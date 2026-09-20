"""
backend/scripts/evaluate_cdvqa.py
==================================
Evaluates the REAL orchestrator end-to-end against CDVQA (Yuan et al.,
"Change Detection Meets Visual Question Answering", IEEE TGRS 2022) --
the official question/answer annotations from the authors' own GitHub
repo (github.com/YZHJessica/CDVQA), which are checked directly into that
repo (not a third-party mirror, no separate verification needed for the
QA text/answers).

The bi-temporal images themselves are NOT hosted in that repo -- CDVQA is
built on a re-split of the official SECOND semantic-change-detection
dataset (captain-whu.github.io/SCD, WHU's own official host). Cross-
referencing confirmed CDVQA's Val/Test splits draw from SECOND's *train*
pool (the im1/im2 image-pair folders inside SECOND_train_set.rar), not
SECOND's own held-out test folder -- e.g. CDVQA's Val split references
"02180.png", which exists only under SECOND_train_set.rar's im1/im2/, not
in SECOND_total_test.zip. Individual pairs are extracted from the RAR
on demand via 7-Zip (avoiding decompressing the full ~2.3GB archive for
a few hundred representative samples).

Every sample goes through app.services.orchestrator.run_analysis() with
mode="bi_temporal_pair", exercising the real change_vqa path
(change_service.py's detector + deterministic template answer, and the
OOD guard) exactly as a real two-image query would be answered.

Usage:
    python scripts/evaluate_cdvqa.py --max-samples 200
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))  # so `app.*` imports resolve when run as a bare script

SEVENZIP = r"C:\Program Files\7-Zip\7z.exe"
GITHUB_RAW = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["Val", "Test"], default="Val")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=str, default=str(_BACKEND_ROOT / "data" / "cdvqa"))
    p.add_argument(
        "--rar-path", type=str, default=str(_BACKEND_ROOT / "data" / "SECOND_train_set.rar"),
        help="Path to SECOND_train_set.rar (from the official SECOND dataset Google Drive release)",
    )
    p.add_argument("--out-dir", type=str, default=str(_BACKEND_ROOT / "evaluation_results"))
    return p.parse_args()


def normalize(text: str) -> str:
    t = (text or "").strip().lower().strip(".,!?;:'\"")
    return " ".join(t.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def load_official_annotations(cache_dir: Path, split: str) -> tuple[list[dict], list[dict], list[dict]]:
    import urllib.request

    official_dir = cache_dir / "official"
    official_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for kind in ("images", "questions", "answers"):
        fn = f"{split}_{kind}.json"
        path = official_dir / fn
        if not path.exists():
            urllib.request.urlretrieve(f"{GITHUB_RAW}/{fn}", path)
        with open(path, encoding="utf-8") as f:
            files[kind] = json.load(f)[kind]
    return files["images"], files["questions"], files["answers"]


def sample_questions(questions: list[dict], max_samples: int, seed: int) -> list[dict]:
    active = [q for q in questions if q.get("active", True)]
    rng = random.Random(seed)
    shuffled = active[:]
    rng.shuffle(shuffled)
    return shuffled[:max_samples]


def fetch_image_pairs(rar_path: Path, cache_dir: Path, file_names: set[str]) -> dict[str, tuple[Path, Path]]:
    """Extracts only the needed im1/im2 pairs from SECOND_train_set.rar via
    7-Zip, instead of decompressing the full ~2.3GB archive."""
    images_dir = cache_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    to_fetch = [
        fn for fn in file_names
        if not (images_dir / "im1" / fn).exists() or not (images_dir / "im2" / fn).exists()
    ]
    if to_fetch:
        print(f"Extracting {len(to_fetch)} image pairs from {rar_path.name} via 7-Zip...", flush=True)
        list_file = cache_dir / "_extract_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for fn in to_fetch:
                f.write(f"im1\\{fn}\n")
                f.write(f"im2\\{fn}\n")
        subprocess.run(
            [SEVENZIP, "x", str(rar_path), f"-i@{list_file}", f"-o{images_dir}", "-y"],
            check=True, capture_output=True,
        )

    return {fn: (images_dir / "im1" / fn, images_dir / "im2" / fn) for fn in file_names}


def run_orchestrator(path_a: Path, path_b: Path, question: str):
    from app.schemas import AnalysisRequest
    from app.services.image_inspector import inspect_image
    from app.services.orchestrator import run_analysis

    meta_a = inspect_image(str(path_a), path_a.name)
    meta_b = inspect_image(str(path_b), path_b.name)
    request = AnalysisRequest(
        query=question, mode="bi_temporal_pair", image_ids=["a", "b"], task_hint="change_vqa",
    )
    result = run_analysis(
        request, {"a": str(path_a), "b": str(path_b)}, {"a": meta_a, "b": meta_b},
    )
    return result


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    rar_path = Path(args.rar_path)
    if not rar_path.exists():
        raise SystemExit(f"SECOND_train_set.rar not found at {rar_path} -- see module docstring for source.")

    images, questions, answers = load_official_annotations(cache_dir, args.split)
    answers_by_id = {a["id"]: a for a in answers}
    images_by_id = {im["id"]: im for im in images}

    sampled_qs = sample_questions(questions, args.max_samples, args.seed)
    print(f"Sampled {len(sampled_qs)}/{len(questions)} CDVQA {args.split} questions (seed={args.seed})", flush=True)

    file_names = {images_by_id[q["img_id"]]["file_name"] for q in sampled_qs}
    pairs = fetch_image_pairs(rar_path, cache_dir, file_names)

    print(f"Evaluating {len(sampled_qs)} samples through the real orchestrator (change_vqa path)...", flush=True)
    by_type: Dict[str, List[dict]] = {}
    for i, q in enumerate(sampled_qs):
        fn = images_by_id[q["img_id"]]["file_name"]
        path_a, path_b = pairs[fn]
        gt = answers_by_id[q["answers_ids"][0]]["answer"]
        result = run_orchestrator(path_a, path_b, q["question"])
        em = exact_match(result.answer, str(gt))
        by_type.setdefault(q["type"], []).append({
            "file_name": fn, "question": q["question"], "ground_truth": str(gt),
            "prediction": result.answer, "exact_match": em,
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(sampled_qs)}", flush=True)

    summary = {}
    n_total = n_correct = 0
    for t, items in by_type.items():
        acc = sum(1 for it in items if it["exact_match"]) / len(items)
        summary[t] = {"n": len(items), "exact_match_accuracy": round(acc, 4)}
        n_total += len(items)
        n_correct += sum(1 for it in items if it["exact_match"])
    overall = round(n_correct / n_total, 4) if n_total else None

    print("\n" + "=" * 60, flush=True)
    print(f"CDVQA {args.split.upper()} RESULTS (via real orchestrator, change_vqa path)", flush=True)
    for t, s in summary.items():
        print(f"  {t:20s}: n={s['n']:4d}  exact_match={s['exact_match_accuracy']:.2%}", flush=True)
    print(f"  {'OVERALL':20s}: n={n_total:4d}  exact_match={overall:.2%}", flush=True)
    print("=" * 60, flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cdvqa_{args.split.lower()}_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": "CDVQA (YZHJessica/CDVQA)", "split": args.split, "summary": summary,
                   "overall_accuracy": overall, "by_type": by_type}, f, indent=2)
    print(f"Saved detailed results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
