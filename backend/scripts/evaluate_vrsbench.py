"""
backend/scripts/evaluate_vrsbench.py
=====================================
Evaluates the REAL orchestrator end-to-end against VRSBench's official eval
splits (captioning, VQA, referring/grounding) -- the authors' own
HuggingFace release (xiang709/VRSBench, the same handle as the paper's
GitHub repo lx709/VRSBench), not a third-party mirror.

Unlike evaluate_rsvqa.py / evaluate_test_split.py (which call the VLM
directly for speed), every sample here goes through
app.services.orchestrator.run_analysis() -- exercising real task
classification, the hallucination guard, and (for the grounding subset)
grounding_service.py's oversized-box size guard, exactly as a real query
would be answered.

Routing note: VRSBench's own question phrasing doesn't naturally trigger
our keyword-based classifier for captioning or grounding ("Describe the
image in detail" doesn't match CAPTION_KEYWORDS; a referring expression
like "The large yellow vehicle situated closest to the green area." isn't
phrased as "highlight/locate X" and doesn't match GROUNDING_KEYWORDS
either). This is expected -- our classifier is tuned for our own app's
phrasing, not VRSBench's -- so this script sets `task_hint` explicitly to
route each subset to its intended path, per the requested subset mapping,
rather than let a classifier phrasing mismatch contaminate the metric.

Images are pulled on demand from the official Images_val.zip via HTTP
range requests (huggingface_hub.HfFileSystem + zipfile), avoiding a ~4GB
full-archive download for a few hundred representative samples.

Usage:
    python scripts/evaluate_vrsbench.py --subset captioning --max-samples 100
    python scripts/evaluate_vrsbench.py --subset vqa --max-samples 150
    python scripts/evaluate_vrsbench.py --subset grounding --max-samples 100
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))  # so `app.*` imports resolve when run as a bare script
DATASET_REPO = "xiang709/VRSBench"
IMAGES_ZIP_PATH = "datasets/xiang709/VRSBench/Images_val.zip"
EVAL_FILES = {
    "captioning": "VRSBench_EVAL_Cap.json",
    "vqa": "VRSBench_EVAL_vqa.json",
    "grounding": "VRSBench_EVAL_referring.json",
}
TASK_HINT = {"captioning": "captioning", "vqa": "vqa", "grounding": "grounding"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=["captioning", "vqa", "grounding"], required=True)
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=str, default=str(_BACKEND_ROOT / "data" / "vrsbench"))
    p.add_argument("--out-dir", type=str, default=str(_BACKEND_ROOT / "evaluation_results"))
    return p.parse_args()


def normalize(text: str) -> str:
    t = (text or "").strip().lower().strip(".,!?;:'\"")
    return " ".join(t.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def load_official_annotations(cache_dir: Path, subset: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    official_dir = cache_dir / "official"
    official_dir.mkdir(parents=True, exist_ok=True)
    fn = EVAL_FILES[subset]
    path = hf_hub_download(repo_id=DATASET_REPO, filename=fn, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_rows(rows: list[dict], max_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[:max_samples]


def fetch_images(cache_dir: Path, image_ids: set[str]) -> dict[str, Path]:
    """Extracts just the needed images from the official Images_val.zip.
    Uses the full local copy if already cached by huggingface_hub (fast,
    no network per file); otherwise falls back to HTTP range requests via
    HfFileSystem (avoids a ~4GB download for a handful of images, but was
    observed to degrade badly in throughput for larger sample counts --
    see evaluate_vrsbench.py history / README notes)."""
    import zipfile
    from huggingface_hub import hf_hub_download, try_to_load_from_cache, HfFileSystem

    images_dir = cache_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    to_fetch = {iid for iid in image_ids if not (images_dir / iid).exists()}
    if not to_fetch:
        return {iid: images_dir / iid for iid in image_ids}

    cached_zip = try_to_load_from_cache(repo_id=DATASET_REPO, filename="Images_val.zip", repo_type="dataset")
    manual_zip = cache_dir / "official" / "Images_val.zip"
    if not (cached_zip and Path(cached_zip).exists()) and manual_zip.exists():
        # huggingface_hub's own downloader (both plain hf_hub_download and the
        # HfFileSystem range-request path) was observed to throttle down to a
        # near-standstill on this specific large file; a direct curl of the
        # resolved URL sustained ~7MB/s with no degradation, so that's used
        # instead and just needs to be found here rather than in the HF cache.
        cached_zip = manual_zip
    if cached_zip and Path(cached_zip).exists():
        print(f"Extracting {len(to_fetch)} images from the locally-cached full zip...", flush=True)
        zf = zipfile.ZipFile(cached_zip)
        for iid in to_fetch:
            with zf.open(f"Images_val/{iid}") as src, open(images_dir / iid, "wb") as dst:
                dst.write(src.read())
    else:
        print(f"Fetching {len(to_fetch)} images from the official zip via range requests...", flush=True)
        fs = HfFileSystem()
        with fs.open(IMAGES_ZIP_PATH) as f:
            zf = zipfile.ZipFile(f)
            for iid in to_fetch:
                with zf.open(f"Images_val/{iid}") as src, open(images_dir / iid, "wb") as dst:
                    dst.write(src.read())

    return {iid: images_dir / iid for iid in image_ids}


def run_orchestrator(image_path: Path, question: str, task_hint: str):
    from app.schemas import AnalysisRequest
    from app.services.image_inspector import inspect_image
    from app.services.orchestrator import run_analysis

    meta = inspect_image(str(image_path), image_path.name)
    request = AnalysisRequest(query=question, mode="single_image", image_ids=["eval_0"], task_hint=task_hint)
    result = run_analysis(request, {"eval_0": str(image_path)}, {"eval_0": meta})
    return result


# ---- grounding-specific helpers ----

_BRACKET_BOX_RE = re.compile(r"<(-?\d+(?:\.\d+)?)>")


def parse_norm_box(ground_truth: str) -> tuple[float, float, float, float] | None:
    """VRSBench referring ground truth is '{<x0><y0><x1><y1>}' with each
    value normalized to a 0-100 scale of image width/height."""
    nums = _BRACKET_BOX_RE.findall(ground_truth)
    if len(nums) != 4:
        return None
    return tuple(float(n) for n in nums)  # type: ignore[return-value]


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def eval_captioning(rows: list[dict], images: dict[str, Path], out_dir: Path):
    import sacrebleu
    from pycocoevalcap.cider.cider import Cider

    items = []
    for i, row in enumerate(rows):
        result = run_orchestrator(images[row["image_id"]], row["question"], TASK_HINT["captioning"])
        items.append({
            "image_id": row["image_id"], "question": row["question"],
            "ground_truth": row["ground_truth"], "prediction": result.answer,
            "task_selected": result.trace.task_selected,
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)}", flush=True)

    hyps = [it["prediction"] for it in items]
    refs = [it["ground_truth"] for it in items]
    bleu4 = sacrebleu.corpus_bleu(hyps, [refs]).score

    gts = {str(i): [refs[i]] for i in range(len(items))}
    res = {str(i): [hyps[i]] for i in range(len(items))}
    cider_score, cider_per_item = Cider().compute_score(gts, res)
    for i, it in enumerate(items):
        it["cider"] = round(float(cider_per_item[i]), 4)

    task_selected_mismatch = sum(1 for it in items if it["task_selected"] != "captioning")
    summary = {
        "n": len(items),
        "bleu4_corpus": round(bleu4, 4),
        "cider_corpus": round(float(cider_score), 4),
        "task_hint_used": True,
        "task_selected_mismatch": task_selected_mismatch,
    }
    _save_and_print("captioning", summary, {"items": items}, out_dir)


def eval_vqa(rows: list[dict], images: dict[str, Path], out_dir: Path):
    by_type: Dict[str, List[dict]] = {}
    for i, row in enumerate(rows):
        result = run_orchestrator(images[row["image_id"]], row["question"], TASK_HINT["vqa"])
        em = exact_match(result.answer, row["ground_truth"])
        by_type.setdefault(row["type"], []).append({
            "image_id": row["image_id"], "question": row["question"],
            "ground_truth": row["ground_truth"], "prediction": result.answer,
            "exact_match": em, "task_selected": result.trace.task_selected,
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)}", flush=True)

    summary = {}
    n_total = n_correct = 0
    for t, its in by_type.items():
        acc = sum(1 for it in its if it["exact_match"]) / len(its)
        summary[t] = {"n": len(its), "exact_match_accuracy": round(acc, 4)}
        n_total += len(its)
        n_correct += sum(1 for it in its if it["exact_match"])
    overall = {"n": n_total, "exact_match_accuracy": round(n_correct / n_total, 4) if n_total else None}
    _save_and_print("vqa", {"by_type": summary, "overall": overall}, {"by_type": by_type}, out_dir)


def eval_grounding(rows: list[dict], images: dict[str, Path], out_dir: Path):
    from PIL import Image

    items = []
    guard_fired_count = 0
    for i, row in enumerate(rows):
        img_path = images[row["image_id"]]
        result = run_orchestrator(img_path, row["question"], TASK_HINT["grounding"])
        w, h = Image.open(img_path).size

        gt_norm = parse_norm_box(row["ground_truth"])
        gt_box_px = None
        if gt_norm:
            gt_box_px = (gt_norm[0] / 100 * w, gt_norm[1] / 100 * h, gt_norm[2] / 100 * w, gt_norm[3] / 100 * h)

        detections = (result.visual_evidence or {}).get("detections") or []
        guard_fired = any(s.step == "grounding_guard" for s in result.trace.steps)
        if guard_fired:
            guard_fired_count += 1

        best_iou = 0.0
        if detections and gt_box_px:
            best_iou = max(iou(tuple(d["box"]), gt_box_px) for d in detections)

        items.append({
            "image_id": row["image_id"], "referring_expression": row["question"],
            "ground_truth_box_norm": row["ground_truth"], "n_detections": len(detections),
            "guard_fired": guard_fired, "iou": round(best_iou, 4),
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)}", flush=True)

    n = len(items)
    precision_at_50 = sum(1 for it in items if it["iou"] >= 0.5) / n if n else None
    mean_iou = sum(it["iou"] for it in items) / n if n else None
    summary = {
        "n": n,
        "precision_at_iou_0.5": round(precision_at_50, 4) if precision_at_50 is not None else None,
        "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
        "grounding_guard_fired_count": guard_fired_count,
        "grounding_guard_fired_rate": round(guard_fired_count / n, 4) if n else None,
    }
    _save_and_print("grounding", summary, {"items": items}, out_dir)


def _save_and_print(subset: str, summary: dict, detail: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vrsbench_{subset}_eval_results.json"
    print("\n" + "=" * 60, flush=True)
    print(f"VRSBench {subset.upper()} RESULTS (via real orchestrator)", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("=" * 60, flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET_REPO, "subset": subset, "summary": summary, **detail}, f, indent=2)
    print(f"Saved detailed results to {out_path}", flush=True)


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    rows_all = load_official_annotations(cache_dir, args.subset)
    rows = sample_rows(rows_all, args.max_samples, args.seed)
    print(f"Sampled {len(rows)}/{len(rows_all)} VRSBench {args.subset} examples (seed={args.seed})", flush=True)

    image_ids = {r["image_id"] for r in rows}
    images = fetch_images(cache_dir, image_ids)

    print(f"Evaluating {len(rows)} samples through the real orchestrator (task_hint={TASK_HINT[args.subset]!r})...", flush=True)
    if args.subset == "captioning":
        eval_captioning(rows, images, out_dir)
    elif args.subset == "vqa":
        eval_vqa(rows, images, out_dir)
    else:
        eval_grounding(rows, images, out_dir)


if __name__ == "__main__":
    main()
