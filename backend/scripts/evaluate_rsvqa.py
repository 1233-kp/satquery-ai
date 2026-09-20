"""
backend/scripts/evaluate_rsvqa.py
==================================
Rough accuracy check against RSVQA-LR (Sentinel-2, low-resolution VQA
benchmark) -- the actual external benchmark named in the brief, as
opposed to BigEarthNet.txt's own held-out split (see
evaluate_test_split.py). Uses dmarsili/RSVQA-LR-2k on the HF Hub, a
self-contained 2000-row sample of RSVQA-LR's validation split with
images embedded directly in the parquet (no separate image archive).

This is true cross-dataset generalization: RSVQA-LR is a different
imagery source and question distribution than the BigEarthNet.txt data
the LoRA adapter was fine-tuned on.

Question categories are inferred heuristically from question text
(RSVQA's own paper uses: count, presence, comparison, rural/urban),
since the 2k sample doesn't carry the original category column.

Usage:
    python scripts/evaluate_rsvqa.py --max-samples 150
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Dict, List

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATASET_REPO = "dmarsili/RSVQA-LR-2k"
DATASET_FILE = "data/validation-00000-of-00001.parquet"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "vqa_lora" / "best"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--image-max-side", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cache-dir", type=str, default=str(_BACKEND_ROOT / "data" / "rsvqa_lr"))
    p.add_argument("--out", type=str, default=str(_BACKEND_ROOT / "data" / "rsvqa_lr" / "rsvqa_eval_results.json"))
    return p.parse_args()


def normalize(text: str) -> str:
    t = (text or "").strip().lower().strip(".,!?;:'\"")
    return " ".join(t.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def infer_category(question: str) -> str:
    q = question.lower()
    if "rural" in q and "urban" in q:
        return "rural_urban"
    if re.search(r"\bhow many\b", q):
        return "count"
    if re.search(r"\b(more|less|greater|fewer|same number|compared to|than)\b", q):
        return "comparison"
    if re.search(r"\b(is there|are there|is it|does|do)\b", q):
        return "presence"
    return "other"


def load_sample(cache_dir: Path, max_samples: int, seed: int) -> List[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print(f"Downloading {DATASET_REPO}/{DATASET_FILE} (small, self-contained parquet)...", flush=True)
    path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    table = pq.read_table(path)
    n_total = table.num_rows
    print(f"  -> {n_total} rows available (RSVQA-LR validation split)", flush=True)

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    indices = indices[:max_samples]

    images_dir = cache_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in indices:
        row = table.slice(i, 1).to_pylist()[0]
        img_bytes = row["image"]["bytes"]
        img_path = images_dir / f"{i}.png"
        if not img_path.exists():
            Image.open(io.BytesIO(img_bytes)).convert("RGB").save(img_path)
        rows.append({"idx": i, "image_path": img_path, "question": row["question"], "answer": row["answer"]})
    return rows


def generate(processor, model, image, question, device, max_new_tokens):
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Remote sensing analysis: {question}"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = gen[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)

    rows = load_sample(Path(args.cache_dir), args.max_samples, args.seed)
    print(f"Evaluating {len(rows)} RSVQA-LR pairs (cross-dataset -- not seen during BigEarthNet.txt fine-tuning)...", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": args.image_max_side}
    base_model = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    by_category: Dict[str, List[dict]] = {}
    for i, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        image.thumbnail((args.image_max_side, args.image_max_side), Image.LANCZOS)
        pred = generate(processor, model, image, row["question"], device, args.max_new_tokens)
        em = exact_match(pred, row["answer"])
        cat = infer_category(row["question"])
        by_category.setdefault(cat, []).append({
            "idx": row["idx"], "question": row["question"],
            "ground_truth": row["answer"], "prediction": pred, "exact_match": em,
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(rows)}", flush=True)

    summary = {}
    n_total = n_correct = 0
    for cat, items in by_category.items():
        acc = sum(1 for it in items if it["exact_match"]) / len(items)
        summary[cat] = {"n": len(items), "exact_match_accuracy": round(acc, 4)}
        n_total += len(items)
        n_correct += sum(1 for it in items if it["exact_match"])
    overall_acc = n_correct / n_total if n_total else None

    print("\n" + "=" * 60, flush=True)
    print("RSVQA-LR RESULTS (cross-dataset, unseen during fine-tuning)", flush=True)
    for cat, s in summary.items():
        print(f"  {cat:12s}: n={s['n']:4d}  exact_match={s['exact_match_accuracy']:.2%}", flush=True)
    print(f"  {'OVERALL':12s}: n={n_total:4d}  exact_match={overall_acc:.2%}", flush=True)
    print("=" * 60, flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"dataset": DATASET_REPO, "summary": summary, "overall_accuracy": overall_acc, "by_category": by_category}, f, indent=2)
    print(f"Saved detailed results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
