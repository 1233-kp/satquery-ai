"""
backend/scripts/evaluate_test_split.py
=======================================
Rough accuracy check on a held-out slice of the BigEarthNet.txt test split
(never seen during training or used for checkpoint selection), broken down
by question type. Binary/MCQ questions have a well-defined right answer, so
exact-match accuracy is meaningful; captioning is free-form text, so those
rows are reported qualitatively instead of forced into an accuracy number.

Note: this evaluates against BigEarthNet.txt's own held-out test split, not
VRSBench or RSVQA -- those are separate datasets that would need their own
acquisition step. Treat this as the same-domain generalization number.

Usage:
    python scripts/evaluate_test_split.py --max-samples 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(_BACKEND_ROOT / "data" / "bigearthnet"))
    p.add_argument("--adapter-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "vqa_lora" / "best"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--image-max-side", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=str(_BACKEND_ROOT / "data" / "bigearthnet" / "test_eval_results.json"))
    return p.parse_args()


def normalize(text: str) -> str:
    t = (text or "").strip().lower().strip(".,!?;:'\"")
    return " ".join(t.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


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
    data_dir = Path(args.data_dir)

    rows = []
    with open(data_dir / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: args.max_samples]
    print(f"Evaluating {len(rows)} held-out test pairs from BigEarthNet.txt (never used in train/val)...", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": args.image_max_side}
    base_model = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    by_type: Dict[str, List[dict]] = {}
    for i, row in enumerate(rows):
        image = Image.open(data_dir / row["image"]).convert("RGB")
        image.thumbnail((args.image_max_side, args.image_max_side), Image.LANCZOS)
        pred = generate(processor, model, image, row["question"], device, args.max_new_tokens)
        em = exact_match(pred, row["answer"])
        by_type.setdefault(row["type"], []).append({
            "patch_id": row["patch_id"], "question": row["question"],
            "ground_truth": row["answer"], "prediction": pred, "exact_match": em,
        })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(rows)}", flush=True)

    summary = {}
    for qtype, items in by_type.items():
        if qtype == "captioning":
            summary[qtype] = {"n": len(items), "note": "free-form text, reported qualitatively (see results file)"}
        else:
            acc = sum(1 for it in items if it["exact_match"]) / len(items)
            summary[qtype] = {"n": len(items), "exact_match_accuracy": round(acc, 4)}

    scored = [it for items in by_type.values() for it in items if "exact_match" in it]
    scored_types = {k: v for k, v in by_type.items() if k != "captioning"}
    n_scored = sum(len(v) for v in scored_types.values())
    n_correct = sum(sum(1 for it in v if it["exact_match"]) for v in scored_types.values())
    overall_acc = n_correct / n_scored if n_scored else None

    print("\n" + "=" * 60, flush=True)
    print("HELD-OUT TEST RESULTS (BigEarthNet.txt test split)", flush=True)
    for qtype, s in summary.items():
        if "exact_match_accuracy" in s:
            print(f"  {qtype:12s}: n={s['n']:4d}  exact_match={s['exact_match_accuracy']:.2%}", flush=True)
        else:
            print(f"  {qtype:12s}: n={s['n']:4d}  ({s['note']})", flush=True)
    if overall_acc is not None:
        print(f"  {'OVERALL (binary+mcq)':12s}: n={n_scored:4d}  exact_match={overall_acc:.2%}", flush=True)
    print("=" * 60, flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "overall_binary_mcq_accuracy": overall_acc, "by_type": by_type}, f, indent=2)
    print(f"Saved detailed results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
