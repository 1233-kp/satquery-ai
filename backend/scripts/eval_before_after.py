"""
backend/scripts/eval_before_after.py
=====================================
Runs the frozen base SmolVLM-500M-Instruct and the LoRA-adapted version
side by side on the same held-out BigEarthNet.txt image-text pairs
(data/bigearthnet/demo_samples.jsonl), so the effect of fine-tuning is
directly visible rather than just inferred from the loss curve.

Usage:
    python scripts/eval_before_after.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=str(_BACKEND_ROOT / "data" / "bigearthnet"))
    p.add_argument("--samples-file", type=str, default="demo_samples.jsonl")
    p.add_argument("--adapter-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "vqa_lora" / "best"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--image-max-side", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=str(_BACKEND_ROOT / "data" / "bigearthnet" / "before_after.json"))
    return p.parse_args()


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
    samples = []
    with open(data_dir / args.samples_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} demo samples from {args.samples_file}", flush=True)

    print(f"Loading base model ({dtype})...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": args.image_max_side}

    base_model = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    base_model.eval()

    print(f"Attaching LoRA adapter from {args.adapter_dir}...", flush=True)
    adapted_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapted_model.eval()

    results = []
    for i, sample in enumerate(samples):
        image = Image.open(data_dir / sample["image"]).convert("RGB")
        image.thumbnail((args.image_max_side, args.image_max_side), Image.LANCZOS)

        print(f"\n[{i + 1}/{len(samples)}] {sample['patch_id']}", flush=True)
        print(f"  Q: {sample['question'][:100]}", flush=True)
        print(f"  Ground truth: {sample['answer'][:150]}", flush=True)

        with adapted_model.disable_adapter():
            before = generate(processor, adapted_model, image, sample["question"], device, args.max_new_tokens)
        print(f"  BEFORE (base, no LoRA): {before[:200]}", flush=True)

        after = generate(processor, adapted_model, image, sample["question"], device, args.max_new_tokens)
        print(f"  AFTER  (LoRA-adapted):  {after[:200]}", flush=True)

        results.append({
            "patch_id": sample["patch_id"], "question": sample["question"], "ground_truth": sample["answer"],
            "before_base_model": before, "after_lora_adapted": after,
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {args.out}", flush=True)


if __name__ == "__main__":
    main()
