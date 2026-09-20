"""
backend/scripts/diagnose_ood_hallucination.py
================================================
One-off diagnostic: compares the frozen base SmolVLM (no LoRA) against the
BigEarthNet.txt-adapted checkpoint on out-of-distribution images (not
satellite imagery), to check whether the rigid "country + climate zone +
sqm area" caption template is a base-model tendency or something the LoRA
fine-tune introduced/amplified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def generate(processor, model, image, question, device, max_new_tokens=150):
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = gen[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "vqa_lora" / "best"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--images", nargs="+", default=[
        str(_BACKEND_ROOT / "data" / "uploads" / "29218fa5f547.jpg"),
        str(_BACKEND_ROOT / "data" / "uploads" / "c3c24f838fb3.png"),
    ])
    p.add_argument("--question", type=str, default="Describe this remote sensing image.")
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)

    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": 448}

    base_model = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    adapted = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapted.eval()

    for img_path in args.images:
        image = Image.open(img_path).convert("RGB")
        image.thumbnail((448, 448), Image.LANCZOS)
        print("=" * 70)
        print(f"Image: {img_path}")
        with adapted.disable_adapter():
            base_out = generate(processor, adapted, image, args.question, device)
        adapted_out = generate(processor, adapted, image, args.question, device)
        print(f"BASE (no LoRA):      {base_out}")
        print(f"ADAPTED (LoRA):      {adapted_out}")


if __name__ == "__main__":
    main()
