"""
One-off comparison of decoding configs on the known repetition-prone case,
to find a repetition fix that doesn't trade one failure mode for another.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "baseline (greedy, no penalty)": dict(do_sample=False),
    "no_repeat_ngram_size=3 only": dict(do_sample=False, no_repeat_ngram_size=3),
    "no_repeat_ngram_size=4 only": dict(do_sample=False, no_repeat_ngram_size=4),
    "repetition_penalty=1.15 only": dict(do_sample=False, repetition_penalty=1.15),
    "repetition_penalty=1.15 + ngram=4": dict(do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=4),
    "repetition_penalty=1.3 + ngram=3 (current)": dict(do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3),
}


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": 448}
    base_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", torch_dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base_model, str(_BACKEND_ROOT / "checkpoints" / "vqa_lora" / "best"))
    model.eval()

    image = Image.open(_BACKEND_ROOT / "data" / "uploads" / "29218fa5f547.jpg").convert("RGB")
    image.thumbnail((448, 448), Image.LANCZOS)
    question = "Describe the land-cover and major objects visible in this image"
    prompt = f"Describe this remote-sensing image in detail. {question}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Remote sensing analysis: {prompt}"}]}]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=[image], return_tensors="pt").to(device)

    for name, gen_kwargs in CONFIGS.items():
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, **gen_kwargs)
        new_tokens = output[:, inputs["input_ids"].shape[-1]:]
        answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        print("=" * 70)
        print(name)
        print("-" * 70)
        print(answer)
        print()


if __name__ == "__main__":
    main()
