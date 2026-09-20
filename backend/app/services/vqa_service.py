"""
Single-image VQA / captioning service.

VQA_MODE=mock  -> deterministic stub answers so the rest of the stack
                  (orchestrator, frontend, trace assembly) can be built
                  and demoed before the real model is trained/wired in.
VQA_MODE=real  -> loads SmolVLM-500M-Instruct (+ optional LoRA adapter)
                  and runs actual inference. Fill in `_load_real_model`
                  and `_run_real_inference` once your Day 2 fine-tune
                  is ready.
"""
from functools import lru_cache
from typing import Union
from PIL import Image

from app.config import get_settings

settings = get_settings()

ImagePathOrPaths = Union[str, list[str]]


class VQAService:
    def __init__(self):
        self.mode = settings.vqa_mode
        self._model = None
        self._processor = None
        if self.mode == "real":
            self._load_real_model()

    def _load_real_model(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        if settings.vqa_device == "cpu":
            self._dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float16

        self._processor = AutoProcessor.from_pretrained(settings.vqa_base_model)
        if hasattr(self._processor, "image_processor") and hasattr(self._processor.image_processor, "do_image_splitting"):
            # Matches training: one resized image per sample (capped at 448px), no multi-crop mosaic.
            self._processor.image_processor.do_image_splitting = False
            self._processor.image_processor.size = {"longest_edge": 448}

        self._model = AutoModelForVision2Seq.from_pretrained(
            settings.vqa_base_model,
            torch_dtype=self._dtype,
        ).to(settings.vqa_device)

        # Attach LoRA adapter if a fine-tuned checkpoint exists
        import os
        if os.path.isdir(settings.vqa_lora_path):
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, settings.vqa_lora_path)

        self._model.eval()

    def answer(
        self,
        image_path: ImagePathOrPaths,
        question: str,
        task: str = "vqa",
        min_new_tokens: int | None = None,
    ) -> tuple[str, float | None, bool, str]:
        if self.mode == "mock":
            first_path = image_path[0] if isinstance(image_path, list) else image_path
            return self._mock_answer(first_path, question, task), None, False, ""
        return self._real_answer(image_path, question, task, min_new_tokens)

    def _mock_answer(self, image_path: str, question: str, task: str) -> str:
        from app.services.geo_preprocessing import load_image_as_rgb

        img = load_image_as_rgb(image_path)
        w, h = img.size
        if task == "captioning":
            return (
                f"[mock] This {w}x{h} remote-sensing scene shows a mix of built-up areas, "
                "vegetation, and open land, consistent with a peri-urban region."
            )
        return (
            f"[mock] Based on visual analysis of the {w}x{h} image, the answer to "
            f"'{question}' is: a mixed land-cover scene with visible built-up structures "
            "and vegetation patches. (Replace VQA_MODE=real once the LoRA-adapted model is wired in.)"
        )

    def _real_answer(
        self, image_path: ImagePathOrPaths, question: str, task: str, min_new_tokens: int | None = None
    ) -> tuple[str, float | None, bool, str]:
        import torch
        from app.services.geo_preprocessing import load_image_as_rgb

        paths = image_path if isinstance(image_path, list) else [image_path]
        images = []
        for p in paths:
            img = load_image_as_rgb(p)
            img.thumbnail((512, 512), Image.LANCZOS)
            images.append(img)

        if task == "captioning":
            # Append to, don't replace, the user's query -- otherwise any instruction
            # they gave (e.g. "do not invent measurements") never reaches the model.
            base_instruction = "Describe this remote-sensing image in detail."
            prompt = f"{base_instruction} {question}" if question else base_instruction
        else:
            prompt = question

        # One {"type": "image"} placeholder per image passed in `images=` below -- the
        # processor requires these counts to match. Note: the LoRA adapter was only ever
        # fine-tuned on single-image examples, so a 2-image call (change-VQA) exercises
        # a base-model capability (Idefics3 supports multi-image) that's outside what the
        # adapter itself was trained on -- untested territory as far as the fine-tune goes.
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                + [{"type": "text", "text": f"Remote sensing analysis: {prompt}"}],
            }
        ]
        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=text_prompt, images=images, return_tensors="pt"
        ).to(settings.vqa_device)

        # Captioning naturally generates far more tokens than VQA (measured on
        # real requests: ~235 tokens for a captioning call vs ~2 for VQA, both
        # reaching EOS naturally -- captioning isn't stuck rambling to a token
        # cap, it's just a genuinely longer task). At ~230ms/token on CPU that
        # alone accounts for the bulk of captioning's latency, so it gets its
        # own, lower cap to bound worst-case latency; VQA's answers are nowhere
        # near either value and are unaffected.
        max_tokens = settings.captioning_max_new_tokens if task == "captioning" else settings.vqa_max_new_tokens

        with torch.no_grad():
            # no_repeat_ngram_size alone kills the greedy-decoding repetition loop this
            # model falls into on long descriptive answers. repetition_penalty was tried
            # too (see scripts/tune_generation_config.py) but pushes it into worse
            # confabulation -- wrong units, invented reasoning -- so it's deliberately
            # left off; this is the more surgical fix.
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=3,
            )

        # Slice off the input prompt tokens rather than string-matching, since the
        # decoded prompt text (with special tokens/whitespace) won't exactly match `prompt`.
        new_tokens = output[:, inputs["input_ids"].shape[-1]:]
        answer = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        guard_fired, guard_detail = False, ""
        if task in ("vqa", "captioning"):
            from app.services.hallucination_guard import check_and_clean
            answer, guard_fired, guard_detail = check_and_clean(answer, question)

        return answer, None, guard_fired, guard_detail  # plug in a calibrated confidence score if/when you have one


@lru_cache
def get_vqa_service() -> VQAService:
    return VQAService()
