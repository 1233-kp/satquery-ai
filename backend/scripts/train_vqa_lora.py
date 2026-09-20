"""
backend/scripts/train_vqa_lora.py
==================================
LoRA fine-tunes HuggingFaceTB/SmolVLM-500M-Instruct on the BigEarthNet.txt
image-text pairs produced by prepare_bigearthnet.py, targeting a 6GB-class
laptop GPU (e.g. RTX 3050): the base model is frozen and loaded in
fp16/bf16, only a small rank-8/16 LoRA adapter on the attention projections
is trained, with a real DataLoader (num_workers=2) and gradient
accumulation to reach an effective batch size of ~16 while keeping the
per-step micro-batch at 1-2 samples.

Falls back to CPU + fp32 automatically when no CUDA device is available
(slow, but correct -- useful for smoke-testing the pipeline).

Usage:
    python scripts/train_vqa_lora.py --probe-steps 5      # GPU memory check only
    python scripts/train_vqa_lora.py                      # full run, default hyperparams
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune SmolVLM on BigEarthNet.txt VQA pairs.")
    p.add_argument("--data-dir", type=str, default=str(_BACKEND_ROOT / "data" / "bigearthnet"))
    p.add_argument("--output-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "vqa_lora"))
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2, help="Micro-batch size (1-2 for 6GB VRAM).")
    p.add_argument("--grad-accum", type=int, default=8, help="With batch-size=2, effective batch = 16.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--image-max-side", type=int, default=448, help="Resize cap fed to the processor (384-512 range).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe-steps", type=int, default=0, help="If >0, run only this many training steps, report GPU memory, and exit (no save/eval).")
    return p.parse_args()


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize_answer(text: str) -> str:
    t = (text or "").strip().lower().strip(".,!?;:'\"")
    return " ".join(t.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def soft_match(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    return g == p or g in p or p in g


class VQADataset:
    """Returns raw (PIL image, question, answer) tuples; tokenization happens in collate_fn
    (in the main process) so DataLoader workers only do image I/O + resizing in parallel."""

    def __init__(self, rows: List[dict], data_dir: Path, image_max_side: int):
        self.rows = rows
        self.data_dir = data_dir
        self.image_max_side = image_max_side

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        row = self.rows[idx]
        img_path = self.data_dir / row["image"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        image.thumbnail((self.image_max_side, self.image_max_side), Image.LANCZOS)
        return {"image": image, "question": row["question"], "answer": row["answer"], "patch_id": row["patch_id"]}


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # Module-level (not a closure) so it's picklable for Windows' spawn-based DataLoader workers.
    return [b for b in batch if b is not None]


def build_train_inputs(processor, sample: Dict[str, Any], device, dtype):
    question_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Remote sensing analysis: {sample['question']}"}]}]
    full_msgs = question_msgs + [{"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}]

    prompt_text = processor.apply_chat_template(question_msgs, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_msgs, add_generation_prompt=False)

    inputs = processor(text=full_text, images=[sample["image"]], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Mask the prompt (image + question) out of the loss -- otherwise the model is
    # penalized for not "predicting" its own input, which dominates and distorts the loss.
    prompt_len = processor(text=prompt_text, images=[sample["image"]], return_tensors="pt")["input_ids"].shape[-1]
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    return inputs, labels


def generate_answer(processor, model, sample: Dict[str, Any], device, max_new_tokens: int = 64) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Remote sensing analysis: {sample['question']}"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[sample["image"]], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = gen[:, inputs["input_ids"].shape[-1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def log_gpu_memory(tag: str, device: str):
    import torch

    if device != "cuda":
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  [GPU MEM] {tag}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB / {total:.2f}GB total", flush=True)


def main():
    args = parse_args()
    t0 = time.perf_counter()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForVision2Seq, AutoProcessor

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device == "cuda"

    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        bf16_ok = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        dtype = torch.float32
        gpu_name, vram_gb, bf16_ok = "None (CPU only)", None, False

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75, flush=True)
    print("SatQuery-AI VQA LoRA Training -- SmolVLM-500M-Instruct on BigEarthNet.txt", flush=True)
    print(f"Device: {device} ({gpu_name}) | dtype: {dtype} | VRAM: {f'{vram_gb:.2f}GB' if vram_gb else 'N/A'}", flush=True)
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} | "
          f"batch={args.batch_size} grad_accum={args.grad_accum} (effective={args.batch_size * args.grad_accum}) | "
          f"image_max_side={args.image_max_side} | workers={args.num_workers}", flush=True)
    print("=" * 75, flush=True)

    train_rows = load_jsonl(data_dir / "train.jsonl", args.max_train_samples)
    val_rows = load_jsonl(data_dir / "val.jsonl", args.max_val_samples)
    if not train_rows:
        raise RuntimeError(f"No training rows found in {data_dir}/train.jsonl -- run prepare_bigearthnet.py first.")
    print(f"[1/5] Dataset: {len(train_rows)} train, {len(val_rows)} val pairs", flush=True)

    print(f"[2/5] Loading base model {args.model_id} ({dtype})...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_image_splitting"):
        # Disable multi-crop mosaic splitting and cap the single resized image at
        # image_max_side, which must stay <= the processor's max_image_size (512)
        # or the Idefics3 image processor raises on the size/max_image_size mismatch.
        processor.image_processor.do_image_splitting = False
        processor.image_processor.size = {"longest_edge": args.image_max_side}

    base_model = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
    base_model.to(device)
    total_params = sum(p.numel() for p in base_model.parameters())

    print("[3/5] Applying frozen-base LoRA adapter (q/k/v/o_proj)...", flush=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds = VQADataset(train_rows, data_dir, args.image_max_side)
    val_ds = VQADataset(val_rows, data_dir, args.image_max_side) if val_rows else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    use_scaler = use_cuda and dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def run_micro_batch(batch: List[Dict[str, Any]]) -> Optional[float]:
        if not batch:
            return None
        losses = []
        for sample in batch:
            inputs, labels = build_train_inputs(processor, sample, device, dtype)
            with torch.autocast(device_type=device, dtype=dtype, enabled=use_cuda):
                out = model(**inputs, labels=labels)
                loss = out.loss
                if loss is None:
                    logits = out.logits[..., :-1, :].contiguous()
                    lab = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1))
            scaled = loss / (len(batch) * args.grad_accum)
            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            losses.append(float(loss.item()))
        return sum(losses) / len(losses)

    print(f"[4/5] Training (epochs={args.epochs}, lr={args.lr})...", flush=True)
    training_log: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    global_step = 0
    probe_done = False

    for epoch in range(args.epochs):
        model.train()
        epoch_t0 = time.perf_counter()
        epoch_losses: List[float] = []
        optimizer.zero_grad()
        accum = 0

        for batch_idx, batch in enumerate(train_loader):
            loss_val = run_micro_batch(batch)
            if loss_val is None:
                continue
            epoch_losses.append(loss_val)
            accum += 1

            if accum >= args.grad_accum:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accum = 0
                global_step += 1
                print(f"  E{epoch + 1} step {global_step} [{batch_idx + 1}/{len(train_loader)} batches] loss={loss_val:.4f}", flush=True)

            if global_step <= 3 and use_cuda:
                log_gpu_memory(f"epoch {epoch + 1} step {global_step}", device)

            if args.probe_steps and global_step >= args.probe_steps:
                probe_done = True
                break

        if probe_done:
            break

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else float("nan")

        val_loss = None
        if val_ds and len(val_ds) > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(len(val_ds)):
                    sample = val_ds[i]
                    if sample is None:
                        continue
                    inputs, labels = build_train_inputs(processor, sample, device, dtype)
                    with torch.autocast(device_type=device, dtype=dtype, enabled=use_cuda):
                        out = model(**inputs, labels=labels)
                        vl = out.loss
                    if vl is not None:
                        val_losses.append(float(vl.item()))
            val_loss = sum(val_losses) / max(len(val_losses), 1) if val_losses else None
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                model.save_pretrained(str(best_dir))
                processor.save_pretrained(str(best_dir))
                print(f"  -> new best checkpoint: val_loss={val_loss:.4f} -> {best_dir}", flush=True)

        # Per-epoch snapshot (kept in addition to best/) -- a cheap safety net so a
        # specific epoch's weights are still recoverable later without retraining,
        # even if a later epoch overwrites best/.
        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(epoch_dir))

        dur = time.perf_counter() - epoch_t0
        vl_str = f" val={val_loss:.4f}" if val_loss is not None else ""
        print(f"*** Epoch {epoch + 1}: train={train_loss:.4f}{vl_str} ({dur:.1f}s) ***", flush=True)
        training_log.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4),
                              "val_loss": round(val_loss, 4) if val_loss is not None else None,
                              "duration_sec": round(dur, 1)})

    if args.probe_steps:
        log_gpu_memory("final (probe)", device)
        print(f"\nProbe complete after {global_step} optimizer steps -- no checkpoint saved. "
              f"Adjust batch-size/grad-accum/image-max-side if peak memory is close to the VRAM total above.", flush=True)
        return

    print(f"[5/5] Saving final adapter to {output_dir}...", flush=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    if not (output_dir / "best").exists():
        # No val split improved on -- just mirror the final adapter as "best" so
        # VQA_LORA_PATH=./checkpoints/vqa_lora/best always resolves.
        best_dir = output_dir / "best"
        best_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(best_dir))
        processor.save_pretrained(str(best_dir))

    cfg = {
        "base_model": args.model_id, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trainable_params": trainable_params, "total_params": total_params,
        "trainable_pct": round(100.0 * trainable_params / total_params, 4),
        "device": device, "gpu_name": gpu_name, "vram_gb": vram_gb, "dtype": str(dtype),
        "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum, "learning_rate": args.lr,
        "image_max_side": args.image_max_side, "n_train_samples": len(train_rows), "n_val_samples": len(val_rows),
        "seed": args.seed, "training_log": training_log,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "best_epoch": best_epoch,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n[Evaluation] Reloading best adapter and scoring held-out val set...", flush=True)
    eval_base = AutoModelForVision2Seq.from_pretrained(args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    adapted = PeftModel.from_pretrained(eval_base, str(output_dir / "best"))
    adapted.eval()

    em_correct = soft_correct = 0
    eval_results = []
    for i in range(len(val_ds) if val_ds else 0):
        sample = val_ds[i]
        if sample is None:
            continue
        pred = generate_answer(processor, adapted, sample, device)
        gold = sample["answer"]
        em, sm = exact_match(pred, gold), soft_match(pred, gold)
        em_correct += int(em)
        soft_correct += int(sm)
        eval_results.append({"patch_id": sample["patch_id"], "question": sample["question"],
                              "ground_truth": gold, "prediction": pred, "exact_match": em, "soft_match": sm})

    n_eval = len(eval_results)
    if n_eval:
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump({"n_evaluated": n_eval, "exact_match_accuracy": round(em_correct / n_eval, 4),
                        "soft_match_accuracy": round(soft_correct / n_eval, 4), "results": eval_results}, f, indent=2)
        print(f"  Exact match: {em_correct}/{n_eval} ({em_correct / n_eval:.2%}) | "
              f"Soft match: {soft_correct}/{n_eval} ({soft_correct / n_eval:.2%})", flush=True)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 75, flush=True)
    print(f"DONE in {elapsed:.0f}s. Best adapter: {output_dir / 'best'} (val_loss={best_val_loss:.4f} @ epoch {best_epoch})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
