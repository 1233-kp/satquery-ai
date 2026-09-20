"""
backend/scripts/train_fusion_net.py
====================================
Trains OpticalSARFusionNet (dual-branch gated fusion, see
app/services/models/fusion_net.py) on BigEarthNet optical+SAR pairs with
weak multi-label land-cover supervision (the official BigEarthNet-19
classes). fp16 mixed precision, per-epoch checkpoints, tracks BCE loss AND
per-class + macro F1 every epoch -- loss alone isn't a reliable quality
signal (same lesson as Days 2-3).

Usage:
    python scripts/train_fusion_net.py --probe-steps 5    # GPU memory check
    python scripts/train_fusion_net.py                    # full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.models.fusion_net import NUM_CLASSES, OpticalSARFusionNet, count_parameters  # noqa: E402

_INPUT_SIZE = 128


def parse_args():
    p = argparse.ArgumentParser(description="Train the dual-branch optical+SAR fusion net.")
    p.add_argument("--data-dir", type=str, default=str(_BACKEND_ROOT / "data" / "fusion"))
    p.add_argument("--output-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "fusion_net"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--probe-steps", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    return p.parse_args()


class FusionDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, sar_mean: np.ndarray, sar_std: np.ndarray,
                 augment: bool, max_items: Optional[int] = None):
        self.data_dir = data_dir
        self.rows = []
        with open(data_dir / split / "manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        if max_items:
            self.rows = self.rows[:max_items]
        self.sar_mean = sar_mean.reshape(-1, 1, 1)
        self.sar_std = sar_std.reshape(-1, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        optical = np.array(Image.open(self.data_dir / row["optical"]).convert("RGB"), dtype=np.float32) / 255.0
        sar = np.load(self.data_dir / row["sar"]).astype(np.float32)  # (2, H, W) dB
        sar = (sar - self.sar_mean) / self.sar_std

        if self.augment:
            if np.random.rand() < 0.5:
                optical, sar = optical[:, ::-1].copy(), sar[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                optical, sar = optical[::-1, :].copy(), sar[:, ::-1, :].copy()

        optical_t = torch.from_numpy(optical).permute(2, 0, 1)
        sar_t = torch.from_numpy(sar)
        optical_t = F.interpolate(optical_t.unsqueeze(0), size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        sar_t = F.interpolate(sar_t.unsqueeze(0), size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        label = torch.tensor(row["label_vector"], dtype=torch.float32)
        return optical_t, sar_t, label


def compute_sar_stats(data_dir: Path, sample_size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(data_dir / "train" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    import random
    random.seed(42)
    sample = random.sample(rows, min(sample_size, len(rows)))
    arrs = [np.load(data_dir / r["sar"]).astype(np.float32) for r in sample]
    stacked = np.stack(arrs)  # (N, 2, H, W)
    mean = stacked.mean(axis=(0, 2, 3))
    std = stacked.std(axis=(0, 2, 3))
    return mean, std


@torch.no_grad()
def compute_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * labels).sum(dim=0)
    fp = (pred * (1 - labels)).sum(dim=0)
    fn = ((1 - pred) * labels).sum(dim=0)
    precision = tp / (tp + fp).clamp(min=1e-8)
    recall = tp / (tp + fn).clamp(min=1e-8)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    # Only count classes that actually appear in this eval set (avoid F1=0 from divide-by-zero classes dominating the macro average).
    support = labels.sum(dim=0)
    valid = support > 0
    macro_f1 = f1[valid].mean().item() if valid.any() else 0.0
    return {"macro_f1": macro_f1, "per_class_f1": f1.tolist(), "support": support.tolist()}


def log_gpu_memory(tag: str, device: str):
    if device != "cuda":
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  [GPU MEM] {tag}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB / {total:.2f}GB total", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device == "cuda"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing SAR normalization stats from training sample...", flush=True)
    sar_mean, sar_std = compute_sar_stats(data_dir)
    print(f"  SAR mean (VH,VV)={sar_mean}, std={sar_std}", flush=True)

    train_ds = FusionDataset(data_dir, "train", sar_mean, sar_std, augment=True, max_items=args.max_train)
    val_ds = FusionDataset(data_dir, "val", sar_mean, sar_std, augment=False, max_items=args.max_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = OpticalSARFusionNet().to(device)
    counts = count_parameters(model)
    print("=" * 75, flush=True)
    print("OpticalSARFusionNet training on BigEarthNet optical+SAR pairs", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})", flush=True)
    print(f"Params: {counts['total']:,} ({counts['total'] / 1e6:.2f}M)", flush=True)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)} | batch={args.batch_size} | fp16={use_cuda}", flush=True)
    print("=" * 75, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    bce = nn.BCEWithLogitsLoss()

    training_log = []
    best_f1 = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_t0 = time.perf_counter()
        train_losses = []

        for step, (optical, sar, labels) in enumerate(train_loader):
            optical, sar, labels = optical.to(device), sar.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_cuda):
                logits = model(optical, sar)
                loss = bce(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            global_step += 1

            if step < 3:
                log_gpu_memory(f"epoch {epoch + 1} step {step + 1}", device)

            if args.probe_steps and global_step >= args.probe_steps:
                log_gpu_memory("final (probe)", device)
                print(f"\nProbe complete after {global_step} steps -- no checkpoint saved.", flush=True)
                return

            if (step + 1) % 50 == 0:
                print(f"  E{epoch + 1} step {step + 1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        scheduler.step()

        model.eval()
        val_losses = []
        all_logits, all_labels = [], []
        with torch.no_grad():
            for optical, sar, labels in val_loader:
                optical, sar, labels = optical.to(device), sar.to(device), labels.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_cuda):
                    logits = model(optical, sar)
                    loss = bce(logits, labels)
                val_losses.append(float(loss.item()))
                all_logits.append(logits.float().cpu())
                all_labels.append(labels.cpu())

        metrics = compute_f1(torch.cat(all_logits), torch.cat(all_labels))
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        dur = time.perf_counter() - epoch_t0

        print(f"*** Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"| macro_F1={metrics['macro_f1']:.4f} ({dur:.1f}s) ***", flush=True)

        epoch_log = {"epoch": epoch + 1, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                     "macro_f1": round(metrics["macro_f1"], 4), "duration_sec": round(dur, 1)}
        training_log.append(epoch_log)

        torch.save(model.state_dict(), output_dir / f"epoch_{epoch + 1}.pt")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            with open(output_dir / "best_eval_per_class.json", "w") as f:
                from app.services.models.fusion_net import NUM_CLASSES as _  # noqa
                json.dump(metrics, f, indent=2)
            print(f"  -> new best checkpoint: macro_F1={metrics['macro_f1']:.4f} -> {output_dir / 'best_model.pt'}", flush=True)

        if epoch >= 4:
            recent = [e["macro_f1"] for e in training_log[-5:]]
            if max(recent) - min(recent) < 0.005:
                print(f"  [FLAG] val macro_F1 has plateaued over the last 5 epochs ({recent}).", flush=True)
            if len(training_log) >= 2 and val_loss > training_log[-2]["val_loss"] * 1.15:
                print(f"  [FLAG] val_loss jumped >15% vs previous epoch -- possible divergence.", flush=True)

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "architecture": "OpticalSARFusionNet (dual-branch gated fusion)",
            "total_params": counts["total"], "num_classes": NUM_CLASSES,
            "input_size": _INPUT_SIZE, "sar_mean": sar_mean.tolist(), "sar_std": sar_std.tolist(),
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "n_train": len(train_ds), "n_val": len(val_ds),
            "best_macro_f1": round(best_f1, 4), "best_epoch": best_epoch,
            "training_log": training_log,
        }, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print(f"DONE. Best checkpoint: epoch {best_epoch}, macro_F1={best_f1:.4f} -> {output_dir / 'best_model.pt'}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
