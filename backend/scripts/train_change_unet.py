"""
backend/scripts/train_change_unet.py
=====================================
Trains the FC-Siam-diff SiameseUNet (see app/services/models/siamese_unet.py)
on LEVIR-CD from scratch, on a 6GB-class GPU: fp16 mixed precision, batch
4-8, BCE + Dice loss (change masks are heavily class-imbalanced -- most
pixels are "no change"). Tracks train/val loss AND precision/recall/F1/IoU
on the changed class every epoch, since loss alone doesn't reliably signal
segmentation quality. Saves a per-epoch snapshot plus the best-by-F1
checkpoint (not best-by-loss) to backend/checkpoints/change_unet/.

Usage:
    python scripts/train_change_unet.py --probe-steps 5     # GPU memory check
    python scripts/train_change_unet.py                     # full run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.models.siamese_unet import SiameseUNet, count_parameters  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train the Siamese U-Net change detector on LEVIR-CD.")
    p.add_argument("--data-dir", type=str, default=str(_BACKEND_ROOT / "data" / "levircd"))
    p.add_argument("--output-dir", type=str, default=str(_BACKEND_ROOT / "checkpoints" / "change_unet"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--probe-steps", type=int, default=0, help="If >0, run this many steps, report GPU memory, and exit.")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    return p.parse_args()


class LevirCDDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, augment: bool, max_items: Optional[int] = None):
        self.data_dir = data_dir
        self.rows = []
        with open(data_dir / split / "manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        if max_items:
            self.rows = self.rows[:max_items]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img_a = np.array(Image.open(self.data_dir / row["imageA"]).convert("RGB"), dtype=np.float32) / 255.0
        img_b = np.array(Image.open(self.data_dir / row["imageB"]).convert("RGB"), dtype=np.float32) / 255.0
        label = np.array(Image.open(self.data_dir / row["label"]).convert("L"), dtype=np.float32) / 255.0

        if self.augment:
            if np.random.rand() < 0.5:
                img_a, img_b, label = img_a[:, ::-1].copy(), img_b[:, ::-1].copy(), label[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img_a, img_b, label = img_a[::-1, :].copy(), img_b[::-1, :].copy(), label[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                img_a, img_b, label = np.rot90(img_a, k).copy(), np.rot90(img_b, k).copy(), np.rot90(label, k).copy()

        img_a = torch.from_numpy(img_a).permute(2, 0, 1)
        img_b = torch.from_numpy(img_b).permute(2, 0, 1)
        label = torch.from_numpy(label).unsqueeze(0)
        return img_a, img_b, label


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    pred = (torch.sigmoid(logits) > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


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

    train_ds = LevirCDDataset(data_dir, "train", augment=True, max_items=args.max_train)
    val_ds = LevirCDDataset(data_dir, "val", augment=False, max_items=args.max_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SiameseUNet().to(device)
    counts = count_parameters(model)
    print("=" * 75, flush=True)
    print("SiameseUNet (FC-Siam-diff) training on LEVIR-CD", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})", flush=True)
    print(f"Params: {counts['total']:,} ({counts['total'] / 1e6:.2f}M)", flush=True)
    print(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)} | batch={args.batch_size} | fp16={use_cuda}", flush=True)
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

        for step, (img_a, img_b, label) in enumerate(train_loader):
            img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_cuda):
                logits = model(img_a, img_b)
                loss = bce(logits, label) + args.dice_weight * dice_loss(logits, label)

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
        agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        with torch.no_grad():
            for img_a, img_b, label in val_loader:
                img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_cuda):
                    logits = model(img_a, img_b)
                    loss = bce(logits, label) + args.dice_weight * dice_loss(logits, label)
                val_losses.append(float(loss.item()))
                m = compute_metrics(logits.float(), label)
                agg["tp"] += m["tp"]
                agg["fp"] += m["fp"]
                agg["fn"] += m["fn"]

        precision = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0.0
        recall = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = agg["tp"] / (agg["tp"] + agg["fp"] + agg["fn"]) if (agg["tp"] + agg["fp"] + agg["fn"]) > 0 else 0.0

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        dur = time.perf_counter() - epoch_t0

        print(f"*** Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"| P={precision:.4f} R={recall:.4f} F1={f1:.4f} IoU={iou:.4f} ({dur:.1f}s) ***", flush=True)

        epoch_log = {
            "epoch": epoch + 1, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "iou": round(iou, 4),
            "duration_sec": round(dur, 1),
        }
        training_log.append(epoch_log)

        # Per-epoch snapshot -- cheap safety net for later comparison.
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch + 1}.pt")

        # Best-by-F1 (not best-by-loss -- loss alone isn't a reliable quality signal).
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  -> new best checkpoint: F1={f1:.4f} -> {output_dir / 'best_model.pt'}", flush=True)

        # Flag plateau/divergence proactively rather than waiting to be asked.
        if epoch >= 4:
            recent_f1 = [e["f1"] for e in training_log[-5:]]
            if max(recent_f1) - min(recent_f1) < 0.005:
                print(f"  [FLAG] val F1 has plateaued over the last 5 epochs ({recent_f1}).", flush=True)
            if len(training_log) >= 2 and val_loss > training_log[-2]["val_loss"] * 1.15:
                print(f"  [FLAG] val_loss jumped >15% vs previous epoch ({training_log[-2]['val_loss']:.4f} -> {val_loss:.4f}) -- possible divergence.", flush=True)

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "architecture": "SiameseUNet (FC-Siam-diff)", "total_params": counts["total"],
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "n_train": len(train_ds), "n_val": len(val_ds), "best_f1": round(best_f1, 4), "best_epoch": best_epoch,
            "training_log": training_log,
        }, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print(f"DONE. Best checkpoint: epoch {best_epoch}, F1={best_f1:.4f} -> {output_dir / 'best_model.pt'}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
