"""
backend/scripts/prepare_levircd.py
===================================
Materializes the official LEVIR-CD building change detection dataset (bi-
temporal 1024x1024 Google Earth pairs, pre-cropped here into non-overlapping
256x256 tiles: 445/64/128 official train/val/test pairs -> 7120/1024/2048
tiles) from ericyu/LEVIRCD_Cropped256 on the HF Hub -- a single small
self-contained parquet per split with images embedded as PNG bytes, so no
separate multi-GB archive download is needed (unlike BigEarthNet's raw
Sentinel-2 archive).

Writes PNG triplets (imageA, imageB, label) per split under
backend/data/levircd/<split>/, a manifest.jsonl per split, and a handful of
side-by-side (A | B | mask) sample composites for visual sanity-checking
before training starts.

Usage:
    python scripts/prepare_levircd.py
    python scripts/prepare_levircd.py --max-per-split 500   # faster iteration
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

REPO_ID = "ericyu/LEVIRCD_Cropped256"
SPLIT_FILES = {
    "train": "data/train-00000-of-00001-737f96f51caac8cd.parquet",
    "val": "data/val-00000-of-00001-d09d88a7419f2427.parquet",
    "test": "data/test-00000-of-00001-31d7c3e3444e5b5d.parquet",
}


def parse_args():
    p = argparse.ArgumentParser(description="Materialize LEVIR-CD from the HF Hub parquet mirror.")
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "levircd"))
    p.add_argument("--max-per-split", type=int, default=None, help="Cap tiles per split (default: use all).")
    p.add_argument("--num-samples", type=int, default=4, help="Side-by-side sample composites to save for visual review.")
    return p.parse_args()


def materialize_split(repo_file: str, split: str, output_dir: Path, max_per_split: Optional[int]) -> dict:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print(f"[{split}] Downloading {REPO_ID}/{repo_file}...", flush=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=repo_file, repo_type="dataset")

    split_dir = output_dir / split
    for sub in ("imageA", "imageB", "label"):
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    n_use = min(n_total, max_per_split) if max_per_split else n_total

    manifest = []
    changed_fracs = []
    idx = 0
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if idx >= n_use:
                break
            img_a = Image.open(io.BytesIO(row["imageA"]["bytes"])).convert("RGB")
            img_b = Image.open(io.BytesIO(row["imageB"]["bytes"])).convert("RGB")
            label = Image.open(io.BytesIO(row["label"]["bytes"])).convert("L")

            name = f"{idx:05d}.png"
            img_a.save(split_dir / "imageA" / name)
            img_b.save(split_dir / "imageB" / name)
            label.save(split_dir / "label" / name)

            frac = float(np.mean(np.array(label) > 127))
            changed_fracs.append(frac)
            manifest.append({
                "id": name,
                "imageA": f"{split}/imageA/{name}",
                "imageB": f"{split}/imageB/{name}",
                "label": f"{split}/label/{name}",
                "changed_frac": round(frac, 4),
            })
            idx += 1
        if idx >= n_use:
            break

    with open(split_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    stats = {
        "n_tiles": len(manifest),
        "mean_changed_frac": round(float(np.mean(changed_fracs)), 4) if changed_fracs else 0.0,
        "n_tiles_with_no_change": sum(1 for c in changed_fracs if c < 1e-6),
    }
    print(f"[{split}] -> {stats['n_tiles']} tiles, mean changed pixels={stats['mean_changed_frac']:.2%}, "
          f"{stats['n_tiles_with_no_change']} tiles with zero change", flush=True)
    return stats


def save_sample_composites(output_dir: Path, num_samples: int) -> None:
    """Side-by-side (imageA | imageB | label) composites for visual sanity-checking."""
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "train" / "manifest.jsonl"
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Prefer tiles with visible change so the samples are actually informative.
    rows.sort(key=lambda r: r["changed_frac"], reverse=True)
    chosen = rows[:num_samples]

    for i, row in enumerate(chosen):
        img_a = Image.open(output_dir / row["imageA"]).convert("RGB")
        img_b = Image.open(output_dir / row["imageB"]).convert("RGB")
        label = Image.open(output_dir / row["label"]).convert("RGB")
        w, h = img_a.size
        composite = Image.new("RGB", (w * 3 + 20, h), color=(255, 255, 255))
        composite.paste(img_a, (0, 0))
        composite.paste(img_b, (w + 10, 0))
        composite.paste(label, (2 * w + 20, 0))
        out_path = samples_dir / f"sample_{i}_changed_{row['changed_frac']:.0%}.png"
        composite.save(out_path)
        print(f"  sample saved: {out_path} (A | B | mask, changed={row['changed_frac']:.1%})", flush=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("Preparing LEVIR-CD (official split, pre-cropped 256x256 tiles)", flush=True)
    print("=" * 70, flush=True)

    summary = {}
    for split, repo_file in SPLIT_FILES.items():
        summary[split] = materialize_split(repo_file, split, output_dir, args.max_per_split)

    print("\nSaving sample composites for visual sanity check...", flush=True)
    save_sample_composites(output_dir, args.num_samples)

    with open(output_dir / "manifest_summary.json", "w") as f:
        json.dump({"source": REPO_ID, "splits": summary}, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("DONE. Dataset ready at", output_dir, flush=True)
    for split, s in summary.items():
        print(f"  {split}: {s['n_tiles']} tiles, mean changed={s['mean_changed_frac']:.2%}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
