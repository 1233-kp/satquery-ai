"""
backend/scripts/prepare_fusion_data.py
=======================================
Prepares co-registered BigEarthNet optical (Sentinel-2) + SAR (Sentinel-1)
pairs with weak multi-label land-cover supervision for the fusion net.

Source: ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2 (HF
Hub) -- a pre-packaged ~13.7K-pair subset of official BigEarthNet
train/validation/test splits (7180/3255/3248), avoiding the ~51-59GB
Zenodo Sentinel-1/-2 archives (which were down/timing out when this was
built) entirely. metadata.parquet carries the real multi-label land-cover
classes per patch (the official BigEarthNet-19 set) plus the s1_name that
links each optical patch to its SAR counterpart -- no text-mining needed.

Per patch:
  - Optical: 10-band uint16 GeoTIFF (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
    in ascending order -- the standard BigEarthNet toolkit convention, no
    embedded band-name metadata to confirm against, so cross-checked via
    the sample composite images this script saves). RGB assembled from
    B04/B03/B02 (indices 2,1,0) with a 2-98% percentile stretch, saved as
    PNG for fast loading during training.
  - SAR: 2-band float32 GeoTIFF (VV, VH), already calibrated to dB
    (confirmed empirically: values are negative, in the -35 to -3 dB
    range typical of gamma0 backscatter) -- saved as-is in a .npy, no unit
    conversion, no percentile stretch (that would destroy the calibrated
    scale a real backscatter value needs to retain).
  - Labels: the official BigEarthNet-19 multi-label set, one-hot encoded
    directly from metadata.parquet's `labels` column.

Usage:
    python scripts/prepare_fusion_data.py
    python scripts/prepare_fusion_data.py --max-per-split 2000   # faster iteration
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
ZIP_REPO = "ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2"
ZIP_FILE = "BigEarthNet_14K.zip"
METADATA_FILE = "metadata.parquet"

# Ascending BigEarthNet-19 order (10m + 20m bands only, no 60m B01/B09).
S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
RGB_INDICES = (2, 1, 0)  # B04, B03, B02 -> R, G, B

BIGEARTHNET_19_CLASSES = [
    "Urban fabric", "Industrial or commercial units", "Arable land", "Permanent crops",
    "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation", "Transitional woodland, shrub",
    "Beaches, dunes, sands", "Inland wetlands", "Coastal wetlands", "Inland waters",
    "Marine waters",
]
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def parse_args():
    p = argparse.ArgumentParser(description="Prepare BigEarthNet optical+SAR fusion training data.")
    p.add_argument("--output-dir", type=str, default=str(_BACKEND_ROOT / "data" / "fusion"))
    p.add_argument("--max-per-split", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=3, help="Sample composites to save for visual review.")
    return p.parse_args()


def _percentile_stretch(ch: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(ch, pmin), np.percentile(ch, pmax)
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((ch - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def assemble_optical_rgb(bands_u16: np.ndarray) -> Image.Image:
    r, g, b = (_percentile_stretch(bands_u16[i].astype(np.float32)) for i in RGB_INDICES)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def label_vector(labels: list) -> list:
    present = set(labels)
    return [1 if c in present else 0 for c in BIGEARTHNET_19_CLASSES]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print("[1/4] Downloading/locating metadata + image zip (cached if already fetched)...", flush=True)
    zip_path = hf_hub_download(repo_id=ZIP_REPO, filename=ZIP_FILE, repo_type="dataset")
    meta_path = hf_hub_download(repo_id=ZIP_REPO, filename=METADATA_FILE, repo_type="dataset")

    meta_table = pq.read_table(meta_path, columns=["patch_id", "labels", "s1_name"])
    meta_by_patch = {row["patch_id"]: row for row in meta_table.to_pylist()}
    print(f"      metadata: {len(meta_by_patch)} patches indexed", flush=True)

    zf = zipfile.ZipFile(zip_path)
    all_names = zf.namelist()
    all_names_set = set(all_names)

    summary = {}
    for zip_split, out_split in SPLIT_MAP.items():
        print(f"[2/4] Processing split '{zip_split}' -> '{out_split}'...", flush=True)
        s2_dir = f"BEN_14k/BigEarthNet-S2/{zip_split}/"
        s2_names = [n for n in all_names if n.startswith(s2_dir) and n.endswith(".tif")]
        if args.max_per_split:
            s2_names = s2_names[: args.max_per_split]

        split_dir = output_dir / out_split
        (split_dir / "optical").mkdir(parents=True, exist_ok=True)
        (split_dir / "sar").mkdir(parents=True, exist_ok=True)

        manifest = []
        skipped = 0
        for i, s2_name in enumerate(s2_names):
            patch_id = Path(s2_name).stem
            meta = meta_by_patch.get(patch_id)
            if meta is None:
                skipped += 1
                continue
            s1_name = meta["s1_name"]
            s1_path_in_zip = f"BEN_14k/BigEarthNet-S1/{zip_split}/{s1_name}.tif"
            if s1_path_in_zip not in all_names_set:
                skipped += 1
                continue

            with rasterio.open(io.BytesIO(zf.read(s2_name))) as src:
                s2_bands = src.read()  # (10, H, W) uint16
            with rasterio.open(io.BytesIO(zf.read(s1_path_in_zip))) as src:
                s1_bands = src.read().astype(np.float32)  # (2, H, W) float32 dB

            rgb = assemble_optical_rgb(s2_bands)
            rgb.save(split_dir / "optical" / f"{patch_id}.png")
            np.save(split_dir / "sar" / f"{patch_id}.npy", s1_bands)

            manifest.append({
                "patch_id": patch_id,
                "optical": f"{out_split}/optical/{patch_id}.png",
                "sar": f"{out_split}/sar/{patch_id}.npy",
                "labels": meta["labels"],
                "label_vector": label_vector(meta["labels"]),
            })
            if (i + 1) % 500 == 0:
                print(f"      ...{i + 1}/{len(s2_names)}", flush=True)

        with open(split_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")
        summary[out_split] = {"n_pairs": len(manifest), "skipped": skipped}
        print(f"      -> {len(manifest)} pairs written ({skipped} skipped: no metadata/SAR match)", flush=True)

    print("\n[3/4] Saving sample composites (optical | SAR-VV | SAR-VH) for visual review...", flush=True)
    save_samples(output_dir, args.num_samples)

    with open(output_dir / "manifest_summary.json", "w") as f:
        json.dump({"source": ZIP_REPO, "classes": BIGEARTHNET_19_CLASSES, "splits": summary}, f, indent=2)

    print("\n[4/4] DONE. Dataset ready at", output_dir, flush=True)
    for split, s in summary.items():
        print(f"  {split}: {s['n_pairs']} pairs", flush=True)


def save_samples(output_dir: Path, num_samples: int) -> None:
    rows = []
    with open(output_dir / "train" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    # Prefer patches with a few labels (more interesting to sanity-check) over single-label ones.
    rows.sort(key=lambda r: len(r["labels"]), reverse=True)

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for i, row in enumerate(rows[:num_samples]):
        optical = Image.open(output_dir / row["optical"]).convert("RGB")
        sar = np.load(output_dir / row["sar"])
        vv_png = Image.fromarray(_percentile_stretch(sar[1])).convert("RGB")  # VV typically higher backscatter
        vh_png = Image.fromarray(_percentile_stretch(sar[0])).convert("RGB")
        w, h = optical.size
        comp = Image.new("RGB", (w * 3 + 20, h), (255, 255, 255))
        comp.paste(optical, (0, 0))
        comp.paste(vh_png, (w + 10, 0))
        comp.paste(vv_png, (2 * w + 20, 0))
        out_path = samples_dir / f"sample_{i}.png"
        comp.save(out_path)
        print(f"  {out_path.name}: labels={row['labels']}", flush=True)
        print(f"    SAR VH range=[{sar[0].min():.1f}, {sar[0].max():.1f}] dB, "
              f"VV range=[{sar[1].min():.1f}, {sar[1].max():.1f}] dB", flush=True)


if __name__ == "__main__":
    main()
