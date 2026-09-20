"""
backend/scripts/compute_levircd_reference_profile.py
======================================================
Computes a lightweight reference profile of LEVIR-CD's training imagery
(per-channel intensity statistics + known ground sample distance) and saves
it for the OOD guard in change_service.py to compare new inputs against.

This is NOT a generalization fix -- it's a heuristic tripwire. It can only
catch inputs whose basic pixel statistics or resolution look obviously
different from LEVIR-CD; it says nothing about whether the model's
predictions are correct on in-distribution-looking inputs.

Usage:
    python scripts/compute_levircd_reference_profile.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_ROOT / "data" / "levircd"
OUTPUT_PATH = _BACKEND_ROOT / "checkpoints" / "change_unet" / "reference_profile.json"

# LEVIR-CD (Chen & Shi, 2020): 0.5m/pixel Google Earth imagery. This is a
# documented fact about the dataset, not something computed from the tiles
# themselves (the PNG tiles carry no geospatial metadata).
LEVIRCD_GSD_METERS = 0.5
SAMPLE_SIZE = 400


def main():
    rows = []
    with open(DATA_DIR / "train" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    random.seed(42)
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    channel_means = []
    channel_stds = []
    for row in sample:
        for key in ("imageA", "imageB"):
            img = np.array(Image.open(DATA_DIR / row[key]).convert("RGB"), dtype=np.float32) / 255.0
            channel_means.append(img.reshape(-1, 3).mean(axis=0))
            channel_stds.append(img.reshape(-1, 3).std(axis=0))

    channel_means = np.array(channel_means)
    channel_stds = np.array(channel_stds)

    profile = {
        "source": "LEVIR-CD official train split (ericyu/LEVIRCD_Cropped256)",
        "n_images_sampled": len(channel_means),
        "gsd_meters": LEVIRCD_GSD_METERS,
        "tile_size_px": 256,
        # Distribution of per-image channel means, across training images --
        # this is what a new image's own channel mean gets compared against.
        "channel_mean_of_means": channel_means.mean(axis=0).tolist(),
        "channel_std_of_means": channel_means.std(axis=0).tolist(),
        "channel_mean_of_stds": channel_stds.mean(axis=0).tolist(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Reference profile saved to {OUTPUT_PATH}")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
