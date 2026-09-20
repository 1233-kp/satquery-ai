"""
backend/scripts/compute_fusion_reference_profile.py
=====================================================
Reference profile for the fusion net's OOD guard (reuses ood_guard.py,
built for change_service.py -- see that module's docstring). Computes
per-channel intensity stats for BOTH modalities: optical (RGB, matching
the "channel_mean_of_means" key ood_guard.py already expects) and SAR
(VV/VH dB, stored under "extra_channel_mean_of_means" so the same
check_ood() call can flag either modality diverging).

Usage:
    python scripts/compute_fusion_reference_profile.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _BACKEND_ROOT / "data" / "fusion"
OUTPUT_PATH = _BACKEND_ROOT / "checkpoints" / "fusion_net" / "reference_profile.json"
SAMPLE_SIZE = 500

# BigEarthNet Sentinel-2 10m/20m bands are ~10m effective GSD for the RGB
# bands used here (B04/B03/B02); Sentinel-1 GRD in this dataset is also
# ~10m. Documented fact about the training data, not computed from it.
TRAINING_GSD_METERS = 10.0


def main():
    rows = []
    with open(DATA_DIR / "train" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    random.seed(42)
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    optical_means, sar_means = [], []
    for row in sample:
        img = np.array(Image.open(DATA_DIR / row["optical"]).convert("RGB"), dtype=np.float32) / 255.0
        optical_means.append(img.reshape(-1, 3).mean(axis=0))
        sar = np.load(DATA_DIR / row["sar"]).astype(np.float32)
        sar_means.append(sar.reshape(2, -1).mean(axis=1))

    optical_means = np.array(optical_means)
    sar_means = np.array(sar_means)

    profile = {
        "source": "BigEarthNet optical+SAR fusion training set (ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2)",
        "n_images_sampled": len(optical_means),
        "gsd_meters": TRAINING_GSD_METERS,
        "channel_mean_of_means": optical_means.mean(axis=0).tolist(),
        "channel_std_of_means": optical_means.std(axis=0).tolist(),
        "extra_channel_mean_of_means": sar_means.mean(axis=0).tolist(),
        "extra_channel_std_of_means": sar_means.std(axis=0).tolist(),
        "extra_channel_note": "SAR (VH, VV) dB values, not [0,1] optical -- check_ood applies the same z-score formula regardless of scale.",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Reference profile saved to {OUTPUT_PATH}")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
