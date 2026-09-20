"""
backend/app/services/ood_guard.py
==================================
Heuristic out-of-distribution tripwire, shared by the change detector and
the optical+SAR fusion net.

Originally built for change_service.py after the Siamese U-Net predicted
zero change on an OSCD scene with ~20% real change -- a silent failure,
with no error and no indication anything was wrong. Reused as-is for
fusion_service.py rather than writing a second copy, since the underlying
check (does this input's basic pixel/resolution statistics look like the
training distribution?) is identical regardless of which model is asking;
only the reference profile (and, for fusion, an extra SAR-stats check)
differs. See compute_levircd_reference_profile.py and
compute_fusion_reference_profile.py for how each profile is built.

What this module actually does: compares a new input's basic pixel
statistics (and GeoTIFF resolution, when available) against a reference
profile of the model's training imagery, and flags large deviations. It
cannot tell you whether the model's prediction is *correct* on an
in-distribution-looking input, and it can both miss real domain shifts (if
they don't happen to move pixel statistics much) and false-positive on
legitimate in-distribution scenes with unusual lighting/season. It is a
tripwire, not a generalization guarantee.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

Z_SCORE_THRESHOLD = 3.0
GSD_RATIO_MIN = 0.2  # flag if input resolution is <0.2x or >5x LEVIR-CD's 0.5m GSD
GSD_RATIO_MAX = 5.0

# Secondary, narrower change-detection-only check: a model reporting ~no change
# at all is only worth flagging if the input also looks somewhat atypical --
# on its own, "near zero change" is common even in-distribution (~35% of a
# real LEVIR-CD test sample) so it can't be the signal by itself. Calibrated
# against 200 real CDVQA (SECOND dataset) samples vs. 100 real in-distribution
# LEVIR-CD pairs: max_z > 2.0 co-occurring with near-zero detected change has a
# 2% false-positive rate in-distribution, while nearly doubling this guard's
# coverage on CDVQA's out-of-domain imagery (9.7% -> 18.2%) -- see
# backend/README.md Evaluation section for the full investigation. Distinct
# from Z_SCORE_THRESHOLD both in value and in what it means: this fires only
# when paired with near-zero detected change, never on its own.
NEAR_ZERO_CHANGE_Z_THRESHOLD = 2.0
NEAR_ZERO_CHANGE_PCT_MAX = 0.01  # detected percent_changed at/below this counts as "near zero"

_PROFILE_CACHE: Optional[dict] = None


def _load_profile(path: str) -> Optional[dict]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        _PROFILE_CACHE = json.load(f)
    return _PROFILE_CACHE


def _channel_zscores(img_float: np.ndarray, ref_mean: list, ref_std: list) -> np.ndarray:
    """img_float: (H, W, C) array, C matching len(ref_mean)/len(ref_std) --
    3 for optical RGB, 2 for SAR VV/VH. Not hardcoded to 3 so the same
    function serves both change_service.py (RGB) and fusion_service.py
    (RGB for the optical check, 2-channel for the SAR check)."""
    c = len(ref_mean)
    mean = img_float.reshape(-1, c).mean(axis=0)
    return np.abs(mean - np.array(ref_mean)) / np.maximum(np.array(ref_std), 1e-6)


def check_ood(
    image_a_float: np.ndarray,
    image_b_float: np.ndarray,
    profile_path: str,
    geotiff_gsd_meters: Optional[float] = None,
    extra_channels_float: Optional[np.ndarray] = None,
    context_label: str = "this model's",
    detected_change_pct: Optional[float] = None,
) -> Tuple[bool, str, bool]:
    """Returns (flagged, detail, near_zero_secondary). detail is empty when
    not flagged. near_zero_secondary is True only when the secondary
    near-zero-change check (see NEAR_ZERO_CHANGE_Z_THRESHOLD above) is what
    fired -- False whenever the primary pixel-stats/GSD check fired instead,
    even if the secondary condition would also have been true, so callers
    can tell which check to attribute a flag to and surface that distinctly
    (e.g. as a differently-named trace step) rather than merging both into
    one indistinguishable "ood_guard" warning.

    image_a_float / image_b_float: (H, W, C) arrays -- for change_service.py
    these are the before/after RGB images; for fusion_service.py, pass the
    same optical image twice (there's only one optical image, not a pair --
    this degenerates cleanly to a single-image check). Must be in the same
    units the profile's channel_mean_of_means/channel_mean_of_stds were
    computed in (e.g. [0,1] for optical).
    profile_path: reference profile JSON (see compute_*_reference_profile.py).
    geotiff_gsd_meters: pixel size in meters if known from GeoTIFF metadata;
    None for ordinary PNG/JPEG (no resolution metadata).
    extra_channels_float: optional second modality to check against a
    "extra_channel_mean_of_means"/"extra_channel_std_of_means" entry in the
    profile -- used by fusion_service.py to also check the SAR branch,
    which change_service.py's profile doesn't have (so this is a no-op
    there).
    context_label: prefix used in the returned message, e.g. "this
    model's" or "the change detector's" -- kept generic by default since
    this same function now serves more than one specialist.
    detected_change_pct: the model's own detected percent-changed, passed
    in AFTER inference (change_service.py calls this function post-model
    now, not pre-model, specifically to supply this). None for
    fusion_service.py, which has no equivalent single number -- the
    secondary check is a no-op whenever this is None.
    """
    profile = _load_profile(profile_path)
    if profile is None:
        return False, "", False

    reasons = []

    z_a = _channel_zscores(image_a_float, profile["channel_mean_of_means"], profile["channel_std_of_means"])
    z_b = _channel_zscores(image_b_float, profile["channel_mean_of_means"], profile["channel_std_of_means"])
    max_z = float(max(z_a.max(), z_b.max()))
    if max_z > Z_SCORE_THRESHOLD:
        reasons.append(
            f"pixel intensity distribution differs sharply from the training imagery "
            f"(max channel z-score={max_z:.1f}, threshold={Z_SCORE_THRESHOLD:.1f})"
        )

    if extra_channels_float is not None and "extra_channel_mean_of_means" in profile:
        z_extra = _channel_zscores(
            extra_channels_float, profile["extra_channel_mean_of_means"], profile["extra_channel_std_of_means"]
        )
        max_z_extra = float(z_extra.max())
        if max_z_extra > Z_SCORE_THRESHOLD:
            reasons.append(
                f"secondary-modality statistics differ sharply from the training imagery "
                f"(max z-score={max_z_extra:.1f}, threshold={Z_SCORE_THRESHOLD:.1f})"
            )

    if geotiff_gsd_meters is not None:
        ref_gsd = profile.get("gsd_meters", 0.5)
        ratio = geotiff_gsd_meters / ref_gsd
        if ratio < GSD_RATIO_MIN or ratio > GSD_RATIO_MAX:
            reasons.append(
                f"GeoTIFF resolution ({geotiff_gsd_meters:.2f}m/px) is far from the "
                f"training resolution (~{ref_gsd}m/px)"
            )

    if reasons:
        detail = (
            f"Input imagery characteristics differ substantially from {context_label} training "
            "distribution (" + "; ".join(reasons) + "); confidence is low and results may be unreliable."
        )
        return True, detail, False

    if (
        detected_change_pct is not None
        and detected_change_pct <= NEAR_ZERO_CHANGE_PCT_MAX
        and max_z > NEAR_ZERO_CHANGE_Z_THRESHOLD
    ):
        detail = (
            f"Detected near-zero change ({detected_change_pct:.2f}%), but this input's pixel "
            f"intensity statistics are still somewhat atypical for {context_label} training "
            f"distribution (max channel z-score={max_z:.1f}, secondary threshold="
            f"{NEAR_ZERO_CHANGE_Z_THRESHOLD:.1f} -- below the primary {Z_SCORE_THRESHOLD:.1f} bar "
            "used above); a 'no significant change' result here is less trustworthy than usual."
        )
        return True, detail, True

    return False, "", False
