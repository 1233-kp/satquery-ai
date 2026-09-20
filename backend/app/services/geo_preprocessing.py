"""
backend/app/services/geo_preprocessing.py
==========================================
Loads an image for the change detector, handling both ordinary PNG/JPEG
(uint8, already display-ready) and GeoTIFF (which the problem statement's
real evaluation imagery will be -- typically uint16 or float32 reflectance
values, and sometimes more than 3 bands).

PIL can open plain uint8 TIFFs fine, but it doesn't know how to rescale
uint16/float32 pixel values into a sane 0-255 range, and it has no concept
of "pick the right 3 bands out of N" -- both of those need to be handled
explicitly, or a GeoTIFF loaded naively renders as solid black/white or
picks nonsensical channels. That's what this module does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

GEOTIFF_EXTENSIONS = {".tif", ".tiff"}


def _percentile_stretch(channel: np.ndarray, pmin: float = 2.0, pmax: float = 98.0) -> np.ndarray:
    """Robust contrast stretch: maps the [pmin, pmax] percentile range to
    [0, 255]. Needed for uint16/float32 reflectance data, which doesn't sit
    in a 0-255 range and would otherwise render as near-black or blown out."""
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(finite, pmin), np.percentile(finite, pmax)
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((channel - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_image_as_rgb(path: str, band_order: Optional[Tuple[int, int, int]] = None) -> Image.Image:
    """Returns a display-ready PIL RGB image regardless of source format.

    band_order: optional explicit 0-indexed (R, G, B) band indices for a
    multi-band GeoTIFF. If not given, a heuristic is used:
      - 1 band  -> grayscale replicated across R/G/B
      - 3 bands -> assumed already ordered (R, G, B)
      - 4+ bands -> assumed Sentinel-2-style (Blue, Green, Red, NIR, ...)
        ordering, so bands (2, 1, 0) are picked for true color (R, G, B).
        This is a documented heuristic, not a universal rule -- a sensor
        with a different band order needs an explicit band_order.
    """
    ext = Path(path).suffix.lower()
    if ext not in GEOTIFF_EXTENSIONS:
        return Image.open(path).convert("RGB")

    import rasterio

    with rasterio.open(path) as src:
        count = src.count
        dtype = src.dtypes[0]

        if band_order is not None:
            idx0 = list(band_order)
        elif count >= 4:
            idx0 = [2, 1, 0]
        elif count == 3:
            idx0 = [0, 1, 2]
        else:
            idx0 = [0, 0, 0]  # single band -> grayscale-as-RGB

        raw = src.read([i + 1 for i in idx0]).astype(np.float32)  # rasterio bands are 1-indexed; shape (3, H, W)

    if dtype == "uint8":
        channels = [raw[c].astype(np.uint8) for c in range(3)]
    else:
        # uint16 / int16 / float32 / etc: percentile-stretch each band independently.
        channels = [_percentile_stretch(raw[c]) for c in range(3)]

    rgb = np.stack(channels, axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def load_sar_backscatter(path: str, num_channels: int = 2) -> Tuple[np.ndarray, str, float]:
    """Loads a SAR image (GeoTIFF or plain PNG/JPEG fallback) as calibrated
    dB backscatter, returning (array of shape (num_channels, H, W) float32
    with NaNs imputed -- see below, a note describing what conversion -- if
    any -- was applied, and the fraction of pixels that were NaN/no-data
    before imputation).

    Real SAR products commonly have genuine NaN pixels -- no-data borders,
    layover/shadow regions the radar geometry can't resolve -- this is
    normal for real data, not corruption. Left unhandled, a NaN silently
    propagates through anything downstream that doesn't explicitly guard
    for it: a plain .mean() becomes NaN, and a single NaN pixel in a CNN's
    input contaminates its entire receptive field, which is how one
    no-data border turned into NaN for every one of the model's 19 output
    classes despite covering only ~6% of the scene. So NaN pixels are
    imputed here (per-channel mean of the valid pixels -- a simple fill,
    not real inpainting) before the array is returned, and the NaN
    fraction is always returned alongside so the caller can decide whether
    to just report it or escalate to a low-confidence warning.

    Real-world SAR GeoTIFFs vary in what they actually store, and there's
    no universal way to tell from the file alone, so this uses a
    best-effort heuristic on the value distribution:
      - Mostly-negative float values in a plausible dB range (roughly -60
        to +10) -> already calibrated dB, used as-is. This is what
        BigEarthNet's Sentinel-1 GRD products (and this model's training
        data) actually are -- verified empirically, not assumed.
      - Small positive float values (~0-2) -> looks like linear power
        (sigma-nought/gamma-nought before the log conversion) -> converted
        via 10*log10.
      - Large positive values (uint16 DN or big floats) -> no real
        calibration is possible without sensor-specific constants (e.g.
        RISAT calibration LUTs), which aren't available here. A labeled
        proxy conversion is applied and clearly flagged as a proxy in the
        returned note -- NOT a substitute for real radiometric
        calibration. Treat any composition estimate derived from this
        path with proportionally less confidence.

    Channel count handling: 1 band is replicated to fill num_channels
    (e.g. single-pol SAR duplicated into both dual-pol slots); 2 bands
    used as-is; >2 bands takes the first `num_channels`.
    """
    ext = Path(path).suffix.lower()
    if ext in GEOTIFF_EXTENSIONS:
        import rasterio

        with rasterio.open(path) as src:
            raw = src.read().astype(np.float32)  # (C, H, W)
    else:
        # Non-GeoTIFF SAR input (e.g. a pre-rendered grayscale PNG stand-in) --
        # no calibration metadata possible; treat pixel values as an uncalibrated
        # proxy from the start.
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        raw = arr[np.newaxis, :, :]

    channels = list(raw)
    if len(channels) == 1:
        channels = channels * num_channels
    elif len(channels) > num_channels:
        channels = channels[:num_channels]
    elif len(channels) < num_channels:
        channels = (channels * num_channels)[:num_channels]
    raw = np.stack(channels, axis=0)

    # Sign is the primary signal, not just magnitude: linear power and raw digital
    # numbers are intensity values and are never negative, while calibrated dB
    # backscatter for real land cover is virtually always negative (occasionally
    # slightly positive only for extreme corner-reflector targets). Checking
    # magnitude alone before sign let a small-positive "linear power" test case
    # (mean ~0.25) wrongly match the dB range below zero -- caught by testing
    # against synthetic linear-power/uncalibrated-DN files, not just real data.
    mean_val = float(np.nanmean(raw))
    if mean_val < 0:
        db = raw
        note = "used as-is (negative mean is consistent with already-calibrated dB)"
    elif mean_val <= 2:
        db = 10 * np.log10(np.clip(raw, 1e-10, None))
        note = "converted from linear power (sigma/gamma-nought) to dB via 10*log10"
    else:
        # Uncalibrated digital numbers -- no sensor calibration constants available.
        db = 20 * np.log10(np.clip(raw, 1, None)) - 83
        note = ("proxy dB conversion from uncalibrated digital numbers -- no real "
                "radiometric calibration constants were available for this input")

    db = db.astype(np.float32)
    nan_mask = ~np.isfinite(db)
    nan_fraction = float(nan_mask.mean())
    if nan_mask.any():
        for c in range(db.shape[0]):
            channel = db[c]
            valid = channel[np.isfinite(channel)]
            fill_value = float(valid.mean()) if valid.size else 0.0
            channel[~np.isfinite(channel)] = fill_value

    return db, note, nan_fraction


def get_geotiff_gsd_meters(path: str) -> Optional[float]:
    """Returns the ground sample distance (pixel size) in meters, if the
    file is a GeoTIFF with resolvable resolution; None for ordinary
    PNG/JPEG (no geospatial metadata) or a GeoTIFF with no CRS/transform.

    For a geographic CRS (degrees, not meters), this converts using a
    fixed ~111,320 m/degree approximation -- accurate enough for a coarse
    "is this roughly the same ballpark as 0.5m LEVIR-CD imagery" check,
    not for precise measurement (it ignores longitude compression with
    latitude).
    """
    if Path(path).suffix.lower() not in GEOTIFF_EXTENSIONS:
        return None

    import rasterio

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            x_res, y_res = src.res
            if src.crs.is_geographic:
                x_res *= 111_320
                y_res *= 111_320
            return float((x_res + y_res) / 2)
    except Exception:
        return None
