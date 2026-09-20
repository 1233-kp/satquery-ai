"""
Inspects uploaded images: format, dimensions, band count, CRS (if geotiff),
and a heuristic modality guess (optical vs SAR). This is the "compatibility
checking" layer the problem statement asks for -- it runs before any model
is invoked so the orchestrator can reject/route incompatible inputs early.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_RASTERIO = False


@dataclass
class ImageMeta:
    format: str
    width: int
    height: int
    bands: int
    crs: Optional[str]
    modality: str  # optical | sar | unknown


# Delimiter-bounded token matching, not a bare substring check -- "s1" as a
# plain substring false-matches inside "ROIs1970_fall_s2_..." (the "s1" in
# "ROIs1970" itself), which previously misclassified that optical file as SAR.
# Requires a filename delimiter (or start/end of string) on both sides of the
# token, so "ROIs1970" doesn't match "s1" but "..._s1_..." or "sar_..." does.
_SAR_TOKEN_RE = re.compile(r"(?:^|[_\-.])(?:sar|s1|radar)(?:[_\-.]|$)", re.IGNORECASE)
_OPTICAL_TOKEN_RE = re.compile(r"(?:^|[_\-.])(?:optical|s2)(?:[_\-.]|$)", re.IGNORECASE)


def _guess_modality(bands: int, filename: str) -> str:
    name = filename.lower()
    if _SAR_TOKEN_RE.search(name):
        return "sar"
    if _OPTICAL_TOKEN_RE.search(name):
        return "optical"
    # heuristic: SAR products are commonly single/dual-band (VV/VH),
    # optical/multispectral are commonly 3+ bands (RGB or more)
    if bands <= 2:
        return "sar"
    return "optical"


def inspect_image(filepath: str, original_filename: str) -> ImageMeta:
    ext = Path(filepath).suffix.lower().lstrip(".")

    if ext in ("tif", "tiff") and HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            bands = src.count
            width, height = src.width, src.height
            crs = str(src.crs) if src.crs else None
        return ImageMeta(
            format=ext, width=width, height=height, bands=bands,
            crs=crs, modality=_guess_modality(bands, original_filename),
        )

    # fallback: PNG/JPEG or TIFF without rasterio available
    with Image.open(filepath) as img:
        width, height = img.size
        bands = len(img.getbands())
    return ImageMeta(
        format=ext, width=width, height=height, bands=bands,
        crs=None, modality=_guess_modality(bands, original_filename),
    )


def check_pair_compatibility(meta_a: ImageMeta, meta_b: ImageMeta) -> list[str]:
    """Returns a list of warning strings; empty list == fully compatible."""
    warnings: list[str] = []
    if meta_a.width != meta_b.width or meta_a.height != meta_b.height:
        warnings.append(
            f"Dimension mismatch: {meta_a.width}x{meta_a.height} vs "
            f"{meta_b.width}x{meta_b.height}. Results may need resampling."
        )
    if meta_a.crs and meta_b.crs and meta_a.crs != meta_b.crs:
        warnings.append(f"CRS mismatch: {meta_a.crs} vs {meta_b.crs}.")
    return warnings
