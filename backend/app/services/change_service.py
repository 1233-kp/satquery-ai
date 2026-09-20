"""
Bi-temporal change detection + change-VQA.

CHANGE_MODE=mock -> deterministic stub mask/stats.
CHANGE_MODE=real -> loads the trained Siamese U-Net checkpoint (Day 3) and
                     computes a real binary change mask + area stats, then
                     renders them into a natural-language description via a
                     fixed template -- no VLM call in this path.

The change-VQA description used to be VLM-generated (grounded in detector
stats via prompting). That approach hit four distinct failure modes in
testing: hallucinated metadata (a country, a climate zone), a repetition
loop, collapsing to a bare "yes"/"no", and -- even after tightening the
prompt and forcing min_new_tokens -- a self-contradicting answer ("no."
followed by a description of a detected change) that also asserted a
"parking lot" label the detector has no way to verify (it's a binary
change mask, not a semantic classifier). A final constrained test asked
the model to only *rephrase* a fact-complete sentence, changing no numbers
-- it still altered the percentages (19.1%->20.9%, 73%->72%). At that
point the VLM was dropped from this path entirely: percent changed,
severity, and coarse location are the only things the detector can
actually verify, so the description states exactly those and nothing
else -- no claims about what changed (building/road/vegetation/etc.).
"""
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings

settings = get_settings()

_MODEL_INPUT_SIZE = 256  # matches the resolution SiameseUNet was trained at (LEVIR-CD tiles)


def _mask_location_summary(mask: np.ndarray) -> str:
    """Derives a coarse location description directly from the mask's own
    pixel positions -- computed, not guessed by the VLM, so this can't be a
    source of fabrication the way a free-form "where did it change" answer
    from the model could be."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return "not localized to any particular region"

    v_half = np.where(ys < h / 2, "top", "bottom")
    h_half = np.where(xs < w / 2, "left", "right")
    quadrants = np.char.add(np.char.add(v_half, "-"), h_half)
    labels, counts = np.unique(quadrants, return_counts=True)
    total = counts.sum()
    order = np.argsort(-counts)
    parts = [f"{labels[i]} ({counts[i] / total:.0%})" for i in order if counts[i] / total > 0.15]
    return "concentrated in the " + " and ".join(parts) if parts else "distributed across the scene"


def _deterministic_description(stats: "ChangeStats", location: str) -> str:
    """The only things the detector can actually verify: percent changed,
    severity, and coarse location (derived from the mask itself, not
    guessed). Deliberately makes no claim about *what* changed -- a binary
    change mask has no semantic label for building/road/vegetation/etc.,
    so asserting one would be exactly the kind of unverifiable claim that
    got the VLM dropped from this path."""
    return (
        f"A {stats.severity} level of change was detected between the two dates, "
        f"affecting approximately {stats.percent_changed}% of the scene "
        f"({stats.changed_pixels}/{stats.total_pixels} pixels). The change is {location}."
    )


@dataclass
class ChangeStats:
    percent_changed: float
    changed_pixels: int
    total_pixels: int
    severity: str  # low | moderate | high


class ChangeService:
    def __init__(self):
        self.mode = settings.change_mode
        self._model = None
        if self.mode == "real":
            self._load_real_model()

    def _load_real_model(self):
        import torch
        from app.services.models.siamese_unet import SiameseUNet

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SiameseUNet()
        state = torch.load(settings.change_detection_checkpoint, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    def detect_change(self, image_a_path: str, image_b_path: str) -> tuple[np.ndarray, ChangeStats, bool, str, bool, float | None]:
        if self.mode == "mock":
            # No real sigmoid output in mock mode (naive pixel-diff threshold) -- no
            # confidence to report, not a fabricated stand-in number.
            mask, stats = self._mock_change(image_a_path, image_b_path)
            return mask, stats, False, "", False, None
        return self._real_change(image_a_path, image_b_path)

    def _mock_change(self, image_a_path: str, image_b_path: str) -> tuple[np.ndarray, ChangeStats]:
        img_a = np.array(Image.open(image_a_path).convert("L"))
        img_b = np.array(Image.open(image_b_path).convert("L"))
        h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
        diff = np.abs(img_a[:h, :w].astype(int) - img_b[:h, :w].astype(int))
        mask = (diff > 40).astype(np.uint8)  # naive threshold, stand-in for the trained model

        changed = int(mask.sum())
        total = int(mask.size)
        pct = round(100 * changed / total, 2)
        severity = "high" if pct > 15 else "moderate" if pct > 5 else "low"
        return mask, ChangeStats(pct, changed, total, severity)

    def _real_change(self, image_a_path: str, image_b_path: str) -> tuple[np.ndarray, ChangeStats, bool, str, bool, float | None]:
        import torch
        import torchvision.transforms.functional as TF
        from app.services.geo_preprocessing import get_geotiff_gsd_meters, load_image_as_rgb
        from app.services.ood_guard import check_ood

        # load_image_as_rgb handles both ordinary PNG/JPEG and GeoTIFF (uint16/float32,
        # possibly multi-band) inputs uniformly -- see geo_preprocessing.py.
        orig_a = load_image_as_rgb(image_a_path)
        output_size = orig_a.size  # (W, H) -- mask is upscaled back to this for display

        img_a = orig_a.resize((_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), Image.LANCZOS)
        img_b = load_image_as_rgb(image_b_path).resize((_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), Image.LANCZOS)

        img_a_t = TF.to_tensor(img_a).unsqueeze(0).to(self._device)
        img_b_t = TF.to_tensor(img_b).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(img_a_t, img_b_t)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        mask_native = (prob > 0.5).astype(np.uint8)
        changed = int(mask_native.sum())
        total = int(mask_native.size)
        pct = round(100 * changed / total, 2)
        severity = "high" if pct > 15 else "moderate" if pct > 5 else "low"

        # Real evidence, not an estimate: mean of the model's own pre-threshold
        # sigmoid probability, taken only over the pixels it actually classified
        # as changed -- i.e. "of the pixels the model called 'changed', how
        # strongly did it believe that." None (not 0.0) when nothing was
        # classified as changed, since there's no changed-pixel subset to average.
        confidence = float(prob[mask_native.astype(bool)].mean()) if changed > 0 else None

        # OOD tripwire: compare the same resized images the model actually sees against
        # the LEVIR-CD training reference profile -- see ood_guard.py for what this can
        # and can't catch. Not a generalization fix, just a heuristic safety net. Runs
        # AFTER inference (not before) so the secondary near-zero-change check can see
        # the model's own detected_change_pct -- see ood_guard.py's module-level
        # NEAR_ZERO_CHANGE_Z_THRESHOLD comment for why that's a separate, later check.
        gsd = get_geotiff_gsd_meters(image_a_path) or get_geotiff_gsd_meters(image_b_path)
        profile_path = str(Path(settings.change_detection_checkpoint).parent / "reference_profile.json")
        img_a_float = np.array(img_a, dtype=np.float32) / 255.0
        img_b_float = np.array(img_b, dtype=np.float32) / 255.0
        ood_flag, ood_detail, ood_near_zero = check_ood(
            img_a_float, img_b_float, profile_path, geotiff_gsd_meters=gsd, detected_change_pct=pct,
        )

        # Upscale (nearest-neighbor, to keep the mask binary/sharp) back to the
        # original image resolution -- otherwise the overlay is a fixed 256x256
        # patch regardless of how large the actual uploaded image is.
        mask_img = Image.fromarray(mask_native * 255).resize(output_size, Image.NEAREST)
        mask = (np.array(mask_img) > 0).astype(np.uint8)

        return mask, ChangeStats(pct, changed, total, severity), ood_flag, ood_detail, ood_near_zero, confidence

    def describe_change(self, image_a_path: str, image_b_path: str, query: str) -> tuple[str, ChangeStats, np.ndarray, bool, str, bool, float | None]:
        # `query` is accepted for interface stability (orchestrator.py passes
        # request.query positionally) but unused: the description is now a fixed
        # template over detector stats, not a query-conditioned VLM answer. See
        # the module docstring for why the VLM was dropped from this path.
        mask, stats, ood_flag, ood_detail, ood_near_zero, confidence = self.detect_change(image_a_path, image_b_path)
        location = _mask_location_summary(mask)

        if stats.changed_pixels == 0:
            full_answer = "No significant change was detected between the two images."
        else:
            full_answer = _deterministic_description(stats, location)

        if ood_flag:
            tag = "ood_guard_near_zero_change" if ood_near_zero else "ood_guard"
            full_answer += f"\n\n[{tag}: {ood_detail}]"
        return full_answer, stats, mask, ood_flag, ood_detail, ood_near_zero, confidence


@lru_cache
def get_change_service() -> ChangeService:
    return ChangeService()
