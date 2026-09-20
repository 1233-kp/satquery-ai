"""
Optical + SAR cross-modal fusion/analysis.

FUSION_MODE=mock -> deterministic stub stats (still runs real basic image
                     stats on the uploaded pair so the demo isn't literally
                     hardcoded text).
FUSION_MODE=real -> loads the dual-branch gated fusion net and computes
                     real land-cover composition estimates from its 19-class
                     multi-label output, plus real optical brightness and
                     calibrated SAR backscatter stats.

Applying the Day 2/3 lesson from the start here, not as a retrofit: every
VLM-narrated answer this week eventually hit a failure mode (fabricated
metadata, a repetition loop, a bare yes/no, a self-contradicting answer
with an invented "parking lot" label the detector had no way to verify).
So this path never calls a VLM at all. The fusion net's only job is to
produce verifiable, computed numbers -- 19-class land-cover presence
probabilities, real backscatter dB, real optical brightness -- and the
answer is a fixed template over those numbers. It makes no claim about
anything the numbers don't directly support (no object identities, no
place names beyond what's asked).
"""
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings

settings = get_settings()

_INPUT_SIZE = 128  # matches training resolution (see train_fusion_net.py)

# Real SAR scenes commonly have some NaN/no-data pixels (border, layover,
# shadow) -- always reported (see FusionStats.sar_nodata_pct), but above
# this fraction the scene is missing enough real data that a confident-
# looking composition estimate would be misleading, so it escalates to an
# explicit warning (same trace-step/warnings-list style as ood_guard.py,
# though this is a distinct concern -- data completeness, not distribution
# shift -- so it's a separate check rather than routed through check_ood).
SAR_NODATA_WARNING_THRESHOLD = 0.30

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
# Coarse bucket each official class rolls up into, for the 3-number template.
# "Beaches, dunes, sands" is genuinely neither built-up, water, nor vegetation
# (bare ground) -- excluded from all three buckets rather than forced into one.
_BUILT_UP = {"Urban fabric", "Industrial or commercial units"}
_WATER = {"Inland wetlands", "Coastal wetlands", "Inland waters", "Marine waters"}
_VEGETATION = set(BIGEARTHNET_19_CLASSES) - _BUILT_UP - _WATER - {"Beaches, dunes, sands"}


def _aggregate_to_buckets(class_probs: np.ndarray) -> tuple[float, float, float, float]:
    """Reduces 19 per-class presence probabilities to 3 coarse composition
    percentages. Takes the MAX probability within each bucket (i.e. "how
    confident is the model that at least one built-up class is present")
    and normalizes the three bucket values to sum to 100 -- this is a
    confidence-weighted composition proxy, not true pixel-area coverage,
    since BigEarthNet only provides patch-level multi-label presence, not
    a pixel segmentation ground truth. Documented here and in the answer
    template's wording ("approximately") rather than presented as exact.

    Also returns a confidence score: the raw (pre-normalization) sigmoid
    probability of whichever bucket the model is most confident about --
    the same number that decided the winning bucket, so it's a direct,
    already-computed read of the model's own certainty, not a new estimate.
    (The model's output head is per-class sigmoid, not softmax -- this is
    a multi-label presence task, not a single mutually-exclusive class
    pick -- so a sigmoid probability is what's actually available here.)"""
    by_class = dict(zip(BIGEARTHNET_19_CLASSES, class_probs))
    built_up = max((by_class[c] for c in _BUILT_UP), default=0.0)
    water = max((by_class[c] for c in _WATER), default=0.0)
    vegetation = max((by_class[c] for c in _VEGETATION), default=0.0)
    confidence = float(max(built_up, water, vegetation))
    total = built_up + water + vegetation
    if total <= 1e-8:
        return 0.0, 0.0, 0.0, confidence
    return (built_up / total * 100, water / total * 100, vegetation / total * 100, confidence)


@dataclass
class FusionStats:
    optical_mean_brightness: float
    sar_mean_backscatter_db: float
    estimated_built_up_pct: float
    estimated_water_pct: float
    estimated_vegetation_pct: float
    sar_nodata_pct: float = 0.0  # fraction (0-100) of SAR pixels that were NaN/no-data


def _deterministic_answer(stats: FusionStats) -> str:
    """The only claim this makes is the composition breakdown -- no object
    identities, no place names, nothing beyond what the fused classifier
    output and raw sensor stats directly support."""
    answer = (
        f"Joint analysis of the optical and SAR imagery indicates the scene is approximately "
        f"{stats.estimated_built_up_pct:.1f}% built-up, {stats.estimated_water_pct:.1f}% "
        f"water-covered, and {stats.estimated_vegetation_pct:.1f}% vegetation, based on fused "
        f"spectral-radar evidence (optical brightness={stats.optical_mean_brightness:.1f}, "
        f"SAR backscatter~{stats.sar_mean_backscatter_db:.1f} dB)."
    )
    # Always surfaced once it's non-trivial, not just when it's bad enough to warn on --
    # "6% no-data" is useful context even when it's not a reliability concern.
    if stats.sar_nodata_pct > 1.0:
        answer += (
            f" {stats.sar_nodata_pct:.1f}% of the SAR scene was no-data (border/layover/shadow) "
            f"and was filled with the scene's own mean backscatter rather than real signal."
        )
    return answer


class FusionService:
    def __init__(self):
        self.mode = settings.fusion_mode
        self._model = None
        if self.mode == "real":
            self._load_real_model()

    def _load_real_model(self):
        import torch
        from app.services.models.fusion_net import OpticalSARFusionNet

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = OpticalSARFusionNet()
        state = torch.load(settings.fusion_checkpoint, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

        config_path = Path(settings.fusion_checkpoint).parent / "config.json"
        import json
        with open(config_path) as f:
            config = json.load(f)
        self._sar_mean = np.array(config["sar_mean"], dtype=np.float32).reshape(-1, 1, 1)
        self._sar_std = np.array(config["sar_std"], dtype=np.float32).reshape(-1, 1, 1)

    def analyze(self, optical_path: str, sar_path: str, query: str) -> tuple[str, FusionStats, bool, str, float | None]:
        # `query` accepted for interface stability (orchestrator.py passes it
        # positionally) but unused in real mode -- the answer is a fixed
        # template over computed stats, not a query-conditioned VLM answer.
        if self.mode == "mock":
            # Mock mode's composition numbers are a simple brightness/backscatter
            # heuristic, not a real classifier output -- no confidence to report.
            stats = self._compute_mock_stats(optical_path, sar_path)
            return _deterministic_answer(stats), stats, False, "", None
        return self._real_analyze(optical_path, sar_path)

    def _real_analyze(self, optical_path: str, sar_path: str) -> tuple[str, FusionStats, bool, str, float | None]:
        import torch
        import torch.nn.functional as F
        from app.services.geo_preprocessing import get_geotiff_gsd_meters, load_image_as_rgb, load_sar_backscatter
        from app.services.ood_guard import check_ood

        optical_img = load_image_as_rgb(optical_path)
        optical_arr = np.array(optical_img, dtype=np.float32) / 255.0
        # sar_db has already had any NaN/no-data pixels imputed (per-channel mean fill) by
        # load_sar_backscatter -- sar_nan_fraction is the fraction that were NaN before that,
        # so it's still meaningful evidence even though the array itself is now NaN-free.
        sar_db, sar_note, sar_nan_fraction = load_sar_backscatter(sar_path, num_channels=2)

        profile_path = str(Path(settings.fusion_checkpoint).parent / "reference_profile.json")
        gsd = get_geotiff_gsd_meters(optical_path) or get_geotiff_gsd_meters(sar_path)
        # Optical checked against itself (fusion has one optical image, not a before/after
        # pair); SAR checked via the same function's extra_channels_float slot. The 3rd
        # return value (near-zero-change secondary check) is change_service.py-only --
        # fusion has no equivalent single "detected change" number, so it's always False
        # here and deliberately discarded.
        ood_flag, ood_detail, _ood_near_zero_unused = check_ood(
            optical_arr, optical_arr, profile_path,
            geotiff_gsd_meters=gsd,
            extra_channels_float=np.transpose(sar_db, (1, 2, 0)),
            context_label="the fusion model's",
        )

        nodata_flag = sar_nan_fraction > SAR_NODATA_WARNING_THRESHOLD
        nodata_detail = (
            f"{sar_nan_fraction:.1%} of the SAR scene is no-data/NaN (likely border, layover, "
            "or shadow regions, which is normal for real SAR products) -- composition estimates "
            "are based on a mean-filled approximation for those pixels and confidence is "
            "correspondingly lower for a scene this incomplete."
        ) if nodata_flag else ""

        optical_t = torch.from_numpy(optical_arr).permute(2, 0, 1).unsqueeze(0)
        optical_t = F.interpolate(optical_t, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False)

        sar_norm = (sar_db - self._sar_mean) / self._sar_std
        sar_t = torch.from_numpy(sar_norm).unsqueeze(0)
        sar_t = F.interpolate(sar_t, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False)

        optical_t, sar_t = optical_t.to(self._device), sar_t.to(self._device)
        with torch.no_grad():
            logits = self._model(optical_t, sar_t)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        built_up, water, vegetation, confidence = _aggregate_to_buckets(probs)
        # _aggregate_to_buckets is typed to return float but numpy arithmetic actually
        # hands back np.float32 scalars (same for sar_nan_fraction) -- harmless for the
        # f-string formatting in _deterministic_answer, but SQLite's JSON column
        # serializer chokes on them ("Object of type float32 is not JSON serializable")
        # once this gets asdict()'d into visual_evidence/trace and saved. Cast explicitly
        # rather than rely on the type hint, same as the two fields just above.
        stats = FusionStats(
            optical_mean_brightness=float(optical_arr.mean() * 255),
            sar_mean_backscatter_db=float(sar_db.mean()),  # sar_db is NaN-free at this point
            estimated_built_up_pct=float(built_up),
            estimated_water_pct=float(water),
            estimated_vegetation_pct=float(vegetation),
            sar_nodata_pct=float(sar_nan_fraction * 100),
        )

        answer = _deterministic_answer(stats)
        if sar_note != "used as-is (negative mean is consistent with already-calibrated dB)":
            answer += f" [sar_calibration: {sar_note}]"
        if ood_flag:
            answer += f"\n\n[ood_guard: {ood_detail}]"
        if nodata_flag:
            answer += f"\n\n[ood_guard: {nodata_detail}]"

        # Combined under the same (flag, detail) slot orchestrator.py already knows how to
        # surface as a trace step + warning -- distinct concerns (distribution shift vs. data
        # completeness) but the same "don't return a confident-looking answer silently" style.
        combined_flag = ood_flag or nodata_flag
        combined_detail = " ".join(d for d in (ood_detail, nodata_detail) if d)
        return answer, stats, combined_flag, combined_detail, confidence

    def _compute_mock_stats(self, optical_path: str, sar_path: str) -> FusionStats:
        optical = np.array(Image.open(optical_path).convert("L"), dtype=float)
        sar = np.array(Image.open(sar_path).convert("L"), dtype=float)

        optical_mean = float(optical.mean())
        # proxy backscatter in dB-like range from normalized SAR intensity -- mock mode
        # only ever gets a plain PNG stand-in, not real calibrated SAR, so this is
        # explicitly a display-value proxy, unlike the real model's actual dB input.
        sar_norm = sar / 255.0
        sar_db_proxy = float(10 * np.log10(np.clip(sar_norm.mean(), 1e-3, 1.0)))

        built_up = float(np.clip((optical_mean - 80) / 175 * 40, 0, 100))
        water = float(np.clip((1 - sar_norm.mean()) * 25, 0, 100))
        vegetation = float(max(0, 100 - built_up - water))

        return FusionStats(
            optical_mean_brightness=optical_mean,
            sar_mean_backscatter_db=sar_db_proxy,
            estimated_built_up_pct=built_up,
            estimated_water_pct=water,
            estimated_vegetation_pct=vegetation,
        )


@lru_cache
def get_fusion_service() -> FusionService:
    return FusionService()
