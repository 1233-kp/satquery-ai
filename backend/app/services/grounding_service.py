"""
backend/app/services/grounding_service.py
===========================================
Zero-shot open-vocabulary object detection, for "how many X are there" and
"highlight/locate X" queries.

Why this exists as a separate specialist rather than just asking the VLM:
SmolVLM-500M has no actual detection mechanism behind a counting question --
it just emits a plausible-looking token (see the Part 1 "1" bug this
routing change works around: a weak VLM asked "how many buildings are
there?" degenerates to a bare guess, and that guess then poisons unrelated
follow-up turns). Grounding DINO is a real zero-shot detector: a count here
is len(detections) from actual bounding boxes with confidence scores, not a
language-model guess, and the boxes are returned as inspectable visual
evidence.

GROUNDING_MODE=mock -> deterministic stub box, no model load (mirrors the
                       mock modes on the other specialists).
GROUNDING_MODE=real -> loads IDEA-Research/grounding-dino-tiny (172M params)
                       zero-shot, no fine-tuning, as requested.

Known limitation (measured, not assumed -- see backend/README.md
Limitations for the full per-image breakdown): tested directly on 4 real
remote-sensing images for "building" and "water body". Results are sharply
bimodal by scene type: on moderate-density scenes with individually
resolvable objects (rural houses + ponds) it's accurate -- 40 correctly
localized building boxes, both ponds found tightly. On dense industrial/
urban scenes and wide-area/low-zoom imagery, it frequently collapses to a
single box covering 80-99% of the image instead of localizing anything,
and on one image confidently (score 0.46) boxed a park as a "water body"
that isn't present at all -- a real false positive, not just an imprecise
box. Net across the 4-image test: 3/8 correct, 1/8 partial, 4/8 failed.
The score and box are still real and inspectable (unlike a VLM's opaque
numeral), but a count should be read as "detector confidence," not a
guaranteed per-instance tally, until a remote-sensing-tuned detector
(LAE-DINO was evaluated and deferred -- see README -- due to an
mmdetection/mmcv + Python 3.8/torch 1.10 stack conflict with this
project's Python 3.12/torch 2.5.1 environment) replaces it.

_apply_size_guard() below catches only one specific failure shape: a
single box covering most of the frame (the wide-area/degenerate-
localization cases above). It does NOT catch a confident, normal-sized box
pointing at the wrong content -- e.g. the measured false positive where a
29%-area box was placed over a park and labeled "water body" survives this
guard untouched, because 29% is nowhere near the oversized threshold. That
failure mode (wrong content, plausible size) is a different problem an
area check can't detect by construction; it would need a semantic check
(e.g. cross-referencing against the VQA answer, or a better detector),
which is out of scope for this guard.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings

settings = get_settings()

# A single box covering most of the image is symptomatic of the detector
# failing to localize, not a real object filling the frame -- see the
# measured failure cases in the module docstring above (a 99%-area box on a
# river estuary, a 99%-area false positive boxing a park as "water body").
# Same "don't pass an unreliable result through as if it were normal"
# discipline as hallucination_guard.py and ood_guard.py.
_OVERSIZED_BOX_AREA_FRAC = 0.75
GROUNDING_GUARD_MESSAGE = (
    "Detection did not localize confidently on this image -- likely a dense "
    "or wide-area scene outside this detector's reliable range."
)

_COUNT_RE = re.compile(r"how\s+many\s+(.+?)\s+(?:are|is)\s+there\b", re.IGNORECASE)
_COUNT_RE_FALLBACK = re.compile(r"how\s+many\s+(.+)", re.IGNORECASE)
_LOCATE_RE = re.compile(
    r"(?:highlight|locate|mark|point\s+to|where\s+is|where\s+are|show\s+me)\s+"
    r"(?:the\s+|a\s+|an\s+)?(.+)",
    re.IGNORECASE,
)


def extract_target_phrase(query: str) -> str:
    """Pulls the noun phrase to detect out of a counting/locating query --
    e.g. "How many buildings are there?" -> "buildings". Falls back to the
    whole query (stripped of punctuation) if no pattern matches, so this
    never raises on unexpected phrasing."""
    q = query.strip()
    for pattern in (_COUNT_RE, _COUNT_RE_FALLBACK, _LOCATE_RE):
        m = pattern.search(q)
        if m:
            return m.group(1).strip().rstrip("?.! ")
    return q.rstrip("?.! ")


# A follow-up that refers back to the prior turn's subject without naming it
# again -- a pronoun ("where are THEY located?") or an elided count noun
# ("how many ARE THERE?", with nothing between "how many" and "are there").
# Deliberately just these two patterns, not general anaphora resolution: the
# only case this needs to catch is a follow-up echoing what the previous
# grounding turn was already about.
_PRONOUN_REFERENT_RE = re.compile(r"\b(?:they|them|these|those|it)\b", re.IGNORECASE)
_ELIDED_COUNT_RE = re.compile(r"how\s+many\s+(?:are|is)\s+there\b", re.IGNORECASE)


def has_vague_referent(query: str) -> bool:
    """True if `query` names no concrete subject of its own and instead
    refers back to something -- see the two patterns above."""
    q = query.strip()
    return bool(_PRONOUN_REFERENT_RE.search(q) or _ELIDED_COUNT_RE.search(q))


def resolve_followup_phrase(query: str, prior_query: str) -> str:
    """Like extract_target_phrase(query), but when query has a vague
    referent (see has_vague_referent), substitutes in the noun phrase from
    the immediately preceding turn's query instead -- e.g. "where are they
    located?" after "how many ships are there?" should search for "ships",
    not the literal, meaningless phrase "they located" that a plain
    extract_target_phrase(query) would produce. Narrow and rule-based on
    purpose (no LLM call): this is context-passing, not new capability.
    Falls back to query's own extraction if the prior query doesn't yield a
    usable phrase either, so this never raises or returns an empty phrase."""
    if not has_vague_referent(query):
        return extract_target_phrase(query)
    prior_phrase = extract_target_phrase(prior_query)
    if prior_phrase and not has_vague_referent(prior_phrase):
        return prior_phrase
    return extract_target_phrase(query)


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # xyxy, pixel coords in the original image


@dataclass
class GroundingResult:
    phrase: str
    detections: list[Detection]
    guard_fired: bool = False
    guard_detail: str = ""

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def confidence(self) -> float | None:
        """Mean of the surviving detections' own scores -- real per-box
        confidence the detector already returned, not an estimate. None
        when no boxes survived (nothing to average), not a misleading 0.0."""
        if not self.detections:
            return None
        return sum(d.score for d in self.detections) / len(self.detections)


def _apply_size_guard(image_size: tuple[int, int], detections: list[Detection]) -> tuple[list[Detection], bool, str]:
    """Drops any box covering more than _OVERSIZED_BOX_AREA_FRAC of the
    image -- a real building/water-body/etc. essentially never fills the
    entire frame, so a box that large is the detector failing to localize,
    not a genuine detection. Mirrors hallucination_guard.py's approach:
    keep whatever plausible content survives, only fall back to an explicit
    "couldn't do this reliably" message when nothing usable is left."""
    img_area = image_size[0] * image_size[1]
    if img_area <= 0:
        return detections, False, ""

    kept, dropped = [], []
    for d in detections:
        x0, y0, x1, y1 = d.box
        area_frac = ((x1 - x0) * (y1 - y0)) / img_area
        (dropped if area_frac > _OVERSIZED_BOX_AREA_FRAC else kept).append(d)

    if not dropped:
        return detections, False, ""

    detail = (
        f"Suppressed {len(dropped)} box(es) covering >{_OVERSIZED_BOX_AREA_FRAC:.0%} of the "
        f"image -- treated as failed localization, not a real detection."
    )
    if kept:
        detail += f" {len(kept)} plausible detection(s) retained."
    return kept, True, detail


class GroundingService:
    def __init__(self):
        self.mode = settings.grounding_mode
        self._model = None
        self._processor = None
        if self.mode == "real":
            self._load_real_model()

    def _load_real_model(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        # Auto-detect rather than trust settings.grounding_device blindly -- same
        # defense-in-depth pattern change_service.py/fusion_service.py already use.
        # settings.grounding_device defaults to "cpu", but a copy-pasted .env (e.g.
        # from a GPU dev machine, which is exactly what happened locally when this
        # was first wired up) could set it to "cuda" and crash outright on a
        # CPU-only deploy target like Render, which has no CUDA driver at all.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(settings.grounding_model)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(settings.grounding_model)
        self._model.to(self._device)
        self._model.eval()

    def detect(self, image_path: str, phrase: str) -> GroundingResult:
        from app.services.geo_preprocessing import load_image_as_rgb

        result = self._mock_detect(image_path, phrase) if self.mode == "mock" else self._real_detect(image_path, phrase)

        img_size = load_image_as_rgb(image_path).size
        kept, guard_fired, guard_detail = _apply_size_guard(img_size, result.detections)
        result.detections = kept
        result.guard_fired = guard_fired
        result.guard_detail = guard_detail
        return result

    def _mock_detect(self, image_path: str, phrase: str) -> GroundingResult:
        from app.services.geo_preprocessing import load_image_as_rgb

        img = load_image_as_rgb(image_path)
        w, h = img.size
        return GroundingResult(
            phrase=phrase,
            detections=[Detection(label=phrase, score=0.99, box=(w * 0.1, h * 0.1, w * 0.4, h * 0.4))],
        )

    def _real_detect(self, image_path: str, phrase: str) -> GroundingResult:
        import torch
        from app.services.geo_preprocessing import load_image_as_rgb

        img = load_image_as_rgb(image_path)
        # Grounding DINO's text-prompt convention: lowercase, ending in a period.
        text_prompt = phrase.lower().strip()
        if not text_prompt.endswith("."):
            text_prompt += "."

        inputs = self._processor(images=img, text=text_prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=settings.grounding_box_threshold,
            text_threshold=settings.grounding_text_threshold,
            target_sizes=[img.size[::-1]],  # (height, width)
        )[0]

        detections = [
            Detection(
                label=(label or phrase).strip(),
                score=float(score),
                box=tuple(float(v) for v in box),
            )
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"])
        ]
        return GroundingResult(phrase=phrase, detections=detections)


@lru_cache
def get_grounding_service() -> GroundingService:
    return GroundingService()
