"""
backend/app/services/hallucination_guard.py
=============================================
Post-processing safety check for VQA/captioning answers.

The BigEarthNet.txt-adapted LoRA was fine-tuned on a narrow, ~3,600-example,
Europe-only dataset whose captions always name one of 10 countries and
report precise land-cover areas in square meters. On complex or
out-of-distribution scenes, the model can regurgitate that learned pattern --
inventing a country/capital and an area figure that have nothing to do with
the actual image (documented finding: "Austria... 1,044,000 square meters"
recurring verbatim across unrelated photos). The area figures aren't always
suspiciously round either -- a decoding fix for a separate repetition-loop
bug revealed the same fabrication in other units/precisions ("1,000,072
square meters", "120,000 square kilometers"), because the real problem is
that this model has no way to derive a physical area from a single photo at
all, regardless of how the number is formatted.

This is a known capacity ceiling of a rank-8 LoRA on a 500M model (see
backend/README.md Limitations), not something a prompt or routing change
can fix. This module is the guardrail: it flags and strips those specific
fabricated-detail patterns before an answer reaches the user, rather than
letting a confident-sounding wrong answer go out during a demo.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# The exact 10 countries BigEarthNet.txt's captions are drawn from, plus
# their capitals (fabricated answers sometimes name the capital instead of
# the country -- e.g. "Vienna" instead of "Austria"). Small and hardcoded on
# purpose: this only needs to catch this one dataset's specific leakage, not
# be a general-purpose geo-NER.
_BIGEARTHNET_COUNTRIES = [
    "austria", "belgium", "finland", "ireland", "kosovo",
    "lithuania", "luxembourg", "portugal", "serbia", "switzerland",
]
_BIGEARTHNET_CAPITALS = [
    "vienna", "brussels", "helsinki", "dublin", "pristina",
    "vilnius", "luxembourg city", "lisbon", "belgrade", "bern",
]
FLAGGED_LOCATIONS = _BIGEARTHNET_COUNTRIES + _BIGEARTHNET_CAPITALS

# Matches any area/size claim in absolute units (square meters/km, sqm, acres,
# hectares), round or not. Originally this only matched the rigid
# round-thousand pattern the fixed BigEarthNet.txt caption template produces
# (e.g. "1,044,000 square meters") -- but changing the generation config to
# stop a repetition loop (see vqa_service.py) revealed the same model, once
# no longer looping, fabricates area figures in other units and precisions
# too ("1,000,072 square meters", "120,000 square kilometers", "965000
# thousand acres"). The real issue isn't "is the number suspiciously round" --
# it's that this model has no way to derive a physical area/scale from a
# single photo at all, so any such claim is fabricated regardless of format.
_AREA_RE = re.compile(
    r"\b[\d,]+(?:\.\d+)?\s*(?:thousand\s+|million\s+|billion\s+)?"
    r"(?:square\s+(?:met(?:er|re)s?|kilomet(?:er|re)s?)|sq\.?\s*k?m|sqm|acres?|hectares?)\b",
    re.IGNORECASE,
)

# Matches a fabricated calendar date claim ("26 January 2017", "January 26,
# 2017", "2017-01-26", "03/14/2019"). Same justification as _AREA_RE: this
# model has no way to derive a real capture date from pixel content alone,
# so any such claim is fabricated regardless of format. Requires a full
# day+month+year (or numeric date triple), not just a lone number, so it
# doesn't collide with pixel dimensions, object counts, or bare years.
_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE,
)

# Matches a fabricated capture *time* ("captured ... at 17:27"). Anchored to
# a preceding "at"/"on" so it doesn't collide with scale ratios ("1:50000")
# or aspect ratios ("16:9"), which share the bare HH:MM shape but never
# follow that anchor word.
_TIME_RE = re.compile(
    r"\b(?:at|on)\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b",
    re.IGNORECASE,
)

# Matches a query asking for a continuous measurement (percentage, area,
# proportion, coverage) as opposed to a discrete count ("how many X") --
# counting questions are deliberately excluded here, since a bare "1" answer
# to "how many buildings" is a routing/capability gap (see grounding_service)
# rather than a fabricated-figure gap this guard is meant to catch.
_MEASUREMENT_QUERY_RE = re.compile(
    r"\b(percentage|percent|proportion|coverage|"
    r"how\s+much\s+(?:area|land|water|vegetation|space|forest)|"
    r"what\s+(?:area|percentage|percent|proportion)|"
    r"how\s+big|size\s+of)\b",
    re.IGNORECASE,
)

# Matches an answer that is *just* a bare number (optionally with a decimal
# point and/or a trailing "%"), with no supporting text at all -- e.g. "1.5"
# or "25.5" answering "how much area is of water?" / "how much percentage is
# the vegetation?". The model has no way to measure a real percentage or area
# from a single photo, so a bare number with zero qualifying context is
# exactly as fabricated as an area figure with a unit suffix -- it just isn't
# caught by _AREA_RE because it has no unit at all.
_BARE_NUMBER_ANSWER_RE = re.compile(r"^\(?\s*[\d,]+(?:\.\d+)?\s*%?\s*\)?\.?$")

SAFE_FALLBACK = "The model could not reliably determine precise regional, area, or measurement details for this scene."


def _location_pattern(term: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def check_and_clean(answer: str, query: str) -> Tuple[str, bool, str]:
    """Flags and strips fabricated country/capital or area/size claims from
    a model answer.

    A location is only flagged if it appears in `answer` but was NOT already
    present in the user's `query` -- if the user explicitly asked about
    Austria, mentioning Austria back isn't a fabrication.

    Returns (cleaned_answer, guard_fired, detail_for_trace). `detail_for_trace`
    is empty when the guard didn't fire.
    """
    if not answer:
        return answer, False, ""

    flagged_locations = sorted({
        term for term in FLAGGED_LOCATIONS
        if _location_pattern(term).search(answer) and not _location_pattern(term).search(query or "")
    })
    has_area_figure = bool(_AREA_RE.search(answer))
    has_date_or_time = bool(_DATE_RE.search(answer)) or bool(_TIME_RE.search(answer))
    has_bare_measurement = bool(_MEASUREMENT_QUERY_RE.search(query or "")) and bool(
        _BARE_NUMBER_ANSWER_RE.match(answer.strip())
    )

    if not flagged_locations and not has_area_figure and not has_date_or_time and not has_bare_measurement:
        return answer, False, ""

    if has_bare_measurement:
        # The whole answer is nothing but the fabricated number -- there's no
        # sentence structure to selectively drop from, so replace it outright.
        detail = (
            "Suppressed unverifiable bare numeric measurement answer "
            "(no model-accessible way to derive a percentage/area from a single image)."
        )
        return SAFE_FALLBACK, True, detail

    sentences = _split_sentences(answer)
    kept, dropped = [], []
    for sentence in sentences:
        hits_location = any(_location_pattern(t).search(sentence) for t in flagged_locations)
        hits_area = bool(_AREA_RE.search(sentence))
        hits_date_or_time = bool(_DATE_RE.search(sentence)) or bool(_TIME_RE.search(sentence))
        (dropped if (hits_location or hits_area or hits_date_or_time) else kept).append(sentence)

    reasons = []
    if flagged_locations:
        reasons.append(f"unstated location claim ({', '.join(flagged_locations)})")
    if has_area_figure:
        reasons.append("unverifiable area/size figure (no model-accessible scale reference)")
    if has_date_or_time:
        reasons.append("fabricated capture date/time (no model-accessible timestamp source)")
    reason_str = " and ".join(reasons)

    remaining = " ".join(kept).strip()
    if len(remaining) >= 20:
        # Enough non-fabricated content survives to stand on its own.
        cleaned = remaining
    else:
        cleaned = SAFE_FALLBACK if not remaining else f"{SAFE_FALLBACK} {remaining}"

    detail = f"Suppressed {reason_str} before returning result ({len(dropped)}/{len(sentences)} sentence(s) removed)."
    return cleaned, True, detail
