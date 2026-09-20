"""
Classifies a natural-language query + input mode into a task type.
Rule-based on purpose: fast, deterministic, fully explainable in the
execution trace (no hidden LLM call that judges can't audit). Swap in
an LLM-based classifier later if you want to handle messier phrasing --
just keep it returning the same TaskType strings.
"""
import re

from app.schemas import TaskType, InputMode

CHANGE_KEYWORDS = ["change", "before", "after", "difference", "increased", "decreased", "over time", "compare these two", "what changed"]
GROUNDING_KEYWORDS = ["highlight", "locate", "where is", "where are", "point to", "mark the", "show me the region"]
# "how many X" / "count the X" / "number of X" -- routed to the real Grounding
# DINO detector (grounding_service.py) instead of free-text VQA, since the VLM
# has no actual counting mechanism and just guesses a number (see the Part 1
# "1" bug this replaced). Narrow on purpose: only counting/locating phrasing
# is redirected here, so a general "describe..."/"what is..." query is
# untouched and still goes through vqa_service as before.
COUNTING_RE = re.compile(r"\bhow many\b|\bcount (?:the|of)\b|\bnumber of\b", re.IGNORECASE)
# Whole-phrase generic caption requests only -- NOT a bare "describe"/"caption",
# which also matches specific-intent queries like "Describe its location" or
# "Describe only visible features, do not invent measurements". Those carry a
# real user instruction that must reach the model, so they go through vqa
# instead (see vqa_service.py), which already handles arbitrary questions
# correctly and doesn't discard the query text.
CAPTION_KEYWORDS = [
    "caption this image", "caption the image", "give a caption", "provide a caption",
    "scene description", "give a scene description", "what is in this image",
    "what's in this image", "summarize this image", "summarise this image",
]
FUSION_KEYWORDS = ["sar", "radar", "optical and sar", "fuse", "combine the optical", "both images together", "cross-modal"]


def classify_task(query: str, mode: InputMode) -> TaskType:
    q = query.lower()

    if mode == "bi_temporal_pair":
        # both change_description and change_vqa map to the same pipeline;
        # default to change_vqa since it returns a full natural-language answer
        return "change_vqa" if _any(q, CHANGE_KEYWORDS) or True else "change_description"

    if mode == "cross_modal_pair":
        return "optical_sar_fusion"

    # single_image
    if COUNTING_RE.search(q) or _any(q, GROUNDING_KEYWORDS):
        return "grounding"
    if _any(q, CAPTION_KEYWORDS):
        return "captioning"
    return "vqa"  # default / mandatory baseline


def _any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)
