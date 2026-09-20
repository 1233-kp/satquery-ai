"""One-off test: can the VLM be trusted to *rephrase* a fixed, fact-complete
sentence without adding/dropping facts or naming an unverifiable object type?
If not, per user instruction, we drop the VLM from this path entirely."""
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.vqa_service import get_vqa_service

template_sentence = (
    "A high level of change was detected between the two dates, affecting "
    "approximately 19.1% of the scene (12518/65536 pixels). The change is "
    "concentrated in the top-right (73%)."
)
prompt = (
    "Rephrase only the wording of the following sentence so it reads naturally. "
    "Do not add, remove, or change any number, percentage, or fact. Do not name any "
    "specific object, structure, or land-cover type (e.g. do not say building, road, "
    "parking lot, vegetation, etc.) -- only rewrite the sentence's phrasing.\n\n"
    f'Sentence: "{template_sentence}"'
)

vqa = get_vqa_service()
image_a = str(_BACKEND_ROOT / "data" / "uploads" / "09d2c8ecfdf2.png")
image_b = str(_BACKEND_ROOT / "data" / "uploads" / "0480ea3d8298.png")
answer, _, guard_fired, guard_detail = vqa.answer([image_a, image_b], prompt, task="vqa", min_new_tokens=10)
print("ORIGINAL: ", template_sentence)
print("REPHRASED:", answer)
print("guard_fired:", guard_fired, guard_detail)
