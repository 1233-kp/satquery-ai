"""
Orchestrator hardening tests: confirm each of these known failure-prone
inputs produces a clear error or an explicit warning -- never a silent
wrong answer with no trace of the problem:
  1. Wrong image count for the requested mode.
  2. A bi-temporal/cross-modal pair with mismatched dimensions.
  3. An SAR-only pair submitted as cross_modal_pair (expects one optical +
     one SAR image).
  4. An ambiguous/empty query with no task_hint -- the classifier must
     still make and *record* an explicit decision, not silently guess.

Runs against VQA_MODE=CHANGE_MODE=FUSION_MODE=mock (see conftest.py) --
these are input-validation/routing tests, not model-quality tests, so
they don't need the GPU or trained checkpoints.
"""
import io

from PIL import Image


def _png_bytes(width: int, height: int, color=(120, 140, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, filename: str, width: int = 64, height: int = 64) -> str:
    resp = client.post(
        "/upload",
        files={"file": (filename, _png_bytes(width, height), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["image_id"]


# ---------------------------------------------------------------------------
# 1. Wrong image count per mode
# ---------------------------------------------------------------------------

def test_single_image_mode_rejects_two_images(client):
    img_a = _upload(client, "a.png")
    img_b = _upload(client, "b.png")
    resp = client.post("/analysis", json={
        "query": "Describe this image.",
        "mode": "single_image",
        "image_ids": [img_a, img_b],
    })
    assert resp.status_code == 400
    assert "requires 1 image" in resp.json()["detail"]


def test_bi_temporal_mode_rejects_one_image(client):
    img_a = _upload(client, "a.png")
    resp = client.post("/analysis", json={
        "query": "What changed?",
        "mode": "bi_temporal_pair",
        "image_ids": [img_a],
    })
    assert resp.status_code == 400
    assert "requires 2 image" in resp.json()["detail"]


def test_cross_modal_mode_rejects_zero_images(client):
    resp = client.post("/analysis", json={
        "query": "Fuse these.",
        "mode": "cross_modal_pair",
        "image_ids": [],
    })
    assert resp.status_code == 400
    assert "requires 2 image" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 2. Mismatched pair dimensions -> warning, not a silent pass
# ---------------------------------------------------------------------------

def test_mismatched_dimensions_produces_warning_not_silent_success(client):
    img_a = _upload(client, "before.png", width=64, height=64)
    img_b = _upload(client, "after.png", width=128, height=96)
    resp = client.post("/analysis", json={
        "query": "What changed between these two dates?",
        "mode": "bi_temporal_pair",
        "image_ids": [img_a, img_b],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert any("Dimension mismatch" in w for w in body["warnings"]), body["warnings"]
    compat_steps = [s for s in body["trace"]["steps"] if s["step"] == "compatibility_check"]
    assert compat_steps, "compatibility_check step missing from trace"
    assert "Dimension mismatch" in compat_steps[0]["detail"]


def test_matching_dimensions_produce_no_dimension_warning(client):
    img_a = _upload(client, "before.png", width=64, height=64)
    img_b = _upload(client, "after.png", width=64, height=64)
    resp = client.post("/analysis", json={
        "query": "What changed between these two dates?",
        "mode": "bi_temporal_pair",
        "image_ids": [img_a, img_b],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert not any("Dimension mismatch" in w for w in body["warnings"]), body["warnings"]


# ---------------------------------------------------------------------------
# 3. SAR-only pair passed as cross_modal_pair -> explicit degraded warning
# ---------------------------------------------------------------------------

def test_sar_only_pair_as_cross_modal_warns_instead_of_silently_fusing(client):
    img_a = _upload(client, "s1_scene_a.png")  # "s1" in filename -> modality guessed as sar
    img_b = _upload(client, "sar_scene_b.png")  # "sar" in filename -> modality guessed as sar
    resp = client.post("/analysis", json={
        "query": "Analyze the land cover using both optical and SAR imagery.",
        "mode": "cross_modal_pair",
        "image_ids": [img_a, img_b],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert any("Expected one optical + one SAR image" in w for w in body["warnings"]), body["warnings"]
    # It's explicitly allowed to proceed in degraded mode -- the point is the
    # warning must be present, not that the request should hard-fail.
    assert body["answer"], "expected a response even in degraded mode, just with a warning"


def test_proper_optical_sar_pair_has_no_modality_warning(client):
    img_optical = _upload(client, "s2_scene.png")
    img_sar = _upload(client, "s1_scene.png")
    resp = client.post("/analysis", json={
        "query": "Analyze the land cover using both optical and SAR imagery.",
        "mode": "cross_modal_pair",
        "image_ids": [img_optical, img_sar],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert not any("Expected one optical + one SAR image" in w for w in body["warnings"]), body["warnings"]


# ---------------------------------------------------------------------------
# 4. Ambiguous / empty query -> classifier still records an explicit decision
# ---------------------------------------------------------------------------

def test_empty_query_gets_an_explicit_recorded_classification(client):
    img_a = _upload(client, "scene.png")
    resp = client.post("/analysis", json={
        "query": "",
        "mode": "single_image",
        "image_ids": [img_a],
    })
    assert resp.status_code == 200
    body = resp.json()
    task_steps = [s for s in body["trace"]["steps"] if s["step"] == "task_classification"]
    assert task_steps, "task_classification step missing from trace"
    # The exact fallback task doesn't matter as much as the fact that it's an
    # explicit, named, auditable decision -- not a silent skip or crash.
    assert body["trace"]["task_selected"] in ("vqa", "captioning", "grounding")
    assert "classified as" in task_steps[0]["detail"]


def test_nonsensical_query_does_not_crash_and_still_answers(client):
    img_a = _upload(client, "scene.png")
    resp = client.post("/analysis", json={
        "query": "asdkjfh qwoeiru zzz???",
        "mode": "single_image",
        "image_ids": [img_a],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["trace"]["task_selected"]
