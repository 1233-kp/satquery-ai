"""
Regression test for the modality-guessing bug: `_guess_modality` used a bare
substring check ("s1" in name), so "ROIs1970_fall_s2_13_p265.png" (an
OPTICAL file) false-matched the "s1" hiding inside "ROIs1970" itself and got
misclassified as SAR. Fixed with delimiter-bounded token matching -- these
are the exact two real filenames that surfaced the bug.
"""
from app.services.image_inspector import _guess_modality


def test_real_roi_filenames_classify_correctly():
    # The actual SAR file: "s1" is its own delimited token.
    assert _guess_modality(bands=2, filename="ROIs1970_fall_s1_13_p265.png") == "sar"
    # The actual OPTICAL file: previously misclassified as "sar" because the
    # bare substring check matched "s1" inside "ROIs1970" before ever reaching
    # the "s2" check.
    assert _guess_modality(bands=3, filename="ROIs1970_fall_s2_13_p265.png") == "optical"


def test_delimited_tokens_still_match():
    assert _guess_modality(bands=2, filename="s1_scene.png") == "sar"
    assert _guess_modality(bands=3, filename="sar_scene.png") == "sar"
    assert _guess_modality(bands=2, filename="scene_s1.tif") == "sar"
    assert _guess_modality(bands=3, filename="s2_scene.png") == "optical"
    assert _guess_modality(bands=4, filename="optical_scene.tif") == "optical"


def test_embedded_substrings_do_not_false_match():
    # "s1"/"sar" appearing as a substring inside an unrelated word must not
    # trigger a modality guess -- falls through to the band-count heuristic.
    assert _guess_modality(bands=3, filename="caesar.png") == "optical"  # "sar" in "cae-sar"
    assert _guess_modality(bands=3, filename="solar_panel.png") == "optical"  # no real token
    assert _guess_modality(bands=1, filename="ROIs1970_something.png") == "sar"  # falls to band-count heuristic (<=2 bands), not the "s1" substring
