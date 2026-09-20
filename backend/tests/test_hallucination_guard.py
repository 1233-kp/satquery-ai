"""
Regression tests for hallucination_guard.py's date/time detection.

Added alongside _DATE_RE/_TIME_RE, which fill a gap surfaced by a real
generated caption: "A satellite image of the city of Paris, France,
captured on 26 January 2017 at 17:27." passed through check_and_clean()
completely unflagged -- the guard only ever checked for fabricated
locations and area figures, never dates/times, even though the model has
exactly as little basis for inventing a capture date as it does an area
in square meters (see the module docstring's justification for _AREA_RE,
which applies here unchanged).
"""
from app.services.hallucination_guard import check_and_clean


def test_real_example_strips_fabricated_date_and_time():
    # The exact caption that surfaced the gap. Only the location/date/time
    # sentence should be dropped -- the second sentence has no fabricated
    # claim in it and should survive.
    answer = (
        "A satellite image of the city of Paris, France, captured on "
        "26 January 2017 at 17:27. The dominant feature is the urban "
        "landscape, which is characterized by a circular, green area."
    )
    cleaned, fired, detail = check_and_clean(answer, "Caption this image")
    assert fired is True
    assert "26 January 2017" not in cleaned
    assert "17:27" not in cleaned
    assert "urban landscape" in cleaned
    assert "date/time" in detail


def test_textual_date_formats_are_flagged():
    for answer in [
        "This image was captured on 26 January 2017.",
        "Captured January 26, 2017 near the coast.",
    ]:
        _, fired, _ = check_and_clean(answer, "Caption this image")
        assert fired is True, answer


def test_numeric_date_formats_are_flagged():
    for answer in [
        "This image was taken on 2017-01-26.",
        "The scene dates to 03/14/2019.",
    ]:
        _, fired, _ = check_and_clean(answer, "Caption this image")
        assert fired is True, answer


def test_anchored_time_claim_is_flagged():
    answer = "The image was captured at 17:27 over a coastal town."
    _, fired, _ = check_and_clean(answer, "Caption this image")
    assert fired is True


def test_pixel_dimensions_are_not_flagged():
    answer = "The image is 512x512 pixels showing a mixed-use urban area."
    cleaned, fired, _ = check_and_clean(answer, "Caption this image")
    assert fired is False
    assert cleaned == answer


def test_scale_and_aspect_ratios_are_not_flagged():
    for answer in [
        "This is a 1:50000 scale map of the region.",
        "The aspect ratio is 16:9 for this crop.",
    ]:
        cleaned, fired, _ = check_and_clean(answer, "Caption this image")
        assert fired is False, answer
        assert cleaned == answer


def test_object_counts_are_not_flagged():
    answer = "Approximately 3 buildings were detected near the water."
    cleaned, fired, _ = check_and_clean(answer, "How many buildings are visible?")
    assert fired is False
    assert cleaned == answer


def test_bare_year_is_not_flagged():
    # A lone year, with no day/month or full numeric date triple, isn't a
    # specific enough claim to match -- avoids false-flagging things like
    # "Class 2017 residential zone" or stylistic year references.
    for answer in [
        "This is the year 2017 style building visible in the scene.",
        "Class 2017 residential zone with scattered vegetation.",
    ]:
        cleaned, fired, _ = check_and_clean(answer, "Caption this image")
        assert fired is False, answer
        assert cleaned == answer
