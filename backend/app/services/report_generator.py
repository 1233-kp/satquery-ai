"""
Generates the downloadable PDF report named in the problem statement
("downloadable reports"). Pure presentation over data that already exists
on AnalysisRecord rows (query, answer, trace, visual_evidence) -- this
module does not call any model or re-derive anything, it only lays out
what the orchestrator already computed and persisted.
"""
import base64
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether,
)
from PIL import Image as PILImage

from app.models import AnalysisRecord, ImageRecord

_styles = getSampleStyleSheet()
_styles.add(ParagraphStyle(name="TurnHeading", parent=_styles["Heading2"], spaceBefore=14, spaceAfter=6))
_styles.add(ParagraphStyle(name="FieldLabel", parent=_styles["Normal"], textColor=colors.HexColor("#666666"), fontSize=8, spaceAfter=2))
_styles.add(ParagraphStyle(name="Answer", parent=_styles["BodyText"], spaceAfter=8, leading=14))
_styles.add(ParagraphStyle(name="StepDetail", parent=_styles["BodyText"], fontSize=9, leading=12, leftIndent=12))
_styles.add(ParagraphStyle(name="Warning", parent=_styles["BodyText"], textColor=colors.HexColor("#a15c00"), fontSize=9))
_styles.add(ParagraphStyle(name="ImageCaption", parent=_styles["Normal"], fontSize=8, textColor=colors.HexColor("#666666"), alignment=1, spaceBefore=3))

# Every embedded image (input photo or computed mask) is capped to this box and
# framed with a border -- fixes the earlier bug where a near-solid-black change
# mask had no visible boundary and could run into whatever followed it. Capping
# both dimensions (not just width, scaled by aspect ratio) is what actually
# fixes it: a source image far taller than it is wide previously produced a
# height that could exceed the remaining space on the page, which is what a
# lone width-only cap had missed.
_IMG_BOX = 2.3 * inch


_IMG_DPI = 150  # print-quality without embedding full-resolution source pixels


def _bordered_image(pil_img, box: float = _IMG_BOX) -> Table:
    """Scales a PIL image to fit within box x box (preserving aspect ratio,
    never upscaling past it) and frames it in a bordered single-cell table --
    a plain Image flowable has no visible edge, so a mostly-black mask (the
    near-zero-change case) looked like an untethered void with nothing to
    show where it actually ends.

    Actually downsamples the pixel data to the target print size (at
    _IMG_DPI) before encoding, rather than just telling reportlab to *display*
    it smaller -- a full-resolution multi-band GeoTIFF source (e.g. 1998x2255)
    embedded at its original pixel size but shrunk only on-screen bloated one
    report to 22MB. Never upscales past the source's own resolution."""
    w, h = pil_img.size
    scale = min(box / w, box / h, 1.0)
    draw_w, draw_h = w * scale, h * scale

    target_px = (max(1, round(draw_w / inch * _IMG_DPI)), max(1, round(draw_h / inch * _IMG_DPI)))
    if target_px[0] < w or target_px[1] < h:
        pil_img = pil_img.resize(target_px, PILImage.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    img_flowable = Image(buf, width=draw_w, height=draw_h)

    table = Table([[img_flowable]], colWidths=[box], rowHeights=[box])
    table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cccccc")),
    ]))
    return table


def _captioned_image(pil_img, caption: str, box: float = _IMG_BOX) -> list:
    return [_bordered_image(pil_img, box), Paragraph(caption, _styles["ImageCaption"])]


def _image_row(captioned_images: list[list]) -> Table:
    """Lays out 1-2 (image, caption) flowable pairs side by side so a
    before/after or optical/SAR comparison is visually obvious in one row,
    rather than stacked in a way that forces flipping between pages."""
    n = len(captioned_images)
    col_width = (6.5 * inch) / n
    row = Table([captioned_images], colWidths=[col_width] * n)
    row.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return row


def _input_imagery_flow(record: AnalysisRecord, image_records: dict[str, ImageRecord]) -> list:
    """The original source image(s) for this turn -- labeled by mode so a
    before/after or optical/SAR pair is unambiguous. Returns [] if any
    referenced image is missing (e.g. deleted from disk) rather than
    raising, since this is a presentation nicety, not critical data."""
    from app.services.geo_preprocessing import load_image_as_rgb

    ids = record.image_ids or []
    labels = {
        "single_image": ["Input image"],
        "bi_temporal_pair": ["Before (T1)", "After (T2)"],
        "cross_modal_pair": ["Optical", "SAR"],
    }.get(record.mode, [f"Image {i + 1}" for i in range(len(ids))])

    captioned = []
    for image_id, label in zip(ids, labels):
        img_record = image_records.get(image_id)
        if not img_record:
            continue
        try:
            pil_img = load_image_as_rgb(img_record.filepath)
        except Exception:
            continue
        captioned.append(_captioned_image(pil_img, label))

    if not captioned:
        return []
    # KeepTogether is safe here (unlike the old evidence block) because this unit
    # is always small and bounded -- at most a label plus one row of _IMG_BOX-capped
    # images, never something that could itself overflow a fresh page. Without it,
    # the label alone was landing at the bottom of one page with the actual images
    # starting on the next.
    return [KeepTogether([Paragraph("Input imagery", _styles["FieldLabel"]), _image_row(captioned)]), Spacer(1, 8)]


def _stats_table(stats: dict) -> Table:
    rows = [["Metric", "Value"]] + [
        [k.replace("_", " "), (f"{v:.2f}" if isinstance(v, float) else str(v))]
        for k, v in stats.items()
    ]
    table = Table(rows, colWidths=[2.6 * inch, 2.6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _turn_flowables(record: AnalysisRecord, index: int, total: int, image_records: dict[str, ImageRecord]) -> list:
    flow = []
    label = "Original analysis" if index == 0 else f"Follow-up {index}"
    flow.append(Paragraph(f"{label} &mdash; <font color='#1a7f74'>{record.task_selected}</font>", _styles["TurnHeading"]))

    flow.append(Paragraph("Query", _styles["FieldLabel"]))
    flow.append(Paragraph(record.query, _styles["BodyText"]))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("Answer", _styles["FieldLabel"]))
    flow.append(Paragraph(record.answer.replace("\n", "<br/>"), _styles["Answer"]))

    trace = record.trace or {}
    flow.append(Paragraph("Execution trace", _styles["FieldLabel"]))
    trace_lines = [
        f"<b>Task:</b> {trace.get('task_selected', '-')}",
        f"<b>Mode:</b> {trace.get('mode', '-')}",
        f"<b>Model(s)/tool:</b> {', '.join(trace.get('models_used', []))}",
    ]
    if trace.get("confidence") is not None:
        trace_lines.append(f"<b>Confidence:</b> {trace['confidence']:.2f}")
    else:
        trace_lines.append(f"<b>Confidence:</b> N/A ({trace.get('confidence_note', 'uncalibrated')})")
    flow.append(Paragraph("<br/>".join(trace_lines), _styles["BodyText"]))
    flow.append(Spacer(1, 4))

    steps = trace.get("steps", [])
    if steps:
        flow.append(Paragraph("Pipeline steps", _styles["FieldLabel"]))
        for i, step in enumerate(steps, start=1):
            flow.append(Paragraph(f"{i}. <b>{step['step']}</b> &mdash; {step['detail']}", _styles["StepDetail"]))
        flow.append(Spacer(1, 6))

    flow.extend(_input_imagery_flow(record, image_records))

    params = trace.get("parameters", {})
    numeric_stats = None
    for key in ("change_stats", "fusion_stats"):
        if key in params:
            numeric_stats = params[key]
            break

    evidence = record.visual_evidence or {}
    mask_b64 = evidence.get("change_mask_png_b64")
    boxes_b64 = evidence.get("detection_boxes_png_b64")
    stats = evidence.get("stats") or numeric_stats

    if mask_b64 or boxes_b64 or stats:
        evidence_label_done = False
        # A bordered, size-capped box per image (see _bordered_image) -- the
        # earlier version scaled only by width, so a near-solid-black change
        # mask with no visible edge could render taller than the space left
        # on the page and run into whatever followed it. The label is kept
        # with just its own image (KeepTogether on a single _IMG_BOX-capped
        # image is always safely bounded) -- but deliberately NOT extended to
        # cover the stats table too: forcing image + table to stay together
        # was the other half of the original bug, since a block too tall for
        # a fresh page has nowhere left to go but overlap.
        for b64, caption in ((mask_b64, "Change mask"), (boxes_b64, "Detected objects")):
            if not b64:
                continue
            try:
                img_bytes = base64.b64decode(b64)
                pil_img = PILImage.open(io.BytesIO(img_bytes))
                unit = []
                if not evidence_label_done:
                    unit.append(Paragraph("Computed evidence", _styles["FieldLabel"]))
                    evidence_label_done = True
                unit.append(_bordered_image(pil_img))
                unit.append(Paragraph(caption, _styles["ImageCaption"]))
                flow.append(KeepTogether(unit))
                flow.append(Spacer(1, 8))
            except Exception:
                flow.append(Paragraph(f"({caption.lower()} image could not be rendered)", _styles["BodyText"]))
        if not evidence_label_done:
            flow.append(Paragraph("Computed evidence", _styles["FieldLabel"]))
        if stats:
            flow.append(_stats_table(stats))
        flow.append(Spacer(1, 6))

    if params.get("reused_from_analysis_id"):
        flow.append(Paragraph(
            f"<i>Evidence reused from analysis {params['reused_from_analysis_id']} "
            "&mdash; no new model inference was run for this turn.</i>",
            _styles["BodyText"],
        ))
        flow.append(Spacer(1, 6))

    if index < total - 1:
        flow.append(Spacer(1, 4))
        flow.append(HRFlowable(width="100%", color=colors.HexColor("#dddddd")))

    return flow


def generate_report_pdf(thread: list[AnalysisRecord], image_records: dict[str, ImageRecord] | None = None) -> bytes:
    """thread: the full chain (root first, follow-ups in order) for one
    analysis. A single-turn list renders the same as a standalone report.
    image_records: {image_id: ImageRecord} for every image referenced by any
    turn in the thread, used to embed the original source image(s) alongside
    computed evidence. Optional (defaults to {}) so this stays a pure
    presentation function the caller can still exercise without a DB-backed
    lookup, but no source images will be embedded without it."""
    image_records = image_records or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="SatQuery AI Analysis Report",
    )

    root = thread[0]
    flow = [
        Paragraph("SatQuery AI &mdash; Analysis Report", _styles["Title"]),
        Paragraph(
            f"Analysis {root.id} &middot; generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            + (f" &middot; {len(thread)} turns" if len(thread) > 1 else ""),
            _styles["FieldLabel"],
        ),
        Spacer(1, 10),
    ]

    for i, record in enumerate(thread):
        flow.extend(_turn_flowables(record, i, len(thread), image_records))

    doc.build(flow)
    return buf.getvalue()
