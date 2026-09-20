from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ImageRecord, AnalysisRecord
from app.schemas import AnalysisRequest, AnalysisResult, AnalysisListItem, AnalysisStats, TaskCount, ThreadListItem
from app.services.orchestrator import run_analysis, OrchestrationError
from app.services.image_inspector import ImageMeta
from app.services.report_generator import generate_report_pdf

router = APIRouter()


def _build_thread(db: Session, analysis_id: str) -> list[AnalysisRecord]:
    """Returns the full chain of an analysis and its follow-ups, oldest
    first, regardless of which turn's id is passed in -- walks up to the
    root, then forward through parent_analysis_id links. Raises
    HTTPException(404) if analysis_id doesn't exist."""
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(404, "Analysis not found")

    root = record
    while root.parent_analysis_id:
        parent = db.query(AnalysisRecord).filter(AnalysisRecord.id == root.parent_analysis_id).first()
        if not parent:
            break
        root = parent

    thread = [root]
    current = root
    while True:
        child = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.parent_analysis_id == current.id)
            .order_by(AnalysisRecord.created_at.asc())
            .first()
        )
        if not child:
            break
        thread.append(child)
        current = child

    return thread


@router.post("/analysis", response_model=AnalysisResult)
def submit_analysis(request: AnalysisRequest, db: Session = Depends(get_db)):
    parent_record = None
    parent_context = None
    if request.parent_analysis_id:
        parent_record = db.query(AnalysisRecord).filter(AnalysisRecord.id == request.parent_analysis_id).first()
        if not parent_record:
            raise HTTPException(404, f"Parent analysis id not found: {request.parent_analysis_id}")
        parent_context = {
            "analysis_id": parent_record.id,
            "task_selected": parent_record.task_selected,
            "trace": parent_record.trace,
            "visual_evidence": parent_record.visual_evidence,
            "answer": parent_record.answer,
            "query": parent_record.query,
        }

    # A follow-up only supplies new query text -- mode/image_ids are inherited
    # from the parent turn rather than being re-specified by the caller. Uses
    # `is not None` (not truthiness) so an explicitly-empty image_ids=[] on a
    # non-follow-up request still flows through to the orchestrator's own
    # "requires N image(s)" validation instead of being silently swapped out.
    effective_mode = request.mode if request.mode is not None else (parent_record.mode if parent_record else None)
    effective_image_ids = (
        request.image_ids if request.image_ids is not None else (parent_record.image_ids if parent_record else None)
    )
    if effective_mode is None or effective_image_ids is None:
        raise HTTPException(400, "mode and image_ids are required unless parent_analysis_id is set")

    records = db.query(ImageRecord).filter(ImageRecord.id.in_(effective_image_ids)).all()
    found_ids = {r.id for r in records}
    missing = set(effective_image_ids) - found_ids
    if missing:
        raise HTTPException(404, f"Image id(s) not found: {sorted(missing)}")

    image_paths = {r.id: r.filepath for r in records}
    image_metas = {
        r.id: ImageMeta(format=r.format, width=r.width, height=r.height, bands=r.bands, crs=r.crs, modality=r.modality)
        for r in records
    }

    effective_request = request.model_copy(update={"mode": effective_mode, "image_ids": effective_image_ids})

    try:
        result = run_analysis(effective_request, image_paths, image_metas, parent_context=parent_context)
    except OrchestrationError as e:
        raise HTTPException(400, str(e))

    record = AnalysisRecord(
        id=result.analysis_id,
        query=result.query,
        mode=effective_mode,
        image_ids=effective_image_ids,
        task_selected=result.trace.task_selected,
        answer=result.answer,
        trace=result.trace.model_dump(),
        visual_evidence=result.visual_evidence,
        confidence=result.trace.confidence,
        parent_analysis_id=request.parent_analysis_id,
    )
    db.add(record)
    db.commit()

    return result


@router.get("/analysis/stats", response_model=AnalysisStats)
def get_analysis_stats(db: Session = Depends(get_db)):
    """Real aggregates over AnalysisRecord only -- no derived/invented
    metrics (no success rate, no percentages that aren't a literal
    count). Registered before /analysis/{analysis_id} so FastAPI doesn't
    match "stats" as a path parameter."""
    total = db.query(AnalysisRecord).count()

    by_task_rows = (
        db.query(AnalysisRecord.task_selected, func.count(AnalysisRecord.id))
        .group_by(AnalysisRecord.task_selected)
        .all()
    )

    recent = (
        db.query(AnalysisRecord)
        .order_by(AnalysisRecord.created_at.desc())
        .limit(5)
        .all()
    )

    return AnalysisStats(
        total=total,
        by_task=[TaskCount(task=task, count=count) for task, count in by_task_rows],
        recent=[
            AnalysisListItem(
                analysis_id=r.id, query=r.query, task_selected=r.task_selected,
                created_at=r.created_at.isoformat(),
            )
            for r in recent
        ],
    )


@router.get("/analysis/threads", response_model=list[ThreadListItem])
def get_analysis_threads(db: Session = Depends(get_db)):
    """Root-level analyses (parent_analysis_id is null) with a count of
    every descendant follow-up in their thread. Follow-ups chain to the
    immediately preceding turn, not always the root (see
    orchestrator.py's context_inherited step), so counting direct
    children of the root alone would undercount a thread with 2+
    follow-ups -- this walks the full descendant chain instead."""
    roots = (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.parent_analysis_id.is_(None))
        .order_by(AnalysisRecord.created_at.desc())
        .all()
    )

    child_rows = (
        db.query(AnalysisRecord.id, AnalysisRecord.parent_analysis_id)
        .filter(AnalysisRecord.parent_analysis_id.isnot(None))
        .all()
    )
    children_by_parent: dict[str, list[str]] = {}
    for child_id, parent_id in child_rows:
        children_by_parent.setdefault(parent_id, []).append(child_id)

    def count_descendants(root_id: str) -> int:
        count = 0
        stack = list(children_by_parent.get(root_id, []))
        while stack:
            current = stack.pop()
            count += 1
            stack.extend(children_by_parent.get(current, []))
        return count

    return [
        ThreadListItem(
            analysis_id=r.id,
            query=r.query,
            task_selected=r.task_selected,
            created_at=r.created_at.isoformat(),
            follow_up_count=count_descendants(r.id),
        )
        for r in roots
    ]


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(404, "Analysis not found")
    from app.schemas import ExecutionTrace
    return AnalysisResult(
        analysis_id=record.id,
        query=record.query,
        answer=record.answer,
        trace=ExecutionTrace(**record.trace),
        visual_evidence=record.visual_evidence,
        parent_analysis_id=record.parent_analysis_id,
    )


@router.get("/analysis/{analysis_id}/thread", response_model=list[AnalysisResult])
def get_analysis_thread(analysis_id: str, db: Session = Depends(get_db)):
    thread = _build_thread(db, analysis_id)
    from app.schemas import ExecutionTrace
    return [
        AnalysisResult(
            analysis_id=r.id,
            query=r.query,
            answer=r.answer,
            trace=ExecutionTrace(**r.trace),
            visual_evidence=r.visual_evidence,
            parent_analysis_id=r.parent_analysis_id,
        )
        for r in thread
    ]


@router.get("/analysis/{analysis_id}/report")
def get_analysis_report(analysis_id: str, db: Session = Depends(get_db)):
    thread = _build_thread(db, analysis_id)
    all_image_ids = {iid for record in thread for iid in (record.image_ids or [])}
    image_records = {
        r.id: r for r in db.query(ImageRecord).filter(ImageRecord.id.in_(all_image_ids)).all()
    }
    pdf_bytes = generate_report_pdf(thread, image_records)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="satquery_report_{analysis_id}.pdf"'},
    )


@router.get("/analysis", response_model=list[AnalysisListItem])
def list_analyses(db: Session = Depends(get_db)):
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).limit(50).all()
    return [
        AnalysisListItem(
            analysis_id=r.id, query=r.query, task_selected=r.task_selected,
            created_at=r.created_at.isoformat(),
        )
        for r in records
    ]
