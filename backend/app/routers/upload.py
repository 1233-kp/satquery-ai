import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import ImageRecord
from app.schemas import UploadResponse
from app.services.image_inspector import inspect_image

router = APIRouter()
settings = get_settings()

ALLOWED_EXTENSIONS = {"tif", "tiff", "png", "jpg", "jpeg"}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file format '.{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    os.makedirs(settings.upload_dir, exist_ok=True)

    record = ImageRecord(filename=file.filename, filepath="", modality="unknown", format=ext)
    db.add(record)
    db.flush()  # get generated id before we know the final path

    dest_path = os.path.join(settings.upload_dir, f"{record.id}.{ext}")
    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    meta = inspect_image(dest_path, file.filename)

    record.filepath = dest_path
    record.modality = meta.modality
    record.width = meta.width
    record.height = meta.height
    record.bands = meta.bands
    record.crs = meta.crs
    db.commit()
    db.refresh(record)

    return UploadResponse(
        image_id=record.id,
        filename=record.filename,
        modality=record.modality,
        format=record.format,
        width=record.width,
        height=record.height,
        bands=record.bands,
        crs=record.crs,
    )
