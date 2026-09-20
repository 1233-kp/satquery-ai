import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Integer, JSON, Float
from app.database import Base


def gen_id() -> str:
    return uuid.uuid4().hex[:12]


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True, default=gen_id)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    modality = Column(String, nullable=False)   # optical | sar | unknown
    format = Column(String, nullable=False)     # tif | tiff | png | jpg
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    bands = Column(Integer, nullable=True)
    crs = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_id)
    email = Column(String, unique=True, nullable=True, index=True)
    hashed_password = Column(String, nullable=True)  # null for OAuth-only accounts
    oauth_provider = Column(String, nullable=True)   # 'google' | 'github' | null for password accounts
    oauth_id = Column(String, nullable=True)          # provider's subject/user id
    display_name = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=gen_id)
    query = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    image_ids = Column(JSON, nullable=False)
    task_selected = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    trace = Column(JSON, nullable=False)
    visual_evidence = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    parent_analysis_id = Column(String, nullable=True)  # set for follow-up turns in a thread
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
