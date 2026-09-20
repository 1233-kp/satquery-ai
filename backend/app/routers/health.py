from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "app_name": "SatQuery AI Backend",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
