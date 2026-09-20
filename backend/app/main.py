import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.config import get_settings
from app.database import Base, engine
from app.routers import health, upload, analysis, tools, auth
from app.models import User  # noqa: F401 -- ensures User is registered on Base before create_all

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Every get_*_service() is @lru_cache'd, so without this the model weights
    # for whichever specialist gets hit first load lazily on that request --
    # fine locally, but on a real deploy that means the first live user query
    # of each task type eats a multi-second-to-a-minute-plus cold-load penalty
    # instead of just inference time. Load everything real once, here, so
    # by the time the process is accepting traffic every specialist is warm.
    # Loading failures surface immediately in boot logs instead of on some
    # later user's request.
    if settings.torch_num_threads is not None:
        import torch
        # Must happen before any model loads/inference -- see config.py's
        # torch_num_threads docstring for why this matters on Render.
        torch.set_num_threads(settings.torch_num_threads)
    if settings.vqa_mode == "real":
        from app.services.vqa_service import get_vqa_service
        get_vqa_service()
    if settings.change_mode == "real":
        from app.services.change_service import get_change_service
        get_change_service()
    if settings.fusion_mode == "real":
        from app.services.fusion_service import get_fusion_service
        get_fusion_service()
    if settings.grounding_mode == "real":
        from app.services.grounding_service import get_grounding_service
        get_grounding_service()
    yield

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.results_dir, exist_ok=True)

Base.metadata.create_all(bind=engine)

# `create_all` only creates missing tables, not missing columns on tables that
# already exist -- needed here because `parent_analysis_id` was added to
# AnalysisRecord after satquery.db files from earlier runs may already exist.
if engine.url.get_backend_name() == "sqlite":
    with engine.connect() as conn:
        from sqlalchemy import text

        existing_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(analyses)"))}
        if existing_cols and "parent_analysis_id" not in existing_cols:
            conn.execute(text("ALTER TABLE analyses ADD COLUMN parent_analysis_id VARCHAR"))
            conn.commit()

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic vision-language assistant for remote-sensing image analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only used for the OAuth handshake's transient state/nonce (authlib's
# authorize_redirect/authorize_access_token read/write request.session).
# Both /auth/*/login and /auth/*/callback are backend routes hit via full
# top-level navigation, so this cookie is always same-site -- the app's
# actual auth session is a bearer JWT, not this cookie (see
# auth_service.py's docstring), so this never needs SameSite=None/Secure.
app.add_middleware(SessionMiddleware, secret_key=settings.jwt_secret_key)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(analysis.router)
app.include_router(tools.router)
app.include_router(auth.router)


@app.get("/")
def root():
    return {
        "app": "SatQuery AI",
        "status": "running",
        "docs": "/docs",
        "modes": {
            "vqa": settings.vqa_mode,
            "change": settings.change_mode,
            "fusion": settings.fusion_mode,
            "grounding": settings.grounding_mode,
        },
    }
