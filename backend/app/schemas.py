from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


InputMode = Literal["single_image", "cross_modal_pair", "bi_temporal_pair"]
TaskType = Literal[
    "vqa",
    "captioning",
    "grounding",
    "change_description",
    "change_vqa",
    "optical_sar_fusion",
]


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    modality: str
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    crs: Optional[str] = None


class AnalysisRequest(BaseModel):
    query: str = Field(..., description="Natural language query from the user")
    mode: Optional[InputMode] = Field(
        default=None, description="Required unless parent_analysis_id is set, in which case it is inherited"
    )
    image_ids: Optional[list[str]] = Field(
        default=None, description="Required unless parent_analysis_id is set, in which case it is inherited"
    )
    task_hint: Optional[TaskType] = Field(
        default=None, description="Optional override; orchestrator classifies if omitted"
    )
    parent_analysis_id: Optional[str] = Field(
        default=None,
        description="If set, this is a follow-up turn: mode/image_ids are inherited from the parent analysis",
    )

    @model_validator(mode="after")
    def _require_mode_and_images_for_a_new_analysis(self):
        if self.parent_analysis_id is None and (self.mode is None or self.image_ids is None):
            raise ValueError("mode and image_ids are required unless parent_analysis_id is set")
        return self


class ExecutionStep(BaseModel):
    step: str
    detail: str


class ExecutionTrace(BaseModel):
    task_selected: TaskType
    mode: InputMode
    models_used: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    steps: list[ExecutionStep] = Field(default_factory=list)
    confidence: Optional[float] = None
    confidence_note: Optional[str] = None


class AnalysisResult(BaseModel):
    analysis_id: str
    query: str
    answer: str
    trace: ExecutionTrace
    visual_evidence: Optional[dict[str, Any]] = None
    warnings: list[str] = Field(default_factory=list)
    parent_analysis_id: Optional[str] = None


class AnalysisListItem(BaseModel):
    analysis_id: str
    query: str
    task_selected: TaskType
    created_at: str


class TaskCount(BaseModel):
    task: str
    count: int


class AnalysisStats(BaseModel):
    total: int
    by_task: list[TaskCount]
    recent: list[AnalysisListItem]


class ThreadListItem(BaseModel):
    analysis_id: str
    query: str
    task_selected: TaskType
    created_at: str
    follow_up_count: int


# ── Auth ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    display_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _basic_email_shape(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Enter a valid email address")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class PublicUser(BaseModel):
    id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    oauth_provider: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: PublicUser
