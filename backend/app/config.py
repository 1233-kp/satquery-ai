from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    database_url: str = "sqlite:///./satquery.db"
    upload_dir: str = "./data/uploads"
    results_dir: str = "./data/results"
    cors_origins: str = "http://localhost:3000"

    vqa_mode: str = "mock"          # real | mock
    change_mode: str = "mock"       # real | mock
    fusion_mode: str = "mock"       # real | mock
    grounding_mode: str = "mock"    # real | mock

    vqa_base_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    vqa_lora_path: str = "./checkpoints/vqa_lora/best"
    vqa_device: str = "cpu"
    vqa_max_new_tokens: int = 256
    # Captioning naturally generates far more tokens than VQA (measured: ~235
    # vs ~2 on the same model/hardware) -- this is a separate cap so it can be
    # tuned for captioning's latency/completeness tradeoff without touching
    # VQA, which is nowhere near either value.
    captioning_max_new_tokens: int = 120
    # None = leave torch's own default thread count alone (fine on a normal
    # host). Containerized CPU targets (e.g. Render) commonly report the
    # HOST's full core count via /proc/cpuinfo rather than the container's
    # actual cgroup-limited allocation, which can make torch spawn more
    # threads than the container really has, causing contention that's a net
    # slowdown, not a speedup. Set this explicitly to the deploy target's
    # real vCPU count to avoid that -- see render.yaml.
    torch_num_threads: int | None = None

    change_detection_checkpoint: str = "./checkpoints/change_unet/best_model.pt"
    fusion_checkpoint: str = "./checkpoints/fusion_net/best_model.pt"

    grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    grounding_device: str = "cpu"
    grounding_box_threshold: float = 0.30
    grounding_text_threshold: float = 0.25

    # ── Auth ──────────────────────────────────────────────
    jwt_secret_key: str = "insecure-dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 1 week

    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""

    frontend_url: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
