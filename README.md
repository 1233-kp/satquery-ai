# SatQuery AI

Agentic vision-language assistant for remote-sensing image analysis — built for
SIH 2026 (PS 26167). Accepts single images, optical+SAR pairs, and bi-temporal
pairs, classifies the query, routes to the right specialist pipeline, and
returns an evidence-grounded answer with a full, auditable execution trace.

```
SatQuery-AI/
├── backend/     FastAPI service — orchestrator, specialist services, API
├── frontend/    React + Vite + TS console UI
└── render.yaml  One-click Render deploy blueprint for both services
```

Everything currently runs in **mock mode**: the orchestrator, task
classification, compatibility checks, upload handling, and trace assembly are
all real; the "model" outputs are deterministic placeholders computed from the
actual uploaded images (real pixel diffing for change detection, real
brightness/backscatter stats for fusion) so the whole stack is demoable today.
Swap in trained models by flipping `VQA_MODE` / `CHANGE_MODE` / `FUSION_MODE`
to `real` in `backend/.env` once they're ready — see "Wiring in real models"
below.

## Run locally (VS Code / terminal)

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
# .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 2. Frontend

In a new terminal:

```bash
cd frontend
npm install
cp .env.example .env      # VITE_API_URL=http://localhost:8000

npm run dev
```

- App: http://localhost:5173

Open the app, pick an input mode, upload the required image(s) (PNG/JPEG/TIFF),
type a query (or click an example chip), and run the analysis. The right-hand
panel shows the full orchestrator execution trace — task classification, tool
selection, pipeline steps, and confidence.

### 3. Run backend tests (once you add them under `backend/tests/`)

```bash
cd backend
pip install pytest
pytest -v
```

## Wiring in real models (Day 2-4 of the build)

The heavy ML dependencies are kept out of `requirements.txt` on purpose so
the base app installs fast (and deploys cleanly on Render's free tier).
When you're ready to plug in trained models:

```bash
cd backend
pip install -r requirements-ml.txt
```

Then in `backend/.env`:

```
VQA_MODE=real
VQA_LORA_PATH=./checkpoints/vqa_lora/best     # your fine-tuned LoRA adapter

CHANGE_MODE=real
CHANGE_DETECTION_CHECKPOINT=./checkpoints/change_unet/best_model.pt

FUSION_MODE=real
FUSION_CHECKPOINT=./checkpoints/fusion_net/best_model.pt
```

Extension points to fill in as you train:
- `backend/app/services/vqa_service.py` — `_load_real_model` / `_real_answer`
  already call SmolVLM-500M-Instruct + an optional PEFT LoRA adapter; point
  `VQA_LORA_PATH` at your fine-tuned checkpoint once it exists.
- `backend/app/services/change_service.py` — expects a `SiameseUNet` class at
  `backend/app/services/models/siamese_unet.py` (create this when you write
  the training script).
- `backend/app/services/fusion_service.py` — expects an `OpticalSARFusionNet`
  class at `backend/app/services/models/fusion_net.py`.

None of the orchestrator, routing, or trace-assembly logic needs to change —
only these three service internals.

**`backend/checkpoints/` is gitignored** — the trained weights themselves
(~757MB across the LoRA adapter, Siamese U-Net, and fusion net, plus their
per-epoch snapshots) aren't in this repo; only empty `vqa_lora/`,
`change_unet/`, and `fusion_net/` placeholder directories (via `.gitkeep`)
are, so the expected layout is visible on a fresh clone. To get real weights
into those folders:
- **Train from scratch** using `backend/scripts/train_vqa_lora.py`,
  `train_change_unet.py`, and `train_fusion_net.py` against the datasets
  `scripts/prepare_*.py` pull from their public sources, or
- **Copy them from wherever they're already deployed** — currently the
  Oracle Cloud VM running the real-model backend has its own full copy,
  transferred there directly (not via this repo); `scp` them down from
  there if you just need the existing trained weights rather than
  retraining.

## Deploying to Render

This repo is on GitHub at
[github.com/1233-kp/satquery-ai](https://github.com/1233-kp/satquery-ai)
but **has never actually been deployed to Render** — `render.yaml`
describes the intended setup, but creating the Render services,
registering OAuth redirect URIs, and setting secrets all require someone
with dashboard access to Render, Google Cloud Console, and the GitHub
OAuth App to do it by hand. What follows is that exact checklist.

**Step 1 — point Render at the GitHub repo.** Render's Blueprint flow
(below) reads `render.yaml` directly from
[github.com/1233-kp/satquery-ai](https://github.com/1233-kp/satquery-ai) —
no separate push needed, it's already there.

**Step 2 — create the services.** In the Render dashboard, "New > Blueprint"
pointed at the repo reads `render.yaml` and creates both services (backend
web service + frontend static site) automatically, with the model modes,
CPU device settings, and plan tier already set correctly for real models
(see below for why). Manual service creation works too if you'd rather not
use Blueprint — just match `render.yaml`'s `buildCommand`/`startCommand`.

**Step 3 — set the 5 secrets `render.yaml` leaves blank.** These are marked
`sync: false` so they're never committed in plaintext; Render prompts for
them on first deploy (Dashboard → satquery-ai-backend → Environment):

| Key | Where it comes from |
|---|---|
| `JWT_SECRET_KEY` | Generate a fresh random secret (e.g. `openssl rand -hex 32`) — do not reuse the local dev value from `backend/.env`. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google Cloud Console → APIs & Services → Credentials → your OAuth client. |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub → Settings → Developer settings → OAuth Apps → your app. |

**Step 4 — register the production redirect URIs.** This was flagged as
pending when auth was first built and still is — these only exist once the
backend is actually deployed (Step 2) and its real URL is known:
- Google Cloud Console, same OAuth client as above → Authorized redirect
  URIs → add `https://<your-backend>.onrender.com/auth/google/callback`
- GitHub OAuth App settings → Authorization callback URL → set (GitHub
  allows only one) `https://<your-backend>.onrender.com/auth/github/callback`

Do this *in addition to*, not instead of, the localhost URIs already there
if you still want OAuth to work locally too.

**Step 5 — verify the three URL env vars.** `render.yaml` sets
`CORS_ORIGINS`, `FRONTEND_URL`, and `VITE_API_URL` to the standard
`https://satquery-ai-<backend|frontend>.onrender.com` pattern, but Render
appends a random suffix instead if those exact names were already taken.
Check the actual assigned URLs in the dashboard after first deploy and
update these three if they don't match — a mismatch here breaks CORS and
OAuth redirects in a way that works fine locally and only surfaces once
deployed.

### Real models on Render's CPU-only infrastructure

`render.yaml` is set up for `VQA_MODE`/`CHANGE_MODE`/`FUSION_MODE`/
`GROUNDING_MODE=real`, not the mock mode this section used to recommend.
Concretely, measured locally with CUDA hidden (`CUDA_VISIBLE_DEVICES=""`) to
simulate Render's CPU-only environment:

- **RAM**: all four real models loaded eagerly at startup (see below) use
  ~2.6GB resident; after exercising one query per task type it settles at
  ~3.5GB and **stays flat** under repeated requests (checked directly — not
  a slow leak). Render's Standard plan (2GB) is not enough and will likely
  OOM; **Pro (4GB)** is the technical minimum with little headroom; **Pro
  Plus (8GB)**, which `render.yaml` uses, gives real headroom for concurrent
  users rather than just single-request steady-state.
- **Latency** (same CPU-only local simulation, one query per task, cold
  models already warm from startup): VQA 3.9s, grounding 1.4s, change
  detection 0.15s, optical+SAR fusion 1.0s, **captioning 43s**. Captioning's
  latency is real and load-bearing for demo planning — it's SmolVLM
  generating up to 256 tokens autoregressively with no batching or KV-cache
  tricks beyond what `transformers` does by default, and CPU token-by-token
  generation is simply slow. Don't rely on a live captioning query fitting
  inside a short demo window; consider having a pre-run example ready as a
  fallback.
- **Dtype**: already correct as of this codebase — `vqa_service.py` uses
  fp32 on CPU (not fp16, which has poor/emulated CPU support and would be
  slower, not faster, unlike on the RTX 3050 this was developed on); the
  change/fusion/grounding checkpoints were all verified fp32-native already,
  so no conversion needed there either.
- **Startup**: all four services now load once during app startup (a
  FastAPI `lifespan` handler in `app/main.py`), not lazily on each
  service's first request — check the boot logs for all four load
  messages before considering the deploy healthy; a failure here surfaces
  immediately instead of on some later user's first query.
- **Build**: `requirements.txt` alone does *not* include torch/transformers/
  rasterio (see its own header comment) — `render.yaml`'s `buildCommand`
  installs from `requirements-ml.txt`, and installs `torch`/`torchvision`
  from the CPU-only wheel index first to avoid pulling PyPI's default
  CUDA-bundled build (several GB larger than needed on a GPU-less target).
  Not yet confirmed against Render's actual build-time/disk limits for the
  Pro Plus tier specifically — that requires a real deploy to observe.
- **Proxy headers**: Render terminates TLS at its edge and forwards plain
  HTTP internally; without `--forwarded-allow-ips='*' --proxy-headers` on
  the uvicorn start command (already in `render.yaml`), `request.url_for()`
  builds `http://` OAuth redirect URIs instead of `https://`, which then
  mismatch whatever's registered in Google/GitHub's console and break OAuth
  in production despite working locally. Verified this fix directly against
  a simulated proxy header, not just reasoned about.

Render's disk for a standard web service is ephemeral, so the SQLite history
resets on redeploy/restart — fine for a demo, not for production data.

**Not yet done — needs an actual deploy, which needs Render dashboard
access this session didn't have:** confirming the build completes within
Pro Plus's real build-time/disk limits, and measuring true end-to-end
latency and OAuth login on the live URL itself rather than this local
CPU-only simulation. Once deployed, re-run the same per-task-type latency
check directly against the live URL — real network latency and Render's
actual CPU class may differ from this local approximation in either
direction.

## What's implemented vs. what's next

| Requirement (PS 26167) | Status |
|---|---|
| Input upload + compatibility checking | ✅ done |
| Single-image VQA | ✅ done (mock; real hook ready) |
| Second single-image task (captioning) | ✅ done (mock; real hook ready) |
| Bi-temporal change description / change-VQA | ✅ done (mock; real hook ready) |
| Optical+SAR cross-modal analysis | ✅ done (mock; real hook ready) |
| Agentic orchestration + auditable trace | ✅ done |
| Remote-sensing adaptation (LoRA fine-tune on BigEarthNet.txt) | ⏳ Day 2 |
| Trained Siamese U-Net on LEVIR-CD | ⏳ Day 3 |
| Trained optical+SAR fusion net | ⏳ Day 4 |
| PDF mission report download | ⏳ Day 5 |
