# Session 10 — Full-Stack Deployment

## Learning objectives

- The whole deploy is one file pair — `backend/Dockerfile` and `docker-compose.yml` — because the previous nine sessions already did the hard work: every stateful thing (the checkpointer, the long-term store, the Chroma index) already lives in a file, never in a Python object. Containerizing a process with no in-memory state left to lose is what makes this session small.
- 12-factor config as leverage, not ceremony: `DESK_DB_PATH` and `CHROMA_DIR` are the only two paths the app hard-codes anywhere, and both already read from the environment — the exact same `app.main:app` runs bare-metal (Session 01's way) and inside the container, just with different env values.
- A healthcheck answers "is this ready to serve," not "is the process running" — `/health` is written to never throw (a `try`/`except` around the Chroma directory listing), because a probe that can itself crash turns a soft failure into compose reporting the container unhealthy for the wrong reason.
- Layer-ordering is an economics lesson, not a style rule: `COPY requirements.txt` → `pip install` → `COPY app` means a code-only edit never re-triggers the requirements layer — and on this image that layer is the expensive one.
- Be honest about image weight up front: `sentence-transformers` pulls in torch, and on Linux pip resolves the full CUDA/cuDNN/NCCL/triton GPU wheel stack right alongside it even though this is CPU-only inference — the measured image was 9.35GB. That number, and its real cause, is the seed for the slimming stretch, not a thing to gloss over.

## Live-coding order

1. `backend/app/memory.py` — `DB_PATH` reads `DESK_DB_PATH` with the existing relative path as fallback: the SQLite checkpointer and the long-term store both point at one env-overridable path.
2. `backend/app/tools.py` and `backend/app/ingest.py` — `CHROMA_DIR` gets the same treatment; the retrieval tool and the ingest CLI must agree on where the index lives, or the container "loses" data that's actually one directory over.
3. `backend/app/main.py` — `GET /health`: `filings_indexed` from listing `CHROMA_DIR`, wrapped so an `OSError` degrades to `False` instead of a 500. This is the compose healthcheck's target, written before compose exists. Caveat: `filings_indexed` is a best-effort directory probe — a filings question asked before any ingest lazily creates the Chroma dir, flipping it true with zero documents indexed.
4. `backend/Dockerfile` — narrate each layer out loud: base image, `curl` installed only for the healthcheck probe, requirements before code, `EXPOSE`, the `uvicorn` CMD binding `0.0.0.0` (not `127.0.0.1` — the container's loopback isn't the host's).
5. `backend/.dockerignore` — what must never enter the build context: `.env` (secrets baked into a layer are permanent, deleted-later or not), `.venv` (host-built, wrong platform), `chroma_db` / `desk_state.db*` (local state, not seed data).
6. `docker-compose.yml` — the `desk-state` named volume, `./data:/data:ro` (filings are read-only from inside the container), the healthcheck block, `ALPHADESK_PORT` as the one host-side knob.
7. Terminal — `docker compose up --build`, then the demo script.

## Demo script

1. Fresh `docker compose up --build` — first build ~7 minutes (the requirements layer doing the work). `curl localhost:8000/health` answers within 5 seconds of the container starting: `filings_indexed:false`. A brand-new volume is honestly empty — the health check isn't lying, there's nothing indexed yet.
2. `docker compose exec backend python -m app.ingest /data/filings` — indexes the sample filing as 22 chunks. Re-run `/health`: `filings_indexed:true`.
3. Chat on thread `deploy-t1`: "I watch AAPL" — teaches the watchlist fact. `docker compose restart backend`. Open a brand-new thread `deploy-t2` and ask "What do I watch?" → "You are watching Apple Inc. (AAPL)." Cross-thread *and* cross-restart: the checkpointer only ever knew `deploy-t1`'s messages, so this answer can only be coming from the volume-backed long-term store, not the checkpoint.
4. `docker compose ps` — reads `(healthy)`. Point at `start_period: 30s` in the compose file against the measured 5-second response: ample margin on this machine — tune it down only if you enjoy living dangerously.
5. Simulate a taken port: `ALPHADESK_PORT=8010 docker compose up` — same desk, new host port, zero code changes. (Not hypothetical here — host 8000 was occupied during verification, so 8010 is the port this was actually checked on.)

## Where it breaks

- **Container can't see the index.** Ingest ran on the host (`.venv/bin/python -m app.ingest`) against `backend/chroma_db/`, not inside the container against `/state/chroma_db` — same code, different `CHROMA_DIR`, two indexes that never converge.
- **State vanished after a rebuild.** `docker compose down -v` deletes the named volume along with the containers — plain `down` is safe, the `-v` is the whole story.
- **Healthcheck flaps `starting` → `unhealthy` during boot.** `start_period` is too short for whatever warms up slowest (an SDK client, a model load). Measured here: the app answered `/health` within 5 seconds, so the 30-second default has margin to spare — a slower box or a heavier model needs `start_period` raised, not the retries loosened.
- **Secrets baked into the image.** `.env` missing from `.dockerignore` means a `COPY` picks up the API key inside a layer forever — deleting the file in a later layer doesn't remove it from the image history.
- **Every code edit re-downloads torch.** `COPY app ./app` landed above `COPY requirements.txt` / `pip install`, so Docker invalidates the cached requirements layer on every source change. Reorder: requirements first, code last.
- **First ingest or query after recreating the container stalls for a while, every time.** The ~79MB ONNX embedding-model cache that `sentence-transformers`/Chroma downloads on first use lands in the container's default cache directory, not under `/state` — it survives a plain `restart` (same container, same filesystem) but is gone the moment the container is *recreated* (`down` then `up`, or an image rebuild), so the next ingest or query silently re-downloads it. The persistence demo above uses `restart`, which is exactly why it doesn't show this; swap in a `down`/`up` cycle and the same "what do I watch" query pauses noticeably before answering.

## Stretch

> **Budget note:** 81 insertions / 4 deletions across app code + Dockerfile + compose + `.gitignore` (Tasks 5–6) — far under the ~250 target.

- Slim the image: 9.35GB isn't "torch is big," it's pip resolving the full CUDA/cuDNN/NCCL/triton GPU wheel stack alongside torch for a workload that never touches a GPU — point `pip install` at PyTorch's CPU-only wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`), or split requirements so `RERANK_ENABLED=false` skips installing `sentence-transformers`/torch at all.
- A docker-compose profile that runs local Opik alongside the desk, so `OPIK_TRACING=true` has somewhere to send traces without a manual `opik server` step.
- A multi-stage build (builder stage installs, runtime stage copies just the venv) to separate build tooling from what actually ships.
- A GitHub Actions workflow that builds the image and runs `eval_retrieval` against it as CI — the eval harness Session 05 built, now gating merges instead of scrolling past in a terminal.
