# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University distributed-systems project: a master/worker LLM-serving system with retry-based fault tolerance, least-loaded routing, and a RAG pipeline. Workers do **not** run the LLM locally — they retrieve context from a FAISS index and proxy generation to remote **GPU servers** (Cloudflare-tunneled URLs configured per worker in `docker-compose.yml`). The repo is the orchestration / RAG / load-test side; the GPU server itself is a separate service.

## Run modes

Two completely separate run paths — they use different ports and different worker→GPU routing.

### Local (Windows, no Docker) — `run.bat`
- Starts 3 workers on `8001/8002/8003`, master on `8000`, then `client/load_generator.py`.
- Master reads `WORKER_URLS=http://localhost:8001,8002,8003`.
- Logs go to `./logs/{worker1,worker2,worker3,master,client}.log`.
- `run.bat` loads vars from `.env` (GROQ_API_KEY, HF_TOKEN, NUM_USERS, RAMP_UP_RATE, MAX_RETRIES, FAIL_RATE, MAX_CONCURRENT) and forwards them per-process. **It does NOT forward `GPU_SERVER_URL`**, so workers started via `run.bat` will fail any real LLM call unless you export `GPU_SERVER_URL` into each worker's environment yourself.
- `run_commands_no_docker.txt` has the manual equivalent if you want to start services one-by-one in separate shells.

### Docker — `docker-compose up`
- nginx (`:80`) → master1 (`:7000`) → worker1..worker4 (`:8000` each, internal) → external GPU servers.
- Each `workerN` service has its own hard-coded `GPU_SERVER_URL` in `docker-compose.yml` (Cloudflare tunnels). These URLs rotate; if a tunnel is dead, that worker will be `degraded` and excluded from routing.
- Client service hits `http://nginx:80`.

**Port-mismatch trap:** `run.bat` runs master on `8000`, but `master/Dockerfile` and `nginx.conf` expect master on `7000`. Don't "fix" one to match the other without checking — they're two separate deployment modes.

### Single-test commands
```bash
# Manual smoke-test of a single worker (PowerShell, from repo root):
$env:WORKER_ID=1; $env:GPU_SERVER_URL="https://..."; $env:HF_TOKEN="..."; $env:PYTHONPATH=$PWD
cd workers; uvicorn app:app --host 0.0.0.0 --port 8001

# Hit endpoints directly:
curl http://localhost:8000/health
curl http://localhost:8000/analytics
curl -X DELETE http://localhost:8000/analytics/reset
curl -X POST http://localhost:8000/handle -H "Content-Type: application/json" -d '{"id":1,"query":"who scored most UCL goals?"}'

# Run the RAG component standalone (sanity check FAISS/embeddings):
cd workers/rag; python test.py    # NOTE: test.py imports `llm` which model.py no longer exports — it's broken; treat as scratch
```

There is **no test suite, no linter config, and no build step.** Don't invent one.

## Architecture

```
client/load_generator.py  ──POST /handle──▶  master/app.py  ──POST /process──▶  workers/app.py  ──POST /generate──▶  remote GPU server
                                                  │                                   │
                                                  └─ polls /health for routing        └─ FAISS retrieval over rag/my_bio.txt
```

### Key responsibilities
- **`master/app.py`** — least-loaded routing with retries. On every `/handle` it calls `get_worker_loads()` which polls each worker's `/health` synchronously, builds a `{url: active_requests}` map of `status=="ok"` workers only, and picks the lowest (random tiebreak). Retries up to `MAX_RETRIES` with exponential-ish backoff (`time.sleep(attempt)`). Tracks per-worker analytics + GPU snapshots under two locks (`lock` for counters, `gpu_metrics_lock` for GPU state). `/analytics` triggers a fresh health-poll so the GPU section is current.

- **`workers/app.py`** — capacity-gated single-worker FastAPI. `Worker.process()` enforces `MAX_CONCURRENT` via a counter under `self.lock`, runs FAISS retrieval (`retriever.invoke`), formats the prompt, and calls `invoke_llm()` which POSTs to `{GPU_SERVER_URL}/generate`. The Worker object is constructed once at import time. `/health` proxies the GPU server's `/health` so the master sees real GPU metrics — if the GPU is unreachable the worker reports `status=="degraded"` and the master excludes it from routing.

- **`workers/rag/model.py`** — module-level RAG init runs at import: loads `my_bio.txt` (UEFA trivia), splits with `RecursiveCharacterTextSplitter(400/100)`, embeds with HuggingFace endpoint (`all-MiniLM-L6-v2`, requires `HF_TOKEN`), builds FAISS retriever (`k=2`). This is why workers take ~20s to start (the `run.bat` `timeout /t 20` is for this). Exports `retriever`, `prompt`, `invoke_llm`. **Does not export `llm`** — `rag/test.py` is stale.

- **`client/load_generator.py`** — threaded load test. `RAMP_UP_RATE` users per second (sleeps `1/RAMP_UP_RATE` between thread starts), each thread fires one `/handle` request and reports latency. After all threads join, prints client-side stats then fetches `/analytics` for server-side stats including per-worker GPU metrics.

- **`common/models.py`** — `Request` and `Response` dataclasses shared across master/worker/client. `Response` has GPU/token/vm fields; the worker fills these from the GPU server's response. Both use `asdict()` for JSON.

- **`LoadBalancer/`** — old/unused. The original design had a separate LB service; the master now does routing inline. Don't add new code here without checking with the user — it may be dead.

- **`main.py`** — fully commented out. Vestigial.

### Routing/health invariants worth preserving
- The master decides routing **only** from worker `/health`. Anything that breaks worker `/health` (e.g. GPU server timeout) silently excludes that worker.
- `active_requests` is incremented inside the lock **before** the LLM call and decremented in `finally`. Per-worker concurrency cap is enforced before increment, so an overloaded worker returns `status="failed"` with `error="Capacity full"` and the master's retry loop will pick a different worker.
- Master's `/analytics` is the canonical scoreboard. Reset it between load tests with `DELETE /analytics/reset`.

## Conventions / gotchas

- `.env` contains real API keys (Groq, HF) and is gitignored. Don't commit it. `GROQ_API_KEYS` is plural (comma-separated rotation list) but the worker code only reads `GPU_SERVER_URL` — Groq keys are not currently wired in (legacy from a prior LangChain+Groq path).
- The query in `client/load_generator.py` is hard-coded to a UEFA-trivia question because `my_bio.txt` is UEFA trivia despite its name.
- `MAX_NEW_TOKENS` defaults to 150. The GPU server's response shape is expected to include `answer`, `metrics`, `total_latency_seconds`, `input_tokens`, `output_tokens`, `vm_label`, and an `ok` boolean — `invoke_llm()` will raise if `ok` is falsy.
- Master's `Dockerfile` uses `COPY Master/...` (capital M); the directory on disk is lowercase `master/`. On case-insensitive filesystems (Windows) this works; on Linux Docker builds it will fail. Same pattern in `client/Dockerfile`. If you touch Dockerfiles, fix the case.
- Worker timeouts: master→worker is 240s, worker→GPU is 240s, client→master is 9000s. Don't lower these without understanding cold-start latency on the remote GPU servers.
- Worker `/health` timeout (master side) is 5s — a slow GPU `/health` will mark the worker unreachable.
