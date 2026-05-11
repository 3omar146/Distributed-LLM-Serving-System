# Distributed LLM Serving System

A five-tier distributed system for serving 1,000+ concurrent LLM + RAG requests across a cluster of local Docker services and remote GPU VMs, with dynamic batching, layered retries, and end-to-end fault tolerance.

> Final project branch: `elbatch_elnha2y_bel_fault_tolerance`
>
> Important: this repository contains several experimental branches (`API_based`, `batching`, `thundercompute`, `thunder1000`, `elbatch-elnha2y`). Those branches were trial stages during development. The final integrated system with batching + fault tolerance is the branch above.

## Table of Contents

- [Overview](#overview)
- [Branch Scope](#branch-scope)
- [Architectural Evolution](#architectural-evolution)
- [Final Architecture](#final-architecture)
- [Request Lifecycle](#request-lifecycle)
- [Core Components](#core-components)
- [Load Balancing and Batching](#load-balancing-and-batching)
- [Fault Tolerance](#fault-tolerance)
- [GPU Monitoring](#gpu-monitoring)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Reported Performance](#reported-performance)

## Overview

This project implements a distributed inference cluster that can absorb 1,000+ concurrent requests while remaining responsive, balanced, and resilient to worker or GPU failures. It was built for **CSE354: Distributed Computing at Ain Shams University (2nd Semester 2025/2026)** and combines:

- Retrieval-Augmented Generation (RAG) over a FAISS index
- A local worker tier that performs retrieval, prompt assembly, and dynamic batching
- A remote GPU tier that hosts **Qwen/Qwen2.5-0.5B-Instruct** on rented Thunder Compute A100 VMs
- Two layers of traffic distribution:
  - NGINX at the edge
  - Load-aware worker selection inside each master

The final design removes the need for a centralized external LLM API and turns the system into a real distributed serving pipeline with observable GPU-backed inference.

## Branch Scope

This README documents the code currently implemented on:

- `elbatch_elnha2y_bel_fault_tolerance`

It does **not** describe the older trial branches as the final architecture. The other branches exist to show the development path, but they should be treated as intermediate experiments rather than the final submission.

## Architectural Evolution

The final branch is the result of several iterations, each solving a bottleneck exposed by the previous design.

### Iteration 1: Single-process prototype

The earliest design used local Python classes in one process to simulate workers and a load balancer. This was useful for validating control flow, but it had no real network distribution and no real fault isolation. The commented `main.py` file in this repository is a leftover reference to that prototype stage.

### Iteration 2: Real LLM calls through an external API

The next step replaced dummy sleeps with actual LLM traffic. This made latency and failure behavior realistic, but all requests were still backed by the same centralized provider, which meant the system was not yet a real distributed GPU cluster.

### Iteration 3: Separate FastAPI processes

The system was then split into client, balancer, and worker processes communicating over HTTP. This introduced real inter-process communication and independent failure domains, but the custom balancing logic was still limited and deployment was awkward.

### Iteration 4: Docker + NGINX + master tier

Containerization and NGINX introduced cleaner deployment and stronger edge failover. The master tier was added to make worker selection and retries stateful and load-aware instead of purely connection-count based.

### Iteration 5: Local orchestration + remote GPU inference

The final branch moves model execution to rented Thunder Compute GPUs and keeps orchestration locally. This required:

- Dynamic batching at the worker tier
- Batched prompt submission to the GPU tier
- Left-padded causal LM inference on the GPU server
- Per-worker health probing and eviction at the master tier
- Propagation of GPU telemetry back to the client analytics layer

## Final Architecture

The final system is a five-tier pipeline. Every adjacent layer talks over JSON-over-HTTP. There is no shared filesystem, no message broker, and no shared database.

```text
+-------------+
|   Client    |  asyncio + httpx load generator
+-------------+
       |
       | HTTP
+------v------+
|    NGINX    |  edge load balancer
+-------------+
       |
   +---+---+
   |       |
+--v----+ +--v----+
|Master1| |Master2|  worker health probes + load-aware retries
+--+--+-+ +--+--+-+
   |  |      |  |
   w1 w2     w3 w4   local Docker workers
   |  |      |  |
  GPU GPU    GPU GPU  remote Thunder Compute VMs
```

### Split-worker design

Each logical worker is split across two environments:

- Local half (`workers/app.py`):
  - Accepts requests from the masters
  - Queues them into a dynamic batcher
  - Runs RAG retrieval in parallel
  - Sends the whole prompt batch to one remote GPU server
- Remote half (`batch_server.py`):
  - Loads the Qwen model
  - Serializes access to the GPU with a lock
  - Runs one batched `model.generate()` call
  - Samples GPU and PyTorch memory metrics

This split lets the orchestrating cluster stay CPU-only while GPU capacity is rented separately.

## Request Lifecycle

1. `client/load_generator.py` creates many concurrent asyncio tasks and POSTs requests to NGINX.
2. `nginx/nginx.conf` forwards each request to the master pool using `least_conn`.
3. The selected master receives `/handle`, increments analytics, and chooses the best worker based on:
   - The worker's most recently reported `active_requests`
   - An in-flight correction maintained by the master between health probes
4. The worker enqueues the request in `DynamicBatchWorker`.
5. The batcher:
   - Waits for the first request
   - Greedily drains already-queued requests
   - Waits up to `BATCH_TIMEOUT` for more arrivals if the batch is not full
6. For the completed batch, the worker:
   - Runs RAG retrieval in parallel using a thread pool
   - Formats prompts
   - Sends the entire prompt list in one HTTP request to the GPU server
7. The GPU server runs batched generation, returns one response per prompt, and the worker maps them back to the waiting callers.

## Core Components

### Client

`client/load_generator.py`

- Generates `NUM_USERS` concurrent requests using `httpx.AsyncClient`
- Measures client-visible latency and throughput
- Fetches analytics directly from both masters after the run
- Aggregates per-master worker and GPU telemetry into one combined report

### Edge load balancer

`nginx/nginx.conf`

- Uses `least_conn` across `master1` and `master2`
- Retries failed POST requests with:
  - `error`
  - `timeout`
  - `http_503`
  - `non_idempotent`
- Uses `max_fails=5` and `fail_timeout=10s` to temporarily bench unhealthy masters
- Keeps long-running inference requests open with a large `proxy_read_timeout`

### Master tier

`master/app.py`

- Spawns one background probe thread per worker
- Polls `/health` on workers every `HEALTH_INTERVAL`
- Evicts a worker after `FAIL_THRESHOLD` consecutive probe failures
- Re-adds the worker automatically on the next successful probe
- Maintains:
  - global request analytics
  - per-worker success/failure counters
  - latest GPU snapshots per worker
  - a cached load view
  - an in-flight correction map to avoid load stampedes

Retry semantics on `/handle`:

- One initial send attempt
- Up to `MAX_RETRIES` additional re-attempts after failures
- By default, this means up to 6 total worker attempts for one client request

### Worker tier

`workers/app.py`

- Runs one `DynamicBatchWorker` thread per container
- Maintains a queue of pending requests
- Tracks `active_requests` as the load signal consumed by the masters
- Uses a persistent `ThreadPoolExecutor` to parallelize RAG retrieval inside each batch
- Returns `degraded` on `/health` if the remote GPU server is unreachable

### RAG pipeline

`workers/rag/model.py`

- Loads `workers/rag/my_bio.txt`
- Splits text into overlapping chunks with:
  - `chunk_size=400`
  - `chunk_overlap=100`
- Builds an in-memory FAISS index at startup
- Uses `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEndpointEmbeddings`
- Retrieves `k=2` chunks per query
- Formats a concise QA prompt before sending it to the GPU tier

### Remote GPU server

`batch_server.py`

- Loads `Qwen/Qwen2.5-0.5B-Instruct`
- Uses FP16 on CUDA when available
- Sets tokenizer padding to the **left**, which is required for proper causal LM batch generation
- Accepts either:
  - a single prompt string
  - a list of prompt strings for true batch inference
- Serializes access with `generate_lock`
- Stores the metrics from the most recently completed batch so `/health` reflects real inference-time GPU usage

### Legacy reference files

- `main.py`: commented prototype from the early single-process design
- `no_batch_server.py`: older non-batched GPU server kept for comparison/debugging

## Load Balancing and Batching

Load balancing happens at two different tiers:

| Strategy | Location | Purpose |
| --- | --- | --- |
| `least_conn` | NGINX -> masters | Choose the less busy master at the edge |
| Load-aware routing | Master -> workers | Choose the worker with the lowest effective load |

The master-side effective load is not just the last reported `active_requests`. It also includes an in-flight correction so a burst of requests does not all get routed to the same worker between probe intervals.

Batching happens only at the worker tier:

- `BATCH_SIZE` caps the batch size
- `BATCH_TIMEOUT` caps how long the worker waits for a fuller batch
- RAG retrieval is still done per request, but in parallel inside the batch
- LLM generation is done once per batch on the GPU server

This design is what makes the system both distributed and GPU-efficient: requests stay independent at the orchestration layer, but generation is amortized at the model layer.

## Fault Tolerance

Fault tolerance is layered. Each tier handles failures in the tier directly below it.

| Failure | Detected by | Recovery |
| --- | --- | --- |
| One worker request fails | Master `/handle` | Retry on another worker, excluding the most recently failed worker when possible |
| Worker becomes slow or unreachable | Master probe thread | Evict after `FAIL_THRESHOLD` failed probes; restore on first healthy probe |
| All workers under one master are unavailable | Master | Return HTTP `503`, allowing NGINX to retry another master |
| Master is down or timing out | NGINX | Retry another master via `proxy_next_upstream` |
| Remote GPU VM dies | Worker `/process` path and later master probes | Batch fails fast, master retries elsewhere, dead worker is eventually evicted |
| Container restarts | Docker Compose health checks + `restart: unless-stopped` | Service returns to the pool after it becomes healthy again |

## GPU Monitoring

The GPU server exports the following metrics for the last completed batch:

```text
gpu_name
gpu_utilization_percent
gpu_memory_used_mb
gpu_memory_total_mb
gpu_memory_used_percent
gpu_temperature_c
gpu_power_watts
torch_memory_allocated_mb
torch_memory_reserved_mb
torch_peak_memory_mb
gpu_available
```

Metric flow through the system:

```text
batch_server.py -> worker /health -> master probe cache -> master /analytics -> client report
```

An important detail in the implementation is that metrics are sampled **inside** the GPU lock and **after** `torch.cuda.synchronize()`. That avoids misleading idle readings from NVML.

## Repository Layout

```text
.
|- batch_server.py           # Final remote GPU batch inference server
|- no_batch_server.py        # Older single-request GPU server
|- docker-compose.yml
|- nginx/
|  `- nginx.conf
|- master/
|  |- app.py
|  |- Dockerfile
|  `- requirements.txt
|- workers/
|  |- app.py
|  |- Dockerfile
|  |- requirements.txt
|  `- rag/
|     |- model.py
|     `- my_bio.txt
|- client/
|  |- load_generator.py
|  |- Dockerfile
|  `- requirements.txt
`- common/
   `- models.py
```

## Configuration

### Master configuration

| Variable | Location | Default | Meaning |
| --- | --- | --- | --- |
| `WORKER_URLS` | `master/app.py` | `http://localhost:8001` | Comma-separated worker URLs |
| `MAX_RETRIES` | `master/app.py` | `5` | Retry count after the initial worker attempt |
| `HEALTH_INTERVAL` | `master/app.py` | `3.0` | Seconds between probe sweeps |
| `HEALTH_TIMEOUT` | `master/app.py` | `10` | Timeout for each worker health probe |
| `FAIL_THRESHOLD` | `master/app.py` | `3` | Failed probes required before eviction |

### Worker configuration

| Variable | Location | Default | Meaning |
| --- | --- | --- | --- |
| `WORKER_ID` | `workers/app.py` | `1` | Numeric worker identifier |
| `BATCH_SIZE` | `workers/app.py` | `30` | Maximum requests per batch |
| `BATCH_TIMEOUT` | `workers/app.py` | `0.5` | Maximum time to wait for a fuller batch |
| `GPU_SERVER_URL` | `workers/app.py` and `workers/rag/model.py` | empty | Remote GPU server URL |
| `RAG_PARALLELISM` | `workers/app.py` | `8` | Size of the RAG retrieval thread pool |
| `MAX_NEW_TOKENS` | `workers/rag/model.py` | `150` | Generation cap forwarded to the GPU tier |
| `HF_TOKEN` | `workers/rag/model.py` | empty | Token used for Hugging Face endpoint embeddings |

### GPU server configuration

| Variable | Location | Default | Meaning |
| --- | --- | --- | --- |
| `MODEL_NAME` | `batch_server.py` | `Qwen/Qwen2.5-0.5B-Instruct` | Model loaded on the GPU VM |
| `VM_LABEL` | `batch_server.py` | `gpu-vm` | Friendly name included in health telemetry |

### Client configuration

| Variable | Location | Default | Meaning |
| --- | --- | --- | --- |
| `MASTER_URL` | `client/load_generator.py` | `http://localhost:8000` | Load-test target, usually NGINX |
| `MASTER_DIRECT_URLS` | `client/load_generator.py` | `http://master1:7000,http://master2:7000` | Direct master URLs used only for analytics |
| `NUM_USERS` | `client/load_generator.py` | `1000` | Number of concurrent users to simulate |
| `RAMP_UP_RATE` | `client/load_generator.py` | `0` | Users per second; `0` means immediate burst |
| `REQUEST_TIMEOUT` | `client/load_generator.py` | `600` | Client-side request timeout in seconds |

### Docker Compose wiring

`docker-compose.yml` expects the `.env` file to provide the remote worker endpoints:

```env
WORKER1=https://<gpu-vm-1>
WORKER2=https://<gpu-vm-2>
WORKER3=https://<gpu-vm-3>
WORKER4=https://<gpu-vm-4>
HF_TOKEN=<your-hugging-face-token>
```

## Deployment

### Prerequisites

- Docker and Docker Compose
- 4 reachable GPU VMs (Thunder Compute in the final setup)
- Python environment on each GPU VM with:
  - `fastapi`
  - `uvicorn`
  - `transformers`
  - `torch`
  - `pynvml`
- Hugging Face access for:
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `sentence-transformers/all-MiniLM-L6-v2` endpoint embeddings

### 1. Start the GPU servers

Run this on each GPU VM:

```bash
pip install fastapi uvicorn transformers torch pynvml
export VM_LABEL=gpu-vm-1
uvicorn batch_server:app --host 0.0.0.0 --port 8000
```

Optional:

- Set `MODEL_NAME` if you want a different compatible Hugging Face model
- Use a process manager (`nohup`, `screen`, `tmux`, `systemd`) in a real deployment

### 2. Configure the local stack

Set the worker endpoint URLs in `.env` so each local worker points at its remote GPU server:

```env
WORKER1=https://<gpu-vm-1-url>
WORKER2=https://<gpu-vm-2-url>
WORKER3=https://<gpu-vm-3-url>
WORKER4=https://<gpu-vm-4-url>
HF_TOKEN=<token>
```

### 3. Launch the distributed system

```bash
docker compose up --build
```

The dependency chain in `docker-compose.yml` ensures the edge does not start until the masters are healthy, and each master waits for its worker pool to become healthy first.

### 4. Run the load test

The `client` container runs `client/load_generator.py`. By default it targets NGINX and simulates 1,000 concurrent users.

If you want to run the client manually instead of through Compose:

```bash
python client/load_generator.py
```

## Reported Performance

The following figures come from the project's recorded load tests and describe the intended behavior of the final branch.

### Clean baseline: 1,000 concurrent users

| Metric | Value |
| --- | --- |
| Success / Fail | 1,000 / 0 (100.00% success) |
| Duration | 202.46 s |
| Throughput | 4.94 req/s |
| Avg / Median latency | 108.34 s / 111.03 s |
| Master split | master1 = 508, master2 = 492 |
| Worker split | w1 = 260, w2 = 248, w3 = 262, w4 = 230 |

### Fault-tolerance run: 1,000 users with a GPU kill mid-run

| Metric | Value |
| --- | --- |
| Success / Fail | 1,000 / 0 (100.00% success) |
| Duration | 316.08 s |
| Avg / Median latency | 147.10 s / 148.05 s |
| Batch failures absorbed | 60 total (w1: 40, w4: 20), all retried successfully |
