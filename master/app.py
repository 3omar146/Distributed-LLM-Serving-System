
from fastapi import FastAPI, HTTPException, Response
import requests
import time
import os
import random
import threading

app = FastAPI()

WORKER_URLS = [u.strip() for u in os.getenv(
    "WORKER_URLS",
    "http://localhost:8001",
).split(",") if u.strip()]

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
HEALTH_INTERVAL = float(os.getenv("HEALTH_INTERVAL", 3.0))   # seconds between sweeps
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", 10))     # per-worker probe timeout
FAIL_PROBES_THRESHOLD = int(os.getenv("FAIL_THRESHOLD", 3))

analytics_lock = threading.Lock()
gpu_metrics_lock = threading.Lock()
load_cache_lock = threading.Lock()

# Analytics tracking
analytics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "attempt_requests_per_worker":  {url: 0 for url in WORKER_URLS},
    "attempt_successes_per_worker": {url: 0 for url in WORKER_URLS},
    "attempt_failures_per_worker":  {url: 0 for url in WORKER_URLS},
}

# Latest GPU snapshot per worker (refreshed by the health-loop thread).
latest_gpu_status = {url: None for url in WORKER_URLS}

# Cached worker loads (refreshed by the health-loop thread).
load_cache = {
    "loads": {},   # url -> active_requests reported by worker
    "fails": {url: 0 for url in WORKER_URLS},  # consecutive failed probes
}


inflight_lock = threading.Lock()
inflight = {url: 0 for url in WORKER_URLS}


def _probe_worker(url):
    """One health probe. Returns (load_or_None, gpu_snapshot_dict)."""
    try:
        r = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
        data = r.json()
        snapshot = {
            "status":      data.get("status"),
            "vm_label":    data.get("vm_label"),
            "device":      data.get("device"),
            "model":       data.get("model"),
            "gpu_backend": data.get("gpu_backend"),
            "gpu_metrics": data.get("gpu_metrics", {}),
        }
        if data.get("status") == "ok":
            return data.get("active_requests", 0), snapshot
        return None, snapshot
    except Exception as e:
        return None, {"status": "unreachable", "error": str(e)}


def _probe_loop(url):
    """Per-worker probe thread. Excludes a worker after FAIL_THRESHOLD
    consecutive failed probes; re-includes immediately on the next success."""
    while True:
        load, snapshot = _probe_worker(url)

        with gpu_metrics_lock:
            latest_gpu_status[url] = snapshot

        with load_cache_lock:
            if load is not None:
                # Healthy: refresh load, reset fail counter
                load_cache["loads"][url] = load
                load_cache["fails"][url] = 0
            else:
                # Failed probe: increment, evict if we've crossed the threshold
                load_cache["fails"][url] += 1
                if load_cache["fails"][url] == FAIL_PROBES_THRESHOLD:
                    load_cache["loads"].pop(url, None)
                    print(f"[Master] Worker {url} evicted after "f"{load_cache['fails'][url]} failed probes")

        time.sleep(HEALTH_INTERVAL)


# Kick off one probe thread per worker at import time.
for _url in WORKER_URLS:
    threading.Thread(target=_probe_loop, args=(_url,), daemon=True).start()


def get_best_worker(exclude=None):
    """Pick the worker with the lowest effective load (cached + in-flight).

    `exclude` is an optional URL to skip — used on retries so we don't
    immediately re-send to the worker that just failed. Falls back to the
    excluded worker if it's the only one healthy (better to retry the same
    one than give up).
    """
    with load_cache_lock:
        loads = dict(load_cache["loads"])

    if not loads:
        return None

    with inflight_lock:
        # Effective load = what the worker reported + what we've already sent
        # since its last health report. Prevents stampedes.
        effective = {
            url: loads[url] + inflight.get(url, 0)
            for url in loads if url != exclude
        }
        if not effective and exclude is not None and exclude in loads:
            effective = {exclude: loads[exclude] + inflight.get(exclude, 0)}

    min_load = min(effective.values())
    candidates = [url for url, load in effective.items() if load == min_load]
    return random.choice(candidates)


@app.post("/handle")
def handle(request: dict, response: Response):
    attempt = 0
    print(f"[Master] Received request {request.get('id')}")

    with analytics_lock:
        analytics["total_requests"] += 1

    last_worker = None

    while attempt <= MAX_RETRIES:
        worker_url = get_best_worker(exclude=last_worker)

        if not worker_url:
            with analytics_lock:
                analytics["failed_requests"] += 1
            print(f"[Master] No healthy workers available for request {request.get('id')}")
            raise HTTPException(
                    status_code=503, 
                    detail={"id": request.get("id"), "error": "No healthy workers available"}
                )

        try:
            print(
                f"[Master] Sending request {request.get('id')} to {worker_url} "
                f"(attempt {attempt + 1})"
            )

            with analytics_lock:
                analytics["requests_per_worker"][worker_url] = \
                    analytics["requests_per_worker"].get(worker_url, 0) + 1
            with inflight_lock:
                inflight[worker_url] = inflight.get(worker_url, 0) + 1
            last_worker = worker_url

            try:
                res = requests.post(
                    f"{worker_url}/process",
                    json=request,
                    timeout=305,
                )
                data = res.json()
            finally:
                # Always release the in-flight slot, success or fail.
                with inflight_lock:
                    inflight[worker_url] = max(0, inflight.get(worker_url, 0) - 1)

            if data.get("status") == "failed":
                raise Exception(data.get("error", "Worker failed"))

            with analytics_lock:
                analytics["successful_requests"] += 1
                analytics["successes_per_worker"][worker_url] = \
                    analytics["successes_per_worker"].get(worker_url, 0) + 1

            return data

        except Exception as e:
            attempt += 1
            print(
                f"[Master] Request {request.get('id')} attempt {attempt} "
                f"failed on {worker_url}: {e}"
            )

            with analytics_lock:
                analytics["failures_per_worker"][worker_url] = \
                    analytics["failures_per_worker"].get(worker_url, 0) + 1

            if attempt > MAX_RETRIES:
                with analytics_lock:
                    analytics["failed_requests"] += 1
                return {
                    "id": request.get("id"),
                    "status": "failed",
                    "worker_id": last_worker,
                    "error": f"All {MAX_RETRIES} retries exhausted: {str(e)}",
                }

            time.sleep(attempt)


@app.get("/health")
def health():
    with load_cache_lock:
        worker_loads = dict(load_cache["loads"])
    return {
        "status": "ok",
        "service": "master",
        "healthy_workers": len(worker_loads),
        "worker_loads": worker_loads,
    }


@app.get("/analytics")
def get_analytics():
    with analytics_lock:
        total = analytics["total_requests"]
        success_rate = (
            round(analytics["successful_requests"] / total * 100, 2)
            if total > 0 else 0
        )
        result = {
            "total_requests":       analytics["total_requests"],
            "successful_requests":  analytics["successful_requests"],
            "failed_requests":      analytics["failed_requests"],
            "success_rate":         f"{success_rate}%",
            "requests_per_worker":  dict(analytics["requests_per_worker"]),
            "successes_per_worker": dict(analytics["successes_per_worker"]),
            "failures_per_worker":  dict(analytics["failures_per_worker"]),
        }

    with gpu_metrics_lock:
        result["gpu_status_per_worker"] = {
            url: dict(info) if info else None
            for url, info in latest_gpu_status.items()
        }

    with load_cache_lock:
        result["current_worker_loads"] = dict(load_cache["loads"])

    with inflight_lock:
        result["in_flight_per_worker"] = dict(inflight)

    return result


@app.delete("/analytics/reset")
def reset_analytics():
    with analytics_lock:
        analytics["total_requests"] = 0
        analytics["successful_requests"] = 0
        analytics["failed_requests"] = 0
        analytics["requests_per_worker"]  = {url: 0 for url in WORKER_URLS}
        analytics["successes_per_worker"] = {url: 0 for url in WORKER_URLS}
        analytics["failures_per_worker"]  = {url: 0 for url in WORKER_URLS}
    with gpu_metrics_lock:
        for url in WORKER_URLS:
            latest_gpu_status[url] = None
    return {"status": "analytics reset"}