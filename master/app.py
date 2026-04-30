from fastapi import FastAPI
import requests
import time
import os
import random
import threading

app = FastAPI()

WORKER_URLS = os.getenv(
    "WORKER_URLS",
    "http://localhost:8001"
).split(",")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))

lock = threading.Lock()

# analytics tracking
analytics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "requests_per_worker": {url: 0 for url in WORKER_URLS},
    "successes_per_worker": {url: 0 for url in WORKER_URLS},
    "failures_per_worker": {url: 0 for url in WORKER_URLS},
}


def get_worker_loads():
    loads = {}
    for url in WORKER_URLS:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            data = r.json()
            if data.get("status") == "ok":
                loads[url] = data.get("active_requests", 0)
        except Exception as e:
            print(f"[Master] Worker {url} excluded from pool: {e}")
    print(f"[Master] Current worker loads: {loads}")
    return loads


def get_best_worker():
    loads = get_worker_loads()
    if not loads:
        return None
    min_load = min(loads.values())
    candidates = [url for url, load in loads.items() if load == min_load]
    return random.choice(candidates)


@app.post("/handle")
def handle(request: dict):
    attempt = 0
    print(f"[Master] Received request {request.get('id')}")

    # count total requests
    with lock:
        analytics["total_requests"] += 1

    while attempt <= MAX_RETRIES:
        worker_url = get_best_worker()

        if not worker_url:
            with lock:
                analytics["failed_requests"] += 1
            return {
                "id": request.get("id"),
                "status": "failed",
                "error": "No healthy workers available"
            }

        try:
            print(f"[Master] Sending request {request.get('id')} to {worker_url} (attempt {attempt + 1})")

            # count request sent to this worker
            with lock:
                analytics["requests_per_worker"][worker_url] += 1

            res = requests.post(
                f"{worker_url}/process",
                json=request,
                timeout=120
            )
            data = res.json()

            if data.get("status") == "failed":
                raise Exception(data.get("error", "Worker failed"))

            # count success
            with lock:
                analytics["successful_requests"] += 1
                analytics["successes_per_worker"][worker_url] += 1

            return data

        except Exception as e:
            attempt += 1
            print(f"[Master] Request {request.get('id')} "
                  f"attempt {attempt} failed on {worker_url}: {e}")

            # count failure for this worker
            with lock:
                analytics["failures_per_worker"][worker_url] += 1

            if attempt > MAX_RETRIES:
                with lock:
                    analytics["failed_requests"] += 1
                return {
                    "id": request.get("id"),
                    "status": "failed",
                    "error": f"All {MAX_RETRIES} retries exhausted: {str(e)}"
                }

            time.sleep(attempt)


@app.get("/health")
def health():
    worker_loads = get_worker_loads()
    return {
        "status": "ok",
        "service": "master",
        "healthy_workers": len(worker_loads),
        "worker_loads": worker_loads
    }


@app.get("/analytics")
def get_analytics():
    with lock:
        total = analytics["total_requests"]
        success_rate = (
            round(analytics["successful_requests"] / total * 100, 2)
            if total > 0 else 0
        )
        return {
            "total_requests":       analytics["total_requests"],
            "successful_requests":  analytics["successful_requests"],
            "failed_requests":      analytics["failed_requests"],
            "success_rate":         f"{success_rate}%",
            "requests_per_worker":  analytics["requests_per_worker"],
            "successes_per_worker": analytics["successes_per_worker"],
            "failures_per_worker":  analytics["failures_per_worker"],
        }


@app.delete("/analytics/reset")
def reset_analytics():
    with lock:
        analytics["total_requests"] = 0
        analytics["successful_requests"] = 0
        analytics["failed_requests"] = 0
        analytics["requests_per_worker"] = {url: 0 for url in WORKER_URLS}
        analytics["successes_per_worker"] = {url: 0 for url in WORKER_URLS}
        analytics["failures_per_worker"] = {url: 0 for url in WORKER_URLS}
    return {"status": "analytics reset"}