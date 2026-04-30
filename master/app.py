from fastapi import FastAPI
import requests
import time
import os
import random
import threading

app = FastAPI()

# workers come from environment variable
WORKER_URLS = os.getenv(
    "WORKER_URLS",
    "http://localhost:8001,http://localhost:8002,http://localhost:8003"
).split(",")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))

index = 0
lock = threading.Lock()


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
        return None  # all workers dead

    min_load = min(loads.values())
    candidates = [url for url, load in loads.items() if load == min_load]
    return random.choice(candidates)


@app.post("/handle")
def handle(request: dict):
    attempt = 0

    print(f"[Master] Received request {request.get('id')}")

    while attempt <= MAX_RETRIES:
        worker_url = get_best_worker()

        if not worker_url:
            return {
                "id": request.get("id"),
                "status": "failed",
                "error": "No healthy workers available"
            }

        try:
            print(f"[Master] Sending request {request.get('id')} to {worker_url} (attempt {attempt + 1})")
            res = requests.post(
                f"{worker_url}/process",
                json=request,
                timeout=120
            )
            data = res.json()

            if data.get("status") == "failed":
                raise Exception(data.get("error", "Worker failed"))

            return data

        except Exception as e:
            attempt += 1
            print(f"[Master] Request {request.get('id')} "
                  f"attempt {attempt} failed on {worker_url}: {e}")

            if attempt > MAX_RETRIES:
                return {
                    "id": request.get("id"),
                    "status": "failed",
                    "error": f"All {MAX_RETRIES} retries exhausted: {str(e)}"
                }

            time.sleep(0.5 * attempt)


@app.get("/health")
def health():
    worker_loads = get_worker_loads()
    return {
        "status": "ok",
        "service": "master",
        "healthy_workers": len(worker_loads),
        "worker_loads": worker_loads
    }