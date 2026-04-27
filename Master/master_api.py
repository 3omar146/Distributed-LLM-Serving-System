from fastapi import FastAPI
import requests
import time

app = FastAPI()

LB_URL = "http://localhost:9000"
MAX_RETRIES = 1


@app.post("/handle")
def handle(request: dict):
    attempt = 0

    while attempt <= MAX_RETRIES:
        try:
            res = requests.post(f"{LB_URL}/dispatch", json=request, timeout=120)
            return res.json()

        except Exception as e:
            attempt += 1

            if attempt > MAX_RETRIES:
                return {
                    "id": request["id"],
                    "status": "failed",
                    "error": str(e)
                }

            time.sleep(0.05 * attempt)