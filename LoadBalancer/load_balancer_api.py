from fastapi import FastAPI
import requests
import threading

app = FastAPI()


class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0
        self.lock = threading.Lock()

    def get_next(self):
        with self.lock:
            w = self.workers[self.index]
            self.index = (self.index + 1) % len(self.workers)
        return w

    def dispatch(self, request):
        for _ in range(len(self.workers)):
            w = self.get_next()
            try:
                res = requests.post(f"{w}/process", json=request, timeout=10)
                return res.json()
            except Exception:
                continue

        raise Exception("All workers failed")


lb = LoadBalancer([
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003"
])


@app.post("/dispatch")
def dispatch(request: dict):
    return lb.dispatch(request)