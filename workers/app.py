from fastapi import FastAPI
import time
import threading
import os
import random

from common.models import Response
from rag.model import retriever, llm, prompt

app = FastAPI()


class Worker:
    def __init__(self, worker_id, max_concurrent=5, fail_rate=0):
        self.id = worker_id
        self.lock = threading.Lock()
        self.active_requests = 0
        self.sem = threading.Semaphore(max_concurrent)
        self.fail_rate = fail_rate

    def process(self, request: dict):
        start = time.time()

        if not self.sem.acquire(blocking=False):
            raise Exception("Capacity full")

        with self.lock:
            self.active_requests += 1

        try:
            if random.random() < self.fail_rate:
                raise Exception("Simulated worker failure")

            query = request["query"]
            req_id = request["id"]

            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            formatted = prompt.format(context=context, question=query)
            result = llm.invoke(formatted).content

            latency = time.time() - start

            return Response(
                id=req_id,
                result=result,
                latency=latency,
                status="success",
                worker_id=self.id,
                created_at=request.get("created_at")
            ).to_dict()

        except Exception as e:
            return Response(
                id=request.get("id"),
                status="failed",
                error=str(e),
                worker_id=self.id
            ).to_dict()

        finally:
            with self.lock:
                self.active_requests -= 1
            self.sem.release()


worker = Worker(
    worker_id=int(os.getenv("WORKER_ID", 1)),
    max_concurrent=int(os.getenv("MAX_CONCURRENT", 2)),
    fail_rate=float(os.getenv("FAIL_RATE", 0.1))
)


@app.post("/process")
def process(request: dict):
    try:
        return worker.process(request)
    except Exception as e:
        return Response(
            id=request.get("id"),
            status="failed",
            error=str(e),
            worker_id=worker.id
        ).to_dict()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "worker_id": worker.id,
        "active_requests": worker.active_requests
    }
