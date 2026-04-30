from aiohttp import request
from fastapi import FastAPI
import time
import threading
import os
import random

from common.models import Response
from rag.model import retriever, llm, prompt

app = FastAPI()

MAX_Concurrent = int(os.getenv("MAX_CONCURRENT", 5))
FAIL_RATE = float(os.getenv("FAIL_RATE", 0.1))

class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.lock = threading.Lock()
        self.active_requests = 0

    def process(self, request: dict):
        start = time.time()

        # check and increment atomically in one lock
        with self.lock:
            if self.active_requests >= MAX_Concurrent:
                return Response(
                    id=request.get("id"),
                    status="failed",
                    error="Capacity full",
                    worker_id=self.id
                ).to_dict()
            self.active_requests += 1

        try:
            query = request["query"]
            req_id = request["id"]

            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            formatted = prompt.format(context=context, question=query)
            result = llm.invoke(formatted).content

            print(f"[Worker {self.id}] Result for request {req_id}: {result}")

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
            print(f"[Worker {self.id}] Error processing request {request.get('id')}: {e}")
            return Response(
                id=request.get("id"),
                status="failed",
                error=str(e),
                worker_id=self.id
            ).to_dict()

        finally:
            with self.lock:
                self.active_requests -= 1

worker = Worker(
    worker_id=int(os.getenv("WORKER_ID", 1)),
)


@app.post("/process")
def process(request: dict):
    try:
        print(f"[Worker {worker.id}] Processing request {request.get('id')}")
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
