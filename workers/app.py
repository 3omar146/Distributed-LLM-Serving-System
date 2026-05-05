from fastapi import FastAPI
import time
import threading
import os
import requests

from common.models import Response
from rag.model import retriever, invoke_llm, prompt

app = FastAPI()

MAX_Concurrent = int(os.getenv("MAX_CONCURRENT", 4))
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "").strip()


class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.lock = threading.Lock()
        self.active_requests = 0

    def process(self, request: dict):
        start = time.time()

        with self.lock:
            if self.active_requests >= MAX_Concurrent:
                return Response(
                    id=request.get("id"),
                    status="failed",
                    error="Capacity full",
                    worker_id=self.id,
                ).to_dict()
            self.active_requests += 1

        try:
            query = request["query"]
            req_id = request["id"]

            # RAG retrieval
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Format prompt and call GPU server
            formatted = prompt.format(context=context, question=query)
            llm_response = invoke_llm(formatted)

            answer = llm_response["answer"]
            gpu_metrics = llm_response.get("metrics", {})
            gpu_latency = llm_response.get("total_latency_seconds")
            input_tokens = llm_response.get("input_tokens")
            output_tokens = llm_response.get("output_tokens")
            vm_label = llm_response.get("vm_label")

            print(f"[Worker {self.id}] Result for request {req_id}: {answer[:80]}...")

            latency = time.time() - start

            return Response(
                id=req_id,
                result=answer,
                latency=latency,
                status="success",
                worker_id=self.id,
                created_at=request.get("created_at"),
                gpu_metrics=gpu_metrics,
                gpu_latency=gpu_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                vm_label=vm_label,
            ).to_dict()

        except Exception as e:
            print(f"[Worker {self.id}] Error processing request {request.get('id')}: {e}")
            return Response(
                id=request.get("id"),
                status="failed",
                error=str(e),
                worker_id=self.id,
            ).to_dict()

        finally:
            with self.lock:
                self.active_requests -= 1


worker = Worker(worker_id=int(os.getenv("WORKER_ID", 1)))


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
            worker_id=worker.id,
        ).to_dict()


@app.get("/health")
def health():
    """
    Probe the GPU server directly. If it's up, return its latest metrics.
    Master uses 'status' for routing decisions and 'gpu_metrics' for monitoring.
    """
    try:
        r = requests.get(f"{GPU_SERVER_URL}/health", timeout=100)
        if r.status_code == 200:
            data = r.json()
            return {
                "status": "ok",
                "worker_id": worker.id,
                "active_requests": worker.active_requests,
                "gpu_backend": True,
                "vm_label": data.get("vm_label"),
                "device": data.get("device"),
                "model": data.get("model"),
                "gpu_metrics": data.get("metrics", {}),
            }
    except Exception as e:
        print(f"[Worker {worker.id}] GPU server health check failed: {e}")

    return {
        "status": "degraded",
        "worker_id": worker.id,
        "active_requests": worker.active_requests,
        "gpu_backend": False,
    }