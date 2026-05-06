from fastapi import FastAPI
import time
import threading
import queue
import os
import requests
from common.models import Response
from rag.model import retriever, invoke_llm_batch, prompt

app = FastAPI()

# Batching Configurations
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))        # Max requests per batch
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 0.5)) # Max seconds to wait for a full batch
MAX_QUEUE = int(os.getenv("MAX_CONCURRENT", 50))    # We can increase this now!
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "").strip()

class DynamicBatchWorker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.request_queue = queue.Queue()
        self.active_requests = 0
        self.lock = threading.Lock()
        
        # Start the "Bus Driver" background thread
        self.processor_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processor_thread.start()

    def _process_batches(self):
        """Runs endlessly in the background, grouping requests into batches."""
        while True:
            batch = []
            try:
                # 1. Wait for at least ONE request to arrive
                first_task = self.request_queue.get(block=True)
                batch.append(first_task)

                # 2. Wait up to 0.5 seconds to see if more requests arrive to fill the batch
                end_time = time.time() + BATCH_TIMEOUT
                while len(batch) < BATCH_SIZE:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        break
                    try:
                        task = self.request_queue.get(block=True, timeout=remaining_time)
                        batch.append(task)
                    except queue.Empty:
                        break # Timeout reached, bus is leaving!

                # 3. Process the collected batch
                self._execute_batch(batch)

            except Exception as e:
                print(f"[Worker {self.id}] Critical batch loop error: {e}")

    def _execute_batch(self, batch):
        """Executes RAG and LLM inference for a group of requests simultaneously."""
        start = time.time()
        queries = [task["request"]["query"] for task in batch]
        req_ids = [task["request"]["id"] for task in batch]

        print(f"[Worker {self.id}] Processing BATCH of {len(batch)} requests...")

        formatted_prompts = []
        try:
            # Step 1: Run RAG retrieval for all queries in the batch
            for query in queries:
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                formatted_prompts.append(prompt.format(context=context, question=query))

            # Step 2: Send ONE massive request to the GPU
            llm_responses = invoke_llm_batch(formatted_prompts)

            # Step 3: Distribute the answers back to the waiting users
            latency = time.time() - start
            for i, task in enumerate(batch):
                resp_data = llm_responses[i]
                answer = resp_data.get("answer", "No answer")
                gpu_metrics = resp_data.get("metrics", {})

                task["response"] = Response(
                    id=req_ids[i],
                    result=answer,
                    latency=latency,
                    status="success",
                    worker_id=self.id,
                    created_at=task["request"].get("created_at"),
                    gpu_metrics=gpu_metrics
                ).to_dict()
                
                # WAKE UP the FastAPI route!
                task["event"].set()

        except Exception as e:
            print(f"[Worker {self.id}] Batch execution failed: {e}")
            for task in batch:
                task["response"] = Response(
                    id=task["request"]["id"],
                    status="failed",
                    error=str(e),
                    worker_id=self.id
                ).to_dict()
                task["event"].set()
        finally:
            with self.lock:
                self.active_requests -= len(batch)

    def queue_request(self, request_dict):
        """FastAPI calls this. It drops the request in the queue and goes to sleep."""
        with self.lock:
            if self.active_requests >= MAX_QUEUE:
                return Response(
                    id=request_dict.get("id"),
                    status="failed",
                    error="Capacity full",
                    worker_id=self.id
                ).to_dict()
            self.active_requests += 1

        # Create a task with a threading Event (like a pager that buzzes when food is ready)
        task = {
            "request": request_dict,
            "event": threading.Event(),
            "response": None
        }
        
        # Put it in the queue for the background thread
        self.request_queue.put(task)

        # Go to sleep until the background thread calls task["event"].set()
        task["event"].wait(timeout=300)

        if task["response"]:
            return task["response"]
        else:
            with self.lock:
                self.active_requests -= 1
            return Response(
                id=request_dict.get("id"),
                status="failed",
                error="Worker timeout waiting for batch processing",
                worker_id=self.id
            ).to_dict()

worker = DynamicBatchWorker(worker_id=int(os.getenv("WORKER_ID", 1)))

@app.post("/process")
def process(request: dict):
    try:
        return worker.queue_request(request)
    except Exception as e:
        return Response(
            id=request.get("id"),
            status="failed",
            error=str(e),
            worker_id=worker.id,
        ).to_dict()

@app.get("/health")
def health():
    try:
        r = requests.get(f"{GPU_SERVER_URL}/health", timeout=5)
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
        pass
    return {
        "status": "degraded",
        "worker_id": worker.id,
        "active_requests": worker.active_requests,
        "gpu_backend": False,
    }