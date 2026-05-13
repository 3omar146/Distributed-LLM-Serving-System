from fastapi import FastAPI
import time
import threading
import queue
import os
import requests
import concurrent.futures
from common.models import Response
from rag.model import retriever, invoke_llm_batch, prompt

app = FastAPI()

# Batching configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 30))           # Max requests per batch
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", 0.5)) # Max seconds to wait for a full batch
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "").strip()
RAG_PARALLELISM = int(os.getenv("RAG_PARALLELISM", 8)) # Threads for RAG retrieval


class DynamicBatchWorker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.request_queue = queue.Queue()
        self.active_requests = 0
        self.lock = threading.Lock()

        # Reused thread pool for parallel RAG retrieval inside each batch.
        # Created once so we don't pay thread-startup cost on every batch.
        self.rag_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=RAG_PARALLELISM,
            thread_name_prefix=f"worker{worker_id}-rag",
        )

        # Single batching thread ("bus driver"). Could be doubled for overlap;
        # for assignment scope one is enough and keeps batching deterministic.
        self.processor_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processor_thread.start()

    def _process_batches(self):
        """Forms batches and dispatches them. Runs forever."""
        while True:
            batch = []
            try:
                # Block until at least one request shows up.
                first_task = self.request_queue.get(block=True)
                batch.append(first_task)

                # Greedily drain any other already-queued requests with no
                # additional wait — they're here right now.
                while len(batch) < BATCH_SIZE:
                    try:
                        batch.append(self.request_queue.get_nowait())
                    except queue.Empty:
                        break

                # If we still have room, wait up to BATCH_TIMEOUT for more.
                if len(batch) < BATCH_SIZE:
                    end_time = time.time() + BATCH_TIMEOUT
                    while len(batch) < BATCH_SIZE:
                        remaining = end_time - time.time()
                        if remaining <= 0:
                            break
                        try:
                            task = self.request_queue.get(block=True, timeout=remaining)
                            batch.append(task)
                        except queue.Empty:
                            break

                self._execute_batch(batch)

            except Exception as e:
                print(f"[Worker {self.id}] Critical batch loop error: {e}")
                # Fail any tasks we already pulled so callers don't hang.
                for task in batch:
                    if not task["event"].is_set():
                        task["response"] = Response(
                            id=task["request"].get("id"),
                            status="failed",
                            error=f"Batch loop error: {e}",
                            worker_id=self.id,
                        ).to_dict()
                        task["event"].set()
                with self.lock:
                    self.active_requests -= len(batch)

    def _retrieve_context(self, query):
        """Runs in the RAG thread pool — one HTTP round-trip to embeddings."""
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

    def _execute_batch(self, batch):
        """Run RAG + batched LLM inference for a group of requests."""
        start = time.time()
        queries = [task["request"]["query"] for task in batch]
        req_ids = [task["request"]["id"] for task in batch]

        print(f"[Worker {self.id}] Processing BATCH of {len(batch)} requests...")

        try:
            # Step 1: RAG retrieval IN PARALLEL. Each retriever.invoke is an
            # HTTP call to the embeddings endpoint, so doing them serially
            # would dominate batch latency.
            contexts = list(self.rag_pool.map(self._retrieve_context, queries))

            formatted_prompts = [
                prompt.format(context=ctx, question=q)
                for ctx, q in zip(contexts, queries)
            ]

            # Step 2: ONE batched call to the GPU. The list arrives at the GPU
            # server as a list, which runs a single batched model.generate().
            llm_responses = invoke_llm_batch(formatted_prompts)

            if len(llm_responses) != len(batch):
                raise RuntimeError(
                    f"LLM returned {len(llm_responses)} responses for "
                    f"{len(batch)} prompts"
                )

            # Step 3: Hand each result back to the right waiting request.
            latency = time.time() - start
            for i, task in enumerate(batch):
                resp_data = llm_responses[i] or {}
                answer = resp_data.get("answer", "No answer")
                gpu_metrics = resp_data.get("metrics", {})

                task["response"] = Response(
                    id=req_ids[i],
                    result=answer,
                    latency=latency,
                    status="success",
                    worker_id=self.id,
                    created_at=task["request"].get("created_at"),
                    gpu_metrics=gpu_metrics,
                ).to_dict()
                task["event"].set()

        except Exception as e:
            print(f"[Worker {self.id}] Batch execution failed: {e}")
            for task in batch:
                if not task["event"].is_set():
                    task["response"] = Response(
                        id=task["request"]["id"],
                        status="failed",
                        error=str(e),
                        worker_id=self.id,
                    ).to_dict()
                    task["event"].set()
        finally:
            # Single source of truth for active_requests bookkeeping.
            # queue_request() never decrements — only this finally does.
            with self.lock:
                self.active_requests -= len(batch)

    def queue_request(self, request_dict):
        """Drop the request into the queue and block until the batch finishes."""
        task = {
            "request": request_dict,
            "event": threading.Event(),
            "response": None,
        }
        with self.lock:
            self.active_requests += 1
        self.request_queue.put(task)

        # Wait for the bus driver to wake us. _execute_batch's finally always
        # runs and decrements active_requests, so we DO NOT decrement here on
        completed = task["event"].wait(timeout=300)

        if completed and task["response"]:
            return task["response"]

        return Response(
            id=request_dict.get("id"),
            status="failed",
            error="Worker timeout waiting for batch processing",
            worker_id=self.id,
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
        r = requests.get(f"{GPU_SERVER_URL}/health", timeout=300)
        if r.status_code == 200:
            data = r.json()
            return {
                "status": "ok",
                "worker_id": worker.id,
                "active_requests": worker.active_requests,
                "queue_depth": worker.request_queue.qsize(),
                "gpu_backend": True,
                "vm_label": data.get("vm_label"),
                "device": data.get("device"),
                "model": data.get("model"),
                "gpu_metrics": data.get("metrics", {}),
            }
    except Exception:
        pass
    print(f"[Worker {worker.id}] GPU server unreachable at {GPU_SERVER_URL}")
    return {
        "status": "degraded",
        "worker_id": worker.id,
        "active_requests": worker.active_requests,
        "queue_depth": worker.request_queue.qsize(),
        "gpu_backend": False,
    }