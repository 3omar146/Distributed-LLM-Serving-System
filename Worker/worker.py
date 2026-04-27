import time
import threading
import random


class Worker:
    def __init__(self, worker_id, max_concurrent=2, fail_rate=0.1):
        self.id = worker_id
        self.active_requests = 0
        self.lock = threading.Lock()

        # 🔥 capacity control
        self.semaphore = threading.Semaphore(max_concurrent)

        # 🔥 failure probability (0.1 = 10%)
        self.fail_rate = fail_rate

    def process(self, request):
        # 🔒 limit concurrent requests
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise Exception(f"Worker {self.id} is at full capacity")

        start_time = time.time()

        with self.lock:
            self.active_requests += 1

        try:
            # 🔥 simulate random failure
            if random.random() < self.fail_rate:
                raise Exception(f"Worker {self.id} randomly failed")

            query = request.query

            # 🔹 Step 1: RAG (stub)
            context = self.retrieve_context(query)

            # 🔹 Step 2: LLM (stub)
            result = self.run_llm(query, context)

            latency = time.time() - start_time

            response = {
                "request_id": request.id,
                "status": "success",
                "worker_id": self.id,
                "result": result,
                "latency": latency
            }

            print(f"[Worker {self.id}] Done request {request.id} in {latency:.3f}s")

            return response

        finally:
            # 🔒 always release resources
            with self.lock:
                self.active_requests -= 1

            self.semaphore.release()

    # ---------------------------
    # RAG (stub)
    # ---------------------------
    def retrieve_context(self, query):
        time.sleep(random.uniform(0.01, 0.05))
        return f"Context for: {query}"

    # ---------------------------
    # LLM (stub)
    # ---------------------------
    def run_llm(self, query, context):
        time.sleep(random.uniform(0.1, 0.3))
        return f"Answer to '{query}' using [{context}]"