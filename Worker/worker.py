import time
import threading
import random


class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.active_requests = 0
        self.lock = threading.Lock()

    def process(self, request):
        start_time = time.time()

        # 🔒 Track active requests (for monitoring / future load-aware)
        with self.lock:
            self.active_requests += 1

        try:
            query = request.get("query", "")

            # 🔹 Step 1: RAG (stub for now)
            context = self.retrieve_context(query)

            # 🔹 Step 2: LLM inference (stub for now)
            result = self.run_llm(query, context)

            latency = time.time() - start_time

            response = {
                "worker_id": self.id,
                "result": result,
                "latency": latency
            }

            print(f"[Worker {self.id}] Done request {request.get('id')} in {latency:.3f}s")

            return response

        finally:
            # 🔒 Decrement active requests even if error happens
            with self.lock:
                self.active_requests -= 1

    # ---------------------------
    # 🔹 RAG (stub)
    # ---------------------------
    def retrieve_context(self, query):
        # simulate retrieval delay
        time.sleep(random.uniform(0.01, 0.05))
        return f"Context for: {query}"

    # ---------------------------
    # 🔹 LLM (stub for now)
    # ---------------------------
    def run_llm(self, query, context):
        # simulate heavy computation
        time.sleep(random.uniform(0.1, 0.3))
        return f"Answer to '{query}' using [{context}]"