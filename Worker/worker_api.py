from fastapi import FastAPI
import time
import threading
import os
import random  # 🔥 added

from Common.models import Response
from RAG.model import retriever, llm, prompt

app = FastAPI()




class Worker:
    def __init__(self, worker_id, max_concurrent=2, fail_rate=0.1):
        self.id = worker_id
        self.lock = threading.Lock()
        self.active_requests = 0
        self.sem = threading.Semaphore(max_concurrent)

        # 🔥 failure rate
        self.fail_rate = fail_rate

    def process(self, request: dict):
        start = time.time()

        if not self.sem.acquire(blocking=False):
            raise Exception("Capacity full")

        with self.lock:
            self.active_requests += 1

        try:
            # 🔥 simulate random failure
            if random.random() < self.fail_rate:
                raise Exception("Simulated worker failure")

            query = request["query"]
            req_id = request["id"]
            time.sleep(0.5) 
            result = f"Simulated response for: {query}"

            latency = time.time() - start

            return Response(
                id=req_id,
                result=result,
                latency=latency,
                status="success",
                worker_id=self.id,
                created_at=request.get("created_at")
            ).to_dict()

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


# 🔥 read from environment variables
worker = Worker(
    worker_id=int(os.getenv("WORKER_ID", 1)),
    max_concurrent=int(os.getenv("MAX_CONCURRENT", 2)),
    fail_rate=float(os.getenv("FAIL_RATE", 0.1))
)


@app.post("/process")
def process(request: dict):
    return worker.process(request)


# 🔥 read from environment variables
worker = Worker(
    worker_id=int(os.getenv("WORKER_ID", 1)),
    max_concurrent=int(os.getenv("MAX_CONCURRENT", 2)),
    fail_rate=float(os.getenv("FAIL_RATE", 0.1))
)


@app.post("/process")
def process(request: dict):
    return worker.process(request)