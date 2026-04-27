
import time
import threading

from rag.model import retriever, llm, prompt

class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.active_requests = 0
        self.lock = threading.Lock()

    def process(self, request):
        start_time = time.time()

        with self.lock:
            self.active_requests += 1

        try:
            if isinstance(request, dict):
                query = request.get("query", "")
                req_id = request.get("id")
            else:
                query = request.query
                req_id = request.id

            context = self.retrieve_context(query)

            result = self.run_llm(query, context)

            latency = time.time() - start_time

            response = {
                "id": req_id,
                "worker_id": self.id,
                "result": result,
                "latency": latency,
                "status": "success"
            }

            print(f"[Worker {self.id}] Done request {req_id} in {latency:.3f}s")
            return response

        except Exception as e:
            return {
                "id": getattr(request, 'id', None) or (request.get('id') if isinstance(request, dict) else "Unknown"),
                "worker_id": self.id,
                "result": "",
                "latency": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            }

        finally:
            with self.lock:
                self.active_requests -= 1

   
    def retrieve_context(self, query):
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def run_llm(self, query, context):
        formatted_prompt = prompt.format(context=context, question=query)
        response = llm.invoke(formatted_prompt)
        return response.content