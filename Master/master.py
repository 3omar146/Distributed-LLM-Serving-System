import time


class MasterScheduler:
    def __init__(self, load_balancer, max_retries=2):
        self.lb = load_balancer
        self.max_retries = max_retries

    def handle_request(self, request):
        attempt = 0

        while attempt <= self.max_retries:
            start_time = time.time()

            try:
                response = self.lb.dispatch(request)

                total_latency = time.time() - start_time

                print(f"[Scheduler] Request {request.id} handled in {total_latency:.3f}s")

                return response

            except Exception as e:
                attempt += 1

                print(f"[Scheduler] Attempt {attempt} failed for request {request.id}: {e}")

                # 🔥 if all retries exhausted → fail
                if attempt > self.max_retries:
                    print(f"[Scheduler] Request {request.id} FAILED after {self.max_retries} retries")

                    return {
                        "request_id": request.id,
                        "status": "failed",
                        "error": str(e)
                    }

                # 🔹 small backoff before retry
                time.sleep(0.05 * attempt)