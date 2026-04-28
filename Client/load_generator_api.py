import time
import threading
import statistics
import requests

from Common.models import Request

MASTER_URL = "http://localhost:7000"


class LoadTestResult:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies = []
        self.lock = threading.Lock()

    def add_success(self, latency):
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.latencies.append(latency)

    def add_failure(self):
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1


def simulate_user(user_id, results):
    req = Request(
        id=user_id,
        query=f"User {user_id}: Explain distributed LLM load balancing"
    )

    start = time.time()

    try:
        res = requests.post(
            f"{MASTER_URL}/handle",
            json=req.to_dict(),
            timeout=120
        ).json()

        latency = time.time() - start

        if res.get("status") == "success":
            results.add_success(latency)
        else:
            # 🔹 ADD THIS PRINT STATEMENT
            error_msg = res.get("error", "No error message provided")
            print(f"[Client] Request {user_id} FAILED | Worker Error: {error_msg}")
            results.add_failure()

    except Exception as e:
        # 🔹 ADD THIS PRINT STATEMENT
        print(f"[Client] Request {user_id} CRASHED | Network Exception: {e}")
        results.add_failure()


def run_load_test(num_users=1000):
    results = LoadTestResult()
    threads = []

    start = time.time()

    for i in range(num_users):
        t = threading.Thread(target=simulate_user, args=(i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total = time.time() - start

    print("\n===== REPORT =====")
    print(f"Total: {results.total_requests}")
    print(f"Success: {results.successful_requests}")
    print(f"Failed: {results.failed_requests}")
    print(f"Throughput: {results.successful_requests / total:.2f} req/s")

    if results.latencies:
        print(f"Avg latency: {statistics.mean(results.latencies):.3f}s")