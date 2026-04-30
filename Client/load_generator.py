import time
import threading
import statistics
import requests
import os

from common.models import Request

MASTER_URL = os.getenv("MASTER_URL", "http://localhost:8000").strip()
RAMP_UP_RATE = int(os.getenv("RAMP_UP_RATE", 10))


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
        query=f"User {user_id}: Who has the most goal in the champions league history?",
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
            print(f"[Client] Request {user_id} OK | "
                  f"Latency: {latency:.3f}s | "
                  f"Worker: {res.get('worker_id')}|"
                  f"Query: {req.query} | "
                  f" Result: {res.get('result')}")
        else:
            error_msg = res.get("error", "No error message")
            print(f"[Client] Request {user_id} FAILED | Error: {error_msg}")
            results.add_failure()

    except Exception as e:
        print(f"[Client] Request {user_id} CRASHED | {e}")
        results.add_failure()


def fetch_master_analytics():
    try:
        res = requests.get(f"{MASTER_URL}/analytics", timeout=5)
        return res.json()
    except Exception as e:
        print(f"[Client] Could not fetch master analytics: {e}")
        return None


def run_load_test(num_users=1000):
    results = LoadTestResult()
    threads = []

    start = time.time()

    for i in range(num_users):
        t = threading.Thread(target=simulate_user, args=(i, results))
        threads.append(t)
        t.start()
        time.sleep(1 / RAMP_UP_RATE)

    for t in threads:
        t.join()

    total = time.time() - start

    # client side report
    print("\n===== LOAD TEST REPORT (CLIENT SIDE) =====")
    print(f"Total requests:      {results.total_requests}")
    print(f"Successful:          {results.successful_requests}")
    print(f"Failed:              {results.failed_requests}")
    print(f"Duration:            {total:.2f}s")
    print(f"Throughput:          {results.successful_requests / total:.2f} req/s")

    if results.latencies:
        print(f"Avg latency:         {statistics.mean(results.latencies):.3f}s")
        print(f"Min latency:         {min(results.latencies):.3f}s")
        print(f"Max latency:         {max(results.latencies):.3f}s")
        print(f"Median latency:      {statistics.median(results.latencies):.3f}s")

    # master side analytics
    analytics = fetch_master_analytics()
    if analytics:
        print("\n===== MASTER ANALYTICS (SERVER SIDE) =====")
        print(f"Total received:      {analytics.get('total_requests')}")
        print(f"Successful:          {analytics.get('successful_requests')}")
        print(f"Failed:              {analytics.get('failed_requests')}")
        print(f"Success rate:        {analytics.get('success_rate')}")

        print("\n--- Requests per worker ---")
        for worker, count in analytics.get("requests_per_worker", {}).items():
            successes = analytics.get("successes_per_worker", {}).get(worker, 0)
            failures  = analytics.get("failures_per_worker", {}).get(worker, 0)
            print(f"  {worker}")
            print(f"    Sent:      {count}")
            print(f"    Success:   {successes}")
            print(f"    Failed:    {failures}")


if __name__ == "__main__":
    num_users = int(os.getenv("NUM_USERS", 50))
    print(f"Starting load test with {num_users} users "
          f"at {RAMP_UP_RATE} users/sec ramp-up...")
    run_load_test(num_users)