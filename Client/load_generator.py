# client/load_generator.py

import time
import threading
import statistics
from Common.models import Request


class LoadTestResult:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies = []
        self.errors = []
        self.lock = threading.Lock()

    def add_success(self, latency):
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.latencies.append(latency)

    def add_failure(self, error):
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.errors.append(str(error))


def simulate_user(scheduler, user_id, results):
    """
    Simulates one user sending one AI request.
    """

    request = Request(
        id=user_id,
        query=f"User {user_id}: Explain distributed LLM load balancing",
        
        # metadata={
        #     "source": "client_load_generator",
        #     "user_id": user_id
        # }
    )

    start_time = time.time()

    try:
        response = scheduler.handle_request(request)

        end_time = time.time()
        latency = end_time - start_time

        # Supports both dict response and Response dataclass response
        if isinstance(response, dict):
            response_id = response.get("request_id", user_id)
            response_status = response.get("status", "success")
            response_error = response.get("error", None)
        else:
            response_id = response.id
            response_status = response.status
            response_error = response.error

        if response_status == "success":
            results.add_success(latency)

            print(
                f"[Client] Request {response_id} completed "
                f"| Latency: {latency:.3f}s"
            )
        else:
            results.add_failure(response_error or "Unknown error")

            print(
                f"[Client] Request {response_id} failed "
                f"| Error: {response_error}"
            )

    except Exception as e:
        results.add_failure(e)

        print(
            f"[Client] Request {user_id} failed "
            f"| Error: {e}"
        )


def run_load_test(scheduler, num_users=1000):
    """
    Runs a load test using many concurrent users.
    """

    print("=" * 60)
    print(f"Starting load test with {num_users} concurrent users")
    print("=" * 60)

    results = LoadTestResult()
    threads = []

    test_start_time = time.time()

    for user_id in range(num_users):
        thread = threading.Thread(
            target=simulate_user,
            args=(scheduler, user_id, results)
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    test_end_time = time.time()
    total_time = test_end_time - test_start_time

    print_final_report(results, total_time)

    return results


def print_final_report(results, total_time):
    """
    Prints final load test metrics.
    """

    print("\n" + "=" * 60)
    print("LOAD TEST REPORT")
    print("=" * 60)

    print(f"Total requests:       {results.total_requests}")
    print(f"Successful requests:  {results.successful_requests}")
    print(f"Failed requests:      {results.failed_requests}")
    print(f"Total test time:      {total_time:.3f}s")

    if total_time > 0:
        throughput = results.successful_requests / total_time
    else:
        throughput = 0

    print(f"Throughput:           {throughput:.2f} requests/second")

    if results.latencies:
        print(f"Average latency:      {statistics.mean(results.latencies):.3f}s")
        print(f"Minimum latency:      {min(results.latencies):.3f}s")
        print(f"Maximum latency:      {max(results.latencies):.3f}s")

        if len(results.latencies) > 1:
            print(f"Latency std dev:      {statistics.stdev(results.latencies):.3f}s")

    if results.errors:
        print("\nErrors:")
        for error in results.errors[:10]:
            print(f"- {error}")

        if len(results.errors) > 10:
            print(f"... and {len(results.errors) - 10} more errors")

    print("=" * 60)