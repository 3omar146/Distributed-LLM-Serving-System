import asyncio
import os
import statistics
import time

import httpx

from common.models import Request

# Where load test traffic goes (NGINX in front of masters)
MASTER_URL = os.getenv("MASTER_URL", "http://localhost:8000").strip()

# Direct URLs to each master, bypassing NGINX. Comma-separated.
# We need this because NGINX would round-robin our analytics requests
# to a random master, and we'd only see that one master's slice.
MASTER_DIRECT_URLS = [
    u.strip()
    for u in os.getenv(
        "MASTER_DIRECT_URLS",
        "http://master1:7000,http://master2:7000",
    ).split(",")
    if u.strip()
]

NUM_USERS = int(os.getenv("NUM_USERS", 1000))
RAMP_UP_RATE = int(os.getenv("RAMP_UP_RATE", 0))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 600))


class LoadTestResult:
    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.latencies = []

    def add_success(self, latency):
        self.total += 1
        self.success += 1
        self.latencies.append(latency)

    def add_failure(self):
        self.total += 1
        self.failed += 1


async def simulate_user(client, user_id, results):
    req = Request(
        id=user_id,
        query=f"User {user_id}: Who has the most goals in the champions league history?",
    )
    start = time.time()
    try:
        r = await client.post(
            f"{MASTER_URL}/handle",
            json=req.to_dict(),
            timeout=REQUEST_TIMEOUT,
        )
        data = r.json()
        latency = time.time() - start
        if data.get("status") == "success":
            results.add_success(latency)
            print(f"[Client] {user_id} SUCCESS in {latency:.2f}s: {data.get('result')}")
        else:
            results.add_failure()
            print(f"[Client] {user_id} FAILED: {data.get('error')}")
    except Exception as e:
        results.add_failure()
        print(f"[Client] {user_id} CRASHED: {e}")


async def fetch_one_master_analytics(client, url):
    """Fetch analytics from one master directly (bypassing NGINX)."""
    try:
        r = await client.get(f"{url}/analytics", timeout=30)
        return url, r.json()
    except Exception as e:
        print(f"[Client] Could not fetch analytics from {url}: {e}")
        return url, None


async def fetch_all_analytics(client):
    """Fetch analytics from every master in parallel."""
    results = await asyncio.gather(
        *[fetch_one_master_analytics(client, url) for url in MASTER_DIRECT_URLS]
    )
    return dict(results)   # {master_url: analytics_dict_or_None}


def aggregate_analytics(per_master):
    """Combine per-master analytics into a single global view.

    Note: total_requests across masters can exceed actual client request count
    because NGINX retries on 5xx will be counted on multiple masters. The
    client-side counter is the source of truth for unique client requests.
    """
    combined = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "requests_per_worker": {},
        "successes_per_worker": {},
        "failures_per_worker": {},
        "gpu_status_per_worker": {},
    }

    for master_url, data in per_master.items():
        if data is None:
            continue

        combined["total_requests"]      += data.get("total_requests", 0)
        combined["successful_requests"] += data.get("successful_requests", 0)
        combined["failed_requests"]     += data.get("failed_requests", 0)

        for worker, count in (data.get("requests_per_worker") or {}).items():
            combined["requests_per_worker"][worker] = \
                combined["requests_per_worker"].get(worker, 0) + count
        for worker, count in (data.get("successes_per_worker") or {}).items():
            combined["successes_per_worker"][worker] = \
                combined["successes_per_worker"].get(worker, 0) + count
        for worker, count in (data.get("failures_per_worker") or {}).items():
            combined["failures_per_worker"][worker] = \
                combined["failures_per_worker"].get(worker, 0) + count

        # GPU status: keep the freshest snapshot per worker.
        # If two masters both have a snapshot for worker1, we just
        # keep whichever one we saw last. This is fine because GPU
        # status is the same physical worker — both masters are
        # describing the same GPU.
        for worker, info in (data.get("gpu_status_per_worker") or {}).items():
            if info:
                combined["gpu_status_per_worker"][worker] = info

    return combined


def print_aggregated_report(per_master, combined):
    print("\n===== AGGREGATED MASTER ANALYTICS =====")
    print(f"Total received (sum across masters):  {combined['total_requests']}")
    print(f"Successful:                            {combined['successful_requests']}")
    print(f"Failed:                                {combined['failed_requests']}")

    if combined["total_requests"] > 0:
        rate = combined["successful_requests"] / combined["total_requests"] * 100
        print(f"Success rate:                          {rate:.2f}%")

    print("\n--- Per-master breakdown ---")
    print("(Note: sum > client total can occur due to NGINX retries on 5xx)")
    for url, data in per_master.items():
        if data is None:
            print(f"  {url}  UNREACHABLE")
            continue
        total = data.get("total_requests", 0)
        succ  = data.get("successful_requests", 0)
        fail  = data.get("failed_requests", 0)
        rate  = (succ / total * 100) if total > 0 else 0.0
        print(f"  {url}  total={total}  ok={succ}  fail={fail}  rate={rate:.1f}%")

    for master_url, data in per_master.items():
        if data is None:
            continue

        reqs_per_worker  = data.get("requests_per_worker") or {}
        succ_per_worker  = data.get("successes_per_worker") or {}
        fail_per_worker  = data.get("failures_per_worker") or {}
        gpu_per_worker   = data.get("gpu_status_per_worker") or {}

        if not reqs_per_worker and not gpu_per_worker:
            continue

        print(f"\n=== Workers under {master_url} ===")

        all_workers = sorted(
            set(reqs_per_worker) | set(gpu_per_worker)
        )

        print("\n  --- Requests per worker ---")
        for worker in all_workers:
            sent = reqs_per_worker.get(worker, 0)
            ok   = succ_per_worker.get(worker, 0)
            fail = fail_per_worker.get(worker, 0)
            print(f"    {worker}  attempts_sent={sent}  attempts_success={ok}  attempts_fail={fail}")

        if gpu_per_worker:
            print("\n  --- GPU metrics per worker ---")
            for worker in all_workers:
                info = gpu_per_worker.get(worker)
                if not info:
                    print(f"    {worker}  (no snapshot)")
                    continue
                device = info.get("device") or "?"
                model  = info.get("model") or "?"
                status = info.get("status") or "?"
                print(f"    {worker} status={status} device={device} model={model}")
                m = info.get("gpu_metrics") or {}
                if not m or not m.get("gpu_available"):
                    print("        gpu_available=False")
                    continue
                name  = m.get("gpu_name") or "?"
                util  = m.get("gpu_utilization_percent")
                mu    = m.get("gpu_memory_used_mb")
                mt    = m.get("gpu_memory_total_mb")
                mp    = m.get("gpu_memory_used_percent")
                temp  = m.get("gpu_temperature_c")
                power = m.get("gpu_power_watts")
                ta    = m.get("torch_memory_allocated_mb")
                tr    = m.get("torch_memory_reserved_mb")
                print(f"        {name}")
                print(f"        util={util}%  temp={temp}C  power={power}W")
                print(f"        mem={mu}/{mt} MB ({mp}%)")
                print(f"        torch alloc={ta} MB  reserved={tr} MB")

async def main():
    results = LoadTestResult()
    limits = httpx.Limits(
        max_connections=NUM_USERS + 50,
        max_keepalive_connections=NUM_USERS + 50,
    )
    print(f"load test: {NUM_USERS} users")
    print(f"Traffic target: {MASTER_URL}")
    print(f"Analytics sources: {MASTER_DIRECT_URLS}")

    start = time.time()
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for i in range(NUM_USERS):
            tasks.append(asyncio.create_task(simulate_user(client, i, results)))
            if RAMP_UP_RATE > 0:
                await asyncio.sleep(1 / RAMP_UP_RATE)
        await asyncio.gather(*tasks)
        total = time.time() - start

        # ===== Client-side report (source of truth for client request count) =====
        print("\n===== LOAD TEST REPORT (CLIENT SIDE) =====")
        print(f"Total client requests:  {results.total}")
        print(f"Success:                {results.success}")
        print(f"Failed:                 {results.failed}")
        print(f"Duration:               {total:.2f}s")
        if total > 0:
            print(f"Throughput:             {results.success / total:.2f} req/s")
        if results.latencies:
            print(f"Avg latency:            {statistics.mean(results.latencies):.3f}s")
            print(f"Median:                 {statistics.median(results.latencies):.3f}s")
            print(f"Min/Max:                {min(results.latencies):.3f}s / "
                  f"{max(results.latencies):.3f}s")
            if len(results.latencies) >= 20:
                s = sorted(results.latencies)
                print(f"p95:                    {s[int(len(s)*0.95)]:.3f}s")
                print(f"p99:                    {s[int(len(s)*0.99)]:.3f}s")

        # ===== Master-side analytics (aggregated across all masters) =====
        per_master = await fetch_all_analytics(client)
        combined = aggregate_analytics(per_master)
        print_aggregated_report(per_master, combined)


if __name__ == "__main__":
    asyncio.run(main())