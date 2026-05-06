"""
Async load generator for the "all 1000 at once" use case.

Why a separate file: 1000 OS threads each blocked on requests.post costs
~8 GB of stack memory and adds context-switch overhead. With httpx + asyncio,
1000 in-flight requests cost a few MB total because they share one event loop.

Run:
    pip install httpx
    python -m client.client_async
"""

import asyncio
import os
import statistics
import time

import httpx

from common.models import Request

MASTER_URL = os.getenv("MASTER_URL", "http://localhost:8000").strip()
NUM_USERS = int(os.getenv("NUM_USERS", 1000))
# 0 = fire all requests instantaneously (the "all at once" case).
# >0 = users per second ramp-up.
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


async def fetch_analytics(client):
    try:
        r = await client.get(f"{MASTER_URL}/analytics", timeout=30)
        return r.json()
    except Exception as e:
        print(f"[Client] Could not fetch analytics: {e}")
        return None


async def main():
    results = LoadTestResult()
    # Generous connection pool so we can actually have N requests in-flight.
    limits = httpx.Limits(
        max_connections=NUM_USERS + 50,
        max_keepalive_connections=NUM_USERS + 50,
    )
    print(f"Async load test: {NUM_USERS} users, ramp_rate={RAMP_UP_RATE}/s")

    start = time.time()
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for i in range(NUM_USERS):
            tasks.append(asyncio.create_task(simulate_user(client, i, results)))
            if RAMP_UP_RATE > 0:
                await asyncio.sleep(1 / RAMP_UP_RATE)
        await asyncio.gather(*tasks)
        total = time.time() - start

        print("\n===== ASYNC LOAD TEST REPORT =====")
        print(f"Total:        {results.total}")
        print(f"Success:      {results.success}")
        print(f"Failed:       {results.failed}")
        print(f"Duration:     {total:.2f}s")
        if total > 0:
            print(f"Throughput:   {results.success / total:.2f} req/s")
        if results.latencies:
            print(f"Avg latency:  {statistics.mean(results.latencies):.3f}s")
            print(f"Median:       {statistics.median(results.latencies):.3f}s")
            print(f"Min/Max:      {min(results.latencies):.3f}s / "
                  f"{max(results.latencies):.3f}s")
            if len(results.latencies) >= 20:
                s = sorted(results.latencies)
                print(f"p95:          {s[int(len(s)*0.95)]:.3f}s")
                print(f"p99:          {s[int(len(s)*0.99)]:.3f}s")

        analytics = await fetch_analytics(client)
        if analytics:
            print("\n--- Requests per worker ---")
            for url, count in analytics.get("requests_per_worker", {}).items():
                succ = analytics.get("successes_per_worker", {}).get(url, 0)
                fail = analytics.get("failures_per_worker", {}).get(url, 0)
                print(f"  {url}  sent={count}  ok={succ}  fail={fail}")

            gpu_status = analytics.get("gpu_status_per_worker", {}) or {}
            if gpu_status:
                print("\n--- GPU metrics per worker ---")
                for url, info in gpu_status.items():
                    if not info:
                        print(f"  {url}  (no snapshot)")
                        continue
                    label = info.get("vm_label") or "?"
                    device = info.get("device") or "?"
                    model = info.get("model") or "?"
                    status = info.get("status") or "?"
                    print(f"  {url}  [{label}] status={status} device={device} model={model}")
                    m = info.get("gpu_metrics") or {}
                    if not m or not m.get("gpu_available"):
                        print("      gpu_available=False")
                        continue
                    name = m.get("gpu_name") or "?"
                    util = m.get("gpu_utilization_percent")
                    mem_used = m.get("gpu_memory_used_mb")
                    mem_total = m.get("gpu_memory_total_mb")
                    mem_pct = m.get("gpu_memory_used_percent")
                    temp = m.get("gpu_temperature_c")
                    power = m.get("gpu_power_watts")
                    torch_alloc = m.get("torch_memory_allocated_mb")
                    torch_res = m.get("torch_memory_reserved_mb")
                    print(f"      {name}")
                    print(f"      util={util}%  temp={temp}C  power={power}W")
                    print(f"      mem={mem_used}/{mem_total} MB ({mem_pct}%)")
                    print(f"      torch alloc={torch_alloc} MB  reserved={torch_res} MB")


if __name__ == "__main__":
    asyncio.run(main())