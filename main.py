# main.py

from Master.master import MasterScheduler
from LoadBalancer.load_balancer import LoadBalancer
from Worker.worker import Worker
from Client.load_generator import run_load_test


def main():
    # 🔹 Step 1: Create workers (simulate GPUs)
    workers = [Worker(worker_id=i, max_concurrent=20, fail_rate=0.1) for i in range(4)]

    # 🔹 Step 2: Create load balancer
    lb = LoadBalancer(workers)

    # 🔹 Step 3: Create scheduler
    scheduler = MasterScheduler(load_balancer=lb, max_retries=4)

    # 🔹 Step 4: Run load test
    run_load_test(scheduler, num_users=100)  # start with 100, then increase


main()