@echo off

echo Starting Workers...

start cmd /k "set WORKER_ID=1 && uvicorn Worker.worker_api:app --port 8001 > logs\worker1.log 2>&1"
start cmd /k "set WORKER_ID=2 && uvicorn Worker.worker_api:app --port 8002 > logs\worker2.log 2>&1"
start cmd /k "set WORKER_ID=3 && uvicorn Worker.worker_api:app --port 8003 > logs\worker3.log 2>&1"

timeout /t 2 >nul

echo Starting Load Balancer...
start cmd /k "uvicorn LoadBalancer.load_balancer_api:app --port 9000 > logs\lb.log 2>&1"

timeout /t 2 >nul

echo Starting Master...
start cmd /k "uvicorn Master.master_api:app --port 7000 > logs\master.log 2>&1"

timeout /t 3 >nul

echo Running Client...
python main.py > logs\client.log 2>&1

pause