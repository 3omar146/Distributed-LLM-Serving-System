@echo off

echo Starting Workers...
start cmd /k "set WORKER_ID=1 && uvicorn Worker.worker_api:app --port 8001"
start cmd /k "set WORKER_ID=2 && uvicorn Worker.worker_api:app --port 8002"
start cmd /k "set WORKER_ID=3 && uvicorn Worker.worker_api:app --port 8003"

timeout /t 2 >nul

echo Starting Load Balancer...
start cmd /k "uvicorn LoadBalancer.load_balancer_api:app --port 9000"

timeout /t 2 >nul

echo Starting Master...
start cmd /k "uvicorn Master.master_api:app --port 7000"

timeout /t 3 >nul

echo Running Client...
python main.py

pause