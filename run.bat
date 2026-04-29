@echo off

:: create logs folder
mkdir logs 2>nul

:: set environment variables
set GROQ_API_KEY=your_key_here
set FAIL_RATE=0.1
set MAX_CONCURRENT=2

echo Starting Worker 1 on port 8001...
start "Worker 1" cmd /c "set PYTHONPATH=%CD% && cd workers && set WORKER_ID=1 && set GROQ_API_KEY=%GROQ_API_KEY% && set FAIL_RATE=%FAIL_RATE% && set MAX_CONCURRENT=%MAX_CONCURRENT% && uvicorn app:app --host 0.0.0.0 --port 8001 > ../logs/worker1.log 2>&1"

echo Starting Worker 2 on port 8002...
start "Worker 2" cmd /c "set PYTHONPATH=%CD% && cd workers && set WORKER_ID=2 && set GROQ_API_KEY=%GROQ_API_KEY% && set FAIL_RATE=%FAIL_RATE% && set MAX_CONCURRENT=%MAX_CONCURRENT% && uvicorn app:app --host 0.0.0.0 --port 8002 > ../logs/worker2.log 2>&1"

echo Starting Worker 3 on port 8003...
start "Worker 3" cmd /c "set PYTHONPATH=%CD% && cd workers && set WORKER_ID=3 && set GROQ_API_KEY=%GROQ_API_KEY% && set FAIL_RATE=%FAIL_RATE% && set MAX_CONCURRENT=%MAX_CONCURRENT% && uvicorn app:app --host 0.0.0.0 --port 8003 > ../logs/worker3.log 2>&1"

echo Waiting for workers to start...
timeout /t 5 /nobreak

echo Starting Master on port 8000...
start "Master" cmd /c "cd master && set WORKER_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003 && set MAX_RETRIES=3 && uvicorn app:app --host 0.0.0.0 --port 8000 > ../logs/master.log 2>&1"

echo Waiting for master to start...
timeout /t 3 /nobreak

echo Starting Client...
start "Client" cmd /c "set PYTHONPATH=%CD% && set MASTER_URL=http://localhost:8000 && set NUM_USERS=20 && set RAMP_UP_RATE=5 && python client\load_generator.py > logs\client.log 2>&1"

echo.
echo ===================================
echo All services started!
echo Logs are being saved to ./logs/
echo.
echo   Master:   logs/master.log
echo   Worker 1: logs/worker1.log
echo   Worker 2: logs/worker2.log
echo   Worker 3: logs/worker3.log
echo   Client:   logs/client.log
echo.
echo To watch logs in real time run:
echo   tail_logs.bat
echo ===================================
pause