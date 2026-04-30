@echo off

:: create logs folder
mkdir logs 2>nul

:: Load API keys and config from .env
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if "%%A"=="GROQ_API_KEY" set "GROQ_API_KEY=%%~B"
    if "%%A"=="HF_TOKEN"     set "HF_TOKEN=%%~B"
    if "%%A"=="NUM_USERS"    set "NUM_USERS=%%~B"
    if "%%A"=="RAMP_UP_RATE" set "RAMP_UP_RATE=%%~B"
    if "%%A"=="MAX_RETRIES"  set "MAX_RETRIES=%%~B"
    if "%%A"=="FAIL_RATE"    set "FAIL_RATE=%%~B"
)

set FAIL_RATE=0.1
set MAX_CONCURRENT=2

echo Starting Worker 1 on port 8001...
start "Worker 1" cmd /c "set PYTHONPATH=%CD%&&set PYTHONUNBUFFERED=1&&cd workers&&set WORKER_ID=1&&set GROQ_API_KEY=%GROQ_API_KEY%&&set HF_TOKEN=%HF_TOKEN%&&set FAIL_RATE=%FAIL_RATE%&&set MAX_CONCURRENT=%MAX_CONCURRENT%&&uvicorn app:app --host 0.0.0.0 --port 8001 > ../logs/worker1.log 2>&1"

echo Starting Worker 2 on port 8002...
start "Worker 2" cmd /c "set PYTHONPATH=%CD%&&set PYTHONUNBUFFERED=1&&cd workers&&set WORKER_ID=2&&set GROQ_API_KEY=%GROQ_API_KEY%&&set HF_TOKEN=%HF_TOKEN%&&set FAIL_RATE=%FAIL_RATE%&&set MAX_CONCURRENT=%MAX_CONCURRENT%&&uvicorn app:app --host 0.0.0.0 --port 8002 > ../logs/worker2.log 2>&1"

echo Starting Worker 3 on port 8003...
start "Worker 3" cmd /c "set PYTHONPATH=%CD%&&set PYTHONUNBUFFERED=1&&cd workers&&set WORKER_ID=3&&set GROQ_API_KEY=%GROQ_API_KEY%&&set HF_TOKEN=%HF_TOKEN%&&set FAIL_RATE=%FAIL_RATE%&&set MAX_CONCURRENT=%MAX_CONCURRENT%&&uvicorn app:app --host 0.0.0.0 --port 8003 > ../logs/worker3.log 2>&1"

echo Waiting for workers to initialize (RAG setup takes ~30s)...
timeout /t 20 /nobreak

echo Starting Master on port 8000...
start "Master" cmd /c "cd master&&set WORKER_URLS=http://localhost:8001,http://localhost:8002,http://localhost:8003&&set MAX_RETRIES=%MAX_RETRIES%&&uvicorn app:app --host 0.0.0.0 --port 8000 > ../logs/master.log 2>&1"

echo Waiting for master to start...
timeout /t 3 /nobreak

echo Starting Client...
start "Client" cmd /c "set PYTHONPATH=%CD%&&set MASTER_URL=http://localhost:8000&&set NUM_USERS=%NUM_USERS%&&set RAMP_UP_RATE=%RAMP_UP_RATE%&&python client\load_generator.py > logs\client.log 2>&1"

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
