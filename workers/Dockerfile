FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY Common/ ./Common/

COPY workers/requirements.txt .

# install CPU-only PyTorch first — prevents CUDA/NVIDIA downloads
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# then install everything else
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY workers/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]