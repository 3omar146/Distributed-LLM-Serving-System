from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import threading
import os

#uvicorn gpu_server:app --host 0.0.0.0 --port 8080

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VM_LABEL = os.getenv("VM_LABEL", "gpu-vm")

print(f"[GPU Server] VM: {VM_LABEL} | Device: {DEVICE} | Model: {MODEL_NAME}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Causal-LM batched generation requires LEFT padding so that newly generated
# tokens line up after the prompt for every sequence in the batch.
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# torch_dtype= (NOT dtype=) is the correct from_pretrained parameter.
# Using the wrong keyword causes it to be silently swallowed by **kwargs and
# the model loads in its default dtype (bfloat16 for Qwen), ignoring your choice.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)
model.to(DEVICE)
model.eval()

actual_dtype = next(model.parameters()).dtype
print(f"[GPU Server] Model loaded | dtype={actual_dtype} | device={DEVICE}", flush=True)

# Serialize at the BATCH level (not per-request).
# Each batch fully owns the GPU while it runs.
generate_lock = threading.Lock()

# Stores the GPU metrics sampled inside the last completed batch.
# /health returns this so the dashboard shows real inference-time utilization
# rather than idle readings taken between batches.
last_batch_metrics: dict = {}
last_batch_lock = threading.Lock()


class GenerateRequest(BaseModel):
    # Accept a single string OR a list of strings.
    # List path = true GPU batching; scalar path = backward compat.
    prompt: Union[str, List[str]]
    max_new_tokens: int = 150


def get_gpu_metrics() -> dict:
    """
    Sample live GPU state via NVML + PyTorch memory APIs.
    Call this INSIDE generate_lock (after synchronize) to capture
    utilization while the GPU is still warm from inference.
    Calling it outside the lock risks reading 0% because NVML's
    utilization counter is a ~1s rolling average that decays quickly.
    """
    metrics = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_utilization_percent": None,
        "gpu_memory_used_mb": None,
        "gpu_memory_total_mb": None,
        "gpu_memory_used_percent": None,
        "gpu_temperature_c": None,
        "gpu_power_watts": None,
        "torch_memory_allocated_mb": None,
        "torch_memory_reserved_mb": None,
        "torch_peak_memory_mb": None,
    }

    if DEVICE != "cuda":
        return metrics

    metrics["gpu_name"] = torch.cuda.get_device_name(0)
    metrics["torch_memory_allocated_mb"] = round(
        torch.cuda.memory_allocated(0) / 1024 / 1024, 2
    )
    metrics["torch_memory_reserved_mb"] = round(
        torch.cuda.memory_reserved(0) / 1024 / 1024, 2
    )
    # Peak is a high-water mark preserved until the next reset_peak_memory_stats().
    # It is valid to read immediately after synchronize(), still inside the lock.
    metrics["torch_peak_memory_mb"] = round(
        torch.cuda.max_memory_allocated(0) / 1024 / 1024, 2
    )

    if NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power = None

        metrics["gpu_utilization_percent"] = util.gpu
        metrics["gpu_memory_used_mb"] = round(mem.used / 1024 / 1024, 2)
        metrics["gpu_memory_total_mb"] = round(mem.total / 1024 / 1024, 2)
        metrics["gpu_memory_used_percent"] = round(mem.used / mem.total * 100, 2)
        metrics["gpu_temperature_c"] = temp
        metrics["gpu_power_watts"] = round(power, 2) if power is not None else None

    return metrics


@app.get("/health")
def health():
    """
    Returns the GPU metrics from the most recently completed batch
    (last_batch_metrics), falling back to a live reading if no batch
    has run yet. This means the dashboard always shows what the GPU
    looked like during real inference, not during idle polling.
    """
    with last_batch_lock:
        snapshot = dict(last_batch_metrics)

    return {
        "ok": True,
        "vm_label": VM_LABEL,
        "model": MODEL_NAME,
        "device": DEVICE,
        "actual_dtype": str(actual_dtype),
        "metrics": snapshot if snapshot else get_gpu_metrics(),
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    is_scalar = isinstance(req.prompt, str)
    prompts = [req.prompt] if is_scalar else list(req.prompt)

    if len(prompts) == 0:
        return {"ok": False, "error": "empty prompt list"}

    start_time = time.time()
    live_metrics: dict = {}

    with generate_lock:
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats(0)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if DEVICE == "cuda":
            # Block until all CUDA kernels finish before reading any metrics.
            torch.cuda.synchronize()

        # Sample GPU state HERE — inside the lock, after synchronize.
        # The GPU is still warm so NVML's utilization counter reflects
        # the actual inference workload rather than idle state.
        live_metrics = get_gpu_metrics()

        # Persist to last_batch_metrics so /health can serve it.
        with last_batch_lock:
            last_batch_metrics.clear()
            last_batch_metrics.update(live_metrics)

    # Everything below is pure Python / CPU — fine to do outside the lock.
    end_time = time.time()
    input_len = inputs["input_ids"].shape[-1]
    batch_latency = round(end_time - start_time, 3)

    results = []
    for i in range(len(prompts)):
        gen_ids = output_ids[i][input_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({
            "ok": True,
            "vm_label": VM_LABEL,
            "answer": answer,
            "model": MODEL_NAME,
            "device": DEVICE,
            "input_tokens": int(input_len),
            "output_tokens": int(gen_ids.shape[-1]),
            "total_latency_seconds": batch_latency,
            "batch_size": len(prompts),
            "metrics": live_metrics,
        })

    # Preserve request shape: scalar in → scalar out, list in → list out.
    return results[0] if is_scalar else results