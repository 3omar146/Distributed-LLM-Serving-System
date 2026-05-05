from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import time

@dataclass
class Request:
    id: int
    query: str
    created_at: float = field(default_factory=time.time)
    #metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

@dataclass
class Response:
    id: int
    result: str = None
    latency: float = None
    status: str = "success"
    error: str = None
    worker_id: int = None
    created_at: float = None
    gpu_metrics: dict = None
    gpu_latency: float = None
    input_tokens: int = None
    output_tokens: int = None
    vm_label: str = None

    def to_dict(self):
        return asdict(self)