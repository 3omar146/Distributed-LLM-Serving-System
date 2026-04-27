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
    result: str = ""
    latency: float = 0.0
    status: str = "success"
    worker_id: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    completed_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)