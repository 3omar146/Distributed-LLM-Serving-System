# common/models.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


@dataclass
class Request:
    id: int
    query: str
    created_at: float = field(default_factory=time.time)
    #metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    id: int
    result: str
    latency: float
    status: str = "success"          # success / failed
    worker_id: Optional[int] = None  # which GPU worker processed it
    error: Optional[str] = None      # failure reason if failed
   # metadata: Dict[str, Any] = field(default_factory=dict)