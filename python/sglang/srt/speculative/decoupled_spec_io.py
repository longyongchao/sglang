"""IPC configuration for decoupled speculative decoding.

Minimal container consumed by PortArgs in server_args.py. The full decoupled
draft/verify engine mesh is not ported to this branch; only the config type is
needed so `--decoupled-spec-role null` (the default) works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DecoupledSpecIpcConfig:
    bind_endpoint: str
    connect_endpoints: Tuple[str, ...]
    rank: int
