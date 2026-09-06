# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-side spec-training capture sink (SpecForge DataFlow transport).

Under ``--enable-spec-capture``, a request's ``spec_capture`` dict tells this
sink to write the prefill's captured tensors straight into a Mooncake store
(one hard-pinned object per tensor at ``{store_id}/{sample_id}/g{gen}/{name}``,
raw bytes — shape/dtype travel on the returned spec). Feature tensors never
touch the response path; ``meta_info["spec_capture"]`` returns only keys +
shapes/dtypes. Strategy naming is the client's (the ``features`` mapping); the
server knows only generic artifacts. Self-contained: the scheduler hooks are
one-liners, every capture decision lives here.

Request schema::

    {"store_id", "sample_id", "gen",           # key namespace / id / staleness
     "features": {"aux": <name>, "last_hidden": <name>},   # artifact -> feature
     "passthrough": [{"name", "data", "shape", "dtype"}]}  # client tensors verbatim

Response (``meta_info["spec_capture"]``): ``{"sample_id", "store_id", "gen",
"aux_layer_ids", "features": {name: {"shape", "dtype"}}}``.

Mooncake connection uses the standard ``MOONCAKE_*`` env vars (see
``MooncakeFeatureStore``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# torch dtype -> the FeatureSpec dtype string SpecForge's zero-copy get() maps
# back to a torch dtype. Keep in sync with MooncakeFeatureStore._TORCH_DTYPES.
_DTYPE_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_DTYPE = {v: k for k, v in _DTYPE_STR.items()}

_ARTIFACT_AUX = "aux"
_ARTIFACT_LAST_HIDDEN = "last_hidden"


class SpecCaptureSink:
    """Writes captured per-request tensors into Mooncake in SpecForge layout."""

    def __init__(self, aux_layer_ids: Optional[List[int]] = None) -> None:
        self.aux_layer_ids = list(aux_layer_ids) if aux_layer_ids else None
        self._store = None
        self._put_config = None
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------
    def _connect(self):
        if self._store is not None:
            return self._store
        with self._lock:
            if self._store is not None:
                return self._store
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig

            store = MooncakeDistributedStore()
            rc = store.setup(
                local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
                metadata_server=os.environ.get(
                    "MOONCAKE_METADATA_SERVER", "http://localhost:8080/metadata"
                ),
                global_segment_size=int(
                    os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)
                ),
                local_buffer_size=int(
                    os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)
                ),
                protocol=os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
                rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
                master_server_addr=os.environ.get(
                    "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
                ),
            )
            if rc is not None and int(rc) != 0:
                raise RuntimeError(f"spec-capture mooncake setup failed (status {rc})")
            # Hard-pin every object: SpecForge (not Mooncake's LRU) is the
            # lifetime authority — a committed feature must never be evicted
            # before the trainer consumes it.
            cfg = ReplicateConfig()
            cfg.replica_num = 1
            cfg.with_hard_pin = True
            self._put_config = cfg
            self._store = store
            logger.info("spec-capture mooncake sink connected")
            return store

    # -- key/put primitives (pinned to MooncakeFeatureStore's layout) --------
    @staticmethod
    def _tkey(store_id: str, sample_id: str, gen: int, name: str) -> str:
        return f"{store_id}/{sample_id}/g{gen}/{name}"

    def _put_tensor(self, key: str, t: torch.Tensor) -> None:
        store = self._connect()
        t = t.detach().to("cpu").contiguous()
        nbytes = t.element_size() * t.numel()
        try:
            store.register_buffer(t.data_ptr(), nbytes)
        except Exception:
            pass  # some builds auto-register
        try:
            rc = store.put_from(key, t.data_ptr(), nbytes, self._put_config)
        finally:
            try:
                store.unregister_buffer(t.data_ptr())
            except Exception:
                pass
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"spec-capture put_from failed (status {rc}) for {key}")

    def _remove_quiet(self, key: str) -> None:
        try:
            self._connect().remove(key)
        except Exception:
            pass

    # -- the one entry point --------------------------------------------------
    def put_sample(
        self,
        spec: Dict[str, Any],
        *,
        aux: Optional[torch.Tensor],
        last_hidden: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Write one sample's artifacts; return the meta_info result dict.

        ``aux``/``last_hidden`` are the per-request (L, W) captured tensors,
        stored with a leading batch dim of 1. On any failure the keys already
        written are best-effort removed (no partial sample is consumable).
        """
        store_id = str(spec["store_id"])
        sample_id = str(spec["sample_id"])
        gen = int(spec.get("gen", 1))
        features: Dict[str, str] = dict(spec.get("features") or {})

        written: List[str] = []
        result_feats: Dict[str, Dict[str, Any]] = {}

        def _write(name: str, t: torch.Tensor) -> None:
            key = self._tkey(store_id, sample_id, gen, name)
            self._put_tensor(key, t)
            written.append(key)
            result_feats[name] = {
                "shape": list(t.shape),
                "dtype": _DTYPE_STR.get(t.dtype, str(t.dtype).replace("torch.", "")),
            }

        try:
            aux_name = features.get(_ARTIFACT_AUX)
            if aux_name is not None:
                if aux is None:
                    raise RuntimeError(
                        "spec_capture requested 'aux' but no aux hidden states were "
                        "captured — launch the server with --enable-spec-capture "
                        "(and optionally --spec-capture-aux-layer-ids)"
                    )
                _write(aux_name, aux.unsqueeze(0))
            lh_name = features.get(_ARTIFACT_LAST_HIDDEN)
            if lh_name is not None:
                if last_hidden is None:
                    raise RuntimeError(
                        "spec_capture requested 'last_hidden' but the logits "
                        "processor did not return it (is aux capture enabled?)"
                    )
                _write(lh_name, last_hidden.unsqueeze(0))
            for item in spec.get("passthrough") or []:
                dtype = _STR_DTYPE.get(str(item.get("dtype", "int64")))
                if dtype is None:
                    raise RuntimeError(
                        f"spec_capture passthrough {item.get('name')!r}: "
                        f"unsupported dtype {item.get('dtype')!r}"
                    )
                t = torch.tensor(item["data"], dtype=dtype).reshape(
                    [int(d) for d in item["shape"]]
                )
                _write(str(item["name"]), t)
        except Exception:
            for key in written:
                self._remove_quiet(key)
            raise

        return {
            "sample_id": sample_id,
            "store_id": store_id,
            "gen": gen,
            "aux_layer_ids": self.aux_layer_ids,
            "features": result_feats,
        }


_SINK: Optional[SpecCaptureSink] = None


def maybe_init_sink(server_args) -> None:
    """Called from Scheduler init on the writer rank when spec capture is on.

    Connection to Mooncake is lazy (first put), so a capture-enabled server
    without a reachable Mooncake master still boots and serves normal traffic.
    """
    global _SINK
    if getattr(server_args, "enable_spec_capture", False) and _SINK is None:
        _SINK = SpecCaptureSink(
            aux_layer_ids=getattr(server_args, "spec_capture_aux_layer_ids", None)
        )


def get_sink() -> Optional[SpecCaptureSink]:
    return _SINK
