# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC message definitions for PLE CPU offload."""

from dataclasses import dataclass

import msgspec
import torch

import vllm.envs as envs

# ---------------------------------------------------------------------------
# Startup rendezvous barrier timeout
# ---------------------------------------------------------------------------
# The GPU worker <-> offload registration handshake (connector.py sends a
# registration, worker.py accepts it and returns an ACK) is a STARTUP-ONLY
# rendezvous. Bounding both ends converts the TP=1 warmup deadlock
# (vllm-project/vllm#53960) -- where a lost registration or a hung sibling
# leaves one side waiting forever -- into a NAMED TimeoutError. The bound is
# deliberately WELL ABOVE a legitimate cold boot (main weight load alone is
# ~510 s and a full boot ~13-14 min here), so it never pre-empts a slow-but-
# healthy start; it only fires on a true hang. It is derived from the existing
# ready-timeout knob so raising that knob raises this bound too.


def barrier_timeout_s() -> float:
    """Return the startup registration-rendezvous timeout in seconds."""
    return max(2.0 * float(envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT), 1200.0)


# ---------------------------------------------------------------------------
# IPC message dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PleOffloadRegistration:
    """Sent once from each GPU worker during offload setup."""

    worker_id: int
    tp_rank: int
    dp_rank: int
    # CUDA tensors are serialized through PyTorch CUDA IPC.
    gpu_output_buffers: dict[str, torch.Tensor]
    sem_flag_tensors: dict[str, torch.Tensor]
    # CPU tensors are allocated in shared memory and registered once.
    input_ids_buf: torch.Tensor
    query_start_loc_buf: torch.Tensor
    ngram_context_buf: torch.Tensor | None
    # ZMQ endpoint the offload worker PUSHes this worker's registration ACK to.
    # The GPU worker binds it before sending and blocks on the ACK as the
    # startup rendezvous barrier. Empty disables the ACK path (legacy/tests).
    ack_addr: str = ""


@dataclass
class PleOffloadRequest:
    """Sent by each DP rank's TP rank zero at every inference step."""

    dp_rank: int
    num_tokens: int
    num_reqs: int


@dataclass
class PleOffloadRegistrationAck:
    """Sent by the offload worker to one GPU worker once that worker's
    registration has been received AND its output targets are built.

    Receipt of this ACK is the startup rendezvous barrier: the GPU worker must
    not dispatch its first (warmup) PLE forward -- which enqueues an untimed
    ``cuStreamWaitValue32`` on the model stream -- until the offload worker is
    known to be serving it. A missing ACK is detected (TimeoutError), never an
    infinite GPU-stream wait (vllm-project/vllm#53960)."""

    worker_id: int
    num_layers: int


_PLE_OFFLOAD_REQUEST_DECODER = msgspec.msgpack.Decoder(PleOffloadRequest)
_PLE_OFFLOAD_ACK_DECODER = msgspec.msgpack.Decoder(PleOffloadRegistrationAck)
