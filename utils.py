from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    state: dict[str, float] = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - state["start"]


def zero_state(batch_size: int, qubits: int, device: torch.device) -> torch.Tensor:
    dim = 2**qubits
    state = torch.zeros((batch_size, dim), dtype=COMPLEX_DTYPE, device=device)
    state[:, 0] = 1.0 + 0.0j
    return state


def normalize_statevectors(states: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(states, dim=-1, keepdim=True).clamp_min(eps)
    return (states / norm).to(COMPLEX_DTYPE)


def statevectors_to_density(states: torch.Tensor) -> torch.Tensor:
    states = normalize_statevectors(states)
    return states.unsqueeze(-1) * states.conj().unsqueeze(-2)


def density_to_real_features(rho: torch.Tensor) -> torch.Tensor:
    flat = rho.reshape(rho.shape[0], -1)
    return torch.cat([flat.real, flat.imag], dim=-1).to(REAL_DTYPE)


def statevector_to_real_features(states: torch.Tensor) -> torch.Tensor:
    states = normalize_statevectors(states)
    return torch.cat([states.real, states.imag], dim=-1).to(REAL_DTYPE)


def _as_batch_angles(theta: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    theta = torch.as_tensor(theta, dtype=REAL_DTYPE, device=device)
    if theta.ndim == 0:
        theta = theta.expand(batch_size)
    return theta.reshape(batch_size)


def apply_single_qubit_gate(
    states: torch.Tensor, gate: torch.Tensor, qubit: int, qubits: int
) -> torch.Tensor:
    batch_size = states.shape[0]
    view = states.reshape(batch_size, *([2] * qubits))
    view = view.movedim(qubit + 1, 1).contiguous()
    rest_shape = view.shape[2:]
    view = view.reshape(batch_size, 2, -1)
    out = torch.einsum("bij,bjk->bik", gate, view)
    out = out.reshape(batch_size, 2, *rest_shape).movedim(1, qubit + 1).contiguous()
    return out.reshape(batch_size, -1).to(COMPLEX_DTYPE)


def apply_ry(states: torch.Tensor, theta: torch.Tensor | float, qubit: int, qubits: int) -> torch.Tensor:
    batch_size = states.shape[0]
    theta = _as_batch_angles(theta, batch_size, states.device)
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    gate = torch.zeros((batch_size, 2, 2), dtype=COMPLEX_DTYPE, device=states.device)
    gate[:, 0, 0] = c
    gate[:, 0, 1] = -s
    gate[:, 1, 0] = s
    gate[:, 1, 1] = c
    return apply_single_qubit_gate(states, gate, qubit, qubits)


def apply_rz(states: torch.Tensor, theta: torch.Tensor | float, qubit: int, qubits: int) -> torch.Tensor:
    batch_size = states.shape[0]
    theta = _as_batch_angles(theta, batch_size, states.device)
    gate = torch.zeros((batch_size, 2, 2), dtype=COMPLEX_DTYPE, device=states.device)
    gate[:, 0, 0] = torch.exp(-0.5j * theta)
    gate[:, 1, 1] = torch.exp(0.5j * theta)
    return apply_single_qubit_gate(states, gate, qubit, qubits)


def _basis_bit_mask(dim: int, qubits: int, qubit: int, device: torch.device) -> torch.Tensor:
    bit_position = qubits - 1 - qubit
    idx = torch.arange(dim, device=device)
    return ((idx >> bit_position) & 1).bool()


def apply_cz(states: torch.Tensor, control: int, target: int, qubits: int) -> torch.Tensor:
    dim = states.shape[-1]
    c_mask = _basis_bit_mask(dim, qubits, control, states.device)
    t_mask = _basis_bit_mask(dim, qubits, target, states.device)
    phase = torch.ones(dim, dtype=COMPLEX_DTYPE, device=states.device)
    phase[c_mask & t_mask] = -1.0 + 0.0j
    return states * phase


def apply_cnot(states: torch.Tensor, control: int, target: int, qubits: int) -> torch.Tensor:
    dim = states.shape[-1]
    idx = torch.arange(dim, device=states.device)
    control_mask = _basis_bit_mask(dim, qubits, control, states.device)
    target_bit = qubits - 1 - target
    flipped = idx ^ (1 << target_bit)
    perm = torch.where(control_mask, flipped, idx)
    return states[:, perm]


def apply_cz_chain(states: torch.Tensor, qubits: int) -> torch.Tensor:
    for q in range(qubits - 1):
        states = apply_cz(states, q, q + 1, qubits)
    return states


def apply_cnot_chain(states: torch.Tensor, qubits: int) -> torch.Tensor:
    for q in range(qubits - 1):
        states = apply_cnot(states, q, q + 1, qubits)
    return states


def gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))
