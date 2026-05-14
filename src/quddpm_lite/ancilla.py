from __future__ import annotations

import torch

from .utils import COMPLEX_DTYPE, apply_cnot, apply_ry, apply_rz, normalize_statevectors


def append_ancilla_zero(states: torch.Tensor) -> torch.Tensor:
    batch_size, dim = states.shape
    out = torch.zeros((batch_size, dim * 2), dtype=COMPLEX_DTYPE, device=states.device)
    out[:, 0::2] = states.to(COMPLEX_DTYPE)
    return out


def project_ancilla(
    system_state: torch.Tensor,
    data_qubits: int,
    outcome: int = 0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if outcome not in {0, 1}:
        raise ValueError("outcome must be 0 or 1")
    del data_qubits
    selected = system_state[:, outcome::2]
    success_probability = selected.abs().pow(2).sum(dim=-1).real.clamp_min(eps)
    post_selected = selected / torch.sqrt(success_probability).unsqueeze(-1)
    return normalize_statevectors(post_selected), success_probability


def controlled_or_entangling_layer_with_ancilla(
    system_state: torch.Tensor,
    data_qubits: int,
) -> torch.Tensor:
    total_qubits = data_qubits + 1
    ancilla_index = data_qubits
    out = system_state
    for q in range(data_qubits):
        out = apply_cnot(out, q, ancilla_index, total_qubits)
    return out


def ancilla_rotation_block(
    system_state: torch.Tensor,
    angles: torch.Tensor,
    data_qubits: int,
) -> torch.Tensor:
    total_qubits = data_qubits + 1
    out = system_state
    for q in range(total_qubits):
        out = apply_ry(out, angles[:, q, 0], q, total_qubits)
        out = apply_rz(out, angles[:, q, 1], q, total_qubits)
    return out
