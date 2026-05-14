from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    import ot
except ImportError:  # pragma: no cover - verified by dependency check.
    ot = None

from utils import gpu_memory_mb, normalize_statevectors


def pure_state_fidelity(target: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    target = normalize_statevectors(target)
    generated = normalize_statevectors(generated)
    overlap = torch.sum(target.conj() * generated, dim=-1)
    return torch.abs(overlap).pow(2).real


def pure_density_fidelity(target: torch.Tensor, generated_rho: torch.Tensor) -> torch.Tensor:
    target = normalize_statevectors(target)
    value = torch.einsum("bi,bij,bj->b", target.conj(), generated_rho, target)
    return value.real.clamp_min(0.0)


def pairwise_fidelity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = normalize_statevectors(x)
    y = normalize_statevectors(y)
    return torch.abs(x @ y.conj().T).pow(2).real


def _take_subset(x: torch.Tensor, subset: int | None) -> torch.Tensor:
    if subset is None or x.shape[0] <= subset:
        return x
    return x[:subset]


def mmd_fidelity(x: torch.Tensor, y: torch.Tensor, subset: int | None = 256) -> float:
    x = _take_subset(x, subset)
    y = _take_subset(y, subset)
    k_xx = pairwise_fidelity_matrix(x, x)
    k_yy = pairwise_fidelity_matrix(y, y)
    k_xy = pairwise_fidelity_matrix(x, y)
    return float((k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).detach().cpu())


def wasserstein_fidelity_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    subset: int | None = 256,
    method: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
) -> float:
    if ot is None:
        return float("nan")
    x = _take_subset(x, subset)
    y = _take_subset(y, subset)
    cost = (1.0 - pairwise_fidelity_matrix(x, y)).clamp_min(0.0).detach().cpu().numpy()
    a = np.ones(cost.shape[0], dtype=np.float64) / cost.shape[0]
    b = np.ones(cost.shape[1], dtype=np.float64) / cost.shape[1]
    if method == "emd":
        return float(ot.emd2(a, b, cost))
    return float(ot.sinkhorn2(a, b, cost, reg=sinkhorn_reg))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resource_metrics(
    model: torch.nn.Module,
    qubits: int,
    depth: int,
    noise_steps: int,
    runtime_sec: float,
    device: torch.device,
    model_name: str,
) -> dict[str, Any]:
    base_two_qubit_gates = depth * (qubits - 1)
    base_single_qubit_rotations = depth * qubits * 2
    return {
        "trainable_parameters": count_parameters(model),
        "estimated_circuit_depth": depth * (2 * qubits + (qubits - 1)),
        "estimated_two_qubit_gate_count": base_two_qubit_gates,
        "estimated_single_qubit_rotation_count": base_single_qubit_rotations,
        "runtime_sec": runtime_sec,
        "gpu_memory_mb": gpu_memory_mb(device),
        "diffusion_steps": noise_steps,
        "model_family": model_name,
    }

