from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    import ot
except ImportError:  # pragma: no cover - verified by dependency check.
    ot = None

from .utils import gpu_memory_mb, normalize_statevectors


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


def differentiable_mmd_fidelity_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    k_xx = pairwise_fidelity_matrix(x, x)
    k_yy = pairwise_fidelity_matrix(y, y)
    k_xy = pairwise_fidelity_matrix(x, y)
    return (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).real


def mean_nearest_fidelity(target: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    pairwise = pairwise_fidelity_matrix(target, generated)
    return pairwise.max(dim=0).values.mean().real


def nearest_fidelity_loss(target: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    return 1.0 - mean_nearest_fidelity(target, generated)


def wasserstein_fidelity_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    subset: int | None = 256,
    method: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
) -> float:
    x = _take_subset(x, subset)
    y = _take_subset(y, subset)
    cost = (1.0 - pairwise_fidelity_matrix(x, y)).clamp_min(0.0).detach().cpu().numpy()
    if ot is None:
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) == 0:
            return float("nan")
        return float(cost[row_ind, col_ind].mean())
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
    generation_sampling_mode: str = "one_step",
) -> dict[str, Any]:
    denoiser_depth_per_step = depth * (2 * qubits + (qubits - 1))
    denoiser_two_qubit_gates_per_step = depth * (qubits - 1)
    denoiser_single_qubit_rotations_per_step = depth * qubits * 2

    if model_name == "cnr":
        total_reverse_depth = 0
        total_reverse_two_qubit_gate_count = 0
        total_reverse_single_qubit_rotation_count = 0
        forward_unitary_depth = 0
        forward_two_qubit_gate_count = 0
        channel_application_count = 0
        generation_call_count = 1
        total_estimated_depth = denoiser_depth_per_step
        total_estimated_two_qubit_gate_count = denoiser_two_qubit_gates_per_step
        resource_notes = "One-step classical-noise reuploading comparator heuristic."
    else:
        total_reverse_depth = noise_steps * denoiser_depth_per_step
        total_reverse_two_qubit_gate_count = noise_steps * denoiser_two_qubit_gates_per_step
        total_reverse_single_qubit_rotation_count = (
            noise_steps * denoiser_single_qubit_rotations_per_step
        )
        if model_name == "quddpm_baseline":
            forward_unitary_depth = noise_steps * denoiser_depth_per_step
            forward_two_qubit_gate_count = noise_steps * denoiser_two_qubit_gates_per_step
            channel_application_count = 0
            resource_notes = "Random-unitary forward and reverse denoiser costs are both counted."
        else:
            forward_unitary_depth = 0
            forward_two_qubit_gate_count = 0
            channel_application_count = noise_steps
            resource_notes = (
                "Depolarizing channels are counted separately from explicit unitary-depth heuristics."
            )
        generation_call_count = 1 if generation_sampling_mode == "one_step" else noise_steps
        total_estimated_depth = forward_unitary_depth + total_reverse_depth
        total_estimated_two_qubit_gate_count = (
            forward_two_qubit_gate_count + total_reverse_two_qubit_gate_count
        )

    return {
        "trainable_parameters": count_parameters(model),
        "estimated_circuit_depth": denoiser_depth_per_step,
        "estimated_two_qubit_gate_count": denoiser_two_qubit_gates_per_step,
        "estimated_single_qubit_rotation_count": denoiser_single_qubit_rotations_per_step,
        "denoiser_depth_per_step": denoiser_depth_per_step,
        "denoiser_two_qubit_gates_per_step": denoiser_two_qubit_gates_per_step,
        "denoiser_single_qubit_rotations_per_step": denoiser_single_qubit_rotations_per_step,
        "total_reverse_depth": total_reverse_depth,
        "total_reverse_two_qubit_gate_count": total_reverse_two_qubit_gate_count,
        "total_reverse_single_qubit_rotation_count": total_reverse_single_qubit_rotation_count,
        "forward_unitary_depth": forward_unitary_depth,
        "forward_two_qubit_gate_count": forward_two_qubit_gate_count,
        "channel_application_count": channel_application_count,
        "total_estimated_depth": total_estimated_depth,
        "total_estimated_two_qubit_gate_count": total_estimated_two_qubit_gate_count,
        "generation_call_count": generation_call_count,
        "resource_notes": resource_notes,
        "runtime_sec": runtime_sec,
        "gpu_memory_mb": gpu_memory_mb(device),
        "diffusion_steps": noise_steps,
        "model_family": model_name,
    }
