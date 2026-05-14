from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from utils import (
    COMPLEX_DTYPE,
    apply_cnot,
    apply_cnot_chain,
    apply_cz_chain,
    apply_ry,
    apply_rz,
    normalize_statevectors,
    zero_state,
)


@dataclass
class StateDataset:
    train: torch.Tensor
    test: torch.Tensor
    all_states: torch.Tensor
    labels: list[str]


def _product_from_angles(theta: torch.Tensor, phi: torch.Tensor, qubits: int) -> torch.Tensor:
    states = zero_state(theta.shape[0], qubits, theta.device)
    for q in range(qubits):
        states = apply_ry(states, theta[:, q], q, qubits)
        states = apply_rz(states, phi[:, q], q, qubits)
    return normalize_statevectors(states)


def make_product_clustered_states(
    size: int, qubits: int, device: torch.device, clusters: int = 4
) -> torch.Tensor:
    cluster_ids = torch.randint(0, clusters, (size,), device=device)
    centers = torch.linspace(0.2, 1.3, clusters, device=device) * math.pi
    phase_centers = torch.linspace(-0.8, 0.8, clusters, device=device) * math.pi
    qubit_offsets = torch.linspace(-0.12, 0.12, qubits, device=device)
    theta = centers[cluster_ids].unsqueeze(1) + qubit_offsets + 0.12 * torch.randn(
        size, qubits, device=device
    )
    phi = phase_centers[cluster_ids].unsqueeze(1) - qubit_offsets + 0.18 * torch.randn(
        size, qubits, device=device
    )
    return _product_from_angles(theta, phi, qubits).to(COMPLEX_DTYPE)


def make_entangled_clustered_states(size: int, qubits: int, device: torch.device) -> torch.Tensor:
    states = make_product_clustered_states(size, qubits, device)
    states = apply_cz_chain(states, qubits)
    states = apply_cnot_chain(states, qubits)
    return normalize_statevectors(states)


def make_bell_pair_product_states(size: int, qubits: int, device: torch.device) -> torch.Tensor:
    if qubits % 2 != 0:
        raise ValueError("Bell-pair product states require an even number of qubits.")
    states = zero_state(size, qubits, device)
    for control in range(0, qubits, 2):
        target = control + 1
        theta = (0.5 * math.pi) + 0.16 * torch.randn(size, device=device)
        phi = 2.0 * math.pi * torch.rand(size, device=device)
        states = apply_ry(states, theta, control, qubits)
        states = apply_rz(states, phi, control, qubits)
        states = apply_cnot(states, control, target, qubits)
    return normalize_statevectors(states)


def make_state_ensemble(
    size: int,
    qubits: int,
    device: torch.device,
    kind: str = "mixed",
) -> tuple[torch.Tensor, list[str]]:
    if kind == "product":
        return make_product_clustered_states(size, qubits, device), ["product"] * size
    if kind == "entangled":
        return make_entangled_clustered_states(size, qubits, device), ["entangled"] * size
    if kind == "bell":
        return make_bell_pair_product_states(size, qubits, device), ["bell"] * size
    if kind != "mixed":
        raise ValueError(f"Unknown dataset kind: {kind}")

    sizes = [size // 3, size // 3, size - 2 * (size // 3)]
    states = torch.cat(
        [
            make_product_clustered_states(sizes[0], qubits, device),
            make_entangled_clustered_states(sizes[1], qubits, device),
            make_bell_pair_product_states(sizes[2], qubits, device),
        ],
        dim=0,
    )
    labels = ["product"] * sizes[0] + ["entangled"] * sizes[1] + ["bell"] * sizes[2]
    perm = torch.randperm(size, device=device)
    states = states[perm]
    labels = [labels[i] for i in perm.detach().cpu().tolist()]
    return states, labels


def build_dataset(
    size: int,
    qubits: int,
    device: torch.device,
    train_fraction: float = 0.8,
    kind: str = "mixed",
) -> StateDataset:
    states, labels = make_state_ensemble(size=size, qubits=qubits, device=device, kind=kind)
    train_size = int(round(size * train_fraction))
    return StateDataset(
        train=states[:train_size],
        test=states[train_size:],
        all_states=states,
        labels=labels,
    )

