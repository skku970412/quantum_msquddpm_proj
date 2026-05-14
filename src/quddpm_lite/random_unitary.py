from __future__ import annotations

import math

import torch

from .utils import apply_cnot_chain, apply_cz_chain, apply_ry, apply_rz, normalize_statevectors, statevectors_to_density


@torch.no_grad()
def random_unitary_scramble_states(
    states: torch.Tensor,
    t: torch.Tensor,
    noise_steps: int,
    depth: int,
    qubits: int,
) -> torch.Tensor:
    out = states.clone()
    batch_size = out.shape[0]
    scale = ((t.float() + 1.0) / float(noise_steps)).clamp(0.0, 1.0) * (math.pi / 2.0)
    for layer in range(depth):
        layer_scale = scale * float(layer + 1) / float(depth)
        for q in range(qubits):
            theta = torch.randn(batch_size, device=states.device) * layer_scale
            phi = torch.randn(batch_size, device=states.device) * layer_scale
            out = apply_ry(out, theta, q, qubits)
            out = apply_rz(out, phi, q, qubits)
        out = apply_cz_chain(out, qubits) if layer % 2 == 0 else apply_cnot_chain(out, qubits)
    return normalize_statevectors(out)


@torch.no_grad()
def random_unitary_forward_from_states(
    states: torch.Tensor,
    t: torch.Tensor,
    noise_steps: int,
    depth: int,
    qubits: int,
) -> torch.Tensor:
    scrambled = random_unitary_scramble_states(states, t, noise_steps, depth, qubits)
    return statevectors_to_density(scrambled)


def estimated_random_unitary_resources(depth: int, qubits: int) -> dict[str, int]:
    return {
        "random_unitary_single_qubit_rotations": depth * qubits * 2,
        "random_unitary_two_qubit_gates": depth * (qubits - 1),
    }

