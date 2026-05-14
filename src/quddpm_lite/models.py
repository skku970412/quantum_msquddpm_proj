from __future__ import annotations

import math

import torch
from torch import nn

from .utils import (
    apply_cz_chain,
    apply_ry,
    apply_rz,
    density_to_real_features,
    normalize_statevectors,
    statevector_to_real_features,
    zero_state,
)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, noise_steps: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t.float().unsqueeze(1)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    scaled_t = t.float().unsqueeze(1) / max(noise_steps - 1, 1)
    args = scaled_t * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[1]))
    return emb


class RotationStateGenerator(nn.Module):
    def __init__(self, qubits: int, depth: int):
        super().__init__()
        self.qubits = qubits
        self.depth = depth

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        batch_size = angles.shape[0]
        states = zero_state(batch_size, self.qubits, angles.device)
        for layer in range(self.depth):
            for q in range(self.qubits):
                states = apply_ry(states, angles[:, layer, q, 0], q, self.qubits)
                states = apply_rz(states, angles[:, layer, q, 1], q, self.qubits)
            states = apply_cz_chain(states, self.qubits)
        return normalize_statevectors(states)


class QuantumDenoiser(nn.Module):
    def __init__(
        self,
        qubits: int,
        noise_steps: int,
        depth: int,
        hidden_dim: int = 128,
        time_embedding_dim: int = 32,
        temporal_sharing: bool = False,
        input_mode: str = "density",
    ):
        super().__init__()
        self.qubits = qubits
        self.dim = 2**qubits
        self.noise_steps = noise_steps
        self.depth = depth
        self.temporal_sharing = temporal_sharing
        self.time_embedding_dim = time_embedding_dim
        self.input_mode = input_mode

        if input_mode == "density":
            feature_dim = 2 * self.dim * self.dim
        elif input_mode == "statevector":
            feature_dim = 2 * self.dim
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        if temporal_sharing:
            self.time_table = None
            self.shared_angle_bias = nn.Parameter(torch.zeros(depth, qubits, 2))
        else:
            self.time_table = nn.Embedding(noise_steps, time_embedding_dim)
            self.time_angle_bias = nn.Parameter(torch.zeros(noise_steps, depth, qubits, 2))

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.angle_head = nn.Linear(hidden_dim, depth * qubits * 2)
        self.generator = RotationStateGenerator(qubits=qubits, depth=depth)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.angle_head.weight)
        nn.init.normal_(self.angle_head.bias, mean=0.0, std=0.02)
        if not self.temporal_sharing:
            nn.init.normal_(self.time_angle_bias, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.shared_angle_bias, mean=0.0, std=0.02)

    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long().clamp(0, self.noise_steps - 1)
        if self.temporal_sharing:
            return sinusoidal_time_embedding(t, self.time_embedding_dim, self.noise_steps)
        if self.time_table is None:
            raise RuntimeError("time_table was not initialized")
        return self.time_table(t)

    def forward(self, noisy_state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "density":
            features = density_to_real_features(noisy_state)
        else:
            features = statevector_to_real_features(noisy_state)
        time_features = self._time_features(t)
        hidden = self.encoder(torch.cat([features, time_features], dim=-1))
        angles = self.angle_head(hidden).reshape(-1, self.depth, self.qubits, 2)
        if self.temporal_sharing:
            angles = angles + self.shared_angle_bias.unsqueeze(0)
        else:
            angles = angles + self.time_angle_bias[t.long().clamp(0, self.noise_steps - 1)]
        return self.generator(angles)


class MSQuDDPMLite(QuantumDenoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, temporal_sharing=False, **kwargs)


class QuDDPMLiteBaselineDenoiser(QuantumDenoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, temporal_sharing=False, **kwargs)


class TMSQuDDPMLite(QuantumDenoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, temporal_sharing=True, **kwargs)


def build_model(
    model_name: str,
    qubits: int,
    noise_steps: int,
    depth: int,
    hidden_dim: int,
    time_embedding_dim: int,
    input_mode: str = "density",
) -> nn.Module:
    common = dict(
        qubits=qubits,
        noise_steps=noise_steps,
        depth=depth,
        hidden_dim=hidden_dim,
        time_embedding_dim=time_embedding_dim,
        input_mode=input_mode,
    )
    if model_name == "msquddpm":
        return MSQuDDPMLite(**common)
    if model_name == "quddpm_baseline":
        return QuDDPMLiteBaselineDenoiser(**common)
    if model_name == "t_msquddpm":
        return TMSQuDDPMLite(**common)
    raise ValueError(f"Unknown model: {model_name}")
