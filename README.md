# GPU-accelerated Resource-Efficient QuDDPM-lite Benchmark

Pure PyTorch benchmark for a resource-efficient **QuDDPM-lite / MSQuDDPM-lite** project. This is not presented as a full paper reproduction; it is a runnable simulation benchmark for 6-qubit density-matrix experiments with CUDA acceleration when available.

## What Was Built

- 6-qubit quantum state ensemble generation with product-clustered, entangled-clustered, and Bell-pair product states.
- Depolarizing forward diffusion:

```text
rho_noisy = (1 - beta) * rho + beta * I / dim
```

- MSQuDDPM-lite denoiser using a quantum-inspired RY/RZ state generator.
- QuDDPM-lite random-unitary baseline with random single-qubit rotations and CZ/CNOT entangling layers.
- T-MSQuDDPM-lite with temporal parameter sharing and sin/cos time embeddings.
- GPU-batched fidelity, fidelity-kernel MMD, and POT Sinkhorn Wasserstein evaluation.
- Resource metrics: trainable parameters, estimated circuit depth, two-qubit gate count, runtime, GPU memory, and diffusion steps.

## Verified Run

The completed run used the `twohour` preset from [twohour_conditions.md](twohour_conditions.md).

```bash
python main.py --preset twohour --results-dir results_twohour
python verify_results.py --results-dir results_twohour
```

Validation passed:

```text
Verified 36 benchmark rows in results_twohour
```

Run conditions:

| Item | Value |
|---|---:|
| Qubits | 6 |
| Input mode | density matrix |
| Hilbert dimension | 64 |
| Density matrix | 64 x 64 complex64 |
| Dataset size | 512 |
| Epochs | 400 |
| Batch size | 128 |
| Seeds | 0, 1, 2 |
| Noise steps | 10, 20 |
| PQC depth | 2, 4 |
| Device used | cuda |
| Total recorded train runtime | 48.87 min |
| Peak recorded GPU memory | 52.65 MB |

## Results

Overall mean across seeds, `T = 10/20`, and depth `2/4`:

| Model | Fidelity ↑ | MMD ↓ | Wasserstein ↓ | Avg Runtime / Row |
|---|---:|---:|---:|---:|
| MSQuDDPM-lite | 0.8012 | 0.0245 | 0.2182 | 74.42 s |
| QuDDPM-lite baseline | 0.2051 | 0.4552 | 0.7168 | 95.77 s |
| T-MSQuDDPM-lite | 0.7927 | 0.0257 | 0.2258 | 74.14 s |

Best high-fidelity settings in this run:

| Model | T | Depth | Fidelity Mean |
|---|---:|---:|---:|
| MSQuDDPM-lite | 20 | 4 | 0.9078 |
| MSQuDDPM-lite | 10 | 4 | 0.9066 |
| T-MSQuDDPM-lite | 10 | 4 | 0.9037 |
| T-MSQuDDPM-lite | 20 | 4 | 0.8880 |

The random-unitary baseline is intentionally harder in this lite benchmark because the forward process scrambles states more aggressively than the depolarizing channel. The temporal sharing model keeps nearly the same fidelity profile as MSQuDDPM-lite while using slightly fewer trainable parameters.

## Figures

### Fidelity Comparison

![Fidelity comparison](results_twohour/fidelity_comparison.png)

### MMD and Wasserstein Comparison

![MMD and Wasserstein comparison](results_twohour/mmd_wasserstein_comparison.png)

### Resource Tradeoff

![Resource tradeoff](results_twohour/resource_tradeoff.png)

### Fidelity Under Depolarizing Noise

![Fidelity under depolarizing noise](results_twohour/fidelity_vs_noise.png)

### Training Loss

![MSQuDDPM-lite loss curve](results_twohour/loss_curve_msquddpm.png)

![QuDDPM-lite baseline loss curve](results_twohour/loss_curve_baseline.png)

![T-MSQuDDPM-lite loss curve](results_twohour/loss_curve_t_msquddpm.png)

## Project Structure

```text
.
├── main.py
├── config.py
├── datasets.py
├── noise.py
├── random_unitary.py
├── metrics.py
├── models.py
├── train.py
├── experiments.py
├── visualize.py
├── utils.py
├── verify_results.py
├── twohour_conditions.md
├── WORKFLOW_TODO.md
├── requirements.txt
└── results_twohour/
```

File roles:

- `main.py`: CLI entry point.
- `config.py`: presets including `smoke`, `mini`, `twohour`, `research`, and `full`.
- `datasets.py`: quantum state ensemble generation.
- `noise.py`: depolarizing diffusion channel and statevector proxy for optional 8-qubit runs.
- `random_unitary.py`: QuDDPM-lite random-unitary forward process.
- `metrics.py`: fidelity, MMD, Wasserstein, and resource metrics.
- `models.py`: MSQuDDPM-lite, QuDDPM-lite baseline denoiser, and T-MSQuDDPM-lite.
- `train.py`: training and evaluation loop.
- `experiments.py`: model/seed/ablation runner and result writer.
- `visualize.py`: CSV-to-PNG plotting.
- `verify_results.py`: integrity check for required outputs.

## Reproduce

Create the environment:

```bash
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Quick smoke test:

```bash
python main.py --preset smoke --results-dir results
python verify_results.py --results-dir results
```

Main verified run:

```bash
python main.py --preset twohour --results-dir results_twohour
python verify_results.py --results-dir results_twohour
```

Research-sized run from the original plan:

```bash
python main.py --preset research --results-dir results_research
```

CUDA is selected automatically through `torch.cuda.is_available()`. If CUDA is unavailable, the same code falls back to CPU.

## Outputs

The verified run produced:

- `results_twohour/metrics.csv`
- `results_twohour/summary_table.csv`
- `results_twohour/seed_summary.csv`
- `results_twohour/loss_history.csv`
- `results_twohour/fidelity_comparison.png`
- `results_twohour/mmd_wasserstein_comparison.png`
- `results_twohour/resource_tradeoff.png`
- `results_twohour/fidelity_vs_noise.png`
- `results_twohour/loss_curve_msquddpm.png`
- `results_twohour/loss_curve_baseline.png`
- `results_twohour/loss_curve_t_msquddpm.png`

## Application Sentence

Implemented a GPU-aware, resource-efficient QuDDPM-lite benchmark in pure PyTorch for 6-qubit density-matrix simulation, comparing MSQuDDPM-lite, a random-unitary QuDDPM-lite baseline, and T-MSQuDDPM-lite temporal parameter sharing across fidelity, MMD, Wasserstein distance, runtime, and circuit resource estimates over three random seeds and T/depth ablations.

