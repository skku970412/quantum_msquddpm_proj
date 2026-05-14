# QuDDPM-lite PyTorch Sandbox

이 프로젝트는 **QuDDPM/MSQuDDPM 아이디어를 바탕으로 한 본선 대비용 lightweight sandbox**입니다. 논문 완전 재현이나 하드웨어 수준 구현이 아니라, forward diffusion, denoising, prior-based generation evaluation, metric calculation, resource comparison을 팀 단위로 빠르게 실험하기 위한 PyTorch 기반 벤치마크입니다.

CUDA가 있으면 GPU를 사용하고, 없으면 CPU에서 실행됩니다.

## Project Scope

- 논문 전체 재현이 아닙니다.
- 본선 문제를 미리 정답 구현한 코드도 아닙니다.
- 목표는 작은 qubit 수와 가벼운 실험 설정에서 QuDDPM 계열 workflow를 반복 검증하는 것입니다.

Application note:

> 이 프로젝트는 본선 세부 문제의 정답을 미리 구현한 것이 아니라, Quantum DDPM 계열 문제에 필요한 forward diffusion, denoising, generation evaluation, metric calculation, resource comparison을 팀 단위로 사전 연습하기 위한 PyTorch 기반 sandbox입니다.

## What Was Added In This Version

- true `smoke` / `mini` preset
- prior-based generation evaluation
- `cnr` one-step comparator
- cumulative depolarizing schedule
- forward corruption logging
- extended resource metrics

## Models

| 모델 | 역할 |
|---|---|
| `msquddpm` | depolarizing forward를 되돌리는 기본 denoiser |
| `quddpm_baseline` | random-unitary forward를 쓰는 baseline denoiser |
| `t_msquddpm` | temporal sharing을 넣은 denoiser 변형 |
| `cnr` | latent classical noise에서 직접 상태를 생성하는 one-step comparator |

`cnr`는 QuDDPM 대체 모델이 아니라, **target-conditioned denoising 없이 latent classical noise만으로 ensemble을 생성하는 single-step comparator**입니다. post-selection 없이 resource-efficient한 비교군으로 두었습니다.

## Presets

| preset | 기본 목적 | 기본 설정 요약 |
|---|---|---|
| `smoke` | CPU/GPU 빠른 검증 | 2 qubits, 16 states, 1 epoch, `msquddpm` |
| `mini` | 짧은 기능 검증 | 4 qubits, 64 states, 10 epochs, `msquddpm/t_msquddpm/cnr` |
| `twohour` | 기존 스타일 장시간 벤치마크 | 6 qubits, 512 states, 400 epochs |
| `research` | 기본 연구용 defaults | 6 qubits, 512 states, 1000 epochs |
| `full` | 더 큰 장기 실행 | 6 qubits, 1024 states, 3000 epochs |

## Reconstruction Vs Generation

- reconstruction:
  clean target state에서 noisy input을 만든 뒤, model이 clean state를 얼마나 복원하는지 평가합니다.
- generation:
  target input을 직접 주지 않고 prior에서 시작해 generated ensemble을 만들고, 그 ensemble이 target test ensemble과 얼마나 가까운지 평가합니다.

이 둘은 의도적으로 분리되어 있으며, `metrics.csv`에는 reconstruction metric과 generation metric이 각각 따로 저장됩니다.

주요 컬럼 예시는 다음과 같습니다.

- reconstruction:
  `reconstruction_fidelity`, `reconstruction_pure_state_fidelity`, `reconstruction_mmd`, `reconstruction_wasserstein`
- generation:
  `generation_mmd`, `generation_wasserstein`, `generation_nearest_fidelity_mean`, `generation_prior_mode`, `generation_sampling_mode`

호환성을 위해 기존 `fidelity`, `mmd`, `wasserstein` 컬럼도 유지합니다.

- denoiser 계열은 reconstruction 값을 넣습니다.
- `cnr`는 generation 지표를 넣습니다.

## Forward Processes

지원하는 forward corruption은 다음 두 계열입니다.

- `random_unitary`
  `quddpm_baseline`에 사용됩니다.
- `depolarizing`
  `msquddpm`, `t_msquddpm`에 사용됩니다.

depolarizing schedule은 두 모드를 지원합니다.

- `single_beta`
  `rho_t = (1 - beta_t) rho_0 + beta_t I / d`
- `cumulative`
  `rho_t = alpha_bar_t rho_0 + (1 - alpha_bar_t) I / d`

`noise_curve.csv`에는 다음 컬럼이 저장됩니다.

- `step`
- `beta`
- `alpha`
- `alpha_bar`
- `expected_fidelity_single_beta`
- `expected_fidelity_cumulative`

## Fairness And Limitations

- random-unitary forward와 depolarizing forward는 corruption strength가 자동으로 같아지지 않습니다.
- 그래서 각 run에 `forward_final_target_noisy_fidelity_mean`과 `forward_process_type`을 같이 기록합니다.
- generation evaluation은 prior-based이지만, 현재 reverse chain은 논문 수준의 exact Markov sampler가 아니라 lightweight surrogate입니다.
- `cnr`는 QuDDPM 대체가 아니라 comparator입니다.
- resource metrics는 실제 hardware transpilation count가 아니라 heuristic estimate입니다.

명시적으로, **estimated resource metrics are heuristic, not transpiled hardware counts** 입니다.

또한 depolarizing channel의 physical cost를 explicit unitary depth와 동일시하지 않기 위해, `channel_application_count`를 별도 컬럼으로 분리해 기록합니다.

## Resource Metrics

`metrics.csv`에는 기존 parameter/depth 추정 외에 다음 컬럼이 추가되었습니다.

- `denoiser_depth_per_step`
- `denoiser_two_qubit_gates_per_step`
- `denoiser_single_qubit_rotations_per_step`
- `total_reverse_depth`
- `total_reverse_two_qubit_gate_count`
- `total_reverse_single_qubit_rotation_count`
- `forward_unitary_depth`
- `forward_two_qubit_gate_count`
- `channel_application_count`
- `total_estimated_depth`
- `total_estimated_two_qubit_gate_count`
- `generation_call_count`
- `resource_notes`

요약 집계는 `summary_table.csv`에 들어가며, generation quality와 total resource 평균도 함께 저장됩니다.

## Outputs

실행하면 결과 폴더에 보통 아래 파일들이 생성됩니다.

- `metrics.csv`
- `summary_table.csv`
- `seed_summary.csv`
- `loss_history.csv`
- `noise_curve.csv`
- `fidelity_comparison.png`
- `mmd_wasserstein_comparison.png`
- `resource_tradeoff.png`
- `fidelity_vs_noise.png`
- `generation_quality_comparison.png`
- `total_resource_tradeoff.png`
- 모델별 `loss_curve_*.png`

## Recommended Commands

smoke:

```bash
python main.py --preset smoke --results-dir results_smoke
python verify_results.py --results-dir results_smoke
```

mini with CNR:

```bash
python main.py --preset mini --models msquddpm t_msquddpm cnr --results-dir results_mini
python verify_results.py --results-dir results_mini
```

twohour original benchmark:

```bash
python main.py --preset twohour --results-dir results_twohour_new
python verify_results.py --results-dir results_twohour_new
```

baseline-only smoke:

```bash
python main.py --preset smoke --models quddpm_baseline --results-dir results_smoke_baseline
python verify_results.py --results-dir results_smoke_baseline
```

## CLI Notes

자주 쓰는 옵션은 아래와 같습니다.

- `--models msquddpm t_msquddpm cnr`
- `--dataset-kind cluster1q`
- `--depolarizing-mode cumulative`
- `--prior-mode random_pure|maximally_mixed_jitter|depolarized_random`
- `--generation-sampling-mode one_step|iterative`

## Code Layout

```text
.
├── main.py
├── verify_results.py
├── src/quddpm_lite/
│   ├── cli.py
│   ├── config.py
│   ├── datasets.py
│   ├── experiments.py
│   ├── metrics.py
│   ├── models.py
│   ├── noise.py
│   ├── random_unitary.py
│   ├── train.py
│   ├── utils.py
│   ├── verify.py
│   └── visualize.py
└── results_twohour/
```

## Setup

```bash
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install -r requirements.txt
```
