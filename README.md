# GPU 기반 자원 효율적 QuDDPM-lite 벤치마크

순수 PyTorch만으로 구현한 6-qubit quantum diffusion simulation 벤치마크입니다. GPU 사용이 가능하면 CUDA를 자동으로 사용하고, CUDA가 없으면 CPU로 fallback합니다.

전체 QuDDPM 논문 재현이 아니라, 실제 실행 가능한 범위의 **QuDDPM-lite / MSQuDDPM-lite / resource-efficient benchmark** 구현과 비교 평가에 초점을 맞췄습니다.

## 프로젝트 개요

6-qubit quantum state ensemble에 forward diffusion을 적용한 뒤, quantum-inspired denoiser가 원래 clean state를 얼마나 잘 복원하는지 비교했습니다.

핵심 조건:

- 기본 실험: 6-qubit density matrix
- Hilbert dimension: `64`
- density matrix shape: `64 x 64`
- tensor dtype: `torch.complex64`
- CUDA 사용 가능 시 자동 GPU 사용
- CUDA가 없으면 CPU fallback
- Qiskit, PennyLane, TensorCircuit 없이 순수 PyTorch simulation으로 구현

## 구현한 내용

- 6-qubit quantum state ensemble 생성
  - product clustered states
  - entangled clustered states
  - Bell-pair product states
- depolarizing noise channel 기반 forward diffusion 구현
- random-unitary scrambling 기반 QuDDPM-lite baseline 구현
- MSQuDDPM-lite denoiser 구현
- T-MSQuDDPM-lite temporal parameter sharing 구조 구현
- seed 3개 반복 실험
- `T=10 vs T=20` ablation
- `depth=2 vs depth=4` ablation
- fidelity, MMD, Wasserstein distance 계산
- parameter count, estimated circuit depth, two-qubit gate count, runtime, GPU memory 기록
- 결과 CSV와 PNG 자동 저장

Depolarizing channel:

```text
rho_noisy = (1 - beta) * rho + beta * I / dim
```

## 비교 모델

| 모델 | 설명 |
|---|---|
| MSQuDDPM-lite | depolarizing forward process와 RY/RZ 기반 quantum-inspired denoiser 사용 |
| QuDDPM-lite baseline | random single-qubit rotation + CZ/CNOT entangling layer 기반 random-unitary scrambling 사용 |
| T-MSQuDDPM-lite | MSQuDDPM-lite에 temporal parameter sharing과 sin/cos time embedding 추가 |

## 검증된 실행 조건

본 README의 결과는 `twohour` preset으로 생성했습니다.

```bash
python main.py --preset twohour --results-dir results_twohour
python verify_results.py --results-dir results_twohour
```

검증 결과:

```text
Verified 36 benchmark rows in results_twohour
```

실험 조건:

| 항목 | 값 |
|---|---:|
| qubits | 6 |
| input mode | density matrix |
| Hilbert dimension | 64 |
| density matrix | 64 x 64 complex64 |
| dataset size | 512 |
| epochs | 400 |
| batch size | 128 |
| seeds | 0, 1, 2 |
| noise steps | 10, 20 |
| PQC depth | 2, 4 |
| device | cuda |
| recorded train runtime | 48.87 min |
| peak recorded GPU memory | 52.65 MB |

## 실험 결과 요약

seed 3개, `T=10/20`, `depth=2/4` 전체 평균입니다.

| 모델 | Fidelity ↑ | MMD ↓ | Wasserstein ↓ | 평균 runtime / row |
|---|---:|---:|---:|---:|
| MSQuDDPM-lite | 0.8012 | 0.0245 | 0.2182 | 74.42 s |
| QuDDPM-lite baseline | 0.2051 | 0.4552 | 0.7168 | 95.77 s |
| T-MSQuDDPM-lite | 0.7927 | 0.0257 | 0.2258 | 74.14 s |

높은 fidelity를 보인 설정:

| 모델 | T | Depth | Fidelity mean |
|---|---:|---:|---:|
| MSQuDDPM-lite | 20 | 4 | 0.9078 |
| MSQuDDPM-lite | 10 | 4 | 0.9066 |
| T-MSQuDDPM-lite | 10 | 4 | 0.9037 |
| T-MSQuDDPM-lite | 20 | 4 | 0.8880 |

결과 해석:

- MSQuDDPM-lite는 depolarizing noise 복원에서 높은 fidelity를 보였습니다.
- T-MSQuDDPM-lite는 parameter sharing을 적용했음에도 MSQuDDPM-lite와 유사한 성능을 유지했습니다.
- QuDDPM-lite random-unitary baseline은 forward process가 더 강하게 state를 scramble하기 때문에 fidelity가 낮고 MMD/Wasserstein이 크게 나타났습니다.
- depth 4 설정이 depth 2보다 fidelity 측면에서 뚜렷하게 유리했습니다.

## 결과 Figure

### Fidelity 비교

![Fidelity comparison](results_twohour/fidelity_comparison.png)

### MMD / Wasserstein 비교

![MMD and Wasserstein comparison](results_twohour/mmd_wasserstein_comparison.png)

### Resource tradeoff

![Resource tradeoff](results_twohour/resource_tradeoff.png)

### Depolarizing noise에 따른 fidelity

![Fidelity under depolarizing noise](results_twohour/fidelity_vs_noise.png)

### Training loss

![MSQuDDPM-lite loss curve](results_twohour/loss_curve_msquddpm.png)

![QuDDPM-lite baseline loss curve](results_twohour/loss_curve_baseline.png)

![T-MSQuDDPM-lite loss curve](results_twohour/loss_curve_t_msquddpm.png)

## 코드 구조

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

파일 역할:

- `main.py`: CLI 실행 진입점
- `config.py`: `smoke`, `mini`, `twohour`, `research`, `full` preset 정의
- `datasets.py`: product, entangled, Bell-pair state ensemble 생성
- `noise.py`: depolarizing channel과 optional 8-qubit statevector proxy
- `random_unitary.py`: QuDDPM-lite random-unitary forward process
- `metrics.py`: fidelity, MMD, Wasserstein, resource metric 계산
- `models.py`: MSQuDDPM-lite, QuDDPM-lite baseline denoiser, T-MSQuDDPM-lite
- `train.py`: 학습 및 평가 루프
- `experiments.py`: seed/model/ablation grid 실행과 결과 저장
- `visualize.py`: CSV 결과를 PNG figure로 변환
- `verify_results.py`: 결과 파일과 metric row 검증

## 실행 방법

환경 생성:

```bash
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install -r requirements.txt
```

빠른 smoke test:

```bash
python main.py --preset smoke --results-dir results
python verify_results.py --results-dir results
```

README의 결과를 재현하는 실행:

```bash
python main.py --preset twohour --results-dir results_twohour
python verify_results.py --results-dir results_twohour
```

원래 계획의 research-sized 실행:

```bash
python main.py --preset research --results-dir results_research
```

CUDA 사용 가능 여부는 `torch.cuda.is_available()`로 자동 판정합니다.

## 생성 산출물

검증된 실행에서 생성된 주요 파일:

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
