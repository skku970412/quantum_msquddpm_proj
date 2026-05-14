# 2시간 안쪽 목표 실행 조건

이 문서는 `gpu-quddpm-lite-benchmark` 프로젝트에서 긴 `research` 실행 대신, 결과표와 그래프를 확보하면서 약 2시간 안쪽 완료를 목표로 잡은 실행 조건을 정리한 것이다.

## 실행 목표

- 6-qubit density matrix 실험 유지
- 모델 3개 비교 유지
  - `msquddpm`
  - `quddpm_baseline`
  - `t_msquddpm`
- seed 3개 반복 유지: `[0, 1, 2]`
- `T = 10 vs 20` ablation 유지
- `depth = 2 vs 4` ablation 유지
- 결과 CSV와 PNG 자동 저장 유지
- 실행 시간은 현재 CUDA 환경 기준 약 1~2시간 안쪽을 목표로 함

## Twohour Preset 조건

`python main.py --preset twohour`는 다음 조건을 사용한다.

| 항목 | 값 |
|---|---:|
| qubits | 6 |
| input mode | density |
| Hilbert dimension | 64 |
| density matrix shape | 64 x 64 complex64 |
| dataset size | 512 |
| epochs | 400 |
| batch size | 128 |
| learning rate | 1e-3 |
| hidden dim | 96 |
| seeds | 0, 1, 2 |
| noise steps grid | 10, 20 |
| depth grid | 2, 4 |
| MMD subset | 128 |
| Wasserstein subset | 128 |
| eval subset | 128 |
| device | auto, CUDA 우선 |

## 실행 명령

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
. .venv/bin/activate
python main.py --preset twohour --results-dir results_twohour
python verify_results.py --results-dir results_twohour
```

## 산출물

실행이 끝나면 `results_twohour/`에 다음 파일들이 생성되어야 한다.

- `metrics.csv`
- `summary_table.csv`
- `seed_summary.csv`
- `loss_curve_msquddpm.png`
- `loss_curve_baseline.png`
- `fidelity_comparison.png`
- `mmd_wasserstein_comparison.png`
- `resource_tradeoff.png`
- `fidelity_vs_noise.png`

## 검증 기준

`verify_results.py` 통과 조건:

- 총 36개 benchmark row 존재
- 모델 3개 모두 포함
- seed `[0, 1, 2]` 포함
- `T=10`, `T=20` 포함
- `depth=2`, `depth=4` 포함
- fidelity 값이 `[0, 1]` 범위
- parameter count와 two-qubit gate count가 양수
- 필수 CSV/PNG 결과 파일 존재

## Research Preset과 차이

`research` 기본값은 `dataset_size=512`, `epochs=1000`이므로 현재 환경에서 2시간을 넘길 가능성이 있다. `twohour`는 dataset size와 실험 조합은 유지하면서, epoch를 400으로 줄이고 batch/eval subset을 조정해 지원서용 결과표와 그래프를 현실적인 시간 안에 확보하는 설정이다.

