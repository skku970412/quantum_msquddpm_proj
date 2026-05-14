# 남은 작업 체크리스트

이 문서는 세션이 끊기거나 수동으로 상태를 확인해야 할 때 이어서 볼 수 있는 작업 목록이다.

## 1. Twohour 벤치마크 실행

실행 명령:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
. .venv/bin/activate
python main.py --preset twohour --results-dir results_twohour
```

완료 조건:

- `results_twohour/metrics.csv` 생성
- 총 36개 benchmark row 생성
- 모델 3개 포함
  - `msquddpm`
  - `quddpm_baseline`
  - `t_msquddpm`
- seed 3개 포함: `0, 1, 2`
- `T=10`, `T=20` 포함
- `depth=2`, `depth=4` 포함

## 2. 결과 검증

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
. .venv/bin/activate
python verify_results.py --results-dir results_twohour
```

성공 메시지 예:

```text
Verified 36 benchmark rows in results_twohour
```

## 3. 결과 요약 확인

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
. .venv/bin/activate
python - <<'PY'
import pandas as pd
m = pd.read_csv("results_twohour/metrics.csv")
print("rows:", len(m))
print("models:", sorted(m.model.unique()))
print("seeds:", sorted(m.seed.unique()))
print("T:", sorted(m.noise_steps.unique()))
print("depths:", sorted(m.depth.unique()))
print("devices:", sorted(m.device.unique()))
print(m.groupby("model")[["fidelity", "mmd", "wasserstein", "runtime_sec"]].mean())
PY
```

확인할 것:

- `devices`에 `cuda`가 들어가 있는지
- fidelity가 0~1 범위인지
- MMD/Wasserstein 값이 생성됐는지
- `results_twohour/*.png` figure 파일들이 생성됐는지

## 4. README 정리

검증이 통과하면 `README.md`에 다음 내용을 반영한다.

- 프로젝트 수행 요약
- twohour 실행 조건
- 주요 결과 표 요약
- figure 삽입
  - `results_twohour/fidelity_comparison.png`
  - `results_twohour/mmd_wasserstein_comparison.png`
  - `results_twohour/resource_tradeoff.png`
  - `results_twohour/fidelity_vs_noise.png`
  - `results_twohour/loss_curve_msquddpm.png`
  - `results_twohour/loss_curve_baseline.png`
- 지원서용 프로젝트 수행 경험 문장

## 5. GitHub 업로드 준비

`.gitignore`에 최소한 다음 항목을 포함한다.

```text
.venv/
__pycache__/
*.pyc
results/
.pytest_cache/
```

업로드에 포함할 것:

- Python source files
- `README.md`
- `requirements.txt`
- `twohour_conditions.md`
- `WORKFLOW_TODO.md`
- `results_twohour/`의 CSV/PNG 결과

업로드에서 제외할 것:

- `.venv/`
- `__pycache__/`
- smoke 임시 결과인 `results/`

## 6. GitHub push

대상 repository:

```text
https://github.com/skku970412/quantum_msquddpm_proj.git
```

명령:

```bash
cd /home/work/llama_young/gpu-quddpm-lite-benchmark
git init
git branch -M main
git remote add origin https://github.com/skku970412/quantum_msquddpm_proj.git
git add .
git commit -m "Add quantum MSQuDDPM-lite benchmark"
git push -u origin main
```

이미 remote가 있으면:

```bash
git remote set-url origin https://github.com/skku970412/quantum_msquddpm_proj.git
git push -u origin main
```

HTTPS 인증이 막히면 Personal Access Token을 password 칸에 입력한다. VS Code askpass socket 오류가 나면 아래처럼 터미널 prompt를 강제한다.

```bash
unset GIT_ASKPASS SSH_ASKPASS VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_EXTRA_ARGS VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_IPC_HANDLE
git push -u origin main
```
