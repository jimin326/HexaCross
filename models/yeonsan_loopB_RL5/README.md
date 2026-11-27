# yeonsan_loopB_RL5: 고친 루프 B 지도(PR #31)에서 RL4 방법으로 다시 학습한 모델 (2026-11-27)

PR #31(회전교차로 한 바퀴 제거 + 반송로×쌍미천로 정지선 위치 정정)로 바뀐 루프 B 지도에서,
`models/yeonsan_loopB_RL4`와 **같은 방법·같은 설정**으로 PPO를 다시 돌린 모델이다 (실행 이름 r9).
바꾼 것은 지도 하나이고, 시작 모델(`checkpoints/il/bcstop6.pt`)·모방학습 데이터·보상·하이퍼파라미터·`SEED=71`은 RL4와 같다.
RL4는 지우지 않고 그대로 두었다.

## RL4와 다른 점

| 항목 | RL4 (r8) | RL5 (r9) |
|---|---|---|
| 지도 | PR #31 이전 | **PR #31 반영 (`5287f95`)** |
| 모방학습 데이터 경로 | `data/lf/*_s.npz` | `models/yeonsan_loopB_RL4/*_s.npz` (같은 파일, #32) |
| 실행 환경 | Linux, Python 3.13 | Windows 11, Python 3.12.10, torch 2.15.0 CPU, SUMO 1.27.1 |
| 채택 저장본 | 102k | 102k |

`r9_config.json`과 RL4 `r8_config.json`을 비교하면 `BC_DATA` 경로 한 줄만 다르다.

## 학습

- `run_r9.ps1` (`run_r8.sh`를 PowerShell로 옮김, 값은 그대로). `N_ENVS=4` 병렬 환경이 Windows에서도 그대로 돌았다.
- 102k까지 약 16분(갱신 25번, 정책 갱신 5번). 시작 모델과 정책 가중치 상대 차 1.1% (RL4 1.2%).
- 22.1만 스텝에서 한 번 멈췄다가 20.48만 스텝 `ckpt.pt`에서 이어 돌렸고, 102k 채택이 정해져 21.7만 스텝에서 끝냈다.
  `r9_train_log.csv`에서 갱신 51~53은 두 번 나온다 (이어 돌리기 전·후).

## 저장본별 평가 (학습 조건, 120회, `--seed 905`)

| 저장본 | 완주 | 충돌 | 신호위반 | 앞차 / 정지선 정지 (중앙값) |
|---|---|---|---|---|
| 28k · 53k · 77k (정책 고정 구간 = bcstop6) | 119 | 1 | 0 | 4.55 / 1.78 m |
| **102k (채택)** | **119** | **1** | 0 | **4.55 / 1.63 m** |
| 126k | 117 | 3 | 0 | 4.53 / 1.63 m |
| 151k | 119 | 1 | 0 | 4.50 / 1.62 m |
| 176k | 116 | 3 | 1 | 4.46 / 1.64 m |

- 28k~151k의 충돌 1회는 모두 같은 판(시드 1004, 82초, 내부 연결로 `:414663644_33`)이다.
- 126k·176k에서 늘어난 충돌은 모두 `-37400617#4` 구간(약 1,160~1,230초)에서 났다.
- 120회에서 충돌 1회와 3회의 차이는 우연 범위다. 102k와 151k가 같은 성적이고, RL4와 같은 102k를 채택했다.

## 최종 평가 (4조건 × 240회, `--seed 60001`, 선택에 쓰지 않은 시드)

같은 시드·같은 지도로 RL4도 평가해 비교한다. 결과는 `eval/` (RL5), `eval_RL4/` (같은 지도의 RL4).

**진행 중 — 끝나면 표를 채운다.**

판단 기준 (미리 정함): RL5가 RL4보다 완주가 낮지 않고 충돌이 많지 않을 때만 교체 후보. 차이가 우연 범위면 "차이 없음".

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 채택 모델 (HybridActorCritic state_dict, r9 102k 스텝) |
| `r9_config.json` | 학습 설정 (환경 변수, 하이퍼파라미터, 보상) |
| `run_r9.ps1` | PPO 실행 스크립트 (Windows PowerShell) |
| `r9_train.txt`, `r9_train_log.csv` | 학습 로그 |

## 평가

```bash
pip install -r requirements.txt   # SUMO 1.27.1
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python tools/rl/eval_parallel.py models/yeonsan_loopB_RL5/model.pt results/eval_RL5 --n 240 --workers 4 --chunk 15 --seed 60001
```

Windows에서 보고서 출력에 `UnicodeEncodeError: 'cp949'`가 나면 `PYTHONIOENCODING=utf-8`을 지정한다 (`eval.json`은 그 전에 저장된다).
