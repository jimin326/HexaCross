# checkpoints: 실험 중간 모델 보관 (11-17 ~ 11-22)

대표 모델은 `models/`에 있다. 이 폴더는 **실험 기록용으로 남긴 중간 모델 전부**다.
회차 이름은 `docs/experiments/README.md`(실험 기록)와 `docs/experiments/loopB_log.md`의 이름과 같다.

- 가중치(`*.pt`)와 학습 설정·로그만 넣었다. 옵티마이저 상태(`ckpt.pt`)와 실행 때 복사해 둔 설정 파일(`*.py`)은 뺐다.
- 단일 신경망 파일은 PPO 정책 `state_dict`다 (`test.load_agent`, `tools/rl/eval_parallel.py`로 바로 평가 가능).
- `ens*.pt`, `runs/ft*/ens_*.pt`는 앙상블 형식이다 (`algorithms/ensemble.py`).
- `runs/res*/residual_*.pt`는 잔차 정책 형식이다 (`algorithms/residual.py`).
- 지도·관측·교통이 단계마다 바뀌었으므로, 예전 모델은 지금 환경에서 성적이 다르게 나온다.

## 폴더

| 폴더 | 내용 | 실험 기록 |
|---|---|---|
| `runs_early/` | 첫 실험. PR #2 지도, 시청 → 교대 한 경로 PPO (`ys_A` ~ `ys_H2`, 7경로 첫 시도 `ys_R0` ~ `ys_R3`) | 1단계 |
| `runs/ys_*` | 7경로 강화학습 (M, N, P, Q, C0~C2, D, K, L, W, X, Y, Z, AA, S0, V 회차). `snap_*k.pt` = 스냅샷 | 3단계 |
| `runs/ys_RF0`, `ys_RB0`, `ys_RB1` | 모방학습 모델에서 이어 한 PPO (RF0: bc5에서, RB: BC에서 + 모방학습 손실) | 4단계 |
| `runs/res1` ~ `res9` | 잔차 PPO (모방학습 앙상블 위에 보정만 학습) | 5-3 |
| `runs/ft1`, `ft2` | 앙상블 전체 PPO + 모방학습 손실. **J8_RL = `ft1/ens_753k.pt`** | 6단계 |
| `runs/lbft1` | 루프 B 앙상블 PPO 1차 (중단) | 7단계 |
| `runs/sr1` | 루프 B 단일 신경망 PPO. **최종 모델 = `sr1/snap_1802k.pt`** (`models/yeonsan_loopB_RL/model.pt`와 같음) | 7단계 |
| `runs/sr2` | 루프 B 충돌 집중 PPO (목표 달성으로 중단) | 7단계 |
| `runs/dtest`, `vt_test`, `merge_smoke`, `run_20260907_202220` | 도구 시험 실행 (실험 아님) | – |
| `il/` | 모방학습·DAgger로 학습한 신경망과 앙상블 (아래) | 4·5·7단계 |
| `misc/` | 이어 학습할 때 쓴 중간 가중치 (`mid_*`, `pad_*`, `init_*`, `dckpt_*`) | 3단계 |

## `il/` 이름

| 이름 | 내용 |
|---|---|
| `bc1` ~ `bc15*`, `test_bc` | 모방학습 첫 시도들. `bc3` 그대로 따라 하기(완주 3%), `bc5` 교사 보정(82%) |
| `bcD*`, `bcL*`, `bcs*`, `bT*`, `b5*`, `dag*` | 모방학습·DAgger 중간 실험 |
| `bB1*`, `bC2*`, `bD3*` | 7경로 DAgger 회차별 구성원 |
| `bE4*` → `ensE8.pt` | E8 (교통 0.85, DAgger B4) |
| `bG5*` → `ensG8.pt` | G8 (DAgger B5) |
| `bH5*`, `bH6*` → `ensH8.pt` | H8 (DAgger B6) |
| `bI7*` → `ensI8.pt` | I8 (넓은 차로 변경 보강) |
| `bJ8*` → `ensJ8.pt` | **J8** (DAgger B7, 7경로 모방학습 최고) |
| `ensF4.pt`, `ensF4E8.pt`, `ensGE16.pt` | 묶음 조합 실험 (채택 안 함) |
| `dist8.pt`, `distC8.pt` | 앙상블을 하나로 증류 |
| `bLB1_*` → `ensLB1.pt` | 루프 B DAgger 1차 |
| `bLB2_*` → `ensLB2.pt` | 루프 B DAgger 2차 (나빠짐) |
| `bLBY3_*` → `ensLBY3.pt` | 루프 B 황색 정답 보정 후. **`bLBY3_3` = m3 = sr1 시작점** (`models/yeonsan_loopB_RL/il_init_m3.pt`) |
| `lbft1_200k.pt`, `sr1_*.pt`, `sr2_*.pt` | 루프 B 강화학습 스냅샷 복사본 (평가에 쓴 이름) |
| `dup_E/` | E8 구성원 재학습 확인용 |
| `*.log` | 각 신경망의 학습 로그 |
