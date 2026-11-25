# yeonsan_loopB_RL3: 시드 버그를 고친 main에서 루프 경로 B 다시 학습 (2026-11-26)

`models/yeonsan_loopB_RL2`와 같은 환경·같은 목표로, 시드 버그를 고친 main(#19 `eval_routes`/`dagger_collect` 시드 수정, #20)에서
DAgger 데이터부터 다시 모아 학습한 단일 신경망이다. RL2는 지우지 않고 그대로 두었다.

## 왜 다시 학습했나 (시드 버그)

`algorithms/ppo.py`의 PPO 생성자가 `np.random.seed(0)`을 다시 건다. 모델을 만들기 **전에** 시드를 거는 스크립트는
어떤 `SEED`를 줘도 0번 시드에서 시작했다 (main #16부터 배경 교통 난수가 `np.random`을 따르므로 영향이 생김).

| 스크립트 | 영향 | 이유 |
|---|---|---|
| `eval_routes.py` | 있음 (main #19에서 고침) | 시드를 건 뒤 모델을 만든다 |
| `tools/rl/dagger_collect.py` | 있음 (고친 main으로 다시 모음) | 같음. RL2 학습에 쓴 DAgger 결과가 136/12/12, 148/8/4처럼 모두 4의 배수였다 (워커 4개가 같은 에피소드를 돎) |
| `tools/rl/eval_parallel.py` | **없음** | 모델을 먼저 불러오고 에피소드마다 `np.random.seed(ep_seed)` |
| `collect_pomdp_data.py`, `residual_train.py` | 없음 | 워커가 PPO를 만들지 않는다 |

RL2의 평가(`eval_parallel.py`, 960회 98.75%)는 유효하다. 다만 RL2의 DAgger 데이터는 서로 다른 에피소드가 1/4뿐이었다.
이번에는 고친 수집기로 모아 결과가 4의 배수가 아니게 나왔다 (152/7/1, 159/1).

## 결과 (결정적 정책, `eval_parallel.py`, 조건마다 240회, 시드 905~)

| 조건 (`loop_flows` 켬) | 완주 | 충돌 | 신호위반 | 차로 오류 | 시간 초과 |
|---|---|---|---|---|---|
| 운전자 성향 + 택시 + 잠깐 정차 (학습 조건) | 235 (97.9%) | 1 (0.42%) | 4 (1.7%) | 0 | 0 |
| 도로 디테일 없음 | 236 (98.3%) | 2 (0.83%) | 2 (0.8%) | 0 | 0 |
| 운전자 성향 | 235 (97.9%) | 2 (0.83%) | 3 (1.3%) | 0 | 0 |
| 택시 + 잠깐 정차 | 235 (97.9%) | 2 (0.83%) | 2 (0.8%) | 1 | 0 |
| **합계 960회** | **941 (98.02%)** | **7 (0.73%)** | 11 (1.15%) | 1 | 0 |
| 목표 | 97% 이상 | 1% 이하 | | | |

네 조건 모두 목표를 넘는다. 같은 평가에서 RL2(948회 98.75%, 충돌 2회 0.21%)보다는 조금 낮다.
조건별 원자료는 `eval/<조건>/` (eval.json, summary.csv, report.md).

## 만든 과정

1. **DAgger 2번 (고친 수집기, 각 160회, 워커 4개, `BETA=0`, `LANE_LABEL=best`)**
   - r3a: 학생 `il_init_bcfix` (RL2의 시작점), SEED 91 → 학생 결과 완주 152, 충돌 7, 차로 오류 1
   - r3b: 학생 RL2, SEED 92 → 학생 결과 완주 159, 충돌 1
2. **정답 보정** (RL2와 같은 도구): 황색 신호 (`relabel_yellow.py`, 4배 복제), 신호 없는 회전·합류 접근 감속 (`relabel_turn.py`, 3배 복제), 차로 변경 보강 (`make_bcdata.py`)
3. **PPO r3** (`tools/rl/residual_train.py`, 시작 `il_init_bcfix`)
   - 모방학습 손실 데이터: RL2 때의 `bc_lby_t`, `dg12_bc` + 새로 모은 `r3a_bc`, `r3b_bc` (약 276만 표본), 가중치 1.0 → 0.2
   - **새로 켠 것: `YELLOW_PEN=20`** (main #20). 4 m/s² 이하로 멈출 수 있었던 황색을 지나가면 −20. 관측만 쓴다.
   - 나머지는 RL2와 같다: 학습률 3e-5, 4,096 스텝마다 갱신, 충돌 −50, 신호위반 −30, 차로 오류 −10, TTC 0.3, 위험 장면 30% 재시작, `SEED=63`
4. **평가**: 200k 스냅샷 120회 → 완주 118, 충돌 0, 신호위반 2. 4조건 × 240회로 확인하고 채택. 목표를 넘겨 약 44만 스텝에서 학습을 멈췄다.

설정 전체는 `r3_config.json`, 실행 스크립트는 `run_r3.sh`, DAgger·보정 로그는 `r3_dagger_prep.txt`, 학습 로그는 `r3_train.txt`·`r3_train_log.csv`.
스냅샷은 `checkpoints/runs/r3/` (200k = 이 모델, 401k).

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 채택 모델 (HybridActorCritic state_dict, r3 200k 스텝) |
| `r3_config.json` | 학습 설정 (환경 변수, 하이퍼파라미터, 보상) |
| `run_r3.sh` | DAgger → 보정 → PPO 실행 스크립트 |
| `r3_dagger_prep.txt` | DAgger 결과와 보정 표본 수 |
| `r3_train.txt`, `r3_train_log.csv` | 학습 로그 |
| `eval/` | 4조건 확인 평가 결과 |

## 평가

```bash
pip install -r requirements.txt   # SUMO 1.27.1
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python tools/rl/eval_parallel.py models/yeonsan_loopB_RL3/model.pt results/eval_RL3 --n 240 --workers 4 --chunk 15
```
