# yeonsan_loopC_RL: 루프 경로 C (모방학습 신경망 → 강화학습 2단계)

연산교차로 북쪽에서 출발해 월드컵대로 남쪽 → 연수로 좌회전 → 쌍미천로 좌회전(북쪽) → 고분로 좌회전 → 연산교차로 동쪽으로 다시 들어와 서쪽으로 나가는 루프 경로 C를 학습한 **단일 신경망**이다 (앙상블 아님).

- 1단계: J8의 구성원 신경망 하나(m6)에서 시작해 PPO로 140만 스텝 학습했다 (`rl_init_1400k.pt`).
- 2단계: 1단계 모델에서 이어서 PPO 80만 스텝을 더 학습했다 (`lc_ft7`, `model.pt`). 황색 신호 장면을 늘린 모방학습 손실과 "설 수 있었던 황색 통과" 감점을 넣었다.

## 결과 (교통량 1.0, 결정적 정책, libsumo, 960회)

| 평가 시드 | 회수 | 완주 | 충돌 | 신호위반 | 차로 오류 |
|---|---|---|---|---|---|
| a: 921~954 | 240 | 95.83% | 0.42% (1) | 3.75% (9) | 0 |
| b: 961~994 | 240 | 96.67% | 0% | 3.33% (8) | 0 |
| c: 1001~1034 | 240 | 96.67% | 0% | 3.33% (8) | 0 |
| d: 1041~1074 | 240 | 95.42% | 0% | 4.58% (11) | 0 |
| **합계** | **960** | **96.15%** | **0.10% (1)** | 3.75% (36) | 0 |
| 목표 | | 95% 이상 | 2% 이하 | | |

- 스냅샷은 60회 선별(a의 앞 4개 시드)로 골랐다. b·c는 후보 3개(401k·602k·802k) 중 고를 때 썼다. d는 고른 뒤 처음 돌린 시드다.
- 완주율의 95% 신뢰구간은 약 94.9~97.4%다.

같은 평가 방식(바른 시드, 아래 "평가 시드 버그" 참고)으로 잰 비교:

| 모델 | 회수 | 완주 | 충돌 | 신호위반 |
|---|---|---|---|---|
| **이 모델 (2단계, ft7 802k)** | 960 | **96.15%** | **0.10%** | 3.75% |
| 1단계 모델 (`rl_init_1400k.pt`) | 480 | 93.75% | 0.83% | 5.42% |
| 1단계 재현 (시드 42, 1,802k) | 480 | 92.50% | 1.04% | 6.46% |
| 루프 B 모델 (`yeonsan_loopB_RL`, 루프 B 경로) | 480 | 92.71% | 4.38% | 1.25% (막힘 8회) |

## 경로와 지도

| 항목 | 내용 |
|---|---|
| 지도 | `env/maps/yeonsan_area.net.xml` (대로 중심 넓은 지도, 루프 B와 같음) |
| 경로 | 연산 N 진입 → 월드컵대로 남쪽(`-26005519`) → 연수로 동쪽(`773756903`) → 쌍미천로 북쪽(`37401760#1`) → 고분로 서쪽(`794688376#1`) → 연산 E 진입 → W 진출 |
| 길이 | 19개 도로, 4,796 m (루프 부분은 약 3.7 km), 신호 7곳 |
| 설정 | `tools/rl/routes7.py`의 `ROUTE_SET=loopC` (경로 1개 = `ego_0`, `N_ROUTES=1`) |
| 에피소드 길이 | `MAX_STEPS=6000` |

## 만든 과정

### 1단계 (2026-11-23)

1. **기준선**: J8은 C를 처음 달려도 차로 문제가 없었다(큰길 좌회전 3번). 실패는 거의 신호위반이었다.
2. **DAgger 1차** (`dgC1`): J8이 운전(BETA 0), 4 × 20회.
3. **황색 신호 정답 보정** (`tools/rl/relabel_yellow.py`): 루프 B와 같은 규칙("정지선까지 4 m/s² 이하로 설 수 있으면 정지"). C 데이터는 황색 표본을 4배로 복사했다. J8_RL의 7경로 모방학습 데이터도 같은 규칙으로 보정했다.
4. **시작 신경망 고르기**: J8 구성원 8개 중 m6.
5. **PPO** (루프 B sr1과 같은 설정): 학습률 3e-5, 모방학습 손실 1 → 0.1, 신호위반 30·충돌 50·차로 오류 10, 위험 장면 재시작 30%. 1,400k 스냅샷을 골랐다.

### 2단계 (2026-11-24): 황색 신호 딜레마 고치기

1단계 모델의 실패는 거의 신호위반이었다. 실패 장면을 기록해 보니, 정지선 30~45 m 앞에서 황색이 켜졌을 때 2~3 m/s²로 충분히 설 수 있는데도 약하게 감속하다 적색에 넘었다(연산 N 진입, 쌍미천로, 연산 E 진입 등 여러 곳).

1. **DAgger 3차** (`dgC3`): 1단계 모델이 운전(BETA 0), 8 × 20회 = 160회, 28만 스텝. 황색 정답 보정, 황색 표본 4배.
2. **차로 정답 바꾸기** (`tools/rl/self_lane_label.py`): 긴 루프에서는 SUMO `getBestLanes`가 "더 멀리 이어지는 차로"를 가리켜, 지금 차로로도 갈 수 있는데 "왼쪽으로 바꿔라" 정답이 15% 생겼다. 차로 오류가 없는 1단계 모델의 선택으로 바꿨다(가감속 정답은 그대로) → `dgC3_ys.npz`.
3. **황색 장면 늘리기** (`tools/rl/yellow_only.py`): dgC3의 황색 장면 1,935개를 80배 복사해 모방학습 손실 데이터의 8%로 만들었다.
4. **설 수 있었던 황색 통과 감점** (`YELLOW_PEN=20`, `tools/rl/residual_train.py`): 황색을 처음 본 순간 4 m/s² 이하로 설 수 있었는데 황색에 정지선을 넘으면 −20. 관측만 쓰고 공용 환경(`env/sumo_env.py`)은 그대로다.
5. **PPO** (`lc_ft7`): 1단계 모델에서 시작, 학습률 1e-5, 모방학습 손실 가중치 0.3 고정, 100만 스텝. 802k 스냅샷을 골랐다.

```bash
python tools/rl/self_lane_label.py results/lc_m6_s41/snap_1400k.pt dgC3_y.npz dgC3_ys.npz
python tools/rl/yellow_only.py dgC3_ys.npz dgC3_yellow80.npz 80 281233
ROUTE_SET=loopC N_ROUTES=1 MAX_STEPS=6000 LANE_PLAN_RANGE=300 OBS_VISIBILITY=100 LANE_ARROWS=1 INTERNAL_OBS=1 \
INIT_MODEL=models/yeonsan_loopC_RL/rl_init_1400k.pt \
BC_DATA=bc_ft_y.npz,dgC1_y.npz,dgC1_y.npz,dgC3_ys.npz,dgC3_ys.npz,dgC3_yellow80.npz BC_W=0.3 BC_W_END=0.3 \
BC_STEPS=20 BC_BATCH=512 LR=1e-5 INIT_LOG_STD=-1.5 LANE_ENT=0 VALUE_WARMUP=20 DANGER_P=0.3 DANGER_RED=1 RING_EVERY=4 RING_N=6 \
YELLOW_PEN=20 REWARD_PATCH='{"collision_penalty": 50.0, "red_light_penalty": 30.0, "wrong_lane_penalty": 10.0, "ttc_penalty": 0.3}' \
N_ENVS=4 STEPS_PER_ENV=1024 TIMESTEPS=1000000 SNAP_EVERY=100000 SEED=57 \
python tools/rl/residual_train.py lc_ft7
```

`bc_ft_y.npz`는 `models/yeonsan_7routes_J8_RL/bc_ft.npz`를 `relabel_yellow.py`로 보정한 것이다(저장소에는 없음).

### 황색 장면에서 실제로 서는지

dgC3에서 "설 수 있었던 황색" 장면 중 정지선 20~50 m 앞, 8 m/s 이상인 152개에서 각 모델의 가감속 출력을 봤다(정답 평균 −0.46).

| 모델 | 출력 평균 | 거의 안 서는 비율 (> −0.1) |
|---|---|---|
| 1단계 모델 | −0.15 | 49% |
| ft2 (신호위반 페널티 60) | −0.18 | 41% |
| ft7 401k | −0.26 | 18% |
| **ft7 802k (이 모델)** | **−0.31** | **8%** |

## 2단계 실험 전체 (모두 1단계 모델에서 시작, 학습률 1e-5, 100만 스텝)

| 실행 | 바꾼 것 | 결과 |
|---|---|---|
| ft1 | 모방학습 가중치 0.1 고정 (이어 학습 기준선) | 60회 88~95%. 나아지지 않음 |
| ft2 | ft1 + 신호위반 페널티 60 | 901k 720회 93.75%, 1,003k 720회 94.03%. 기준 모델과 같음 |
| ft3 | ft2를 시드만 바꿔 재현 | ft2가 효과 없어 200k에서 중단 |
| ft4 | ft1 + dgC3 데이터 | 60회 88~95%, 400k에서 중단 |
| ft5 | ft4 + 모방학습 가중치 0.3 | 학습할수록 신호위반 증가(8%), 600k에서 중단 |
| ft6 | ft5 + 황색 장면 80배 | 신호위반 10%까지 증가, 400k에서 중단 |
| **ft7** | **ft6 + 설 수 있었던 황색 통과 감점 20** | **802k 960회 96.15% / 0.10% (채택)**, 401k 720회 95.28% / 0 |
| ft8 | ft7 + 학습률 3e-5 | 60회 88~97%, 500k에서 중단 |
| ft9 | ft7을 시드만 바꿔 재현 (시드 59) | 802k 480회(b·c) 92.29% / 2.29%. **재현 안 됨** |
| ft10 | ft7 + 감점 40 | 503k 480회(b·c) 93.54% / 0.21%, 602k 480회 87.92% / 0.21% |

## 평가 시드 버그 (1단계 숫자를 다시 잰 이유)

- `eval_routes.py`는 `SEED`를 건 뒤 `load_agent`로 모델을 불러왔다. 그런데 모델을 만드는 PPO 생성자가 `np.random.seed(0)`을 다시 걸어 `SEED`가 무시됐다.
  - main #16 이전: SUMO 난수는 시드와 관계없이 매번 같았고, 배경 차 배치만 OS 난수로 달라졌다. 그래서 "새 시드 확인"이 완전히 독립된 표본이 아니었고, "같은 시드로 다시 돌려도 결과가 다르다"는 현상도 이것 때문이었다.
  - main #16 이후: 교통 난수도 `np.random`을 따르게 바뀌어, 모든 시드가 같은 15회를 반복했다.
- 시드를 모델을 불러온 뒤에 걸도록 고쳤다(`dagger_collect.py`도 같은 문제를 같이 고침). 이제 같은 시드면 같은 결과가 나온다.
- 1단계 README의 480회 97.50% / 0.21%는 이 버그가 있던 평가다. 바른 평가로는 93.75% / 0.83%였다(`round1/eval_old_seedbug.json`은 예전 숫자).

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 채택 모델 (HybridActorCritic state_dict, `lc_ft7` 802k 스텝) |
| `rl_init_1400k.pt` | 2단계 시작점 = 1단계 모델 (`lc_m6_s41` 1,400k) |
| `il_init_m6.pt` | 1단계 시작점 (J8의 구성원 m6, 모방학습만) |
| `dgC1_y.npz` | C DAgger 1차 (J8 운전, 황색 보정, 황색 4배) |
| `dgC3_ys.npz` | C DAgger 3차 (1단계 모델 운전, 황색 보정, 황색 4배, 차로 정답 = 모델 선택) |
| `train_config.json`, `train.txt`, `train_log.csv` | 2단계 학습 설정·로그 (`lc_ft7`) |
| `round1/` | 1단계 학습 설정·로그, 예전(시드 버그) 평가 |
| `eval.json` | 960회 평가 결과 (시드 묶음별) |

## 평가

```bash
ROUTE_SET=loopC MAX_STEPS=6000 LANE_PLAN_RANGE=300 OBS_VISIBILITY=100 LANE_ARROWS=1 INTERNAL_OBS=1 OMP_NUM_THREADS=1 SEED=961 \
python tools/rl/eval_routes.py models/yeonsan_loopC_RL/model.pt <SUMO 작업 폴더> 15 out.json
```

- SUMO 작업 폴더는 `routes7.apply()`를 먼저 부른 뒤 `road_builder.build`로 만든다. 그렇지 않으면 기본 7경로 지도로 만들어진다.
  ```bash
  ROUTE_SET=loopC N_ROUTES=1 python -c "
  import sys; sys.path.insert(0, 'tools/rl'); import routes7; routes7.apply()
  from env.road_config import ROAD, EGO, TRAFFIC
  from env import road_builder; road_builder.build(ROAD, EGO, TRAFFIC, 'sumo_loopC')"
  ```
- `eval_routes.py`는 libsumo로 돈다 (`EVAL_TRACI=1`이면 TraCI). 숫자를 섞어 비교하지 않는다.
- 여러 평가를 동시에 돌릴 때는 `OMP_NUM_THREADS=1`을 준다.

## 주의

- **60회 선별은 낙관적이다.** 60회 98.3%였던 스냅샷이 240회에서 87~94%로 떨어진 경우가 여럿 있었다(ft10 602k: 60회 98.3% → 480회 87.92%). 후보는 반드시 선택에 안 쓴 시드로 확인한다.
- **재현성이 약하다.** 같은 설정을 시드만 바꾼 ft9는 목표에 못 미쳤다(802k 92.29% / 2.29%). 이 모델은 시드 57에서 나온 결과다.
- 남은 실패는 거의 신호위반(3.75%)이다. 충돌은 960회 중 1회다.
