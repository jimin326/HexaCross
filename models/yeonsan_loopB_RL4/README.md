# yeonsan_loopB_RL4: 정지선·앞차에 알맞게 붙어 서는 루프 경로 B 모델 (2026-11-27)

`models/yeonsan_loopB_RL2`, `RL3`와 같은 환경·같은 목표(조건마다 완주 97% 이상, 충돌 1% 이하)에서,
**멈출 때 정지선·앞차와 너무 멀리 떨어져 서는 문제**를 고친 단일 신경망이다. RL2·RL3는 지우지 않고 그대로 두었다.
주행 중 규칙 개입은 없다. 규칙은 모방학습 정답 보정과 보상에만 쓴다.

## 무엇이 바뀌었나

| 모델 | 앞차 뒤 정지 간격 (범퍼, 중앙값) | 정지선 앞 정지 거리 (중앙값) |
|---|---|---|
| RL2 | 7.5 m | 6.8 m |
| **RL4** | **4.5 m** | **1.6 m** |

- 앞차 간격은 3~3.5 m를 먼저 시도했지만 대기 줄에서 충돌이 많아, 4.0 m 목표(실제 4.5 m)로 넓혔다.
  SUMO는 AV의 `minGap` 2.5 m보다 범퍼 간격이 좁아지면 닿지 않아도 충돌로 처리하므로 이보다 크게 줄이기 어렵다.
- 측정: `eval_parallel.py`가 멈춘 순간(속도 0.1 m/s 미만)의 앞차 범퍼 간격과 적·황색 정지선까지 거리를 모아 중앙값을 낸다 (`eval.json`의 `metrics.stop_gap_veh`, `stop_gap_line`).

## 왜 멀리 섰나 (모방학습 정답 버그)

DAgger 정답 가감속은 `env.step` 뒤 `getSpeedWithoutTraCI`로 만들었다. 그런데 AV를 `setSpeed`로 몰면 이 값은
SUMO 운전 모델이 고를 속도가 아니라 **학생이 명령한 속도**이고, SUMO 안전 속도보다 빠를 때만 줄어든다.
그래서 "더 다가가라"·"미리 줄여라" 정답이 거의 없었고(서 있는 AV의 정답은 앞차 간격과 관계없이 0), 학생이 멀리 서면 그대로 정답이 됐다.

`tools/rl/relabel_stop.py`로 기존 모방학습 데이터 4개의 정답을 보정했다 (학습 데이터만):

1. **천천히 다가가기**: 서 있는 앞차·적색 정지선 25 m 안, 2.5 m/s 미만이면 속도 = 0.5 × (남은 거리 − 목표), 최대 2 m/s. 목표 = 앞차 범퍼 4.0 m(`getLeader` 1.5 m), 정지선 1.5 m.
2. **확실히 멈추기**: 목표 0.6 m(앞차는 1.0 m) 안이면 −3 m/s². 비례 정답은 0 근처라 신경망 오차에 묻혀 기어가다 닿았다.
3. **뒤차 무작위 복사**: 정지 구간 표본에 뒤차 관측만 바꾼 복사본 4개. 뒤에서 빠른 차가 오면 출발해 앞차에 닿는 잘못된 연관을 끊는다.
4. **빠른 접근 감속**: 적색 정지선·느린 앞차까지 서는 데 1 m/s² 넘게 필요한데 정답이 더 약하면 그 1.15배로.
5. **황색**: 1.15배 여유 감속이 4.5 m/s² 안이면 서고, 아니면 통과.
6. **속도 관측 포화**: 관측 속도는 13.89 m/s에서 잘리므로, 포화 표본의 가속 정답은 −0.05로.

## 결과 (결정적 정책, `eval_parallel.py`, 조건마다 240회, 시드 905~)

| 조건 (`loop_flows` 켬) | 완주 | 충돌 | 신호위반 | 차로 오류 | 시간 초과 | 앞차 / 정지선 정지 |
|---|---|---|---|---|---|---|
| 운전자 성향 + 택시 + 잠깐 정차 (학습 조건) | 237 (98.75%) | 1 (0.42%) | 2 | 0 | 0 | 4.51 / 1.61 m |
| 도로 디테일 없음 | 239 (99.58%) | 1 (0.42%) | 0 | 0 | 0 | 4.50 / 1.60 m |
| 운전자 성향 | 240 (100%) | 0 (0%) | 0 | 0 | 0 | 4.52 / 1.60 m |
| 택시 + 잠깐 정차 | 238 (99.17%) | 2 (0.83%) | 0 | 0 | 0 | 4.53 / 1.61 m |
| **합계 960회** | **954 (99.38%)** | **4 (0.42%)** | 2 (0.21%) | 0 | 0 | |
| 목표 | 97% 이상 | 1% 이하 | | | | |

네 조건 모두 목표를 넘는다. 같은 평가에서 RL2는 948회(98.75%)·충돌 2회, RL3는 941회(98.02%)·충돌 7회였다.
조건별 원자료는 `eval/<조건>/` (eval.json, summary.csv, report.md).

## 만든 과정

1. **정답 보정** (위 1~6, `tools/rl/relabel_stop.py`): `bc_lby_t`, `dg12_bc`, `r3a_bc`, `r3b_bc` → `*_s.npz`
2. **모방학습 미세조정**: RL2에서 보정 데이터로 3,000번 (학습률 5e-5) → `checkpoints/il/bcstop6.pt`.
   빠른 시험(`tools/rl/early_check.py`, 이전 실패 시드 20개 × 4조건 100초)에서 80회 중 실패 1, 40회 평가 완주 40·충돌 0.
3. **PPO r8** (`tools/rl/residual_train.py`, 시작 bcstop6)
   - 학습률 1.5e-5, 4,096 스텝마다 갱신, 처음 20번 갱신은 가치 함수만 학습(정책 고정)
   - 모방학습 손실: 보정 데이터 4개, 가중치 1.0 → 0.2
   - 보상: 충돌 −50, 신호위반 −30, 차로 오류 −10, TTC 0.3, 그리고 **가까이 서기 감점** (`idle_mode "close"`):
     앞차·적황색 정지선이 30 m 안에 있을 때 앞차 `getLeader` 1.5 m·정지선 1.5 m보다 넓게 비우고 서면 8 m에 걸쳐 스텝당 최대 0.1 감점, 근접 감점은 3 m/s 이상일 때만
   - 위험 장면 30% 재시작, `SEED=71`
4. **평가**: 2.5만 스텝마다 120회 → 102k(정책 갱신 5번, 모방학습 시작점과 가중치 상대 차 1.2%) 완주 118·충돌 0이 가장 좋음.
   126k·151k는 충돌 6으로 나빠져 학습을 멈췄다. 102k를 4조건 × 240회로 확인하고 채택.

설정 전체는 `r8_config.json`, 실행 스크립트는 `run_r8.sh`, 학습 로그는 `r8_train.txt`·`r8_train_log.csv`.
스냅샷은 `checkpoints/runs/r8/` (28k·53k = 정책 고정 구간이라 bcstop6과 같음, 102k = 이 모델, 126k, 151k).

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 채택 모델 (HybridActorCritic state_dict, r8 102k 스텝) |
| `r8_config.json` | 학습 설정 (환경 변수, 하이퍼파라미터, 보상) |
| `run_r8.sh` | PPO 실행 스크립트 |
| `r8_train.txt`, `r8_train_log.csv` | 학습 로그 |
| `eval/` | 4조건 확인 평가 결과 |
| `bc_lby_t_s.npz` | 모방학습 데이터: 예전 루프 DAgger + 황색·회전 보정 (1,380,903 표본) |
| `dg12_bc_s.npz` | 모방학습 데이터: 고친 환경 DAgger 1·2차 (1,645,536 표본) |
| `r3a_bc_s.npz` | 모방학습 데이터: 시드 버그를 고친 DAgger, 학생 bcfix (818,832 표본) |
| `r3b_bc_s.npz` | 모방학습 데이터: 시드 버그를 고친 DAgger, 학생 RL2 (908,467 표본) |

### 모방학습(전문가) 데이터

위 `*_s.npz` 4개는 bcstop6 미세조정과 PPO r8의 모방학습 손실에 쓴 데이터다 (합계 4,753,738 표본).
DAgger로 모은 SUMO 전문가 정답에 황색·회전 보정을 한 원본(`bc_lby_t`, `dg12_bc`, `r3a_bc`, `r3b_bc`)을
`tools/rl/relabel_stop.py`로 한 번 더 보정한 결과다 (위 "왜 멀리 섰나"의 1~6. 바뀐 표본은 복제해 원본보다 크다).

| 키 | 모양 | 내용 |
|---|---|---|
| `obs` | (N, 56) float | 관측 (DATA_FORMAT.md 4절) |
| `accel` | (N,) float | 정답 가감속 [-1, 1] (−1 = −5 m/s², +1 = +3 m/s²) |
| `lane_cls` | (N,) int | 정답 차로 선택 (0 왼쪽 · 1 유지 · 2 오른쪽, `HybridActorCritic.LANE_CMDS` 순서) |

모방학습만 다시 하려면 (bcstop6과 같은 설정):

```bash
STEPS=3000 LR=5e-5 python tools/rl/bc_finetune.py models/yeonsan_loopB_RL2/model.pt results/bcstop6/model.pt \
    models/yeonsan_loopB_RL4/bc_lby_t_s.npz models/yeonsan_loopB_RL4/dg12_bc_s.npz \
    models/yeonsan_loopB_RL4/r3a_bc_s.npz models/yeonsan_loopB_RL4/r3b_bc_s.npz
```

## 평가

```bash
pip install -r requirements.txt   # SUMO 1.27.1
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python tools/rl/eval_parallel.py models/yeonsan_loopB_RL4/model.pt results/eval_RL4 --n 240 --workers 4 --chunk 15
```
