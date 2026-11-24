# yeonsan_loopB_RL2: 루프 경로 B, 조교 답변 반영 환경에서 다시 학습 (2026-11-23)

`models/yeonsan_loopB_RL`(이전 최종 모델)과 같은 루프 경로 B를 **고친 환경**에서 다시 학습한 단일 신경망이다.
이전 모델은 지우지 않고 그대로 두었다.

## 이전 모델과 달라진 환경

| 항목 | 이전 (`yeonsan_loopB_RL`) | 이 모델 |
|---|---|---|
| SUMO | 1.28 | **1.27.1** (requirements.txt, libsumo 1.27.1) |
| 배경 교통 | 기본 | **순환 흐름 `loop_flows` 켬 (+4,550대/시)** |
| 도로 디테일 | 끔 | **`ROAD_DETAILS=driver_styles,taxi,short_stops`** (운전자 성향, 택시, 잠깐 정차) |
| 가속 명령 속도 상한 | 제한속도에서 자름 (조교: 강제라 수정 필요) | **차량 성능 최고속도(100 km/h)까지만 자름**, 제한속도 초과는 보상에서 감점 (`overspeed_penalty` 0.5) |
| AV 차종 | speedFactor 기본 | speedFactor 1, speedDev 0 (전문가 정답이 제한속도를 따름) |

## 결과 (결정적 정책, 조건마다 240회, 시드 905~)

| 조건 (`loop_flows` 켬) | 완주 | 충돌 | 신호위반 | 차로 오류 | 시간 초과 |
|---|---|---|---|---|---|
| 운전자 성향 + 택시 + 잠깐 정차 (학습 조건) | 236 (98.3%) | 0 (0%) | 4 (1.7%) | 0 | 0 |
| 도로 디테일 없음 | 237 (98.8%) | 0 (0%) | 3 (1.3%) | 0 | 0 |
| 운전자 성향 | 238 (99.2%) | 1 (0.4%) | 1 (0.4%) | 0 | 0 |
| 택시 + 잠깐 정차 | 237 (98.8%) | 1 (0.4%) | 2 (0.8%) | 0 | 0 |
| **합계 960회** | **948 (98.75%)** | **2 (0.21%)** | 10 (1.04%) | 0 | 0 |
| 목표 | 97% 이상 | 1% 이하 | | | |

## 만든 과정

1. **속도 상한을 고친 뒤 이전 모델 그대로 PPO (nc1) → 실패**
   - 이전 모델은 제한속도에서도 가속을 계속 냈다 (예전 환경은 속도를 잘라서 문제가 안 됐음). DAgger 160회 중 완주 12회.
   - 원인: 모방학습 손실이 다음 속도를 제한속도에서 잘라 비교해서, 제한속도에서 가속해도 손실이 0이었다. `tools/rl/bc_train.py`의 `VNEXT_MAX`를 차량 최고속도로 바꿨다.
2. **모방학습 손실만으로 미세조정 → `il_init_bcfix.pt`** (`tools/rl/bc_finetune.py`)
   - 제한속도에서 평균 가속 출력 +0.057 → +0.008. 고친 환경 60회: 완주 95.0%, 충돌 3.3%.
3. **DAgger (학생 bcfix, 160회) + 정답 보정**
   - 황색 신호 보정 (`relabel_yellow.py`), 신호 없는 회전·합류 접근 감속 보정 (`relabel_turn.py`, 관측만 씀), 차로 변경 보강.
4. **PPO nc2** (`tools/rl/residual_train.py`, 시작 bcfix)
   - 모방학습 손실 1.0 → 0.2, 학습률 3e-5, 충돌·신호위반 직전 장면 30% 재시작.
   - 20만 스텝마다 120회 평가: 200k 97.5%/0.8%, **401k 97.5%/0%**, 602k 98.3%/0.8%.
   - 200k는 240회에서 충돌 1.25%로 미달. 401k를 4조건 × 240회로 확인하고 채택했다. 목표를 넘겨 676k에서 학습을 멈췄다.

자세한 기록은 저장소 루트의 `진행상황.md`에 있다. 스냅샷은 `checkpoints/runs/nc2/`.

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 채택 모델 (HybridActorCritic state_dict, nc2 401k 스텝) |
| `il_init_bcfix.pt` | 강화학습 시작점 (이전 최종 모델을 모방학습 손실로 미세조정) |
| `nc2_config.json` | 학습 설정 (환경 변수, 하이퍼파라미터, 보상) |
| `nc2_train.txt`, `nc2_train_log.csv` | 학습 로그 |

## 평가

```bash
pip install -r requirements.txt   # SUMO 1.27.1
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python tools/rl/eval_parallel.py models/yeonsan_loopB_RL2/model.pt results/eval_RL2 --n 240 --workers 4 --chunk 15
```
