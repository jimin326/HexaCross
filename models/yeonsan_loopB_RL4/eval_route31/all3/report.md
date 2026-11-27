# 주행 평가 결과

| 모델 | 평가 횟수 | 완주 | 충돌 | 신호 위반 | 차로 갇힘 | 막힘(순간이동) | 시간 초과 | 충돌률 기준(5% 미만) |
|---|---:|---|---|---|---|---|---|---|
| rl4_r31_all3 | 240 | 238 (99.2%, 97.0%~99.8%) | 2 (0.8%, 0.2%~3.0%) | 0 (0.0%, 0.0%~1.6%) | 0 (0.0%, 0.0%~1.6%) | 0 (0.0%, 0.0%~1.6%) | 0 (0.0%, 0.0%~1.6%) | 통과 (신뢰구간 상한 3.0% < 5%) |

괄호 안: 비율, 95% 신뢰구간 (Wilson). 투입 실패(AV가 도로에 못 들어감)는 평가 횟수에서 뺐다.

## 주행 품질

| 모델 | 완주 시 이동 시간 (s) | 평균 속도 (m/s) | 정지 시간 (s/주행) | 최소 차간거리 평균 (m) |
|---|---:|---:|---:|---:|
| rl4_r31_all3 | 1451.8 | 5.0 | 600.3 | 0.7 |

## rl4_r31_all3: 경로별

조건: model=models/yeonsan_loopB_RL4/model.pt, route_set=loopB, road_details=short_stops+taxi+driver_styles, max_steps=6000, n_per_route=240, seed_start=905, traffic_scale=1.0, sumo_dir=env/sumo_eval/loopB_scale_1.0_details-short_stops-taxi-driver_styles, commit=0148913, git_dirty=False, date=2026-11-27 15:07, minutes=15.4, libsumo=True

| 경로 | 평가 | 완주 | 충돌 | 신호 위반 | 차로 갇힘 | 막힘(순간이동) | 시간 초과 | 투입 실패 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 루프B N→S→회전교차로→E→W | 240 | 238 (99.2%) | 2 (0.8%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 |

## 그래프

![fig_compare.png](fig_compare.png)
![fig_routes_rl4_r31_all3.png](fig_routes_rl4_r31_all3.png)
