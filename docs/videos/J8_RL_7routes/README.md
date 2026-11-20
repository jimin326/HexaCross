# 7경로 주행 영상: J8_RL (모방학습 J8 + PPO 미세조정 75만 스텝) — 2026-11-20

`models/yeonsan_7routes_J8_RL/model.pt`의 주행 영상입니다. 이 모델은 `results/ft1/ens_753k.pt`와 같은 파일입니다.
배경 교통량은 실측의 0.85배이고 버스가 들어 있습니다. 빨간 차(분홍 테두리)가 AV이고, 화면이 AV를 따라갑니다.

- 영상 1초는 시뮬레이션 8초입니다.
- 배경 교통이 매번 달라서 영상 한 편의 결과는 평가 통계와 다를 수 있습니다. 평가 통계는 교통 0.85, 840회 기준 완주 97.4%, 충돌 0.24%입니다(`models/yeonsan_7routes_J8_RL/README.md`).

| 경로 | 영상 | 이 영상의 결과 | 주행 시간 |
|---|---|---|---|
| S→W→회전교차로→W→E | [route0](route0_S-W-roundabout-W-E.mp4) | 완주 | 715초 |
| N→S | [route1](route1_N-S.mp4) | 완주 | 536초 |
| NE→SW | [route2](route2_NE-SW.mp4) | 완주 | 1037초 |
| E→W | [route3](route3_E-W.mp4) | 완주 | 310초 |
| S→N | [route4](route4_S-N.mp4) | 완주 | 149초 |
| SW→N | [route5](route5_SW-N.mp4) | 완주 | 461초 |
| W→E | [route6](route6_W-E.mp4) | 완주 | 156초 |

녹화 명령 (교통 0.85 지도 폴더를 `SUMO_DIR`로):

```bash
INTERNAL_OBS=1 LANE_ARROWS=1 OBS_VISIBILITY=100 MAX_STEPS=3000 LANE_PLAN_RANGE=300 xvfb-run -a python tools/rl/record_routes.py models/yeonsan_7routes_J8_RL/model.pt SUMO_DIR docs/videos/J8_RL_7routes ego_0,ego_1,ego_2,ego_3,ego_4,ego_5,ego_6
```
