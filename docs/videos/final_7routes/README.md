# 7경로 주행 영상: 최종 모델 (모방학습 + DAgger, 8개 앙상블) — 2026-11-19

`models/yeonsan_7routes_final/model.pt`의 주행 영상입니다. 빨간 차(분홍 테두리)가 AV이고, 화면이 AV를 따라갑니다.

- 영상 1초는 시뮬레이션 8초입니다.
- 배경 교통이 매번 달라서 영상 한 편의 결과는 평가 통계와 다를 수 있습니다. 평가 통계는 모델 README에 있고, 처음 보는 교통 840회 기준 완주 97.9%, 충돌 1.79%입니다.

| 경로 | 영상 | 이 영상의 결과 | 주행 시간 |
|---|---|---|---|
| S→W→회전교차로→W→E | [route0](route0_S-W-roundabout-W-E.mp4) | 완주 | 786초 |
| N→S | [route1](route1_N-S.mp4) | 완주 | 295초 |
| NE→SW | [route2](route2_NE-SW.mp4) | 완주 | 468초 |
| E→W | [route3](route3_E-W.mp4) | 완주 | 331초 |
| S→N | [route4](route4_S-N.mp4) | 완주 | 312초 |
| SW→N | [route5](route5_SW-N.mp4) | 완주 | 610초 |
| W→E | [route6](route6_W-E.mp4) | 완주 | 300초 |

녹화 명령:

```bash
xvfb-run -a python tools/rl/record_routes.py models/yeonsan_7routes_final/model.pt env/sumo_expert/w0 docs/videos/final_7routes ego_0,ego_1,ego_2,ego_3,ego_4,ego_5,ego_6
```

녹화 시드는 3입니다(기본값).
