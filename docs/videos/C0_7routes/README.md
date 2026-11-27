# 7경로 주행 영상 — 최종 모델 C0 (프로젝트 규칙 준수 설정)

`models/yeonsan_7routes_C0/model.pt`(PPO 강화학습)의 주행 영상입니다. 빨간 차(분홍 테두리)가 AV이고 화면이 AV를 따라갑니다.
영상 1초 = 시뮬레이션 8초. 배경 교통이 매번 달라서, 영상 한 편의 결과는 평가 통계(모델 README)와 다를 수 있습니다.

| 길 | 영상 | 이 영상의 결과 | 주행 시간 |
|---|---|---|---|
| S→W→회전교차로→W→E | [route0](route0_S-W-roundabout-W-E.mp4) | 차로 갇힘 | 92초 |
| N→S | [route1](route1_N-S.mp4) | 완주 | 322초 |
| NE→SW | [route2](route2_NE-SW.mp4) | 완주 | 590초 |
| E→W | [route3](route3_E-W.mp4) | 신호 위반 | 184초 |
| S→N | [route4](route4_S-N.mp4) | 완주 | 284초 |
| SW→N | [route5](route5_SW-N.mp4) | 충돌 | 276초 |
| W→E | [route6](route6_W-E.mp4) | 완주 | 389초 |

녹화: `tools/rl/record_routes.py` (sumo-gui 화면을 매 스텝 캡처, 1280×720).
