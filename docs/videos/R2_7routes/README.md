# 7경로 주행 영상 — 모델 R2

연산교차로 7경로를 섞어 PPO로 이어 학습한 모델 `ys_R2`(G1 → F0 → R2, main 병합 전 환경)의 주행 영상입니다.
AV는 화면 가운데 빨간 차(분홍 테두리)이고, 화면이 AV를 따라갑니다. 영상 1초 = 시뮬레이션 8초 (0.5초 스텝 16장/초).
에피소드마다 결과가 다를 수 있습니다 (배경 교통이 무작위). 경로별 평가 통계는 보고서 페이지 참고.

| 경로 | 영상 | 이 영상의 결과 | 주행 시간 |
|---|---|---|---|
| S→W→회전교차로→W→E | [route0_S-W-roundabout-W-E.mp4](route0_S-W-roundabout-W-E.mp4) | 충돌 | 20초 |
| N→S | [route1_N-S.mp4](route1_N-S.mp4) | 완주 | 478초 |
| NE→SW | [route2_NE-SW.mp4](route2_NE-SW.mp4) | 완주 | 961초 |
| E→W | [route3_E-W.mp4](route3_E-W.mp4) | 차로 갇힘 | 449초 |
| S→N | [route4_S-N.mp4](route4_S-N.mp4) | 차로 갇힘 | 239초 |
| SW→N | [route5_SW-N.mp4](route5_SW-N.mp4) | 충돌 | 2초 |
| W→E | [route6_W-E.mp4](route6_W-E.mp4) | 완주 | 382초 |

- 녹화: `sumo-gui` 화면을 매 스텝 캡처해 ffmpeg로 묶음 (1280×720).
- W→E는 출발 위치를 도로망 끝 엣지(91 m, 배경 차량 투입 지점)에서 그다음 엣지로 옮긴 뒤 녹화했습니다.
