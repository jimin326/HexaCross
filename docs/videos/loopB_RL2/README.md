# 최종 모델 RL2 주행 영상 (루프 경로 B)

`loopB_RL2_seed905.mp4` (56.4초, 1280×720)

- 모델 `models/yeonsan_loopB_RL2/model.pt`, 결정적 정책
- 학습 조건과 같은 환경: SUMO 1.27.1, 순환 교통 `loop_flows`, `ROAD_DETAILS=driver_styles,taxi,short_stops`
- 평가 시드 905 (평가 `tools/rl/eval_parallel.py --seed 905`의 첫 에피소드와 같은 주행). 결과: 완주, 2,765스텝(23분 2.5초)
- 왼쪽: AV를 따라가는 확대 화면(가로 170 m, 북쪽 위). AV 분홍, 일반 차 회색, 버스 주황, 택시 노랑, 정지선 색 = 신호 색, 갈색 = 버스 전용 차로
- 오른쪽: 전체 경로 지도(지나온 길 진하게, 분홍 점 = 현재 위치, 분홍 네모 = 왼쪽 화면 범위)와 속도·도로 이름·진행률
- 배속 (화면 오른쪽 위에 실제 시간 대비 배속 표시)
  - **×12.8**: 움직이면서 신호 교차로 30 m 안, 교차로 안, 꺾는 교차로 25 m 안, 앞차 12 m 안, 차로 변경, 1.5 m/s² 넘는 가감속 (앞뒤 3초 포함). 전체 주행 시간의 31%
  - **×38.4** (위의 3배): 그냥 달리는 구간과 신호 대기처럼 서 있는 시간
  - 배속이 바뀔 때는 1.5초(실제 시간)에 걸쳐 부드럽게 바꾼다

다시 만들기:

```bash
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python tools/rl/record_loop_map.py log models/yeonsan_loopB_RL2/model.pt env/sumo_eval/loopB_scale_1.0_details-short_stops-taxi-driver_styles 905 rl2_905.pkl
python tools/rl/record_loop_map.py render rl2_905.pkl env/sumo_eval/loopB_scale_1.0_details-short_stops-taxi-driver_styles docs/videos/loopB_RL2/loopB_RL2_seed905.mp4
```

SUMO 폴더는 `eval_parallel.py`를 같은 환경 변수로 한 번 돌리면 생긴다.
