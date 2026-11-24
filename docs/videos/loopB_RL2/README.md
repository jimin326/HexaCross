# 최종 모델 RL2 주행 영상 (루프 경로 B)

`loopB_RL2_seed905.mp4` (56.3초, 1600×900)

- **왼쪽 = sumo-gui 실제 화면** (SUMO 1.27.1). 매 스텝(0.5초) 캡처한 2,765장. 화면이 AV를 따라가고, AV는 분홍 원 안의 빨간 차다. 건물은 분홍 면.
- 오른쪽 = 전체 경로 지도(지나온 길 진하게, 분홍 점 = 지금 위치)와 속도·제한속도·도로 이름·진행률.
- 모델 `models/yeonsan_loopB_RL2/model.pt`, 결정적 정책
- 학습 조건과 같은 환경: 순환 교통 `loop_flows`, `ROAD_DETAILS=driver_styles,taxi,short_stops`
- 평가 시드 905 (평가 `tools/rl/eval_parallel.py --seed 905`의 첫 에피소드). sumo-gui로 다시 돌린 주행도 평가와 같이 2,765스텝(23분 2.5초)에 완주
- 배속 (화면 오른쪽 위에 실제 시간 대비 배속 표시)
  - **×13.4**: 움직이면서 신호 교차로 30 m 안, 교차로 안, 꺾는 교차로 25 m 안, 앞차 12 m 안, 차로 변경, 1.5 m/s² 넘는 가감속 (앞뒤 3초 포함). 주행 시간의 34%
  - **×40.1** (위의 3배): 그냥 달리는 구간과 신호 대기로 서 있는 시간
  - 23분을 1분 안에 담으려면 평균 약 25배가 필요해서 실제 속도(×1)로는 보여 줄 수 없다. 배속이 바뀔 때는 1.5초(실제 시간)에 걸쳐 부드럽게 바꾼다

다시 만들기 (화면 없는 서버에서는 xvfb-run):

```bash
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops ZOOM=1100 \
xvfb-run -a -s "-screen 0 1920x1200x24" python tools/rl/record_loop_map.py gui models/yeonsan_loopB_RL2/model.pt \
    env/sumo_eval/build_loopB_scale_1.0_details-short_stops-taxi-driver_styles 905 gui905.pkl gframes
python tools/rl/record_loop_map.py compose gui905.pkl gframes \
    env/sumo_eval/build_loopB_scale_1.0_details-short_stops-taxi-driver_styles docs/videos/loopB_RL2/loopB_RL2_seed905.mp4
```

SUMO 폴더는 `eval_parallel.py`를 같은 환경 변수로 한 번 돌리면 생긴다 (`build_…` 폴더에 건물 파일이 있다).
