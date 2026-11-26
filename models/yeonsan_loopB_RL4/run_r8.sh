#!/bin/bash
# RL4 (r8): bcstop6(빠른 접근 감속·황색·속도 포화·뒤차 복사 보정)에서 PPO. 학습률 1.5e-5, 2.5만 스텝마다 스냅샷
S=/tmp/claude-0/-home-user-HexaCross/24027832-5621-5d90-929a-fa67f688310f/scratchpad
cd $S/ymerge
P=$S/venv127/bin/python
export SUMO_HOME=$S/venv127/lib/python3.13/site-packages/sumo ROUTE_SET=loopB N_ROUTES=1 MAX_STEPS=6000 LANE_PLAN_RANGE=300
export ROAD_DETAILS=driver_styles,taxi,short_stops
export INIT_MODEL=checkpoints/il/bcstop6.pt INIT_LOG_STD=-1.5 LR=1.5e-5 VALUE_WARMUP=20 LANE_ENT=0
export BC_DATA=data/lf/bc_lby_t_s.npz,data/lf/dg12_bc_s.npz,data/lf/r3a_bc_s.npz,data/lf/r3b_bc_s.npz BC_W=1.0 BC_W_END=0.2 BC_STEPS=20 BC_BATCH=512
export DANGER_P=0.3 DANGER_RED=1 RING_EVERY=4 RING_N=6
# 멈춤 거리 목표: 앞차 범퍼 4.0 m (3.25·3.5 m는 대기 줄 충돌), 정지선 1.5 m.
# 앞차는 범퍼 4.0 m(getLeader 1.5 m), 정지선은 1.5 m 넘게 비우고 서면 8 m에 걸쳐 스텝당 최대 0.1 감점
# 천천히 다가가는 것은 막지 않도록 근접 감점은 3 m/s 이상일 때만
export REWARD_PATCH='{"collision_penalty": 50.0, "red_light_penalty": 30.0, "wrong_lane_penalty": 10.0, "ttc_penalty": 0.3, "idle_penalty": 0.1, "idle_mode": "close", "idle_speed": 1.0, "idle_gap": 30.0, "idle_free_min": 2.0, "idle_free_min_veh": 1.5, "idle_free_min_line": 1.5, "idle_free_ramp": 8.0, "close_gap_min_speed": 3.0}'
export N_ENVS=4 STEPS_PER_ENV=1024 TIMESTEPS=400000 SNAP_EVERY=25000 SEED=71
exec $P tools/rl/residual_train.py r8
