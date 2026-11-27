# r9 (RL5): run_r8.sh와 같은 값, 경로(S, cd, P, SUMO_HOME)만 Windows용. 지도는 PR #31 반영본
Set-Location (Split-Path (Split-Path $PSScriptRoot))   # 저장소 루트
foreach ($f in 'bc_lby_t_s','dg12_bc_s','r3a_bc_s','r3b_bc_s') {
  if (-not (Test-Path "models/yeonsan_loopB_RL4/$f.npz")) { Write-Error "models/yeonsan_loopB_RL4/$f.npz 없음 - 학습 중단"; exit 1 }
}
$P = ".venv\Scripts\python.exe"
$env:SUMO_HOME = (& $P -c "import sumo; print(sumo.SUMO_HOME)")
$env:ROUTE_SET = "loopB"; $env:N_ROUTES = "1"; $env:MAX_STEPS = "6000"; $env:LANE_PLAN_RANGE = "300"
$env:ROAD_DETAILS = "driver_styles,taxi,short_stops"
$env:INIT_MODEL = "checkpoints/il/bcstop6.pt"; $env:INIT_LOG_STD = "-1.5"; $env:LR = "1.5e-5"; $env:VALUE_WARMUP = "20"; $env:LANE_ENT = "0"
$env:BC_DATA = "models/yeonsan_loopB_RL4/bc_lby_t_s.npz,models/yeonsan_loopB_RL4/dg12_bc_s.npz,models/yeonsan_loopB_RL4/r3a_bc_s.npz,models/yeonsan_loopB_RL4/r3b_bc_s.npz"
$env:BC_W = "1.0"; $env:BC_W_END = "0.2"; $env:BC_STEPS = "20"; $env:BC_BATCH = "512"
$env:DANGER_P = "0.3"; $env:DANGER_RED = "1"; $env:RING_EVERY = "4"; $env:RING_N = "6"
$env:REWARD_PATCH = '{"collision_penalty": 50.0, "red_light_penalty": 30.0, "wrong_lane_penalty": 10.0, "ttc_penalty": 0.3, "idle_penalty": 0.1, "idle_mode": "close", "idle_speed": 1.0, "idle_gap": 30.0, "idle_free_min": 2.0, "idle_free_min_veh": 1.5, "idle_free_min_line": 1.5, "idle_free_ramp": 8.0, "close_gap_min_speed": 3.0}'
$env:N_ENVS = "4"; $env:STEPS_PER_ENV = "1024"; $env:TIMESTEPS = "400000"; $env:SNAP_EVERY = "25000"; $env:SEED = "71"
& $P tools/rl/residual_train.py r9
