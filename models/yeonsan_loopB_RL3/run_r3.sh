#!/bin/bash
# RL3: 시드 버그를 고친 dagger_collect로 DAgger 2번(학생 bcfix, RL2) → 보정 → PPO r3 (bcfix에서, YELLOW_PEN=20)
S=${S:-$(cd "$(dirname "$0")/../../.." && pwd)}  # 작업 폴더 (저장소 ymerge 와 venv127 이 있는 곳)
cd $S/ymerge
P=$S/venv127/bin/python
export SUMO_HOME=$S/venv127/lib/python3.13/site-packages/sumo ROUTE_SET=loopB N_ROUTES=1 MAX_STEPS=6000 LANE_PLAN_RANGE=300
export ROAD_DETAILS=driver_styles,taxi,short_stops
log() { echo "$(TZ=Asia/Seoul date +%H:%M) $*" >> $S/r3_prep.log; }
for job in "r3a results/bcfix/model.pt 91" "r3b models/yeonsan_loopB_RL2/model.pt 92"; do
  set -- $job
  if [ ! -f data/lf/$1.npz ]; then
    log "DAgger $1 (학생 $2, SEED $3) 시작"
    N_WORKERS=4 SEED=$3 BETA=0 LANE_LABEL=best $P tools/rl/dagger_collect.py $2 data/lf/$1 40 > $S/$1.log 2>&1
    grep -v proj.db $S/$1.log | grep -E "^steps|ego_0|arrival" | tail -3 >> $S/r3_prep.log
    [ -f data/lf/$1.npz ] || { log "DAGGER $1 FAILED"; exit 1; }
  fi
  $P tools/rl/relabel_yellow.py data/lf/$1.npz data/lf/$1y.npz 4 >> $S/r3_prep.log 2>&1 &&
  $P tools/rl/relabel_turn.py data/lf/$1y.npz data/lf/$1yt.npz 3 >> $S/r3_prep.log 2>&1 &&
  $P tools/rl/make_bcdata.py data/lf/$1_bc.npz data/lf/$1yt.npz >> $S/r3_prep.log 2>&1 || { log "PREP $1 FAILED"; exit 1; }
done
export INIT_MODEL=results/bcfix/model.pt INIT_LOG_STD=-1.5 LR=3e-5 VALUE_WARMUP=20 LANE_ENT=0
export BC_DATA=data/lf/bc_lby_t.npz,data/lf/dg12_bc.npz,data/lf/r3a_bc.npz,data/lf/r3b_bc.npz BC_W=1.0 BC_W_END=0.2 BC_STEPS=20 BC_BATCH=512
export DANGER_P=0.3 DANGER_RED=1 RING_EVERY=4 RING_N=6 YELLOW_PEN=20
export REWARD_PATCH='{"collision_penalty": 50.0, "red_light_penalty": 30.0, "wrong_lane_penalty": 10.0, "ttc_penalty": 0.3}'
export N_ENVS=4 STEPS_PER_ENV=1024 TIMESTEPS=2000000 SNAP_EVERY=200000 SEED=63
log "start r3"
exec $P tools/rl/residual_train.py r3
