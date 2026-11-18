"""PR #2 도로 검증용 스크립트 (레포 루트에서 실행).

AV 대신 SUMO 기본 운전자 모델이 ego 경로를 운전하게 해서
도로 자체가 정상인지(완주, teleport, 진입 대기) 확인한다.

    python docs/research/check_route.py 8
"""
import json, os, sys
sys.path.insert(0, os.getcwd())  # 레포 루트에서 실행하므로 env 패키지를 찾게 함
import numpy as np
import sumolib, traci
from env import road_builder
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from env.sumo_env import SumoHighwayEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = road_builder.build(ROAD, EGO, TRAFFIC, os.path.join("env", "sumo"))
env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                     action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
res = []
for ep in range(N):
    env.reset(seed=100 + ep)
    ego = env.ego_id
    traci.vehicle.setSpeed(ego, -1)
    traci.vehicle.setSpeedMode(ego, 31)
    traci.vehicle.setLaneChangeMode(ego, 0b011001010101)
    t = stopped = tele = maxveh = pend = 0
    arrived = False
    while t < 2400:
        traci.simulationStep(); t += 1
        tele += traci.simulation.getStartingTeleportNumber()
        if ego in traci.simulation.getArrivedIDList():
            arrived = True; break
        if ego not in traci.vehicle.getIDList():
            break
        if traci.vehicle.getSpeed(ego) < 0.1:
            stopped += 1
        if t % 40 == 0:
            maxveh = max(maxveh, traci.vehicle.getIDCount())
            pend = max(pend, len(traci.simulation.getPendingVehicles()))
    r = dict(ep=ep, arrived=arrived, time_s=t * 0.5, stop_s=stopped * 0.5,
             teleports=tele, max_veh=maxveh, max_pending=pend)
    print(r, flush=True); res.append(r)
env.close()
print("arrived", sum(r["arrived"] for r in res), "/", N,
      "| teleports", sum(r["teleports"] for r in res),
      "| mean time", round(float(np.mean([r["time_s"] for r in res])), 1), "s",
      "| max pending", max(r["max_pending"] for r in res))
