"""경로별 평가: python tools/rl/eval_routes.py MODEL SUMO_DIR N_PER_ROUTE OUT.json
경로마다 N회 (결정적 정책, seed 777). 결과: 경로별 끝난 이유(완주/충돌/차로 갇힘/신호 위반/시간 초과) 횟수."""
import os, sys, json, time
import numpy as np, torch
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(BASE)
import routes7 as routes_cfg
routes_cfg.apply()
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from test import load_agent
from train import REWARD
model, sdir, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
env = SumoHighwayEnv(cfg_path=os.path.join(sdir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                     mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=False)
NAMES = {"ego_0": "S→W→회전교차로→W→E", "ego_1": "N→S", "ego_2": "NE→SW", "ego_3": "E→W", "ego_4": "S→N", "ego_5": "SW→N", "ego_6": "W→E"}
np.random.seed(777); torch.manual_seed(777)
agent = load_agent(model, env.observation_space.shape[0], 2)
res, t0 = {}, time.time()
for rid in (os.environ["ONLY"].split(",") if os.environ.get("ONLY") else [f"ego_{i}" for i in range(7)]):
    env.meta["ego_routes"] = [rid]
    ends = {}
    fails = 0
    for ep in range(n):
        try:
            obs, _ = env.reset()
        except RuntimeError as e:
            fails += 1
            continue
        done = False
        while not done:
            obs, rew, te, tr, info = env.step(agent.predict(obs, deterministic=True)); done = te or tr
        end = ("collision" if info["collided"] else "red_light" if info.get("red_light") and te else
               "wrong_lane" if info.get("wrong_lane") else "arrival" if info["arrived"] else "timeout")
        ends[end] = ends.get(end, 0) + 1
    res[NAMES[rid]] = {"episodes": sum(ends.values()), "insert_fail": fails, **ends}
    print(NAMES[rid], res[NAMES[rid]], f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
