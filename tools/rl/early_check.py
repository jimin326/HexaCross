# usage: python tools/rl/early_check.py MODEL NSTEPS SEED...  — 출발 후 NSTEPS 스텝만 돌려 충돌·신호위반 확인 (환경변수 SUMO_DIR, ROAD_DETAILS). r7 53k 실패가 몰린 출발 30~45초 지점 빠른 시험용
import os, sys
BASE=os.getcwd(); sys.path.insert(0, os.path.join(BASE,"tools","rl")); sys.path.insert(0,BASE)
import common; common.setup(BASE)
import numpy as np, torch; torch.set_num_threads(1)
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from test import load_agent
model, ns = sys.argv[1], int(sys.argv[2])
env = SumoHighwayEnv(cfg_path=os.path.join(os.environ["SUMO_DIR"], "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION, reward=common.reward(), traffic=TRAFFIC, gui=False)
agent = load_agent(model, env.observation_space.shape[0], 2)
for seed in map(int, sys.argv[3:]):
    np.random.seed(seed); torch.manual_seed(seed)
    obs,_ = env.reset(); done=False; t=0; info={}
    while not done and t < ns:
        obs,_,te,tr,info = env.step(agent.predict(obs, deterministic=True)); done=te or tr; t+=1
    r = 'collision' if info.get('collided') else 'red_light' if info.get('red_light') else 'wrong_lane' if info.get('wrong_lane') else 'ok'
    print(os.environ.get("ROAD_DETAILS"), seed, r, t, flush=True)
env.close()
