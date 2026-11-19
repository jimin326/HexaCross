"""전문가 데이터 수집: SUMO 운전 모델이 AV(ego)를 직접 운전하고, 그때의 AV 관측(공지 준수 관측)과
SUMO가 실제로 한 행동(가감속, 차로 변경)을 기록한다. 모방학습(BC)용.

usage: python tools/rl/expert_collect.py OUT_PREFIX EPISODES_PER_WORKER
환경 변수: N_WORKERS(4), SEED(0) + routes7.py가 읽는 것 (MAX_STEPS ...). 관측 설정은 env/mdp_config.py
출력: OUT_PREFIX.npz (obs, accel, lane_cls, episode, route) + OUT_PREFIX.json (경로별 결과)
lane_cls: 0 = 왼쪽(+1), 1 = 유지, 2 = 오른쪽(-1)  (HybridActorCritic.LANE_CMDS 순서)"""
import os, sys, json, shutil
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, setup as _setup, reward as _reward


def worker(q, repo, sdir, seed, n_ep):
    _setup(repo)
    import traci
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    np.random.seed(seed)
    import random
    random.seed(seed)
    env = SumoHighwayEnv(cfg_path=os.path.join(sdir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=_reward(), traffic=TRAFFIC, gui=False)
    real_set_speed = traci.vehicle.setSpeed
    routes = [f"ego_{i}" for i in range(7)]
    O, A, L, E, R = [], [], [], [], []
    res = {}
    dt = env.step_length
    for ep in range(n_ep):
        rid = routes[(seed + ep) % 7]
        env.meta["ego_routes"] = [rid]
        for k in range(10):
            try:
                obs, _ = env.reset()
                break
            except RuntimeError:
                if k == 9:
                    raise
        ego = env.ego_id
        traci.vehicle.setImperfection(ego, 0.0)       # 전문가는 운전자 실수(sigma) 없이
        real_set_speed(ego, -1)                       # 속도 고정 해제 → SUMO 차량 추종 모델이 운전
        traci.vehicle.setSpeedMode(ego, 31)           # 안전거리·신호·교차로 우선순위 모두 지킴
        traci.vehicle.setLaneChangeMode(ego, 1621)    # SUMO 기본 차로 변경 (경로·속도·협조·오른쪽 유지)
        env._apply_lane_change = lambda lc: True      # 정책 차로 명령 끔
        traci.vehicle.setSpeed = lambda vid, v: None if vid == ego else real_set_speed(vid, v)
        done, end = False, "other"
        while not done:
            v0 = traci.vehicle.getSpeed(ego); road0 = traci.vehicle.getRoadID(ego); lane0 = traci.vehicle.getLaneIndex(ego)
            o_t = obs
            obs, r, te, tr, info = env.step(np.zeros(2, dtype=np.float32))
            done = te or tr
            alive = ego in traci.vehicle.getIDList()
            if alive:
                v1 = traci.vehicle.getSpeed(ego)
                acc = (v1 - v0) / dt
                a = acc / env.max_accel if acc >= 0 else acc / env.max_decel
                lc = 1
                if traci.vehicle.getRoadID(ego) == road0:
                    d = traci.vehicle.getLaneIndex(ego) - lane0
                    lc = 0 if d > 0 else (2 if d < 0 else 1)
                O.append(o_t); A.append(float(np.clip(a, -1, 1))); L.append(lc); E.append(seed * 100000 + ep); R.append(int(rid[4:]))
            if done:
                end = ("arrival" if info.get("arrived") else "collision" if info.get("collided") else "red_light" if info.get("red_light")
                       else "wrong_lane" if info.get("wrong_lane") else "timeout" if tr else "other")
        traci.vehicle.setSpeed = real_set_speed
        res.setdefault(rid, {}).setdefault(end, 0)
        res[rid][end] += 1
    env.close()
    q.put((np.asarray(O, np.float32), np.asarray(A, np.float32), np.asarray(L, np.int64), np.asarray(E), np.asarray(R), res))


def main():
    out, n_ep = sys.argv[1], int(sys.argv[2])
    repo = BASE
    _setup(repo)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env import road_builder
    nw = int(os.environ.get("N_WORKERS", "4"))
    seed = int(os.environ.get("SEED", "0"))
    base = os.path.join(BASE, "env", "sumo_expert")
    if not os.path.exists(os.path.join(base, "w0", "highway.sumocfg")):
        road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(base, "w0"))
    for i in range(1, nw):
        if not os.path.exists(os.path.join(base, f"w{i}", "highway.sumocfg")):
            shutil.copytree(os.path.join(base, "w0"), os.path.join(base, f"w{i}"), dirs_exist_ok=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(q, repo, os.path.join(base, f"w{i}"), seed * 10 + i, n_ep)) for i in range(nw)]
    for p in ps:
        p.start()
    parts = [q.get() for _ in ps]
    for p in ps:
        p.join()
    O, A, L, E, R = (np.concatenate([p[k] for p in parts]) for k in range(5))
    res = {}
    for p in parts:
        for rid, d in p[5].items():
            for k, v in d.items():
                res.setdefault(rid, {}).setdefault(k, 0)
                res[rid][k] += v
    np.savez_compressed(out + ".npz", obs=O, accel=A, lane_cls=L, episode=E, route=R)
    json.dump(res, open(out + ".json", "w"), ensure_ascii=False, indent=1)
    tot = {}
    for d in res.values():
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + v
    print("steps", len(O), "lane_cls", np.bincount(L, minlength=3).tolist(), "results", tot, flush=True)
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
