"""DAgger 수집: 학생 정책(BC 모델)이 AV를 운전하고, 매 스텝 "SUMO 운전 모델이라면 어떻게 했을지"를 정답으로 기록한다.
  가감속 정답 = (getSpeedWithoutTraCI - 직전 속도) / dt   (SUMO 차량 추종 모델이 고른 속도)
  차로 정답   = wantsAndCouldChangeLane(+1 왼쪽 / -1 오른쪽)  (SUMO 차로 변경 모델의 바람, 경로 포함)
학생이 만든 상황(전문가가 안 가 본 상황)에서도 정답을 얻어 모방학습의 누적 오차를 줄인다.

usage: python tools/rl/dagger_collect.py MODEL.pt OUT_PREFIX EPISODES_PER_WORKER
환경 변수: N_WORKERS(4), SEED, BETA(0: 전문가가 대신 운전할 확률), STOCH(0/1 학생 확률적 행동),
  LANE_LABEL: best(권장: 경로를 이어 가려면 옮겨야 할 차로, getBestLanes) / wants(SUMO가 바꾸고 싶고 바꿀 수 있음, 추월 포함) / strategic"""
import os, sys, json, shutil
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE, setup as _setup, reward as _reward


def worker(q, repo, sdir, seed, n_ep, model):
    # 수집 프로세스가 죽으면 본 프로세스가 결과를 영원히 기다리므로, 오류를 큐로 알리고 끝낸다
    try:
        _worker(q, repo, sdir, seed, n_ep, model)
    except BaseException:
        import traceback
        q.put(("ERR", sdir, traceback.format_exc()))
        raise


def _worker(q, repo, sdir, seed, n_ep, model):
    _setup(repo)
    import traci, torch
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from algorithms.ppo import PPO
    from train import HPARAMS
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    env = SumoHighwayEnv(cfg_path=os.path.join(sdir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=_reward(), traffic=TRAFFIC, gui=False)
    def _mk(path):
        a_ = PPO(obs_dim=env.observation_space.shape[0], act_dim=2, **{**HPARAMS, "device": "cpu"})
        a_.policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        return a_
    ck_ = None if model.endswith(".ens") else torch.load(model, map_location="cpu", weights_only=True)
    if isinstance(ck_, dict) and ck_.get("algorithm") == "ensemble":     # 앙상블 파일 (algorithms/ensemble.py)
        from algorithms.ensemble import EnsemblePolicy
        agent = EnsemblePolicy.build(env.observation_space.shape[0], 2, ck_["members"], HPARAMS)
    elif model.endswith(".ens"):      # 앙상블 학생(구성원 경로 목록 파일): 가감속 평균, 차로 로짓(log softmax) 평균
        class _Ens:
            def __init__(self, ps):
                self.ms = [_mk(p_) for p_ in ps]
            def predict(self, obs, deterministic=True):
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    outs = [m.policy(o) for m in self.ms]
                mu = torch.stack([x[0] for x in outs]).mean(0); lg = torch.stack([torch.log_softmax(x[2], -1) for x in outs]).mean(0)
                return np.array([float(mu.reshape(-1)[0]), (1.0, 0.0, -1.0)[int(lg.argmax(-1))]], dtype=np.float32)
        agent = _Ens(open(model).read().split())
    else:
        agent = _mk(model)
    # 모델을 만든 PPO가 np.random.seed(고정값)를 다시 걸므로 시드를 다시 건다 (안 하면 워커들이 같은 에피소드를 모음)
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    beta = float(os.environ.get("BETA", "0")); stoch = os.environ.get("STOCH") == "1"
    dt = env.step_length
    O, A, L, E, R = [], [], [], [], []
    res = {}
    for ep in range(n_ep):
        rid = f"ego_{(seed + ep) % int(os.environ.get('N_ROUTES', '7'))}"  # 루프 경로 B(ROUTE_SET=loopB)는 N_ROUTES=1
        env.meta["ego_routes"] = [rid]
        for k in range(10):
            try:
                obs, _ = env.reset()
                break
            except RuntimeError:
                if k == 9:
                    raise
        ego = env.ego_id
        traci.vehicle.setImperfection(ego, 0.0)       # 정답(SUMO 모델)에 운전자 실수 없이
        done, end = False, "other"
        while not done:
            v0 = traci.vehicle.getSpeed(ego)
            o_t = obs
            # 전문가 차로 의도 (이번 스텝 전 상태 기준)
            lc = 1
            try:
                # 경로 때문에 필요한 변경(strategic)이고 그쪽이 막혀 있지 않을 때만 "변경", 아니면 "유지"
                # (추월·오른쪽 유지 같은 선택적 변경은 정답으로 쓰지 않음)
                # 'stay'+'strategic' = 경로상 머물러야 함. 바꾸고 싶을 때는 방향 단어(left/right)가 붙는다
                if os.environ.get("LANE_LABEL", "strategic") == "best":
                    # 경로를 이어 가려면 옮겨야 할 차로 수(bestLanes 오프셋, + = 왼쪽). 그쪽이 막혀 있으면 "유지"
                    cur = traci.vehicle.getLaneID(ego)
                    if not cur.startswith(":"):
                        off = next((b[3] for b in traci.vehicle.getBestLanes(ego) if b[0] == cur), 0)
                        if off != 0:
                            d_ = 1 if off > 0 else -1
                            st_ = traci.vehicle.getLaneChangeStatePretty(ego, d_)[0]
                            if not any(x.startswith("blocked") or x == "overlapping" for x in st_):
                                lc = 0 if d_ > 0 else 2
                if os.environ.get("LANE_LABEL", "strategic") == "wants":
                    # 예전 방식(dg1·dg2): SUMO 차로 변경 모델이 바꾸고 싶고 바꿀 수 있으면 (추월 포함)
                    if traci.vehicle.wantsAndCouldChangeLane(ego, 1):
                        lc = 0
                    elif traci.vehicle.wantsAndCouldChangeLane(ego, -1):
                        lc = 2
                for d_, cls_, w_ in (() if os.environ.get("LANE_LABEL", "strategic") in ("wants", "best") else ((1, 0, "left"), (-1, 2, "right"))):
                    st_ = traci.vehicle.getLaneChangeStatePretty(ego, d_)[0]
                    if w_ in st_ and "strategic" in st_ and not any(x.startswith("blocked") or x == "overlapping" for x in st_):
                        lc = cls_
                        break
            except traci.TraCIException:
                pass
            a = agent.predict(obs, deterministic=not stoch)
            if np.random.rand() < beta:      # 전문가가 대신 운전 (가감속은 아래 정답으로, 차로는 의도대로)
                a = np.array([a[0], (1.0, 0.0, -1.0)[lc]], dtype=np.float32)
            obs, r, te, tr, info = env.step(a)
            done = te or tr
            if ego in traci.vehicle.getIDList():
                try:
                    vw = traci.vehicle.getSpeedWithoutTraCI(ego)
                except traci.TraCIException:
                    vw = v0
                acc = (vw - v0) / dt
                lab = acc / env.max_accel if acc >= 0 else acc / env.max_decel
                O.append(o_t); A.append(float(np.clip(lab, -1, 1))); L.append(lc); E.append(seed * 100000 + ep); R.append(int(rid[4:]))
            if done:
                end = ("arrival" if info.get("arrived") else "collision" if info.get("collided") else "red_light" if info.get("red_light")
                       else "wrong_lane" if info.get("wrong_lane") else "timeout" if tr else "other")
        res.setdefault(rid, {}).setdefault(end, 0)
        res[rid][end] += 1
    env.close()
    q.put((np.asarray(O, np.float32), np.asarray(A, np.float32), np.asarray(L, np.int64), np.asarray(E), np.asarray(R), res))


def main():
    model, out, n_ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
    repo = BASE
    _setup(repo)
    nw = int(os.environ.get("N_WORKERS", "4"))
    seed = int(os.environ.get("SEED", "0"))
    base = os.environ.get("SUMO_BASE") or os.path.join(BASE, "env", "sumo_expert")   # w0..w{N-1} 폴더가 있는 곳
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(q, repo, os.path.join(base, f"w{i}"), seed * 10 + i, n_ep, model)) for i in range(nw)]
    for p in ps:
        p.start()
    parts = []
    for _ in ps:
        x = q.get()
        if isinstance(x[0], str) and x[0] == "ERR":
            print("WORKER ERROR", x[1], x[2], flush=True)
            for p in ps:
                p.terminate()
            sys.exit(1)
        parts.append(x)
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
    print("steps", len(O), "lane_cls", np.bincount(L, minlength=3).tolist(), "학생 결과", tot, flush=True)
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
