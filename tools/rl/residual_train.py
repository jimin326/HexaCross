"""병렬 PPO 학습 + 잔차 강화학습 (모방학습 앙상블 위에 보정만 PPO로 학습).

SUMO 환경 N개(프로세스 N개, libsumo)에서 동시에 경험을 모아 PPO 하나를 학습한다.
알고리즘(PPO 업데이트·GAE·정책 신경망)과 환경(관측·보상)은 저장소 코드를 그대로 쓴다.

usage (저장소 루트에서): python tools/rl/residual_train.py RUN_NAME
결과: results/RUN_NAME/ (model.pt, ckpt.pt, residual_*k.pt, train_log.csv, run_patch.json),
      SUMO 작업 폴더 results/RUN_NAME/sumo/w0..  (없으면 지금 설정으로 새로 만든다)

기본 환경 변수:
  N_ENVS (4), STEPS_PER_ENV (1024), TIMESTEPS, SEED, LR, N_EPOCHS, MINIBATCH, ENT_COEF, LANE_ENT,
  REWARD_PATCH (보상 덮어쓰기 JSON), INIT_MODEL, INIT_LOG_STD, SNAP_EVERY, CKPT_EVERY, VALUE_WARMUP (처음 몇 번은 가치 head만)
  MAX_STEPS, LANE_PLAN_RANGE (tools/rl/routes7.py)
잔차 강화학습 (algorithms/residual.py):
  RESIDUAL_BASE   기본 운전자 앙상블 파일 (예: models/yeonsan_7routes_I8/model.pt). 이게 있으면 잔차 학습
  RESIDUAL_SCALE  잔차 가감속 배율 (0.3)
  RESIDUAL_LANE   1이면 잔차 차로 출력을 "거부권"으로 씀 (기본 끔)
  MAX_VETO        거부권으로 미룰 수 있는 최대 연속 스텝 (0 = 제한 없음)
  VETO_COST       거부한 스텝마다 감점, FOLLOW_BIAS 처음 거부 확률을 정하는 로짓 편향 (4.0)
  RES_L2          잔차 가감속 크기 비용 (스텝마다 RES_L2 × 잔차²)
위험 상황 재시작:
  DANGER_P        새 에피소드 중 이 비율은 "충돌 몇 초 전" 저장 상태(SUMO saveState)에서 다시 시작
  RING_EVERY (2), RING_N (4): 몇 스텝마다, 몇 개 상태를 들고 있을지 (가장 오래된 것 = 충돌 RING_EVERY×RING_N 스텝 전)
  DANGER_DIR, DANGER_MAX (워커당 150)
모방학습 손실 함께 (BC_DATA, BC_W, BC_STEPS, BC_BATCH, LC_WEIGHT): 잔차가 아닐 때 PPO + BC
  BC_DATA는 쉼표로 여러 파일. tools/rl/make_bcdata.py로 DAgger 데이터 + 넓은 차로 변경 보강을 만든다.
앙상블 전체 미세조정 (PPO + BC):
  INIT_ENSEMBLE   앙상블 파일 (예: models/yeonsan_7routes_J8/model.pt). 구성원 8개를 모두 학습하고
                  가치 신경망은 새로 만든다. INIT_LOG_STD 기본 -1.5. 결과는 ens_model.pt, ens_*k.pt (앙상블 형식)
  예: INIT_ENSEMBLE=models/yeonsan_7routes_J8/model.pt LR=1e-5 BC_DATA=data/bc_ft.npz BC_W=1 VALUE_WARMUP=20 \
      LANE_ENT=0 DANGER_P=0.2 RING_N=6 TIMESTEPS=1000000 SEED=21 \
      REWARD_PATCH='{"collision_penalty": 50.0, "red_light_penalty": 20.0, "wrong_lane_penalty": 10.0, "ttc_penalty": 0.3}' \
      python tools/rl/residual_train.py ft1
  (결과: models/yeonsan_7routes_J8_RL/README.md)

결과 요약 (2026-11-20, 교통량 0.85): 잔차 강화학습은 여러 설정에서 모두 기본 운전자보다 나아지지 않았다.
models/yeonsan_7routes_I8/README.md "강화학습 사후학습" 참고.
"""
import os, sys, json, time, shutil
import multiprocessing as mp
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))


def _setup(repo):
    import sumo
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
    import libsumo
    sys.modules["traci"] = libsumo
    sys.path.insert(0, repo)
    sys.path.insert(0, S)
    os.chdir(repo)
    import routes7
    routes7.apply()


def _reward():
    from train import REWARD
    r = dict(REWARD)
    r.update(json.loads(os.environ.get("REWARD_PATCH", "{}")))
    return r


def worker(remote, repo, sdir, seed):
    _setup(repo)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    np.random.seed(seed)
    import random
    random.seed(seed)
    env = SumoHighwayEnv(cfg_path=os.path.join(sdir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=_reward(), traffic=TRAFFIC, gui=False)
    # 잔차 강화학습 (RESIDUAL_BASE = 모방학습 앙상블 파일): 정책 출력은 앙상블 행동에 더할 보정 (algorithms/residual.py)
    res_base, res_scale, cur_b = None, float(os.environ.get("RESIDUAL_SCALE", "0.3")), None
    if os.environ.get("RESIDUAL_BASE"):
        import torch
        from algorithms.ensemble import EnsemblePolicy
        from algorithms.residual import ResidualPolicy
        from train import HPARAMS
        ck_ = torch.load(os.environ["RESIDUAL_BASE"], map_location="cpu", weights_only=True)
        res_base = EnsemblePolicy.build(ck_["obs_dim"], ck_["act_dim"], ck_["members"], HPARAMS)

    def _aug(o):
        nonlocal cur_b
        if res_base is None:
            return o
        cur_b = res_base.predict(o, deterministic=True)
        return ResidualPolicy.augment(o, cur_b)

    # ── 위험 상황 재시작 (DANGER_P > 0) ──
    # 충돌한 에피소드의 충돌 몇 초 전 SUMO 상태(saveState)를 모아 두고, 새 에피소드의 DANGER_P 비율은
    # 그 상태에서 다시 시작한다. 드문 충돌 상황을 반복해서 겪게 해 강화학습 신호를 늘린다.
    import glob, collections, traci
    d_p = float(os.environ.get("DANGER_P", "0"))
    d_dir = os.environ.get("DANGER_DIR") or os.path.join(os.path.dirname(sdir), "danger")
    os.makedirs(d_dir, exist_ok=True)
    wtag = os.path.basename(sdir)
    ring_every, ring_n = int(os.environ.get("RING_EVERY", "2")), int(os.environ.get("RING_N", "4"))
    ring = collections.deque(maxlen=ring_n)
    st = {"k": 0, "n_saved": 0, "rid": None, "danger": False, "veto_n": 0}
    max_veto = int(os.environ.get("MAX_VETO", "0"))

    def _snap():
        k = st["k"] // ring_every
        path = os.path.join(sdir, f"ring_{k % (ring_n + 1)}.xml")
        traci.simulation.saveState(path)
        ring.append((path, {"rid": st["rid"], "step_count": env.step_count, "wrong_lane_steps": env.wrong_lane_steps}))

    def _reset():
        ring.clear(); st["k"] = 0; st["danger"] = False; st["veto_n"] = 0
        pool = glob.glob(os.path.join(d_dir, "*.xml")) if d_p > 0 else []
        if pool and np.random.rand() < d_p:
            path = pool[np.random.randint(len(pool))]
            meta_ = json.load(open(path[:-4] + ".json"))
            keep = env.meta["ego_routes"]
            env.meta["ego_routes"] = [meta_["rid"]]
            try:
                env.reset()
                traci.simulation.loadState(path)
                if env.ego_id in traci.vehicle.getIDList():
                    traci.vehicle.setSpeedMode(env.ego_id, 0)
                    traci.vehicle.setLaneChangeMode(env.ego_id, 0)
                    env.step_count = int(meta_["step_count"]); env.wrong_lane_steps = int(meta_["wrong_lane_steps"])
                    env._prev_odo = traci.vehicle.getDistance(env.ego_id)
                    st["rid"] = meta_["rid"]; st["danger"] = True
                    return env._get_obs()
            except Exception:
                pass
            finally:
                env.meta["ego_routes"] = keep
        o_ = env.reset()[0]
        try:
            st["rid"] = traci.vehicle.getRouteID(env.ego_id)
        except Exception:
            st["rid"] = None
        return o_

    def _after_step(info):
        if d_p <= 0:
            return
        # DANGER_RED=1: 신호위반 직전 장면도 위험 장면으로 저장 (황색 딜레마 반복 학습)
        if info.get("collided") or (os.environ.get("DANGER_RED") == "1" and info.get("red_light")):
            if len(ring) >= 2 and ring[0][1]["rid"]:
                src, m_ = ring[0]
                n_ = st["n_saved"] % int(os.environ.get("DANGER_MAX", "150"))
                dst = os.path.join(d_dir, f"{wtag}_{n_}.xml")
                shutil.copy(src, dst); json.dump(m_, open(dst[:-4] + ".json", "w"))
                st["n_saved"] += 1
            return
        st["k"] += 1
        if st["k"] % ring_every == 0 and env.ego_id in traci.vehicle.getIDList():
            _snap()

    # ── 설 수 있었던 황색 통과 감점 (YELLOW_PEN > 0) ──
    # 황색을 처음 본 순간 정지선까지 편안한 감속(≤ YELLOW_BCOMF m/s², 기본 4)으로 설 수 있었는데 황색에 정지선을
    # 넘으면 감점한다 (relabel_yellow.py와 같은 규칙). 적색 통과는 환경의 red_light_penalty가 따로 감점한다.
    # 관측만 쓴다: [0]=속도/13.89, [31]=내 신호(+1 녹/0 황/-1 적), [37]=sqrt(정지선 거리/150). 정책 입력은 그대로.
    y_pen, y_b = float(os.environ.get("YELLOW_PEN", "0")), float(os.environ.get("YELLOW_BCOMF", "4.0"))
    ys = {"po": None, "ok": None}

    def _is_y(o_):
        return abs(float(o_[31])) < 0.01 and float(o_[37]) < 0.999

    def _yellow_pen(o_):
        if y_pen <= 0:
            return 0.0
        po, pen = ys["po"], 0.0
        ys["po"] = o_
        if po is None:
            return 0.0
        d0, d1 = float(po[37]) ** 2 * 150.0, float(o_[37]) ** 2 * 150.0
        if _is_y(po) and ys["ok"] is None:          # 이번 신호에서 황색을 처음 본 스텝
            v = float(po[0]) * 13.89
            ys["ok"] = v > 0.5 and v * v / (2 * max(d0 - 2.0, 0.1)) <= y_b
        if d1 > d0 + 5.0:                            # 정지선을 넘음 (다음 신호로 바뀜)
            if _is_y(po) and ys["ok"]:
                pen = y_pen
            ys["ok"] = None
        elif float(o_[31]) > 0.5:                    # 녹색으로 바뀜
            ys["ok"] = None
        return pen

    remote.send(env.observation_space.shape[0] + (2 if res_base is not None else 0))
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            o0 = _reset(); ys["po"], ys["ok"] = o0, None
            remote.send(_aug(o0))
        elif cmd == "step":
            lane_ok = os.environ.get("RESIDUAL_LANE") == "1"
            if res_base is not None and lane_ok and max_veto > 0:
                lane_ok, st["veto_n"] = ResidualPolicy.veto_step(cur_b, data, st["veto_n"], max_veto)
            act = ResidualPolicy.combine(cur_b, data, res_scale, lane_ok) if res_base is not None else data
            # 거부권 비용: 기본 운전자가 차로를 바꾸려는데 잔차가 막으면 매 스텝 조금 감점 (꼭 위험할 때만 막도록)
            veto = res_base is not None and lane_ok and float(cur_b[1]) != 0.0 and float(data[1]) != 0.0
            o, r, te, tr, info = env.step(act)
            r -= _yellow_pen(o)
            if veto:
                r -= float(os.environ.get("VETO_COST", "0.0"))
            if res_base is not None:
                # 보정량 비용: 기본 운전자에서 벗어나는 만큼 감점 (꼭 필요할 때만 보정하도록, KL 정규화와 비슷)
                r -= float(os.environ.get("RES_L2", "0.0")) * float(data[0]) ** 2
            _after_step(info)
            was_danger = st["danger"]
            o = _aug(o)
            nxt = o
            if te or tr:
                for k in range(5):
                    try:
                        n0 = _reset(); ys["po"], ys["ok"] = n0, None
                        nxt = _aug(n0)
                        break
                    except RuntimeError:
                        if k == 4:
                            raise
            remote.send((o, float(r), bool(te), bool(tr), nxt,
                         {**{k: bool(info.get(k)) for k in ("collided", "arrived", "red_light", "wrong_lane")}, "danger": was_danger}))
        elif cmd == "close":
            env.close()
            remote.close()
            return


def main():
    name = sys.argv[1]
    repo = os.environ.get("REPO", os.path.dirname(os.path.dirname(S)))
    _setup(repo)
    import torch
    from algorithms.ppo import PPO
    from utils.buffer import RolloutBuffer
    from train import HPARAMS
    from env.road_config import ROAD, EGO, TRAFFIC
    from env import road_builder

    n_env = int(os.environ.get("N_ENVS", "4"))
    T = int(os.environ.get("STEPS_PER_ENV", "1024"))
    total = int(os.environ.get("TIMESTEPS", "3000000"))
    seed = int(os.environ.get("SEED", "0"))
    run_dir = os.path.join(repo, "results", name)
    os.makedirs(run_dir, exist_ok=True)
    hp = dict(HPARAMS)
    hp.update(lr=float(os.environ.get("LR", hp["lr"])), n_steps=T * n_env,
              n_epochs=int(os.environ.get("N_EPOCHS", hp["n_epochs"])),
              minibatch_size=int(os.environ.get("MINIBATCH", hp["minibatch_size"])),
              entropy_coef=float(os.environ.get("ENT_COEF", hp.get("entropy_coef", 0.0))),
              lane_entropy_coef=float(os.environ.get("LANE_ENT", hp.get("lane_entropy_coef", 0.0))), seed=seed, device="cpu")
    keep = {k: os.environ.get(k) for k in ("N_ENVS", "STEPS_PER_ENV", "TIMESTEPS", "SEED", "LR", "N_EPOCHS", "MINIBATCH",
                                           "ENT_COEF", "REWARD_PATCH", "INIT_MODEL", "INIT_ENSEMBLE", "DANGER_P", "RING_N", "VALUE_WARMUP", "LANE_ENT", "INIT_LOG_STD", "MAX_STEPS", "LANE_PLAN_RANGE",
                                           "OBS_VISIBILITY", "LANE_ARROWS", "INTERNAL_OBS", "BC_DATA", "BC_W", "BC_W_END", "BC_STEPS", "DANGER_P", "DANGER_RED", "YELLOW_PEN", "YELLOW_BCOMF", "ROUTE_SET", "RESIDUAL_BASE", "RESIDUAL_SCALE", "RESIDUAL_LANE")}
    json.dump({"env": keep, "hparams": {k: (list(v) if isinstance(v, tuple) else v) for k, v in hp.items()},
               "reward": _reward()}, open(os.path.join(run_dir, "run_patch.json"), "w"), ensure_ascii=False, indent=1)
    for f in ("env/road_config.py", "env/mdp_config.py", "train.py"):
        shutil.copy(os.path.join(repo, f), run_dir)

    base = os.path.join(repo, "results", name, "sumo")
    if not os.path.exists(os.path.join(base, "w0", "highway.sumocfg")):
        road_builder.build(ROAD, EGO, TRAFFIC, os.path.join(base, "w0"))
        for i in range(1, n_env):
            shutil.copytree(os.path.join(base, "w0"), os.path.join(base, f"w{i}"), dirs_exist_ok=True)

    ctx = mp.get_context("spawn")
    remotes, procs = [], []
    for i in range(n_env):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, repo, os.path.join(base, f"w{i}"), seed * 1000 + i), daemon=True)
        p.start()
        remotes.append(a); procs.append(p)
    obs_dim = remotes[0].recv()
    for r in remotes[1:]:
        r.recv()

    agent = PPO(obs_dim=obs_dim, act_dim=2, buffer=RolloutBuffer(), **hp)
    ens_init = os.environ.get("INIT_ENSEMBLE")
    if ens_init:
        # 앙상블 전체 미세조정: 정책 = 구성원 평균 (EnsembleActorCritic), 가치는 새 신경망
        from algorithms.ensemble import EnsembleActorCritic
        from train import HPARAMS
        ek_ = torch.load(ens_init, map_location="cpu", weights_only=False)
        agent.policy = EnsembleActorCritic(obs_dim, ek_["members"], HPARAMS, init_log_std=float(os.environ.get("INIT_LOG_STD", "-1.5")))
        agent.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=hp["lr"])
        print(f"[ensemble] {ens_init} 구성원 {len(ek_['members'])}개 미세조정", flush=True)
    ck = os.path.join(run_dir, "ckpt.pt")
    steps, n_upd = 0, 0
    if os.path.exists(ck):
        d = torch.load(ck, map_location="cpu", weights_only=False)
        agent.policy.load_state_dict(d["policy"]); agent.optimizer.load_state_dict(d["optim"])
        steps, n_upd = d["steps"], d["updates"]
        print(f"[ckpt] 이어서 학습: {steps} 스텝부터", flush=True)
    elif ens_init:
        pass
    elif os.environ.get("INIT_MODEL"):
        agent.policy.load_state_dict(torch.load(os.environ["INIT_MODEL"], map_location="cpu", weights_only=False))
        print(f"[init] {os.environ['INIT_MODEL']}", flush=True)
        if os.environ.get("INIT_LOG_STD"):
            with torch.no_grad():
                agent.policy.log_std.fill_(float(os.environ["INIT_LOG_STD"]))
    elif os.environ.get("RESIDUAL_BASE"):
        from algorithms.residual import ResidualPolicy
        ResidualPolicy.init_residual(agent.policy, log_std=float(os.environ.get("INIT_LOG_STD", "-1.0")), follow_bias=float(os.environ.get("FOLLOW_BIAS", "4.0")))
        print(f"[residual] 기본 운전자 {os.environ['RESIDUAL_BASE']}, scale {os.environ.get('RESIDUAL_SCALE', '0.3')}", flush=True)
    print(f"obs_dim={obs_dim} envs={n_env} batch={T * n_env} lr={hp['lr']}", flush=True)

    bc = None
    if os.environ.get("BC_DATA"):
        sys.path.insert(0, S)
        import bc_train as bc_mod
        globals()["bc_mod"] = bc_mod
        ds = [np.load(p) for p in os.environ["BC_DATA"].split(",")]
        bc = (torch.as_tensor(np.concatenate([d["obs"] for d in ds])), torch.as_tensor(np.concatenate([d["accel"] for d in ds])),
              torch.as_tensor(np.concatenate([d["lane_cls"] for d in ds])))
        w_ = float(os.environ.get("LC_WEIGHT", "8"))
        globals()["bc_cw"] = torch.tensor([w_, 1.0, w_])
        print(f"[bc] 모방학습 손실 데이터 {len(bc[0])}개, 가중치 {os.environ.get('BC_W', '1.0')}, 업데이트당 {os.environ.get('BC_STEPS', '20')}번", flush=True)
    for r in remotes:
        r.send(("reset", None))
    obs = [r.recv() for r in remotes]
    ep_ret, ep_len = [0.0] * n_env, [0] * n_env
    recent = []          # (return, length, end)
    recent_d = []        # 위험 재시작 에피소드 (따로 집계)
    log_f = open(os.path.join(run_dir, "train_log.csv"), "a")
    if os.path.getsize(os.path.join(run_dir, "train_log.csv")) == 0:
        log_f.write("update,steps,ep_rew_mean,ep_len_mean,arrival,collision,red_light,wrong_lane,other,episodes,policy_loss,value_loss,entropy,approx_kl,std,sec\n")
    snap_every = int(os.environ.get("SNAP_EVERY", "250000"))
    ck_every = int(os.environ.get("CKPT_EVERY", "5"))
    g = agent.gamma
    while steps < total:
        t0 = time.time()
        bufs = [RolloutBuffer() for _ in range(n_env)]
        agent.policy.eval()
        for _ in range(T):
            with torch.no_grad():
                a, lp, v = agent.policy.sample(torch.as_tensor(np.stack(obs), dtype=torch.float32))
            a, lp, v = a.numpy().astype(np.float32), lp.numpy(), v.numpy()
            for i, r in enumerate(remotes):
                r.send(("step", a[i]))
            res = [r.recv() for r in remotes]
            for i, (o, rew, te, tr, nxt, info) in enumerate(res):
                stored = rew
                if tr and not te:
                    with torch.no_grad():
                        stored += g * float(agent.policy(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[-1].item())
                bufs[i].add(obs[i], a[i], float(lp[i]), stored, float(v[i]), te or tr)
                ep_ret[i] += rew; ep_len[i] += 1
                if te or tr:
                    end = ("arrival" if info["arrived"] else "collision" if info["collided"] else "red_light" if info["red_light"]
                           else "wrong_lane" if info["wrong_lane"] else "other")
                    (recent_d if info.get("danger") else recent).append((ep_ret[i], ep_len[i], end))
                    del recent_d[:-1000]
                    del recent[:-1000]
                    ep_ret[i], ep_len[i] = 0.0, 0
                obs[i] = nxt
            steps += n_env
        agent.policy.train()
        with torch.no_grad():
            last_v = agent.policy(torch.as_tensor(np.stack(obs), dtype=torch.float32))[-1].numpy().reshape(-1)
        advs, rets, big = [], [], RolloutBuffer()
        for i, b in enumerate(bufs):
            ad, rt = b.compute_gae(float(last_v[i]), agent.gamma, agent.gae_lambda)
            advs.append(ad); rets.append(rt)
            for k in ("obs", "actions", "log_probs", "rewards", "values", "dones"):
                getattr(big, k).extend(getattr(b, k))
        agent.buffer = big
        agent.num_timesteps = steps
        warm = n_upd < int(os.environ.get("VALUE_WARMUP", "0"))
        if warm:   # 처음 몇 번은 가치 head만 학습 (모방학습 정책을 흔들지 않게)
            for n_, p_ in agent.policy.named_parameters():
                p_.requires_grad_(n_.startswith("v_head") or n_.startswith("v_net"))
        st = agent.update(np.concatenate(advs), np.concatenate(rets))
        if bc is not None and not warm:
            # 모방학습 손실을 함께: PPO 업데이트 뒤 전문가 데이터로 BC_STEPS번 (가중치 BC_W) — 모방학습으로 익힌 행동이 흐트러지지 않게
            bo, ba, bl = bc
            # BC_W_END가 있으면 학습이 진행될수록 모방학습 가중치를 BC_W → BC_W_END로 선형 감소 (강화학습이 점점 주도)
            w0_ = float(os.environ.get("BC_W", "1.0")); w1_ = float(os.environ.get("BC_W_END", w0_))
            bc_w_now = w0_ + (w1_ - w0_) * min(1.0, steps / float(os.environ.get("BC_DECAY_STEPS", total)))
            for _ in range(int(os.environ.get("BC_STEPS", "20"))):
                ix = torch.randint(0, len(bo), (int(os.environ.get("BC_BATCH", "512")),))
                mu_, _, lg_, _ = agent.policy(bo[ix])
                lbc = bc_mod.accel_loss(mu_.squeeze(-1), ba[ix], bo[ix]) + torch.nn.functional.cross_entropy(lg_, bl[ix], weight=bc_cw)
                agent.optimizer.zero_grad(); (bc_w_now * lbc).backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5); agent.optimizer.step()
            st["bc_loss"] = float(lbc.item()); st["bc_w"] = bc_w_now
        if warm:
            for p_ in agent.policy.parameters():
                p_.requires_grad_(True)
        n_upd += 1
        win = recent[-100:]
        wd = recent_d[-100:]
        cd = {k: sum(1 for x in wd if x[2] == k) for k in ("arrival", "collision", "wrong_lane")}
        cnt = {k: sum(1 for x in win if x[2] == k) for k in ("arrival", "collision", "red_light", "wrong_lane", "other")}
        line = (f"{n_upd},{steps},{np.mean([x[0] for x in win]) if win else float('nan'):.3f},{np.mean([x[1] for x in win]) if win else float('nan'):.0f},"
                + ",".join(str(cnt[k]) for k in cnt) + f",{len(win)},{st['policy_loss']:.5f},{st['value_loss']:.4f},{st['entropy']:.4f},{st['approx_kl']:.5f},{agent.policy.current_std:.4f},{time.time() - t0:.1f}")
        log_f.write(line + "\n"); log_f.flush()
        if "bc_loss" in st:
            log_f.write(f"# bc_loss {n_upd} {st['bc_loss']:.4f}\n")
        print(f"[update {n_upd:4d}] steps={steps:8d} rew={np.mean([x[0] for x in win]) if win else 0:.2f} "
              f"최근{len(win)}회 완주{cnt['arrival']} 충돌{cnt['collision']} 신호{cnt['red_light']} 차로{cnt['wrong_lane']} "
              f"kl={st['approx_kl']:.4f} {time.time() - t0:.0f}s"
              + (f" | 위험재시작{len(wd)}회 완주{cd['arrival']} 충돌{cd['collision']} 차로{cd['wrong_lane']}" if wd else ""), flush=True)
        if n_upd % ck_every == 0 or steps >= total:
            torch.save({"policy": agent.policy.state_dict(), "optim": agent.optimizer.state_dict(), "steps": steps, "updates": n_upd},
                       ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            torch.save(agent.policy.state_dict(), os.path.join(run_dir, "model.pt"))
            if ens_init:
                from algorithms.ensemble import EnsemblePolicy
                EnsemblePolicy.save(os.path.join(run_dir, "ens_model.pt"), obs_dim, 2, agent.policy.members())
            if os.environ.get("RESIDUAL_BASE"):
                from algorithms.residual import ResidualPolicy
                ResidualPolicy.save(os.path.join(run_dir, "residual_model.pt"), obs_dim - 2, 2, float(os.environ.get("RESIDUAL_SCALE", "0.3")),
                                    torch.load(os.environ["RESIDUAL_BASE"], map_location="cpu", weights_only=True), agent.policy.state_dict(), os.environ.get("RESIDUAL_LANE") == "1", int(os.environ.get("MAX_VETO", "0")))
        if steps // snap_every != (steps - T * n_env) // snap_every:
            torch.save(agent.policy.state_dict(), os.path.join(run_dir, f"snap_{steps // 1000}k.pt"))
            if ens_init:
                from algorithms.ensemble import EnsemblePolicy
                EnsemblePolicy.save(os.path.join(run_dir, f"ens_{steps // 1000}k.pt"), obs_dim, 2, agent.policy.members())
            if os.environ.get("RESIDUAL_BASE"):
                from algorithms.residual import ResidualPolicy
                ResidualPolicy.save(os.path.join(run_dir, f"residual_{steps // 1000}k.pt"), obs_dim - 2, 2, float(os.environ.get("RESIDUAL_SCALE", "0.3")),
                                    torch.load(os.environ["RESIDUAL_BASE"], map_location="cpu", weights_only=True), agent.policy.state_dict(), os.environ.get("RESIDUAL_LANE") == "1", int(os.environ.get("MAX_VETO", "0")))
    for r in remotes:
        r.send(("close", None))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
