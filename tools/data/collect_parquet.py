"""대량 주행 데이터셋 수집 (DATA_FORMAT v2, Parquet 샤드).

1행 = 1 transition = SUMO 1스텝(0.5초): (o_t, a_t, r_t, o_{t+1}, terminated, truncated) + s_t, s_{t+1}, 감지 차량, 이벤트.
v1(collect_pomdp_data.py)과 값은 같고 저장 방식만 다르다 (DATA_FORMAT.md 10절 "v2").
- 파일: Parquet(zstd) 샤드. 에피소드가 샤드 경계에서 잘리지 않는다 (샤드는 에피소드 단위로 닫음).
- state는 펼친 컬럼(ego_x, tl_color_0 … )이고, s_{t+1}은 같은 이름에 next_ 를 붙인 컬럼이다 (모든 행에 있음).
- 주변 차량(state.vehicles)은 AV 반경 RADIUS m 안의 차만 저장한다 (AV 자신 제외).
- 정책: 결정적(μ, argmax 차로)과 확률적(정책 분포에서 샘플)을 에피소드 단위로 번갈아 쓴다 (policy_mode 0/1).

usage (저장소 루트에서, 루프 경로 B):
    ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 \
    python tools/data/collect_parquet.py --model models/yeonsan_loopB_RL/model.pt \
        --transitions 10000000 --workers 4 --out-dir data/v2/loopB_RL
환경 변수 RADIUS (기본 100), SHARD_ROWS (기본 250000: 샤드 파일 하나의 행 수), ROWGROUP_ROWS (기본 25000: 이만큼 모이면 디스크에 씀).
메모리: 행을 ROWGROUP_ROWS만큼만 들고 있다가 열린 샤드에 row group으로 쓴다 (워커당 수백 MB).
샤드는 다 쓴 뒤에 .tmp에서 이름을 바꾸므로, 끊겼을 때 남은 .tmp는 다음 실행이 지우고 그만큼 다시 모은다.
진행 상황: OUT_DIR/progress_w*.json (워커별 누적 행 수·에피소드 수·끝난 이유), 샤드: OUT_DIR/shard_w{워커}_{번호}.parquet
중간에 끊기면 같은 명령으로 다시 실행하면 이미 쓴 샤드를 세고 이어서 모은다.
"""
import argparse, glob, json, multiprocessing as mp, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE, "tools", "rl"))

AV_KEYS = ["ego_alive", "ego_x", "ego_y", "ego_speed", "ego_lane", "ego_lane_pos", "ego_leg", "ego_lane_count",
           "ego_dist_to_junction"] + [f"tl_color_{i}" for i in range(6)] + [f"tl_remaining_{i}" for i in range(6)] + ["ego_tl", "ego_tl_dist"]
INT_KEYS = {"ego_lane", "ego_leg", "ego_lane_count", "ego_tl"}


def _schema():
    import pyarrow as pa
    obs = pa.list_(pa.float32(), 56)
    veh = pa.list_(pa.struct([("x", pa.float32()), ("y", pa.float32()), ("speed", pa.float32()), ("lane", pa.int8()),
                              ("lane_pos", pa.float32()), ("leg", pa.int8()), ("type", pa.int8())]))
    det = pa.list_(pa.struct([("x", pa.float32()), ("y", pa.float32()), ("rel_pos", pa.float32()),
                              ("speed", pa.float32()), ("lane_offset", pa.int8())]))
    f = [("episode", pa.int32()), ("t", pa.int32()), ("worker", pa.int8()), ("policy_mode", pa.int8()),
         ("observation", obs), ("action_raw", pa.list_(pa.float32(), 2)), ("accel_raw", pa.float32()),
         ("lane_change_raw", pa.float32()), ("lane_change", pa.int8()), ("lane_change_applied", pa.bool_()),
         ("reward", pa.float32()), ("next_observation", obs), ("terminated", pa.bool_()), ("truncated", pa.bool_())]
    for pre in ("", "next_"):
        for k in AV_KEYS:
            t = pa.bool_() if k == "ego_alive" else (pa.int16() if k in INT_KEYS else pa.float32())
            f.append((pre + k, t))
    f += [("vehicles", veh), ("detected_vehicles", det),
          ("collided", pa.bool_()), ("arrived", pa.bool_()), ("red_light", pa.bool_()), ("wrong_lane", pa.bool_()),
          ("lane_before", pa.int8()), ("lane_after", pa.int8())]
    return pa.schema(f)


def _flat_state(s):
    out = {}
    alive = bool(s.get("ego_alive", False))
    for k in AV_KEYS:
        if k == "ego_alive":
            out[k] = alive
        elif not alive or k not in s:
            out[k] = None
        else:
            out[k] = int(s[k]) if k in INT_KEYS else float(s[k])
    return out


def _nearby(traci, env, lay, radius):
    """AV 반경 radius m 안의 차량 (AV 제외). 위치를 먼저 보고 가까운 차만 나머지 값을 읽는다."""
    if env.ego_id not in traci.vehicle.getIDList():
        return []
    ex, ey = traci.vehicle.getPosition(env.ego_id)
    r2 = radius * radius
    out = []
    for vid in traci.vehicle.getIDList():
        if vid == env.ego_id:
            continue
        try:
            x, y = traci.vehicle.getPosition(vid)
            if (x - ex) ** 2 + (y - ey) ** 2 > r2:
                continue
            out.append({"x": x, "y": y, "speed": traci.vehicle.getSpeed(vid), "lane": traci.vehicle.getLaneIndex(vid),
                        "lane_pos": traci.vehicle.getLanePosition(vid),
                        "leg": int(lay["edge_leg"].get(traci.vehicle.getRoadID(vid), -1)),
                        "type": 1 if traci.vehicle.getVehicleClass(vid) == "bus" else 0})
        except traci.TraCIException:
            continue
    return out


def _state(traci, env, lay, radius):
    """get_privileged_state와 같은 값이되, 주변 차량은 반경 안만 (전체 순회 비용을 줄임)."""
    if env.ego_id not in traci.vehicle.getIDList():
        return {"ego_alive": False}, []
    try:
        x, y = traci.vehicle.getPosition(env.ego_id)
        road = traci.vehicle.getRoadID(env.ego_id)
        s = {"ego_alive": True, "ego_x": x, "ego_y": y, "ego_speed": traci.vehicle.getSpeed(env.ego_id),
             "ego_lane": traci.vehicle.getLaneIndex(env.ego_id), "ego_lane_pos": traci.vehicle.getLanePosition(env.ego_id),
             "ego_leg": int(lay["edge_leg"].get(road, -1)),
             "ego_lane_count": int(lay["lane_count"].get(road, traci.edge.getLaneNumber(road))),
             "ego_dist_to_junction": env._ego_dist_to_junction()}
        s.update(env._signal_state())
        return s, _nearby(traci, env, lay, radius)
    except traci.TraCIException:
        return {"ego_alive": False}, []


def worker(wid, args, target, start_episode, q):
    import common
    common.setup(BASE)
    import traci, torch
    import pyarrow as pa, pyarrow.parquet as pq
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from env import road_builder
    from algorithms.ppo import PPO
    from train import HPARAMS
    torch.set_num_threads(1)
    sdir = os.path.join(args.out_dir, "_sumo", f"w{wid}")
    if not os.path.exists(os.path.join(sdir, "highway.sumocfg")):
        road_builder.build(ROAD, EGO, TRAFFIC, sdir)
    env = SumoHighwayEnv(cfg_path=os.path.join(sdir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=common.reward(), traffic=TRAFFIC, gui=False)
    agent = PPO(obs_dim=env.observation_space.shape[0], act_dim=2, **{**HPARAMS, "device": "cpu", "seed": args.seed * 100 + wid})
    agent.policy.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=False))
    np.random.seed(args.seed * 1000 + wid)          # SUMO 교통 시드 (PPO 생성자가 전역 시드를 덮으므로 뒤에)
    torch.manual_seed(args.seed * 1000 + wid)
    schema = _schema()
    radius = float(os.environ.get("RADIUS", "100"))
    shard_rows = int(os.environ.get("SHARD_ROWS", "250000"))
    prog_path = os.path.join(args.out_dir, f"progress_w{wid}.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"rows": 0, "episodes": 0, "shards": 0, "ends": {}}
    prog = {"rows": prog["rows"], "episodes": prog.get("closed_episodes", prog["episodes"] if prog["rows"] else 0),
            "shards": prog["shards"], "ends": prog.get("closed_ends", prog["ends"] if prog["rows"] else {})}
    for f_ in glob.glob(os.path.join(args.out_dir, f"shard_w{wid}_*.parquet.tmp")):
        os.remove(f_)                          # 끊긴 실행이 다 못 쓴 샤드
    rowgroup = int(os.environ.get("ROWGROUP_ROWS", "25000"))
    rows = {f.name: [] for f in schema}
    n_buf, n_open, ep, t0, writer = 0, 0, start_episode + prog["episodes"], time.time(), None
    open_eps, open_ends = 0, {}
    start_rows = prog["rows"]

    def write_group():
        nonlocal rows, n_buf, n_open, writer
        if not n_buf:
            return
        if writer is None:
            writer = pq.ParquetWriter(os.path.join(args.out_dir, f"shard_w{wid}_{prog['shards']:04d}.parquet.tmp"), schema,
                                      compression="zstd", compression_level=6)
        writer.write_table(pa.Table.from_pydict(rows, schema=schema))
        n_open += n_buf
        rows = {f.name: [] for f in schema}; n_buf = 0

    def close_shard():
        nonlocal writer, n_open, open_eps, open_ends
        write_group()
        if writer is None:
            return
        writer.close(); writer = None
        path = os.path.join(args.out_dir, f"shard_w{wid}_{prog['shards']:04d}.parquet")
        os.replace(path + ".tmp", path)
        prog["rows"] += n_open; prog["shards"] += 1; prog["episodes"] += open_eps
        for k_, v_ in open_ends.items():
            prog["ends"][k_] = prog["ends"].get(k_, 0) + v_
        n_open, open_eps, open_ends = 0, 0, {}
        json.dump(prog, open(prog_path, "w"))

    while prog["rows"] + n_open + n_buf < target:
        mode = ep % 2                         # 0 결정적, 1 확률적
        env.meta["ego_routes"] = ["ego_0"]
        for k in range(10):
            try:
                obs, _ = env.reset()
                break
            except RuntimeError:
                if k == 9:
                    raise
        lay = env._state_layout()
        s_t, veh_t = _state(traci, env, lay, radius)
        done, t, end = False, 0, "other"
        while not done:
            det = env.get_detected_vehicles()
            a = agent.predict(obs, deterministic=(mode == 0))
            a = np.clip(a, -1.0, 1.0).astype(np.float32)
            nobs, r, te, tr, info = env.step(a)
            s_n, veh_n = _state(traci, env, lay, radius)
            done = te or tr
            fs, fn = _flat_state(s_t), _flat_state(s_n)
            row = {"episode": ep, "t": t, "worker": wid, "policy_mode": mode,
                   "observation": np.asarray(obs, np.float32).tolist(), "action_raw": a.tolist(),
                   "accel_raw": float(info["accel_raw"]), "lane_change_raw": float(info["lane_change_raw"]),
                   "lane_change": int(info["lane_change"]), "lane_change_applied": bool(info["lane_change_applied"]),
                   "reward": float(r), "next_observation": np.asarray(nobs, np.float32).tolist(),
                   "terminated": bool(te), "truncated": bool(tr),
                   "vehicles": veh_t, "detected_vehicles": det,
                   "collided": bool(info["collided"]), "arrived": bool(info["arrived"]),
                   "red_light": bool(info.get("red_light", False)), "wrong_lane": bool(info.get("wrong_lane", False)),
                   "lane_before": info["lane_before"], "lane_after": info["lane_after"]}
            row.update(fs); row.update({"next_" + k: v for k, v in fn.items()})
            for k, v in row.items():
                rows[k].append(v)
            n_buf += 1; t += 1
            obs, s_t, veh_t = nobs, s_n, veh_n
        end = ("collision" if info["collided"] else "red_light" if info.get("red_light") else "wrong_lane" if info.get("wrong_lane")
               else "arrival" if info["arrived"] else "timeout" if tr else "other")
        open_eps += 1
        key = f"{'det' if mode == 0 else 'sto'}_{end}"
        open_ends[key] = open_ends.get(key, 0) + 1
        ep += 1
        if n_buf >= rowgroup:
            write_group()
        if n_open + n_buf >= shard_rows:
            close_shard()
        done_rows = prog["rows"] + n_open + n_buf
        q.put((wid, done_rows, prog["episodes"] + open_eps, round((done_rows - start_rows) / max(time.time() - t0, 1e-6), 1)))
    close_shard()
    env.close()
    q.put((wid, "done", prog["episodes"], None))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--transitions", type=int, default=10_000_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    args.model = os.path.abspath(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    per = [args.transitions // args.workers + (1 if i < args.transitions % args.workers else 0) for i in range(args.workers)]
    q = mp.get_context("spawn").Queue()
    ps = [mp.get_context("spawn").Process(target=worker, args=(i, args, per[i], 1_000_000 * (i + 1), q)) for i in range(args.workers)]
    for x in ps:
        x.start()
    live, last = set(range(args.workers)), {}
    while live:
        try:
            wid, n, eps, rate = q.get(timeout=600)
        except Exception:
            dead = [i for i in live if not ps[i].is_alive()]
            if dead:
                print("WORKER DIED", dead, flush=True); sys.exit(1)
            continue
        if n == "done":
            live.discard(wid); continue
        last[wid] = (n, eps, rate)
        tot = sum(v[0] for v in last.values())
        print(f"{time.strftime('%H:%M:%S')} w{wid} rows={n} eps={eps} {rate}/s | total {tot:,} / {args.transitions:,}", flush=True)
    for x in ps:
        x.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
