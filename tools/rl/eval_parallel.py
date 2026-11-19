"""경로별 평가를 여러 SUMO로 나눠 돌리고, 끝나면 보고서(표·신뢰구간·그래프)까지 만든다.

usage:
    python tools/rl/eval_parallel.py MODEL OUT_DIR [옵션]

옵션:
    --n N            경로마다 평가 횟수 (기본 100)
    --workers W      동시에 돌릴 SUMO 수 (기본: CPU 코어 수 - 1, 최대 8)
    --seed S         시드 시작값 (기본 905). 경로·묶음마다 다른 시드를 쓴다
    --routes R       평가할 경로 id (쉼표 구분, 기본: 7개 전부 ego_0..ego_6)
    --scale X        교통량 배율 (TRAFFIC["demand"]["scale"]). 주면 그 배율로
                     SUMO 파일을 env/sumo_eval/scale_X/ 에 만들어서 평가한다
    --sumo-dir DIR   이미 만든 SUMO 폴더로 평가 (--scale 대신)
    --rebuild        env/sumo_eval/ 폴더가 있어도 다시 만든다
    --chunk K        한 작업이 맡는 에피소드 수 (기본 25)
    --label NAME     보고서에 쓸 모델 이름 (기본: 모델 폴더 이름)
    --compare L=JSON 같이 비교할 다른 결과 (여러 번 줄 수 있음, eval_routes 형식 가능)

예 (BC 모델, 실측 교통량 그대로, 경로마다 100회):
    python tools/rl/eval_parallel.py models/yeonsan_7routes_BC/model.pt results/eval_BC_s1 \\
        --scale 1.0 --compare BC_s0.5=models/yeonsan_7routes_BC/eval_560.json

결과 (OUT_DIR):
    eval.json   eval_routes.py와 같은 형식 + "_meta"(조건 기록) + 경로별 "metrics"
    report.md, summary.csv, per_route.csv, fig_*.png   (tools/rl/eval_report.py)

주행이 끝난 이유의 판정은 tools/rl/eval_routes.py와 같다 (결정적 정책).
주행 품질 지표 (경로별 평균):
    travel_time_s  완주한 주행의 이동 시간 (s)
    mean_speed     주행 중 평균 속도 (m/s)
    stop_time_s    속도 0.1 m/s 미만으로 서 있던 시간 (s/주행, 신호 대기 포함)
    min_gap        주행 중 앞차와의 최소 거리 (m, 앞차가 감지된 주행만)
"""
import argparse
import datetime
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))


def _worker(task):
    """한 작업 = (경로 하나, 에피소드 n개, 시드). 새 프로세스에서 실행된다."""
    model, sumo_dir, rid, n, seed = task
    sys.path.insert(0, HERE)
    import common
    common.setup(BASE)                       # libsumo가 있으면 사용 + 7경로 설정
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from test import load_agent
    env = SumoHighwayEnv(cfg_path=os.path.join(sumo_dir, "highway.sumocfg"), road=ROAD, ego=EGO,
                         mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION,
                         reward=common.reward(), traffic=TRAFFIC, gui=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = load_agent(model, env.observation_space.shape[0], 2)
    dt = float(SIMULATION["step_length"])
    env.meta["ego_routes"] = [rid]
    eps, fails = [], 0
    try:
        for _ in range(n):
            try:
                obs, _ = env.reset()
            except RuntimeError:
                fails += 1
                continue
            done, steps, speeds, stopped, gaps, info, te = False, 0, [], 0, [], {}, False
            while not done:
                obs, _, te, tr, info = env.step(agent.predict(obs, deterministic=True))
                done = te or tr
                steps += 1
                if info.get("speed") is not None:
                    speeds.append(float(info["speed"]))
                    stopped += float(info["speed"]) < 0.1
                if info.get("gap") is not None:
                    gaps.append(float(info["gap"]))
            end = ("collision" if info.get("collided") else
                   "red_light" if info.get("red_light") and te else
                   "wrong_lane" if info.get("wrong_lane") else
                   "arrival" if info.get("arrived") else "timeout")
            eps.append({"end": end, "time_s": steps * dt,
                        "mean_speed": sum(speeds) / len(speeds) if speeds else None,
                        "stop_time_s": stopped * dt,
                        "min_gap": min(gaps) if gaps else None})
    finally:
        env.close()
    return rid, eps, fails


def _headless_copy(src, dst):
    """SUMO 폴더를 평가용으로 복사하면서 건물(*.poly.xml)을 뺀다.

    건물은 GUI 배경일 뿐이라 화면 없는 평가에는 필요 없다. 그리고 새 SUMO에서
    만든 poly 파일(height 속성 등)은 옛 SUMO가 읽지 못해 시작부터 멈춘다.
    """
    import re
    import shutil
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if os.path.isfile(p) and not fn.endswith(".poly.xml"):
            shutil.copy2(p, os.path.join(dst, fn))
    cfg = os.path.join(dst, "highway.sumocfg")
    with open(cfg, encoding="utf-8") as f:
        s = f.read()

    def drop_poly(m):
        files = [x for x in m.group(1).split(",") if x.strip() and not x.strip().endswith(".poly.xml")]
        return f'<additional-files value="{",".join(files)}"/>'
    s = re.sub(r'<additional-files value="([^"]*)"\s*/>', drop_poly, s)
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(s)
    return dst


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(description="경로별 병렬 평가 + 보고서")
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed", type=int, default=905)
    ap.add_argument("--routes", default="")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--sumo-dir", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--label", default=None)
    ap.add_argument("--compare", action="append", default=[])
    a = ap.parse_args()

    sys.path.insert(0, HERE)
    import common
    common.setup(BASE)
    import routes7
    from env.road_config import ROAD, EGO, TRAFFIC
    from env import road_builder

    # ── SUMO 폴더 (모든 작업이 같이 읽기만 한다) ──
    if a.sumo_dir:
        src = os.path.abspath(a.sumo_dir)
        sumo_dir = _headless_copy(src, os.path.join(BASE, "env", "sumo_eval",
                                                    "dir_" + os.path.basename(src.rstrip("\\/"))))
        scale = None
    else:
        demand = TRAFFIC.get("demand") or {}
        scale = a.scale if a.scale is not None else demand.get("scale")
        if a.scale is not None:
            if not demand:
                sys.exit("TRAFFIC['demand']가 없어 --scale을 쓸 수 없습니다. --sumo-dir를 쓰세요.")
            demand["scale"] = a.scale
        built = os.path.join(BASE, "env", "sumo_eval", f"build_scale_{scale}")
        if a.rebuild or not os.path.exists(os.path.join(built, "highway.sumocfg")):
            print(f"SUMO 파일 생성: {built} (교통량 배율 {scale})", flush=True)
            road_builder.build(ROAD, EGO, TRAFFIC, built)
        sumo_dir = _headless_copy(built, os.path.join(BASE, "env", "sumo_eval", f"scale_{scale}"))

    route_ids = a.routes.split(",") if a.routes else [f"ego_{i}" for i in range(len(routes7.ROUTES))]
    tasks = []
    for ri, rid in enumerate(route_ids):
        done = 0
        k = 0
        while done < a.n:
            m = min(a.chunk, a.n - done)
            tasks.append((os.path.abspath(a.model), sumo_dir, rid, m, a.seed + 1000 * ri + k))
            done += m
            k += 1
    print(f"평가: 경로 {len(route_ids)}개 × {a.n}회 = {len(route_ids) * a.n}회, "
          f"작업 {len(tasks)}개, 동시 {a.workers}개", flush=True)

    t0 = time.time()
    per = {rid: {"eps": [], "fails": 0} for rid in route_ids}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (rid, eps, fails) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            per[rid]["eps"] += eps
            per[rid]["fails"] += fails
            el = time.time() - t0
            print(f"  [{i}/{len(tasks)}] {routes7.NAMES.get(rid, rid)} +{len(eps)}회  "
                  f"경과 {el / 60:.1f}분, 남은 예상 {el / i * (len(tasks) - i) / 60:.1f}분", flush=True)

    res = {"_meta": {
        "model": os.path.relpath(os.path.abspath(a.model), BASE).replace("\\", "/"),
        "n_per_route": a.n, "seed_start": a.seed, "traffic_scale": scale,
        "sumo_dir": os.path.relpath(sumo_dir, BASE).replace("\\", "/"),
        "commit": _git_commit(), "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "minutes": round((time.time() - t0) / 60, 1),
        "libsumo": "libsumo" in str(sys.modules.get("traci")),
    }}
    for rid in route_ids:
        eps = per[rid]["eps"]
        ends = {}
        for e in eps:
            ends[e["end"]] = ends.get(e["end"], 0) + 1
        tt, tt_n = _mean([e["time_s"] for e in eps if e["end"] == "arrival"])
        sp, sp_n = _mean([e["mean_speed"] for e in eps])
        st, st_n = _mean([e["stop_time_s"] for e in eps])
        mg, mg_n = _mean([e["min_gap"] for e in eps])
        res[routes7.NAMES.get(rid, rid)] = {
            "episodes": len(eps), "insert_fail": per[rid]["fails"], **ends,
            "metrics": {"travel_time_s": tt, "travel_time_s_n": tt_n,
                        "mean_speed": sp, "mean_speed_n": sp_n,
                        "stop_time_s": st, "stop_time_s_n": st_n,
                        "min_gap": mg, "min_gap_n": mg_n}}
    os.makedirs(a.out_dir, exist_ok=True)
    out_json = os.path.join(a.out_dir, "eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"저장: {out_json} ({res['_meta']['minutes']}분)", flush=True)

    import eval_report
    label = a.label or os.path.basename(os.path.dirname(os.path.abspath(a.model)))
    eval_report.main([a.out_dir, f"{label}={out_json}", *a.compare])


if __name__ == "__main__":
    main()
