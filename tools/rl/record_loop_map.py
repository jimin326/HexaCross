"""루프 B 주행 영상 (지도 패널 + 구간별 배속). sumo-gui 없이 직접 그린다.

1) log: 한 에피소드를 평가와 같은 방법(eval_parallel._worker, 같은 시드)으로 돌리며 스텝마다 기록
   python tools/rl/record_loop_map.py log MODEL SUMO_DIR SEED OUT.pkl
2) render: 기록으로 영상 만들기. 차량·신호가 얽히는 구간은 느리게(기본 배속), 그냥 달리는 구간은 FAST배 빠르게.
   python tools/rl/record_loop_map.py render OUT.pkl SUMO_DIR OUT.mp4 [--seconds 58] [--fast 3] [--workers 4]

화면: 왼쪽 = AV를 따라가는 확대 화면(북쪽 위, 약 170 m), 오른쪽 = 전체 지도(경로·지나온 길·현재 위치) + 상태판.
배속은 화면에 "실제 시간 대비 ×N"으로 표시한다.
"""
import os, sys, math, pickle, subprocess, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))


# ───────────────────────── 1) 기록 ─────────────────────────
def do_log(model, sumo_dir, seed, out):
    sys.path.insert(0, HERE); sys.path.insert(0, BASE); os.chdir(BASE)
    import common
    common.setup(BASE)
    import torch
    torch.set_num_threads(1)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from test import load_agent
    env = SumoHighwayEnv(cfg_path=os.path.join(sumo_dir, "highway.sumocfg"), road=ROAD, ego=EGO,
                         mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION,
                         reward=common.reward(), traffic=TRAFFIC, gui=False)
    import traci
    agent = load_agent(model, env.observation_space.shape[0], 2)
    env.meta["ego_routes"] = ["ego_0"]
    np.random.seed(seed); torch.manual_seed(seed)
    obs, _ = env.reset()
    route = list(traci.vehicle.getRoute(env.ego_id))
    tls_ids = list(traci.trafficlight.getIDList())
    # 신호등별 (링크 번호, 들어오는 차로) — 정지선 색을 그리려고
    tls_links = {}
    for t in tls_ids:
        links = []
        for i, group in enumerate(traci.trafficlight.getControlledLinks(t)):
            for (inc, _out, _via) in group:
                links.append((i, inc))
        tls_links[t] = links
    dims = {}
    steps = []
    done, info, k = False, {}, 0
    R = 140.0
    while not done:
        act = agent.predict(obs, deterministic=True)
        rec = {"k": k}
        try:
            ex, ey = traci.vehicle.getPosition(env.ego_id)
            rec.update(x=ex, y=ey, ang=traci.vehicle.getAngle(env.ego_id), v=traci.vehicle.getSpeed(env.ego_id),
                       lane=traci.vehicle.getLaneID(env.ego_id), edge=traci.vehicle.getRoadID(env.ego_id),
                       vmax=traci.lane.getMaxSpeed(traci.vehicle.getLaneID(env.ego_id)))
        except Exception:
            break
        vs = []
        for vid in traci.vehicle.getIDList():
            if vid == env.ego_id:
                continue
            x, y = traci.vehicle.getPosition(vid)
            if abs(x - ex) > R or abs(y - ey) > R:
                continue
            if vid not in dims:
                dims[vid] = (traci.vehicle.getLength(vid), traci.vehicle.getWidth(vid), traci.vehicle.getVehicleClass(vid))
            vs.append((vid, x, y, traci.vehicle.getAngle(vid), traci.vehicle.getSpeed(vid)))
        rec["veh"] = vs
        sig = {}
        for t in tls_ids:
            sig[t] = traci.trafficlight.getRedYellowGreenState(t)
        rec["sig"] = sig
        rec["obs"] = np.asarray(obs, dtype=np.float32)
        obs, r, te, tr, info = env.step(act)
        rec["act"] = np.asarray(act, dtype=np.float32).ravel().tolist() if not isinstance(act, tuple) else [float(a) for a in np.ravel(act[0])] + [float(np.ravel(act[1])[0])]
        rec["r"] = float(r)
        rec["lc"] = int(info.get("lane_change", 0) or 0)
        rec["ev"] = {k2: bool(info.get(k2)) for k2 in ("collided", "red_light", "wrong_lane", "arrived", "teleported")}
        steps.append(rec)
        done = te or tr
        k += 1
    ego_dims = (traci.vehicle.getLength(env.ego_id), traci.vehicle.getWidth(env.ego_id)) if env.ego_id in traci.vehicle.getIDList() else (5.0, 1.8)
    end = ("collision" if info.get("collided") else "teleported" if info.get("teleported") else
           "red_light" if info.get("red_light") else "wrong_lane" if info.get("wrong_lane") else
           "arrival" if info.get("arrived") else "timeout")
    env.close()
    pickle.dump({"steps": steps, "route": route, "tls_links": tls_links, "dims": dims, "ego_dims": ego_dims,
                 "end": end, "seed": seed, "model": model, "dt": 0.5,
                 "road_details": os.environ.get("ROAD_DETAILS", "")}, open(out, "wb"))
    print("logged", len(steps), "steps, end", end, flush=True)


# ───────────────────────── 1b) sumo-gui 실제 화면 캡처 + 기록 ─────────────────────────
def do_gui(model, sumo_dir, seed, out_pkl, frames_dir, w=1100, h=900, max_steps=0):
    """sumo-gui로 같은 에피소드를 돌리며 매 스텝 화면을 캡처하고(frames_dir/00000.png …) 상태도 기록한다.
    화면 i = i번째 env.step 뒤 상태 (스크린샷은 다음 simulationStep 때 써지므로 마지막에 한 스텝 더 돈다)."""
    os.environ["USE_LIBSUMO"] = "0"
    sys.path.insert(0, HERE); sys.path.insert(0, BASE); os.chdir(BASE)
    import routes7; routes7.apply()
    import torch
    torch.set_num_threads(1)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from test import load_agent
    import common
    import traci
    sim = {**SIMULATION, "gui_delay": 0.0}
    env = SumoHighwayEnv(cfg_path=os.path.join(sumo_dir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=sim,
                         mdp_obs=OBSERVATION, action=ACTION, reward=common.reward(), traffic=TRAFFIC, gui=True)
    agent = load_agent(model, env.observation_space.shape[0], 2)
    env.meta["ego_routes"] = ["ego_0"]
    np.random.seed(seed); torch.manual_seed(seed)
    obs, _ = env.reset()
    view = "View #0"
    traci.gui.trackVehicle(view, env.ego_id)
    traci.gui.setZoom(view, float(os.environ.get("ZOOM", "1400")))
    traci.vehicle.highlight(env.ego_id, (255, 0, 255, 255), 6, 255, 100000, 0)
    route = list(traci.vehicle.getRoute(env.ego_id))
    shutil.rmtree(frames_dir, ignore_errors=True); os.makedirs(frames_dir)
    steps, done, info, k = [], False, {}, 0
    while not done:
        act = agent.predict(obs, deterministic=True)
        o_before = np.asarray(obs, dtype=np.float32)
        obs, r, te, tr, info = env.step(act)
        done = te or tr
        rec = {"k": k, "obs": o_before, "r": float(r), "lc": int(info.get("lane_change", 0) or 0)}
        try:
            ex, ey = traci.vehicle.getPosition(env.ego_id)
            ln = traci.vehicle.getLaneID(env.ego_id)
            rec.update(x=ex, y=ey, ang=traci.vehicle.getAngle(env.ego_id), v=traci.vehicle.getSpeed(env.ego_id),
                       lane=ln, edge=traci.vehicle.getRoadID(env.ego_id), vmax=traci.lane.getMaxSpeed(ln), veh=[])
        except Exception:
            last = steps[-1] if steps else None
            if last is None:
                break
            rec.update({kk: last[kk] for kk in ("x", "y", "ang", "v", "lane", "edge", "vmax")}, veh=[])
        steps.append(rec)
        traci.gui.screenshot(view, os.path.join(frames_dir, f"{k:05d}.png"), w, h)
        k += 1
        if max_steps and k >= max_steps:
            break
    try:
        traci.simulationStep()          # 마지막 캡처가 써지도록
    except Exception:
        pass
    end = ("collision" if info.get("collided") else "teleported" if info.get("teleported") else
           "red_light" if info.get("red_light") else "wrong_lane" if info.get("wrong_lane") else
           "arrival" if info.get("arrived") else "timeout")
    pickle.dump({"steps": steps, "route": route, "end": end, "seed": seed, "model": model, "dt": 0.5,
                 "road_details": os.environ.get("ROAD_DETAILS", "")}, open(out_pkl, "wb"))
    print("gui logged", len(steps), "steps, end", end, flush=True)
    try:
        traci.close(False)
    except Exception:
        pass
    os._exit(0)


# ───────────────────────── 2) 그리기 ─────────────────────────
def _offset_poly(shape, width):
    """차로 중심선 + 폭 → 다각형 (양쪽으로 width/2)."""
    pts = np.asarray(shape, dtype=float)
    if len(pts) < 2:
        return None
    left, right = [], []
    for i in range(len(pts)):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == len(pts) - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        n = np.hypot(*d)
        if n == 0:
            continue
        nx, ny = -d[1] / n, d[0] / n
        left.append(pts[i] + np.array([nx, ny]) * width / 2)
        right.append(pts[i] - np.array([nx, ny]) * width / 2)
    if len(left) < 2:
        return None
    return np.vstack([left, right[::-1]])


def build_geometry(sumo_dir, route):
    import sumolib
    net = sumolib.net.readNet(os.path.join(sumo_dir, "highway.net.xml"), withInternal=True)
    lanes, marks, junc, edges_map = [], [], [], []
    for e in net.getEdges(withInternal=True):
        internal = e.getFunction() == "internal"
        for ln in e.getLanes():
            p = _offset_poly(ln.getShape(), ln.getWidth())
            if p is not None:
                bus = ln.allows("bus") and not ln.allows("passenger")
                lanes.append((p, bus))
        if not internal:
            edges_map.append((np.asarray(e.getShape()), e.getPriority(), e.getLaneNumber()))
            ls = e.getLanes()
            for ln in ls[1:]:            # 차로 사이 점선 (오른쪽 차로의 왼쪽 경계)
                sh = np.asarray(ln.getShape(), dtype=float)
                if len(sh) < 2:
                    continue
                off = _offset_poly(sh, ln.getWidth())
                if off is None:
                    continue
                n = len(off) // 2
                marks.append(off[n:][::-1] if False else off[n:][::-1])
    for j in net.getNodes():
        sh = j.getShape()
        if len(sh) >= 3:
            junc.append(np.asarray(sh, dtype=float))
    # 경로 중심선과 누적 거리
    rpts = []
    for eid in route:
        e = net.getEdge(eid)
        sh = list(e.getLanes()[0].getShape())
        rpts += sh if not rpts else sh[1:]
    rpts = np.asarray(rpts, dtype=float)
    names = {}
    for eid in route:
        e = net.getEdge(eid)
        names[eid] = e.getName() or ""
    xmin, ymin, xmax, ymax = net.getBoundary()
    return {"lanes": lanes, "marks": marks, "junc": junc, "edges": edges_map, "route_pts": rpts,
            "names": names, "bbox": (xmin, ymin, xmax, ymax), "net": None}


def interaction_mask(steps, _unused=None):
    """천천히 보여 줄 장면: 움직이면서 (신호 교차로 30 m 안 / 교차로 안 / 꺾는 교차로 25 m 안 / 앞차 12 m 안)
    이거나, 차로 변경, 세게 가감속(1.5 m/s² 넘음). 신호 대기처럼 서 있는 시간과 그냥 달리는 구간은 빠르게."""
    n = len(steps)
    m = np.zeros(n, bool)
    for i, s in enumerate(steps):
        ex, ey, v = s["x"], s["y"], s["v"]
        th = math.radians(s["ang"]); fx, fy = math.sin(th), math.cos(th)
        front = False
        for (_vid, x, y, _a, _sp) in s["veh"]:
            dx, dy = x - ex, y - ey
            along = dx * fx + dy * fy
            if 0 < along < 12.0 and abs(dx * fy - dy * fx) < 2.5:
                front = True
                break
        o = s["obs"]
        sig_d = 150.0 * float(o[37]) ** 2 if o[37] < 0.999 else 999.0
        turn_d = 300.0 * float(o[40]) if o[40] < 0.999 else 999.0
        dv = abs(steps[i + 1]["v"] - v) / 0.5 if i + 1 < n else 0.0
        moving = v > 0.5
        m[i] = (moving and (sig_d < 30.0 or s["lane"].startswith(":") or turn_d < 25.0 or (front and v > 1.0))
                or s["lc"] != 0 or dv > 1.5)
    w = 6
    mm = m.copy()
    for i in np.where(m)[0]:
        mm[max(0, i - w):i + w + 1] = True
    gap = 0
    for i in range(n):
        if not mm[i]:
            gap += 1
        else:
            if 0 < gap <= 6:
                mm[i - gap:i] = True
            gap = 0
    return mm


def plan_frames(mask, dt, seconds, fast, fps):
    """스텝마다 배속 정하기: 상호작용 = base, 아니면 base*fast. 전체가 seconds초가 되게 base를 고른다."""
    n = len(mask)
    sim_slow = mask.sum() * dt
    sim_fast = (~mask).sum() * dt
    base = (sim_slow + sim_fast / fast) / seconds
    mult = np.where(mask, base, base * fast)
    # 배속 바뀔 때 1.5초(실제) 동안 부드럽게
    k = max(1, int(1.5 / dt))
    sm = np.convolve(np.pad(mult, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    vt = np.concatenate([[0.0], np.cumsum(dt / sm)])     # 각 스텝 시작의 영상 시각
    total = vt[-1]
    frames = []
    for f in range(int(total * fps)):
        t = f / fps
        i = int(np.searchsorted(vt, t, side="right") - 1)
        i = min(max(i, 0), n - 1)
        a = (t - vt[i]) / max(vt[i + 1] - vt[i], 1e-9)
        frames.append((i, float(min(max(a, 0.0), 1.0)), float(sm[i])))
    return frames, base


def _veh_poly(x, y, ang, L, W):
    """SUMO 위치 = 앞 범퍼 가운데, 각도 0 = 북쪽, 시계방향."""
    th = math.radians(ang)
    fx, fy = math.sin(th), math.cos(th)
    rx, ry = fy, -fx
    cx, cy = x - fx * L / 2, y - fy * L / 2
    hw = W / 2
    return [(cx + fx * L / 2 + rx * hw, cy + fy * L / 2 + ry * hw), (cx + fx * L / 2 - rx * hw, cy + fy * L / 2 - ry * hw),
            (cx - fx * L / 2 - rx * hw, cy - fy * L / 2 - ry * hw), (cx - fx * L / 2 + rx * hw, cy - fy * L / 2 + ry * hw)]


ROAD_KO = {"월드컵대로": "월드컵대로", "중앙대로": "중앙대로", "연수로": "연수로", "중앙천로": "중앙천로", "신촌로": "신촌로",
           "거제천로": "거제천로", "반송로": "반송로", "쌍미천로": "쌍미천로", "고분로": "고분로", "세병로": "세병로"}


def render_chunk(args):
    (pkl, geo_pkl, frames, out_dir, start_idx, W, H, view_m) = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "wqy-zenhei" in f.lower() or "wenquanyi" in f.lower():
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    D = pickle.load(open(pkl, "rb")); G = pickle.load(open(geo_pkl, "rb"))
    steps, dims, tls_links = D["steps"], D["dims"], D["tls_links"]
    BG, ROAD, BUS, MARK, JUNC = "#1d2421", "#4a524d", "#5a4a3a", "#c9cfc9", "#454d48"
    INK, MUTED, ACC, EGO, OTHER = "#eef1ee", "#9aa39c", "#58b4ff", "#ff3d7f", "#c9d3da"
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)
    axv = fig.add_axes([0.0, 0.0, H / W, 1.0]); axv.set_facecolor(BG)
    axm = fig.add_axes([H / W + 0.012, 0.30, 1 - H / W - 0.024, 0.62]); axm.set_facecolor(BG)
    axh = fig.add_axes([H / W + 0.012, 0.0, 1 - H / W - 0.024, 0.29]); axh.set_facecolor(BG)
    for a in (axv, axm, axh):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    # 확대 화면: 도로 (한 번만 추가, 축 범위만 바꿈)
    lane_polys = [p for p, bus in G["lanes"]]
    axv.add_collection(PolyCollection([p for p, b in G["lanes"] if not b], facecolors=ROAD, edgecolors="none"))
    axv.add_collection(PolyCollection([p for p, b in G["lanes"] if b], facecolors=BUS, edgecolors="none"))
    axv.add_collection(PolyCollection(G["junc"], facecolors=JUNC, edgecolors="none"))
    axv.add_collection(LineCollection(G["marks"], colors=MARK, linewidths=0.8, linestyles=(0, (4, 6)), alpha=0.7))
    axv.set_aspect("equal")
    # 전체 지도
    xmin, ymin, xmax, ymax = G["bbox"]
    segs = [e[0] for e in G["edges"]]
    lw = [0.5 + 0.35 * min(e[2], 4) for e in G["edges"]]
    axm.add_collection(LineCollection(segs, colors="#3c4540", linewidths=lw))
    rp = G["route_pts"]
    axm.plot(rp[:, 0], rp[:, 1], color=ACC, lw=2.2, alpha=0.45, solid_capstyle="round")
    axm.plot(rp[0, 0], rp[0, 1], "o", color=ACC, ms=7)
    axm.plot(rp[-1, 0], rp[-1, 1], "^", color=ACC, ms=8)
    axm.text(rp[0, 0] + 40, rp[0, 1], "출발", color=INK, fontsize=10, va="center")
    axm.text(rp[-1, 0] + 40, rp[-1, 1] - 30, "도착", color=INK, fontsize=10, va="center")
    pad = 120
    rx0, rx1, ry0, ry1 = rp[:, 0].min() - pad, rp[:, 0].max() + pad, rp[:, 1].min() - pad, rp[:, 1].max() + pad
    axm.set_xlim(rx0, rx1); axm.set_ylim(ry0, ry1); axm.set_aspect("equal")
    axm.set_title("전체 경로 (6.7 km, 북쪽 위)", color=INK, fontsize=12, loc="left")
    trail, = axm.plot([], [], color=ACC, lw=3.0, solid_capstyle="round")
    egodot, = axm.plot([], [], "o", color=EGO, ms=10, mec="white", mew=1.5)
    box, = axm.plot([], [], color=EGO, lw=1.0)
    ego_xy = np.array([[s["x"], s["y"]] for s in steps])
    dyn = []
    names = G["names"]
    total_steps = len(steps)
    for fi, (i, a, mult) in enumerate(frames):
        for d in dyn:
            d.remove()
        dyn = []
        s = steps[i]; s2 = steps[min(i + 1, total_steps - 1)]
        ex = s["x"] + (s2["x"] - s["x"]) * a; ey = s["y"] + (s2["y"] - s["y"]) * a
        half = view_m / 2
        axv.set_xlim(ex - half, ex + half); axv.set_ylim(ey - half, ey + half)
        # 신호 (정지선 색)
        segs_c = []
        for t, links in tls_links.items():
            st = s["sig"].get(t)
            if not st:
                continue
            for (li, inc) in links:
                lp = G["lane_end"].get(inc)
                if lp is None:
                    continue
                (px, py), (nx, ny), w = lp
                if abs(px - ex) > half + 10 or abs(py - ey) > half + 10:
                    continue
                c = st[li] if li < len(st) else "r"
                col = "#2fd06f" if c in "Gg" else "#ffcc33" if c in "yY" else "#ff4d4d" if c in "rRs" else "#888888"
                segs_c.append(([(px + nx * w / 2, py + ny * w / 2), (px - nx * w / 2, py - ny * w / 2)], col))
        if segs_c:
            lc = LineCollection([p for p, _ in segs_c], colors=[c for _, c in segs_c], linewidths=4.0, zorder=4)
            axv.add_collection(lc); dyn.append(lc)
        # 다른 차
        prev = {v[0]: v for v in s2["veh"]}
        polys, cols = [], []
        for (vid, x, y, ang, sp) in s["veh"]:
            if abs(x - ex) > half + 15 or abs(y - ey) > half + 15:
                continue
            if vid in prev:
                x2, y2 = prev[vid][1], prev[vid][2]
                x, y = x + (x2 - x) * a, y + (y2 - y) * a
            L, Wd, vc = dims.get(vid, (5.0, 1.8, "passenger"))
            polys.append(_veh_poly(x, y, ang, L, Wd))
            cols.append("#f2a65a" if vc == "bus" else "#ffe08a" if vc == "taxi" else OTHER)
        if polys:
            pc = PolyCollection(polys, facecolors=cols, edgecolors="#1d2421", linewidths=0.6, zorder=5)
            axv.add_collection(pc); dyn.append(pc)
        L, Wd = D["ego_dims"]
        pe = PolyCollection([_veh_poly(ex, ey, s["ang"], L, Wd)], facecolors=EGO, edgecolors="white", linewidths=1.4, zorder=6)
        axv.add_collection(pe); dyn.append(pe)
        # 확대 화면 글씨
        t_sim = (i + a) * D["dt"]
        rname = names.get(s["edge"], "") or ("교차로 안" if s["edge"].startswith(":") else "")
        dyn.append(axv.text(0.02, 0.975, f"{int(t_sim // 60):02d}:{int(t_sim % 60):02d}  (주행 시간)", transform=axv.transAxes,
                            color=INK, fontsize=15, va="top", fontweight="bold",
                            bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=4)))
        slow = mult < 1.5 * frames_base
        dyn.append(axv.text(0.98, 0.975, f"×{mult:.0f}  {'차량·교차로 구간' if slow else ('빠르게 (신호 대기)' if s['v'] < 0.5 else '빠르게 (직진)')}", transform=axv.transAxes,
                            color=("#ffcc33" if not slow else "#2fd06f"), fontsize=14, va="top", ha="right", fontweight="bold",
                            bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=4)))
        dyn.append(axv.text(0.02, 0.025, "AV(분홍)  일반 차(회색)  버스(주황)  택시(노랑)  정지선 = 신호 색", transform=axv.transAxes,
                            color=MUTED, fontsize=10.5, va="bottom",
                            bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=3)))
        # 지도
        j = i + 1
        trail.set_data(ego_xy[:j, 0], ego_xy[:j, 1])
        egodot.set_data([ex], [ey])
        box.set_data([ex - half, ex + half, ex + half, ex - half, ex - half], [ey - half, ey - half, ey + half, ey + half, ey - half])
        # 상태판
        axh.cla(); axh.set_facecolor(BG); axh.set_xticks([]); axh.set_yticks([])
        for sp in axh.spines.values():
            sp.set_visible(False)
        axh.set_xlim(0, 1); axh.set_ylim(0, 1)
        v = s["v"] + (s2["v"] - s["v"]) * a
        lim = s.get("vmax", 13.89)
        prog = 100.0 * (1 - float(s["obs"][38])) if len(s["obs"]) > 38 else 0
        axh.text(0.0, 0.92, "최종 모델 RL2 (yeonsan_loopB_RL2) · 루프 경로 B", color=INK, fontsize=13, fontweight="bold", va="top")
        axh.text(0.0, 0.72, f"속도 {v * 3.6:4.0f} km/h   (제한 {lim * 3.6:.0f})", color=INK, fontsize=12, va="top")
        axh.text(0.0, 0.54, f"도로  {rname}", color=INK, fontsize=12, va="top")
        axh.text(0.0, 0.36, f"진행 {prog:4.0f}%", color=INK, fontsize=12, va="top")
        axh.add_patch(plt.Rectangle((0.28, 0.27), 0.70, 0.07, color="#3c4540"))
        axh.add_patch(plt.Rectangle((0.28, 0.27), 0.70 * max(0, min(prog, 100)) / 100, 0.07, color=ACC))
        axh.text(0.0, 0.14, f"SUMO 1.27.1 · 순환 교통 + 운전자 성향·택시·잠깐 정차 · 시드 {D['seed']}", color=MUTED, fontsize=9.5, va="top")
        fig.savefig(os.path.join(out_dir, f"f{start_idx + fi:05d}.png"), dpi=dpi, facecolor=BG)
    plt.close(fig)
    return len(frames)


frames_base = 1.0


def do_render(pkl, sumo_dir, out_mp4, seconds=58.0, fast=3.0, workers=4, fps=25, W=1280, H=720, view_m=170.0):
    global frames_base
    import multiprocessing as mp
    D = pickle.load(open(pkl, "rb"))
    G = build_geometry(sumo_dir, D["route"])
    # 신호 정지선 위치: 들어오는 차로 끝점과 법선
    import sumolib
    net = sumolib.net.readNet(os.path.join(sumo_dir, "highway.net.xml"), withInternal=True)
    lane_end = {}
    for t, links in D["tls_links"].items():
        for (_i, inc) in links:
            if inc in lane_end:
                continue
            try:
                ln = net.getLane(inc)
            except Exception:
                continue
            sh = ln.getShape()
            (x1, y1), (x2, y2) = sh[-2], sh[-1]
            d = math.hypot(x2 - x1, y2 - y1) or 1.0
            lane_end[inc] = ((x2, y2), (-(y2 - y1) / d, (x2 - x1) / d), ln.getWidth())
    G["lane_end"] = lane_end
    tmp = out_mp4 + "_frames"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    geo_pkl = os.path.join(tmp, "geo.pkl"); pickle.dump(G, open(geo_pkl, "wb"))
    mask = interaction_mask(D["steps"], None)
    frames, base = plan_frames(mask, D["dt"], seconds, fast, fps)
    frames_base = base
    keep = int(os.environ.get("ONLY_FRAMES", "0"))
    if keep:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, keep).astype(int)]
    print(f"steps {len(D['steps'])}, 느린 구간 {mask.mean() * 100:.0f}%, 기본 배속 ×{base:.1f}, 빠른 배속 ×{base * fast:.1f}, "
          f"프레임 {len(frames)} ({len(frames) / fps:.1f}초)", flush=True)
    n = len(frames); per = math.ceil(n / workers)
    tasks = [(pkl, geo_pkl, frames[i:i + per], tmp, i, W, H, view_m) for i in range(0, n, per)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_set_base, initargs=(base,)) as pool:
        pool.map(render_chunk, tasks)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium", "-movflags", "+faststart",
                    out_mp4], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out_mp4, f"{os.path.getsize(out_mp4) / 1e6:.1f} MB", flush=True)
    return {"slow_frac": float(mask.mean()), "base_mult": float(base), "fast_mult": float(base * fast),
            "seconds": len(frames) / fps, "steps": len(D["steps"]), "end": D["end"], "seed": D["seed"]}


def _set_base(b):
    global frames_base
    frames_base = b


# ───────────────────────── 3) sumo-gui 화면 + 지도 패널 합성 ─────────────────────────
def gui_mask(steps):
    """천천히 보여 줄 스텝 (gui 기록용: 앞차 거리는 관측 9번(앞차 거리/100 m)으로 판단)."""
    n = len(steps)
    m = np.zeros(n, bool)
    for i, s in enumerate(steps):
        o, v = s["obs"], s["v"]
        sig_d = 150.0 * float(o[37]) ** 2 if o[37] < 0.999 else 999.0
        turn_d = 300.0 * float(o[40]) if o[40] < 0.999 else 999.0
        front = float(o[9]) < 0.12
        dv = abs(steps[i + 1]["v"] - v) / 0.5 if i + 1 < n else 0.0
        moving = v > 0.5
        m[i] = (moving and (sig_d < 30.0 or s["lane"].startswith(":") or turn_d < 25.0 or (front and v > 1.0))
                or s["lc"] != 0 or dv > 1.5)
    w = 6
    mm = m.copy()
    for i in np.where(m)[0]:
        mm[max(0, i - w):i + w + 1] = True
    gap = 0
    for i in range(n):
        if not mm[i]:
            gap += 1
        else:
            if 0 < gap <= 6:
                mm[i - gap:i] = True
            gap = 0
    return mm


def compose_chunk(args):
    (pkl, geo_pkl, frames_dir, frames, out_dir, start_idx, base) = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont
    fpath = None
    for f in font_manager.findSystemFonts():
        if "wqy-zenhei" in f.lower():
            fpath = f
            font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    F = lambda sz: ImageFont.truetype(fpath, sz) if fpath else ImageFont.load_default()
    f_big, f_mid, f_small = F(26), F(20), F(15)
    D = pickle.load(open(pkl, "rb")); G = pickle.load(open(geo_pkl, "rb"))
    steps = D["steps"]; n = len(steps)
    BG, INK, MUTED, ACC, EGO = "#1d2421", "#eef1ee", "#9aa39c", "#58b4ff", "#ff3d7f"
    PW, PH = 500, 900
    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100, facecolor=BG)
    axm = fig.add_axes([0.04, 0.36, 0.92, 0.58]); axm.set_facecolor(BG)
    axh = fig.add_axes([0.06, 0.02, 0.90, 0.32]); axh.set_facecolor(BG)
    for a in (axm, axh):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    segs = [e[0] for e in G["edges"]]
    lw = [0.5 + 0.35 * min(e[2], 4) for e in G["edges"]]
    axm.add_collection(LineCollection(segs, colors="#3c4540", linewidths=lw))
    rp = G["route_pts"]
    axm.plot(rp[:, 0], rp[:, 1], color=ACC, lw=2.2, alpha=0.4, solid_capstyle="round")
    axm.plot(rp[0, 0], rp[0, 1], "o", color=ACC, ms=7); axm.plot(rp[-1, 0], rp[-1, 1], "^", color=ACC, ms=8)
    axm.text(rp[0, 0] + 40, rp[0, 1], "출발", color=INK, fontsize=10, va="center")
    axm.text(rp[-1, 0] + 40, rp[-1, 1] - 30, "도착", color=INK, fontsize=10, va="center")
    pad = 120
    axm.set_xlim(rp[:, 0].min() - pad, rp[:, 0].max() + pad); axm.set_ylim(rp[:, 1].min() - pad, rp[:, 1].max() + pad)
    axm.set_aspect("equal")
    axm.set_title("전체 경로 (6.7 km, 북쪽 위)", color=INK, fontsize=13, loc="left")
    trail, = axm.plot([], [], color=ACC, lw=3.0, solid_capstyle="round")
    egodot, = axm.plot([], [], "o", color=EGO, ms=11, mec="white", mew=1.5)
    exy = np.array([[s["x"], s["y"]] for s in steps])
    names = G["names"]
    for fi, (i, _a, mult) in enumerate(frames):
        j = min(i + 1, n - 1)                 # 스크린샷 i = (i+1)번째 스텝 뒤 화면
        s = steps[j]
        trail.set_data(exy[:j + 1, 0], exy[:j + 1, 1]); egodot.set_data([s["x"]], [s["y"]])
        axh.cla(); axh.set_facecolor(BG); axh.set_xticks([]); axh.set_yticks([]); axh.set_xlim(0, 1); axh.set_ylim(0, 1)
        for sp in axh.spines.values():
            sp.set_visible(False)
        rname = names.get(s["edge"], "") or ("교차로 안" if s["edge"].startswith(":") else "")
        prog = 100.0 * (1 - float(s["obs"][38]))
        axh.text(0, 0.97, "최종 모델 RL2 · 루프 경로 B", color=INK, fontsize=14, fontweight="bold", va="top")
        axh.text(0, 0.80, f"속도  {s['v'] * 3.6:4.0f} km/h   (제한 {s['vmax'] * 3.6:.0f})", color=INK, fontsize=13, va="top")
        axh.text(0, 0.65, f"도로  {rname}", color=INK, fontsize=13, va="top")
        axh.text(0, 0.50, f"진행  {prog:3.0f}%", color=INK, fontsize=13, va="top")
        axh.add_patch(plt.Rectangle((0.32, 0.405), 0.66, 0.07, color="#3c4540"))
        axh.add_patch(plt.Rectangle((0.32, 0.405), 0.66 * max(0, min(prog, 100)) / 100, 0.07, color=ACC))
        axh.text(0, 0.28, "SUMO 1.27.1 (sumo-gui 실제 화면)", color=MUTED, fontsize=10.5, va="top")
        axh.text(0, 0.17, "순환 교통 + 운전자 성향·택시·잠깐 정차", color=MUTED, fontsize=10.5, va="top")
        axh.text(0, 0.06, f"평가 시드 {D['seed']} · 결과: 완주", color=MUTED, fontsize=10.5, va="top")
        fig.canvas.draw()
        panel = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        try:
            shot = Image.open(os.path.join(frames_dir, f"{i:05d}.png")).convert("RGB")
        except Exception:
            shot = Image.new("RGB", (1100, 900), (0, 0, 0))
        img = Image.new("RGB", (1600, 900), BG)
        img.paste(shot.resize((1100, 900)), (0, 0)); img.paste(panel, (1100, 0))
        d = ImageDraw.Draw(img, "RGBA")
        t_sim = (j + 1) * D["dt"]
        d.rectangle([0, 0, 1100, 50], fill=(20, 24, 22, 215))
        d.text((16, 10), f"{int(t_sim // 60):02d}:{int(t_sim % 60):02d}  주행 시간", font=f_big, fill=(238, 241, 238))
        slow = mult < 1.5 * base
        lab = f"실제의 ×{mult:.0f}  " + ("차량·교차로 구간" if slow else ("빠르게 (신호 대기)" if s["v"] < 0.5 else "빠르게 (직진)"))
        col = (47, 208, 111) if slow else (255, 204, 51)
        tw = d.textlength(lab, font=f_mid)
        d.text((1100 - 16 - tw, 14), lab, font=f_mid, fill=col)
        d.rectangle([0, 868, 1100, 900], fill=(20, 24, 22, 200))
        d.text((14, 873), "AV = 분홍 원 안의 빨간 차   ·   SUMO 화면이 AV를 따라감   ·   건물 = 분홍 면", font=f_small, fill=(200, 205, 200))
        img.save(os.path.join(out_dir, f"c{start_idx + fi:05d}.png"))
    plt.close(fig)
    return len(frames)


def do_compose(pkl, frames_dir, sumo_dir, out_mp4, seconds=58.0, fast=3.0, workers=4, fps=25):
    import multiprocessing as mp
    D = pickle.load(open(pkl, "rb"))
    G = build_geometry(sumo_dir, D["route"])
    mask = gui_mask(D["steps"])
    nshots = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    frames, base = plan_frames(mask, D["dt"], seconds, fast, fps)
    frames = [(min(i, nshots - 1), a, m) for (i, a, m) in frames]
    print(f"steps {len(D['steps'])}, 화면 {nshots}, 느린 구간 {mask.mean() * 100:.0f}%, 배속 ×{base:.1f} / ×{base * fast:.1f}, "
          f"프레임 {len(frames)} ({len(frames) / fps:.1f}초)", flush=True)
    tmp = out_mp4 + "_frames"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    geo_pkl = os.path.join(tmp, "geo.pkl"); pickle.dump(G, open(geo_pkl, "wb"))
    per = math.ceil(len(frames) / workers)
    tasks = [(pkl, geo_pkl, frames_dir, frames[k:k + per], tmp, k, base) for k in range(0, len(frames), per)]
    with mp.get_context("fork").Pool(workers) as pool:
        pool.map(compose_chunk, tasks)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "c%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium", "-movflags", "+faststart",
                    out_mp4], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    info = {"source": "sumo-gui screenshots", "slow_frac": float(mask.mean()), "base_mult": float(base),
            "fast_mult": float(base * fast), "seconds": len(frames) / fps, "steps": len(D["steps"]), "end": D["end"], "seed": D["seed"]}
    print("wrote", out_mp4, f"{os.path.getsize(out_mp4) / 1e6:.1f} MB", info, flush=True)
    return info


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compose":
        import json
        info = do_compose(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        json.dump(info, open(os.path.splitext(sys.argv[5])[0] + ".json", "w"), ensure_ascii=False, indent=1)
    if cmd == "gui":
        do_gui(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6],
               max_steps=int(os.environ.get("MAX_REC", "0")))
    if cmd == "log":
        do_log(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif cmd == "render":
        import argparse, json
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("pkl"); ap.add_argument("sumo_dir"); ap.add_argument("out")
        ap.add_argument("--seconds", type=float, default=58.0); ap.add_argument("--fast", type=float, default=3.0)
        ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
        a = ap.parse_args()
        info = do_render(a.pkl, a.sumo_dir, a.out, a.seconds, a.fast, a.workers)
        json.dump(info, open(os.path.splitext(a.out)[0] + ".json", "w"), ensure_ascii=False, indent=1)
        print(info)
