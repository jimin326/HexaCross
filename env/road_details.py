"""도로 디테일 (road_config.DETAILS, 기본은 전부 끔).

실제 도심 도로에 있는 요소를 배경 교통에 더한다. 모두 AV 입장에서는 "앞·옆에 있는 차"로만 보이므로
AV 관측(56개) 형식은 그대로이고, 기존 모델을 다시 학습하지 않고 그대로 평가할 수 있다.

    bus_stops      버스 정류장 정차. 시내버스는 가로변 차로 정류장, BRT 버스는 BRT 차로 정류장에 선다.
    parked         불법 주정차. 에피소드 내내 가로변 차로 한 칸을 막고 서 있는 차.
    short_stops    잠깐 정차한 차 (택배·비상등 켜고 정차). 일반 차 일부가 가로변 차로에 30~120초 선다.
    taxi           택시. 실측 OD의 택시 대수를 택시 차종으로 바꾸고, 그중 일부가 손님을 태우거나 내리려고 선다.
    driver_styles  운전자 성향. 공격적인 운전자(짧은 차간·끼어들기·황색 통과)와 조심스러운 운전자를 섞는다.
    emergency      긴급차량 (구급차·소방차). 경광등(bluelight) 장치가 있어 주변 차가 길을 비켜 준다.
    slope          도로 경사. 기준점 고도로 도로망 높이를 넣고, 버스·소방차는 경사를 반영하는 KraussPS 모델로 달린다.

켜는 법:
    1) env/road_config.py의 DETAILS에서 항목마다 "enabled": True
    2) 또는 환경변수 (설정 파일을 안 고치고 평가할 때):
         ROAD_DETAILS=all                     전부 켬 (slope는 고도 기준점이 있어야 함)
         ROAD_DETAILS=bus_stops,parked        고른 것만 켬 (나머지는 끔)
         ROAD_DETAILS=none                    전부 끔
       환경변수가 있으면 DETAILS의 enabled보다 우선한다.

※ 기본값(전부 끔)이면 이 파일은 도로·교통 파일을 하나도 바꾸지 않는다 (학습 중인 환경 그대로).
※ 정류장·주정차 위치와 비율은 실측이 아닌 가정값이다 (DETAILS 설명 참고).
"""
import copy
import math
import os
import random

ALL = ("bus_stops", "parked", "short_stops", "taxi", "driver_styles", "emergency", "slope")


def config():
    """DETAILS 설정 (+ ROAD_DETAILS 환경변수)."""
    try:
        from env.road_config import DETAILS as base
    except ImportError:
        base = {}
    d = copy.deepcopy(base)
    env = os.environ.get("ROAD_DETAILS")
    if env is not None:
        s = env.strip()
        on = set(ALL) if s == "all" else set() if s in ("", "none") else {x.strip() for x in s.split(",") if x.strip()}
        bad = sorted(on - set(ALL))
        if bad:
            raise ValueError(f"ROAD_DETAILS에 모르는 항목 {bad}. 쓸 수 있는 것: {', '.join(ALL)}, all, none")
        if s == "all" and not ((d.get("slope") or {}).get("points")):
            on.discard("slope")          # 고도 기준점이 없으면 all에서 경사는 뺀다
        for k in ALL:
            d.setdefault(k, {})["enabled"] = k in on
    return d


def enabled(d, key):
    return bool((d.get(key) or {}).get("enabled"))


def any_enabled(d):
    return any(enabled(d, k) for k in ALL)


def tag(d=None):
    """켜진 항목 이름 (평가 폴더 이름·기록용). 전부 끄면 'none'."""
    d = config() if d is None else d
    on = [k for k in ALL if enabled(d, k)]
    return "+".join(on) if on else "none"


# ──────────────────────────────────────────────────────────────
# 1) 차종 (vType)
# ──────────────────────────────────────────────────────────────
def _vtype(vid, attrs, params=None):
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    if not params:
        return f'    <vType id="{vid}" {a}/>'
    p = "".join(f'\n        <param key="{k}" value="{v}"/>' for k, v in params.items())
    return f'    <vType id="{vid}" {a}>{p}\n    </vType>'


def style_vtype_lines(traffic, d, base_total):
    """운전자 성향 vType (carDist 안에 들어갈 줄). base_total: 기존 컨트롤러 확률을 줄인 뒤 남는 몫."""
    cfg = d.get("driver_styles") or {}
    lines = []
    for name, st in (cfg.get("styles") or {}).items():
        attrs = {"id": f"car_{name}", "probability": f'{float(st["share"]):.4f}',
                 "carFollowModel": st.get("carFollowModel", "Krauss"),
                 "accel": st.get("accel", 2.6), "decel": st.get("decel", 4.5),
                 "tau": st.get("tau", 1.0), "minGap": st.get("min_gap", 2.0),
                 "maxSpeed": st.get("max_speed", traffic["max_speed"]),
                 "speedFactor": st.get("speed_factor", 1.0), "speedDev": st.get("speed_dev", 0.1),
                 "color": st.get("color", "1,1,1"), "vClass": "passenger", "guiShape": "passenger/sedan"}
        if "sigma" in st:
            attrs["sigma"] = st["sigma"]
        for k in ("lcAssertive", "lcSpeedGain", "lcImpatience", "lcCooperative", "lcKeepRight",
                  "jmDriveAfterYellowTime", "jmDriveAfterRedTime", "jmIgnoreFoeProb", "jmIgnoreFoeSpeed",
                  "impatience"):
            if k in st:
                attrs[k] = st[k]
        lines.append("        <vType " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>")
    return lines


def style_share(d):
    if not enabled(d, "driver_styles"):
        return 0.0
    s = sum(float(st["share"]) for st in ((d.get("driver_styles") or {}).get("styles") or {}).values())
    if not 0.0 <= s < 1.0:
        raise ValueError(f"DETAILS['driver_styles'] 비율 합 {s}는 0 이상 1 미만이어야 합니다.")
    return s


def bus_cfg_with_slope(bus_cfg, d):
    """경사를 켜면 버스를 KraussPS(경사 반영 Krauss)로."""
    if not enabled(d, "slope"):
        return bus_cfg
    model = (d.get("slope") or {}).get("heavy_model", "KraussPS")
    return {**bus_cfg, "vtype": {**(bus_cfg.get("vtype") or {}), "carFollowModel": model}}


# ──────────────────────────────────────────────────────────────
# 2) 길가 위치 고르기
# ──────────────────────────────────────────────────────────────
def _bus_only(ln):
    return ln.allows("bus") and not ln.allows("passenger")


def _far_from_center(edge, cx, cy, r):
    (x0, y0), (x1, y1) = edge.getShape()[0], edge.getShape()[-1]
    return min(math.hypot(x0 - cx, y0 - cy), math.hypot(x1 - cx, y1 - cy)) >= r


class Curb:
    """차로마다 이미 쓴 구간을 기억해 정류장·주정차·정차가 겹치지 않게 한다."""

    def __init__(self, net, cx, cy, d, exclude_edges=()):
        self.net, self.cx, self.cy = net, cx, cy
        g = d.get("placement") or {}
        self.center_r = float(g.get("center_clear_m", 120.0))
        self.start_gap = float(g.get("start_gap_m", 25.0))
        self.end_gap = float(g.get("end_gap_m", 60.0))
        self.min_len = float(g.get("min_edge_m", 100.0))
        self.exclude = set(exclude_edges)
        self.used = {}                      # lane id -> [(s, e)]

    def car_lane(self, edge, strict=False):
        """승용차가 서는 가로변 차로 (맨 오른쪽 일반 차로). 옆으로 비켜 갈 일반 차로가 하나 이상 더 있어야 한다.
        strict(불법 주정차처럼 계속 서 있는 차): 맨 오른쪽 차로로만 갈 수 있는 방향이 있는 도로는 뺀다."""
        lanes = edge.getLanes()
        if not lanes or _bus_only(lanes[0]):
            return None                     # 가로변이 버스전용차로면 승용차 정차는 넣지 않는다
        car = [ln for ln in lanes if ln.allows("passenger")]
        if len(car) < 2:
            return None
        if not strict:
            return lanes[0]
        # 맨 오른쪽 차로로만 갈 수 있는 방향(우회전 전용 등)이 있으면 그 차로는 막지 않는다.
        # 막으면 그 방향으로 가는 차는 비켜 갈 방법이 없어 끝없이 기다린다 (실제 불법 주정차도 회전 차로 앞은 드묾).
        def outs(ln):
            return {c.getToLane().getEdge().getID() for c in ln.getOutgoing()}
        others = set().union(*(outs(ln) for ln in car[1:]))
        return lanes[0] if outs(lanes[0]) <= others else None

    def edge_ok(self, edge):
        return (edge.getID() not in self.exclude and edge.getFunction() == ""
                and edge.getLength() >= self.min_len and _far_from_center(edge, self.cx, self.cy, self.center_r))

    def free(self, lane, s, e):
        return all(e + 3.0 <= a or s - 3.0 >= b for a, b in self.used.get(lane.getID(), []))

    def take(self, lane, s, e):
        self.used.setdefault(lane.getID(), []).append((s, e))

    def spot(self, lane, length, rng, near_start=False, tries=12, end_gap=None):
        """lane 위에 길이 length의 빈 자리 (start, end) 또는 None. end_gap: 차로 끝(정지선)까지 비울 거리."""
        lo = self.start_gap
        hi = lane.getLength() - (self.end_gap if end_gap is None else end_gap) - length
        if hi < lo:
            return None
        for k in range(tries):
            s = lo + (min(15.0, hi - lo) * rng.random() if near_start and k == 0 else (hi - lo) * rng.random())
            if self.free(lane, s, s + length):
                self.take(lane, s, s + length)
                return s, s + length
        return None


# ──────────────────────────────────────────────────────────────
# 3) 버스 정류장
# ──────────────────────────────────────────────────────────────
def place_bus_stops(curb, bus_routes, d, rng):
    """bus_routes: [(rid, edges, brt)] → (정류장 {id: (lane id, start, end, 이름)}, {rid: [정류장 id, ...]})

    위치 규칙 (가정값): 경로를 따라 spacing_m마다, 교차로를 지난 직후(엣지 시작 쪽)에 둔다.
    여러 노선이 같은 엣지를 지나면 같은 정류장을 함께 쓴다. 교차로 중심 반경 center_clear_m 안에는 두지 않는다.
    시내버스는 맨 오른쪽 차로(가로변 버스전용차로 포함), BRT 버스는 BRT 차로에 선다.
    """
    cfg = d.get("bus_stops") or {}
    spacing = float(cfg.get("spacing_m", 400.0))
    length = float(cfg.get("length_m", 30.0))
    stops, on_edge, route_stops = {}, {}, {}
    for rid, edges, brt in bus_routes:
        acc, seq = spacing * 0.5, []
        for eid in edges:
            e = curb.net.getEdge(eid)
            acc += e.getLength()
            if acc < spacing or not curb.edge_ok(e):
                continue
            key = (eid, brt)
            if key not in on_edge:
                if brt:
                    lane = next((ln for ln in reversed(e.getLanes()) if _bus_only(ln)), None)
                else:
                    lane = e.getLanes()[0] if e.getLanes()[0].allows("bus") else None
                if lane is None:
                    continue
                sp = curb.spot(lane, length, rng, near_start=True)
                if sp is None:
                    continue
                sid = f"bs_{len(stops)}"
                stops[sid] = (lane.getID(), sp[0], sp[1], (e.getName() or eid) + (" BRT" if brt else ""))
                on_edge[key] = sid
            seq.append(on_edge[key])
            acc = 0.0
        route_stops[rid] = seq
    return stops, route_stops


def bus_stops_xml(stops):
    return "\n".join(f'    <busStop id="{sid}" lane="{lane}" startPos="{s:.1f}" endPos="{e:.1f}" name="{nm}" '
                     f'lines="hexacross" friendlyPos="true"/>' for sid, (lane, s, e, nm) in stops.items())


def bus_flow_lines(base_attrs, rate, poisson, stop_ids, d, rng):
    """정류장에 서는 버스 flow 줄들. 정차 시간을 다르게 하려고 같은 노선을 sub_flows개로 나눈다."""
    cfg = d.get("bus_stops") or {}
    lo, hi = cfg.get("dwell_s", (15, 40))
    n = max(1, int(cfg.get("sub_flows", 3)))
    out = []
    for j in range(n):
        r = rate / n
        arrival = f'period="exp({r / 3600.0:.6f})"' if poisson else f'vehsPerHour="{r:.2f}"'
        stops = "".join(f'\n        <stop busStop="{sid}" duration="{rng.uniform(lo, hi):.0f}"/>' for sid in stop_ids)
        out.append(f'    <flow {base_attrs.replace("@ID@", f"_{j}")} {arrival}>{stops}\n    </flow>')
    return out


# ──────────────────────────────────────────────────────────────
# 4) 택시·잠깐 정차·주정차·긴급차량 (rou.xml에 더할 줄)
# ──────────────────────────────────────────────────────────────
def carve_rates(demand, d):
    """승용차 OD에서 떼어 낼 대수 {(출발 방면, 도착 방면): {"taxi": 대/시, "short_stops": 대/시}}.

    택시는 실측 CSV의 택시 칸 (합계에 들어 있으므로 승용차 flow에서 빼서 택시 차종으로 바꾼다 — 총량 그대로).
    잠깐 정차는 승용차 OD의 share 비율.
    """
    out = {}
    if not (enabled(d, "taxi") or enabled(d, "short_stops")):
        return out
    import csv
    from env.road_builder import load_od_demand
    od = load_od_demand(demand)
    taxi = {}
    if enabled(d, "taxi"):
        hour = str(demand["scenarios"][demand["scenario"]]["hour"]).zfill(2)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), demand["csv"])
        scale = float(demand.get("scale", 1.0)) * float((d.get("taxi") or {}).get("scale", 1.0))
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["hour"] == hour:
                    taxi[(row["from"], row["to"])] = float(row["택시"]) * scale
    share = float((d.get("short_stops") or {}).get("share", 0.02)) if enabled(d, "short_stops") else 0.0
    for key, vph in od.items():
        t = min(taxi.get(key, 0.0), vph)
        s = (vph - t) * share
        out[key] = {"taxi": t, "short_stops": s}
    return out


def _stop_on_route(curb, edges, rng, length):
    """경로 위 가로변 차로의 빈 자리 하나: (lane id, endPos) 또는 None. 경로 앞뒤 두 엣지는 피한다."""
    cand = [curb.net.getEdge(e) for e in edges[1:-1]]
    rng.shuffle(cand)
    for e in cand:
        if not curb.edge_ok(e):
            continue
        lane = curb.car_lane(e)
        if lane is None:
            continue
        sp = curb.spot(lane, length, rng)
        if sp:
            return lane.getID(), sp[1]
    return None


def extra_traffic(net, curb, od_paths, d, poisson, depart_speed, rng, ego_edges=(), car_type="carDist"):
    """택시·잠깐 정차·주정차·긴급차량.

    od_paths: [(경로 id 앞부분, edges, {"taxi": 대/시, "short_stops": 대/시, "vph": 승용차 OD 대/시})]
    ego_edges: AV 경로가 지나는 엣지 (주정차 일부를 AV가 실제로 만나는 곳에 두려고)
    Returns: (vtype 줄, route 줄, flow·vehicle 줄, meta)
    """
    vtypes, routes, flows, meta = [], [], [], {}

    def arrival(r):
        return f'period="exp({r / 3600.0:.6f})"' if poisson else f'vehsPerHour="{r:.3f}"'

    def stop_xml(lane, end, dur):
        return f'\n        <stop lane="{lane}" startPos="{max(end - 10.0, 0.0):.1f}" endPos="{end:.1f}" duration="{dur:.0f}" parking="false"/>'

    # 택시
    if enabled(d, "taxi"):
        cfg = d.get("taxi") or {}
        vtypes.append(_vtype("taxi", {"vClass": "passenger", "guiShape": "taxi", "color": cfg.get("color", "1,0.55,0"),
                                      "accel": 2.6, "decel": 4.5, "tau": 1.0, "minGap": 2.0, "speedDev": 0.1,
                                      "carFollowModel": "Krauss", "sigma": 0.3, "lcKeepRight": 0}))
        p_stop = float(cfg.get("stop_share", 0.3))
        lo, hi = cfg.get("dwell_s", (10, 30))
        n_stop, total = 0, 0.0
        for k, (rid0, edges, rates) in enumerate(od_paths):
            r = rates.get("taxi", 0.0)
            if r <= 0:
                continue
            total += r
            rid = f"taxi_{k}"
            routes.append(f'    <route id="{rid}" edges="{" ".join(edges)}"/>')
            sp = _stop_on_route(curb, edges, rng, 10.0) if p_stop > 0 else None
            plain = r * (1.0 - p_stop) if sp else r        # 설 자리가 없으면 모두 서지 않는 택시로
            if plain > 0:
                flows.append(f'    <flow id="f_{rid}" type="taxi" route="{rid}" begin="0" end="100000" {arrival(plain)} '
                             f'departLane="best" departSpeed="{depart_speed}"/>')
            if sp:
                flows.append(f'    <flow id="f_{rid}_stop" type="taxi" route="{rid}" begin="0" end="100000" {arrival(r * p_stop)} '
                             f'departLane="best" departSpeed="{depart_speed}">{stop_xml(sp[0], sp[1], rng.uniform(lo, hi))}\n    </flow>')
                n_stop += 1
        meta["taxi"] = {"veh_per_hour": round(total, 1), "stop_share": p_stop, "stop_points": n_stop}

    # 잠깐 정차 (택배·비상등 정차)
    if enabled(d, "short_stops"):
        cfg = d.get("short_stops") or {}
        vtypes.append(_vtype("car_stopper", {"vClass": "passenger", "guiShape": "passenger/van", "color": cfg.get("color", "0.95,0.95,0.95"),
                                             "accel": 2.4, "decel": 4.5, "tau": 1.0, "minGap": 2.0, "speedDev": 0.1,
                                             "carFollowModel": "Krauss", "sigma": 0.3, "lcKeepRight": 0}))
        lo, hi = cfg.get("dwell_s", (30, 120))
        n, total = 0, 0.0
        for k, (rid0, edges, rates) in enumerate(od_paths):
            r = rates.get("short_stops", 0.0)
            if r <= 0:
                continue
            sp = _stop_on_route(curb, edges, rng, 10.0)
            rid = f"stopper_{k}"
            routes.append(f'    <route id="{rid}" edges="{" ".join(edges)}"/>')
            if sp is None:
                # 설 자리가 없는 경로(짧은 엣지뿐): 떼어 낸 대수를 일반 승용차로 되돌린다 (총량 그대로)
                flows.append(f'    <flow id="f_{rid}" type="{car_type}" route="{rid}" begin="0" end="100000" {arrival(r)} '
                             f'departLane="best" departSpeed="{depart_speed}"/>')
                continue
            flows.append(f'    <flow id="f_{rid}" type="car_stopper" route="{rid}" begin="0" end="100000" {arrival(r)} '
                         f'departLane="best" departSpeed="{depart_speed}">{stop_xml(sp[0], sp[1], rng.uniform(lo, hi))}\n    </flow>')
            n += 1
            total += r
        meta["short_stops"] = {"veh_per_hour": round(total, 1), "stop_points": n}

    # 긴급차량
    if enabled(d, "emergency"):
        cfg = d.get("emergency") or {}
        heavy = (d.get("slope") or {}).get("heavy_model", "KraussPS") if enabled(d, "slope") else "Krauss"
        kinds = {
            "ambulance": ({"vClass": "emergency", "guiShape": "emergency", "color": "1,1,1", "length": 6.0,
                           "accel": 2.6, "decel": 5.0, "maxSpeed": 22.0, "speedFactor": 1.3, "carFollowModel": "Krauss"},
                          float(cfg.get("ambulance_per_hour", 4.0))),
            "firetruck": ({"vClass": "emergency", "guiShape": "firebrigade", "color": "0.8,0.1,0.1", "length": 10.0,
                           "accel": 1.5, "decel": 4.5, "maxSpeed": 20.0, "speedFactor": 1.2, "carFollowModel": heavy},
                          float(cfg.get("firetruck_per_hour", 1.0))),
        }
        params = {"has.bluelight.device": "true"}
        total_od = sum(rates.get("vph", 0.0) for _, _, rates in od_paths) or 1.0
        meta["emergency"] = {}
        for kind, (attrs, per_hour) in kinds.items():
            if per_hour <= 0:
                continue
            vtypes.append(_vtype(kind, {**attrs, "sigma": 0.2, "minGap": 2.5}, params))
            n = 0
            for k, (rid0, edges, rates) in enumerate(od_paths):
                r = per_hour * rates.get("vph", 0.0) / total_od
                if r <= 0:
                    continue
                if not all(net.getEdge(e).allows("emergency") for e in edges):
                    continue
                rid = f"{kind}_{k}"
                routes.append(f'    <route id="{rid}" edges="{" ".join(edges)}"/>')
                flows.append(f'    <flow id="f_{rid}" type="{kind}" route="{rid}" begin="0" end="100000" {arrival(r)} '
                             f'departLane="best" departSpeed="{depart_speed}"/>')
                n += 1
            meta["emergency"][kind] = {"veh_per_hour": per_hour, "routes": n}

    # 불법 주정차 (에피소드 내내 서 있는 차)
    if enabled(d, "parked"):
        cfg = d.get("parked") or {}
        vtypes.append(_vtype("car_parked", {"vClass": "passenger", "guiShape": "passenger/sedan",
                                            "color": cfg.get("color", "0.55,0.55,0.6"), "length": 4.5, "minGap": 1.0}))
        cand = [e for e in net.getEdges() if curb.edge_ok(e) and curb.car_lane(e, strict=True) is not None]
        total_km = sum(e.getLength() for e in cand) / 1000.0
        n_target = min(int(round(total_km * float(cfg.get("per_km", 1.5)))), int(cfg.get("max_count", 40)))
        # ego_route_share만큼은 AV 경로 위 엣지에 먼저 둔다 (도로망 전체에 흩으면 AV가 거의 못 만난다)
        on_ego = set(ego_edges)
        ego_c = [e for e in cand if e.getID() in on_ego]
        rest = [e for e in cand if e.getID() not in on_ego]
        rng.shuffle(ego_c)
        rng.shuffle(rest)
        n_ego = min(len(ego_c), int(round(n_target * float(cfg.get("ego_route_share", 0.5)))))
        cand = ego_c[:n_ego] + rest + ego_c[n_ego:]
        placed = []
        for e in cand:
            if len(placed) >= n_target:
                break
            lane = curb.car_lane(e, strict=True)
            sp = curb.spot(lane, 6.0, rng, end_gap=float(cfg.get("end_gap_m", 60.0)))
            if sp is None:
                continue
            end = sp[1]
            vid = f"park_{len(placed)}"
            routes.append(f'    <route id="r_{vid}" edges="{e.getID()}"/>')
            flows.append(f'    <vehicle id="{vid}" type="car_parked" route="r_{vid}" depart="0" departLane="{lane.getIndex()}" '
                         f'departPos="{max(end - 6.0, 0.0):.1f}" departSpeed="0">{stop_xml(lane.getID(), end, 10_000_000)}\n    </vehicle>')
            placed.append(e.getID())
        meta["parked"] = {"count": len(placed), "candidate_km": round(total_km, 2), "edges": placed}
    return vtypes, routes, flows, meta


# ──────────────────────────────────────────────────────────────
# 5) 도로 경사 (도로망 높이)
# ──────────────────────────────────────────────────────────────
def _points_xy(net, points):
    out = []
    for p in points:
        if "lon" in p:
            x, y = net.convertLonLat2XY(float(p["lon"]), float(p["lat"]))
        else:
            x, y = float(p["x"]), float(p["y"])
        out.append((x, y, float(p["z"])))
    return out


def _idw(pts, x, y, power=2.0):
    num = den = 0.0
    for px, py, pz in pts:
        dd = math.hypot(x - px, y - py)
        if dd < 1e-6:
            return pz
        w = 1.0 / dd ** power
        num += w * pz
        den += w
    return num / den


def apply_elevation(net_path, net, d):
    """기준점 고도(DETAILS['slope']['points'])를 거리 역가중(IDW)으로 펴서 도로망의 모든 차로 모양에 z를 넣는다.

    연결·신호·차로는 그대로 두고 높이만 더한다 (netconvert를 다시 돌리지 않는다).
    Returns: 요약 (가장 가파른 차로 경사 등)
    """
    import xml.etree.ElementTree as ET
    cfg = d.get("slope") or {}
    pts = _points_xy(net, cfg.get("points") or [])
    if len(pts) < 2:
        raise ValueError("DETAILS['slope']['points']에 고도 기준점이 2개 이상 있어야 합니다 "
                         "(예: {'lon': 129.08, 'lat': 35.186, 'z': 20}).")
    power = float(cfg.get("power", 2.0))
    tree = ET.parse(net_path)
    root = tree.getroot()

    def with_z(shape):
        out = []
        for xy in shape.split():
            c = xy.split(",")
            x, y = float(c[0]), float(c[1])
            out.append(f"{c[0]},{c[1]},{_idw(pts, x, y, power):.2f}")
        return " ".join(out)

    steep = []
    for edge in root.iter("edge"):
        if edge.get("shape"):
            edge.set("shape", with_z(edge.get("shape")))
        for lane in edge.iter("lane"):
            lane.set("shape", with_z(lane.get("shape")))
            c = [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
            run = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(c, c[1:]))
            if run > 30 and edge.get("function") != "internal":
                steep.append((abs(c[-1][2] - c[0][2]) / run * 100.0, edge.get("id")))
    for j in root.iter("junction"):
        if j.get("x") is not None:
            j.set("z", f"{_idw(pts, float(j.get('x')), float(j.get('y')), power):.2f}")
        if j.get("shape"):
            j.set("shape", with_z(j.get("shape")))
    tree.write(net_path, encoding="UTF-8", xml_declaration=True)
    steep.sort(reverse=True)
    top = list(dict.fromkeys(e for _, e in steep))[:5]       # 차로마다 한 줄씩이라 엣지 중복 제거
    return {"points": len(pts), "max_grade_pct": round(steep[0][0], 2) if steep else 0.0,
            "steepest_edges": top,
            "heavy_model": cfg.get("heavy_model", "KraussPS")}


def rng_for(d):
    return random.Random(int((d.get("placement") or {}).get("seed", 0)))
