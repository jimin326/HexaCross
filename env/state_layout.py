"""데이터셋 state의 번호표 (DATA_FORMAT.md 2절).

SUMO의 문자열 이름(도로 edge id, 신호 id + 연결 번호)을 데이터에 적을 숫자로 바꾸는 표를
도로 파일에서 한 번 만든다. SUMO를 실행하지 않고 지도·경로·신호 파일만 읽는다.

    lay = build_state_layout(cfg_path, meta, road_legs, traffic)
    lay["edge_leg"]["773756904#2"]          → 15 (중앙천로, 루프 방향)
    lay["tl_number"][("4633975089", 7)]     → 12 (중앙천로 끝 신호)

매 스텝 실제 값을 읽는 부분(get_privileged_state, _signal_state)은 SUMO 연결이 필요해서
env/sumo_env.py에 있고, 이 표를 찾아본다. 정책 입력(observation)에는 쓰지 않는다.

번호 (DATA_FORMAT.md 2절 번호표와 같아야 한다)
    leg   0~5 연산교차로로 들어가는 도로(LEG_ORDER 순) · 6~11 나가는 도로 · 12 연산교차로 안
          13~25 루프 B 주변 도로(방향별, LOOP_LEG_CODES) · 19 회전교차로 안 · -1 그 밖
    신호  0~5 연산교차로 진입로 그룹 A · 6~8 시청 인근 · 9 그 밖 · 10~ AV 경로의 연산 밖 신호
          21~ 연산교차로 진입로 안 차로 그룹 B·C · -1 앞에 신호 없음
"""
import math
import os
import xml.etree.ElementTree as ET

LEG_ORDER = ("SW", "N", "W", "S", "E", "NE")   # 연산교차로 진입부·신호등 번호 0~5
LEG_INSIDE = 12        # 연산교차로 안 (우회전 연결로 포함)
LEG_ROUNDABOUT = 19    # AV 경로가 지나는 회전교차로 안
# 루프 B 주변 도로 번호: 도로 이름 → (루프 방향, 반대 방향)
LOOP_LEG_CODES = {"연수로": (13, 14), "중앙천로": (15, 16), "신촌로": (17, 18),
                  "거제천로": (20, 21), "쌍미천로": (22, 23), "세병로": (24, 25)}
TL_NONE = -1           # 앞에 신호 없음
TL_OTHER = 9           # 그 밖의 신호 (지도와 상관없이 고정. 6~8은 시청 인근)
TL_ROUTE_FIRST = 10    # AV 경로가 지나는 연산 밖 신호 진입로 (경로 순서, 루프 B는 10~20)
TL_GROUP_FIRST = 21    # 연산교차로 진입로 안 차로 그룹 B·C (LEG_ORDER 순)


def empty_layout():
    """번호표가 없는 도로(연산교차로 OSM 도로망이 아님)에서 쓰는 빈 표: 모두 -1 / 0."""
    return {"edge_leg": {}, "lane_count": {}, "stop_edges": set(),
            "center_tls": None, "leg_link": {}, "leg_edges": {}, "tl_number": {},
            "tl_names": list(LEG_ORDER), "other_tl": TL_OTHER,
            "tl_link": {}, "tl_order": {}, "tl_groups": {}, "route_length_m": None}


def build_state_layout(cfg_path, meta, road_legs, traffic):
    """연산교차로 기준 도로 번호와 신호 번호표를 도로 파일에서 만든다.

    cfg_path : highway.sumocfg (같은 폴더의 highway.net.xml, 경로·추가 파일을 읽는다)
    meta     : 도로 빌더가 남긴 highway.meta.json 내용 (layout, center, center_tls, ego_routes, extra_tls)
    road_legs: road_config.ROAD["legs"] (연산교차로 진입로 방위각)
    traffic  : road_config.TRAFFIC (loop_flows: 루프 방향 기준 edge)

    반환 dict:
    edge_leg  : edge id → leg 번호. 들어가는/나가는 도로는 교차로에 붙은 edge에서 같은 이름
                도로를 따라 도로망 끝까지 이어 간다 (예: SW는 시청 인근 교차로 너머까지).
    lane_count: edge id → 승용차가 다닐 수 있는 차로 수 (버스 전용 차로 제외)
    stop_edges: 연산교차로로 들어가는 edge (정지선)
    tl_number : (신호 id, linkIndex) → ego_tl 번호
    tl_link   : 신호 칸 번호 → (신호 id, 대표 linkIndex) — tl_color_i / tl_remaining_i
    tl_order  : 신호 칸 번호 → 이름 (meta.json)
    tl_groups : 신호 칸 번호 → {"edge", "lanes"} (차로 번호 0 = 맨 오른쪽)
    route_length_m: 첫 AV 경로 길이
    """
    lay = empty_layout()
    if not (meta and meta.get("layout") == "osm"):
        return lay          # 연산교차로 OSM 도로망이 아니면 번호표 없음 (-1)

    import sumolib
    net = sumolib.net.readNet(
        os.path.join(os.path.dirname(os.path.abspath(cfg_path)), "highway.net.xml"),
        withInternal=True, withPrograms=True)
    center_id = str(meta["center"])
    cx, cy = net.getNode(center_id).getCoord()
    bearings = {lg["id"]: float(lg["bearing"]) for lg in road_legs}
    order = [lg for lg in LEG_ORDER if lg in bearings]

    def bearing(px, py, ox=cx, oy=cy):
        return math.degrees(math.atan2(px - ox, py - oy)) % 360.0

    def adiff(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def leg_at(px, py):
        return min(order, key=lambda lg: adiff(bearing(px, py), bearings[lg]))

    def heading(e, at_end):
        s = e.getShape()
        (x0, y0), (x1, y1) = (s[-2], s[-1]) if at_end else (s[0], s[1])
        return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0

    def trace(e, outgoing, code, limit=5000.0):
        """같은 이름 도로를 따라 바깥으로 이어 가며 edge에 번호를 붙인다."""
        seen, total = set(), 0.0
        while e is not None and e.getID() not in seen and total < limit:
            seen.add(e.getID())
            lay["edge_leg"].setdefault(e.getID(), code)
            total += e.getLength()
            nxt = list(e.getOutgoing()) if outgoing else list(e.getIncoming())
            nxt = [n for n in nxt if n.getFunction() != "internal"
                   and n.allows("passenger") and n.getName() == e.getName()
                   and n.getID().lstrip("-") != e.getID().lstrip("-")]
            if not nxt:
                break
            h = heading(e, at_end=outgoing)
            e = min(nxt, key=lambda n: adiff(heading(n, at_end=not outgoing), h))

    center = net.getNode(center_id)
    center_tls = meta.get("center_tls")
    lay["center_tls"] = center_tls
    for e in center.getIncoming():
        if e.getFunction() == "internal" or not e.allows("passenger"):
            continue
        leg = leg_at(*e.getShape()[0])
        lay["stop_edges"].add(e.getID())
        lay["leg_edges"].setdefault(leg, []).append(e.getID())
        trace(e, outgoing=False, code=LEG_ORDER.index(leg))
        for ln in e.getLanes():
            for c in ln.getOutgoing():
                if c.getTLSID() == center_tls:
                    lay["tl_number"][(center_tls, c.getTLLinkIndex())] = LEG_ORDER.index(leg)
                    lay["leg_link"].setdefault(leg, c.getTLLinkIndex())
    for e in center.getOutgoing():
        if e.getFunction() == "internal" or not e.allows("passenger"):
            continue
        far, total = e, e.getLength()     # 교차로에서 60m 이상 떨어진 점으로 방향 판정
        while total < 60.0:
            nxt = [n for n in far.getOutgoing() if n.getName() == far.getName()]
            if not nxt:
                break
            far = nxt[0]
            total += far.getLength()
        trace(e, outgoing=True, code=6 + LEG_ORDER.index(leg_at(*far.getShape()[-1])))

    for e in net.getEdges(withInternal=True):
        eid = e.getID()
        lay["lane_count"][eid] = sum(1 for ln in e.getLanes() if ln.allows("passenger"))
        if eid.startswith(f":{center_id}_"):
            lay["edge_leg"][eid] = LEG_INSIDE          # 교차로 안
        elif "_link" in (e.getType() or "") and min(
                math.hypot(px - cx, py - cy) for px, py in e.getShape()) < 150.0:
            lay["edge_leg"][eid] = LEG_INSIDE          # 우회전 연결로

    # 루프 B 주변 도로(13~25)와 회전교차로(19). 다른 교차로 안 번호 잇기보다 먼저 한다.
    route_edges = ego_route_edges(cfg_path, meta)
    if route_edges:
        lay["route_length_m"] = round(sum(net.getEdge(eid).getLength() for eid in route_edges[0]
                                          if net.hasEdge(eid)), 1)
    number_loop_roads(net, lay, route_edges, traffic)

    # 다른 교차로 안(내부 연결)은 들어온 도로의 번호를 이어받는다
    for e in net.getEdges(withInternal=False):
        code = lay["edge_leg"].get(e.getID())
        if code is None or code == LEG_INSIDE:
            continue
        for ln in e.getLanes():
            for c in ln.getOutgoing():
                via = c.getViaLaneID()
                if via:
                    lay["edge_leg"].setdefault(via.rsplit("_", 1)[0], code)

    # extra_signals 교차로(시청 인근)의 진입 그룹: osm_builder가 meta에 남긴 순서대로
    number = len(LEG_ORDER)
    for jid, groups in (meta.get("extra_tls") or {}).items():
        jx, jy = net.getNode(jid).getCoord()
        for e in net.getEdges(withInternal=False):
            for ln in e.getLanes():
                for c in ln.getOutgoing():
                    if c.getTLSID() != jid:
                        continue
                    b = bearing(*e.getShape()[0], ox=jx, oy=jy)
                    k = min(range(len(groups)), key=lambda i: (
                        groups[i]["road"] != (e.getName() or ""),
                        adiff(b, groups[i]["bearing"])))
                    lay["tl_number"][(jid, c.getTLLinkIndex())] = number + k
        lay["tl_names"] += [f"{jid}:{g['road']}@{g['bearing']}" for g in groups]
        number += len(groups)
    if number > TL_OTHER:
        raise ValueError(f"시청 인근 신호 그룹이 너무 많아 번호 {number - 1}이 '그 밖'({TL_OTHER})과 겹칩니다")
    lay["tl_names"].append("other")
    number_signals(net, lay, route_edges, meta, cfg_path)
    return lay


def cfg_input_files(cfg_path, key):
    """sumocfg <input>의 key(route-files, additional-files 등)에 적힌 파일 경로 목록."""
    try:
        node = ET.parse(cfg_path).getroot().find(f"input/{key}")
    except (OSError, ET.ParseError):
        return []
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    value = (node.get("value") if node is not None else "") or ""
    return [os.path.join(cfg_dir, f.strip()) for f in value.split(",") if f.strip()]


def ego_route_edges(cfg_path, meta):
    """meta의 AV 경로 id들을 경로 파일에서 찾아 [[edge id, ...], ...] (meta 순서)로 돌려준다."""
    ids = [str(r) for r in ((meta or {}).get("ego_routes") or [])]
    found = {}
    for path in cfg_input_files(cfg_path, "route-files"):
        try:
            for r in ET.parse(path).getroot().iter("route"):
                if r.get("id") in ids and r.get("edges"):
                    found.setdefault(r.get("id"), r.get("edges").split())
        except (OSError, ET.ParseError):
            continue
    return [found[i] for i in ids if i in found]


def tls_programs(net, cfg_path):
    """신호 id → 현시 상태 문자열 목록.

    도로 파일의 기본 프로그램에, sumocfg의 추가 파일(highway.tls.xml 등)에 있는
    tlLogic을 덮어쓴다 (SUMO도 나중에 읽은 프로그램을 쓴다).
    """
    progs = {}
    for t in net.getTrafficLights():
        ps = t.getPrograms()
        if ps:
            progs[t.getID()] = [ph.state for ph in ps[sorted(ps)[0]].getPhases()]
    for path in cfg_input_files(cfg_path, "additional-files"):
        try:
            for tl in ET.parse(path).getroot().iter("tlLogic"):
                progs[tl.get("id")] = [ph.get("state") for ph in tl.iter("phase")]
        except (OSError, ET.ParseError):
            continue
    return progs


def number_loop_roads(net, lay, route_edges, traffic):
    """루프 B 주변 도로(LOOP_LEG_CODES, 방향별)와 AV 경로의 회전교차로에 leg 번호를 붙인다.

    이미 번호가 있는 edge(연산교차로 도로 0~12)는 건드리지 않는다.
    방향: AV 경로 edge면 루프 방향, 그 반대편 edge(id 앞 "-" 차이)면 반대 방향,
    그 밖에는 같은 이름의 가장 가까운 기준 edge와 진행 방향을 비교한다.
    기준 edge = AV 경로 + 순환 흐름 중 이름에 "(루프 방향)"·"(1)"이 있는 것의 via.
    """
    edge_leg = lay["edge_leg"]
    route_set = {eid for r in route_edges for eid in r}
    refs_by_name = {}
    ref_ids = list(route_set)
    for fl in ((traffic or {}).get("loop_flows") or {}).get("flows", []):
        if "(루프 방향)" in fl.get("name", "") or "(1)" in fl.get("name", ""):
            ref_ids += fl.get("via", [])
    for eid in ref_ids:
        if net.hasEdge(eid):
            e = net.getEdge(eid)
            refs_by_name.setdefault(e.getName(), []).append(e)

    for rb in net.getRoundabouts():
        if not route_set & set(rb.getEdges()):
            continue                    # AV 경로가 지나는 회전교차로만
        for eid in rb.getEdges():
            edge_leg.setdefault(eid, LEG_ROUNDABOUT)
        inner = tuple(f":{nid}_" for nid in rb.getNodes())
        for e in net.getEdges(withInternal=True):
            if e.getID().startswith(inner):
                edge_leg.setdefault(e.getID(), LEG_ROUNDABOUT)

    def mid(e):
        s = e.getShape()
        return (s[0][0] + s[-1][0]) / 2.0, (s[0][1] + s[-1][1]) / 2.0

    def vec(e):
        s = e.getShape()
        return s[-1][0] - s[0][0], s[-1][1] - s[0][1]

    for e in net.getEdges(withInternal=False):
        eid = e.getID()
        codes = LOOP_LEG_CODES.get(e.getName())
        refs = refs_by_name.get(e.getName())
        if not codes or not refs or eid in edge_leg or not e.allows("passenger"):
            continue
        reverse = eid[1:] if eid.startswith("-") else "-" + eid
        if eid in route_set:
            same = True
        elif reverse in route_set:
            same = False
        else:
            mx, my = mid(e)
            r = min(refs, key=lambda q: (mid(q)[0] - mx) ** 2 + (mid(q)[1] - my) ** 2)
            (ax, ay), (bx, by) = vec(e), vec(r)
            same = ax * bx + ay * by >= 0.0
        edge_leg[eid] = codes[0] if same else codes[1]

    # 이름 없는 연결로(*_link)는 들어오는 도로의 번호를 이어받는다
    for _ in range(3):
        for e in net.getEdges(withInternal=False):
            if e.getID() in edge_leg or e.getName() or "_link" not in (e.getType() or ""):
                continue
            codes = [edge_leg.get(p.getID()) for p in e.getIncoming()]
            codes = [c for c in codes if c is not None and c != LEG_INSIDE]
            if codes:
                edge_leg[e.getID()] = codes[0]


def lane_groups(net, edges, tid, phases):
    """진입 edge들의 차로를 '현시마다 녹색인지' 패턴이 같은 것끼리 묶는다.

    반환: [{"edge", "lanes", "links"}] (edge 순, 가장 오른쪽 차로가 있는 묶음부터)
    """
    out = {}
    for eid in edges:
        for ln in net.getEdge(eid).getLanes():
            pats, links = set(), []
            for c in ln.getOutgoing():
                if c.getTLSID() == tid:
                    li = c.getTLLinkIndex()
                    links.append(li)
                    pats.add(tuple(len(p) > li and p[li] in "Gg" for p in phases))
            if links:
                g = out.setdefault((eid, tuple(sorted(pats))), {"edge": eid, "lanes": [], "links": []})
                g["lanes"].append(ln.getIndex())
                g["links"] += links
    return sorted(out.values(), key=lambda g: (g["edge"], min(g["lanes"])))


def number_signals(net, lay, route_edges, meta, cfg_path):
    """신호 칸 번호를 정한다: tl_link · tl_order · tl_groups, tl_number 보충.

    0~5   연산교차로 진입로 — 대표 링크(leg_link)가 속한 차로 그룹 A
    10~   AV 경로가 지나는 연산 밖 신호 진입로 (진입 edge마다 하나, 경로 순서)
    21~   연산교차로 진입로 안 차로 그룹 B·C (LEG_ORDER 순)
    """
    center_tls = lay["center_tls"]
    lay["tl_order"] = {i: n for i, n in enumerate(lay["tl_names"][:-1])}
    lay["tl_order"][TL_OTHER] = "other"
    center_phases = tls_programs(net, cfg_path).get(center_tls) or []

    extra_groups = []                   # (leg, 그룹 A가 아닌 차로 그룹)
    for i, leg in enumerate(LEG_ORDER):
        link = lay["leg_link"].get(leg)
        if link is None:
            continue
        lay["tl_link"][i] = (center_tls, link)
        for g in lane_groups(net, lay["leg_edges"].get(leg, []), center_tls, center_phases):
            if link in g["links"]:
                lay["tl_groups"][i] = {"edge": g["edge"], "lanes": sorted(g["lanes"])}
            else:
                extra_groups.append((leg, g))

    num = TL_ROUTE_FIRST
    extra_ids = set((meta.get("extra_tls") or {}).keys())
    seen = set()
    for r in route_edges:
        for a, b in zip(r, r[1:]):
            if not (net.hasEdge(a) and net.hasEdge(b)):
                continue
            ea = net.getEdge(a)
            for c in ea.getConnections(net.getEdge(b)):
                tid = c.getTLSID()
                if not tid or tid == center_tls or tid in extra_ids or (tid, a) in seen:
                    continue
                seen.add((tid, a))
                lanes = set()
                for ln in ea.getLanes():
                    for cc in ln.getOutgoing():
                        if cc.getTLSID() == tid:
                            lay["tl_number"].setdefault((tid, cc.getTLLinkIndex()), num)
                            lanes.add(ln.getIndex())
                lay["tl_link"][num] = (tid, c.getTLLinkIndex())
                lay["tl_order"][num] = f"{tid}@{a}"
                lay["tl_groups"][num] = {"edge": a, "lanes": sorted(lanes)}
                num += 1

    num = max(TL_GROUP_FIRST, num)
    count = {}
    for leg, g in extra_groups:
        count[leg] = count.get(leg, 0) + 1
        for li in g["links"]:
            lay["tl_number"][(center_tls, li)] = num
        lay["tl_link"][num] = (center_tls, min(g["links"]))
        lay["tl_order"][num] = f"{leg} 그룹 {'BCDEFG'[count[leg] - 1]}"
        lay["tl_groups"][num] = {"edge": g["edge"], "lanes": sorted(g["lanes"])}
        num += 1
