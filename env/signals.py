"""신호 배치 규칙 — 영역 지도(tools/build_area_map.py)와 학습 지도(env/osm_builder.py) 공통.

  1) OSM 신호: OSM에 신호등(highway=traffic_signals)이나 신호 횡단보도(crossing=traffic_signals)가 있는 곳은
     모두 신호. netconvert --tls.guess-signals는 교차로에서 먼 신호 횡단보도를 버리므로 여기서 다시 넣는다.
     도로망 끝(진입/진출 토막 끝, dead_end)에 걸린 OSM 신호는 막을 교통이 없어 넣지 않는다.
  2) 가정 신호: 이름이 다른 간선(trunk/primary/secondary, 연결로 제외) 2개 이상이 만나는 교차로.
     OSM에 신호 태그가 빠진 곳이 많아서다. 회전교차로는 제외.
  3) 나머지(간선 × 이면도로, 이면도로끼리)는 신호 없음 — 우선도로 규칙(이면도로가 양보).
  3-1) 현시 (korean_phase_programs): 마주 보는 진입로의 직진·우회전을 함께 녹색 → 그 축의 보호 좌회전 →
     다른 축 반복. 직진·좌회전 공용 차로가 있는 진입로는 단독 녹색(동시신호). 간선 축 직진 40초, 부도로 25초, 좌회전 12초, 황색 3초, 전적색 2초 (가정값).
     netconvert 기본 현시는 비보호 좌회전(교차로 안에서 대기)이나 진입로별 차례 녹색(간선 녹색이 너무 적음)이다.
  4) 신호 횡단보도처럼 엇갈리는 차량 이동이 없는 신호(단일로 신호)는 양방향 차량을 함께
     녹색 → 황색 → 보행자 시간(적색)으로 둔다. (netconvert는 진입로마다 번갈아 녹색을 줘서 현실과 다름)
     시간은 가정값: MIDBLOCK = 차량 녹색 60초, 황색 3초, 보행 27초 (주기 90초).
"""
import math
import os
import re
import xml.etree.ElementTree as ET

MAJOR = {"highway.trunk", "highway.primary", "highway.secondary"}
MIDBLOCK = {"green": 60.0, "yellow": 3.0, "ped": 27.0}


def osm_signal_nodes(osm_path):
    """[(osm id, lat, lon, 종류)]"""
    out = []
    for n in ET.parse(osm_path).getroot().iter("node"):
        t = {x.get("k"): x.get("v") for x in n.iter("tag")}
        if t.get("highway") == "traffic_signals" or t.get("crossing") == "traffic_signals":
            out.append((n.get("id"), float(n.get("lat")), float(n.get("lon")),
                        "신호 횡단보도" if t.get("highway") == "crossing" else "신호등"))
    return out


def _tls_nodes(net):
    return {c[0].getEdge().getToNode().getID() for tl in net.getTrafficLights() for c in tl.getConnections()}


def missing_osm_signals(net, osm_signals, radius=30.0, rename=None):
    """도로망에 아직 신호가 없는 OSM 신호 → 신호로 지정할 도로망 노드 id 목록, 건너뛴 것 목록."""
    rename = rename or {}
    tls = _tls_nodes(net)
    tls_xy = [net.getNode(n).getCoord() for n in tls]
    nodes = net.getNodes()
    add, skipped = [], []
    for oid, lat, lon, kind in osm_signals:
        x, y = net.convertLonLat2XY(lon, lat)
        oid = rename.get(oid, oid)
        if any(math.dist((x, y), p) <= radius for p in tls_xy):
            continue
        # 같은 id(또는 합쳐진 cluster_… 안의 id)의 노드, 없으면 15 m 안의 가장 가까운 노드
        nd = next((n for n in nodes if n.getID() == oid or oid in n.getID().split("_")), None)
        if nd is None:
            near = sorted(nodes, key=lambda n: math.dist(n.getCoord(), (x, y)))
            nd = near[0] if near and math.dist(near[0].getCoord(), (x, y)) <= 15 else None
        if nd is None:
            skipped.append((oid, kind, "도로망에 그 지점의 노드가 없음(영역 밖 또는 도로 중간)"))
            continue
        if nd.getType() == "dead_end" or not nd.getIncoming() or not nd.getOutgoing():
            skipped.append((oid, kind, "도로망 끝(진입/진출 토막 끝)"))
            continue
        if nd.getID() not in add:
            add.append(nd.getID())
    return add, skipped


def assumed_major_signals(net, extra_names=(), major=MAJOR):
    """이름이 다른 간선 2개 이상이 만나는 신호 없는 교차로 (회전교차로 제외). extra_names = 간선으로 볼 도로 이름."""
    tls = _tls_nodes(net)
    rb = {n for r in net.getRoundabouts() for n in r.getNodes()}
    out = []
    for nd in net.getNodes():
        if nd.getID() in tls or nd.getID() in rb or nd.getType() == "dead_end":
            continue
        names = set()
        for e in nd.getIncoming() + nd.getOutgoing():
            tp, nm = e.getType() or "", e.getName()
            if tp.endswith("_link") or not nm:
                continue
            if tp in major or nm in extra_names:
                names.add(nm)
        if len(names) >= 2:
            out.append(nd.getID())
    return out


def fix_midblock_programs(net_path, skip=(), timing=None):
    """엇갈리는 이동이 없는 신호(모든 링크가 직진, 진입로 ≤ 2)를 '양방향 함께 녹색 → 황색 → 보행 적색'으로.
    net.xml의 <tlLogic>을 직접 고친다. 반환: 고친 신호 id 목록."""
    import sumolib
    tm = {**MIDBLOCK, **(timing or {})}
    net = sumolib.net.readNet(net_path)
    targets = {}
    for tl in net.getTrafficLights():
        if tl.getID() in skip:
            continue
        conns = tl.getConnections()
        if not conns:
            continue
        ins = {c[0].getEdge().getID() for c in conns}
        dirs = set()
        for fl, tl_lane, idx in conns:
            c = next((c for c in fl.getOutgoing() if c.getToLane() == tl_lane), None)
            dirs.add(c.getDirection() if c else "?")
        if dirs <= {"s"} and len(ins) <= 2:
            targets[tl.getID()] = 1 + max(c[2] for c in conns)
    if not targets:
        return []
    txt = open(net_path, encoding="utf-8").read()
    for tid, n in targets.items():
        pat = re.compile(r'(<tlLogic id="' + re.escape(tid) + r'"[^>]*>)(.*?)(</tlLogic>)', re.S)
        new = (f'\n        <phase duration="{tm["green"]:g}" state="{"G" * n}"/>'
               f'\n        <phase duration="{tm["yellow"]:g}" state="{"y" * n}"/>'
               f'\n        <phase duration="{tm["ped"]:g}" state="{"r" * n}"/>\n    ')
        txt = pat.sub(lambda m: m.group(1) + new + m.group(3), txt, count=1)
    open(net_path, "w", encoding="utf-8").write(txt)
    return sorted(targets)


def fix_joined_programs(net_path):
    """노드 여러 개가 신호 하나로 묶인 교차로(--tls.join)에서, 묶인 노드 사이 짧은 도로의 정지선(내부 링크) 중
    직진·우회전은 항상 녹색으로 둔다 (좌회전은 korean_phase_programs의 보호 좌회전 현시) — 연산교차로와 같은 방식. 진입 정지선을 녹색에 통과한 차가 몇 m 뒤 내부 정지선 적색에
    갇히거나 그 적색을 지나 교차 차량과 부딪히지 않게 한다 (환경 관측도 내부 링크를 항상 녹색으로 본다).
    반환: {신호 id: 고친 링크 수}"""
    import sumolib
    net = sumolib.net.readNet(net_path)
    fix = {}
    for tl in net.getTrafficLights():
        conns = tl.getConnections()
        nodes = {c[0].getEdge().getToNode().getID() for c in conns}
        if len(nodes) < 2:
            continue
        inner = []
        for fl, tl_lane, idx in conns:
            if fl.getEdge().getFromNode().getID() not in nodes:
                continue
            c = next((c for c in fl.getOutgoing() if c.getToLane() == tl_lane), None)
            if c is None or c.getDirection() not in ("l", "L"):      # 좌회전은 보호 좌회전 현시 그대로
                inner.append(idx)
        inner = sorted(set(inner))
        if inner:
            fix[tl.getID()] = inner
    if not fix:
        return {}
    txt = open(net_path, encoding="utf-8").read()
    for tid, idxs in fix.items():
        pat = re.compile(r'(<tlLogic id="' + re.escape(tid) + r'"[^>]*>)(.*?)(</tlLogic>)', re.S)

        def sub(m):
            body = re.sub(r'state="([^"]*)"',
                          lambda s: 'state="' + "".join("G" if i in idxs else ch for i, ch in enumerate(s.group(1))) + '"',
                          m.group(2))
            return m.group(1) + body + m.group(3)
        txt = pat.sub(sub, txt, count=1)
    open(net_path, "w", encoding="utf-8").write(txt)
    return {k: len(v) for k, v in fix.items()}


PHASE = {"major_green": 40.0, "minor_green": 25.0, "left_green": 12.0, "yellow": 3.0, "all_red": 2.0}


def korean_phase_programs(net_path, skip=(), timing=None):
    """신호 현시를 '축별 직진 → 축별 보호 좌회전'으로 다시 만든다 (netconvert 기본 현시 대신).

    축 = 마주 보는 두 진입로(방위 차이 135~225°), 짝이 없는 진입로는 혼자 한 축.
      축 직진 현시 : 그 축 진입로들의 직진·우회전 (짝 없는 진입로는 좌회전까지 전부) 녹색
      축 좌회전 현시: 짝 있는 축의 좌회전만 녹색 (보호 좌회전 — 교차로 안에서 기다리는 비보호 좌회전 없음)
    각 녹색 뒤 황색·전적색. 간선(도로 우선순위가 가장 높은) 축은 직진 녹색을 길게.
    노드 여러 개가 하나로 묶인 신호의 내부 링크는 항상 녹색. 시간은 가정값(PHASE).
    반환: {신호 id: 현시 수}"""
    import sumolib
    tm = {**PHASE, **(timing or {})}
    net = sumolib.net.readNet(net_path)
    progs = {}
    for tl in net.getTrafficLights():
        if tl.getID() in skip:
            continue
        conns = tl.getConnections()
        nodes = {c[0].getEdge().getToNode().getID() for c in conns}
        cx = sum(net.getNode(n).getCoord()[0] for n in nodes) / len(nodes)
        cy = sum(net.getNode(n).getCoord()[1] for n in nodes) / len(nodes)
        n_links = 1 + max(c[2] for c in conns)
        inner, appr, inner_left = set(), {}, {}
        for fl, tl_lane, idx in conns:
            e = fl.getEdge()
            if e.getFromNode().getID() in nodes:
                c = next((c for c in fl.getOutgoing() if c.getToLane() == tl_lane), None)
                if c is not None and c.getDirection() in ("l", "L"):
                    # 묶인 노드 사이 도로에서의 좌회전: 마주 오는 직진을 가로지르므로 들어온 진입로 축의 보호 좌회전 현시로
                    src = next((x.getID() for x in e.getFromNode().getIncoming()
                                if x.getFromNode().getID() not in nodes and e in x.getOutgoing()), None)
                    inner_left[idx] = src
                else:
                    inner.add(idx)
                continue
            c = next((c for c in fl.getOutgoing() if c.getToLane() == tl_lane), None)
            x, y = e.getShape()[0]
            b = math.degrees(math.atan2(x - cx, y - cy)) % 360
            a_ = appr.setdefault(e.getID(), {"b": b, "prio": e.getPriority(), "links": [], "lane_dirs": {}})
            a_["links"].append((idx, c.getDirection() if c else "s"))
            a_["lane_dirs"].setdefault(fl.getIndex(), set()).add("l" if c and c.getDirection() in ("l", "L") else "o")
        if not appr:
            continue
        for i, src in list(inner_left.items()):
            if src not in appr:
                inner.add(i)
                del inner_left[i]
        # 축 묶기
        ids = sorted(appr, key=lambda k: -appr[k]["prio"])
        axes, used = [], set()
        for a in ids:
            if a in used:
                continue
            used.add(a)
            mate = min((b for b in ids if b not in used and 135 <= abs(appr[a]["b"] - appr[b]["b"]) % 360 <= 225),
                       key=lambda b: abs(abs(appr[a]["b"] - appr[b]["b"]) - 180), default=None)
            ax = [a] + ([mate] if mate else [])
            used |= set(ax)
            axes.append(ax)
        top = max(appr[a]["prio"] for a in appr)
        # 직진·좌회전 공용 차로가 있는 진입로는 좌회전만 떼어 낼 수 없다(맨 앞 좌회전 차가 직진을 막음) →
        # 그 축은 진입로마다 단독 녹색(동시신호)
        def shared(a):
            return any(len(v) > 1 for v in appr[a]["lane_dirs"].values())
        axes2 = []
        for ax in axes:
            if len(ax) == 2 and any(shared(a) for a in ax):
                axes2 += [[a] for a in ax]
            else:
                axes2.append(ax)
        axes = axes2
        phases = []
        for ax in axes:
            major = max(appr[a]["prio"] for a in ax) == top
            if len(ax) == 1:
                main = {i for a in ax for i, d in appr[a]["links"]}
                left = set()
            else:
                main = {i for a in ax for i, d in appr[a]["links"] if d not in ("l", "L")}
                left = {i for a in ax for i, d in appr[a]["links"] if d in ("l", "L")}
            (left if len(ax) > 1 else main).update(i for i, src in inner_left.items() if src in ax)
            for grp, g in ((main, tm["major_green"] if major else tm["minor_green"]), (left, tm["left_green"])):
                if not grp:
                    continue
                def st(ch):
                    return "".join("G" if i in inner else (ch if i in grp else "r") for i in range(n_links))
                phases += [(g, st("G")), (tm["yellow"], st("y"))]
                if tm["all_red"] > 0:
                    phases.append((tm["all_red"], st("r")))
        progs[tl.getID()] = phases
    if not progs:
        return {}
    txt = open(net_path, encoding="utf-8").read()
    for tid, phases in progs.items():
        pat = re.compile(r'(<tlLogic id="' + re.escape(tid) + r'"[^>]*>)(.*?)(</tlLogic>)', re.S)
        body = "".join(f'\n        <phase duration="{d:g}" state="{s}"/>' for d, s in phases) + "\n    "
        txt = pat.sub(lambda m: m.group(1) + body + m.group(3), txt, count=1)
    open(net_path, "w", encoding="utf-8").write(txt)
    return {k: len(v) for k, v in progs.items()}


def brt_split_programs(net_path, skip=(), bus_green=10.0, yellow=3.0):
    """버스전용차로(BRT) 링크가 있는 신호: 버스 직진과 경로가 엇갈리는 승용차 링크가 같은 녹색에 있으면
    그 녹색 끝에 BRT만 녹색인 구간을 둔다 (env/bus_lanes.bus_subphases — 연산교차로와 같은 규칙).
    SUMO는 같은 진입 도로의 연결끼리 서로 양보시키지 않아서 신호로 떼어 놓아야 한다. 반환: 고친 신호 id 목록"""
    import sumolib
    from env.bus_lanes import bus_link_conflicts, bus_subphases
    net = sumolib.net.readNet(net_path)
    txt = open(net_path, encoding="utf-8").read()
    done = []
    for tl in net.getTrafficLights():
        if tl.getID() in skip:
            continue
        conf = bus_link_conflicts(net_path, tl.getID())
        if not conf:
            continue
        nodes = {c[0].getEdge().getToNode().getID() for c in tl.getConnections()}
        approach = {idx for fl, _, idx in tl.getConnections() if fl.getEdge().getFromNode().getID() not in nodes}
        pat = re.compile(r'(<tlLogic id="' + re.escape(tl.getID()) + r'"[^>]*>)(.*?)(</tlLogic>)', re.S)
        m = pat.search(txt)
        phases = [(float(d), st) for d, st in re.findall(r'<phase duration="([^"]*)"\s+state="([^"]*)"', m.group(2))]
        new = bus_subphases(phases, conf, set().union(*conf.values()), bus_green, yellow, approach=approach)
        if new == phases:
            continue
        body = "".join(f'\n        <phase duration="{d:g}" state="{s}"/>' for d, s in new) + "\n    "
        txt = pat.sub(lambda mm: mm.group(1) + body + mm.group(3), txt, count=1)
        done.append(tl.getID())
    open(net_path, "w", encoding="utf-8").write(txt)
    return done


def demand_green_split(net_path, rou_path, skip=(), sat=1800.0, min_green=10.0, cycle_range=(60.0, 150.0), tls_add=None):
    """배경 교통 수요(rou의 flow 경로와 도착률)에 맞춰 신호 녹색 시간을 다시 나눈다 (Webster).

    현시 구성(어떤 연결이 함께 녹색인지)과 황색·전적색은 그대로 두고 녹색 시간만 바꾼다.
      연결별 교통량 = 그 연결(진입 도로 → 진출 도로)을 지나는 flow들의 시간당 대수 합
      현시 p의 포화도 y_p = 그 현시에 녹색인 진입 차로 중 가장 붐비는 차로의 (교통량 / sat)
      주기 C = (1.5 L + 5) / (1 − Y) (L = 황색·전적색 합, Y = Σ y_p), cycle_range로 자름
      녹색 g_p = (C − L) × y_p / Y, 최소 min_green
    교통량이 0인 현시는 min_green. 늘 녹색(G)인 묶인 노드 내부 링크는 계산에서 뺀다.
    tls_add: 따로 쓴 신호 프로그램 파일(.tls.xml)이 있으면 그 안의 같은 id 프로그램도 같이 고친다.
    tls_add에 프로그램이 있는 신호는 SUMO가 그 프로그램으로 돌리므로 그 현시로 계산하고 그 파일만 고친다.
    반환: {신호 id: (주기, [녹색들], Y)}"""
    import collections
    import sumolib
    net = sumolib.net.readNet(net_path, withPrograms=True)
    # 연결별 교통량 (진입 도로, 진출 도로) → 대/시
    flow = collections.Counter()
    root = ET.parse(rou_path).getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.iter("route") if r.get("id")}
    for f in root.iter("flow"):
        edges = routes.get(f.get("route")) or (f.get("route") and None)
        if not edges:
            continue
        per = f.get("period", "")
        if per.startswith("exp("):
            vph = float(per[4:-1]) * 3600.0
        elif f.get("vehsPerHour"):
            vph = float(f.get("vehsPerHour"))
        elif per:
            vph = 3600.0 / float(per)
        else:
            continue
        for a, b in zip(edges, edges[1:]):
            flow[(a, b)] += vph
    add_phases = {}
    if tls_add and os.path.exists(tls_add):
        for t in ET.parse(tls_add).getroot().iter("tlLogic"):
            add_phases[t.get("id")] = [(float(p.get("duration")), p.get("state")) for p in t.iter("phase")]
    out = {}
    for tl in net.getTrafficLights():
        tid = tl.getID()
        if tid in skip:
            continue
        progs = tl.getPrograms()
        if not progs:
            continue
        prog = list(progs.values())[0]
        phases = add_phases.get(tid) or [(float(p.duration), p.state) for p in prog.getPhases()]
        conns = tl.getConnections()
        n = len(phases[0][1])
        always = {i for i in range(n) if all(st[i] in "Gg" for _, st in phases)}
        lane_links = collections.defaultdict(set)     # 진입 차로 → 연결 index
        link_flow = {}
        lane_count = collections.Counter()            # (진입 도로, 진출 도로) 연결이 쓰는 진입 차로 수
        for inl, outl, idx in conns:
            lane_count[(inl.getEdge().getID(), outl.getEdge().getID())] += 1
        for inl, outl, idx in conns:
            if idx in always:
                continue
            k = (inl.getEdge().getID(), outl.getEdge().getID())
            link_flow[idx] = link_flow.get(idx, 0.0) + flow.get(k, 0.0) / max(lane_count[k], 1)
            lane_links[inl.getID()].add(idx)
        green_idx = [k for k, (d, st) in enumerate(phases) if any(st[i] in "Gg" for i in range(n) if i not in always)
                     and not any(st[i] in "yY" for i in range(n))]
        if not green_idx:
            continue
        lost = sum(d for k, (d, st) in enumerate(phases) if k not in green_idx)
        ys = []
        for k in green_idx:
            st = phases[k][1]
            yk = 0.0
            for lane, idxs in lane_links.items():
                g = [i for i in idxs if st[i] in "Gg"]
                if g:
                    yk = max(yk, sum(link_flow.get(i, 0.0) for i in idxs) / sat)
            ys.append(yk)
        Y = sum(ys)
        if Y <= 0:
            continue
        C = (1.5 * lost + 5.0) / max(1.0 - Y, 0.1)
        C = min(max(C, cycle_range[0]), cycle_range[1])
        avail = max(C - lost, min_green * len(green_idx))
        greens = [max(min_green, avail * y / Y) for y in ys]
        new = list(phases)
        for k, g in zip(green_idx, greens):
            new[k] = (round(g), phases[k][1])
        out[tid] = (round(sum(d for d, _ in new)), [round(g) for g in greens], round(Y, 2))
        for path in ([tls_add] if tid in add_phases else [net_path]):
            txt = open(path, encoding="utf-8").read()
            pat = re.compile(r'(<tlLogic id="' + re.escape(tid) + r'"[^>]*>)(.*?)(</tlLogic>)', re.S)
            if not pat.search(txt):
                continue
            body = "".join(f'\n        <phase duration="{d:g}" state="{st}"/>' for d, st in new) + "\n    "
            txt = pat.sub(lambda m: m.group(1) + body + m.group(3), txt, count=1)
            open(path, "w", encoding="utf-8").write(txt)
    return out
