"""SUMO 도로망 ↔ OSM 전수 대조 (영역 지도·학습 지도 공통).

  python tools/audit_osm.py area [NET]   # env/maps/yeonsan_area.net.xml ↔ env/osm/yeonsan_area.osm
  python tools/audit_osm.py train [NET]  # 학습 지도(env/sumo/highway.net.xml) ↔ ROAD["osm"]["file"]

엣지 id로 OSM way를 찾지 않고 "위치"로 짝짓는다: netconvert(--geometry.remove)는 이어진 OSM way 여러 개를
엣지 하나로 합치고 첫 way의 id만 남기기 때문이다. 엣지 모양을 3 m 간격으로 찍어 가장 가까운 OSM 도로 선에
붙이고(양방향 도로는 방향마다 차로 폭 절반만큼 옆에 놓이므로 그만큼 허용), 진행 방향은 두 선의 방향으로 판정.

  A 도로 유무  : 들어가야 할 OSM 도로(방향별)가 도로망에 덮여 있는가
  B 모양      : 엣지가 OSM 선에서 벗어난 곳 (교차로 안 짧은 구간 제외)
  C 방향      : 일방통행 도로를 거꾸로 달리는 엣지 / 양방향 도로인데 한쪽만 있는 곳
  D 차로 수    : 엣지가 덮는 OSM way들의 그 방향 차로 수(lanes, lanes:forward·backward) ↔ 엣지 차로 수
  E 회전 제한  : OSM relation(no_* / only_*) ↔ 도로망 연결
  F 신호      : OSM 신호 노드(신호등·신호 횡단보도) 자리에 도로망 신호가 있는가 + 신호 없는 교차로
"""
import collections
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from tools.audit_lanes import expect  # noqa: E402

ONEWAY = ("yes", "1", "true")
DRIVE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}
CELL = 30.0


def _proj(t_a, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0.0, min(1.0, ((t_a[0] - a[0]) * dx + (t_a[1] - a[1]) * dy) / L2))
    q = (a[0] + t * dx, a[1] + t * dy)
    return math.dist(t_a, q), (dx, dy)


class SegIndex:
    """OSM 도로 선분 격자 색인: 점 → (거리, way, 선분 방향)"""

    def __init__(self, lines):
        self.g = collections.defaultdict(list)
        for w, pl in lines.items():
            for a, b in zip(pl, pl[1:]):
                x0, x1 = sorted((a[0], b[0]))
                y0, y1 = sorted((a[1], b[1]))
                for i in range(int(x0 // CELL) - 1, int(x1 // CELL) + 2):
                    for j in range(int(y0 // CELL) - 1, int(y1 // CELL) + 2):
                        self.g[(i, j)].append((w, a, b))

    def near(self, p, r):
        out = {}
        i, j = int(p[0] // CELL), int(p[1] // CELL)
        k = int(r // CELL) + 1
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                for w, a, b in self.g.get((i + di, j + dj), ()):
                    d, v = _proj(p, a, b)
                    if d <= r and (w not in out or d < out[w][0]):
                        out[w] = (d, v)
        return out          # {way: (거리, 선분 방향벡터)}


def load(osm_path):
    r = ET.parse(osm_path).getroot()
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in r.iter("node")}
    ntags = {n.get("id"): {t.get("k"): t.get("v") for t in n.iter("tag")} for n in r.iter("node")}
    ways = {w.get("id"): ({t.get("k"): t.get("v") for t in w.iter("tag")}, [x.get("ref") for x in w.iter("nd")])
            for w in r.iter("way")}
    rels = [(rel.get("id"), {t.get("k"): t.get("v") for t in rel.iter("tag")},
             [(m.get("type"), m.get("ref"), m.get("role")) for m in rel.iter("member")]) for rel in r.iter("relation")]
    return nodes, ntags, ways, rels


def main(which, net_path=None):
    import sumolib
    merged, assumed, splits = {}, {}, {}
    if which == "area":
        cfg = json.load(open(os.path.join(BASE, "env/maps/yeonsan_area.json"), encoding="utf-8"))
        meta = json.load(open(os.path.join(BASE, "env/maps/yeonsan_area.meta.json"), encoding="utf-8"))
        net_path = net_path or os.path.join(BASE, "env/maps/yeonsan_area.net.xml")
        osm_path = os.path.join(BASE, cfg["osm_file"])
        merges = meta.get("carriageway_merges", [])
        assumed = {w: int(a["lanes"]) for w, a in (cfg.get("lane_assumptions") or {}).items()}
        splits = {s["way"]: s for s in cfg.get("way_splits", [])}
        selected = set(meta["selected_ways"])
        bbox = None
        region_ll = cfg["polygon"]                 # 빨간 선 (순환 도로) — 이 안 + 버퍼만 비교, 밖은 80 m 토막
        region_buf = float(cfg.get("buffer_m", 25)) + 10
    else:
        from env.road_config import ROAD
        net_path = net_path or os.path.join(BASE, "env/sumo/highway.net.xml")
        osm_path = os.path.join(BASE, ROAD["osm"]["file"])
        bbox = ROAD["osm"]["bbox"]
        mp = os.path.join(os.path.dirname(net_path), "highway.meta.json")
        meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        merges = meta.get("carriageway_merges") or (meta.get("signal_build") or {}).get("carriageway_merges", [])
        selected = None
        W_, S_, E__, N_ = bbox
        region_ll = [(S_, W_), (S_, E__), (N_, E__), (N_, W_)]
        region_buf = 0.0
    net = sumolib.net.readNet(net_path, withInternal=False)
    inet = sumolib.net.readNet(net_path, withInternal=True)
    nodes, ntags, ways, rels = load(osm_path)
    lat0 = sum(v[0] for v in nodes.values()) / len(nodes)
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110574

    def en(lat, lon):
        return (lon * kx, lat * ky)

    def xy_en(x, y):
        lon, lat = net.convertXY2LonLat(x, y)
        return en(lat, lon)

    region = [en(*p) for p in region_ll]

    def in_region(p):
        inside = False
        for (x1, y1), (x2, y2) in zip(region, region[1:] + region[:1]):
            if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1) + x1:
                inside = not inside
        if inside:
            return True
        return region_buf > 0 and min(_proj(p, a, b)[0] for a, b in zip(region, region[1:] + region[:1])) <= region_buf

    merged_ways = {w for md in merges for w in md["merged"]}
    busways = {b["busway"]: md for md in merges for b in md["lanes"]["busway_evidence"]}
    # 들어가야 할 OSM 도로
    if selected is not None:
        expected = {w for w in selected if w in ways}
    else:
        W, Sb, E_, N = bbox
        mlat, mlon = 0.0005, 0.0006                 # 약 55 m: 노드는 범위 밖이어도 선분이 범위를 지나는 도로 포함
        expected = {w for w, (t, nds) in ways.items()
                    if t.get("highway") in DRIVE and t.get("access") not in ("no", "private")
                    and any(n in nodes and W - mlon <= nodes[n][1] <= E_ + mlon and Sb - mlat <= nodes[n][0] <= N + mlat
                            for n in nds)}
    expected -= merged_ways
    lines_all = {w: [en(*nodes[n]) for n in ways[w][1] if n in nodes] for w in expected | set(busways) | merged_ways}
    lines_all = {w: pl for w, pl in lines_all.items() if len(pl) > 1}
    idx = SegIndex({w: pl for w, pl in lines_all.items() if w in expected})
    idx_m = SegIndex({w: pl for w, pl in lines_all.items() if w in busways or w in merged_ways})
    merged_prefix = tuple(md["new_id"] for md in merges)
    edges = net.getEdges()

    def opposite(e):
        i = e.getID()
        return net.hasEdge(i[1:] if i.startswith("-") else "-" + i)

    # ── 엣지 → OSM way 짝짓기
    edge_info = {}
    for e in edges:
        half = sum(ln.getWidth() for ln in e.getLanes()) / 2
        is_m = e.getID().lstrip("-").startswith(merged_prefix) if merged_prefix else False
        ix = idx_m if is_m else idx
        # 양방향 도로는 방향마다 차로 폭 절반만큼 옆에 놓인다. 엣지 id로는 반대 방향을 못 찾을 수 있어(합쳐진 엣지)
        # 항상 차로 폭 절반 + 2.5 m를 허용한다 (일방통행 도로에는 그만큼 너그러움)
        tol = half + 2.5 + (7.0 if is_m else 0.0)     # 합친 도로: 원래 차도 선은 가운데에서 약 9 m 옆
        pts = [xy_en(x, y) for x, y in e.getShape()]
        dense = []
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(math.dist(a, b) / 3))
            dense += [((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n), (b[0] - a[0], b[1] - a[1]))
                      for k in range(n)]
        cover = collections.Counter()
        dirs = collections.defaultdict(float)
        off = []
        for p, tv in dense:
            cand = ix.near(p, tol + 1.0)
            if not cand:
                off.append(p)
                continue
            w, (d, sv) = min(cand.items(), key=lambda kv: kv[1][0])
            if d > tol:
                off.append(p)
                continue
            cover[w] += 1
            dirs[w] += tv[0] * sv[0] + tv[1] * sv[1]
        edge_info[e.getID()] = {"cover": cover, "dir": {w: (1 if dirs[w] >= 0 else -1) for w in cover},
                                "n": len(dense), "off": len(off), "tol": tol, "merged": is_m}

    rep = {"net": os.path.relpath(net_path, BASE), "osm": os.path.relpath(osm_path, BASE)}

    # ── A 도로 유무 (방향별)
    need = []
    for w in expected:
        t = ways[w][0]
        if t.get("oneway") == "-1":
            need.append((w, -1))
        elif t.get("oneway") in ONEWAY or t.get("junction") in ("roundabout", "circular"):
            need.append((w, 1))
        else:
            need += [(w, 1), (w, -1)]
    # 덮음 = 엣지 + 교차로 내부 차로의 모양 (방향 포함). OSM 선을 3 m 간격으로 찍어 영역 안 점만 센다.
    cov_lines = [[xy_en(x, y) for x, y in e.getShape()] for e in edges]
    for n_ in inet.getNodes():
        for ln in n_.getInternal() if hasattr(n_, "getInternal") else []:
            pass
    for e in inet.getEdges():
        if e.getFunction() == "internal":
            for ln in e.getLanes():
                cov_lines.append([xy_en(x, y) for x, y in ln.getShape()])
    cov_idx = SegIndex({i: pl for i, pl in enumerate(cov_lines) if len(pl) > 1})
    missing = []
    for w, d in need:
        pl = lines_all.get(w, [])
        if len(pl) < 2:
            continue
        tot = got = 0
        for a, b in zip(pl, pl[1:]):
            n = max(1, int(math.dist(a, b) / 3))
            sv = ((b[0] - a[0]) * d, (b[1] - a[1]) * d)
            for k in range(n):
                p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                if not in_region(p):
                    continue
                tot += 3
                near = cov_idx.near(p, 9.0)
                if any(v[0] * sv[0] + v[1] * sv[1] > 0 for _, v in near.values()):
                    got += 3
        L = tot
        if L and got < L - 12:          # 영역 안 OSM 도로 중 12 m 넘게 빈 곳
            missing.append({"way": w, "name": ways[w][0].get("name"), "highway": ways[w][0].get("highway"),
                            "dir": "정" if d == 1 else "역", "osm_len_in_region_m": round(L), "covered_m": round(got)})
    rep["A_roads"] = {"expected_way_dirs": len(need), "missing": missing}

    # ── B 모양
    bad = [{"edge": e, "name": net.getEdge(e).getName(), "off_m": inf["off"] * 3, "len_m": round(net.getEdge(e).getLength())}
           for e, inf in edge_info.items() if inf["off"] * 3 > 12 and inf["off"] > 0.15 * inf["n"]]
    rep["B_shape"] = {"edges": len(edges), "off_line": bad}

    # ── C 방향
    wrong = []
    for eid, inf in edge_info.items():
        for w, c in inf["cover"].items():
            if c < 4 or w in busways:
                continue
            t = ways[w][0]
            ow = t.get("oneway") in ONEWAY or t.get("junction") in ("roundabout", "circular")
            if (ow and inf["dir"][w] < 0) or (t.get("oneway") == "-1" and inf["dir"][w] > 0):
                wrong.append({"edge": eid, "way": w, "name": t.get("name")})
    rep["C_direction"] = {"against_oneway": wrong,
                          "two_way_one_side": [m for m in missing if m["highway"] and
                                               not (ways[m["way"]][0].get("oneway") in ONEWAY)]}

    # ── D 차로 수
    cnt, mism, unk = collections.Counter(), [], collections.defaultdict(float)
    for e in edges:
        inf = edge_info[e.getID()]
        n_net = e.getLaneNumber()
        n_bus = sum(1 for ln in e.getLanes() if ln.allows("bus") and not ln.allows("passenger"))
        if inf["merged"]:
            md = next(m for m in merges if e.getID().lstrip("-").startswith(m["new_id"]))
            d = 1 if e.getID().startswith("-") else 0
            sid = e.getID().lstrip("-").split("#")[0]
            segl = (md["lanes"].get("segments") or {}).get(sid)
            if segl:
                g, b = (segl[0], segl[2]) if d == 0 else (segl[1], segl[3])
            else:
                g, b = md["lanes"]["general_per_dir"][d], md["lanes"]["bus_per_dir"][d]
            # 원본 근거: 이 엣지 옆 원래 차도(같은 방향)의 OSM lanes, 옆 busway의 lanes/2
            side = "정방향 차도(나감)" if d == 0 else "역방향 차도(들어옴)"
            e_pts = [xy_en(x, y) for x, y in e.getShape()]
            mid = e_pts[len(e_pts) // 2]
            near_w = sorted(((min(_proj(mid, a, b2)[0] for a, b2 in zip(lines_all[w], lines_all[w][1:])), w)
                             for w in md["merged"] if w in lines_all
                             and (ways[w][1][0] in md.get("one_side_nodes", {}) or True)), key=lambda x: x[0])
            src = {ways[w][0].get("lanes") for dd, w in near_w[:2] if dd < 25}
            near_b = [w for w in busways if w in lines_all and
                      min(_proj(pt, a, b2)[0] for pt in e_pts for a, b2 in zip(lines_all[w], lines_all[w][1:])) < 10]
            bl = [int(ways[w][0].get("lanes", "0") or 0) for w in near_b]
            ok = (n_net, n_bus) == (g + b, b) and str(g) in src and ((b == 0 and not bl) or all(v == 2 * b for v in bl) and bl)
            cnt["일치(합친 도로: 차도 lanes + busway/2)" if ok else "불일치"] += 1
            if not ok:
                mism.append({"edge": e.getID(), "net": (n_net, n_bus), "osm": f"차도 {src} busway {bl}"})
            continue
        main_w = [w for w, c in inf["cover"].items() if c * 3 >= max(9, 0.25 * inf["n"] * 3)]
        if not main_w:
            cnt["짝 없음(교차로 안 짧은 엣지)"] += 1
            continue
        exps, why_all = set(), []
        for w in main_w:
            if w in busways:
                md = busways[w]
                d = 0 if inf["dir"][w] > 0 else 1
                g, b = md["lanes"]["general_per_dir"], md["lanes"]["bus_per_dir"]
                # busway 방향과 합친 도로 방향이 같다고 볼 수 없으므로 두 방향 값이 같을 때만 비교
                if g[0] == g[1] and b[0] == b[1]:
                    src = {ways[x][0].get("lanes") for x in md["merged"] if x in ways}
                    blanes = int(ways[w][0].get("lanes", "0") or 0)
                    exps.add((g[0] + b[0], b[0]))
                    why_all.append(f"합친 도로: 차도 lanes {sorted(src)} + busway {blanes}/2")
                continue
            t = dict(ways[w][0])
            if w in splits:
                pass
            ex, why = expect(t)
            d = 0 if inf["dir"][w] > 0 else 1
            if ex is None and w in assumed and d == 0:
                ex, why = (assumed[w], 0), "가정"
            if ex is None:
                why_all.append(why)
                continue
            key = "bus:lanes" if t.get("oneway") in ONEWAY else ("bus:lanes:forward" if d == 0 else "bus:lanes:backward")
            nb = (t.get(key) or "").split("|").count("designated")
            exps.add((ex[d], nb))
            why_all.append(why)
        if not exps:
            reason = "OSM 모호(양방향 홀수)" if any(str(x).startswith("모호") for x in why_all) else "OSM에 차로 수 없음"
            cnt[reason] += 1
            unk[(e.getName() or "(이름없음)", reason, n_net)] += e.getLength()
            continue
        if (n_net, n_bus) in exps and len({x[0] for x in exps}) == 1:
            cnt["일치"] += 1
        elif (n_net, n_bus) in exps:
            cnt["일치(덮는 way들끼리 값이 다름 — 한쪽과 일치)"] += 1
        else:
            # way_splits 보정 구간 (OSM lanes를 일부러 바꾼 곳)
            if any(w in splits for w in main_w):
                cnt["보정 구간(way_splits)"] += 1
                continue
            cnt["불일치"] += 1
            mism.append({"edge": e.getID(), "name": e.getName(), "net": (n_net, n_bus), "osm": sorted(exps),
                         "ways": main_w, "why": why_all})
    rep["D_lanes"] = {"count": dict(cnt), "mismatch": mism,
                      "no_osm_value": [{"도로": k[0], "사유": k[1], "도로망_차로": k[2], "길이_m": round(v)}
                                       for k, v in sorted(unk.items(), key=lambda x: -x[1])]}

    # ── E 회전 제한
    way_edges = collections.defaultdict(list)
    for eid, inf in edge_info.items():
        for w, c in inf["cover"].items():
            if c >= 2:
                way_edges[w].append(eid)
    for md in merges:                        # 합친 도로: 원래 차도 → 합친 도로 엣지
        for w in md["merged"]:
            way_edges[w] = [e.getID() for e in edges if e.getID().lstrip("-").startswith(md["new_id"])]
    res, viol = collections.Counter(), []
    for rid, t, mem in rels:
        if t.get("type") != "restriction" or not t.get("restriction"):
            continue
        fr = next((r for ty, r, ro in mem if ro == "from"), None)
        to = next((r for ty, r, ro in mem if ro == "to"), None)
        via = next((r for ty, r, ro in mem if ro == "via" and ty == "node"), None)
        if via is None or via not in nodes:
            res["via가 way(검사 생략)"] += 1
            continue
        vp = en(*nodes[via])
        fe = [net.getEdge(x) for x in way_edges.get(fr, [])
              if math.dist(xy_en(*net.getEdge(x).getToNode().getCoord()), vp) < 25]
        te = {x for x in way_edges.get(to, []) if math.dist(xy_en(*net.getEdge(x).getFromNode().getCoord()), vp) < 25}
        if not fe or not te:
            res["도로망 밖"] += 1
            continue
        if fr == to or (fr in merged_ways and to in merged_ways):
            res["유턴 제한(도로망에 유턴 없음)"] += 1
            continue
        conn = any(c.getTo().getID() in te for e in fe for cl in e.getOutgoing().values() for c in cl)
        ok = (not conn) if t["restriction"].startswith("no_") else conn
        res["OK" if ok else "위반"] += 1
        if not ok:
            viol.append({"rel": rid, "type": t["restriction"], "from": fr, "via": via, "to": to})
    rep["E_restrictions"] = {"count": dict(res), "violations": viol}

    # ── F 신호
    tls_pts = []
    for tl in net.getTrafficLights():
        for c in tl.getConnections():
            tls_pts.append(xy_en(*c[0].getEdge().getToNode().getCoord()))
    sig = []
    for nid, tg in ntags.items():
        if not (tg.get("highway") == "traffic_signals" or tg.get("crossing") == "traffic_signals"):
            continue
        p = en(*nodes[nid])
        if not idx.near(p, 3.0):
            continue                            # 도로망 도로 위가 아님
        dmin = min((math.dist(p, q) for q in tls_pts), default=1e9)
        x, y = net.convertLonLat2XY(nodes[nid][1], nodes[nid][0])
        nd = min(net.getNodes(), key=lambda n: math.dist(n.getCoord(), (x, y)))
        dn = math.dist(nd.getCoord(), (x, y))
        where = ("영역 밖/경계" if not in_region(p) or dn > 15 or nd.getType() == "dead_end" else "")
        sig.append({"node": nid, "kind": "신호 횡단보도" if tg.get("highway") == "crossing" else "신호등",
                    "nearest_tls_m": round(dmin), "ok": dmin <= 30 or bool(where), "note": where})
    juncs = []
    tls_nodes = {c[0].getEdge().getToNode().getID() for tl in net.getTrafficLights() for c in tl.getConnections()}
    for n in net.getNodes():
        if n.getID() in tls_nodes or n.getType() in ("dead_end",):
            continue
        es = n.getIncoming() + n.getOutgoing()
        nb = {(e.getToNode() if e.getFromNode() == n else e.getFromNode()).getID() for e in es}
        if len(nb) < 3:
            continue
        cls = set()
        for e in es:
            cv = edge_info[e.getID()]["cover"]
            if cv:
                w = cv.most_common(1)[0][0]
                cls.add(ways[w][0].get("highway", "busway") if w in ways else "?")
        juncs.append({"node": n.getID(), "type": n.getType(), "legs": len(nb),
                      "roads": sorted({e.getName() or "(이름없음)" for e in es}), "classes": sorted(cls)})
    rep["F_signals"] = {"tls_in_net": len(net.getTrafficLights()), "osm_signals_on_roads": sig,
                        "unsignalized_junctions": juncs}
    return rep


def print_rep(rep):
    print(f"== {rep['net']}  ↔  {rep['osm']}")
    a = rep["A_roads"]
    print(f"A 도로 유무: OSM 도로(방향별) {a['expected_way_dirs']}개 중 도로망에 없음/부족 {len(a['missing'])}")
    for x in a["missing"][:20]:
        print("     ", x)
    b = rep["B_shape"]
    print(f"B 모양: 엣지 {b['edges']}개 중 OSM 선에서 벗어난 엣지 {len(b['off_line'])}")
    for x in b["off_line"][:15]:
        print("     ", x)
    c = rep["C_direction"]
    print(f"C 방향: 일방통행 역주행 엣지 {len(c['against_oneway'])} {c['against_oneway'][:5]}")
    d = rep["D_lanes"]
    print(f"D 차로 수: {d['count']}")
    for x in d["mismatch"][:20]:
        print("     불일치:", x)
    e = rep["E_restrictions"]
    print(f"E 회전 제한: {e['count']}")
    for x in e["violations"][:10]:
        print("     위반:", x)
    f = rep["F_signals"]
    miss = [s for s in f["osm_signals_on_roads"] if not s["ok"]]
    edge_ = [s["node"] for s in f["osm_signals_on_roads"] if s.get("note")]
    print(f"F 신호: 도로망 신호 {f['tls_in_net']}곳 | 도로 위 OSM 신호 노드 {len(f['osm_signals_on_roads'])}개 중 빠짐 {len(miss)} {miss[:10]}"
          f" (영역 밖/도로망 끝이라 넣지 않음 {len(edge_)}: {edge_})"
          f" | 신호 없는 3갈래 이상 교차로 {len(f['unsignalized_junctions'])}곳")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "area"
    rep = main(which, sys.argv[2] if len(sys.argv) > 2 else None)
    print_rep(rep)
    out = os.path.join(BASE, "env/maps", f"audit_{which}.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→", os.path.relpath(out, BASE))
