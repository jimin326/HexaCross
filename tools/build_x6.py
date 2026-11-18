"""연산교차로(6지) 정밀 모델: 차로 조사 자료(env/maps/yeonsan_x6.json)대로 교차로를 다시 짓는다.

  python tools/build_x6.py NET_IN NET_OUT [SPEC]      # 다시 짓고 검사 결과 출력
  python tools/build_x6.py --check NET [SPEC]         # 검사만

교차로 중심 반경 rebuild_radius_m 안의 6개 대로·우회전 연결로를 지우고 다음처럼 새로 만든다.
  - 진입로: 경계 → (차로 수가 바뀌는 지점들, 분리 우회전 분기점) → 교차로 노드 하나.
    정지선 차로 수와 차로마다 갈 수 있는 진출 도로는 자료 그대로 (차로 id 1 = 중앙선 쪽).
    BRT 버스전용차로는 맨 왼쪽 차로 allow="bus".
  - 진출로: 교차로 → (분리 우회전 합류점) → 경계. 차로 수 = 자료의 out(+BRT).
  - 분리 우회전 차로: 진입로 맨 오른쪽 차로에서 갈라져 신호 없이 진출로 맨 오른쪽 차로로 합류(양보).
  - 교차로 연결은 자료의 차로별 진출 도로만 남긴다 (좌회전 금지 방향은 연결 자체가 없음).
    같은 진출 도로로 가는 차로들은 진출 차로에 순서대로 배정 (우회전 성격이면 바깥 차로부터) — 자료 그림과 같은 규칙.
  - 신호: 진입로별 동시신호 (자료: E·S·W는 녹색+좌회전 화살표가 함께, N은 방면별 신호등, 주기는 공개 자료 없음 → 가정).
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
SPEC_DEFAULT = os.path.join(BASE, "env/maps/yeonsan_x6.json")
MIN_SEG = 4.0          # 차로 수 변화·분기 지점 사이 최소 거리 (m). 더 가까우면 한 지점으로 합침


def _ang(a, b):
    return abs((a - b + 180) % 360 - 180)


def _brg(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360


def _turn(spec, o, d):
    """o에서 들어와 d로 나갈 때 꺾는 각 (+ = 오른쪽)"""
    hin = (spec["legs"][o]["bearing"] + 180) % 360
    return ((spec["legs"][d]["bearing"] - hin + 540) % 360) - 180


def n_main(spec, o):
    return len(spec["legs"][o]["lanes"])


def n_out(spec, d):
    lg = spec["legs"][d]
    return lg["out_general"] + (1 if lg.get("brt") else 0)


def exit_map(spec, o, d):
    """자료 그림과 같은 규칙: d로 가는 o의 차로들 → d의 진출 차로 (id 1 = 중앙선 쪽).
    BRT는 BRT로, 일반 차로는 우회전 성격(+25° 넘게 꺾음)이면 바깥 차로부터, 아니면 안쪽부터 순서대로."""
    L = [l for l in spec["legs"][o]["lanes"] if d in l["to"]]
    res, first, last = {}, (2 if spec["legs"][d].get("brt") else 1), n_out(spec, d)
    dbrt = bool(spec["legs"][d].get("brt"))
    for l in L:
        if l["kind"] == "brt" and dbrt:
            res[l["id"]] = 1                      # BRT → 진출로 BRT
    # BRT가 없는 진출로로 가는 BRT 차로(예: SW BRT2 → 안락)는 일반 차로처럼 순서대로
    g = [l for l in L if not (l["kind"] == "brt" and dbrt)]
    right = _turn(spec, o, d) > 25
    for k, l in enumerate(g):
        j = last - (len(g) - 1 - k) if right else first + k
        res[l["id"]] = max(first, min(last, j))
    return res


def _poly(e, nodes):
    if e.get("shape"):
        return [tuple(map(float, p.split(",")[:2])) for p in e.get("shape").split()]
    return [nodes[e.get("from")], nodes[e.get("to")]]


def _cut_at(pts, C, R):
    """pts(밖 → 안 순서)가 반경 R 원과 처음 만나는 점까지. 반환 (밖 부분 점 목록, 교점)"""
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if math.dist(a, C) > R >= math.dist(b, C):
            d = (b[0] - a[0], b[1] - a[1])
            f = (a[0] - C[0], a[1] - C[1])
            A = d[0] ** 2 + d[1] ** 2
            Bq = 2 * (f[0] * d[0] + f[1] * d[1])
            Cq = f[0] ** 2 + f[1] ** 2 - R * R
            disc = max(Bq * Bq - 4 * A * Cq, 0.0)
            t = (-Bq - math.sqrt(disc)) / (2 * A)
            t = min(max(t, 0.0), 1.0)
            P = (a[0] + t * d[0], a[1] + t * d[1])
            return out + [P], P
        out.append(b)
    return None, None


LANE_W = 3.2           # netconvert 기본 차로 폭


def _left_shift(a, b, p, d):
    """a→b 진행 방향의 왼쪽으로 p를 d만큼 옮긴 점"""
    L = math.dist(a, b)
    return (p[0] - (b[1] - a[1]) / L * d, p[1] + (b[0] - a[0]) / L * d)


def _lanes_of(e):
    return int(e.get("numLanes", "1"))


def _gen_at(lg, dist_from_stop, gen_outside):
    v = None
    for d, n in lg["general_profile"]:
        if dist_from_stop >= d:
            v = n
    return gen_outside if v is None else v


def _spread(nu, nd):
    """nu개 차로 → nd개 차로를 고르게 잇는 (위, 아래) 쌍"""
    pairs = []
    if nd >= nu:
        for j in range(nd):
            pairs.append((0 if nu == 1 else round(j * (nu - 1) / (nd - 1)), j))
    else:
        for i in range(nu):
            pairs.append((i, 0 if nd == 1 else round(i * (nd - 1) / (nu - 1))))
    return pairs


def _brt_at(lg, dist_from_stop):
    """정지선에서 dist m 떨어진 곳의 BRT 차로 수. brt_profile [[거리, 수], ...]가 있으면 그것, 없으면 BRT면 1"""
    if not lg.get("brt"):
        return 0
    v = None
    for d, n in lg.get("brt_profile", []):
        if dist_from_stop >= d:
            v = n
    return 1 if v is None else int(v)


def _map_lanes(nu, nd, bu, bd=None):
    """차로 수가 바뀌는 지점의 연결 (위 차로 → 아래 차로). bu/bd = 위/아래의 BRT 차로 수 (맨 왼쪽).
    BRT는 BRT끼리 (BRT 1 → 2처럼 늘어나면 고르게), 일반 차로는 일반 차로끼리."""
    bu = int(bu); bd = bu if bd is None else int(bd)
    gu, gd = nu - bu, nd - bd
    pairs = []
    if bu and bd:
        pairs += [(gu + i, gd + j) for i, j in _spread(bu, bd)]
    if gd >= gu:
        for j in range(gd):
            i = 0 if gd == 1 else round(j * (gu - 1) / (gd - 1))
            pairs.append((i, j))
    else:
        for i in range(gu):
            j = 0 if gu == 1 else round(i * (gd - 1) / (gu - 1))
            pairs.append((i, j))
    return sorted(set(pairs))


def _stop_radii(net_path, spec):
    """지은 도로망에서 진입로별 실제 정지선(정지선 엣지 끝)의 중심 거리"""
    import sumolib
    net = sumolib.net.readNet(net_path)
    C = net.getNode(str(spec["center_node"])).getCoord()
    out = {}
    for o in spec["legs"]:
        es = [e for e in net.getEdges() if re.match(rf"x6_{o}_in\d+$", e.getID()) and e.getToNode().getID() == str(spec["center_node"])]
        if es:
            out[o] = round(math.dist(es[0].getShape()[-1], C), 1)
    return out


def apply_x6(net_in, net_out, spec_path=SPEC_DEFAULT, workdir=None):
    """두 번 짓는다: 1차로 교차로 모양(정지선 위치)을 정하고, 2차에서 자료의 거리(정지선 앞 m)를
    실제 정지선 기준으로 맞춘다 (가정값 stop_r 대신 netconvert가 만든 정지선 위치)."""
    import shutil
    spec = json.load(open(spec_path, encoding="utf-8"))
    tmp = workdir or tempfile.mkdtemp(prefix="x6_")
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, "x6_source.net.xml")
    if os.path.abspath(net_in) != os.path.abspath(src):
        shutil.copyfile(net_in, src)
    stop = {o: float(lg["stop_r"]) for o, lg in spec["legs"].items()}
    for it in range(3):
        rep = _apply_once(src, net_out, spec, os.path.join(tmp, f"pass{it}"), stop)
        got = _stop_radii(net_out, spec)
        diff = max(abs(got[o] - stop[o]) for o in stop)
        rep["stop_line_r"] = got
        rep["stop_line_iter"] = it + 1
        if diff <= 1.0:
            break
        stop = got
    return rep


def _apply_once(net_in, net_out, spec, tmp, stop_r):
    import sumolib
    from tools.merge_nets import export, _plain_args, _files, _run
    os.makedirs(tmp, exist_ok=True)
    prefix = os.path.join(tmp, "base")
    export(net_in, prefix)
    f = _files(prefix)
    nod_t, edg_t, con_t = ET.parse(f["nod"]), ET.parse(f["edg"]), ET.parse(f["con"])
    tll_t = ET.parse(f["tll"]) if "tll" in f else None
    nroot, eroot, croot = nod_t.getroot(), edg_t.getroot(), con_t.getroot()
    nel = {n.get("id"): n for n in nroot.findall("node")}
    nodes = {k: (float(v.get("x")), float(v.get("y"))) for k, v in nel.items()}
    Cid = str(spec["center_node"])
    if Cid not in nodes:
        raise RuntimeError(f"중심 노드 {Cid}가 도로망에 없음")
    C = nodes[Cid]
    R = float(spec["rebuild_radius_m"])
    inside = {k for k, p in nodes.items() if math.dist(p, C) <= R}
    eel = {e.get("id"): e for e in eroot.findall("edge")}

    # 지울 신호: 반경 안 노드를 제어하던 신호
    src = sumolib.net.readNet(net_in)
    old_tls = {tl.getID() for tl in src.getTrafficLights()
               if any(c[0].getEdge().getToNode().getID() in inside for c in tl.getConnections())}

    deleted, cross = set(), []          # cross: (edge id, 방향 'in'/'out', 교점)
    bnodes = []                          # (점, id)

    def bnode(P, leg_hint):
        for q, nid in bnodes:
            if math.dist(q, P) < 3.0:
                return nid
        nid = f"x6b_{len(bnodes)}"
        el = ET.SubElement(nroot, "node", {"id": nid, "x": f"{P[0]:.2f}", "y": f"{P[1]:.2f}", "type": "priority"})
        nel[nid] = el
        nodes[nid] = P
        bnodes.append((P, nid))
        return nid

    for eid, e in list(eel.items()):
        fi, ti = e.get("from") in inside, e.get("to") in inside
        if fi and ti:
            deleted.add(eid)
        elif ti or fi:
            pts = _poly(e, nodes)
            inbound = ti
            seq = pts if inbound else pts[::-1]
            part, P = _cut_at(seq, C, R)
            if part is None:
                raise RuntimeError(f"{eid}: 반경 경계와 만나는 점을 못 찾음")
            nid = bnode(P, None)
            if inbound:
                e.set("to", nid)
                e.set("shape", " ".join(f"{x:.2f},{y:.2f}" for x, y in part))
            else:
                e.set("from", nid)
                e.set("shape", " ".join(f"{x:.2f},{y:.2f}" for x, y in part[::-1]))
            for ln in e.findall("lane"):
                ln.attrib.pop("shape", None)
            cross.append((eid, "in" if inbound else "out", P))
    for eid in deleted:
        eroot.remove(eel.pop(eid))
    for nid in inside - {Cid}:
        nroot.remove(nel.pop(nid))
    cn = nel[Cid]
    cn.set("type", "traffic_light")
    cn.set("tl", Cid)
    for k in [k for k in cn.attrib if k in ("shape", "radius", "keepClear", "controlledInner")]:
        del cn.attrib[k]

    # 경계 도로 → 진입로(leg)
    legs = spec["legs"]
    leg_in, leg_out = {}, {}
    for eid, kind, P in cross:
        b = _brg(P[0] - C[0], P[1] - C[1])
        o = min(legs, key=lambda k: _ang(b, legs[k]["bearing"]))
        if _ang(b, legs[o]["bearing"]) > 35:
            raise RuntimeError(f"{eid}: 방위 {b:.0f}°가 어느 대로와도 맞지 않음 (반경 안에 다른 도로?)")
        tgt = leg_in if kind == "in" else leg_out
        if o in tgt:
            raise RuntimeError(f"{o}: {kind} 도로가 둘 이상 ({tgt[o]}, {eid})")
        tgt[o] = (eid, P)
    for o in legs:
        if o not in leg_in or o not in leg_out:
            raise RuntimeError(f"{o}: 경계에서 진입/진출 도로를 못 찾음 (in={leg_in.get(o)}, out={leg_out.get(o)})")

    cons = []

    def add_con(a, b, fl, tl):
        cons.append((a, b, int(fl), int(tl)))

    def new_node(nid, P, typ="priority"):
        el = ET.SubElement(nroot, "node", {"id": nid, "x": f"{P[0]:.2f}", "y": f"{P[1]:.2f}", "type": typ})
        nel[nid] = el
        nodes[nid] = P
        return nid

    def new_edge(eid, a, b, n, proto, brt=False, shape=None, typ=None, speed=None, prio=None, name=None):
        attrs = {"id": eid, "from": a, "to": b, "numLanes": str(n), "spreadType": "right",
                 "priority": prio or proto.get("priority", "10"), "type": typ or proto.get("type", ""),
                 "speed": speed or proto.get("speed", "13.89")}
        nm = proto.get("name") if name is None else name
        if nm:
            attrs["name"] = nm
        if shape:
            attrs["shape"] = " ".join(f"{x:.2f},{y:.2f}" for x, y in shape)
        el = ET.SubElement(eroot, "edge", attrs)
        for k in range(int(brt)):                 # 맨 왼쪽 brt개 차로 = 버스전용
            ET.SubElement(el, "lane", {"index": str(n - 1 - k), "allow": "bus"})
        eel[eid] = el
        return eid

    final_in, first_out, split_info, merge_info, report = {}, {}, {}, {}, {"legs": {}}
    # 진출로 먼저 (분리 우회전 합류점이 필요)
    slip_to = {lg["slip"]["to"]: o for o, lg in legs.items() if lg.get("slip")}
    for d, lg in legs.items():
        eid, P = leg_out[d]
        proto = eel[eid]
        brt = bool(lg.get("brt"))
        m = n_out(spec, d)
        if _lanes_of(proto) != m:
            report.setdefault("warnings", []).append(f"{d} 진출: 경계 밖 도로 {_lanes_of(proto)}차로, 자료 {m}차로 → 경계에서 맞춤")
        u = ((P[0] - C[0]) / math.dist(P, C), (P[1] - C[1]) / math.dist(P, C))
        stop = stop_r[d]
        chain = [Cid]
        if d in slip_to:
            rm = stop + float(legs[slip_to[d]]["slip"]["merge_m"])
            mid = new_node(f"x6_{d}_merge", (C[0] + u[0] * rm, C[1] + u[1] * rm))
            merge_info[d] = mid
            chain.append(mid)
        chain.append(eel[eid].get("from"))
        prev = None
        for k, (a, b) in enumerate(zip(chain, chain[1:])):
            shp = None
            if k == len(chain) - 2 and proto.get("spreadType") == "center":
                # 경계 밖 도로가 가운데 기준 차로 배치면, 마지막 조각 끝을 옮겨 맨 오른쪽 차로 가장자리가 그대로 이어지게
                shp = [nodes[a], _left_shift(nodes[a], nodes[b], nodes[b], (m - _lanes_of(proto) / 2) * LANE_W)]
            sid = new_edge(f"x6_{d}_out{k}", a, b, m, proto, brt=brt, shape=shp)
            if k == 0:
                first_out[d] = sid
            if prev:
                for i in range(m):
                    add_con(prev, sid, i, i)
            prev = sid
        for i in range(min(m, _lanes_of(proto))):
            add_con(prev, eid, i, i)
        if _lanes_of(proto) > m:      # 경계 밖이 더 넓으면 바깥쪽 차로를 맨 오른쪽에서 이어 줌
            for i in range(m, _lanes_of(proto)):
                add_con(prev, eid, m - 1, i)
        elif _lanes_of(proto) < m:    # 경계 밖이 더 좁으면 남는 왼쪽 차로를 맨 왼쪽 차로로 합류 (막다른 차로 방지)
            for i in range(_lanes_of(proto), m):
                add_con(prev, eid, i, _lanes_of(proto) - 1)
        merge_info.setdefault(d, None)
        report["legs"].setdefault(d, {})["out"] = {"lanes": m, "edges": [f"x6_{d}_out{k}" for k in range(len(chain) - 1)]}

    for o, lg in legs.items():
        eid, P = leg_in[o]
        proto = eel[eid]
        brt = bool(lg.get("brt"))
        n_side = _lanes_of(proto)
        gen_out = n_side - (1 if brt else 0)
        u = ((P[0] - C[0]) / math.dist(P, C), (P[1] - C[1]) / math.dist(P, C))
        rb = math.dist(P, C)
        stop = stop_r[o]
        brk = {}
        for dprof, _ in lg["general_profile"]:
            r = stop + float(dprof)
            if dprof > 0 and r < rb - 3:
                brk[round(r, 1)] = "taper"
        if lg.get("slip"):
            rs = stop + float(lg["slip"]["split_m"])
            if rs >= rb - 3:
                raise RuntimeError(f"{o}: 분리 우회전 분기점({rs} m)이 반경 밖")
            brk[round(rs, 1)] = "split"
        # MIN_SEG 안에 붙은 지점만 하나로 (조각 길이가 0이 되는 것 방지). 분기점을 우선,
        # 경계에서 MIN_SEG 안의 차로 변화는 경계에서 바로 바뀌게 둔다
        merged_brk = []
        for r in sorted(brk, reverse=True):
            if rb - r < MIN_SEG:
                if brk[r] == "split":
                    raise RuntimeError(f"{o}: 분리 우회전 분기점이 반경 경계와 너무 가까움")
                continue
            if merged_brk and merged_brk[-1][0] - r < MIN_SEG:
                pr, pk = merged_brk[-1]
                merged_brk[-1] = (r if brk[r] == "split" else pr, "split" if "split" in (pk, brk[r]) else pk)
            else:
                merged_brk.append((r, brk[r]))
        brk = dict(merged_brk)
        rs_list = sorted(brk, reverse=True)
        chain = [eel[eid].get("to")]
        kinds = ["boundary"]
        for r in rs_list:
            chain.append(new_node(f"x6_{o}_in_{int(r)}", (C[0] + u[0] * r, C[1] + u[1] * r)))
            kinds.append(brk[r])
        chain.append(Cid)
        radii = [rb] + rs_list + [0.0]
        segs = []
        for k, (a, b) in enumerate(zip(chain, chain[1:])):
            rmid = (radii[k] + max(radii[k + 1], stop)) / 2
            g = _gen_at(lg, max(rmid - stop, 0.0), gen_out)
            nb = _brt_at(lg, max(rmid - stop, 0.0))
            n = g + nb
            shp = None
            if k == 0 and proto.get("spreadType") == "center":
                # 경계 밖 도로가 가운데 기준 차로 배치면, 첫 조각 시작을 옮겨 맨 오른쪽 차로 가장자리가 그대로 이어지게
                shp = [_left_shift(nodes[a], nodes[b], nodes[a], (n - n_side / 2) * LANE_W), nodes[b]]
            sid = new_edge(f"x6_{o}_in{k}", a, b, n, proto, brt=nb, shape=shp)
            segs.append((sid, n, b, nb))
        if segs[-1][1] != n_main(spec, o) or segs[-1][3] != sum(1 for l in lg["lanes"] if l["kind"] == "brt"):
            raise RuntimeError(f"{o}: 정지선 차로 {segs[-1][1]} ≠ 자료 {n_main(spec, o)}")
        final_in[o] = segs[-1][0]
        fe = eel[segs[-1][0]]
        for l in lg["lanes"]:
            if l.get("stop_offset_m"):
                idx = str(segs[-1][1] - l["id"])
                le = next((x for x in fe.findall("lane") if x.get("index") == idx), None)
                if le is None:
                    le = ET.SubElement(fe, "lane", {"index": idx})
                ET.SubElement(le, "stopOffset", {"value": f'{float(l["stop_offset_m"]):g}'})
        # 경계 → 첫 조각
        for i, j in _map_lanes(n_side, segs[0][1], 1 if brt else 0, segs[0][3]):
            add_con(eid, segs[0][0], i, j)
        for (s1, n1, nd, b1), (s2, n2, _, b2) in zip(segs, segs[1:]):
            for i, j in _map_lanes(n1, n2, b1, b2):
                add_con(s1, s2, i, j)
            if nd in chain and kinds[chain.index(nd)] == "split":
                split_info[o] = (nd, s1)
        report["legs"].setdefault(o, {})["in"] = {"segments": [(s_[0], s_[1]) for s_ in segs], "boundary_lanes": n_side}

    # 분리 우회전 차로
    for o, lg in legs.items():
        sl = lg.get("slip")
        if not sl:
            continue
        d = sl["to"]
        nd, up = split_info[o]
        mid = merge_info[d]
        a, b = nodes[nd], nodes[mid]
        uo = (a[0] - C[0], a[1] - C[1])
        ud = (b[0] - C[0], b[1] - C[1])
        lo, ld = math.hypot(*uo), math.hypot(*ud)
        bis = (uo[0] / lo + ud[0] / ld, uo[1] / lo + ud[1] / ld)
        bl = math.hypot(*bis)
        rc = min(lo, ld) * 0.95
        ctrl = (C[0] + bis[0] / bl * rc, C[1] + bis[1] / bl * rc)
        shape = [((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * ctrl[0] + t * t * b[0],
                  (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * ctrl[1] + t * t * b[1]) for t in [i / 12 for i in range(13)]]
        sid = new_edge(f"x6_slip_{o}_{d}", nd, mid, 1, eel[leg_in[o][0]], shape=shape,
                       typ="highway.secondary_link", speed="8.33",
                       prio=str(max(int(eel[leg_in[o][0]].get("priority", "10")) - 2, 1)), name="")
        add_con(up, sid, 0, 0)
        merged_next = f"x6_{d}_out1"
        add_con(sid, merged_next, 0, 0)

    # 교차로 연결: 자료의 차로별 진출 도로
    for o, lg in legs.items():
        n = n_main(spec, o)
        m_map = {}
        for d in legs:
            if d == o:
                continue
            for lid, j in exit_map(spec, o, d).items():
                add_con(final_in[o], first_out[d], n - lid, n_out(spec, d) - j)

    # 기존 연결 정리: 지운 도로·끊은 도로의 옛 연결
    cut_in = {e for e, k, _ in cross if k == "in"}
    for c in list(croot.findall("connection")):
        a, b = c.get("from"), c.get("to")
        if a in deleted or b in deleted or a in cut_in or (b is None and a in deleted):
            croot.remove(c)
    for c in list(croot.findall("crossing")) + list(croot.findall("walkingArea")):
        croot.remove(c)
    for a, b, fl, tl in sorted(set(cons)):
        ET.SubElement(croot, "connection", {"from": a, "to": b, "fromLane": str(fl), "toLane": str(tl)})
    # 신호: 반경 안을 제어하던 옛 신호 삭제 (교차로 노드 신호는 netconvert가 새로 만들고 아래에서 현시를 씀)
    if tll_t is not None:
        troot = tll_t.getroot()
        for el in list(troot):
            if el.tag == "tlLogic" and el.get("id") in old_tls:
                troot.remove(el)
            elif el.tag == "connection" and (el.get("tl") in old_tls or el.get("from") in deleted
                                             or el.get("to") in deleted or el.get("from") in cut_in):
                troot.remove(el)
        tll_t.write(f["tll"], encoding="UTF-8", xml_declaration=True)
    for n in nroot.findall("node"):
        if n.get("tl") in old_tls and n.get("id") != Cid:
            n.attrib.pop("tl", None)
            if n.get("type") == "traffic_light":
                n.set("type", "priority")
    nod_t.write(f["nod"], encoding="UTF-8", xml_declaration=True)
    edg_t.write(f["edg"], encoding="UTF-8", xml_declaration=True)
    con_t.write(f["con"], encoding="UTF-8", xml_declaration=True)
    _run(_plain_args(f) + ["--output.street-names", "--tls.default-type", "static", "-o", net_out])

    # netconvert가 더 만든 연결은 지운다 (새로 지은 도로에서 나가는 것)
    want = set(cons)
    net = sumolib.net.readNet(net_out)
    extra = []
    new_edges = {e for e in eel if e.startswith("x6_")} | cut_in
    for eid in new_edges:
        if not net.hasEdge(eid):
            continue
        for to, cl in net.getEdge(eid).getOutgoing().items():
            for c in cl:
                key = (eid, to.getID(), c.getFromLane().getIndex(), c.getToLane().getIndex())
                if key not in want:
                    extra.append(key)
    if extra:
        dpath = os.path.join(tmp, "x6_delete.con.xml")
        with open(dpath, "w") as fh:
            fh.write("<connections>\n" + "".join(
                f'    <delete from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n' for a, b, fl, tl in extra)
                     + "</connections>\n")
        _run(["-s", net_out, "--connection-files", dpath, "--output.street-names", "-o", net_out])
    report["extra_connections_removed"] = len(extra)
    report["center_signal"] = write_center_program(net_out, spec)
    return report


def leg_slot(g, Y, AR, lanes, wl=None):
    """한 진입로 차례(녹색 g + 황색 Y + 전적색 AR)의 현시들. lanes = {링크 번호: 차로 id(1 = 중앙선 쪽)}.
    wl (자료의 진입로 안 차이, signal_plan_assumed.within_leg):
      lead_lanes / lead_frac(또는 lead_s)                : 이 차로들이 먼저 켜지고 나머지는 녹색의 lead_frac 뒤에 켜짐
      early_end_lanes / early_end_frac(또는 early_end_s) : 이 차로들이 녹색의 early_end_frac만큼 먼저 꺼짐
    반환 [(시간, {링크 번호: 'G'|'y'|'r'})]"""
    wl = wl or {}
    win = {}
    for i, lid in lanes.items():
        on, off = 0.0, g
        if wl.get("lead_lanes") and lid not in wl["lead_lanes"]:
            on = min(round(float(wl["lead_frac"]) * g) if "lead_frac" in wl else float(wl["lead_s"]), g - 1.0)
        if wl.get("early_end_lanes") and lid in wl["early_end_lanes"]:
            off = max(g - (round(float(wl["early_end_frac"]) * g) if "early_end_frac" in wl else float(wl["early_end_s"])), 1.0)
        win[i] = (on, off)
    slot = g + Y + AR
    cuts = sorted({0.0, slot} | {t for on, off in win.values() for t in (on, off, off + Y)})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 1e-6:
            continue
        out.append((b - a, {i: ("G" if on <= a < off else "y" if off <= a < off + Y else "r") for i, (on, off) in win.items()}))
    return out


def link_lane_ids(net_path, tl_id):
    """신호 링크 번호 → (진입로 id, 차로 id) — 정밀 모델 진입로(x6_<leg>_in*)만"""
    import sumolib
    net = sumolib.net.readNet(net_path)
    out = {}
    for fl, _, idx in net.getTLS(tl_id).getConnections():
        e = fl.getEdge()
        m = re.match(r"x6_(\w+?)_in\d+$", e.getID())
        if m:
            out[idx] = (m.group(1), e.getLaneNumber() - fl.getIndex())
    return out


def write_center_program(net_path, spec):
    """교차로 신호 현시: 진입로별 동시신호 (자료에 주기 없음 → spec['signal_plan_assumed']).
    진입로 안에서 자료가 말한 차로 묶음별 차이를 넣는다:
      within_leg[o].lead_lanes / lead_s        : 이 차로들이 먼저 켜지고 나머지는 lead_s 뒤에 켜짐 (N: 방면별 신호 따로)
      within_leg[o].early_end_lanes / early_end_s : 이 차로들이 녹색보다 early_end_s 먼저 꺼짐 (SW: 교대 방면 화살표)"""
    import sumolib
    net = sumolib.net.readNet(net_path)
    Cid = str(spec["center_node"])
    tl = next((t for t in net.getTrafficLights()
               if any(c[0].getEdge().getToNode().getID() == Cid for c in t.getConnections())), None)
    if tl is None:
        raise RuntimeError("교차로 신호가 만들어지지 않음")
    plan = spec["signal_plan_assumed"]
    Y, AR = float(plan["yellow"]), float(plan["all_red"])
    conns = tl.getConnections()
    n = 1 + max(c[2] for c in conns)
    group = {}
    for fl, _, idx in conns:
        e = fl.getEdge()
        m = re.match(r"x6_(\w+?)_in\d+$", e.getID())
        group[idx] = (m.group(1), e.getLaneNumber() - fl.getIndex()) if m else (None, None)
    phases = []
    for leg in plan["order"]:
        lanes = {i: lid for i, (lg_, lid) in group.items() if lg_ == leg}
        for dur, st in leg_slot(float(plan["green"][leg]), Y, AR, lanes, (plan.get("within_leg") or {}).get(leg)):
            phases.append((dur, "".join(st.get(i, "r") for i in range(n))))
    txt = open(net_path, encoding="utf-8").read()
    pat = re.compile(r'(<tlLogic id="' + re.escape(tl.getID()) + r'"[^>]*>)(.*?)(</tlLogic>)', re.S)
    body = "".join(f'\n        <phase duration="{d:g}" state="{s}"/>' for d, s in phases) + "\n    "
    txt = pat.sub(lambda mm: mm.group(1) + body + mm.group(3), txt, count=1)
    open(net_path, "w", encoding="utf-8").write(txt)
    return {"tls": tl.getID(), "links": n, "cycle_s": sum(d for d, _ in phases), "phases": len(phases)}


def check_x6(net_path, spec_path=SPEC_DEFAULT):
    """자료와 도로망 대조: 정지선 차로 수, BRT, 차로별 진출 도로(정확히 같아야), 금지 방향, 분리 우회전, 진출 차로 수."""
    import sumolib
    spec = json.load(open(spec_path, encoding="utf-8"))
    net = sumolib.net.readNet(net_path)
    legs = spec["legs"]
    Cid = str(spec["center_node"])

    def leg_of_out(e):
        mm = re.match(r"x6_(\w+?)_out\d+$", e.getID())
        return mm.group(1) if mm else None
    rows, problems = [], []
    for o, lg in legs.items():
        fin = [e for e in net.getEdges() if re.match(rf"x6_{o}_in\d+$", e.getID()) and e.getToNode().getID() == Cid]
        if len(fin) != 1:
            problems.append(f"{o}: 정지선 엣지 {len(fin)}개")
            continue
        e = fin[0]
        n = e.getLaneNumber()
        if n != len(lg["lanes"]):
            problems.append(f"{o}: 정지선 차로 {n} ≠ 자료 {len(lg['lanes'])}")
        for l in lg["lanes"]:
            ln = e.getLane(n - l["id"])
            got = sorted({leg_of_out(c.getTo()) for c in ln.getOutgoing()})
            want = sorted(l["to"])
            bus_only = ln.allows("bus") and not ln.allows("passenger")
            ok = got == want and bus_only == (l["kind"] == "brt")
            rows.append({"leg": o, "lane": l["id"], "kind": l["kind"], "want": want, "got": got, "bus_only": bus_only, "ok": ok})
            if not ok:
                problems.append(f"{o} {l['id']}차로: 자료 {want} / 도로망 {got} / 버스전용 {bus_only}")
        allg = {leg_of_out(c.getTo()) for ln in e.getLanes() for c in ln.getOutgoing()}
        bad = allg & set(lg.get("ban", []))
        if bad:
            problems.append(f"{o}: 금지 방향으로 연결 있음 {bad}")
        if lg.get("slip"):
            sl = net.getEdge(f"x6_slip_{o}_{lg['slip']['to']}") if net.hasEdge(f"x6_slip_{o}_{lg['slip']['to']}") else None
            nxt = [c.getTo().getID() for c in sl.getLane(0).getOutgoing()] if sl else []
            ok = sl is not None and any(x.startswith(f"x6_{lg['slip']['to']}_out") for x in nxt) and sl.getToNode().getType() != "traffic_light"
            rows.append({"leg": o, "lane": "분리", "kind": "slip", "want": [lg["slip"]["to"]], "got": [lg["slip"]["to"]] if ok else [], "ok": ok})
            if not ok:
                problems.append(f"{o}: 분리 우회전 → {lg['slip']['to']} 연결 문제")
        outs = [x for x in net.getEdges() if x.getID() == f"x6_{o}_out0"]
        if outs and outs[0].getLaneNumber() != n_out(spec, o):
            problems.append(f"{o} 진출: {outs[0].getLaneNumber()}차로 ≠ 자료 {n_out(spec, o)}")
    return {"rows": rows, "problems": problems, "lanes_checked": len(rows), "ok": sum(r["ok"] for r in rows)}


def print_check(rep):
    print(f"[x6 검사] 차로 {rep['lanes_checked']}개 중 자료와 일치 {rep['ok']} | 문제 {len(rep['problems'])}")
    for p in rep["problems"]:
        print("   ", p)


def concurrent_phases(plan, greens, lane_of, n, yellow=3.0, all_red=2.0):
    """동시 녹색 현시표(spec['signal_plan_concurrent']) → SUMO 현시 [(초, 상태)].
    lane_of[i] = (진입로, 차로 id) (정밀 모델 링크만), 그 밖의 링크(교차로 내부)는 항상 녹색.
    다음 현시에도 녹색인 링크는 황색·전적색 동안 녹색을 이어 간다."""
    sets = [{i for i, (lg, lid) in lane_of.items() if lid in ph["lanes"].get(lg, [])} for ph in plan["phases"]]
    inner = set(range(n)) - set(lane_of)
    out = []
    for k, ph in enumerate(plan["phases"]):
        cur, nxt = sets[k], sets[(k + 1) % len(sets)]
        g = float(greens[ph["name"]])
        for dur, end in ((g, "G"), (yellow, "y"), (all_red, "r")):
            st = "".join("G" if (i in inner or i in (cur & nxt) or (end == "G" and i in cur)) else (end if i in cur else "r")
                         for i in range(n))
            if dur > 0:
                out.append((dur, st))
    return out


def green_conflicts(net_path, tls_id, phases):
    """동시에 녹색(G/g)인 링크 중 교차로 foes(엇갈림·합류)가 있는 쌍 → [(현시 번호, i, j)]"""
    import xml.etree.ElementTree as ET
    root = ET.parse(net_path).getroot()
    J = next(j for j in root.iter("junction") if j.get("id") == tls_id)
    foes = {int(r.get("index")): r.get("foes")[::-1] for r in J.iter("request")}
    # 같은 진입 도로에서 같은 진출 차로로 합류하는 링크끼리는 빼고 본다 (진입로 안의 합류는 SUMO가 양보 규칙으로 처리.
    # 예: 신리 3·4차로가 둘 다 교대로 → 교대 진출 일반 2차로로 합류)
    same = {}
    for c in root.iter("connection"):
        if c.get("tl") == tls_id and c.get("linkIndex") is not None:
            same[int(c.get("linkIndex"))] = (c.get("from"), c.get("to"), c.get("toLane"))
    bad = []
    for k, (_, st) in enumerate(phases):
        on = [i for i, c in enumerate(st) if c in "Gg" and i in foes]
        for a in on:
            for b in on:
                if a < b and ((b < len(foes[a]) and foes[a][b] == "1") or (a < len(foes[b]) and foes[b][a] == "1")):
                    if a in same and b in same and same[a] == same[b]:
                        continue
                    bad.append((k, a, b))
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        print_check(check_x6(sys.argv[2], *(sys.argv[3:4])))
    else:
        r = apply_x6(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
        print(json.dumps({k: v for k, v in r.items() if k != "legs"}, ensure_ascii=False))
        print_check(check_x6(sys.argv[2], *(sys.argv[3:4])))
