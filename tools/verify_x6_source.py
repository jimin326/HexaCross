"""연산교차로: 차로 조사 페이지 원본 데이터(env/maps/yeonsan_x6_source.json) ↔ SUMO 도로망 독립 대조.

빌드 코드(tools/build_x6.py)·옮겨 적은 설정(yeonsan_x6.json)을 쓰지 않고, 도로망 기하(방위)로 진입로/진출로를 찾아
페이지의 차로표·좌회전 금지·분리 우회전·진출 차로 수·신호 설명·'참고'에 적힌 거리를 하나씩 확인한다.

  python tools/verify_x6_source.py NET [TLS_ADDITIONAL]
    NET            : 도로망 (예: env/maps/yeonsan_area.net.xml, env/sumo/highway.net.xml)
    TLS_ADDITIONAL : 시뮬레이션이 실제로 쓰는 신호 프로그램 파일 (학습 환경 env/sumo/highway.tls.xml)
페이지의 방면 id: N NE E SSE SSW WNW (= 이 저장소의 N NE E S SW W)"""
import json, math, sys, collections
import sumolib

import os
D = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env/maps/yeonsan_x6_source.json"), encoding="utf-8"))
LEGS, A, ORDER, EXIT = D["LEGS"], D["A"], D["ORDER"], D["exit"]
net = sumolib.net.readNet(sys.argv[1], withPrograms=True, withConnections=True)
C = net.getNode("414663644")
cx, cy = C.getCoord()
P, F = [], []          # 문제, 확인된 사실


def ok(cond, msg_ok, msg_bad):
    (F if cond else P).append(msg_ok if cond else msg_bad)


def brg(x, y):
    return math.degrees(math.atan2(x - cx, y - cy)) % 360


def angd(a, b):
    return abs((a - b + 180) % 360 - 180)


def leg_of_point(x, y):
    b = brg(x, y)
    o = min(ORDER, key=lambda o: angd(LEGS[o]["b"], b))
    return o, angd(LEGS[o]["b"], b)


def is_bus(l):
    return l.allows("bus") and not l.allows("passenger")


# ---- 진출로: 중심에서 나가는 엣지 → 하류 150 m 체인
out_leg, out_chain = {}, {}
for e in C.getOutgoing():
    sh = e.getShape()
    pt = sh[-1] if e.getLength() < 60 else e.getShape()[min(len(sh) - 1, 1)]
    o, err = leg_of_point(*sh[-1])
    out_leg[e.getID()] = o
    chain, cur, dist = [e], e, e.getLength()
    while dist < 200:
        nx = [c for c in cur.getOutgoing() if True]
        nx = [x for x in cur.getOutgoing() if out_leg.get(x.getID()) is None and angd(brg(*x.getShape()[-1]), LEGS[o]["b"]) < 30]
        if not nx:
            break
        cur = max(nx, key=lambda x: x.getLaneNumber())
        chain.append(cur); dist += cur.getLength()
    out_chain[o] = chain
    for x in chain:
        out_leg.setdefault(x.getID(), o)
ok(sorted(out_chain) == sorted(ORDER), f"진출로 6개 방면 모두 있음", f"진출로 방면 {sorted(out_chain)}")
for o in ORDER:
    e = next(x for x in C.getOutgoing() if out_leg[x.getID()] == o)
    want = LEGS[o]["out"] + (1 if LEGS[o].get("brt") else 0)
    nb = sum(is_bus(l) for l in e.getLanes())
    b = brg(*e.getShape()[-1])
    ok(e.getLaneNumber() == want and nb == (1 if LEGS[o].get("brt") else 0),
       f"{o} 진출 {e.getLaneNumber()}차로(버스전용 {nb}) = 자료 out {LEGS[o]['out']}+BRT {int(bool(LEGS[o].get('brt')))}",
       f"{o} 진출 {e.getLaneNumber()}차로(버스전용 {nb}) ≠ 자료 {want}")
    if nb:
        ok(is_bus(e.getLanes()[-1]), f"{o} 진출 BRT = 1차로(중앙선 쪽)", f"{o} 진출 BRT 위치 틀림")

# ---- 진입로
inc, PROF, SLIP = {}, {}, {}
for e in C.getIncoming():
    o, err = leg_of_point(*e.getShape()[0])
    inc[o] = e
def cross_r(edges, R):
    """엣지들의 차로 모양이 중심 거리 R과 만나는 점들의 평균"""
    pts = []
    for e in edges:
        for l in e.getLanes():
            sh = l.getShape()
            for p, q in zip(sh, sh[1:]):
                rp, rq = math.dist(p, (cx, cy)), math.dist(q, (cx, cy))
                if (rp - R) * (rq - R) <= 0 and rp != rq:
                    t = (R - rp) / (rq - rp)
                    pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return pts


for o in ORDER:
    # 방면 방위: 중심에서 60 m 거리의 진입·진출 차로들(분리 우회전 제외)의 가운데 점
    up, cur = [inc[o]], inc[o]
    for _ in range(4):
        pr = [x for x in cur.getIncoming() if x.getFromNode() != C and angd(brg(*x.getShape()[0]), LEGS[o]["b"]) < 35]
        if not pr:
            break
        cur = max(pr, key=lambda x: x.getLaneNumber()); up.append(cur)
    pts = cross_r(up + out_chain[o], 60)
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    b = brg(mx, my)
    ok(angd(b, LEGS[o]["b"]) <= 10, f"{o} 방위 {b:.0f}° (자료 {LEGS[o]['b']}°, 10° 안)", f"{o} 방위 {b:.0f}° ≠ 자료 {LEGS[o]['b']}°")
ok(sorted(inc) == sorted(ORDER), "진입로 6개", f"진입로 {sorted(inc)}")

tls = {t.getID(): t for t in net.getTrafficLights()}
tl = next(t for t in tls.values() if any(c[0].getEdge().getToNode() == C for c in t.getConnections()))
prog = list(tl.getPrograms().values())[0]
phases = [(float(p.duration), p.state) for p in prog.getPhases()]
if len(sys.argv) > 2:      # 시뮬레이션이 실제로 쓰는 신호 프로그램 (추가 파일, 예: 학습 환경 highway.tls.xml)
    import xml.etree.ElementTree as _ET
    _tl = next(t for t in _ET.parse(sys.argv[2]).getroot().iter("tlLogic") if t.get("id") == tl.getID())
    phases = [(float(p.get("duration")), p.get("state")) for p in _tl.iter("phase")]
    F.append(f"신호 프로그램: {sys.argv[2]} programID={_tl.get('programID')}")
link = {}  # (fromLane, toLane) -> idx
for fl, tl_, idx in tl.getConnections():
    link[(fl.getID(), tl_.getID())] = idx

lane_dest = {}
for o in ORDER:
    e = inc[o]
    ms = A[o]["lanes"]
    main = [l for l in ms if l["kind"] != "slip"]
    n = e.getLaneNumber()
    ok(n == len(main), f"{o} 정지선 {n}차로 = 자료 {len(main)}", f"{o} 정지선 {n}차로 ≠ 자료 {len(main)}")
    for i, l in enumerate(main):
        lid = int(l["id"])
        ln = e.getLane(n - lid)
        got = collections.defaultdict(list)
        for c in ln.getOutgoing():
            d = out_leg.get(c.getTo().getID())
            got[d].append((c.getToLane().getIndex(), c.getDirection()))
        gotset = sorted(got)
        want = sorted(l["to"])
        ok(gotset == want, f"{o} {lid}차로 → {gotset} = 자료 {want}", f"{o} {lid}차로 → 도로망 {gotset} ≠ 자료 {want}")
        ok(is_bus(ln) == (l["kind"] == "brt"), f"{o} {lid}차로 버스전용={is_bus(ln)}", f"{o} {lid}차로 버스전용={is_bus(ln)} ≠ 자료 {l['kind']}")
        for d in l["to"]:
            if d not in got:
                continue
            nd = net.getEdge(next(c.getTo().getID() for c in ln.getOutgoing() if out_leg.get(c.getTo().getID()) == d)).getLaneNumber()
            gotj = sorted({nd - ti for ti, _ in got[d]})
            wj = EXIT[o][d].get(l["id"])
            ok(gotj == [wj], f"{o} {lid}차로 → {d} 진출 {gotj}차로 = 자료 그림 {wj}", f"{o} {lid}차로 → {d} 진출 {gotj}차로 ≠ 자료 그림 {wj}")
            lane_dest[(o, lid, d)] = [(c, link.get((c.getFromLane().getID(), c.getToLane().getID()))) for c in ln.getOutgoing() if out_leg.get(c.getTo().getID()) == d]
    # 금지 방향: 진입로 어느 차로(분리 포함)도 금지 방면으로 연결 없음
    bans = A[o]["ban"]
    alld = {out_leg.get(c.getTo().getID()) for l in e.getLanes() for c in l.getOutgoing()}
    ok(not (alld & set(bans)), f"{o} 좌회전 금지 {bans}: 연결 없음", f"{o} 금지 방면 연결 있음 {alld & set(bans)}")
    ok(all(c.getDirection() not in ("t", "T") for l in e.getLanes() for c in l.getOutgoing()), f"{o} 교차로 유턴 연결 없음", f"{o} 유턴 연결 있음")
    # 상류 체인: 차로 수 변화, 분리 우회전
    chain, cur, dist, prof, slips = [], e, 0.0, [], []
    while dist < 160:
        sr = math.dist(e.getShape()[-1], (cx, cy))
        sh = cur.getShape()
        prof.append((round(math.dist(sh[-1], (cx, cy)) - sr), round(math.dist(sh[0], (cx, cy)) - sr), cur.getLaneNumber(), sum(is_bus(l) for l in cur.getLanes()), cur.getID()))
        for c in [c for l in cur.getLanes() for c in l.getOutgoing()]:
            t = c.getTo()
            if t.getToNode() != C and t != e and t not in chain and c.getFrom() != e:
                # 체인 아래쪽(하류)이 아닌 갈래 = 분리 우회전 후보
                if not any(x is t for x in chain):
                    slips.append((round(dist), c.getFromLane().getIndex(), cur.getLaneNumber(), t))
        chain.append(cur)
        dist += cur.getLength()
        pr = [x for x in cur.getIncoming() if angd(brg(*x.getShape()[0]), LEGS[o]["b"]) < 35 and x.getFromNode() != C]
        if not pr:
            break
        cur = max(pr, key=lambda x: x.getLaneNumber())
    F.append(f"{o} 진입 차로 수(정지선에서 거리 m: 전체/버스): " + ", ".join(f"{a}-{b}m {n}/{bb}" for a, b, n, bb, _ in prof))
    PROF[o] = prof
    want_slip = [l for l in ms if l["kind"] == "slip"][0]["to"][0]
    found = []
    for d0, li, nl, t in [s for s in slips if s[3] not in chain]:
        # 갈래를 따라가서 (중심을 거치지 않고) 다른 방면 진출로에 닿으면 분리 우회전
        stack, best = [(t, [t], t.getLength())], None
        while stack and best is None:
            cur, path, L = stack.pop()
            if out_leg.get(cur.getID()) not in (None, o) and cur not in C.getOutgoing():
                best = path
                break
            if L > 150 or math.dist(cur.getShape()[-1], (cx, cy)) > 105:
                continue
            for x in {c.getTo() for l in cur.getLanes() for c in l.getOutgoing()}:
                if x.getFromNode() != C and x not in path and x.getToNode() != C:
                    stack.append((x, path + [x], L + x.getLength()))
        if best:
            found.append((d0, li, nl, best))
    ok(len(found) == 1, f"{o} 분리 우회전 1개", f"{o} 분리 우회전 {len(found)}개")
    for d0, li, nl, path in found:
        d = out_leg.get(path[-1].getID())
        merge_node = path[-2].getToNode()
        SLIP[o] = (round(math.dist(path[0].getShape()[0], (cx, cy)) - math.dist(e.getShape()[-1], (cx, cy))), 0)
        sig = any(merge_node.getID() in [c[0].getEdge().getToNode().getID() for c in x.getConnections()] for x in tls.values())
        mlane = [c.getToLane().getIndex() for c in path[-2].getLanes()[0].getOutgoing()] if len(path) > 1 else []
        d0 = SLIP[o][0]
        ok(d == want_slip and li == 0, f"{o} 분리 우회전: 정지선 {d0}m 앞 {nl}차로 중 맨 오른쪽에서 갈라져 → {d} (자료 {want_slip})",
           f"{o} 분리 우회전 → {d}, 갈라지는 차로 index {li} (자료 {want_slip}, 맨 오른쪽)")
        ok(not sig and merge_node.getType() != "traffic_light", f"{o} 분리 우회전 합류 신호 없음 ({merge_node.getType()})", f"{o} 분리 우회전 합류에 신호")
        ok(mlane == [0], f"{o} 분리 우회전 → {d} 맨 바깥 차로로 합류", f"{o} 분리 우회전 합류 차로 {mlane}")
        # 합류: 분리 우회전이 양보하는지 (분리 쪽 연결이 minor)
        cs = [c for c in path[-2].getLanes()[0].getOutgoing()]
        st = [c.getState() for c in cs]
        ok(all(s in ("m", "=", "w", "s", "Z", "z") for s in st) and st, f"{o} 분리 우회전 합류 양보(state {st})", f"{o} 분리 우회전 합류 state {st} — 양보 아님")

# ---- 신호 (자료 문장과 대조)
def greens(o, lid, d):
    return [k for k, (_, s) in enumerate(phases) if any(i is not None and s[i] in "Gg" for _, i in lane_dest.get((o, lid, d), []))]
F.append(f"신호 {tl.getID()} 현시 {len(phases)}개, 주기 {sum(x for x, _ in phases):g}s: " + " | ".join(f"{d:g}s" for d, _ in phases))
for o in ORDER:
    ks = {(int(l["id"]), d): tuple(greens(o, int(l["id"]), d)) for l in A[o]["lanes"] if l["kind"] != "slip" for d in l["to"]}
    F.append(f"{o} 녹색 현시: " + ", ".join(f"{k[0]}차로→{k[1]}:{v}" for k, v in ks.items()))
    ok(all(v for v in ks.values()), f"{o} 모든 차로-방면에 녹색 있음", f"{o} 녹색 없는 차로 {[k for k, v in ks.items() if not v]}")
    other = [k for k, (_, s) in enumerate(phases)]
# E·SSE·WNW: 녹색+좌회전 화살표 함께 = 접근로 모든 방향 한 번에
def green_starts(o, lid, d):
    """녹색이 켜지는 현시 번호들 (앞 현시에서 녹색이 아니다가 녹색이 되는 곳)"""
    gs = set(greens(o, lid, d)); n = len(phases)
    return tuple(sorted(k for k in gs if (k - 1) % n not in gs))
for o in ("E", "SSE", "WNW"):
    sets = {green_starts(o, int(l["id"]), d) for l in A[o]["lanes"] if l["kind"] != "slip" for d in l["to"]}
    ok(len(sets) == 1, f"{o}: 모든 방향 녹색이 함께 켜짐 (자료: 녹색과 좌회전 화살표가 함께 켜짐)", f"{o}: 방향마다 녹색 켜지는 때가 다름 {sets}")
# N: 수영 방면 차로(2,3)와 시청 방면 차로(4,5) 신호가 따로 켜짐
nS = {tuple(greens("N", l, "SSE")) for l in (2, 3)}
nW = {tuple(greens("N", l, "SSW")) for l in (4, 5)}
nB = {tuple(greens("N", 1, "SSW"))}
F.append(f"N 수영 방면 {nS} / 시청 방면 {nW} / BRT {nB}")
ok(nS != nW, "N: 수영 방면·시청 방면 차로 신호 따로 켜짐", "N: 수영 방면·시청 방면 차로가 같은 시점에 켜지고 꺼짐 (자료: 따로 켜짐)")
# SSW: 교대 방면 좌회전 화살표가 녹색보다 먼저 꺼짐
def last_green_end(o, lid, d):
    gs = greens(o, lid, d)
    t, ends = 0, []
    for k, (du, s) in enumerate(phases):
        t += du
        if k in gs:
            ends.append(t)
    return max(ends) if ends else None
eN = [last_green_end("SSW", l, "N") for l in (2, 3)]
eG = [last_green_end("SSW", l, d) for l, d in ((4, "NE"), (5, "NE"), (5, "E"), (6, "E"))]
F.append(f"SSW 녹색 끝(주기 안 초): 교대 방면 {eN}, 안락·연산터널 {eG}")
ok(max(eN) < min(eG), "SSW: 교대 방면 좌회전 화살표가 녹색보다 먼저 꺼짐", "SSW: 교대 방면 화살표가 녹색과 같이/늦게 꺼짐 (자료: 먼저 꺼짐)")
# SSW 2·3차로 정지선 약 3 m 뒤
e = inc["SSW"]
so = {}
try:
    import xml.etree.ElementTree as ET
    root = ET.parse(sys.argv[1]).getroot()
    for ed in root.iter("edge"):
        if ed.get("id") == e.getID():
            for la in ed.iter("lane"):
                s = la.find("stopOffset")
                so[la.get("id")] = float(s.get("value")) if s is not None else 0.0
            s = ed.find("stopOffset")
            if s is not None:
                so["edge"] = float(s.get("value"))
except Exception as ex:
    so["err"] = str(ex)
n = e.getLaneNumber()
o23 = [so.get(e.getLane(n - k).getID(), 0) for k in (2, 3)]
o46 = [so.get(e.getLane(n - k).getID(), 0) for k in (4, 5, 6)]
ok(min(o23) - max(o46) >= 2.5, f"SSW 2·3차로 정지선 {o23} m 뒤 / 4~6차로 {o46}", f"SSW 2·3차로 정지선 뒤로 물림 없음 (stopOffset 2·3={o23}, 4~6={o46}; 자료: 약 3 m 뒤)")

# ---- 자료 '참고'에 적힌 거리 (정지선 앞 m, 일반 차로 수 = 버스전용 뺀 것)
def gen_at(o, dist):
    for a, b, n, nb, _ in PROF[o]:
        if a <= dist < b:
            return n - nb
    return "교차로 안"
stop_r = {o: math.dist(inc[o].getShape()[-1], (cx, cy)) for o in ORDER}
F.append("실제 정지선 중심 거리 m: " + ", ".join(f"{o} {stop_r[o]:.1f}" for o in ORDER))
checks = [
    ("N", [(5, 4), (55, 4)], "2차로는 정지선 60~85 m 앞에서 새로 생김 → 60 m 안은 일반 4차로"),
    ("NE", [(5, 4), (30, 4), (49, 3), (80 - stop_r["NE"] + 8, 2)], "정지선 앞 40 m 사이에 4차로, 그 바깥은 넓어지는 3차로, 중심에서 80 m 바깥은 2차로"),
    ("E", [(5, 4), (24, 3), (34, 3), (50, 2)], "정지선 21 m 앞은 3차로, 동쪽 횡단보도(약 40 m) 너머는 2차로"),
    ("SSE", [(5, 4), (38, 4), (48, 3)], "정지선 앞 40~46 m 횡단보도 뒤는 3차로"),
]
for o, pts, txt in checks:
    got = [(d, gen_at(o, d)) for d, _ in pts]
    good = all(g == w for (d, g), (_, w) in zip(got, pts))
    ok(good, f"{o} 차로 수 변화 = 자료 ({txt}): " + ", ".join(f"{d:.0f}m {g}" for d, g in got),
       f"{o} 차로 수 변화 다름 ({txt}): 도로망 " + ", ".join(f"{d:.0f}m {g}(자료 {w})" for (d, g), (_, w) in zip(got, pts)))
if "WNW" in SLIP:
    ok(15 <= SLIP["WNW"][0] <= 25, f"WNW 분리 우회전 갈라지는 곳 정지선 {SLIP['WNW'][0]} m 앞 (자료 약 20 m)",
       f"WNW 분리 우회전 갈라지는 곳 정지선 {SLIP['WNW'][0]} m 앞 ≠ 자료 약 20 m")
F.append("분리 우회전 갈라지는 곳(정지선 앞 m, 자료에 없는 것은 가정): " + ", ".join(f"{o} {v[0]}" for o, v in SLIP.items()))

print("== 문제", len(P))
EXIT_CODE = 1 if P else 0
for x in P:
    print("  ✗", x)
print("== 확인", len(F))
for x in F:
    print("  ✓", x)

sys.exit(EXIT_CODE)
