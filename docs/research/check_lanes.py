"""연산교차로 정지선 차로·연결 검증 스크립트 (레포 루트에서 실행).

    python docs/research/check_lanes.py           # 이미 만들어진 env/sumo/highway.net.xml 검사
    python docs/research/check_lanes.py --build   # 도로를 새로 만든 뒤 검사

출력 (마크다운 표):
    A. 6개 진입부 정지선 edge의 일반 차로 수와 차로별 목적지 (조사 md와 비교)
    B. 금지 회전 연결 수 (전부 0이어야 함)
    C. 우회전 연결로 6개 (출발·도착 방향, 가짜 연결)
    D. 배경 교통 경로 점검 (교차로를 2번 이상 지나는 경로, 가짜 연결 사용 경로)
    E. 진출 차로 수 (교차로를 나온 직후 edge, 조사 md와 비교)
    F. BRT 버스 전용 차로 (중앙대로 SW·N, 정지선 앞·상류·진출)

기대값은 docs/research/연산교차로_도로조사.md의 각 진입부 "정지선 앞" 표에서
옮겨 적은 것이다. 도로 보정 파일(env/osm/yeonsan_fix.*.xml)과 따로 적어 두어서
보정 파일을 잘못 적으면 여기서 걸리게 했다. BRT(버스 전용) 차로는 제외한다.
하나라도 다르면 종료 코드 1.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.getcwd())  # 레포 루트에서 실행하므로 env 패키지를 찾게 함
import sumolib
from env.road_config import ROAD, EGO, TRAFFIC

NET_DIR = os.path.join("env", "sumo")

# 정지선 앞 일반 차로별 목적지. SUMO 차로 번호 순서 = 인도 쪽(0) → 중앙 쪽.
# 방면 → leg: 교대·동래=N, 시청·서면=SW, 거제·법원·사직=W, 수영·광안리=S,
#             안락·반송로=NE, 연산터널·고분로=E
EXPECTED = {
    "SW": ["E", "NE E", "NE", "N", "N"],   # 연산터널 / 안락·연산터널 / 안락 / 교대 / 교대
    "N":  ["SW", "SW", "S", "S"],          # 시청 / 시청 / 수영 / 수영
    "W":  ["S", "E S", "NE E", "N NE"],    # 수영 / 연산터널·수영 / 안락·연산터널 / 교대·안락
    "S":  ["NE N", "N", "W N", "SW"],      # 안락·동래 / 교대 / 거제·교대 / 시청
    "E":  ["N", "W N", "SW W", "S SW"],    # 교대 / 거제·교대 / 시청·거제 / 수영·시청
    "NE": ["W", "W", "SW", "SW"],          # 거제 / 거제 / 시청 / 시청
}
# 진출 일반 차로 수 (교차로 → 바깥, 조사 md "진출부"). SW 진출은 조사 없음
EXIT = {"N": 2, "W": 3, "S": 3, "E": 2, "NE": 3}
# BRT 버스 전용 차로 (조사 md + HANDOFF 5번). 정지선 앞은 차로별 목적지
# (차로 번호 순서 = 인도 쪽 → 중앙), 나머지는 차로 수. SW 진출은 가정.
BRT_STOP = {"SW": ["NE", "N"], "N": ["SW"]}      # SW: BRT2 안락, BRT1 교대 / N: 시청
BRT_UP = {"SW": 1, "N": 1}                       # 정지선 200m 이상 상류
BRT_EXIT = {"SW": 1, "N": 1}
# 우회전 연결로 (교통섬으로 분리, 신호 없음): 출발 leg → 도착 leg
SLIP = {"SW": "S", "N": "W", "W": "SW", "S": "E", "E": "NE", "NE": "N"}
# 금지 회전 (표준노드링크 TURNINFO + 로드뷰). 반송로 유턴은 조사 md 목록,
# 동래 유턴은 HANDOFF 목록에 있어서 둘 다 검사한다.
BANNED = [("SW", "W", "좌회전"), ("N", "NE", "좌회전"), ("N", "E", "좌회전"),
          ("NE", "S", "좌회전"), ("NE", "E", "좌회전"),
          ("N", "N", "유턴"), ("NE", "NE", "유턴")]
ORDER = ["SW", "N", "W", "S", "E", "NE"]


def bearing(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360.0


def adiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def main():
    if "--build" in sys.argv:
        from env import road_builder
        road_builder.build(ROAD, EGO, TRAFFIC, NET_DIR)

    net = sumolib.net.readNet(os.path.join(NET_DIR, "highway.net.xml"))
    center = net.getNode(str(ROAD["osm"]["center_node"]))
    cx, cy = center.getCoord()
    legs = ROAD["legs"]

    def leg_at(x, y):
        b = bearing(x - cx, y - cy)
        return min(legs, key=lambda lg: adiff(b, lg["bearing"]))["id"]

    def dist(p):
        return math.hypot(p[0] - cx, p[1] - cy)

    def out_leg(edge):
        """교차점에서 나가는 edge → 도착 leg (같은 이름 도로를 60m 이상 따라가서 판정)."""
        e, total = edge, edge.getLength()
        while total < 60.0:
            nxt = [n for n in e.getOutgoing() if n.getName() == e.getName()] or list(e.getOutgoing())
            if not nxt:
                break
            e = nxt[0]
            total += e.getLength()
        return leg_at(*e.getShape()[-1])

    def inbound(e):
        return dist(e.getShape()[-1]) < dist(e.getShape()[0])

    def is_slip(e):
        """연산교차로 우회전 연결로 (교차점 150m 안의 *_link)."""
        return ("_link" in (e.getType() or "")
                and min(dist(e.getShape()[0]), dist(e.getShape()[-1])) <= 150.0)

    ok = True
    stop_edge = {}
    for e in center.getIncoming():
        if e.allows("passenger"):
            stop_edge[leg_at(*e.getShape()[0])] = e
    exit_edge = {}
    for e in center.getOutgoing():
        if e.allows("passenger"):
            exit_edge[out_leg(e)] = e

    # ── A. 정지선 차로 ──
    print("### A. 정지선 앞 일반 차로 (차로 0 = 인도 쪽)\n")
    print("| 진입부 | 정지선 edge | 차로 수 조사/결과 | 차로별 목적지: 조사 | 결과 | 일치 |")
    print("|---|---|---|---|---|---|")
    for leg in ORDER:
        e = stop_edge.get(leg)
        exp = [set(s.split()) for s in EXPECTED[leg]]
        if e is None:
            print(f"| {leg} | (없음) | {len(exp)}/- | | | ❌ |")
            ok = False
            continue
        got = []
        for ln in e.getLanes():
            if not ln.allows("passenger"):
                continue   # 버스 전용 차로 제외
            got.append({out_leg(c.getTo()) for c in ln.getOutgoing()})
        match = got == exp
        ok &= match
        fmt = lambda lst: " / ".join("·".join(sorted(s)) or "∅" for s in lst)
        print(f"| {leg} | `{e.getID()}` | {len(exp)}/{len(got)} | {fmt(exp)} | {fmt(got)} | "
              f"{'✅' if match else '❌'} |")

    # ── B. 금지 회전 ──
    print("\n### B. 금지 회전 연결 수 (0이어야 함)\n")
    print("| 이동 | 종류 | from → to | 연결 수 |")
    print("|---|---|---|---|")
    for a, b, kind in BANNED:
        fe, te = stop_edge.get(a), exit_edge.get(b)
        n = 0
        if fe is not None and te is not None:
            n = sum(1 for ln in fe.getLanes() for c in ln.getOutgoing() if c.getTo() == te)
        ok &= n == 0
        ids = f"`{fe.getID() if fe else '-'}` → `{te.getID() if te else '-'}`"
        print(f"| {a}→{b} | {kind} | {ids} | {n}{' ❌' if n else ''} |")

    # ── C. 우회전 연결로 ──
    print("\n### C. 우회전 연결로 (조사: SW→S, N→W, W→SW, S→E, E→NE, NE→N)\n")
    print("| 연결로 edge | 출발→도착 | 조사와 같음 | 가짜 연결 (연결로→연결로/진입로, 진출로→연결로) |")
    print("|---|---|---|---|")
    found = set()
    for e in net.getEdges():
        if not is_slip(e):
            continue
        p0, p1 = e.getShape()[0], e.getShape()[-1]
        # 출발·도착 방향은 연결로 끝점 대신 이어진 진입로(시작점)·진출로(멀리 따라간
        # 끝점)로 판정한다. 진입로 폭이 바뀌면 연결로 끝점이 방위 경계를 넘을 수 있다.
        feeders = [p for p in e.getIncoming() if inbound(p) and not is_slip(p)]
        nexts = [n for n in e.getOutgoing() if not inbound(n) and not is_slip(n)]
        a = leg_at(*feeders[0].getShape()[0]) if feeders else leg_at(*p0)
        b = out_leg(nexts[0]) if nexts else leg_at(*p1)
        found.add((a, b))
        fake = [f"→{n.getID()}" for n in e.getOutgoing()
                if "_link" in (n.getType() or "") or inbound(n)]
        fake += [f"{p.getID()}→" for p in e.getIncoming()
                 if not inbound(p) and "_link" not in (p.getType() or "")]
        print(f"| `{e.getID()}` | {a}→{b} | {'✅' if SLIP.get(a) == b else '❌'} | "
              f"{len(fake)}개 {' '.join(f'`{x}`' for x in fake)} |")
    missing = [f"{a}→{b}" for a, b in SLIP.items() if (a, b) not in found]
    if missing:
        ok = False
        print(f"\n없는 연결로: {', '.join(missing)} ❌")

    # ── D. 배경 교통 경로 ──
    rou = open(os.path.join(NET_DIR, "highway.rou.xml"), encoding="utf-8").read()
    routes = dict(re.findall(r'<route id="(bg_\d+)" edges="([^"]+)"', rou))
    rates = {r: float(v) for r, v in re.findall(r'route="(bg_\d+)"[^>]*vehsPerHour="([\d.]+)"', rou)}
    cid = center.getID()
    twice, via_fake = [], []
    for rid, edges in routes.items():
        es = [net.getEdge(x) for x in edges.split()]
        if sum(1 for x in es if x.getToNode().getID() == cid) >= 2:
            twice.append(rid)
        for x, y in zip(es, es[1:]):
            xl, yl = is_slip(x), is_slip(y)
            if (xl and (yl or inbound(y))) or (yl and not xl and not inbound(x)):
                via_fake.append(rid)
                break
    total = sum(rates.values())
    print("\n### D. 배경 교통 경로\n")
    print("| 항목 | 경로 수 | 교통량 (대/h) |")
    print("|---|---|---|")
    print(f"| 전체 | {len(routes)} | {total:.0f} |")
    print(f"| 교차로를 2번 이상 지나는 경로 | {len(twice)} | {sum(rates.get(r, 0) for r in twice):.0f} |")
    print(f"| 연결로 가짜 연결을 지나는 경로 | {len(via_fake)} | {sum(rates.get(r, 0) for r in via_fake):.0f} |")

    # ── E. 진출 차로 ──
    print("\n### E. 진출 차로 (교차로를 나온 직후 edge, 5m 미만 조각은 건너뜀)\n")
    print("| 진출 | edge | 차로 수 조사/결과 | 들어오는 연결 수 (차로 0 → 끝) | 일치 |")
    print("|---|---|---|---|---|")
    for leg in ORDER:
        e = exit_edge.get(leg)
        if e is None:
            print(f"| {leg} | (없음) | | | ❌ |")
            ok = False
            continue
        # 교차로 영역에 거의 다 들어간 짧은 조각(5m 미만)은 건너뛰고 다음 조각을 본다
        while e.getLength() < 5.0:
            nxt = [o for o in e.getOutgoing() if o.getName() == e.getName()]
            if not nxt:
                break
            e = nxt[0]
        n = sum(1 for ln in e.getLanes() if ln.allows("passenger"))
        feed = [sum(1 for f in e.getIncoming() for fl in f.getLanes()
                    for c in fl.getOutgoing() if c.getToLane() == ln)
                for ln in e.getLanes() if ln.allows("passenger")]
        exp = EXIT.get(leg)
        match = (exp is None or n == exp) and all(feed)
        ok &= match
        print(f"| {leg} | `{e.getID()}` | {exp if exp else '조사 없음'}/{n} | "
              f"{' / '.join(map(str, feed))} | {'✅' if match else '❌'} |")

    # ── F. BRT ──
    def bus_lanes(e):
        return [ln for ln in e.getLanes() if ln.allows("bus") and not ln.allows("passenger")]

    print("\n### F. BRT 버스 전용 차로 (일반 차량 진입 불가)\n")
    print("| 위치 | edge | 조사 | 결과 | 일치 |")
    print("|---|---|---|---|---|")
    for leg in ORDER:
        e = stop_edge.get(leg)
        exp = BRT_STOP.get(leg, [])
        got = [sorted({out_leg(c.getTo()) for c in ln.getOutgoing()}) for ln in bus_lanes(e)] if e else []
        got_s = " / ".join("·".join(g) or "∅" for g in got) or "없음"
        match = [g for g in got] == [[x] for x in exp]
        ok &= match
        if exp or got:
            print(f"| {leg} 정지선 앞 (목적지) | `{e.getID()}` | {' / '.join(exp) or '없음'} | "
                  f"{got_s} | {'✅' if match else '❌'} |")
        # 상류: 같은 이름 도로를 거슬러 200m 이상
        cur, tot = e, 0.0
        while cur is not None and tot < 200.0:
            tot += cur.getLength()
            prv = [p for p in cur.getIncoming() if p.getName() == cur.getName()
                   and p.getID().lstrip("-") != cur.getID().lstrip("-")]
            cur = prv[0] if prv else None
        n_up = len(bus_lanes(cur)) if cur is not None else 0
        exp_up = BRT_UP.get(leg, 0)
        ok &= n_up == exp_up
        if exp_up or n_up:
            print(f"| {leg} 진입 상류 (차로 수) | `{cur.getID() if cur else '-'}` | {exp_up} | {n_up} | "
                  f"{'✅' if n_up == exp_up else '❌'} |")
        x = exit_edge.get(leg)
        n_ex = len(bus_lanes(x)) if x is not None else 0
        exp_ex = BRT_EXIT.get(leg, 0)
        ok &= n_ex == exp_ex
        if exp_ex or n_ex:
            print(f"| {leg} 진출 (차로 수) | `{x.getID()}` | {exp_ex}{' (가정)' if leg == 'SW' else ''} | "
                  f"{n_ex} | {'✅' if n_ex == exp_ex else '❌'} |")
    # 일반 차로 → 버스 차로로 들어가는 연결이 없어야 함 (교차로 안)
    leak = [f"{fl.getID()}→{c.getToLane().getID()}" for f in center.getIncoming() for fl in f.getLanes()
            if fl.allows("passenger") for c in fl.getOutgoing()
            if not c.getToLane().allows("passenger")]
    ok &= not leak
    print(f"| 교차로에서 일반 차로 → 버스 차로 연결 | | 0 | {len(leak)} | {'✅' if not leak else '❌ ' + ' '.join(leak)} |")

    print(f"\n결과: {'통과' if ok else '불일치 있음'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
