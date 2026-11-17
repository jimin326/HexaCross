"""버스전용차로 — 모든 도로에 같은 규칙으로 적용.

SUMO에서 버스전용차로는 그 차로의 통행 권한이 allow="bus" 인 차로다 (승용차는 들어갈 수도, 차로를 바꿔
들어갈 수도 없음). OSM에서 버스전용차로를 적는 방법이 여러 가지인데 netconvert(--osm.lane-access)가
제대로 읽는 것은 일부뿐이라, netconvert에 넘기기 전에 모두 한 가지 형식으로 바꾼다.

  OSM 태그                                   netconvert 그대로      여기서 바꾸는 형식
  bus:lanes / psv:lanes (+access:lanes)      O
  busway:right=lane (= 가로변 버스차로)        O
  busway:left=lane  (= 중앙/왼쪽 버스차로)      X 반대 방향 차로를 만듦  bus:lanes(:forward) 맨 왼쪽 designated
  busway:both=lane, busway=lane              일부                  양방향 맨 오른쪽 designated
  lanes:bus=N, lanes:psv=N                   X 무시                맨 오른쪽 N개 designated
  (차로 조사표) 버스전용_정방향/역방향           -                     아래 형식

바꾼 결과: bus:lanes[:forward|:backward] = "designated|yes|..." (OSM 순서 = 왼쪽 차로부터),
          access:lanes[...] = "no|yes|..." (designated 자리는 일반 차량 no).
`lanes`는 OSM 규칙대로 버스전용차로를 포함한 전체 차로 수다.

차로 조사표 형식 (버스전용_정방향 / 버스전용_역방향 칸):
  "오른쪽1" = 맨 오른쪽 1개 차로 (가로변 버스전용차로)
  "왼쪽1"   = 맨 왼쪽 1개 차로 (중앙 버스전용차로)
  "없음" 또는 빈칸 = 버스전용차로 없음 (빈칸은 OSM 값 유지)
"""
import math
import re

ONEWAY = ("yes", "1", "true")


def _lane_counts(t):
    """(정방향 차로 수, 역방향 차로 수) — OSM 태그에서 정해지는 만큼만, 모르면 None."""
    oneway = t.get("oneway") in ONEWAY or t.get("junction") in ("roundabout", "circular")
    if oneway:
        return (int(t["lanes"]) if t.get("lanes", "").isdigit() else None), 0
    f, b = t.get("lanes:forward", ""), t.get("lanes:backward", "")
    if f.isdigit() and b.isdigit():
        return int(f), int(b)
    n = t.get("lanes", "")
    if n.isdigit() and int(n) % 2 == 0:
        return int(n) // 2, int(n) // 2
    return None, None


def _pattern(n, side, k):
    """n개 차로 중 side(left/right) 쪽 k개가 버스전용 → (bus:lanes 값, access:lanes 값), 왼쪽 차로부터."""
    if not n or k <= 0 or k >= n:
        return None
    bus = ["designated" if (i < k if side == "left" else i >= n - k) else "yes" for i in range(n)]
    acc = ["no" if v == "designated" else "yes" for v in bus]
    return "|".join(bus), "|".join(acc)


def parse_survey(value):
    """조사표 값 → (side, k) / ("none", 0) / None(빈칸)"""
    v = (value or "").strip()
    if not v:
        return None
    if v in ("없음", "0", "none", "no"):
        return ("none", 0)
    m = re.fullmatch(r"(왼쪽|오른쪽|left|right)\s*(\d+)?", v)
    if not m:
        raise ValueError(f"버스전용 칸 형식 오류: '{v}' (예: 오른쪽1, 왼쪽1, 없음)")
    side = "left" if m.group(1) in ("왼쪽", "left") else "right"
    return side, int(m.group(2) or 1)


def normalize_bus_tags(tags, survey=None):
    """한 way의 태그 dict를 받아 바꿀 태그 dict를 돌려준다 (바꿀 것이 없으면 {}).
    survey = {"forward": (side, k) | None, "backward": ...} (차로 조사표, OSM보다 우선)."""
    t = dict(tags)
    oneway = t.get("oneway") in ONEWAY or t.get("junction") in ("roundabout", "circular")
    nf, nb = _lane_counts(t)
    want = {"forward": None, "backward": None}        # 방향별 (side, k)
    # 1) OSM의 여러 표기
    for key, side in (("busway:left", "left"), ("busway:right", "right"), ("busway:both", "both"), ("busway", "both")):
        v = t.get(key)
        if v not in ("lane",):
            continue
        if oneway:
            want["forward"] = ("left" if side == "left" else "right", 1)
        else:
            # 양방향 도로: OSM 그린 방향의 오른쪽 = 정방향 차로 맨 오른쪽, 왼쪽 = 역방향 차로 맨 오른쪽(우측통행)
            if side in ("right", "both"):
                want["forward"] = ("right", 1)
            if side in ("left", "both"):
                want["backward"] = ("right", 1)
    for key in ("lanes:bus", "lanes:psv"):
        for suf, d in (("", "forward"), (":forward", "forward"), (":backward", "backward")):
            v = t.get(key + suf, "")
            if v.isdigit() and int(v) > 0:
                want[d] = ("right", int(v))
    # 2) 조사표 (우선)
    for d in ("forward", "backward"):
        s = (survey or {}).get(d)
        if s is not None:
            want[d] = s
    new = {}
    for d, n in (("forward", nf), ("backward", nb)):
        w = want[d]
        if w is None:
            continue
        sfx = "" if oneway else f":{d}"
        if w[0] == "none":
            for k in (f"bus:lanes{sfx}", f"psv:lanes{sfx}", f"access:lanes{sfx}"):
                if k in t:
                    new[k] = "|".join(["yes"] * n) if n else ""
            continue
        p = _pattern(n, *w)
        if p is None:
            continue                                      # 차로 수를 모르거나 전부 버스차로 → 손대지 않음
        new[f"bus:lanes{sfx}"], new[f"access:lanes{sfx}"] = p
    # netconvert가 잘못 읽는 원래 표기는 지운다 (busway:left=lane → 반대 방향 차로가 생김)
    if new:
        for key in ("busway:left", "busway:right", "busway:both", "busway", "lanes:bus", "lanes:psv"):
            if key in t:
                new[key] = None
    return new


def bus_lane_summary(net):
    """도로망의 버스전용차로 목록과 검사 (빌드 검증용). 승용차가 실제로 들어가지 않는지는 주행 시험으로 따로 확인.
    반환: {"lanes": [...], "problems": [...]}"""
    rows, problems = [], []
    for e in net.getEdges():
        for ln in e.getLanes():
            bus_only = ln.allows("bus") and not ln.allows("passenger")
            if not bus_only:
                continue
            rows.append({"edge": e.getID(), "name": e.getName(), "lane": ln.getIndex(), "n_lanes": e.getLaneNumber(),
                         "length_m": round(e.getLength(), 1)})
            # 버스가 이 차로에서 다음 도로로 이어지는가 (도로망 끝 = 진출 토막은 제외)
            if not ln.getOutgoing() and e.getOutgoing():
                problems.append(f"{e.getID()}_{ln.getIndex()}: 버스전용차로가 다음 도로로 이어지지 않음")
    return {"lanes": rows, "problems": problems}


def normalize_osm_file(osm_path, out_path):
    """OSM 파일 전체에 normalize_bus_tags를 적용해 out_path에 쓴다. 바꾼 way가 없으면 원본 경로를 돌려준다."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(osm_path)
    changed = 0
    for w in tree.getroot().iter("way"):
        tag_el = {x.get("k"): x for x in w.iter("tag")}
        new = normalize_bus_tags({k: el.get("v") for k, el in tag_el.items()})
        for k, v in new.items():
            if v is None:
                if k in tag_el:
                    w.remove(tag_el[k])
            elif k in tag_el:
                tag_el[k].set("v", v)
            else:
                ET.SubElement(w, "tag", {"k": k, "v": v})
        changed += bool(new)
    if not changed:
        return osm_path
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return out_path


def _bus_only(lane):
    return lane.allows("bus") and not lane.allows("passenger")


def brt_through_deletes(net):
    """버스전용차로 다음에 버스전용차로가 이어지는 곳(BRT 같은 중앙 버스차로)은 그 연결만 남긴다.
    netconvert는 버스전용차로를 모든 진출 도로(좌·우회전 포함)에 이어서, 맨 안쪽 버스차로에서 우회전하는 등
    일반 차로를 가로지르는 이동이 생긴다. 반환: <delete>로 지울 (from, to, fromLane, toLane) 목록."""
    out = []
    for e in net.getEdges():
        for ln in e.getLanes():
            if not _bus_only(ln):
                continue
            cons = ln.getOutgoing()
            if not any(_bus_only(c.getToLane()) for c in cons):
                continue
            for c in cons:
                if not _bus_only(c.getToLane()):
                    out.append((e.getID(), c.getTo().getID(), ln.getIndex(), c.getToLane().getIndex()))
    return sorted(set(out))


def _segs(shape):
    return list(zip(shape, shape[1:]))


def _cross(p1, p2, q1, q2):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = o(q1, q2, p1), o(q1, q2, p2), o(p1, p2, q1), o(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def bus_link_conflicts(net_path, tl_id, near_m=2.4, max_m=90.0):
    """신호 tl_id의 링크 중 버스전용 링크와 경로가 겹치는 승용차 링크: {승용차 linkIndex: {버스 linkIndex}}.
    경로 = 교차로 내부 차로 + (노드 여러 개가 신호 하나로 묶인 교차로면) 그 사이 짧은 도로까지, 갈래는 모두.
    두 경로(차로 중심선)가 엇갈리면 겹친다고 본다. SUMO는 같은 진입 도로에서 나온 연결끼리는 서로 양보하지
    않으므로(차로 순서대로 방향이 나뉜다고 가정), 맨 안쪽 버스차로 직진 × 그 옆 차로 좌회전 같은 엇갈림은
    신호로 떼어 놓아야 한다 → bus_subphases."""
    import sumolib
    net = sumolib.net.readNet(net_path, withInternal=True)
    tls = net.getTLS(tl_id)
    tl_nodes = {fl.getEdge().getToNode().getID() for fl, _, _ in tls.getConnections()}

    def inside_cluster(lane):
        e = lane.getEdge()
        return lane.getID().startswith(":") or (e.getFromNode().getID() in tl_nodes and e.getToNode().getID() in tl_nodes)

    def path_points(start):
        """갈래마다 1 m 간격 점 목록 (갈래 사이를 잇지 않도록 갈래 경계에 None)"""
        out_pts = []

        def walk(lane, acc, prefix, depth):
            sh = lane.getShape()
            pts = list(prefix)
            for a, b in zip(sh, sh[1:]):
                n = max(1, int(math.dist(a, b)))
                pts += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]
            pts.append(sh[-1])
            nxts = []
            for c in lane.getOutgoing():
                nxt = c.getViaLaneID() and net.getLane(c.getViaLaneID()) or c.getToLane()
                if inside_cluster(nxt) and depth < 8 and acc + lane.getLength() < max_m:
                    nxts.append(nxt)
            if not nxts:
                out_pts.extend(pts + [None])
            for nx in nxts:
                walk(nx, acc + lane.getLength(), pts, depth + 1)
        walk(start, 0.0, [], 0)
        return out_pts

    paths = {}
    for fl, tl, idx in tls.getConnections():
        c = next((c for c in fl.getOutgoing() if c.getToLane() == tl), None)
        if c is None:
            continue
        start = net.getLane(c.getViaLaneID()) if c.getViaLaneID() else tl
        paths[idx] = (_bus_only(fl), path_points(start), fl.getID())
    out = {}
    for i, (bi, pi, fi) in paths.items():
        if bi:
            continue
        for j, (bj, pj, fj) in paths.items():
            if not bj or fi == fj:
                continue
            if any(_cross(a, b, c, d) for a, b in zip(pi, pi[1:]) if a and b
                   for c, d in zip(pj, pj[1:]) if c and d):
                out.setdefault(i, set()).add(j)
    return out


def bus_subphases(phases, conflicts, bus_links, bus_green=6.0, yellow=3.0, min_green=5.0, approach=None):
    """현시 목록 [(초, 상태)]에서, 녹색 버스 링크와 엇갈리는 승용차 링크가 같은 녹색 구간에 함께 녹색이면
    그 녹색 구간을  [일반 녹색(버스 적색)] → [엇갈리는 승용차 황색] → [BRT 녹색(엇갈리는 승용차 적색)]  으로 나눈다.
    뒤따르는 황색·전적색에서도 그 승용차 링크는 적색을 유지한다.
    conflicts = bus_link_conflicts(...) 결과, bus_links = 버스전용 링크 번호 집합."""
    if approach is not None:      # 진입 정지선 링크끼리만 (교차로 내부 링크는 항상 녹색 — 진입에서 이미 나뉨)
        conflicts = {i: bs & approach for i, bs in conflicts.items() if i in approach and bs & approach}
        bus_links = bus_links & approach
    if not conflicts:
        return phases
    out, hold = [], set()                   # hold: 앞 BRT 녹색에서 적색으로 바뀐 승용차 링크
    for dur, st in phases:
        s = list(st)
        if "G" not in st and "g" not in st or all(s[j] not in "Gg" for j in bus_links if j < len(s)):
            for i in hold:
                if s[i] in "yYGg":
                    s[i] = "r"
            if not any(c in "Gg" for k, c in enumerate(s) if k not in bus_links) and not any(s[j] in "Gg" for j in bus_links):
                hold = set()
            out.append((dur, "".join(s)))
            continue
        greens_bus = {j for j in bus_links if j < len(s) and s[j] in "Gg"}
        clash = {i for i, bs in conflicts.items() if i < len(s) and s[i] in "Gg" and bs & greens_bus}
        if not clash or dur < min_green + yellow + 1:
            out.append((dur, st))
            continue
        tb = min(bus_green, dur - yellow - min_green)
        a = list(s)
        for j in greens_bus:
            a[j] = "r"
        out.append((dur - tb - yellow, "".join(a)))
        y = list(a)
        for i in clash:
            y[i] = "y"
        out.append((yellow, "".join(y)))
        b = list(s)
        for i in clash:
            b[i] = "r"
        out.append((tb, "".join(b)))
        hold = clash
    # 전체 현시 순환: 마지막 hold는 다음 주기 시작 전 정리된다고 본다
    return out
