"""영역 지도 생성기: '외곽 순환 도로(빨간 선) + 그 안쪽 큰 대로'만으로 SUMO 도로망을 만든다.

설정: env/maps/<name>.json  (예: env/maps/yeonsan_area.json)
    osm_file        : OSM 원본 (저장소에 포함)
    polygon         : 사용자 지도 캡처의 빨간 선 꼭짓점 [[lat, lon], ...] (순서대로)
    ring_waypoints  : 순환 도로 추적용 경유점 [{"lat":..,"lon":..,"road": "도로명" | "roundabout"}, ...]
    ring_roads      : 순환 도로를 이루는 도로 이름들
    major_highways  : '큰 대로'로 볼 OSM highway 등급 (안쪽 도로 선택 기준)
    major_names     : 등급과 상관없이 넣을 안쪽 도로 이름 (선택)
    signal_policy   : "osm+major" → OSM 신호 + 큰 대로끼리 만나는 교차로를 신호로 (가정)
    buffer_m        : 순환 도로 바깥으로 이만큼(m)까지는 영역 안으로 본다

결과: env/maps/<name>.net.xml, <name>.{nod,edg,con,tll,typ}.xml (합치기용 plain XML),
      <name>.check.json / <name>.check.md (검증 보고서)

검증 항목 (매번 생성 후 자동 실행)
    V1 빨간 선 ↔ 도로망 거리      : 빨간 선을 10m 간격으로 찍어 가장 가까운 도로까지 거리
    V2 순환 주행 가능              : 순환 도로를 시계/반시계 방향으로 한 바퀴 갈 수 있는지
    V3 안쪽 큰 대로 포함           : 도로 이름별 영역 안 길이 (OSM vs 도로망)
    V4 차로 수                    : 도로망 차로 수 vs OSM lanes 태그 (태그 없는 구간은 '미확인')
    V5 회전 제한                  : OSM 회전 금지(relation)가 도로망 연결에 반영됐는지
    V6 연결성                     : 모든 도로가 서로 오갈 수 있는지 (강연결 성분)
    V7 일방통행                   : OSM oneway 도로에 역방향 도로가 생기지 않았는지
    V8 신호                       : OSM 신호 / 가정 신호 교차로 목록

사용법:  python tools/build_area_map.py env/maps/yeonsan_area.json
"""
import collections
import heapq
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DRIVE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street"}

TYPES_XML = """<types>
  <type id="highway.trunk"          priority="13" numLanes="2" speed="16.67" oneway="false"/>
  <type id="highway.trunk_link"     priority="12" numLanes="1" speed="8.33"  oneway="true"/>
  <type id="highway.primary"        priority="12" numLanes="2" speed="13.89" oneway="false"/>
  <type id="highway.primary_link"   priority="11" numLanes="1" speed="8.33"  oneway="true"/>
  <type id="highway.secondary"      priority="11" numLanes="2" speed="13.89" oneway="false"/>
  <type id="highway.secondary_link" priority="10" numLanes="1" speed="8.33"  oneway="true"/>
  <type id="highway.tertiary"       priority="8"  numLanes="1" speed="13.89" oneway="false"/>
  <type id="highway.tertiary_link"  priority="7"  numLanes="1" speed="8.33"  oneway="true"/>
  <type id="highway.unclassified"   priority="5"  numLanes="1" speed="8.33"  oneway="false"/>
  <type id="highway.residential"    priority="4"  numLanes="1" speed="8.33"  oneway="false"/>
</types>
"""


# ──────────────────────────────────────────────────────────────
# 좌표 / 기하
# ──────────────────────────────────────────────────────────────
class Proj:
    """위경도 ↔ 국소 평면(m). 영역이 수 km라 등장방형 근사로 충분 (오차 < 0.1%)."""

    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = 111320 * math.cos(math.radians(lat0))
        self.ky = 110574

    def en(self, lat, lon):
        return ((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky)


def point_seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def dist_to_polyline(p, pl):
    return min(point_seg_dist(p, pl[i], pl[i + 1]) for i in range(len(pl) - 1))


def inside(p, poly):
    x, y = p
    c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def densify(pl, step=10.0):
    out = []
    for p, q in zip(pl, pl[1:]):
        n = max(1, int(math.dist(p, q) / step))
        out += [(p[0] + (q[0] - p[0]) * k / n, p[1] + (q[1] - p[1]) * k / n) for k in range(n)]
    out.append(pl[-1])
    return out


# ──────────────────────────────────────────────────────────────
# OSM 읽기 / 순환 도로 추적 / 도로 선택
# ──────────────────────────────────────────────────────────────
def load_osm(path, proj):
    root = ET.parse(path).getroot()
    nodes, ntags = {}, {}
    for n in root.iter("node"):
        nodes[n.get("id")] = proj.en(float(n.get("lat")), float(n.get("lon")))
        t = {x.get("k"): x.get("v") for x in n.iter("tag")}
        if t:
            ntags[n.get("id")] = t
    ways = {}
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if t.get("highway") in DRIVE:
            ways[w.get("id")] = (t, [nd.get("ref") for nd in w.iter("nd") if nd.get("ref") in nodes])
    rels = []
    for r in root.iter("relation"):
        t = {x.get("k"): x.get("v") for x in r.iter("tag")}
        if t.get("type") == "restriction":
            rels.append((r, t))
    return root, nodes, ntags, ways, rels


def apply_way_splits(root, ways, splits):
    """OSM way를 지정한 노드에서 둘로 나누고, 뒷부분에 태그를 덮어쓴다 (way_splits 설정).

    예: OSM lanes 태그가 way 전체에 붙어 있는데 중간에 우회전 연결로가 갈라져서
        뒷부분은 실제로 차로가 적은 경우. 뒷부분은 새 way id(9 + 원래 id)가 되고,
        그 구간을 지나는 회전 제한(relation)의 from/to도 새 way로 옮긴다.
    """
    done = []
    for sp in splits or []:
        wid, at = str(sp["way"]), str(sp["at_node"])
        w = next((x for x in root.iter("way") if x.get("id") == wid), None)
        if w is None:
            continue
        nds = [n for n in w if n.tag == "nd"]
        refs = [n.get("ref") for n in nds]
        if at not in refs or refs.index(at) in (0, len(refs) - 1):
            raise ValueError(f"way_splits: {at}는 way {wid}의 중간 노드가 아님")
        k = refs.index(at)
        new_id = "9" + wid
        nw = ET.Element("way", {**w.attrib, "id": new_id})
        for r in refs[k:]:
            ET.SubElement(nw, "nd", {"ref": r})
        tags = {t.get("k"): t.get("v") for t in w if t.tag == "tag"}
        tags.update({k2: str(v) for k2, v in (sp.get("tail_tags") or {}).items()})
        for k2, v in tags.items():
            ET.SubElement(nw, "tag", {"k": k2, "v": v})
        for n in nds[k + 1:]:
            w.remove(n)
        root.append(nw)
        tail = set(refs[k + 1:])
        for rel in root.iter("relation"):
            via = {m.get("ref") for m in rel.iter("member") if m.get("role") == "via"}
            for m in rel.iter("member"):
                if m.get("type") == "way" and m.get("ref") == wid and m.get("role") in ("from", "to") and via & (tail | {refs[-1]}):
                    m.set("ref", new_id)
        if wid in ways:
            t0 = dict(ways[wid][0])
            ways[wid] = (t0, refs[:k + 1])
            ways[new_id] = (tags, refs[k:])
        done.append((wid, new_id, at))
    return done


def trace_ring(cfg, proj, nodes, ways, directed=False, reverse=False):
    """경유점을 차례로 최단경로로 이어 순환 도로를 찾는다 (순환 도로 이름엔 가중치 1, 그 외 큰 벌점).

    directed=False: 일방통행 무시 (영역 모양을 정할 때)
    directed=True : 일방통행을 지켜서 실제 차가 갈 수 있는 순환 경로 (reverse=True면 반대 방향으로 한 바퀴)
    """
    ring_names = set(cfg["ring_roads"])
    G = collections.defaultdict(list)
    for wid, (t, nds) in ways.items():
        ring = t.get("name") in ring_names or t.get("junction") in ("roundabout", "circular")
        hw = t.get("highway", "")
        f = 1.0 if ring else (3.0 if hw.startswith(("secondary", "primary", "tertiary")) else 10.0)
        ow = t.get("oneway")
        oneway_fwd = ow in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular")
        oneway_bwd = ow == "-1"
        for a, b in zip(nds, nds[1:]):
            d = math.dist(nodes[a], nodes[b]) * f
            if not directed or not oneway_bwd:
                G[a].append((b, d, wid))
            if not directed or not oneway_fwd:
                G[b].append((a, d, wid))

    node_ways = collections.defaultdict(set)
    for wid, (t, nds) in ways.items():
        for n in nds:
            node_ways[n].add(wid)

    def nearest(wp):
        p = proj.en(wp["lat"], wp["lon"])
        road = wp.get("road")
        best = None
        for wid, (t, nds) in ways.items():
            roads = road if isinstance(road, list) else [road]
            ok = (t.get("junction") in ("roundabout", "circular")) if road == "roundabout" else \
                (road is None or t.get("name") in roads)
            if not ok:
                continue
            for n in nds:
                d = math.dist(nodes[n], p)
                if isinstance(road, list) and len(road) > 1:
                    meet = {ways[w][0].get("name") for w in node_ways.get(n, ())}
                    if not set(road) <= meet:
                        d += 1000.0           # 모든 도로가 만나는 노드가 아니면 뒤로
                if best is None or d < best[0]:
                    best = (d, n)
        if best is None:
            raise RuntimeError(f"경유점 근처에서 도로 '{road}'를 찾지 못함: {wp}")
        return best[1]

    def path(a, b):
        dist, prev, h = {a: 0.0}, {}, [(0.0, a)]
        while h:
            d, u = heapq.heappop(h)
            if u == b:
                break
            if d > dist[u]:
                continue
            for v, w, wid in G[u]:
                if d + w < dist.get(v, 1e18):
                    dist[v], prev[v] = d + w, (u, wid)
                    heapq.heappush(h, (d + w, v))
        if b not in dist:
            raise RuntimeError("순환 도로 경유점 사이 경로 없음")
        ns, ws = [b], []
        while ns[-1] != a:
            u, wid = prev[ns[-1]]
            ns.append(u)
            ws.append(wid)
        return ns[::-1], ws[::-1]

    wps = [nearest(w) for w in cfg["ring_waypoints"]]
    if reverse:
        wps = wps[::-1]
    ring_nodes, ring_ways = [], []
    for a, b in zip(wps, wps[1:] + wps[:1]):
        ns, ws = path(a, b)
        ring_nodes += ns if not ring_nodes else ns[1:]
        ring_ways += ws
    return ring_nodes, ring_ways


def select_ways(cfg, nodes, ways, ring_line, ring_poly):
    buf = float(cfg.get("buffer_m", 25.0))
    ring_names = set(cfg["ring_roads"])
    major = set(cfg.get("major_highways", ["primary", "secondary", "primary_link", "secondary_link"]))
    major_names = set(cfg.get("major_names", []))
    sel, why = set(), {}
    for wid, (t, nds) in ways.items():
        pts = [nodes[n] for n in nds]
        near_ring = [dist_to_polyline(p, ring_line) <= buf for p in pts]
        in_poly = [inside(p, ring_poly) for p in pts]
        if (t.get("name") in ring_names or t.get("junction") in ("roundabout", "circular")) \
                and sum(near_ring) >= max(1, len(pts) // 2):
            sel.add(wid); why[wid] = "순환"
        elif (t.get("highway") in major or t.get("name") in major_names) and any(
                i or r for i, r in zip(in_poly, near_ring)):
            sel.add(wid); why[wid] = "안쪽 큰 대로"
    return sel, why


def write_filtered_osm(root, sel, rels_keep, path):
    """선택한 way + 그 노드 + 회전 제한 relation만 담은 OSM 파일."""
    keep_nodes = set()
    out = ET.Element("osm", {"version": "0.6", "generator": "build_area_map"})
    b = root.find("bounds")
    if b is not None:
        out.append(b)
    ways_out = []
    for w in root.iter("way"):
        if w.get("id") in sel:
            ways_out.append(w)
            keep_nodes |= {nd.get("ref") for nd in w.iter("nd")}
    for n in root.iter("node"):
        if n.get("id") in keep_nodes:
            out.append(n)
    for w in ways_out:
        out.append(w)
    for r in rels_keep:
        out.append(r)
    ET.ElementTree(out).write(path, encoding="UTF-8", xml_declaration=True)


# ──────────────────────────────────────────────────────────────
# 차로 조사표 (CSV)
#   OSM에 차로 수/차로별 방향이 없는 도로가 많아서, 팀이 로드뷰로 확인해 채우는 표.
#   빌드할 때마다 현재 값으로 다시 쓰되, 사람이 채운 칸(실제_*, turn_*, 메모)은 유지한다.
#   turn 칸은 OSM turn:lanes 형식: 왼쪽 차로부터 '|'로 구분, 한 차로에 여러 방향은 ';'
#     예) "left|through|through;right"  (좌회전 전용 | 직진 | 직진+우회전)
#   정방향 = OSM way 그리기 방향 (도로망 엣지 id에 '-'가 없는 쪽)
# ──────────────────────────────────────────────────────────────
LANE_COLS = ["way_id", "도로명", "등급", "일방", "길이_m", "OSM_lanes", "OSM_정방향", "OSM_역방향",
             "도로망_정방향", "도로망_역방향", "실제_정방향", "실제_역방향", "turn_정방향", "turn_역방향", "메모"]


def read_lane_table(path):
    import csv
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vals = {k: (row.get(k) or "").strip() for k in ("실제_정방향", "실제_역방향", "turn_정방향", "turn_역방향", "메모")}
            if any(vals[k] for k in ("실제_정방향", "실제_역방향", "turn_정방향", "turn_역방향")):
                out[row["way_id"]] = vals
    return out


def apply_overrides(root, ways, overrides):
    """조사표 값을 OSM way 태그로 바꿔 넣는다 (netconvert가 그대로 읽음)."""
    if not overrides:
        return
    for w in root.iter("way"):
        o = overrides.get(w.get("id"))
        if not o:
            continue
        tags = {t.get("k"): t for t in w.iter("tag")}
        oneway = tags.get("oneway") is not None and tags["oneway"].get("v") in ("yes", "1", "true")
        new = {}
        if oneway:
            if o["실제_정방향"]:
                new["lanes"] = o["실제_정방향"]
            if o["turn_정방향"]:
                new["turn:lanes"] = o["turn_정방향"]
        else:
            f, b = o["실제_정방향"], o["실제_역방향"]
            if f:
                new["lanes:forward"] = f
            if b:
                new["lanes:backward"] = b
            if f and b:
                new["lanes"] = str(int(f) + int(b))
            if o["turn_정방향"]:
                new["turn:lanes:forward"] = o["turn_정방향"]
            if o["turn_역방향"]:
                new["turn:lanes:backward"] = o["turn_역방향"]
        for k, v in new.items():
            if k in tags:
                tags[k].set("v", v)
            else:
                ET.SubElement(w, "tag", {"k": k, "v": v})
        if w.get("id") in ways:
            ways[w.get("id")][0].update(new)


def write_lane_table(path, net_f, ways, sel, why, overrides):
    import csv
    import sumolib
    net = sumolib.net.readNet(net_f)
    lanes_net = collections.defaultdict(lambda: [None, None])
    length = collections.Counter()
    for e in net.getEdges():
        wid = e.getID().lstrip("-").split("#")[0]
        lanes_net[wid][1 if e.getID().startswith("-") else 0] = e.getLaneNumber()
        if not e.getID().startswith("-"):
            length[wid] += e.getLength()
    dead = collections.defaultdict(list)          # 막힌 차로(V10)를 조사표 메모에 표시
    for e in net.getEdges():
        outs = [len(ln.getOutgoing()) for ln in e.getLanes()]
        if any(outs) and not all(outs):
            wid = e.getID().lstrip("-").split("#")[0]
            side = "역방향" if e.getID().startswith("-") else "정방향"
            dead[wid].append(f"{side} {e.getID()} 차로 {[i for i, n in enumerate(outs) if n == 0]} 갈 곳 없음(확인 필요)")
    rows = []
    for wid in sorted(sel, key=lambda w: (ways[w][0].get("name") or "~", w)):
        if wid not in lanes_net:
            continue
        t, nds = ways[wid]
        f, b = osm_lanes_expect(t)
        o = overrides.get(wid, {})
        rows.append({"way_id": wid, "도로명": t.get("name", ""), "등급": t.get("highway"),
                     "일방": "예" if t.get("oneway") in ("yes", "1", "true") or t.get("junction") else "",
                     "길이_m": round(length[wid]) if length[wid] else "", "OSM_lanes": t.get("lanes", ""),
                     "OSM_정방향": "" if f is None else f, "OSM_역방향": "" if b is None else b,
                     "도로망_정방향": lanes_net[wid][0] or "", "도로망_역방향": lanes_net[wid][1] or "",
                     "실제_정방향": o.get("실제_정방향", ""), "실제_역방향": o.get("실제_역방향", ""),
                     "turn_정방향": o.get("turn_정방향", ""), "turn_역방향": o.get("turn_역방향", ""),
                     "메모": o.get("메모", "") or "; ".join(dead.get(wid, []))})
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LANE_COLS)
        w.writeheader()
        w.writerows(rows)


# ──────────────────────────────────────────────────────────────
# 빌드
# ──────────────────────────────────────────────────────────────
def build(cfg_path):
    import sumolib
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    out_dir = os.path.join(BASE, cfg.get("out_dir", "env/maps"))
    os.makedirs(out_dir, exist_ok=True)
    poly_ll = cfg["polygon"]
    lat0 = sum(p[0] for p in poly_ll) / len(poly_ll)
    lon0 = sum(p[1] for p in poly_ll) / len(poly_ll)
    proj = Proj(lat0, lon0)
    red = [proj.en(*p) for p in poly_ll]

    root, nodes, ntags, ways, rels = load_osm(os.path.join(BASE, cfg["osm_file"]), proj)
    splits_done = apply_way_splits(root, ways, cfg.get("way_splits"))
    ring_nodes, ring_ways = trace_ring(cfg, proj, nodes, ways)
    ring_line = [nodes[n] for n in ring_nodes]
    ring_poly = ring_line[:-1]
    sel, why = select_ways(cfg, nodes, ways, ring_line, ring_poly)
    sel |= set(ring_ways)
    for w in ring_ways:
        why.setdefault(w, "순환")

    # 회전 제한: from/to way가 둘 다 선택된 것만
    rels_keep = []
    for r, t in rels:
        mem = {m.get("role"): m.get("ref") for m in r.iter("member") if m.get("type") == "way"}
        if mem.get("from") in sel and mem.get("to") in sel:
            rels_keep.append(r)

    # 차로 조사표(CSV) 반영: 팀이 로드뷰로 확인한 차로 수 / 차로별 진행 방향(turn:lanes)
    lane_csv = os.path.join(BASE, cfg.get("lane_table", f"env/maps/{name}.lanes.csv"))
    overrides = read_lane_table(lane_csv)
    apply_overrides(root, ways, overrides)

    tmp = tempfile.mkdtemp(prefix="area_")
    osm_f = os.path.join(tmp, "area.osm")
    write_filtered_osm(root, sel, rels_keep, osm_f)
    types_f = os.path.join(tmp, "types.typ.xml")
    open(types_f, "w").write(TYPES_XML)

    net_f = os.path.join(out_dir, f"{name}.net.xml")
    raw0_f = os.path.join(tmp, "raw0.net.xml")
    raw_f = os.path.join(tmp, "raw.net.xml")
    # 0) 차로 연결 지정 (connections 설정): 그 도로에서 나가는 연결은 여기 적은 것만 생긴다
    con_args = []
    if cfg.get("connections"):
        con_f = os.path.join(tmp, "fixed.con.xml")
        with open(con_f, "w") as f:
            f.write("<connections>\n")
            for c in cfg["connections"]:
                f.write(f'    <connection from="{c["from"]}" to="{c["to"]}" fromLane="{c["fromLane"]}" '
                        f'toLane="{c["toLane"]}"/>\n')
            f.write("</connections>\n")
        con_args = ["--connection-files", con_f]
    # 1) OSM → 도로망 (신호는 OSM 신호 노드만: --tls.guess-signals)
    res = subprocess.run(
        [sumolib.checkBinary("netconvert"), "--osm-files", osm_f, "--type-files", types_f, *con_args,
         "--geometry.remove", "--ramps.guess", "--junctions.join",
         "--tls.guess-signals", "--tls.join", "--tls.default-type", "static",
         "--remove-edges.isolated", "--keep-edges.by-vclass", "passenger",
         "--no-turnarounds", "true", "--output.street-names",     # 유턴 없음 (경계 토막 끝 포함)
         "--osm.turn-lanes", "true", "-o", raw0_f], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-3000:])

    # 2) 신호 가정: 만들어진 도로망의 교차로(합쳐진 노드 포함) 중 큰 대로/순환 도로가
    #    2개 이상 만나는 곳 (OSM에 신호 태그가 빠진 곳이 많아서). 회전교차로·연결로만 만나는 곳 제외.
    tls_set = []
    if cfg.get("signal_policy", "osm+major") == "osm+major":
        major = set(cfg.get("major_highways", []))
        ring_names = set(cfg["ring_roads"])
        rb_ways = {w for w in sel if ways[w][0].get("junction") in ("roundabout", "circular")}
        net0 = sumolib.net.readNet(raw0_f)
        for nd in net0.getNodes():
            if nd.getType() == "traffic_light":
                continue
            names, rb = set(), False
            for e in nd.getIncoming() + nd.getOutgoing():
                wid = e.getID().lstrip("-").split("#")[0]
                t = ways.get(wid, ({}, []))[0]
                if wid in rb_ways:
                    rb = True
                if t.get("highway", "").endswith("_link") or not t.get("name"):
                    continue
                if t.get("highway") in major or t.get("name") in ring_names:
                    names.add(t["name"])
            if len(names) >= 2 and not rb:
                tls_set.append(nd.getID())
        tls_set += [str(x) for x in cfg.get("extra_signals", [])]
    if tls_set:
        # --tls.join: 가까운 신호들(예: 연산교차로를 이루는 노드 4개)을 하나의 신호로 묶는다
        subprocess.run([sumolib.checkBinary("netconvert"), "-s", raw0_f, "--tls.set", ",".join(tls_set),
                        "--tls.join", "--tls.default-type", "static", "--output.street-names", "-o", raw_f],
                       check=True, capture_output=True)
    else:
        raw_f = raw0_f

    # 영역 밖 도로 자르기: 도로 중간점이 (순환 도로 + buffer) 밖이면 제거
    net = sumolib.net.readNet(raw_f)
    buf = float(cfg.get("buffer_m", 25.0))

    def xy_to_en(x, y):
        lon, lat = net.convertXY2LonLat(x, y)
        return proj.en(lat, lon)

    keep = set()
    for e in net.getEdges():
        sh = e.getShape()
        mx, my = sh[len(sh) // 2] if len(sh) > 2 else ((sh[0][0] + sh[-1][0]) / 2, (sh[0][1] + sh[-1][1]) / 2)
        p = xy_to_en(mx, my)
        if inside(p, ring_poly) or dist_to_polyline(p, ring_line) <= buf:
            keep.add(e.getID())
    # 진입/진출 토막: 순환 도로 밖으로 이어지는 큰 대로·순환 도로는 바로 바깥 도로 하나를
    # 남겨 두었다가(아래에서 stub_m 길이로 자름) 배경 교통이 영역에 드나드는 입구/출구로 쓴다.
    major = set(cfg.get("major_highways", []))
    stub_m = float(cfg.get("stub_m", 80.0))
    core_nodes = {n for eid in keep for n in (net.getEdge(eid).getFromNode().getID(),
                                              net.getEdge(eid).getToNode().getID())}
    stubs = {}
    for e in net.getEdges():
        if e.getID() in keep:
            continue
        wid = e.getID().lstrip("-").split("#")[0]
        t = ways.get(wid, ({}, []))[0]
        if not (t.get("highway") in major or t.get("name") in set(cfg["ring_roads"])):
            continue
        a, b = e.getFromNode().getID(), e.getToNode().getID()
        if a in core_nodes and b not in core_nodes:
            stubs[e.getID()] = b            # 바깥 끝 노드
        elif b in core_nodes and a not in core_nodes:
            stubs[e.getID()] = a
    keep |= set(stubs)
    keep_f = os.path.join(tmp, "keep.txt")
    open(keep_f, "w").write("\n".join(sorted(keep)))
    clip_f = os.path.join(tmp, "clip.net.xml")
    subprocess.run([sumolib.checkBinary("netconvert"), "-s", raw_f, "--keep-edges.input-file", keep_f,
                    "--remove-edges.isolated", "--output.street-names", "-o", clip_f],
                   check=True, capture_output=True)
    # 토막 자르기: 바깥 끝 노드를 순환 쪽 노드에서 stub_m 지점으로 옮긴다
    from tools.merge_nets import export, _plain_args, _files, _run
    plain = os.path.join(tmp, "clip")
    export(clip_f, plain)
    cnet = sumolib.net.readNet(clip_f)
    nod_t = ET.parse(f"{plain}.nod.xml")
    nod_by_id = {n.get("id"): n for n in nod_t.getroot().iter("node")}
    edg_t = ET.parse(f"{plain}.edg.xml")
    moved = {}
    for eid, outer in stubs.items():
        if not cnet.hasEdge(eid) or outer in moved:
            continue
        e = cnet.getEdge(eid)
        sh = e.getShape() if e.getToNode().getID() == outer else e.getShape()[::-1]   # 순환 쪽 → 바깥 끝
        acc, cut = 0.0, None
        for p, q in zip(sh, sh[1:]):
            seg = math.dist(p, q)
            if acc + seg >= stub_m:
                f = (stub_m - acc) / seg
                cut = (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
                break
            acc += seg
        if cut is None or outer not in nod_by_id:
            continue                                   # 이미 stub_m보다 짧음
        nod_by_id[outer].set("x", f"{cut[0]:.2f}")
        nod_by_id[outer].set("y", f"{cut[1]:.2f}")
        moved[outer] = cut
    if moved:
        # 바깥 끝 노드에 닿는 도로들의 모양(shape)은 직선 근사로 다시 계산되게 지운다
        for ed in edg_t.getroot().iter("edge"):
            if ed.get("from") in moved or ed.get("to") in moved:
                if "shape" in ed.attrib:
                    del ed.attrib["shape"]
                for ln in ed.iter("lane"):
                    if "shape" in ln.attrib:
                        del ln.attrib["shape"]
    if moved:
        nod_t.write(f"{plain}.nod.xml", encoding="UTF-8", xml_declaration=True)
        edg_t.write(f"{plain}.edg.xml", encoding="UTF-8", xml_declaration=True)
    _run(_plain_args(_files(plain)) + ["--output.street-names", "-o", net_f])
    # 합치기용 plain XML
    export(net_f, os.path.join(out_dir, name))
    # 생성 파일 머리 주석(빌드 컴퓨터의 임시 경로가 들어감)과 .netccfg는 지운다 (저장소에 올리기 위해)
    import re
    cfg_f = os.path.join(out_dir, f"{name}.netccfg")
    if os.path.exists(cfg_f):
        os.remove(cfg_f)
    for ext in ("net", "nod", "edg", "con", "tll", "typ"):
        fpath = net_f if ext == "net" else os.path.join(out_dir, f"{name}.{ext}.xml")
        if os.path.exists(fpath):
            txt = open(fpath, encoding="utf-8").read()
            txt = re.sub(r"<!-- generated on .*?-->\s*", "", txt, count=1, flags=re.S)
            open(fpath, "w", encoding="utf-8").write(txt)

    write_lane_table(lane_csv, net_f, ways, sel, why, overrides)
    meta = {"name": name, "ring_ways": ring_ways, "selected_ways": {w: why[w] for w in sorted(sel)},
            "lane_overrides": len(overrides),
            "restrictions_kept": len(rels_keep), "tls_assumed": tls_set, "lat0": lat0, "lon0": lon0}
    json.dump(meta, open(os.path.join(out_dir, f"{name}.meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    report = check(cfg, name, out_dir, net_f, proj, red, ring_line, ring_poly, nodes, ntags, ways, sel,
                   rels_keep, tls_set, ring_nodes, ring_ways)
    return net_f, report


# ──────────────────────────────────────────────────────────────
# 검증
# ──────────────────────────────────────────────────────────────
def osm_lanes_expect(t):
    """OSM 태그에서 (정방향 차로, 역방향 차로) 기대값. 모르면 None."""
    oneway = t.get("oneway") in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular")
    if oneway:
        return (int(t["lanes"]) if t.get("lanes", "").isdigit() else None, 0)
    f = t.get("lanes:forward")
    b = t.get("lanes:backward")
    if f and b and f.isdigit() and b.isdigit():
        return (int(f), int(b))
    L = t.get("lanes", "")
    if L.isdigit():
        n = int(L)
        return ((n + 1) // 2, n // 2)
    return (None, None)


def check(cfg, name, out_dir, net_f, proj, red, ring_line, ring_poly, nodes, ntags, ways, sel, rels_keep, tls_set,
          ring_nodes=(), ring_ways=()):
    import sumolib
    net = sumolib.net.readNet(net_f, withInternal=False)

    def edge_en(e):
        return [proj.en(*reversed(net.convertXY2LonLat(x, y))) for x, y in e.getShape()]

    E = {e.getID(): edge_en(e) for e in net.getEdges()}
    rep = {}

    # V1 빨간 선 ↔ 도로망
    seg = []
    for i in range(len(red)):
        a, b = red[i], red[(i + 1) % len(red)]
        ds = []
        for p in densify([a, b], 10):
            ds.append(min(dist_to_polyline(p, pl) for pl in E.values() if pl))
        ds.sort()
        seg.append({"from": i, "to": (i + 1) % len(red), "len_m": round(math.dist(a, b)),
                    "median_m": round(ds[len(ds) // 2], 1), "p90_m": round(ds[int(len(ds) * .9)], 1),
                    "max_m": round(ds[-1], 1)})
    rep["V1_red_line"] = seg

    # V2 순환 주행 (시계/반시계): 순환 도로 위 경유점들을 잇는 경로
    def nearest_edge(p, prefer=None):
        best = None
        for eid, pl in E.items():
            if not pl:
                continue
            d = dist_to_polyline(p, pl)
            if best is None or d < best[0]:
                best = (d, eid)
        return net.getEdge(best[1])

    L = sum(math.dist(p, q) for p, q in zip(ring_line, ring_line[1:]))
    # 순환 도로(OSM에서 추적한 way 순서)를 진행 방향 도로망 엣지 순서로 바꿔서,
    # 앞 엣지 → 다음 엣지가 우회 없이(직접 연결 또는 짧은 교차로 내부 경로) 이어지는지 본다.
    def ring_edge_sequence(rev):
        # 일방통행을 지키는 순환 경로를 방향별로 따로 추적 (실제로 차가 도는 길)
        nseq, wseq = trace_ring(cfg, proj, nodes, ways, directed=True, reverse=rev)
        out = []
        for (a, b), wid in zip(zip(nseq, nseq[1:]), wseq):
            nds = ways[wid][1]
            fwd = nds.index(a) < nds.index(b) if (a in nds and b in nds) else True
            cand = [e for e in net.getEdges()
                    if e.getID().lstrip("-").split("#")[0] == wid and e.getID().startswith("-") != fwd]
            if not cand:
                continue
            # a→b 구간을 덮는 엣지 = 두 끝점 모두에서 가까운 엣지 (같은 way가 여러 엣지로 나뉜 경우)
            e = min(cand, key=lambda x: max(dist_to_polyline(nodes[a], E[x.getID()]),
                                            dist_to_polyline(nodes[b], E[x.getID()]))
                    if len(E[x.getID()]) > 1 else 1e9)
            if not out or out[-1].getID() != e.getID():
                out.append(e)
        if len(out) > 1 and out[0].getID() == out[-1].getID():
            out = out[:-1]
        return out

    v2 = {}
    for label, rev in (("forward", False), ("reverse", True)):
        seq = ring_edge_sequence(rev)
        total, detours, broken = 0.0, [], []
        for x, y in zip(seq, seq[1:] + seq[:1]):
            total += x.getLength()
            if y in x.getOutgoing():
                continue
            path, c = net.getShortestPath(x, y, vClass="passenger")
            if not path:
                broken.append(f"{x.getID()}({x.getName()}) → {y.getID()}({y.getName()})")
                continue
            extra = c - x.getLength() - y.getLength()
            total += extra
            ex, ey = E[x.getID()][-1], E[y.getID()][0]
            if extra > math.dist(ex, ey) + 60:     # 끝점 사이 직선보다 60m 넘게 돌아감 = 우회
                detours.append({"from": f"{x.getID()}({x.getName()})", "to": f"{y.getID()}({y.getName()})",
                                "extra_m": round(extra - math.dist(ex, ey)),
                                "via": [p.getName() or p.getID() for p in path[1:-1]][:6]})
        v2[label] = {"ok": not broken, "edges": len(seq), "length_m": round(total), "ring_m": round(L),
                     "detours": detours, "broken": broken}
    rep["V2_ring_drive"] = v2

    # V3 도로명별 영역 길이 (방향별): OSM 양방향 도로는 2배로 센다 (도로망은 방향마다 도로가 따로)
    buf = float(cfg.get("buffer_m", 25.0))

    def in_region(m):
        return inside(m, ring_poly) or dist_to_polyline(m, ring_line) <= buf

    osm_len = collections.Counter()
    for wid in sel:
        t, nds = ways[wid]
        mult = 1 if (t.get("oneway") in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular")) else 2
        for a, b in zip(nds, nds[1:]):
            m = ((nodes[a][0] + nodes[b][0]) / 2, (nodes[a][1] + nodes[b][1]) / 2)
            if in_region(m):
                osm_len[t.get("name") or "(이름없음)"] += math.dist(nodes[a], nodes[b]) * mult
    net_len = collections.Counter()
    for e in net.getEdges():
        pl = E[e.getID()]
        for a, b in zip(pl, pl[1:]):
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if in_region(m):
                net_len[e.getName() or "(이름없음)"] += math.dist(a, b)
    rep["V3_roads_inside"] = {k: {"osm_m": round(v), "net_m": round(net_len.get(k, 0))}
                              for k, v in osm_len.most_common()}

    # V4 차로 수: OSM way id는 edge id 앞부분 (netconvert 규칙: "-"는 역방향, "#k"는 분할)
    lane = collections.Counter()
    unknown = collections.Counter()
    mism = []
    for e in net.getEdges():
        eid = e.getID()
        rev = eid.startswith("-")
        wid = eid.lstrip("-").split("#")[0]
        if wid not in ways:
            continue
        t = ways[wid][0]
        fwd, bwd = osm_lanes_expect(t)
        exp = bwd if rev else fwd
        if exp is None:
            unknown[t.get("name") or "(이름없음)"] += 1
            lane["unknown"] += 1
        elif exp == e.getLaneNumber():
            lane["match"] += 1
        else:
            lane["mismatch"] += 1
            mism.append({"edge": eid, "name": t.get("name"), "net": e.getLaneNumber(), "osm": exp})
    rep["V4_lanes"] = {"count": dict(lane), "unknown_by_road": dict(unknown), "mismatch": mism[:30]}

    # V5 회전 제한
    v5 = []
    for r in rels_keep:
        t = {x.get("k"): x.get("v") for x in r.iter("tag")}
        mem = [(m.get("type"), m.get("ref"), m.get("role")) for m in r.iter("member")]
        fw = next((ref for typ, ref, role in mem if role == "from"), None)
        tw = next((ref for typ, ref, role in mem if role == "to"), None)
        via = next((ref for typ, ref, role in mem if role == "via" and typ == "node"), None)
        kind = t.get("restriction", "")

        def at(node, osm_id):          # junctions.join으로 합쳐진 노드(cluster_..._id_...)도 인정
            return osm_id is not None and (node.getID() == osm_id or osm_id in node.getID().split("_"))
        # via 노드에서 끝나는 from 도로 → via 노드에서 시작하는 to 도로
        f_edges = [e for e in net.getEdges() if e.getID().lstrip("-").split("#")[0] == fw
                   and (via is None or at(e.getToNode(), via))]
        t_edges = {e.getID() for e in net.getEdges() if e.getID().lstrip("-").split("#")[0] == tw
                   and (via is None or at(e.getFromNode(), via))}
        same = fw == tw                 # 같은 도로 = 유턴: 방향이 뒤집히는 연결만 센다
        conn = any(c.getTo().getID() in t_edges
                   and (not same or c.getTo().getID().startswith("-") != e.getID().startswith("-"))
                   for e in f_edges for cl in e.getOutgoing().values() for c in cl)
        if not f_edges or not t_edges:
            status = "도로망에 없음(영역 밖)"
        elif kind.startswith("no_"):
            status = "OK" if not conn else "위반(연결 있음)"
        elif kind.startswith("only_"):
            status = "OK" if conn else "위반(연결 없음)"
        else:
            status = "?"
        v5.append({"id": r.get("id"), "type": kind, "status": status})
    rep["V5_restrictions"] = {"total": len(v5), "by_status": dict(collections.Counter(x["status"] for x in v5)),
                              "violations": [x for x in v5 if x["status"].startswith("위반")]}

    # V6 연결성 (강연결 성분)
    edges = [e for e in net.getEdges()]
    idx = {e.getID(): i for i, e in enumerate(edges)}
    adj = {e.getID(): [o.getID() for o in e.getOutgoing()] for e in edges}
    radj = collections.defaultdict(list)
    for a, bs in adj.items():
        for b in bs:
            radj[b].append(a)

    def reach(start, g):
        seen, st = {start}, [start]
        while st:
            u = st.pop()
            for v in g.get(u, []):
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        return seen
    # 가장 긴 순환 도로 엣지에서 출발한 강연결 성분
    seed = max(edges, key=lambda e: e.getLength()).getID()
    scc = reach(seed, adj) & reach(seed, radj)
    dead = [e.getID() for e in edges if not adj[e.getID()]]
    rep["V6_connectivity"] = {"edges": len(edges), "largest_scc": len(scc),
                              "scc_share": round(len(scc) / max(1, len(edges)), 3),
                              "dead_end_edges": len(dead)}

    # V7 일방통행
    v7 = []
    for e in edges:
        eid = e.getID()
        if not eid.startswith("-"):
            continue
        wid = eid.lstrip("-").split("#")[0]
        if wid in ways and ways[wid][0].get("oneway") in ("yes", "1", "true"):
            v7.append(eid)
    rep["V7_oneway_violations"] = v7

    # V8 신호
    tls_nodes = [tl.getID() for tl in net.getTrafficLights()]
    osm_sig = [n for n, t in ntags.items() if t.get("highway") == "traffic_signals"
               and n in nodes and inside(nodes[n], ring_poly)]
    rep["V8_signals"] = {"tls_in_net": len(tls_nodes), "osm_signal_nodes_inside": len(osm_sig),
                         "assumed_signal_nodes": len(tls_set)}

    # V10 막힌 차로: 다른 차로는 나갈 곳이 있는데 이 차로만 갈 곳이 없음
    #   (조사표 turn:lanes 방향에 해당하는 도로가 도로망에 없을 때 생김 → 차가 갇힘)
    dead_lanes = []
    for e in edges:
        outs = [len(ln.getOutgoing()) for ln in e.getLanes()]
        if any(outs) and not all(outs):
            dead_lanes.append({"edge": e.getID(), "name": e.getName(),
                               "lanes_without_exit": [i for i, n in enumerate(outs) if n == 0]})
    rep["V10_dead_lanes"] = dead_lanes

    # V6 보충: 입구/출구 (배경 교통이 드나드는 도로)
    rep["V6_connectivity"]["entries"] = [e.getID() for e in edges if not e.getIncoming()]
    rep["V6_connectivity"]["exits"] = [e.getID() for e in edges if not e.getOutgoing()]

    # V9 사용자 지도 캡처 위에 도로망 겹쳐 그리기 (캡처 파일이 있을 때만)
    pic = cfg.get("picture") or {}
    pic_file = os.environ.get("AREA_PICTURE") or pic.get("file")
    if pic_file and os.path.exists(pic_file) and "a" in pic:
        from PIL import Image, ImageDraw
        img = Image.open(pic_file).convert("RGB")
        k = 2
        img = img.resize((img.size[0] * k, img.size[1] * k))
        d = ImageDraw.Draw(img)
        pp = Proj(pic["lat0"], pic["lon0"])

        def to_px(lat, lon):
            e_, n_ = pp.en(lat, lon)
            return ((pic["a"] * e_ + pic["bx"]) * k, (-pic["a"] * n_ + pic["by"]) * k)
        for e in edges:
            pts = [to_px(*reversed(net.convertXY2LonLat(x, y))) for x, y in e.getShape()]
            col = (0, 90, 255) if e.getLaneNumber() >= 2 else (0, 170, 90)
            d.line(pts, fill=col, width=1 + e.getLaneNumber())
        for tl in net.getTrafficLights():
            for nd in {c[0].getEdge().getToNode() for c in tl.getConnections()}:
                x, y = to_px(*reversed(net.convertXY2LonLat(*nd.getCoord())))
                d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 140, 0), width=2)
        out_png = os.path.join(out_dir, f"{name}.overlay.png")
        img.save(out_png)
        rep["V9_overlay"] = out_png

    json.dump(rep, open(os.path.join(out_dir, f"{name}.check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print_report(rep)
    return rep


def print_report(rep):
    print("== V1 빨간 선 ↔ 도로망 거리 (구간별 중앙값/90%/최대, m)")
    for s in rep["V1_red_line"]:
        flag = "  ← 확인" if s["p90_m"] > 40 else ""
        print(f"   P{s['from']}→P{s['to']} ({s['len_m']}m): {s['median_m']} / {s['p90_m']} / {s['max_m']}{flag}")
    for k, v in rep["V2_ring_drive"].items():
        print(f"== V2 순환 주행 ({k}): 가능={v['ok']}, 도로 {v['edges']}개, 길이 {v['length_m']}m (순환 중심선 {v['ring_m']}m), "
              f"우회 {len(v['detours'])}곳, 끊김 {len(v['broken'])}곳")
        for d in v["detours"]:
            print("      우회:", d)
        for b in v["broken"]:
            print("      끊김:", b)
    print("== V3 영역 안 도로 길이 (OSM → 도로망, m)")
    for k, v in rep["V3_roads_inside"].items():
        r = v["net_m"] / v["osm_m"] if v["osm_m"] else 0
        print(f"   {k:16s} {v['osm_m']:5d} → {v['net_m']:5d}  ({r:.0%})")
    print("== V4 차로 수:", rep["V4_lanes"]["count"], "| 미확인 도로:", rep["V4_lanes"]["unknown_by_road"])
    for m in rep["V4_lanes"]["mismatch"][:10]:
        print("   불일치:", m)
    print("== V5 회전 제한:", rep["V5_restrictions"]["by_status"])
    v6 = rep["V6_connectivity"]
    print("== V6 연결성:", {k: v for k, v in v6.items() if k not in ("entries", "exits")},
          f"| 입구 {len(v6.get('entries', []))}개, 출구 {len(v6.get('exits', []))}개")
    print("== V7 일방통행 위반:", len(rep["V7_oneway_violations"]))
    print("== V8 신호:", rep["V8_signals"])
    print("== V10 막힌 차로:", len(rep.get("V10_dead_lanes", [])), rep.get("V10_dead_lanes", [])[:5])


if __name__ == "__main__":
    build(sys.argv[1])
