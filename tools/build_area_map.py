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


def apply_carriageway_merges(root, proj, nodes, ways, merges):
    """분리된 일방 차도 2개(상행·하행)를 중앙선 하나의 양방향 도로로 합친다 (merge_carriageways 설정).

    OSM은 중앙대로처럼 가운데 BRT 버스전용차로가 있는 도로를 '상행 차도 + 하행 차도 + busway'
    3개 선으로 그린다. busway를 빼면 SUMO에서 두 차도 사이가 비어 따로 떨어진 도로처럼 보이므로,
    두 차도의 가운데 선으로 양방향 도로 하나를 만들고 안쪽 차로를 버스 전용으로 둔다.

    설정 항목: start_node(합치기 시작 노드, 보통 교차로 중심), name(도로 이름), bearing(방향),
              max_m(시작 노드에서 이 거리까지), tags(새 도로 태그), new_id(새 way id)
    - 두 차도에 붙어 있던 다른 도로(연결로·교차 도로)와 신호 노드는 가운데 선 위로 옮기고,
      양쪽 차도의 붙는 점이 group_m 안에 있으면 하나의 교차로로 합친다.
    - 회전 제한(relation)의 from/to/via도 새 도로·새 노드로 바꾼다.
    """
    done = []
    way_el = {w.get("id"): w for w in root.iter("way")}
    node_ll = {n.get("id"): n for n in root.iter("node")}
    busways = {}
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if t.get("highway") == "busway":
            pts = [proj.en(float(node_ll[x.get("ref")].get("lat")), float(node_ll[x.get("ref")].get("lon")))
                   for x in w.iter("nd") if x.get("ref") in node_ll]
            busways[w.get("id")] = (t, pts)
    node_el = {n.get("id"): n for n in root.iter("node")}

    def to_ll(p):
        return (proj.lat0 + p[1] / proj.ky, proj.lon0 + p[0] / proj.kx)

    def chain(start, name, bearing, outward, max_m):
        """start에서 bearing 방향으로 이어지는 같은 이름의 일방 차도 way들 (outward=True: 바깥으로 가는 차도)."""
        out, cur, seen = [], start, set()
        c = nodes[start]
        for _ in range(50):
            cands = []
            for wid, (t, nds) in ways.items():
                if wid in seen or t.get("name") != name or t.get("oneway") not in ("yes", "1", "true"):
                    continue
                if (nds[0] if outward else nds[-1]) != cur:
                    continue
                far = nds[-1] if outward else nds[0]
                b = math.degrees(math.atan2(nodes[far][0] - c[0], nodes[far][1] - c[1])) % 360
                if abs((b - bearing + 180) % 360 - 180) > 35 or math.dist(nodes[far], c) <= math.dist(nodes[cur], c):
                    continue
                cands.append((abs((b - bearing + 180) % 360 - 180), wid, far))
            if not cands:
                break
            _, wid, far = min(cands)
            out.append(wid)
            seen.add(wid)
            cur = far
            if math.dist(nodes[cur], c) > max_m:
                break
        return out

    def polyline(wids, outward):
        pts = []
        for wid in wids:
            nds = ways[wid][1] if outward else ways[wid][1][::-1]
            pts += nds if not pts else nds[1:]
        return pts

    def project(p, pl):
        """p를 폴리라인 pl에 투영: (거리, 호 길이 s, 투영점)"""
        best, acc = None, 0.0
        for a, b in zip(pl, pl[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy
            t = 0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
            q = (a[0] + t * dx, a[1] + t * dy)
            d = math.dist(p, q)
            if best is None or d < best[0]:
                best = (d, acc + math.sqrt(L2) * t, q)
            acc += math.sqrt(L2)
        return best

    for k, m in enumerate(merges or []):
        start, name, bearing = str(m["start_node"]), m["name"], float(m["bearing"])
        orig_pos = dict(nodes)                       # 옮기기 전 위치 (없어진 도로의 방향 계산용)
        max_m = float(m.get("max_m", 900))
        fw = chain(start, name, bearing, True, max_m)
        bw = chain(start, name, bearing, False, max_m)
        if not fw or not bw:
            raise RuntimeError(f"merge_carriageways '{m.get('label', name)}': 차도를 찾지 못함 (정방향 {fw}, 역방향 {bw})")
        F = [nodes[n] for n in polyline(fw, True)]
        Bk = [nodes[n] for n in polyline(bw, False)]
        # 가운데 선: 정방향 차도를 10m 간격으로 찍고 역방향 차도에 투영한 점과의 중점 (두 차도가 겹치는 구간만)
        center = [nodes[start]]
        dens = densify(F, 10.0)
        for p in dens[1:]:
            d, _, q = project(p, Bk)
            if d > 45 or math.dist(p, nodes[start]) > max_m:
                break
            center.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
        L = sum(math.dist(a, b) for a, b in zip(center, center[1:]))
        # 두 차도의 붙는 점(다른 도로가 닿는 노드, 신호 노드)
        merged_ids = set(fw) | set(bw)
        att = []
        for wid in merged_ids:
            for n in ways[wid][1]:
                if n == start:
                    continue
                others = [w for w, (t, nds) in ways.items() if w not in merged_ids and n in nds]
                nt = ntags_of(node_el.get(n))
                sig = nt.get("highway") == "traffic_signals" or nt.get("crossing") == "traffic_signals"   # 신호 횡단보도 포함
                if not others and not sig:
                    continue
                d, s, q = project(nodes[n], center)
                if s >= L - 1 or d > 45:
                    continue                       # 합치는 구간 밖
                att.append((s, n, q, bool(others), sig))
        att.sort()
        groups = []
        group_m = float(m.get("group_m", 20))
        for a in att:
            if groups and a[0] - groups[-1][-1][0] <= group_m:
                groups[-1].append(a)
            else:
                groups.append([a])
        # 두 차도 사이(중앙분리대) 위에 있던 교차 도로의 노드도 교차점 하나로 흡수 (예: 거제천로가 BRT 선과 만나던 점)
        absorb_m = float(m.get("absorb_m", 12))
        in_att = {a[1] for a in att} | {start}
        for wid, (t, nds) in ways.items():
            if wid in merged_ids:
                continue
            for n in nds:
                if n in in_att:
                    continue
                d, s_, q = project(nodes[n], center)
                if d > absorb_m or s_ >= L - 1:
                    continue
                for g in groups:
                    if g[0][0] - 8 <= s_ <= g[-1][0] + 8:
                        g.append((s_, n, q, False, ntags_of(node_el.get(n)).get("highway") == "traffic_signals"))
                        in_att.add(n)
                        break
        rename = {}
        reps = set()
        one_side = {}                               # 한쪽 차도에만 붙어 있던 교차점 → "out"(바깥으로 가는 차도) / "in"
        fw_nodes = {n for w in fw for n in ways[w][1]}
        bw_nodes = {n for w in bw for n in ways[w][1]}
        new_nodes = []                              # (s, id)
        for g in groups:
            rep = next((a for a in g if a[3]), g[0])   # 다른 도로가 붙는 노드를 대표로
            s = sum(a[0] for a in g) / len(g)
            _, _, q = project(nodes[rep[1]], center)
            nid = rep[1]
            nodes[nid] = q
            lat, lon = to_ll(q)
            node_el[nid].set("lat", f"{lat:.7f}")
            node_el[nid].set("lon", f"{lon:.7f}")
            if any(a[4] for a in g) and ntags_of(node_el[nid]).get("highway") != "traffic_signals":
                ET.SubElement(node_el[nid], "tag", {"k": "highway", "v": "traffic_signals"})
            for a in g:
                if a[1] != nid:
                    rename[a[1]] = nid
            sides = {("out" if a[1] in fw_nodes else "in") for a in g if a[1] in fw_nodes or a[1] in bw_nodes}
            if len(sides) == 1 and any(a[3] for a in g):
                one_side[nid] = sides.pop()
            reps.add(nid)
            new_nodes.append((s, nid))
        # 모양 점 (붙는 점 사이 20m 간격)
        acc, last_s = 0.0, -1e9
        for a, b in zip(center, center[1:]):
            acc += math.dist(a, b)
            if acc - last_s >= 20 and all(abs(acc - s) > 8 for s, _ in new_nodes) and acc < L - 1:
                nid = str(990000000000 + 10000 * k + len(node_el))
                lat, lon = to_ll(b)
                el = ET.SubElement(root, "node", {"id": nid, "lat": f"{lat:.7f}", "lon": f"{lon:.7f}", "version": "1"})
                node_el[nid] = el
                nodes[nid] = b
                new_nodes.append((acc, nid))
                last_s = acc
        end_id = str(990000000000 + 10000 * k + 9999)
        lat, lon = to_ll(center[-1])
        node_el[end_id] = ET.SubElement(root, "node", {"id": end_id, "lat": f"{lat:.7f}", "lon": f"{lon:.7f}", "version": "1"})
        nodes[end_id] = center[-1]
        seq = [start] + [n for s, n in sorted(new_nodes)] + [end_id]
        s_of = {start: 0.0, end_id: L, **{n: s for s, n in new_nodes}}
        # 끊는 곳: 합친 차도를 from/to로 쓰는 회전 제한의 via 교차점만 (나머지는 netconvert가 알아서 나눈다;
        # 다 끊으면 순환 도로 밖으로 나가는 첫 조각이 짧아져 진입/진출 토막이 짧아진다)
        need = set()
        for rel in root.iter("relation"):
            mems = list(rel.iter("member"))
            if any(x.get("type") == "way" and x.get("ref") in merged_ids for x in mems):
                need |= {rename.get(x.get("ref"), x.get("ref")) for x in mems
                         if x.get("type") == "node" and x.get("role") == "via"}
        junc = {start, end_id} | (reps & need)
        new_id = str(m.get("new_id", 880000000 + k))
        tags = {"name": name, "highway": ways[fw[0]][0].get("highway", "secondary"), "oneway": "no"}
        # 일반 차로 = 각 차도의 OSM lanes (가장 흔한 값), 버스전용차로 = 가운데 선을 따라가는 OSM busway의 방향별 차로
        def common_lanes(wids):
            c = collections.Counter(ways[w][0].get("lanes") for w in wids if ways[w][0].get("lanes", "").isdigit())
            return int(c.most_common(1)[0][0]) if c else None
        gen_f, gen_b = common_lanes(fw), common_lanes(bw)
        bus_ev = []
        for bwid, (bt, bpts) in busways.items():
            # 합친 도로 가운데 선의 몇 %가 이 busway 옆(8 m 안)인가 (busway는 합친 구간보다 길 수 있음)
            cov = sum(1 for p in densify(center, 5.0) if dist_to_polyline(p, bpts) < 8) / max(len(densify(center, 5.0)), 1)
            if cov < 0.05:          # 합친 도로의 일부라도 따라가는 busway는 모두 근거로 (구간별 버스차로 판정에 씀)
                continue
            bl = int(bt["lanes"]) if bt.get("lanes", "").isdigit() else (1 if bt.get("oneway") in ("yes", "1") else 2)
            bus_ev.append({"busway": bwid, "lanes": bl, "oneway": bt.get("oneway", "no"), "coverage": round(cov, 2)})
        bus_f = bus_b = 0
        if bus_ev:
            best = max(bus_ev, key=lambda b: b["coverage"])
            if best["oneway"] in ("yes", "1"):
                raise RuntimeError(f"{m.get('label')}: 일방 busway({best['busway']})는 아직 지원하지 않음")
            bus_f = bus_b = best["lanes"] // 2 if best["lanes"] >= 2 else 1
        if gen_f is None or gen_b is None:
            raise RuntimeError(f"{m.get('label')}: 차도의 OSM lanes 태그가 없어 차로 수를 정할 수 없음")
        tags.update({"lanes": str(gen_f + bus_f + gen_b + bus_b), "lanes:forward": str(gen_f + bus_f),
                     "lanes:backward": str(gen_b + bus_b)})
        if bus_f:   # 중앙 BRT: 방향마다 맨 왼쪽(안쪽) 차로가 버스전용 (OSM 순서 = 왼쪽부터)
            for d, g, b_ in (("forward", gen_f, bus_f), ("backward", gen_b, bus_b)):
                tags[f"bus:lanes:{d}"] = "|".join(["designated"] * b_ + ["yes"] * g)
                tags[f"access:lanes:{d}"] = "|".join(["no"] * b_ + ["yes"] * g)
        tags.update({kk: str(v) for kk, v in (m.get("tags") or {}).items()})       # 설정에서 직접 덮어쓰기 (선택)

        # 구간별 차로: 차도의 OSM lanes가 중간에 바뀌거나 busway가 끝나면 그 지점에서 끊고 조각마다 따로 정한다
        def spans(wids):
            out = []
            for w in wids:
                nds_ = ways[w][1]
                a_, b_ = project(orig_pos[nds_[0]], center)[1], project(orig_pos[nds_[-1]], center)[1]
                L_ = ways[w][0].get("lanes", "")
                out.append((min(a_, b_), max(a_, b_), int(L_) if L_.isdigit() else None))
            return out
        sp_f, sp_b = spans(fw), spans(bw)
        bus_lines = [busways[b["busway"]][1] for b in bus_ev]
        cpts = densify(center, 5.0)
        cum = [0.0]
        for a_, b_ in zip(cpts, cpts[1:]):
            cum.append(cum[-1] + math.dist(a_, b_))

        def at(s_):
            def pick(sp, dflt):
                hit = [x for x in sp if x[0] - 1 <= s_ <= x[1] + 1 and x[2]]
                return hit[0][2] if hit else dflt
            i_ = min(range(len(cum)), key=lambda j: abs(cum[j] - s_))
            on_bus = any(dist_to_polyline(cpts[i_], bl) < 8 for bl in bus_lines)
            return (pick(sp_f, gen_f), pick(sp_b, gen_b), bus_f if on_bus else 0, bus_b if on_bus else 0)
        prof = [(x, at(x)) for x in [i_ * 10.0 for i_ in range(int(L // 10) + 1)]]
        cuts = [prof[i_][0] for i_ in range(1, len(prof)) if prof[i_][1] != prof[i_ - 1][1]]
        inner_seq = [n for n in seq[1:-1]]
        for c_ in cuts:
            if inner_seq:
                junc.add(min(inner_seq, key=lambda n: abs(s_of[n] - c_)))

        def seg_tags(nds_):
            gf, gb, bf, bb = at((s_of[nds_[0]] + s_of[nds_[-1]]) / 2)
            t_ = dict(tags)
            t_.update({"lanes": str(gf + bf + gb + bb), "lanes:forward": str(gf + bf), "lanes:backward": str(gb + bb)})
            for k_ in [k_ for k_ in t_ if k_.startswith(("bus:lanes", "access:lanes"))]:
                del t_[k_]
            for d_, g_, b_ in (("forward", gf, bf), ("backward", gb, bb)):
                if b_:
                    t_[f"bus:lanes:{d_}"] = "|".join(["designated"] * b_ + ["yes"] * g_)
                    t_[f"access:lanes:{d_}"] = "|".join(["no"] * b_ + ["yes"] * g_)
            t_.update({kk: str(v) for kk, v in (m.get("tags") or {}).items()})
            return t_, (gf, gb, bf, bb)
        # 교차점마다 끊어 여러 way로 (회전 제한의 from/to가 어느 쪽에서 오는지 정확히 가리키도록)
        segs, cur = [], [seq[0]]
        for n in seq[1:]:
            cur.append(n)
            if n in junc:
                segs.append(cur)
                cur = [n]
        seg_ids, seg_lanes = [], {}
        for i, nds in enumerate(segs):
            sid = f"{new_id}{i:02d}"
            nw = ET.SubElement(root, "way", {"id": sid, "version": "1"})
            for n in nds:
                ET.SubElement(nw, "nd", {"ref": n})
            t_seg, seg_lanes[sid] = seg_tags(nds)
            for kk, v in t_seg.items():
                ET.SubElement(nw, "tag", {"k": kk, "v": v})
            seg_ids.append((sid, nds))
        merged_nds = {wid: ways[wid][1] for wid in merged_ids}

        def seg_at(via, toward_s):
            """via 교차점에 붙은 합친 도로 조각 중 toward_s 쪽 조각"""
            for sid, nds in seg_ids:
                if via in (nds[0], nds[-1]):
                    other = nds[-1] if nds[0] == via else nds[0]
                    if (s_of[other] < s_of[via]) == (toward_s < s_of[via]):
                        return sid
            return None
        # 원래 차도 제거, 다른 도로의 노드 참조 바꾸기
        for wid in merged_ids:
            if wid in way_el:
                root.remove(way_el[wid])
            ways.pop(wid, None)
        old_ends = {wid: (nds[0], nds[-1]) for wid, (t, nds) in ways.items()}
        ways_tags_dropped = {wid: t for wid, (t, nds) in ways.items()}
        old_pos = orig_pos
        dropped = {}                                # 길이가 0이 된 도로 → (원래 시작, 원래 끝)
        for w in list(root.iter("way")):
            refs = []
            for nd in list(w.iter("nd")):
                r_ = rename.get(nd.get("ref"), nd.get("ref"))
                if refs and refs[-1] == r_:
                    w.remove(nd)
                    continue
                nd.set("ref", r_)
                refs.append(r_)
            if len(refs) < 2 and w.get("id") in ways:
                dropped[w.get("id")] = old_ends[w.get("id")]
                root.remove(w)
        for wid, (t, nds) in list(ways.items()):
            if wid in dropped:
                ways.pop(wid)
            elif any(n in rename for n in nds):
                nn = []
                for n in nds:
                    n = rename.get(n, n)
                    if not nn or nn[-1] != n:
                        nn.append(n)
                ways[wid] = (t, nn)
        for sid, nds in seg_ids:
            ways[sid] = (seg_tags(nds)[0], nds)

        def heading(a, b):
            return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360

        def successor(wid, role, via):
            """길이 0이 되어 없어진 도로 대신, 같은 방향으로 이어지는 같은 이름의 도로"""
            a, b = dropped[wid]
            want = heading(old_pos[a], old_pos[b])          # to: via에서 멀어지는 방향, from: via로 오는 방향
            name = ways_tags_dropped.get(wid, {}).get("name")
            best = None
            for w2, (t2, nds2) in ways.items():
                if w2.startswith(new_id) or via not in nds2 or (name and t2.get("name") != name):
                    continue
                i = nds2.index(via)
                for j in (i - 1, i + 1):
                    if 0 <= j < len(nds2):
                        h = heading(nodes[via], nodes[nds2[j]]) if role == "to" else heading(nodes[nds2[j]], nodes[via])
                        diff = abs((h - want + 180) % 360 - 180)
                        if best is None or diff < best[0]:
                            best = (diff, w2)
            return best[1] if best and best[0] < 60 else None

        # 회전 제한: from/to가 합친 차도면 새 도로로, via 노드 이름 바꾸기, 없어진 도로는 이어지는 도로로
        removed_rels = []
        for rel in list(root.iter("relation")):
            mems = list(rel.iter("member"))
            wmem = [x for x in mems if x.get("type") == "way"]
            if not any(x.get("ref") in merged_ids or x.get("ref") in dropped for x in wmem):
                for mem in mems:
                    if mem.get("type") == "node" and mem.get("ref") in rename:
                        mem.set("ref", rename[mem.get("ref")])
                continue
            via = next((rename.get(x.get("ref"), x.get("ref")) for x in mems if x.get("role") == "via" and x.get("type") == "node"), None)
            fr = next((x.get("ref") for x in wmem if x.get("role") == "from"), None)
            to = next((x.get("ref") for x in wmem if x.get("role") == "to"), None)
            # 유턴 제한(같은 도로, 또는 합친 두 차도 사이) → 도로망은 유턴이 없으므로 의미 없음: 뺀다
            if fr == to or (fr in merged_ids and to in merged_ids) or via is None or via not in s_of and any(
                    x.get("ref") in merged_ids for x in wmem):
                root.remove(rel)
                removed_rels.append(rel.get("id"))
                continue
            ok = True
            for mem in wmem:
                ref = mem.get("ref")
                if ref in merged_ids:
                    # 원래 차도의 via 반대쪽 끝이 가운데 선의 어느 쪽인지 → 그쪽 조각
                    nds = merged_nds[ref]
                    far = nds[-1] if rename.get(nds[0], nds[0]) == via else nds[0]
                    sid = seg_at(via, project(orig_pos[far], center)[1])
                    ok &= sid is not None
                    mem.set("ref", sid or ref)
                elif ref in dropped:
                    rep_w = successor(ref, mem.get("role"), via)
                    ok &= rep_w is not None
                    mem.set("ref", rep_w or ref)
            for mem in mems:
                if mem.get("type") == "node" and mem.get("ref") in rename:
                    mem.set("ref", rename[mem.get("ref")])
            if not ok:
                root.remove(rel)
                removed_rels.append(rel.get("id"))
        done.append({"label": m.get("label", name), "new_id": new_id, "merged": sorted(merged_ids),
                     "absorbed_nodes": sorted(rename), "rename": dict(rename), "dropped_ways": sorted(dropped),
                     "one_side_nodes": one_side,
                     "lanes": {"general_per_dir": [gen_f, gen_b], "bus_per_dir": [bus_f, bus_b], "busway_evidence": bus_ev,
                               "segments": {k_: list(v) for k_, v in seg_lanes.items()},
                               "segment_src": {"정방향 차도(나감)": [[round(a_), round(b_), L_] for a_, b_, L_ in sp_f],
                                               "역방향 차도(들어옴)": [[round(a_), round(b_), L_] for a_, b_, L_ in sp_b]}},
                     "segments": [sid for sid, _ in seg_ids], "removed_restrictions": removed_rels,
                     "length_m": round(L), "junctions": len(groups)})
    return done


def median_deletes(net, merges_done):
    """합친 도로에서 원래 한쪽 차도에만 붙어 있던 교차점(골목길·연결로)은 가운데 분리대(BRT)를 건너는 연결을 지운다.
    "out" 쪽(합친 way 방향 = 시작 교차로에서 멀어지는 쪽, 엣지 id에 '-' 없음)에 붙었던 도로는
    '-' 엣지(들어오는 방향)와 서로 오갈 수 없다 — 실제로는 분리대 때문에 우회전으로만 드나든다.
    반환: [(from, to, fromLane, toLane)] (<delete>용)"""
    out = []
    for md in merges_done:
        pre = md["new_id"]
        for rep, side in md.get("one_side_nodes", {}).items():
            nd = next((n for n in net.getNodes() if n.getID() == rep or rep in n.getID().split("_")), None)
            if nd is None:
                continue
            def merged(e):
                return e.getID().lstrip("-").startswith(pre)
            def far_dir(e):          # 분리대 건너편 방향의 합친 도로 엣지인가
                return merged(e) and (e.getID().startswith("-") == (side == "out"))
            for e in nd.getIncoming():
                for to, cl in e.getOutgoing().items():
                    if to.getToNode() is None:
                        continue
                    bad = (not merged(e) and far_dir(to)) or (far_dir(e) and not merged(to))
                    if bad:
                        for c in cl:
                            out.append((e.getID(), to.getID(), c.getFromLane().getIndex(), c.getToLane().getIndex()))
    return sorted(set(out))


def ntags_of(el):
    return {} if el is None else {t.get("k"): t.get("v") for t in el.iter("tag")}


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
             "도로망_정방향", "도로망_역방향", "도로망_버스전용", "실제_정방향", "실제_역방향", "turn_정방향", "turn_역방향",
             "버스전용_정방향", "버스전용_역방향", "메모"]
SURVEY_COLS = ("실제_정방향", "실제_역방향", "turn_정방향", "turn_역방향", "버스전용_정방향", "버스전용_역방향")


def read_lane_table(path):
    import csv
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vals = {k: (row.get(k) or "").strip() for k in (*SURVEY_COLS, "메모")}
            if any(vals[k] for k in SURVEY_COLS):
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


def write_lane_table(path, net_f, ways, sel, why, overrides, osm_tags=None):
    import csv
    import sumolib
    net = sumolib.net.readNet(net_f)
    lanes_net = collections.defaultdict(lambda: [None, None])
    bus_net = collections.defaultdict(set)
    length = collections.Counter()
    for e in net.getEdges():
        wid = e.getID().lstrip("-").split("#")[0]
        lanes_net[wid][1 if e.getID().startswith("-") else 0] = e.getLaneNumber()
        for ln in e.getLanes():
            if ln.allows("bus") and not ln.allows("passenger"):
                side = "역" if e.getID().startswith("-") else "정"
                pos = "왼쪽" if ln.getIndex() == e.getLaneNumber() - 1 else ("오른쪽" if ln.getIndex() == 0 else f"{ln.getIndex()}번")
                bus_net[wid].add(f"{side}:{pos}")
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
        t = (osm_tags or {}).get(wid) or ways[wid][0]
        f, b = osm_lanes_expect(t)
        o = overrides.get(wid, {})
        rows.append({"way_id": wid, "도로명": t.get("name", ""), "등급": t.get("highway"),
                     "일방": "예" if t.get("oneway") in ("yes", "1", "true") or t.get("junction") else "",
                     "길이_m": round(length[wid]) if length[wid] else "", "OSM_lanes": t.get("lanes", ""),
                     "OSM_정방향": "" if f is None else f, "OSM_역방향": "" if b is None else b,
                     "도로망_정방향": lanes_net[wid][0] or "", "도로망_역방향": lanes_net[wid][1] or "",
                     "도로망_버스전용": " ".join(sorted(bus_net.get(wid, []))),
                     "버스전용_정방향": o.get("버스전용_정방향", ""), "버스전용_역방향": o.get("버스전용_역방향", ""),
                     "실제_정방향": "" if o.get("_assumed") else o.get("실제_정방향", ""), "실제_역방향": o.get("실제_역방향", ""),
                     "turn_정방향": o.get("turn_정방향", ""), "turn_역방향": o.get("turn_역방향", ""),
                     "메모": "; ".join(x for x in [o.get("메모", ""), (f"가정 {o.get('실제_정방향')}차로: {o['_assumed']}"
                                                          if o.get("_assumed") else ""), *dead.get(wid, [])] if x)})
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
    merges_done = apply_carriageway_merges(root, proj, nodes, ways, cfg.get("merge_carriageways"))
    for md in merges_done:
        print(f"[merge] {md['label']}: 차도 {len(md['merged'])}개 → 양방향 도로 {md['new_id']} ({md['length_m']}m, "
              f"교차점 {md['junctions']}곳, 조각 {len(md['segments'])}개, 뺀 유턴 제한 {len(md['removed_restrictions'])}건)")
    gone = {rid for md in merges_done for rid in md["removed_restrictions"]}
    rels = [(r, t) for r, t in rels if r.get("id") not in gone]
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
    osm_tags = {w: dict(t) for w, (t, _) in ways.items()}      # 조사표·가정·버스 정리 전의 OSM 원래 태그 (조사표 OSM_* 칸)
    # 차로 수 가정(설정 lane_assumptions: OSM에 없고 설계 기준으로 정한 값). 조사표에 실제 값이 있으면 그쪽이 우선.
    for wid, a in (cfg.get("lane_assumptions") or {}).items():
        if wid in ways and not (overrides.get(wid) or {}).get("실제_정방향"):
            overrides.setdefault(wid, {k: "" for k in (*SURVEY_COLS, "메모")})
            overrides[wid] = {**overrides[wid], "실제_정방향": str(a["lanes"]), "_assumed": a.get("reason", "")}
    apply_overrides(root, ways, overrides)
    # 버스전용차로: 모든 도로에 같은 규칙 (env/bus_lanes.py) — OSM 표기 정리 + 조사표 버스전용 칸
    from env.bus_lanes import normalize_bus_tags, parse_survey
    bus_changed = {}
    for w in root.iter("way"):
        wid = w.get("id")
        if wid not in ways:
            continue
        o = overrides.get(wid) or {}
        survey = {"forward": parse_survey(o.get("버스전용_정방향")), "backward": parse_survey(o.get("버스전용_역방향"))}
        new = normalize_bus_tags(ways[wid][0], survey)
        if not new:
            continue
        tag_el = {x.get("k"): x for x in w.iter("tag")}
        for k, v in new.items():
            if v is None:
                if k in tag_el:
                    w.remove(tag_el[k])
                ways[wid][0].pop(k, None)
            else:
                (tag_el[k].set("v", v) if k in tag_el else ET.SubElement(w, "tag", {"k": k, "v": v}))
                ways[wid][0][k] = v
        bus_changed[wid] = new

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
    # 신호 위상 배치: "incoming" = 진입 도로마다 차례로 녹색(직진+좌회전 동시, 보호 좌회전) — 한국 교차로에 흔한 방식.
    # netconvert 기본 "opposites"는 마주 보는 두 도로를 함께 녹색으로 두어 좌회전이 비보호(교차로 안에서 대기)가 된다.
    tls_layout = cfg.get("tls_layout", "opposites")
    # 1) OSM → 도로망 (신호는 OSM 신호 노드만: --tls.guess-signals)
    res = subprocess.run(
        [sumolib.checkBinary("netconvert"), "--osm-files", osm_f, "--type-files", types_f, *con_args,
         "--geometry.remove", "--ramps.guess", "--junctions.join",
         "--tls.guess-signals", "--tls.join", "--tls.default-type", "static", "--tls.layout", tls_layout,
         "--remove-edges.isolated", "--keep-edges.by-vclass", "passenger",
         "--no-turnarounds", "true", "--osm.lane-access", "true", "--output.street-names",     # 유턴 없음 (경계 토막 끝 포함)
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
        # OSM 신호 중 아직 신호가 아닌 곳 (교차로에서 먼 신호 횡단보도 등, env/signals.py)
        from env.signals import osm_signal_nodes, missing_osm_signals
        rename_all = {}
        for md in merges_done:
            rename_all.update(md.get("rename", {}))
        sig_add, sig_skip = missing_osm_signals(net0, osm_signal_nodes(osm_f), rename=rename_all)
        tls_set += [n for n in sig_add if n not in tls_set]
        print(f"[signals] 가정 신호 {len(tls_set) - len(sig_add)}곳 + 빠졌던 OSM 신호 {sig_add} | 넣지 않은 OSM 신호 {sig_skip}")
    if tls_set:
        # --tls.join: 가까운 신호들(예: 연산교차로를 이루는 노드 4개)을 하나의 신호로 묶는다
        subprocess.run([sumolib.checkBinary("netconvert"), "-s", raw0_f, "--tls.set", ",".join(tls_set),
                        "--tls.join", "--tls.default-type", "static", "--tls.layout", tls_layout,
                        "--tls.allred.time", str(int(cfg.get("tls_allred", 2))),     # 황색 뒤 전적색
                        "--output.street-names", "-o", raw_f],
                       check=True, capture_output=True)
    else:
        raw_f = raw0_f

    # 합친 도로: 한쪽 차도에만 붙어 있던 교차점에서 가운데 분리대(BRT)를 건너는 연결 지우기
    med_del = median_deletes(sumolib.net.readNet(raw_f), merges_done)
    if med_del:
        md_f = os.path.join(tmp, "median.con.xml")
        with open(md_f, "w") as f:
            f.write("<connections>\n" + "".join(
                f'    <delete from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n' for a, b, fl, tl in med_del)
                    + "</connections>\n")
        med_f = os.path.join(tmp, "raw_median.net.xml")
        subprocess.run([sumolib.checkBinary("netconvert"), "-s", raw_f, "--connection-files", md_f,
                        "--output.street-names", "-o", med_f], check=True, capture_output=True)
        raw_f = med_f
    print(f"[median] 분리대를 건너는 연결 지움 {len(med_del)}개: {med_del[:6]}")

    # 갈 곳 없는 차로(차로 수가 줄어드는 곳, 예: 2차로 진입로 → 1차로 회전교차로) → 옆 차로와 같은 곳으로 합류.
    # netconvert는 차로가 줄면 남는 차로의 연결을 만들지 않아 그 차로 끝에서 차가 멈춘다.
    merge_cons = []
    rnet = sumolib.net.readNet(raw_f)
    for e in rnet.getEdges():
        lanes = [ln for ln in e.getLanes() if ln.allows("passenger")]
        if not e.getOutgoing() or not lanes:
            continue
        for ln in lanes:
            if ln.getOutgoing():
                continue
            # 가장 가까운 연결 있는 차로의 연결 중, 그 차로 쪽으로 가장 가까운 진출 차로로
            nb = sorted((l for l in lanes if l.getOutgoing()), key=lambda l: abs(l.getIndex() - ln.getIndex()))
            if not nb:
                continue
            src = nb[0]
            for c in src.getOutgoing():
                if src.getIndex() < ln.getIndex():
                    pick = max(c2.getToLane().getIndex() for c2 in src.getOutgoing() if c2.getTo() == c.getTo())
                else:
                    pick = min(c2.getToLane().getIndex() for c2 in src.getOutgoing() if c2.getTo() == c.getTo())
                if c.getToLane().getIndex() != pick:
                    continue
                merge_cons.append((e.getID(), c.getTo().getID(), ln.getIndex(), pick))
    merge_cons = sorted(set(merge_cons))
    if merge_cons:
        mc_f = os.path.join(tmp, "lane_merge.con.xml")
        with open(mc_f, "w") as f:
            f.write("<connections>\n" + "".join(
                f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n' for a, b, fl, tl in merge_cons)
                    + "</connections>\n")
        merged_f = os.path.join(tmp, "raw_merge.net.xml")
        subprocess.run([sumolib.checkBinary("netconvert"), "-s", raw_f, "--connection-files", mc_f,
                        "--output.street-names", "-o", merged_f], check=True, capture_output=True)
        raw_f = merged_f
    print(f"[lane-merge] 차로가 줄어드는 곳 합류 연결 {len(merge_cons)}개: {merge_cons[:8]}")
    # BRT: 버스전용차로 다음에 버스전용차로가 이어지면 그 직진 연결만 남김 (env/bus_lanes.py)
    from env.bus_lanes import brt_through_deletes
    brt_del = brt_through_deletes(sumolib.net.readNet(raw_f))
    if brt_del:
        bd_f = os.path.join(tmp, "brt.con.xml")
        with open(bd_f, "w") as f:
            f.write("<connections>\n" + "".join(
                f'    <delete from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n' for a, b, fl, tl in brt_del)
                    + "</connections>\n")
        brt_f = os.path.join(tmp, "raw_brt.net.xml")
        subprocess.run([sumolib.checkBinary("netconvert"), "-s", raw_f, "--connection-files", bd_f,
                        "--output.street-names", "-o", brt_f], check=True, capture_output=True)
        raw_f = brt_f
    print(f"[brt] 버스전용차로 → 버스전용차로 직진만 남기고 지운 연결 {len(brt_del)}개")

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
            stubs[e.getID()] = (b, stub_m)            # 바깥 끝 노드, 이 도로에서 남길 길이
        elif b in core_nodes and a not in core_nodes:
            stubs[e.getID()] = (a, stub_m)
    # 첫 바깥 도로가 stub_m보다 짧으면(곁길이 붙어 끊긴 경우) 같은 이름으로 이어지는 도로까지 붙여 stub_m를 채운다
    for eid in list(stubs):
        if eid not in stubs:
            continue
        outer, _ = stubs[eid]
        e = net.getEdge(eid)
        outward = e.getToNode().getID() == outer
        acc, cur = e.getLength(), e
        while acc < stub_m:
            o = net.getNode(outer)
            back = cur.getFromNode().getID() if outward else cur.getToNode().getID()
            cands = [x for x in (o.getOutgoing() if outward else o.getIncoming())
                     if x.getName() == e.getName() and x.getID() not in keep
                     and (x.getToNode() if outward else x.getFromNode()).getID() != back]
            if len(cands) != 1:
                break
            keep.add(cur.getID())
            del stubs[cur.getID()]
            cur = cands[0]
            outer = (cur.getToNode() if outward else cur.getFromNode()).getID()
            stubs[cur.getID()] = (outer, stub_m - acc)
            acc += cur.getLength()
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
    edg_all = list(edg_t.getroot().iter("edge"))

    def node_xy(n):
        return (float(nod_by_id[n].get("x")), float(nod_by_id[n].get("y")))

    def centerline(ed):
        if ed.get("shape"):
            pts = [tuple(map(float, p.split(",")[:2])) for p in ed.get("shape").split()]
        else:
            pts = [node_xy(ed.get("from")), node_xy(ed.get("to"))]
        return pts
    for eid, (outer, want) in stubs.items():
        if not cnet.hasEdge(eid) or outer in moved or outer not in nod_by_id:
            continue
        ce = cnet.getEdge(eid)
        core = ce.getFromNode().getID() if ce.getToNode().getID() == outer else ce.getToNode().getID()
        E = [ed for ed in edg_all if {ed.get("from"), ed.get("to")} == {core, outer}]
        if not E or core not in nod_by_id:
            continue
        pl = centerline(E[0])
        if E[0].get("from") != core:
            pl = pl[::-1]                                   # 순환 쪽 → 바깥 끝
        acc, cut, keep_pts = 0.0, None, [pl[0]]
        for p_, q in zip(pl, pl[1:]):
            seg = math.dist(p_, q)
            if acc + seg >= want:
                f = (want - acc) / seg
                cut = (p_[0] + (q[0] - p_[0]) * f, p_[1] + (q[1] - p_[1]) * f)
                keep_pts.append(cut)
                break
            acc += seg
            keep_pts.append(q)
        if cut is None:
            continue                                   # 이미 stub_m보다 짧음
        nod_by_id[outer].set("x", f"{cut[0]:.2f}")
        nod_by_id[outer].set("y", f"{cut[1]:.2f}")
        moved[outer] = cut
        # OSM 모양 그대로 잘라 붙인다 (직선으로 다시 그리지 않음)
        for ed in E:
            pts = keep_pts if ed.get("from") == core else keep_pts[::-1]
            ed.set("shape", " ".join(f"{x:.2f},{y:.2f}" for x, y in pts))
            for ln in ed.iter("lane"):
                ln.attrib.pop("shape", None)
    if moved:
        nod_t.write(f"{plain}.nod.xml", encoding="UTF-8", xml_declaration=True)
        edg_t.write(f"{plain}.edg.xml", encoding="UTF-8", xml_declaration=True)
    _run(_plain_args(_files(plain)) + ["--output.street-names", "-o", net_f])
    # 단일로 신호(신호 횡단보도 등): 양방향 함께 녹색 → 황색 → 보행 적색 (env/signals.py)
    from env.signals import fix_midblock_programs, fix_joined_programs, korean_phase_programs
    print(f"[signals] 축별 직진 → 보호 좌회전 현시: {korean_phase_programs(net_f)}")
    from env.signals import brt_split_programs
    print(f"[signals] BRT 구간 분리: {brt_split_programs(net_f)}")
    midblock = fix_midblock_programs(net_f)
    print(f"[signals] 묶인 신호의 내부 정지선 항상 녹색: {fix_joined_programs(net_f)}")
    # 연산교차로 정밀 모델 (차로 조사 자료, tools/build_x6.py)
    x6_rep = None
    if cfg.get("x6_spec"):
        from tools.build_x6 import apply_x6, check_x6, print_check
        spec_path = os.path.join(BASE, cfg["x6_spec"])
        x6_rep = apply_x6(net_f, net_f, spec_path, workdir=os.path.join(tmp, "x6"))
        chk = check_x6(net_f, spec_path)
        print_check(chk)
        if chk["problems"]:
            raise RuntimeError(f"연산교차로 정밀 모델이 자료와 다름: {chk['problems']}")
        x6_rep = {"center_signal": x6_rep["center_signal"], "warnings": x6_rep.get("warnings", []),
                  "stop_line_r": x6_rep.get("stop_line_r"),
                  "lanes_checked": chk["lanes_checked"], "ok": chk["ok"], "rows": chk["rows"]}
    print(f"[signals] 단일로 신호 현시 고침: {midblock}")
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

    write_lane_table(lane_csv, net_f, ways, sel, why, overrides, osm_tags)
    meta = {"name": name, "ring_ways": ring_ways, "selected_ways": {w: why[w] for w in sorted(sel)},
            "lane_overrides": len(overrides), "carriageway_merges": merges_done,
            "lane_merge_connections": merge_cons, "bus_tag_changes": bus_changed, "brt_deleted_connections": brt_del,
            "median_deleted_connections": med_del, "x6": x6_rep,
            "lane_assumptions": cfg.get("lane_assumptions", {}),
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

    # V11 꼭 넣어야 하는 큰 대로 (radial_roads): 중심 교차로에서 순환 도로까지 양방향으로 끊김 없이
    #     이어지는지. 예) 연산교차로 6개 방향 대로
    v11 = []
    rr = cfg.get("radial_roads") or []
    if rr and net.hasNode(str(cfg.get("center_node", ""))):
        cx, cy = net.getNode(str(cfg["center_node"])).getCoord()

        def bear(x, y):
            return math.degrees(math.atan2(x - cx, y - cy)) % 360

        for road in rr:
            sector = []
            for e in edges:
                if e.getName() != road["name"]:
                    continue
                sh = e.getShape()
                mid = sh[len(sh) // 2]
                if abs((bear(*mid) - road["bearing"] + 180) % 360 - 180) > float(road.get("tol", 25)):
                    continue
                d0, d1 = math.dist(sh[0], (cx, cy)), math.dist(sh[-1], (cx, cy))
                sector.append((e, d0, d1))
            res = {"id": road["id"], "name": road["name"], "bearing": road["bearing"]}
            for label, out in (("나가는 방향", True), ("들어오는 방향", False)):
                es = [(e, d0, d1) for e, d0, d1 in sector if (d1 > d0) == out]
                if not es:
                    res[label] = "없음"
                    continue
                near = min(es, key=lambda x: min(x[1], x[2]))[0]
                far = max(es, key=lambda x: max(x[1], x[2]))[0]
                a, b = (near, far) if out else (far, near)
                path, c = net.getShortestPath(a, b, vClass="passenger")
                straight = math.dist(a.getShape()[0], b.getShape()[-1])
                lanes = sorted({e.getLaneNumber() for e in (path or [])})
                ok = bool(path) and c <= straight * 1.3 + 100
                res[label] = {"ok": ok, "길이_m": round(c) if path else None,
                              "차로": f"{lanes[0]}~{lanes[-1]}" if lanes else None,
                              "다른_도로_경유": sorted({e.getName() or "(연결로)" for e in (path or [])} - {road["name"]})}
            res["ok"] = all(isinstance(res[k], dict) and res[k]["ok"] for k in ("나가는 방향", "들어오는 방향"))
            v11.append(res)
    rep["V11_radial_roads"] = v11

    # V12 버스전용차로: OSM(정리된 태그)의 버스전용 차로 위치 ↔ 도로망 차로 권한(allow=bus), 버스 연결 끊김
    from env.bus_lanes import bus_lane_summary
    v12 = {"edges_checked": 0, "match": 0, "mismatch": [], "bus_lanes": 0}
    for e in edges:
        wid = e.getID().lstrip("-").split("#")[0]
        if wid not in ways:
            continue
        t = ways[wid][0]
        oneway = t.get("oneway") in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular")
        key = "bus:lanes" if oneway else ("bus:lanes:backward" if e.getID().startswith("-") else "bus:lanes:forward")
        pv = (t.get(key) or t.get(key.replace("bus:", "psv:")) or "").split("|")
        n = e.getLaneNumber()
        # OSM 순서(왼쪽부터) → SUMO 차로 번호(오른쪽부터 0)
        want = sorted(n - 1 - i for i, v in enumerate(pv) if v == "designated") if len(pv) == n else []
        got = sorted(ln.getIndex() for ln in e.getLanes() if ln.allows("bus") and not ln.allows("passenger"))
        v12["edges_checked"] += 1
        v12["bus_lanes"] += len(got)
        if want == got:
            v12["match"] += 1
        else:
            v12["mismatch"].append({"edge": e.getID(), "name": e.getName(), "osm": want, "net": got})
    summ = bus_lane_summary(net)
    v12["problems"] = summ["problems"]
    v12["by_road"] = dict(collections.Counter(r["name"] or "(이름없음)" for r in summ["lanes"]))
    rep["V12_bus_lanes"] = v12

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
    v11 = rep.get("V11_radial_roads", [])
    if v11:
        print(f"== V11 반드시 포함할 큰 대로: {sum(r['ok'] for r in v11)}/{len(v11)} 통과")
        for r in v11:
            o, i = r["나가는 방향"], r["들어오는 방향"]
            fmt = lambda x: x if isinstance(x, str) else f"{'OK' if x['ok'] else '문제'} {x['길이_m']}m {x['차로']}차로"
            print(f"   {r['id']:2s} {r['name']:6s} ({r['bearing']}°) 나감: {fmt(o)} | 들어옴: {fmt(i)}")
    v12 = rep.get("V12_bus_lanes")
    if v12:
        print(f"== V12 버스전용차로: 도로망 버스전용 차로 {v12['bus_lanes']}개 (도로별 {v12['by_road']}) | "
              f"OSM과 위치 일치 {v12['match']}/{v12['edges_checked']} 엣지, 불일치 {len(v12['mismatch'])}, 문제 {len(v12['problems'])}")
        for x in v12["mismatch"][:10] + v12["problems"][:10]:
            print("     ", x)


if __name__ == "__main__":
    build(sys.argv[1])
