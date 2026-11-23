"""OpenStreetMap 데이터로 교차로 주변 도로망 전체를 만드는 생성기.

road_config.ROAD["osm"]["enabled"]가 True면 road_builder.build()가
이 모듈의 build_osm()을 호출한다.

[흐름]
    env/osm/*.osm (OpenStreetMap에서 받은 원본, 저장소에 포함)
        │  netconvert : 도로·차로·연결·신호 → highway.net.xml
        │  polyconvert: 건물 → highway.poly.xml (GUI 배경, 시뮬레이션과 무관)
        ▼
    도로 끝점(fringe) 탐색
        · ROAD["legs"]의 6개 간선: 도로 이름 + 교차점 기준 방위각으로 찾음
        · 나머지 끝점: 골목길 입구 (작은 교통량)
        ▼
    경로(최단경로)와 flow 생성 → highway.rou.xml
    환경용 정보 → highway.meta.json

[OSM 데이터 갱신 방법]
    Overpass API(https://overpass-api.de)에서 같은 범위를 다시 받아
    env/osm/yeonsan.osm을 덮어쓰면 된다. 받는 범위는 ROAD["osm"]["bbox"].
"""
import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET

from sumolib import checkBinary
import sumolib

from env.mdp_config import SIMULATION
from env import road_details

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OSM 도로 등급별 기본값 (OSM에 차로 수/제한속도 태그가 없을 때만 쓰임).
# 도심 안전속도 5030: 간선 50 km/h, 이면도로 30 km/h.
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
  <type id="highway.living_street"  priority="3"  numLanes="1" speed="5.56"  oneway="false"/>
</types>
"""


def _bearing(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def apply_lane_rules(net_path: str, road: dict, center_id: str, rules: dict) -> dict:
    """LANE_RULES 표대로 교차로의 "차로 → 진출 도로" 연결만 남기고 나머지를 지운다.

    연산교차로는 SUMO에서 노드 여러 개(신호 하나로 묶임)와 그 사이의 짧은
    내부 도로로 이루어져 있어서, 진입 차로에서 진출 도로까지 연결을 여러 번
    거친다. 그래서
      1) 진입 차로마다 "갈 수 있는 진출 도로" 집합(표)을 정하고,
      2) 내부 도로 차로에는 거기로 들어오는 진입 차로들의 집합을 합쳐 물려준 뒤,
      3) 각 연결(차로 → 다음 차로)을 "그 연결로 표에 있는 진출 도로에 갈 수
         있을 때만" 남긴다.
    지운 연결은 <delete>로 적어 netconvert로 도로망을 다시 만든다.

    Returns: {진입 도로 id: {차로: [실제로 갈 수 있는 진출 도로]}} (검증용)
    """
    net = sumolib.net.readNet(net_path)
    cx, cy = net.getNode(center_id).getCoord()
    legs = road["legs"]

    def leg_at(x, y):
        b = _bearing(x - cx, y - cy)
        lg = min(legs, key=lambda l: _angle_diff(b, l["bearing"]))
        return lg["id"] if _angle_diff(b, lg["bearing"]) < 40.0 else None

    # 교차로 = 중심 노드로 들어오는 연결을 제어하는 신호가 묶고 있는 노드들
    center_tl = next((c.getTLSID() for e in net.getNode(center_id).getIncoming()
                      for c in e.getOutgoing().values() for c in c if c.getTLSID()), None)
    nodes = {center_id}
    if center_tl:
        nodes |= {c[0].getEdge().getToNode().getID()
                  for c in net.getTLS(center_tl).getConnections()}
    inner = {e.getID() for e in net.getEdges()
             if e.getFromNode().getID() in nodes and e.getToNode().getID() in nodes}
    approach = {}     # 진입 엣지 id -> leg
    exit_leg = {}     # 진출 엣지 id -> leg
    for e in net.getEdges():
        if e.getID() in inner or not e.allows("passenger"):
            continue
        if e.getToNode().getID() in nodes:
            approach[e.getID()] = leg_at(*e.getShape()[0])
        if e.getFromNode().getID() in nodes:
            exit_leg[e.getID()] = leg_at(*e.getShape()[-1])

    def conns(lane):
        return [(c, c.getToLane()) for c in lane.getOutgoing()]

    def reach(lane, keep=None, depth=0):
        """lane에서 (keep 연결만 따라) 갈 수 있는 진출 leg 집합."""
        out = set()
        for c, tl in conns(lane):
            if keep is not None and (lane.getID(), tl.getID()) not in keep:
                continue
            te = tl.getEdge().getID()
            if te in exit_leg:
                out.add(exit_leg[te])
            elif te in inner and depth < 6:
                out |= reach(tl, keep, depth + 1)
        return out

    # 1) 진입 차로별 허용 집합
    allowed = {}      # lane id -> set(leg) | None(제한 없음)
    for eid, lg in approach.items():
        table = rules.get(lg) if lg else None
        n_lanes = net.getEdge(eid).getLaneNumber()
        for ln in net.getEdge(eid).getLanes():
            if table is None or not ln.allows("passenger"):
                allowed[ln.getID()] = None      # 표 없음 / 버스전용차로(BRT)는 netconvert 연결 그대로
            else:
                # 차로 번호: 0 = 맨 오른쪽(양수, 오른쪽부터) / -1 = 맨 왼쪽(음수, 왼쪽부터)
                # 연결로가 갈라진 뒤 차로 수가 줄어드는 도로는 왼쪽부터 세는 쪽이 안정적이다.
                i, j = ln.getIndex(), ln.getIndex() - n_lanes
                v = next((table[k] for k in (i, str(i), j, str(j)) if k in table), [])
                allowed[ln.getID()] = set(v)
    # 2) 내부 도로 차로로 전파 (들어오는 차로들의 합집합)
    frontier = [l for l in allowed]
    while frontier:
        lid = frontier.pop()
        for _, tl in conns(net.getLane(lid)):
            if tl.getEdge().getID() not in inner:
                continue
            a, b = allowed.get(tl.getID(), set()), allowed[lid]
            new = None if (a is None or b is None) else (a | b)
            if tl.getID() not in allowed or new != allowed[tl.getID()]:
                allowed[tl.getID()] = new
                frontier.append(tl.getID())
    # 3) 연결 거르기
    keep, deletes = set(), []
    for lid, al in allowed.items():
        lane = net.getLane(lid)
        for c, tl in conns(lane):
            te = tl.getEdge().getID()
            ok = (al is None
                  or (te in exit_leg and exit_leg[te] in al)
                  or (te in inner and reach(tl) & al))
            if ok:
                keep.add((lid, tl.getID()))
            else:
                deletes.append((lane.getEdge().getID(), te, lane.getIndex(), tl.getIndex()))
    # 표에 없는 엣지의 연결은 모두 유지
    for e in net.getEdges():
        for ln in e.getLanes():
            if ln.getID() not in allowed:
                for _, tl in conns(ln):
                    keep.add((ln.getID(), tl.getID()))

    if deletes:
        con_path = os.path.join(os.path.dirname(net_path), "lane_rules.con.xml")
        with open(con_path, "w") as f:
            f.write("<connections>\n")
            for fe, te, fl, tl in deletes:
                f.write(f'    <delete from="{fe}" to="{te}" fromLane="{fl}" toLane="{tl}"/>\n')
            f.write("</connections>\n")
        subprocess.run(
            [checkBinary("netconvert"), "-s", net_path, "--connection-files", con_path,
             "--output.street-names", "-o", net_path],
            check=True, capture_output=True)

    # 검증: 실제로 갈 수 있는 진출 도로 (거른 연결 기준)
    result = {}
    for eid, lg in approach.items():
        if lg is None or lg not in rules:
            continue
        for ln in net.getEdge(eid).getLanes():
            if not ln.allows("passenger"):
                continue
            got = sorted(reach(ln, keep))
            want = sorted(allowed[ln.getID()] or [])
            result.setdefault(lg, {})[ln.getIndex()] = got
            if got != want:
                print(f"[lane_rules] 경고: {lg} {ln.getIndex()}번 차로 — 표 {want}, "
                      f"실제 도로망 {got} (표의 이동 중 도로망에 없는 것은 만들 수 없음)")
    return result


def _link_exits(net, root, tl_id, center_id, legs):
    """신호 tl_id의 linkIndex마다 "그 링크로 갈 수 있는 진출 도로(leg) 집합".

    교차로가 노드 여러 개 + 내부 도로로 되어 있으면 첫 링크(진입 차로 →
    내부 도로)에서 진출 도로가 여러 개로 갈라질 수 있어, 내부 도로를 따라가며 모은다.
    """
    cx, cy = net.getNode(center_id).getCoord()
    nodes = {c[0].getEdge().getToNode().getID() for c in net.getTLS(tl_id).getConnections()}
    inner = {e.getID() for e in net.getEdges()
             if e.getFromNode().getID() in nodes and e.getToNode().getID() in nodes}

    def leg_at(x, y):
        b = _bearing(x - cx, y - cy)
        return min(legs, key=lambda l: _angle_diff(b, l["bearing"]))["id"]

    def reach(lane, depth=0):
        out = set()
        for c in lane.getOutgoing():
            te = c.getTo()
            if te.getID() in inner and depth < 6:
                out |= reach(c.getToLane(), depth + 1)
            elif te.getID() not in inner:
                out.add(leg_at(*te.getShape()[-1]))
        return out

    exits = {}
    for c in root.iter("connection"):
        if c.get("tl") != tl_id:
            continue
        to_lane = net.getEdge(c.get("to")).getLane(int(c.get("toLane")))
        te = to_lane.getEdge().getID()
        exits[int(c.get("linkIndex"))] = (reach(to_lane) if te in inner
                                          else {leg_at(*to_lane.getEdge().getShape()[-1])})
    return exits


def plan_to_phases(plan: dict, group: dict, exits: dict, n: int, signal: dict,
                   name: str = "") -> list:
    """신호 현시표(plan) 하나 → SUMO 현시 목록 [(초, 상태 문자열), ...].

    group[i] : 링크 i의 진입 도로 (None = 교차로 내부 링크 → 항상 녹색)
    exits[i] : 링크 i로 갈 수 있는 진출 도로 집합
    현시마다: 녹색 → 황색 → 전적색. 다음 현시에도 녹색인 링크는 황색/전적색
    동안에도 녹색을 유지한다 (끊기지 않는 이동류).
    """
    legs_known = {g for g in group.values() if g}

    def links_of(item):
        src, _, dst = str(item).partition(":")
        if src not in legs_known:
            raise ValueError(f"신호 계획 '{name}': 진입 도로 '{src}'가 교차로에 없습니다 "
                             f"(가능: {sorted(legs_known)})")
        idx = {i for i, g in group.items() if g == src
               and (not dst or dst in exits.get(i, set()))}
        if not idx:
            raise ValueError(f"신호 계획 '{name}': '{item}' 이동류에 해당하는 차로가 없습니다 "
                             f"(LANE_RULES 확인)")
        if dst:
            extra = sorted(set().union(*(exits.get(i, set()) for i in idx)) - {dst})
            if extra:
                print(f"[signal] 참고: 계획 '{name}'의 '{item}'은 {extra} 방향과 차로를 같이 "
                      f"써서(LANE_RULES) 그 방향도 함께 녹색이 됩니다.")
        return idx

    phases_in = plan["phases"]
    greens = [set().union(*(links_of(it) for it in ph["green"])) for ph in phases_in]
    out, total = [], 0.0
    for k, ph in enumerate(phases_in):
        y = float(ph.get("yellow", signal.get("yellow", 3.0)))
        ar = float(ph.get("all_red", signal.get("all_red", 2.0)))
        if "split" in ph:                       # 현시시간 (황색·전적색 포함)
            g = float(ph["split"]) - y - ar
        else:                                   # 녹색 시간만
            g = float(ph["green_time"])
        if g <= 0:
            raise ValueError(f"신호 계획 '{name}' {k + 1}현시: 녹색 시간이 0 이하입니다.")
        cur, nxt = greens[k], greens[(k + 1) % len(greens)]

        def st(ch_end):
            return "".join("G" if group.get(i) is None or i in (cur & nxt)
                           else (ch_end if i in cur else "r") for i in range(n))
        for dur, ch in ((g, "G"), (y, "y"), (ar, "r")):
            if dur > 0:
                out.append((dur, st(ch)))
        total += g + y + ar
    if "cycle" in plan and abs(total - float(plan["cycle"])) > 0.5:
        raise ValueError(f"신호 계획 '{name}': 현시 합 {total:g}초 ≠ cycle {plan['cycle']}초")
    return out


def _merge_carriageways(osm_file, merges, out_dir):
    """분리 차도 + 가운데 busway를 하나의 양방향 도로로 (tools/build_area_map.apply_carriageway_merges).
    반환: (새 OSM 파일, 노드 이름 바꿈 {옛 id: 새 id}, 합친 결과 목록)"""
    import sys
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    from tools.build_area_map import Proj, load_osm, apply_carriageway_merges
    tree = ET.parse(osm_file)
    lats = [float(n.get("lat")) for n in tree.getroot().iter("node")]
    lons = [float(n.get("lon")) for n in tree.getroot().iter("node")]
    proj = Proj(sum(lats) / len(lats), sum(lons) / len(lons))
    tmp = os.path.join(out_dir, "src.osm")
    tree.write(tmp, encoding="UTF-8", xml_declaration=True)
    root, nodes, ntags, ways, rels = load_osm(tmp, proj)
    done = apply_carriageway_merges(root, proj, nodes, ways, merges)
    rename = {}
    for d in done:
        rename.update(d.get("rename", {}))
    # OSM 파일은 node → way → relation 순서여야 netconvert가 읽는다 (새로 만든 노드가 뒤에 붙어 있음)
    order = {"bounds": 0, "node": 1, "way": 2, "relation": 3}
    kids = sorted(list(root), key=lambda el: order.get(el.tag, 0))
    for el in list(root):
        root.remove(el)
    root.extend(kids)
    out = os.path.join(out_dir, "merged.osm")
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    return out, rename, done


def bus_vtype_line(bus_cfg: dict) -> str:
    """버스 차종 (시내버스 12 m). bus_cfg["vtype"]으로 값을 바꿀 수 있다."""
    v = {"accel": 1.2, "decel": 4.0, "emergencyDecel": 7.0, "length": 12.0, "minGap": 2.5, "tau": 1.2,
         "maxSpeed": 13.89, "speedDev": 0.05, "sigma": 0.3, **(bus_cfg.get("vtype") or {})}
    attrs = " ".join(f'{k}="{val}"' for k, val in v.items())
    # bus_brt: 중앙 BRT 차로 버스. 실제 BRT는 연석·펜스로 갈라져 있어 중간에 일반 차로로 못 나간다.
    #   SUMO 경로 차로 선택은 줄이 짧은 차로를 고르기도 해서(BRT에 버스가 서 있으면 일반 차로로 빠짐) 차로 변경을 끈다
    #   (lcStrategic -1). BRT 경로는 brt_depart_lane이 차로 변경 없이 끝까지 이어지는지 확인한 것만 쓴다.
    return "\n".join(
        f'    <vType id="{vid}" vClass="bus" guiShape="bus" color="{col}" {attrs} '
        f'lcCooperative="0" lcSpeedGain="0" lcKeepRight="0"{extra}/>'
        for vid, col, extra in (("bus_city", "0.15,0.45,0.95", ""), ("bus_brt", "0.1,0.3,0.8", ' lcStrategic="-1"')))


def brt_depart_lane(net, edges):
    """첫 엣지의 버스전용차로에서 출발해 차로 변경 없이(연결만 따라) 경로 끝까지 갈 수 있으면 그 차로 번호, 아니면 None."""
    first = net.getEdge(edges[0])
    for ln in first.getLanes():
        if not (ln.allows("bus") and not ln.allows("passenger")):
            continue
        cur = {ln}
        for nxt in edges[1:]:
            cur = {c.getToLane() for l_ in cur for c in l_.getOutgoing() if c.getToLane().getEdge().getID() == nxt}
            if not cur:
                break
        if cur:
            return ln.getIndex()
    return None


def build_osm(road: dict, ego: dict, traffic: dict, signal: dict,
              out_dir: str) -> str:
    from env.road_builder import make_vtypes_block   # 순환 import 방지

    cfg = road["osm"]
    osm_file = os.path.join(BASE, cfg["file"])
    center_id = str(cfg["center_node"])
    net_path = os.path.join(out_dir, "highway.net.xml")
    types_path = os.path.join(out_dir, "osm_types.typ.xml")
    with open(types_path, "w") as f:
        f.write(TYPES_XML)

    # ──────────────────────────────────────────────────────
    # 1) netconvert: OSM → SUMO 도로망
    #    --tls.guess-signals : OSM 신호 노드를 가까운 교차로 신호로 합침
    #    --tls.set           : OSM에 신호 표시가 없는 연산교차로 중심과
    #                          extra_signals 교차로(예: 시청 인근)를 신호로 지정
    #    --keep-edges.in-geo-boundary : bbox에 걸친 도로만 남김
    # ──────────────────────────────────────────────────────
    minlon, minlat, maxlon, maxlat = cfg["bbox"]
    # extra_signals 항목: 노드 id 하나, 또는 하나로 합쳐지는 노드 묶음(리스트).
    # 묶음은 netconvert --junctions.join이 "cluster_<정렬된 id들>"로 합친 뒤에
    # 신호를 붙여야 하므로 그 이름으로 바꿔서 넘긴다.
    extra_ids = [("cluster_" + "_".join(sorted(map(str, s)))) if isinstance(s, (list, tuple))
                 else str(s) for s in cfg.get("extra_signals", [])]
    # 1-0) 합친 맵 모드: 따로 만든 맵(교차로 정밀맵 + 주변부 등)을 쓰면 OSM 변환을 건너뛴다
    #   merge_parts: plain XML 묶음 목록 → 빌드할 때마다 tools/merge_nets.py로 합침
    #   net_file   : 이미 합쳐 둔 net.xml
    #   이후 단계(LANE_RULES, 신호, 경로/교통 생성)는 OSM 모드와 같다.
    #   center_node / extra_signals 는 합친 맵의 노드 id와 맞춰야 한다.
    if cfg.get("merge_parts"):
        import sys
        if BASE not in sys.path:
            sys.path.insert(0, BASE)
        from tools.merge_nets import merge_parts
        merge_parts([os.path.join(BASE, p) for p in cfg["merge_parts"]], net_path,
                    strict=cfg.get("merge_strict", True))
    elif cfg.get("net_file"):
        import shutil
        shutil.copy(os.path.join(BASE, cfg["net_file"]), net_path)
    else:
        # 1-1) 분리 차도 + 가운데 BRT busway → 하나의 양방향 도로 (영역 지도와 같은 함수, cfg["merge_carriageways"])
        osm_in, rename = osm_file, {}
        merges_done = []
        if cfg.get("merge_carriageways"):
            osm_in, rename, merges_done = _merge_carriageways(osm_file, cfg["merge_carriageways"], out_dir)
        # 버스전용차로: 모든 도로에 같은 규칙 (env/bus_lanes.py) — OSM 버스 태그를 netconvert가 읽는 형식으로 정리
        from env.bus_lanes import normalize_osm_file
        osm_in = normalize_osm_file(osm_in, os.path.join(out_dir, "bus_normalized.osm"))
        # 1-2) 1차 변환 (신호는 OSM 신호만)
        raw0 = os.path.join(out_dir, "raw0.net.xml")
        # 합친 도로는 범위(bbox) 밖 조각도 남긴다 — 범위 밖에서 골목길과 만나 끊긴 바깥 조각이 버려지면
        # 간선 입구가 범위 경계(바로 앞 신호 교차로)로 당겨진다. (--keep-edges.explicit는 범위 조건과 합집합)
        keep_merged = [f"{d}{sid}{('#' + str(k)) if k >= 0 else ''}" for m in merges_done
                       for sid in m.get("segments", []) for d in ("", "-") for k in range(-1, 80)]
        keep_args = ["--keep-edges.explicit", ",".join(keep_merged)] if keep_merged else []
        # 보정 파일 (ROAD["osm"]의 fix_nodes / fix_edges / fix_connections): OSM과 같은 netconvert 실행에 넣어야
        # 바뀐 차로의 연결이 다시 계산된다. 연산교차로는 지금 x6_spec 정밀 모델이 대신하므로 기본은 꺼 둔다.
        fix_args = []
        if cfg.get("fix_nodes"):
            fix_args += ["--node-files", os.path.join(BASE, cfg["fix_nodes"])]
        if cfg.get("fix_edges"):
            fix_args += ["--edge-files", os.path.join(BASE, cfg["fix_edges"])]
        if cfg.get("fix_connections"):
            fix_args += ["--connection-files", os.path.join(BASE, cfg["fix_connections"])]
        subprocess.run(
            [checkBinary("netconvert"),
             "--osm-files", osm_in, "--type-files", types_path,
             "--geometry.remove", "--ramps.guess", "--junctions.join",
             "--junctions.join-exclude", center_id,           # 연산교차로 중심 노드 id 유지 (LANE_RULES·신호 기준)
             "--tls.guess-signals", "--tls.join", "--tls.default-type", "static",
             "--remove-edges.isolated", "--keep-edges.by-vclass", "passenger",
             "--no-turnarounds.except-deadend", "--output.street-names",
             "--osm.lane-access", "true",                     # 버스전용차로 → 그 차로 allow="bus"
             "--keep-edges.in-geo-boundary", f"{minlon},{minlat},{maxlon},{maxlat}", *keep_args,
             *fix_args,
             "-o", raw0],
            check=True, capture_output=True)
        # 1-3) 신호 배치 (env/signals.py): 중심·extra + 빠진 OSM 신호(신호 횡단보도 등) + 간선끼리 만나는 교차로
        from env.signals import osm_signal_nodes, missing_osm_signals, assumed_major_signals
        net0 = sumolib.net.readNet(raw0)

        def node_of(osm_ids):
            ids = [rename.get(str(i), str(i)) for i in osm_ids]
            return next((n.getID() for n in net0.getNodes()
                         if n.getID() in ids or set(ids) & set(n.getID().split("_"))), None)
        tls_list = [node_of([center_id])]
        for spec in cfg.get("extra_signals", []):
            tls_list.append(node_of(spec if isinstance(spec, (list, tuple)) else [spec]))
        if None in tls_list:
            raise RuntimeError(f"신호로 지정할 노드를 도로망에서 못 찾음: center/extra_signals {tls_list}")
        sig_add, sig_skip = missing_osm_signals(net0, osm_signal_nodes(osm_file), rename=rename)
        major_add = assumed_major_signals(net0) if cfg.get("assume_major_signals", True) else []
        tls_list += [n for n in sig_add + major_add if n not in tls_list]
        subprocess.run(
            [checkBinary("netconvert"), "-s", raw0,
             "--tls.set", ",".join(tls_list), "--tls.join", "--tls.default-type", "static",
             "--tls.layout", "incoming",                      # 아래 korean_phase_programs가 현시를 다시 만듦
             "--tls.allred.time", str(int(round(signal.get("all_red", 2.0)))),   # 황색 뒤 전적색 (교차로 비우기)
             "--tls.yellow.time", str(int(round(signal.get("yellow", 3.0)))),  # 정수만 허용
             "--output.street-names", "-o", net_path],
            check=True, capture_output=True)
        from env.signals import fix_midblock_programs, fix_joined_programs, korean_phase_programs
        phased = korean_phase_programs(net_path)        # 연산교차로·extra_signals는 아래에서 다시 덮어씀
        from env.signals import brt_split_programs
        brt_split = (brt_split_programs(net_path, bus_green=float(signal.get("bus_green", 0.0)),
                                        yellow=float(signal.get("yellow", 3.0)))
                     if float(signal.get("bus_green", 0.0)) > 0 else [])
        midblock = fix_midblock_programs(net_path)
        joined_inner = fix_joined_programs(net_path)
        # 1-4) 연산교차로 정밀 모델 (차로 조사 자료, tools/build_x6.py)
        x6 = None
        if cfg.get("x6_spec"):
            import sys
            if BASE not in sys.path:
                sys.path.insert(0, BASE)
            from tools.build_x6 import apply_x6, check_x6
            spec_path = os.path.join(BASE, cfg["x6_spec"])
            x6 = apply_x6(net_path, net_path, spec_path, workdir=os.path.join(out_dir, "x6"))
            chk = check_x6(net_path, spec_path)
            if chk["problems"]:
                raise RuntimeError(f"연산교차로 정밀 모델이 자료와 다름: {chk['problems']}")
            x6 = {"center_signal": x6["center_signal"], "lanes_checked": chk["lanes_checked"], "ok": chk["ok"],
                  "warnings": x6.get("warnings", []), "stop_line_r": x6.get("stop_line_r")}
        signal_build = {"x6": x6, "osm_signals_added": sig_add, "osm_signals_skipped": sig_skip,
                        "assumed_major": major_add, "phase_programs": phased, "brt_split": brt_split, "midblock_programs": midblock,
                        "joined_inner_green": joined_inner,
                        "carriageway_merges": [{k: v for k, v in m.items() if k != "rename"} for m in merges_done]}
        extra_osm = [spec if isinstance(spec, (list, tuple)) else [spec] for spec in cfg.get("extra_signals", [])]

    # 버스전용차로가 이어지는 곳(BRT)은 버스차로 직진 연결만 (env/bus_lanes.py, 버스차로가 없으면 변화 없음)
    from env.bus_lanes import brt_through_deletes
    _bdel = brt_through_deletes(sumolib.net.readNet(net_path))
    if _bdel:
        _bcon = os.path.join(out_dir, "brt.con.xml")
        with open(_bcon, "w") as f:
            f.write("<connections>\n" + "".join(
                f'    <delete from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n' for a, b, fl, tl in _bdel)
                    + "</connections>\n")
        subprocess.run([checkBinary("netconvert"), "-s", net_path, "--connection-files", _bcon,
                        "--output.street-names", "-o", net_path], check=True, capture_output=True)

    # 차로별 진행 가능 방향 (road_config.LANE_RULES) — 신호/경로 생성 전에 적용
    try:
        from env.road_config import LANE_RULES
    except ImportError:
        LANE_RULES = {"enabled": False}
    lane_table = None
    if LANE_RULES.get("enabled", False):
        lane_table = apply_lane_rules(net_path, road, center_id,
                                      LANE_RULES.get("rules", {}))

    # 건물 (GUI 배경). 실패해도 시뮬레이션에는 영향 없음.
    additional = []
    if cfg.get("buildings", True) and os.path.exists(osm_file):
        poly_path = os.path.join(out_dir, "highway.poly.xml")
        typ = os.path.join(os.environ["SUMO_HOME"], "data", "typemap",
                           "osmPolyconvert.typ.xml")
        res = subprocess.run(
            [checkBinary("polyconvert"), "--osm-files", osm_file,
             "--net-file", net_path, "--type-file", typ, "-o", poly_path],
            capture_output=True)
        if res.returncode == 0:
            additional.append("highway.poly.xml")

    net = sumolib.net.readNet(net_path)
    if not (cfg.get("merge_parts") or cfg.get("net_file")):
        # extra_signals → 최종 도로망에서 그 노드를 제어하는 신호 id (합치기·--tls.join으로 이름이 바뀔 수 있음)
        tl_of = {c[0].getEdge().getToNode().getID(): tl.getID() for tl in net.getTrafficLights() for c in tl.getConnections()}
        extra_ids = []
        for ids in extra_osm:
            ids = [rename.get(str(i), str(i)) for i in ids]
            nid = next((n for n in tl_of if n in ids or set(ids) & set(n.split("_"))), None)
            if nid is None:
                raise RuntimeError(f"extra_signals {ids}에 신호가 만들어지지 않았습니다.")
            extra_ids.append(tl_of[nid])
    if not net.hasNode(center_id):
        raise RuntimeError(f"도로망에 중심 노드 '{center_id}'가 없습니다. 합친 맵을 쓰면 "
                           f"ROAD['osm']['center_node']를 그 맵의 연산교차로 노드 id로 바꾸세요.")
    for jid in extra_ids:
        if jid not in {tl.getID() for tl in net.getTrafficLights()}:
            raise RuntimeError(f"extra_signals의 '{jid}'에 신호가 만들어지지 않았습니다. "
                               f"노드 id 또는 합쳐지는 노드 묶음을 확인하세요.")
    center = net.getNode(center_id)
    cx, cy = center.getCoord()

    # ──────────────────────────────────────────────────────
    # 2) 도로 끝점(fringe) 찾기
    #    끝점 = 이웃 노드가 하나뿐이거나 막다른(dead_end) 노드 (도로망 바깥 경계)
    #    진입 엣지 = 끝점에서 나가는 엣지, 진출 엣지 = 끝점으로 들어오는 엣지
    # ──────────────────────────────────────────────────────
    entries, exits = [], []      # (edge, 방위각, 거리, 도로 이름)
    for nd in net.getNodes():
        nbrs = ({e.getFromNode() for e in nd.getIncoming()}
                | {e.getToNode() for e in nd.getOutgoing()})
        if len(nbrs) != 1 and nd.getType() != "dead_end":
            continue
        x, y = nd.getCoord()
        b, d = _bearing(x - cx, y - cy), math.hypot(x - cx, y - cy)
        for e in nd.getOutgoing():
            if e.allows("passenger"):
                entries.append((e, b, d, e.getName()))
        for e in nd.getIncoming():
            if e.allows("passenger"):
                exits.append((e, b, d, e.getName()))

    def heading(e, at_end):
        """엣지의 끝(또는 시작) 부분 진행 방향 (방위각)."""
        s = e.getShape()
        (x0, y0), (x1, y1) = (s[-2], s[-1]) if at_end else (s[0], s[1])
        return _bearing(x1 - x0, y1 - y0)

    def near_edge(lg, outgoing):
        """교차점에 바로 붙은, 간선 lg 방향의 엣지 (나가는/들어오는 쪽)."""
        def far_point(e):
            s = e.getShape()
            return s[-1] if outgoing else s[0]
        first = center.getOutgoing() if outgoing else center.getIncoming()
        cands = [e for e in first if e.allows("passenger")
                 and _angle_diff(_bearing(far_point(e)[0] - cx, far_point(e)[1] - cy),
                                 lg["bearing"]) < 40.0]
        cands = [e for e in cands if lg["road"] in (e.getName() or "")] or cands
        if not cands:
            raise RuntimeError(f"OSM 도로망에서 '{lg['road']}' ({lg['bearing']}°) "
                               f"방향 도로를 교차점에서 찾지 못했습니다.")
        return min(cands, key=lambda e: _angle_diff(
            _bearing(far_point(e)[0] - cx, far_point(e)[1] - cy), lg["bearing"]))

    def trace_leg(lg, outgoing):
        """교차점에서 간선을 따라 바깥으로 이어 가서 도로망 끝 엣지를 찾는다.

        outgoing=True : 교차점에서 나가는 차도를 따라감 → 진출 끝 엣지
        outgoing=False: 교차점으로 들어오는 차도를 거슬러 감 → 진입 시작 엣지
        같은 도로 이름이면서 가장 덜 꺾이는 다음 엣지를 고른다.
        """
        e = near_edge(lg, outgoing)
        seen = {e.getID()}
        for _ in range(200):
            nxt = list(e.getOutgoing().keys()) if outgoing else list(e.getIncoming().keys())
            nxt = [n for n in nxt if n.getID() not in seen and n.allows("passenger")
                   and n.getName() == e.getName()
                   and n.getID().lstrip("-") != e.getID().lstrip("-")]   # 유턴 제외
            if not nxt:
                return e
            h = heading(e, at_end=outgoing)
            e = min(nxt, key=lambda n: _angle_diff(heading(n, at_end=not outgoing), h))
            seen.add(e.getID())
        return e

    legs = road["legs"]
    leg_in = {lg["id"]: trace_leg(lg, outgoing=False) for lg in legs}
    leg_out = {lg["id"]: trace_leg(lg, outgoing=True) for lg in legs}
    near_in = {lg["id"]: near_edge(lg, outgoing=False) for lg in legs}
    near_out = {lg["id"]: near_edge(lg, outgoing=True) for lg in legs}
    used = {e.getID() for e in list(leg_in.values()) + list(leg_out.values())}
    side_in = [c[0] for c in entries if c[0].getID() not in used]
    side_out = [c[0] for c in exits if c[0].getID() not in used]

    # 금지 이동이 정말 지워졌는지 확인. 없는 edge id는 netconvert가 에러로
    # 멈추지만, OSM을 다시 받아 조각이 바뀌면 id는 있어도 엉뚱한 연결을 지울 수
    # 있으므로 결과를 직접 확인한다.
    banned = {tuple(t) for t in cfg.get("banned_turns", [])}
    for a, b in sorted(banned):
        left = [f"{ln.getID()}→{c.getToLane().getID()}"
                for ln in near_in[a].getLanes() for c in ln.getOutgoing()
                if c.getTo().getID() == near_out[b].getID()]
        if left:
            raise RuntimeError(f"금지 이동 {a}→{b} 연결이 남아 있습니다: {left}. "
                               f"fix_connections 파일의 edge id를 확인하세요.")

    # ──────────────────────────────────────────────────────
    # 3) 교통 수요
    #    · 간선 → 간선
    #      - 실측 OD(TRAFFIC["demand"], 부산시 2025 교통조사)가 켜져 있으면
    #        방향별 실측 교통량 그대로. 연산교차로를 지나는 교통량 = 실측.
    #        간선 → 골목길은 그 간선 실측 합의 side_share만큼 따로 더한다.
    #      - 꺼져 있으면 도로별 inflow(가정값)를 다른 간선으로 배분
    #        (진출 차로 수 비율) + 그중 side_share만큼은 골목길로 빠짐.
    #        금지 이동(banned_turns)은 빼고 나머지 방향에 다시 나눈다.
    #    · 골목길 → 간선: 입구마다 side_inflow 대/시간 (실측 수요가 켜져 있으면 × scale)
    #    경로는 sumolib 최단경로 (실제 연결/회전 금지를 따름)
    # ──────────────────────────────────────────────────────
    path_cache = {}

    def path(a, b):
        key = (a.getID(), b.getID())
        if key not in path_cache:
            p, _ = net.getShortestPath(a, b, vClass="passenger")
            path_cache[key] = [e.getID() for e in p] if p else None
        return path_cache[key]

    # 간선 → 간선 경로: 그냥 최단거리로 찾으면 일부 방향(예: SW→E, E→S, W→SW)이
    # 큰 교차로 대신 이면도로나 다른 도로로 질러간다. 실제 간선 교통은 간선을
    # 타고 연산교차로를 지나므로, "출발 간선 도로 → 교차로 또는 우회전 연결로 →
    # 도착 간선 도로"에 속한 edge만 써서 경로를 찾는다.
    def corridor(lg, outgoing):
        """교차점에 붙은 edge에서 같은 이름 도로를 따라 도로망 끝까지 (trace_leg와 같은 규칙)."""
        e = near_edge(lg, outgoing)
        out = [e.getID()]
        for _ in range(200):
            nxt = list(e.getOutgoing().keys()) if outgoing else list(e.getIncoming().keys())
            nxt = [n for n in nxt if n.getID() not in out and n.allows("passenger")
                   and n.getName() == e.getName()
                   and n.getID().lstrip("-") != e.getID().lstrip("-")]
            if not nxt:
                break
            h = heading(e, at_end=outgoing)
            e = min(nxt, key=lambda n: _angle_diff(heading(n, at_end=not outgoing), h))
            out.append(e.getID())
        return set(out)

    corr_in = {lg["id"]: corridor(lg, outgoing=False) for lg in legs}
    corr_out = {lg["id"]: corridor(lg, outgoing=True) for lg in legs}
    slip_edges = {e.getID() for e in net.getEdges()
                  if "_link" in (e.getType() or "") and min(
                      math.hypot(x - cx, y - cy) for x, y in e.getShape()) < 150.0}

    def through_path(a_id, b_id):
        """leg a 진입 끝 → leg b 진출 끝, 두 간선 도로와 연산교차로·연결로만 쓰는 경로."""
        import heapq
        key = ("through", a_id, b_id)
        if key in path_cache:
            return path_cache[key]
        allowed = corr_in[a_id] | corr_out[b_id] | slip_edges
        src, dst = leg_in[a_id].getID(), leg_out[b_id].getID()
        dist, prev, heap = {src: 0.0}, {}, [(0.0, src)]
        out = None
        while heap:
            d, eid = heapq.heappop(heap)
            if eid == dst:
                out = [eid]
                while out[-1] in prev:
                    out.append(prev[out[-1]])
                out.reverse()
                break
            if d > dist.get(eid, 1e18):
                continue
            for n in net.getEdge(eid).getOutgoing():
                nid = n.getID()
                if nid not in allowed or not n.allows("passenger"):
                    continue
                if d + n.getLength() < dist.get(nid, 1e18):
                    dist[nid], prev[nid] = d + n.getLength(), eid
                    heapq.heappush(heap, (d + n.getLength(), nid))
        path_cache[key] = out
        return out

    def passes_center(edges):
        """경로가 연산교차로를 지나는가 (교차로 안 통과 또는 우회전 연결로)."""
        for eid in edges:
            e = net.getEdge(eid)
            if e.getToNode().getID() == center_id:
                return True
            if "_link" in (e.getType() or "") and min(
                    math.hypot(x - cx, y - cy) for x, y in e.getShape()) < 150.0:
                return True
        return False

    vtypes_block, flow_type = make_vtypes_block(traffic, keep_strategic=True)
    route_lines, flow_lines = [], []
    meta = {"layout": "osm", "center": center_id, "center_xy": [cx, cy],
            "ego_routes": [], "ego_depart_pos": {}, "bg_routes": {}, "in_edges": {}}
    side_share = float(cfg.get("side_share", 0.1))
    side_inflow = float(cfg.get("side_inflow", 40.0))
    demand = traffic.get("demand") or {}
    od_legs = {name: lg for name, lg in (demand.get("legs") or {}).items() if lg}
    use_od = bool(od_legs)
    poisson = use_od and demand.get("poisson", True)
    k = 0
    # 도로 디테일 (road_config.DETAILS / ROAD_DETAILS 환경변수, 기본 전부 끔 → 아래 결과 파일은 예전과 같다)
    details = road_details.config()
    details_rng = road_details.rng_for(details)
    curb = road_details.Curb(net, cx, cy, details)
    od_paths, details_meta = [], {}

    def add_flow(src, dst, rate, finder=None):
        nonlocal k
        p = (finder or path)(src, dst)
        if p is None or rate <= 0:
            return
        rid = f"bg_{k}"
        k += 1
        route_lines.append(f'    <route id="{rid}" edges="{" ".join(p)}"/>')
        # 포아송 도착: 초당 λ대 (등간격보다 실제 도착 패턴에 가깝다)
        arrival = (f'period="exp({rate / 3600.0:.6f})"' if poisson
                   else f'vehsPerHour="{rate:.1f}"')
        flow_lines.append(
            f'    <flow id="f_{rid}" type="{flow_type}" route="{rid}" begin="0" '
            f'end="100000" {arrival} departLane="best" '
            f'departSpeed="{traffic.get("depart_speed", "max")}"/>')
        meta["bg_routes"][rid] = {"first_edge": src.getID(), "rate": rate}
        # lanes: 일반 차량이 다닐 수 있는 차로 수 (BRT 버스 전용 차로 제외).
        # 버스 차로는 맨 왼쪽(가장 큰 번호)이라 0 ~ lanes-1이 일반 차로다.
        # sumo_env의 prefill이 이 범위에만 배경차를 놓는다.
        meta["in_edges"].setdefault(src.getID(), {
            "lanes": sum(1 for ln in src.getLanes() if ln.allows("passenger")),
            "length": src.getLength(), "inflow": 0.0})
        meta["in_edges"][src.getID()]["inflow"] += rate

    by_id = {lg["id"]: lg for lg in legs}
    if use_od:
        from env.road_builder import load_od_demand   # 순환 import 방지
        missing = [name for name in demand["legs"] if not demand["legs"][name]]
        if missing:
            raise ValueError(f"demand.legs에 leg가 비어 있는 방면이 있습니다: {missing}")
        unknown = sorted(set(od_legs.values()) - set(by_id))
        if unknown:
            raise ValueError(f"demand.legs의 leg id가 ROAD['legs']에 없습니다: {unknown}")
        od = load_od_demand(demand)
        leg_total = {a: 0.0 for a in by_id}
        # 도로 디테일(택시·잠깐 정차)을 켜면 그 대수만큼 승용차 flow에서 떼어 내 따로 만든다 (총량 그대로).
        carve = road_details.carve_rates(demand, details)
        for (o, d), vph in od.items():
            if o not in od_legs or d not in od_legs:
                raise ValueError(f"CSV의 방면 '{o}' 또는 '{d}'가 demand.legs에 없습니다.")
            a, b = od_legs[o], od_legs[d]
            if (a, b) in banned or a == b:
                continue      # 금지 이동 (실측에도 없음)
            p = through_path(a, b)
            if p is None or not passes_center(p):
                raise RuntimeError(f"실측 OD {o}→{d} ({a}→{b}): 두 간선 도로와 연산교차로만으로 "
                                   f"가는 경로가 없습니다 (연결·금지 회전 확인).")
            cut = carve.get((o, d)) or {}
            add_flow(leg_in[a], leg_out[b], vph - sum(cut.values()) if cut else vph,
                     finder=lambda s_, t_, a=a, b=b: through_path(a, b))
            od_paths.append((f"{a}_{b}", p, {**cut, "vph": vph}))
            leg_total[a] += vph
        meta["demand"] = {"scenario": demand.get("scenario"),
                          "veh_per_hour": round(sum(leg_total.values()), 1)}
        # 버스 (실제 버스 차종): 실측 OD의 버스 중형+대형. BRT 이동(버스전용차로만으로 끝까지 갈 수 있는 경로)은
        # 버스전용차로에서 출발하고, 나머지는 일반 차로로 다닌다. include_bus(버스를 승용차로 셈)와 같이 켜면 두 번 세므로 막는다.
        bus_cfg = demand.get("bus_vehicles") or {}
        if bus_cfg.get("enabled"):
            if demand.get("include_bus"):
                raise ValueError("demand.include_bus와 demand.bus_vehicles.enabled를 같이 켜면 버스를 두 번 셉니다.")
            from env.road_builder import load_od_bus
            vtypes_block += "\n" + bus_vtype_line(road_details.bus_cfg_with_slope(bus_cfg, details))
            meta["bus"] = {"routes": {}, "veh_per_hour": 0.0}
            bus_routes = []          # 정류장(DETAILS['bus_stops'])을 켜면 노선을 다 모은 뒤 flow를 만든다
            for (o, d), vph in load_od_bus(demand).items():
                a, b = od_legs.get(o), od_legs.get(d)
                if a is None or b is None or (a, b) in banned or a == b:
                    continue
                p = through_path(a, b)
                if p is None or not all(net.getEdge(e).allows("bus") for e in p):
                    continue
                lane = brt_depart_lane(net, p)
                rid = f"bus_{len(meta['bus']['routes'])}"
                route_lines.append(f'    <route id="{rid}" edges="{" ".join(p)}"/>')
                arrival = (f'period="exp({vph / 3600.0:.6f})"' if poisson else f'vehsPerHour="{vph:.1f}"')
                base = (f'id="f_{rid}@ID@" type="{"bus_city" if lane is None else "bus_brt"}" route="{rid}" begin="0" end="100000"',
                        f'departLane="{"best" if lane is None else lane}" departSpeed="{traffic.get("depart_speed", "max")}"')
                if road_details.enabled(details, "bus_stops"):
                    bus_routes.append((rid, p, lane is not None, vph, base))
                else:
                    flow_lines.append(f'    <flow {base[0].replace("@ID@", "")} {arrival} {base[1]}/>')
                meta["bus"]["routes"][rid] = {"od": [a, b], "rate": vph, "brt": lane is not None}
                meta["bus"]["veh_per_hour"] += vph
            if bus_routes:
                stops, route_stops = road_details.place_bus_stops(
                    curb, [(r, p_, brt) for r, p_, brt, _, _ in bus_routes], details, details_rng)
                for rid, p_, brt, vph, (b0, b1) in bus_routes:
                    flow_lines += road_details.bus_flow_lines(f"{b0} {b1}", vph, poisson, route_stops[rid],
                                                              details, details_rng)
                stops_path = os.path.join(out_dir, "highway.stops.xml")
                with open(stops_path, "w", encoding="utf-8") as f:
                    f.write("<additional>\n" + road_details.bus_stops_xml(stops) + "\n</additional>\n")
                additional.append("highway.stops.xml")
                details_meta["bus_stops"] = {"count": len(stops),
                                             "per_route": {r: len(s) for r, s in route_stops.items()}}
    else:
        leg_total = {a: float(by_id[a]["inflow"]) for a in by_id}
        for a in by_id:
            dests = [b for b in by_id if b != a and (a, b) not in banned]
            wsum = sum(by_id[b]["lanes_out"] for b in dests)
            main = leg_total[a] * (1.0 - side_share)
            for b in dests:
                add_flow(leg_in[a], leg_out[b], main * by_id[b]["lanes_out"] / wsum,
                         finder=lambda s_, t_, a=a, b=b: through_path(a, b))
    for a in by_id:
        if side_out:
            side_rate = leg_total[a] * side_share / len(side_out)
            for s in side_out:
                add_flow(leg_in[a], s, side_rate)
    # 골목 → 간선: 실측 수요가 켜져 있으면 실측 배율(scale)도 같이 적용한다
    # (골목에서 들어온 차도 연산교차로 실측 통과량에 포함되기 때문)
    side_rate_in = side_inflow * (float(demand.get("scale", 1.0)) if use_od else 1.0)
    for s in side_in:
        for b in by_id:
            add_flow(s, leg_out[b], side_rate_in / len(by_id))

    # ego 경로: 출발 간선 끝점 → 도착 간선 끝점. 출발 위치는 교차점에서
    # ego_approach(m) 떨어진 지점 (긴 도로 끝에서 출발하면 에피소드가 너무 길어짐)
    #
    # 경로 형식: (진입, 진출) = 교차로 1번 통과.
    #   (진입1, 진출1, 진입2, 진출2, ...) = 여러 번 통과. 진출1로 나간 뒤
    #   주변 도로를 돌아 진입2로 다시 들어온다 (중간 경로는 최단경로).
    approach = float(cfg.get("ego_approach", 400.0))
    loop_r = float(cfg.get("loop_min_radius", 150.0))

    def near_center(e):
        (x0, y0), (x1, y1) = e.getShape()[0], e.getShape()[-1]
        return min(math.hypot(x0 - cx, y0 - cy), math.hypot(x1 - cx, y1 - cy)) < loop_r

    def loop_path(src, dst, names):
        """src(교차점에서 나가는 엣지) → dst(교차점으로 들어오는 엣지) 우회 경로.

        교차점 반경 loop_r 안에서는 나간 도로·다시 들어올 도로(names)만 쓰고,
        우회전 연결로(*_link)와 교차점 통과는 금지한다. 그래서 연결로로
        제자리에서 되돌아오는 대신 바깥 이면도로를 한 바퀴 돌아온다.
        """
        import heapq
        dist, prev = {src.getID(): 0.0}, {}
        heap = [(0.0, src.getID())]
        while heap:
            d, eid = heapq.heappop(heap)
            if eid == dst.getID():
                out = [eid]
                while out[-1] in prev:
                    out.append(prev[out[-1]])
                return out[::-1]
            if d > dist.get(eid, 1e18):
                continue
            for n in net.getEdge(eid).getOutgoing():
                nid = n.getID()
                if not n.allows("passenger") or "_link" in (n.getType() or ""):
                    continue
                if n.getToNode().getID() == center_id and nid != dst.getID():
                    continue
                if near_center(n) and n.getName() not in names:
                    continue
                nd = d + n.getLength()
                if nd < dist.get(nid, 1e18):
                    dist[nid], prev[nid] = nd, eid
                    heapq.heappush(heap, (nd, nid))
        return None

    def via_path(s, t, via):
        """s → via 엣지들 → t (연산교차로를 거치지 않는 최단경로를 이어 붙임). 회전교차로 등을 들르는 경로용."""
        out, cur = [s.getID()], s
        for v in [net.getEdge(x) for x in via] + [t]:
            p, _ = net.getShortestPath(cur, v, vClass="passenger")
            if not p:
                return None
            ids = [e.getID() for e in p]
            if any(net.getEdge(e).getToNode().getID() == center_id for e in ids[:-1]):
                return None
            out += ids[1:]
            cur = v
        return out

    # 순환 도로 배경 흐름 (TRAFFIC["loop_flows"]): 연산교차로를 지나지 않고 루프 주변 도로만 다니는 차.
    #   연산교차로 실측 OD는 교차로를 지나는 차만 담고 있어서, 루프 경로의 나머지 구간(연수로·중앙천로·
    #   거제천로 등)이 거의 비어 있었다. 엣지 id는 net_file로 적은 지도 기준이라 그 지도일 때만 넣는다.
    #   대/시 = vph × loop_flows.scale × demand.scale (연산교차로 수요와 같은 배율을 따름)
    lf = traffic.get("loop_flows") or {}
    if lf.get("enabled") and cfg.get("net_file") and \
            os.path.normpath(cfg["net_file"]) == os.path.normpath(lf.get("net_file", "")):
        lscale = float(lf.get("scale", 1.0)) * float(demand.get("scale", 1.0))
        meta["loop_flows"] = {"veh_per_hour": 0.0, "flows": {}}
        for f in lf.get("flows", []):
            via = list(f["via"])
            missing = [e for e in via if not net.hasEdge(e)]
            if missing:
                raise RuntimeError(f"loop_flows '{f['name']}': 지도에 없는 엣지 {missing}")
            vph = float(f["vph"]) * lscale
            src, dst = net.getEdge(via[0]), net.getEdge(via[-1])
            p = via_path(src, dst, via[1:-1])
            if p is None:
                raise RuntimeError(f"loop_flows '{f['name']}': 연산교차로를 지나지 않는 경로가 없습니다")
            before = k
            # 도로 디테일의 잠깐 정차(DETAILS['short_stops'])를 켜면 순환 도로 흐름에서도 같은 비율을 떼어 낸다
            cut = ({"short_stops": vph * float((details.get("short_stops") or {}).get("share", 0.02))}
                   if road_details.enabled(details, "short_stops") else {})
            add_flow(src, dst, vph - sum(cut.values()) if cut else vph, finder=lambda s_, t_, p=p: p)
            od_paths.append((f"loop_{len(od_paths)}", p, {**cut, "vph": vph}))
            if k > before:
                meta["loop_flows"]["flows"][f"bg_{before}"] = {"name": f["name"], "rate": vph,
                                                               "source": f.get("source", "")}
                meta["loop_flows"]["veh_per_hour"] += vph

    def ego_path(spec):
        via = None
        if isinstance(spec, dict):       # {"legs": (진입, 진출, 진입, 진출), "via": [엣지...]} — 재진입 전에 들를 엣지
            via, spec = spec.get("via"), tuple(spec["legs"])
        if len(spec) < 2 or len(spec) % 2:
            raise RuntimeError(f"ego 경로 {spec}: (진입, 진출) 쌍으로 적어야 합니다.")
        pairs = [(spec[j], spec[j + 1]) for j in range(0, len(spec), 2)]
        full, cur = [], leg_in[pairs[0][0]]
        prev_exit = None
        for j, (a, b) in enumerate(pairs):
            steps = []
            if j > 0 and via:   # 지정한 엣지(예: 회전교차로)를 들러 a 방향에서 재진입
                steps.append((near_in[a], lambda s, t, v=via: via_path(s, t, v)))
            elif j > 0:   # 주변 이면도로를 돌아 a 방향에서 재진입
                names = {by_id[prev_exit]["road"], by_id[a]["road"]}
                steps.append((near_in[a], lambda s, t, nm=names: loop_path(s, t, nm)))
            steps.append((leg_out[b] if j == len(pairs) - 1 else near_out[b], path))
            for t, finder in steps:
                seg = finder(cur, t)
                if seg is None:
                    raise RuntimeError(f"ego 경로 {spec}: {cur.getID()}→{t.getID()} 연결 없음")
                full += seg if not full else seg[1:]
                cur = t
            prev_exit = b
        passes = sum(1 for e in full if net.getEdge(e).getToNode().getID() == center_id)
        if passes != len(pairs):
            raise RuntimeError(f"ego 경로 {spec}: 교차로를 {passes}번 지납니다 "
                               f"(의도: {len(pairs)}번). 다른 조합을 쓰세요.")
        return full

    meta["ego_demo_routes"] = []
    ego_edges = {}               # ego 경로 id -> 엣지 목록 (도로 디테일이 AV 출발 엣지를 비우는 데 쓴다)
    all_specs = ([(f"ego_{i}", s, "ego_routes") for i, s in enumerate(ego.get("routes", []))]
                 + [(f"ego_demo_{i}", s, "ego_demo_routes")
                    for i, s in enumerate(ego.get("demo_routes", []))])
    for rid, spec, key in all_specs:
        p = ego_path(spec)
        # 출발 위치 = 연산교차로까지 경로상 approach(m) 앞. 그 지점이 들어 있는 엣지부터 경로를 시작한다
        # (도로망 끝 엣지가 짧으면 첫 엣지 안에서는 그만큼 앞에 설 수 없음)
        to_center, edges = 0.0, [net.getEdge(e) for e in p]
        for e in edges:
            to_center += e.getLength()
            if e.getToNode().getID() == center_id or center_id in e.getToNode().getID():
                break
        d0, acc, k = max(to_center - approach, 0.0), 0.0, 0
        while k < len(edges) - 1 and acc + edges[k].getLength() <= d0:
            acc += edges[k].getLength()
            k += 1
        # 도로망 끝(dead_end)에서 바로 출발하면 그 자리가 배경 차량 투입 지점과 겹쳐, 교통이 많은 도로(W 등)에서는
        # 입구가 늘 막혀 ego를 넣지 못한다 → 다음 엣지에서 출발 (교차로 쪽으로 그 길이만큼 가까워짐)
        if (d0 - acc) <= 0.0 and k + 1 < len(edges) and edges[k].getFromNode().getType() == "dead_end" \
                and edges[k].getToNode().getID() != center_id:
            k += 1
            acc = d0
            # 너무 짧은 엣지(교차로 사이 조각)에서 출발하면 곧바로 정지선에 서서 "차로 갇힘"으로 잘못 끝날 수 있음
            while k + 1 < len(edges) and edges[k].getLength() < 60.0 and edges[k].getToNode().getID() != center_id:
                k += 1
        p = p[k:]
        route_lines.append(f'    <route id="{rid}" edges="{" ".join(p)}"/>')
        meta[key].append(rid)
        meta["ego_depart_pos"][rid] = float(min(max(d0 - acc, 0.0), max(edges[k].getLength() - 10.0, 0.0)))
        ego_edges[rid] = p

    # 도로 디테일: 택시·잠깐 정차·긴급차량 flow, 불법 주정차 차량 (켠 항목만, 기본 전부 끔)
    if road_details.any_enabled(details):
        curb.exclude |= {p_[0] for p_ in ego_edges.values()}       # AV 출발 엣지는 비워 둔다 (투입 실패 방지)
        if not use_od and (road_details.enabled(details, "taxi") or road_details.enabled(details, "short_stops")):
            raise ValueError("DETAILS의 taxi·short_stops는 실측 수요(TRAFFIC['demand'])가 켜져 있어야 씁니다.")
        vt, rl, fl, dm = road_details.extra_traffic(
            net, curb, od_paths, details, poisson, traffic.get("depart_speed", "max"), details_rng,
            ego_edges=[e for p_ in ego_edges.values() for e in p_[1:]], car_type=flow_type)
        vtypes_block += "".join("\n" + v for v in vt)
        route_lines += rl
        flow_lines += fl
        details_meta.update(dm)

    with open(os.path.join(out_dir, "highway.rou.xml"), "w", encoding="utf-8") as f:
        f.write(f"""<routes>
    <vType id="ego_type" accel="{ego['accel']}" decel="{ego['decel']}" maxSpeed="{ego['max_speed']}" speedFactor="1" speedDev="0" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>
{vtypes_block}
{chr(10).join(route_lines)}
{chr(10).join(flow_lines)}
</routes>
""")

    # ──────────────────────────────────────────────────────
    # 4) 중심 교차로 신호: 도로별 동시신호로 교체
    #    netconvert가 만든 신호(중심 + 바로 옆 OSM 신호 지점이 합쳐진 것)의
    #    연결마다 "어느 간선에서 들어온 차량인가"를 출발 엣지 위치로 판정해
    #    legs 순서대로 녹색을 준다.
    # ──────────────────────────────────────────────────────
    root = ET.parse(net_path).getroot()
    center_tl = None
    for c in root.iter("connection"):
        if c.get("tl") and net.getEdge(c.get("from")).getToNode().getID() == center_id:
            center_tl = c.get("tl")
            break
    if center_tl and signal.get("enabled", True):
        # 교차로를 이루는 노드들과 그 사이 내부 도로 (연산교차로는 노드 4개)
        tl_nodes = {cn[0].getEdge().getToNode().getID()
                    for cn in net.getTLS(center_tl).getConnections()}
        group = {}
        for c in root.iter("connection"):
            if c.get("tl") != center_tl:
                continue
            e = net.getEdge(c.get("from"))
            # 교차로 내부 도로의 링크(두 번째 정지선)는 항상 녹색: 진입 정지선을
            # 녹색에 통과한 차가 8m 뒤 내부 정지선의 적색에 갇히지 않게 한다.
            if (e.getFromNode().getID() in tl_nodes
                    and e.getToNode().getID() in tl_nodes):
                group[int(c.get("linkIndex"))] = None
                continue
            # 엣지 시작점(교차점에서 먼 쪽)의 방위각으로 간선 판정
            sx, sy = e.getShape()[0]
            b = _bearing(sx - cx, sy - cy)
            # 교차점에서 멀어지는 방향의 엣지(진출 차도)는 신호를 받을 필요가 없음
            ex, ey = e.getShape()[-1]
            outward = math.hypot(ex - cx, ey - cy) > math.hypot(sx - cx, sy - cy) + 1.0
            leg = min(legs, key=lambda lg: _angle_diff(b, lg["bearing"]))["id"]
            group[int(c.get("linkIndex"))] = None if outward else leg
        n = max(group) + 1
        yellow = float(signal.get("yellow", 3.0))
        all_red = float(signal.get("all_red", 2.0))
        # 녹색 시간: 실측 수요가 켜져 있으면 그 시나리오의 값(SIGNAL["green_by_scenario"]),
        # 없으면 ROAD["legs"]의 green
        by_scn = ((signal.get("green_by_scenario") or {}).get(demand.get("scenario"))
                  if use_od else None) or {}
        green = {lg["id"]: float(by_scn.get(lg["id"], lg.get("green", 20.0))) for lg in legs}
        meta["signal_green_s"] = green
        plans = signal.get("plans") or {}
        lines = []
        if plans:
            # 실제 신호 현시표 입력 (road_config.SIGNAL["plans"]) — 계획마다 SUMO 프로그램 하나
            exits = _link_exits(net, root, center_tl, center_id, legs)
            for pname, plan in plans.items():
                lines.append(f'    <tlLogic id="{center_tl}" type="static" programID="{pname}" '
                             f'offset="{float(plan.get("offset", 0.0)):g}">')
                for dur, st in plan_to_phases(plan, group, exits, n, signal, pname):
                    lines.append(f'        <phase duration="{dur:g}" state="{st}"/>')
                lines.append("    </tlLogic>")
            meta["signal_plans"] = {
                "tls": center_tl, "programs": list(plans),
                "active": signal.get("active_plan", list(plans)[0]),
                "tod": signal.get("tod") or []}
        else:
            lines.append(f'    <tlLogic id="{center_tl}" type="static" programID="custom" offset="0">')
        # 연산교차로 정밀 모델이면 자료의 진입로 안 차이(N 방면별 신호 따로 켜짐, SW 교대 방면 화살표 먼저 꺼짐)를 넣는다
        import json as _json
        x6_within, x6_lanes = {}, {}
        if cfg.get("x6_spec") and not plans:
            import sys as _sys
            if BASE not in _sys.path:
                _sys.path.insert(0, BASE)
            from tools.build_x6 import leg_slot, link_lane_ids
            x6_within = _json.load(open(os.path.join(BASE, cfg["x6_spec"]), encoding="utf-8"))["signal_plan_assumed"].get("within_leg") or {}
            x6_lanes = link_lane_ids(net_path, center_tl)
        # 연산교차로 정밀 모델에 동시 녹색 현시표(yeonsan_x6.json signal_plan_concurrent)가 있으면 그것을 쓴다
        #   (서로 안 엇갈리는 차로끼리 함께 녹색 — 진입로 하나씩만 녹색인 6현시보다 처리 능력이 큼).
        #   SIGNAL["concurrent"] = False면 예전 진입로별 동시신호.
        conc = None
        if cfg.get("x6_spec") and not plans and signal.get("concurrent", True):
            conc = _json.load(open(os.path.join(BASE, cfg["x6_spec"]), encoding="utf-8")).get("signal_plan_concurrent")
        if conc and conc.get("green"):
            from tools.build_x6 import concurrent_phases, green_conflicts
            scn = demand.get("scenario") if use_od else None
            cg = conc["green"].get(scn) or conc["green"].get("normal")
            cph = concurrent_phases(conc, cg, x6_lanes, n, float(conc.get("yellow", yellow)), float(conc.get("all_red", all_red)))
            bad = green_conflicts(net_path, center_tl, cph)
            if bad:
                raise RuntimeError(f"동시 녹색 현시에 엇갈리는 링크가 있음: {bad[:5]}")
            for dur, st in cph:
                lines.append(f'        <phase duration="{dur:g}" state="{st}"/>')
            meta["signal_green_s"] = cg
            meta["signal_structure"] = {"type": "concurrent", "phases": [p["name"] for p in conc["phases"]],
                                        "cycle_s": sum(d for d, _ in cph), "scenario": scn or "normal"}
            legs_loop = []
        else:
            legs_loop = [] if plans else legs
        for lg in legs_loop:
            lanes = {i: x6_lanes.get(i, (None, 0))[1] for i, g_ in group.items() if g_ == lg["id"]}
            for dur, st in leg_slot(green[lg["id"]], yellow, all_red, lanes, x6_within.get(lg["id"])):
                state = "".join("G" if group.get(i) is None else st.get(i, "r") for i in range(n))
                lines.append(f'        <phase duration="{dur:g}" state="{state}"/>')
        if not plans:
            lines.append("    </tlLogic>")
        # 버스전용차로(BRT)가 있으면: 녹색 버스 직진과 경로가 엇갈리는 승용차 링크는 양보 녹색(g)
        # (같은 진입로의 일반 차로 좌회전이 맨 안쪽 버스차로를 가로지르는 경우 등). 버스차로가 없으면 변화 없음.
        # 버스전용차로(BRT)가 있으면: 버스 직진과 엇갈리는 일반 이동류는 같은 순간 녹색이 되지 않게
        # 진입로 녹색 끝에 짧은 BRT 현시를 둔다 (env/bus_lanes.py). 버스차로가 없으면 변화 없음.
        from env.bus_lanes import bus_link_conflicts, bus_subphases
        bconf = bus_link_conflicts(net_path, center_tl) if float(signal.get("bus_green", 0.0)) > 0 else {}
        if bconf:
            bus_links = set().union(*bconf.values())
            new_lines, buf = [], []

            def flush():
                for d_, s_ in bus_subphases(buf, bconf, bus_links, float(signal.get("bus_green", 0.0)), yellow,
                                            approach={i for i, g_ in group.items() if g_}):
                    new_lines.append(f'        <phase duration="{d_:g}" state="{s_}"/>')
                buf.clear()
            for ln in lines:
                mt = re.search(r'<phase duration="([^"]*)" state="([^"]*)"', ln)
                if mt:
                    buf.append((float(mt.group(1)), mt.group(2)))
                else:
                    flush()
                    new_lines.append(ln)
            flush()
            lines = new_lines
            meta["bus_signal"] = {"conflicts": {str(k): sorted(v) for k, v in bconf.items()},
                                  "bus_green": float(signal.get("bus_green", 0.0))}
    else:
        lines = []

    # ──────────────────────────────────────────────────────
    # 4-1) extra_signals 교차로(예: 시청 인근)도 진입 방향별 동시신호로
    #    netconvert 자동 신호는 2단계 + 비보호 좌회전 + 전적색 없음이라
    #    녹색에 들어가도 교차로 안에서 다른 방향 차와 마주칠 수 있다.
    #    AV 관측에는 교차로 안을 가로지르는 차가 없으므로, 방향별로 한 번에
    #    한 진입로만 녹색을 주고 각 녹색 뒤에 황색·전적색을 넣는다.
    #    녹색 시간: extra_signal_plan["green_by_road"][도로 이름] 또는 기본값 (가정값)
    # ──────────────────────────────────────────────────────
    plan = cfg.get("extra_signal_plan") or {}
    green_by_road = plan.get("green_by_road", {})
    green_default = float(plan.get("green_default", 20.0))
    if signal.get("enabled", True):
        for jid in extra_ids:
            _jn = {c[0].getEdge().getToNode() for c in net.getTLS(jid).getConnections()}
            jx = sum(n.getCoord()[0] for n in _jn) / len(_jn)
            jy = sum(n.getCoord()[1] for n in _jn) / len(_jn)
            groups = []                        # [(이름, 방위각, [linkIndex...])]
            group_of = {}
            for c in root.iter("connection"):
                if c.get("tl") != jid:
                    continue
                e = net.getEdge(c.get("from"))
                sx, sy = e.getShape()[0]
                b, nm = _bearing(sx - jx, sy - jy), e.getName() or ""
                g = next((g for g in groups if g[0] == nm and _angle_diff(g[1], b) < 40.0),
                         None)
                if g is None:
                    g = (nm, b, [])
                    groups.append(g)
                g[2].append(int(c.get("linkIndex")))
            if not groups:
                continue
            groups.sort(key=lambda g: g[1])
            n = 1 + max(i for g in groups for i in g[2])
            yellow = float(signal.get("yellow", 3.0))
            all_red = float(signal.get("all_red", 2.0))
            lines.append(f'    <tlLogic id="{jid}" type="static" programID="custom" offset="0">')
            for nm, b, idx in groups:
                green = float(green_by_road.get(nm, green_default))
                for dur, ch in ((green, "G"), (yellow, "y"), (all_red, "r")):
                    if dur <= 0:
                        continue
                    state = "".join(ch if i in idx else "r" for i in range(n))
                    lines.append(f'        <phase duration="{dur:g}" state="{state}"/>')
            lines.append("    </tlLogic>")
            meta.setdefault("extra_tls", {})[jid] = [
                {"road": nm, "bearing": round(b), "green": float(green_by_road.get(nm, green_default))}
                for nm, b, _ in groups]

    if lines:
        with open(os.path.join(out_dir, "highway.tls.xml"), "w") as f:
            f.write("<additional>\n" + "\n".join(lines) + "\n</additional>\n")
        additional.append("highway.tls.xml")
    meta["center_tls"] = center_tl
    # 주변(상류) 신호 녹색 시간을 배경 교통 수요에 맞춰 다시 나눈다 (env/signals.demand_green_split, Webster).
    # 연산교차로 중심은 6현시 동시 신호 시간을 그대로 둔다 (수요 분배로 바꾸면 주기 156초 → 통과량이 오히려 줄었음).
    if signal.get("enabled", True) and signal.get("demand_split", False):
        from env.signals import demand_green_split
        tls_path = os.path.join(out_dir, "highway.tls.xml")
        ds = demand_green_split(net_path, os.path.join(out_dir, "highway.rou.xml"), skip=(center_tl,),
                                tls_add=tls_path if os.path.exists(tls_path) else None,
                                cycle_range=tuple(signal.get("demand_split_cycle", (60.0, 150.0))))
        meta["demand_split"] = {t: {"cycle": round(c, 1), "greens": [round(g, 1) for g in gs], "Y": round(y, 3)}
                                for t, (c, gs, y) in ds.items()}
    if not (cfg.get("merge_parts") or cfg.get("net_file")):
        meta["signal_build"] = signal_build
    meta["lane_rules"] = lane_table      # 실제 적용된 차로 → 진출 도로 표
    # 도로 경사: 마지막에 도로망 높이만 더한다 (연결·신호·차로는 위에서 끝난 그대로)
    if road_details.enabled(details, "slope"):
        details_meta["slope"] = road_details.apply_elevation(net_path, net, details)
    if road_details.any_enabled(details):
        details_meta["driver_styles"] = (list(((details.get("driver_styles") or {}).get("styles") or {}))
                                         if road_details.enabled(details, "driver_styles") else None)
        meta["details"] = {"enabled": road_details.tag(details), **details_meta}
    with open(os.path.join(out_dir, "highway.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # ──────────────────────────────────────────────────────
    # 5) GUI / sumocfg
    # ──────────────────────────────────────────────────────
    xmin, ymin, xmax, ymax = net.getBoundary()
    extent = max(xmax - xmin, ymax - ymin)
    view_width = float(SIMULATION.get("gui_view_width", 350.0))
    with open(os.path.join(out_dir, "highway.gui.xml"), "w") as f:
        f.write(f"""<viewsettings>
    <scheme name="real world">
        <!-- realisticLinkRules: 실제로 서거나 양보해야 하는 곳(신호·양보·일시정지)에만 정지선을 그린다.
             끄면 우선 도로가 지나가기만 하는 모든 노드(차로 수 바뀌는 곳, 옆길 붙는 곳)에도 흰 선이 그려진다. -->
        <edges showLinkRules="1" realisticLinkRules="1"/>
    </scheme>
    <viewport zoom="{extent / view_width * 100.0:.0f}" x="{cx:.1f}" y="{cy:.1f}" angle="0"/>
    <delay value="{float(SIMULATION.get("gui_delay", 100.0))}"/>
</viewsettings>
""")
    with open(os.path.join(out_dir, "highway.scenery.xml"), "w") as f:
        f.write("<additional>\n</additional>\n")
    additional.insert(0, "highway.scenery.xml")
    cfg_path = os.path.join(out_dir, "highway.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(f"""<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="highway.rou.xml"/>
        <additional-files value="{",".join(additional)}"/>
    </input>
    <gui_only>
        <gui-settings-file value="highway.gui.xml"/>
    </gui_only>
</configuration>
""")
    return cfg_path
