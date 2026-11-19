"""SUMO 맵 나누기 / 합치기 도구.

연산교차로(6지)를 정교하게 만드는 사람과 주변부 도로망을 만드는 사람이
따로 작업한 맵을 하나로 합칠 때 쓴다. 맵은 plain XML 묶음으로 주고받는다:
    <prefix>.nod.xml  노드      <prefix>.edg.xml  도로(edge)
    <prefix>.con.xml  차로 연결 <prefix>.tll.xml  신호      <prefix>.typ.xml  도로 유형
(netconvert --plain-output-prefix 가 만드는 형식)

사용법 (저장소 루트에서):
    # net.xml → plain XML
    python tools/merge_nets.py export env/sumo/highway.net.xml env/maps/base

    # 기준 맵을 "교차로 영역(반경 R 안)"과 "주변부"로 나누기
    #   경계 원을 가로지르는 도로에는 분할점(seam_*)을 넣어 경계가 도로 중간에 오게 한다
    python tools/merge_nets.py split env/sumo/highway.net.xml --center 414663644 \\
        --radius 150 --inner env/maps/x6 --outer env/maps/surround

    # 여러 맵 합치기 → net.xml
    python tools/merge_nets.py merge env/maps/merged.net.xml env/maps/surround env/maps/x6

합칠 때 지켜야 할 것 (merge가 검사한다):
  1. 좌표계가 같아야 한다 (nod.xml의 <location netOffset/projParameter>).
     같은 OSM 기반 맵에서 출발해 netedit로 고치면 자동으로 맞는다.
  2. 경계점(두 맵이 만나는 노드)은 같은 id·같은 좌표여야 한다.
     한쪽 맵에서 도로가 끊긴 끝점(dead_end)이면 다른 쪽 정의를 쓴다.
  3. 경계점은 도로 중간에 둔다. 교차로(특히 신호 교차로)를 경계로 자르면
     그 교차로의 차로 연결/신호가 양쪽으로 나뉘어 다시 계산된다 (경고 출력).
  4. 같은 id의 도로가 양쪽에 있으면 안 된다 (새로 만든 도로는 접두어 권장: x6_, sur_).
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

PARTS = ("nod", "edg", "con", "tll", "typ")


def _netconvert():
    try:
        from sumolib import checkBinary
        return checkBinary("netconvert")
    except Exception:
        return "netconvert"


def _run(args):
    res = subprocess.run([_netconvert()] + args, capture_output=True, text=True)
    errs = [l for l in (res.stdout + res.stderr).splitlines() if l.startswith("Error")]
    if res.returncode != 0:
        raise RuntimeError("netconvert 실패:\n  " + "\n  ".join(errs[:10] or [res.stderr[-2000:]]))
    return res


def _files(prefix):
    return {p: f"{prefix}.{p}.xml" for p in PARTS if os.path.exists(f"{prefix}.{p}.xml")}


# ──────────────────────────────────────────────────────────────
# export
# ──────────────────────────────────────────────────────────────
def export(net_file: str, out_prefix: str):
    """net.xml → plain XML 묶음 (out_prefix.nod.xml 등)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _run(["-s", net_file, "--plain-output-prefix", out_prefix,
              "--output.street-names", "-o", os.path.join(tmp, "x.net.xml")])
    return _files(out_prefix)


# ──────────────────────────────────────────────────────────────
# split
# ──────────────────────────────────────────────────────────────
def split(net_file: str, center: str, radius: float, inner_prefix: str, outer_prefix: str,
          cut: bool = True) -> dict:
    """맵을 중심 노드 반경 radius 안(inner)과 밖(outer)으로 나눈다.

    cut=True면 경계 원을 가로지르는 도로에 분할점(seam_N)을 넣어 경계가
    도로 중간에 오게 한다 (교차로 끝 8m 이내로 가로지르는 도로는 그대로).
    도로는 중간점 위치로 inner/outer를 정한다.
    """
    import sumolib
    net = sumolib.net.readNet(net_file)
    cx, cy = net.getNode(center).getCoord()
    with tempfile.TemporaryDirectory() as tmp:
        base = net_file
        if cut:
            full = os.path.join(tmp, "full")
            export(net_file, full)
            tree = ET.parse(f"{full}.edg.xml")
            renamed, k = {}, 0
            for ed in tree.getroot().iter("edge"):
                e = net.getEdge(ed.get("id"))
                sh = e.getShape()
                length = sum(math.dist(sh[i], sh[i + 1]) for i in range(len(sh) - 1))
                acc = 0.0
                for i in range(len(sh) - 1):
                    d0, d1 = math.dist(sh[i], (cx, cy)), math.dist(sh[i + 1], (cx, cy))
                    seg = math.dist(sh[i], sh[i + 1])
                    if (d0 - radius) * (d1 - radius) < 0:
                        pos = acc + seg * (d0 - radius) / (d0 - d1)
                        if 8.0 < pos < length - 8.0:
                            eid = ed.get("id")
                            renamed[eid] = eid + "_s"
                            ET.SubElement(ed, "split", pos=f"{pos:.2f}", id=f"seam_{k}",
                                          idBefore=eid, idAfter=eid + "_s")
                            k += 1
                        break
                    acc += seg
            tree.write(f"{full}.edg.xml", encoding="UTF-8", xml_declaration=True)
            # 도로 끝에서 나가는 연결은 뒷조각(새 id)에서 나가도록
            for p in ("con", "tll"):
                if not os.path.exists(f"{full}.{p}.xml"):
                    continue
                t = ET.parse(f"{full}.{p}.xml")
                for c in t.getroot().iter("connection"):
                    if c.get("from") in renamed:
                        c.set("from", renamed[c.get("from")])
                t.write(f"{full}.{p}.xml", encoding="UTF-8", xml_declaration=True)
            base = os.path.join(tmp, "base.net.xml")
            _run(_plain_args(_files(full)) + ["--output.street-names", "-o", base])
            net = sumolib.net.readNet(base)
        mid = lambda e: e.getShape()[len(e.getShape()) // 2]
        inner = {e.getID() for e in net.getEdges() if math.dist(mid(e), (cx, cy)) < radius}
        outer = {e.getID() for e in net.getEdges()} - inner
        ends = lambda es: {n.getID() for e in net.getEdges() if e.getID() in es
                           for n in (e.getFromNode(), e.getToNode())}
        seam = ends(inner) & ends(outer)
        # 경계점에서 두 맵을 가로지르는 차로 연결 (어느 쪽 con.xml에도 안 들어감) → 따로 저장
        cross = []
        for nid in seam:
            for e_in in net.getNode(nid).getIncoming():
                for cl in e_in.getOutgoing().values():
                    for c in cl:
                        a, b = c.getFrom().getID(), c.getTo().getID()
                        if (a in inner) != (b in inner):
                            cross.append((a, b, c.getFromLane().getIndex(), c.getToLane().getIndex()))
        os.makedirs(os.path.dirname(os.path.abspath(outer_prefix)), exist_ok=True)
        with open(f"{outer_prefix}.seam.con.xml", "w") as f:
            f.write("<connections>\n")
            for a, b, fl, tl in sorted(cross):
                f.write(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n')
            f.write("</connections>\n")
        for prefix, keep in ((inner_prefix, inner), (outer_prefix, outer)):
            lst = os.path.join(tmp, "keep.txt")
            with open(lst, "w") as f:
                f.write("\n".join(sorted(keep)))
            os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
            _run(["-s", base, "--keep-edges.input-file", lst, "--plain-output-prefix", prefix,
                  "--output.street-names", "-o", os.path.join(tmp, "part.net.xml")])
    junction_seams = sorted(n for n in seam if len(
        {e.getFromNode().getID() for e in net.getNode(n).getIncoming()}
        | {e.getToNode().getID() for e in net.getNode(n).getOutgoing()}) > 2
        or net.getNode(n).getType() == "traffic_light")
    report = {"inner_edges": len(inner), "outer_edges": len(outer), "seam_nodes": len(seam),
              "junction_seams": junction_seams, "seam_connections": len(cross)}
    print(f"[split] 교차로 영역 도로 {len(inner)} | 주변부 도로 {len(outer)} | 경계점 {len(seam)}개 "
          f"| 경계를 가로지르는 차로 연결 {len(cross)}개 → {outer_prefix}.seam.con.xml")
    if junction_seams:
        print(f"[split] 경고: 경계점 {len(junction_seams)}개가 교차로 위에 있습니다 "
              f"(합칠 때 그 교차로의 연결/신호가 다시 계산됨). 반경을 바꿔 보세요: "
              f"{junction_seams[:8]}{' ...' if len(junction_seams) > 8 else ''}")
    return report


def _plain_args(files):
    args = []
    for p, opt in (("nod", "--node-files"), ("edg", "--edge-files"), ("con", "--connection-files"),
                   ("tll", "--tllogic-files"), ("typ", "--type-files")):
        lst = files[p] if isinstance(files.get(p), list) else ([files[p]] if p in files else [])
        if lst:
            args += [opt, ",".join(lst)]
    return args


# ──────────────────────────────────────────────────────────────
# merge
# ──────────────────────────────────────────────────────────────
def merge_parts(prefixes, out_net: str, tol: float = 0.5, strict: bool = True) -> dict:
    """plain XML 맵 여러 개를 하나의 net.xml로 합친다. 검사 결과(report)를 반환.

    strict=True면 좌표계 불일치 / 경계점 좌표 어긋남 / 도로 id 충돌을 오류로 처리.
    """
    parts = [_files(p) for p in prefixes]
    for p, f in zip(prefixes, parts):
        if "nod" not in f or "edg" not in f:
            raise FileNotFoundError(f"{p}.nod.xml / {p}.edg.xml 이 필요합니다.")
    problems, warnings = [], []

    # 1) 좌표계
    locs = []
    for p, f in zip(prefixes, parts):
        loc = ET.parse(f["nod"]).getroot().find("location")
        locs.append((p, None if loc is None else (loc.get("netOffset"), loc.get("projParameter"))))
    ref = locs[0][1]
    for p, l in locs[1:]:
        if l != ref:
            problems.append(f"좌표계가 다릅니다: {locs[0][0]} {ref} vs {p} {l} "
                            f"(같은 OSM 기반 맵에서 출발해야 경계가 맞습니다)")

    # 2) 노드 (경계점 규칙)
    nodes, order, owner = {}, [], {}
    for p, f in zip(prefixes, parts):
        for n in ET.parse(f["nod"]).getroot().iter("node"):
            nid = n.get("id")
            if nid not in nodes:
                nodes[nid], owner[nid] = n, {p}
                order.append(nid)
                continue
            owner[nid].add(p)
            old = nodes[nid]
            d = math.hypot(float(old.get("x")) - float(n.get("x")),
                           float(old.get("y")) - float(n.get("y")))
            if d > tol:
                problems.append(f"경계점 '{nid}' 좌표가 {d:.2f}m 어긋남")
            if old.get("type") == "dead_end" and n.get("type") != "dead_end":
                nodes[nid] = n
            elif old.get("type") != "dead_end" and n.get("type") != "dead_end" \
                    and old.get("type") != n.get("type"):
                warnings.append(f"경계점 '{nid}' 종류가 다름: {old.get('type')} / {n.get('type')} "
                                f"(앞쪽 맵 정의 사용)")
    seams = [nid for nid in order if len(owner[nid]) > 1]

    # 3) 도로 id 충돌 + 경계점 차로 수
    edges, by_node = {}, {}
    for p, f in zip(prefixes, parts):
        for e in ET.parse(f["edg"]).getroot().iter("edge"):
            eid = e.get("id")
            if eid in edges:
                problems.append(f"도로 id 충돌: '{eid}' ({edges[eid][0]} / {p})")
            edges[eid] = (p, e)
            nl = len(list(e.iter("lane"))) or int(e.get("numLanes", 1))
            for nd, side in ((e.get("from"), "out"), (e.get("to"), "in")):
                by_node.setdefault(nd, []).append((p, side, eid, nl))
    junction_seams = []
    for nid in seams:
        es = by_node.get(nid, [])
        nbrs = {x[2] for x in es}
        if len(nbrs) > 2 or nodes[nid].get("type") == "traffic_light":
            junction_seams.append(nid)
            continue
        ins = [x for x in es if x[1] == "in"]
        outs = [x for x in es if x[1] == "out"]
        for a in ins:
            for b in outs:
                if a[0] != b[0] and a[3] != b[3]:
                    warnings.append(f"경계점 '{nid}' 차로 수 불일치: {a[2]}({a[3]}차로) → "
                                    f"{b[2]}({b[3]}차로)")
    # 경계 연결: split이 만든 <prefix>.seam.con.xml 이 있으면 그대로 쓰고,
    # 없으면 "경계점에서 다른 맵 도로로 가는 연결이 빠진 도로"의 연결을 지워서
    # netconvert가 그 도로의 연결을 통째로 다시 계산하게 한다 (도로가 끊기는 것 방지).
    seam_con = [f"{p}.seam.con.xml" for p in prefixes if os.path.exists(f"{p}.seam.con.xml")]
    given = set()
    for f in seam_con:
        for c in ET.parse(f).getroot().iter("connection"):
            given.add(c.get("from"))
    reguess = set()
    # split이 만든 경계 연결 파일이 있으면 그게 완전한 목록이므로 다시 계산하지 않는다
    for nid in ([] if seam_con else junction_seams):
        es = by_node.get(nid, [])
        for p_in, side, eid, _ in es:
            if side != "in" or eid in given:
                continue
            other_out = [x for x in es if x[1] == "out" and x[0] != p_in]
            if other_out:
                reguess.add(eid)
    if junction_seams:
        warnings.append(f"경계점 {len(junction_seams)}개가 교차로 위에 있음: {junction_seams[:8]}"
                        + (f" → 경계 연결 파일 사용 ({len(given)}개 도로)" if given else "")
                        + (f" → 도로 {len(reguess)}개는 연결을 netconvert가 다시 계산 "
                           f"(그 도로의 차로별 진행방향은 기본 추정값이 됨): {sorted(reguess)[:6]}"
                           if reguess else ""))

    for w in warnings:
        print("[merge] 경고:", w)
    if problems:
        msg = "\n  ".join(problems[:20])
        if strict:
            raise ValueError("맵을 합칠 수 없습니다:\n  " + msg)
        for pr in problems:
            print("[merge] 문제(무시하고 진행):", pr)

    # 4) 합친 노드 파일 + netconvert
    os.makedirs(os.path.dirname(os.path.abspath(out_net)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = ET.parse(parts[0]["nod"]).getroot()
        for ch in [c for c in root if c.tag == "node"]:
            root.remove(ch)
        for nid in order:
            root.append(nodes[nid])
        nod = os.path.join(tmp, "merged.nod.xml")
        ET.ElementTree(root).write(nod, encoding="UTF-8", xml_declaration=True)
        files = {"nod": [nod]}
        for p in ("edg", "con", "tll", "typ"):
            files[p] = [f[p] for f in parts if p in f]
        if reguess:                      # 끊긴 경계 도로의 연결 정의를 빼서 다시 계산되게
            filtered = []
            for i, cf in enumerate(files["con"]):
                t = ET.parse(cf)
                r = t.getroot()
                for c in [c for c in r if c.tag == "connection" and c.get("from") in reguess]:
                    r.remove(c)
                out = os.path.join(tmp, f"con{i}.con.xml")
                t.write(out, encoding="UTF-8", xml_declaration=True)
                filtered.append(out)
            files["con"] = filtered
        files["con"] += seam_con
        _run(_plain_args(files) + ["--output.street-names", "-o", out_net])
    report = {"nodes": len(order), "seams": len(seams), "junction_seams": junction_seams,
              "seam_connections": len(given), "reguessed": sorted(reguess),
              "edges": len(edges), "warnings": warnings, "problems": problems}
    print(f"[merge] {out_net}: 도로 {len(edges)} | 노드 {len(order)} | 경계점 {len(seams)} "
          f"(교차로 위 {len(junction_seams)})")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("export", help="net.xml → plain XML")
    a.add_argument("net"); a.add_argument("out_prefix")
    b = sub.add_parser("split", help="중심 반경으로 교차로/주변부 나누기")
    b.add_argument("net"); b.add_argument("--center", required=True)
    b.add_argument("--radius", type=float, default=150.0)
    b.add_argument("--inner", required=True); b.add_argument("--outer", required=True)
    b.add_argument("--no-cut", action="store_true", help="경계 도로에 분할점을 넣지 않음")
    c = sub.add_parser("merge", help="plain XML 맵들 합치기")
    c.add_argument("out_net"); c.add_argument("prefixes", nargs="+")
    c.add_argument("--tol", type=float, default=0.5, help="경계점 좌표 허용 오차 (m)")
    c.add_argument("--force", action="store_true", help="검사 문제가 있어도 합치기")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        print(export(args.net, args.out_prefix))
    elif args.cmd == "split":
        split(args.net, args.center, args.radius, args.inner, args.outer, cut=not args.no_cut)
    else:
        merge_parts(args.prefixes, args.out_net, tol=args.tol, strict=not args.force)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
