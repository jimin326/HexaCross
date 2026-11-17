"""차로 수 전수 검사: SUMO 도로망의 모든 엣지(방향별)를 원본 OSM 태그와 대조한다.

빌드 코드(build_area_map.py)와 따로, 원본 OSM 파일을 다시 읽어서 확인하는 독립 검사다.

  python tools/audit_lanes.py env/maps/yeonsan_area.json      → env/maps/yeonsan_area.lane_audit.csv

판정 (엣지 = 한 방향 도로 조각)
  일치            도로망 차로 수 = OSM에서 정해지는 차로 수
  일치(조사표)     로드뷰 조사표(lanes.csv 실제_*) 값과 일치 — OSM보다 우선
  일치(가정)       설정 lane_assumptions(설계 기준으로 정한 값)와 일치
  일치(보정)       설정 way_splits의 보정값과 일치
  불일치          ← 고쳐야 함
  OSM 모호        양방향 도로인데 lanes가 홀수이고 방향별 값이 없음 (어느 쪽이 많은지 OSM으로 알 수 없음)
  OSM 없음        lanes 태그 없음 → 도로 등급 기본값을 씀 (로드뷰 확인 필요)
합친 도로(중앙대로 ①⑤): 일반 차로 = 원래 두 차도의 OSM lanes, 버스전용 = 가운데 busway의 방향별 차로.
"""
import collections
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
ONEWAY = ("yes", "1", "true")


def expect(t):
    """원본 OSM 태그 → ((정방향, 역방향), 근거). 값을 모르면 None."""
    oneway = t.get("oneway") in ONEWAY or t.get("junction") in ("roundabout", "circular")
    if oneway:
        return ((int(t["lanes"]), 0), "lanes(일방)") if t.get("lanes", "").isdigit() else (None, "없음")
    f, b = t.get("lanes:forward", ""), t.get("lanes:backward", "")
    if f.isdigit() and b.isdigit():
        return (int(f), int(b)), "lanes:forward/backward"
    n = t.get("lanes", "")
    if n.isdigit():
        n = int(n)
        if f.isdigit():
            return (int(f), n - int(f)), "lanes - lanes:forward"
        if b.isdigit():
            return (n - int(b), int(b)), "lanes - lanes:backward"
        if n % 2 == 0:
            return (n // 2, n // 2), "lanes/2"
        return (None, f"모호(lanes={n}, 홀수)")
    return (None, "없음")


def main(cfg_path):
    import sumolib
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    out_dir = os.path.join(BASE, cfg.get("out_dir", "env/maps"))
    net = sumolib.net.readNet(os.path.join(out_dir, f"{name}.net.xml"))
    meta = json.load(open(os.path.join(out_dir, f"{name}.meta.json"), encoding="utf-8"))
    root = ET.parse(os.path.join(BASE, cfg["osm_file"])).getroot()
    tags = {w.get("id"): {x.get("k"): x.get("v") for x in w.iter("tag")} for w in root.iter("way")}
    survey = {}
    lane_csv = os.path.join(BASE, cfg.get("lane_table", f"env/maps/{name}.lanes.csv"))
    if os.path.exists(lane_csv):
        for row in csv.DictReader(open(lane_csv, encoding="utf-8-sig")):
            for k, d in (("실제_정방향", 0), ("실제_역방향", 1)):
                if (row.get(k) or "").strip().isdigit():
                    survey[(row["way_id"], d)] = int(row[k])
    assumed = {(w, 0): int(a["lanes"]) for w, a in (cfg.get("lane_assumptions") or {}).items()}
    splits = {"9" + s["way"]: s for s in cfg.get("way_splits", [])}
    merged = {}                                        # 합친 도로 조각 id 앞부분 → 기대값
    for md in meta.get("carriageway_merges", []):
        src = [tags[w] for w in md["merged"] if w in tags]
        bus = md["lanes"]["busway_evidence"]
        merged[md["new_id"]] = {"src_lanes": sorted({t.get("lanes") for t in src}), "busway": bus,
                                "gen": md["lanes"]["general_per_dir"], "bus": md["lanes"]["bus_per_dir"],
                                "segments": md["lanes"].get("segments"), "spans": md["lanes"].get("segment_src", {})}
    rows = []
    for e in net.getEdges():
        eid = e.getID()
        rev = eid.startswith("-")
        d = 1 if rev else 0
        wid = eid.lstrip("-").split("#")[0]
        n_net = e.getLaneNumber()
        n_bus = sum(1 for ln in e.getLanes() if ln.allows("bus") and not ln.allows("passenger"))
        row = {"edge": eid, "도로명": e.getName(), "way": wid, "방향": "역" if rev else "정", "길이_m": round(e.getLength()),
               "도로망_차로": n_net, "도로망_버스전용": n_bus}
        m = next((k for k in merged if wid.startswith(k)), None)
        if m:
            seg = (merged[m].get("segments") or {}).get(wid)
            g, b = ((seg[0], seg[2]) if d == 0 else (seg[1], seg[3])) if seg else (merged[m]["gen"][d], merged[m]["bus"][d])
            # 원본 근거: 같은 방향 원래 차도들의 OSM lanes (합친 구간 안에 걸친 것), busway lanes = 방향별 2배
            spans = merged[m]["spans"].get("정방향 차도(나감)" if d == 0 else "역방향 차도(들어옴)", [])
            src_l = sorted({L for a, b_, L in spans if b_ - a > 1})
            src_ok = src_l == [g] or not spans
            bw_ok = all(x["lanes"] == 2 * b for x in merged[m]["busway"]) if b else True
            row.update({"OSM_근거": f"차도 lanes={src_l} + busway {[(x['busway'], x['lanes']) for x in merged[m]['busway']]}",
                        "기대_차로": g + b, "기대_버스전용": b})
            ok = n_net == g + b and n_bus == b and src_ok and bw_ok
            row["판정"] = "일치(합친 도로)" if ok else "불일치"
        else:
            if wid in splits:
                t = {**tags.get(splits[wid]["way"], {}), **splits[wid].get("tail_tags", {})}
                src = "보정"
            else:
                t = tags.get(wid)
                src = None
            if t is None:
                row.update({"판정": "OSM에 없는 도로", "OSM_근거": ""})
                rows.append(row)
                continue
            exp, why = expect(t)
            row["OSM_근거"] = f"{why}: " + ", ".join(f"{k}={t[k]}" for k in ("lanes", "lanes:forward", "lanes:backward", "oneway", "junction") if k in t)
            if (wid, d) in survey:
                row.update({"기대_차로": survey[(wid, d)], "판정": "일치(조사표)" if n_net == survey[(wid, d)] else "불일치"})
            elif (wid, d) in assumed:
                row.update({"기대_차로": assumed[(wid, d)], "판정": "일치(가정)" if n_net == assumed[(wid, d)] else "불일치"})
            elif exp is None:
                row["판정"] = "OSM 모호" if why.startswith("모호") else "OSM 없음"
            else:
                row["기대_차로"] = exp[d]
                row["판정"] = ("일치(보정)" if src else "일치") if n_net == exp[d] else "불일치"
            # 버스전용: OSM bus:lanes의 designated 수
            key = "bus:lanes" if t.get("oneway") in ONEWAY else ("bus:lanes:backward" if rev else "bus:lanes:forward")
            exp_bus = (t.get(key) or "").split("|").count("designated")
            row["기대_버스전용"] = exp_bus
            if n_bus != exp_bus:
                row["판정"] = "불일치"
        rows.append(row)
    cols = ["edge", "도로명", "way", "방향", "길이_m", "도로망_차로", "기대_차로", "도로망_버스전용", "기대_버스전용", "판정", "OSM_근거"]
    out = os.path.join(out_dir, f"{name}.lane_audit.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["판정"], r["도로명"] or "~", r["edge"])):
            w.writerow({k: r.get(k, "") for k in cols})
    cnt = collections.Counter(r["판정"] for r in rows)
    length = collections.defaultdict(float)
    for r in rows:
        length[r["판정"]] += r["길이_m"]
    print(f"엣지 {len(rows)}개 (방향별 도로 조각) → {out}")
    for k, v in sorted(cnt.items()):
        print(f"  {k:12s} {v:4d}개  {length[k] / 1000:5.2f} km")
    bad = [r for r in rows if r["판정"] == "불일치"]
    for r in bad:
        print("  불일치:", r)
    unknown = collections.defaultdict(float)
    for r in rows:
        if r["판정"] in ("OSM 없음", "OSM 모호"):
            unknown[(r["도로명"] or "(이름없음)", r["판정"], r["도로망_차로"])] += r["길이_m"]
    print("  확인 필요(도로명, 사유, 지금 차로 수 → 길이):")
    for (nm, why, n), L in sorted(unknown.items(), key=lambda x: -x[1]):
        print(f"    {nm:8s} {why:8s} {n}차로  {L:5.0f} m")
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "env/maps/yeonsan_area.json"))
