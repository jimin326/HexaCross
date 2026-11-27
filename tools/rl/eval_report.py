"""평가 결과(JSON) → 발표용 표·신뢰구간·그래프.

usage:
    python tools/rl/eval_report.py OUT_DIR LABEL=결과.json [LABEL2=결과2.json ...]

예:
    python tools/rl/eval_report.py results/eval_report \\
        BC=models/yeonsan_7routes_BC/eval_560.json \\
        PPO_C0=models/yeonsan_7routes_C0/eval_20per_route.json

입력: tools/rl/eval_routes.py 또는 tools/rl/eval_parallel.py가 만든 JSON.
    {"경로 이름": {"episodes": N, "insert_fail": k, "arrival": a, "collision": c,
                  "wrong_lane": w, "red_light": r, "teleported": p, "timeout": t, ["metrics": {...}]}, ...}
    (teleported: AV가 300초 넘게 못 움직여 SUMO가 순간이동시켜 끝남. 없는 예전 결과는 0으로 본다)
    eval_routes.py 형식(경로가 맨 위) 그대로 읽는다. eval_parallel.py 결과에는
    "_meta"(조건 기록)와 경로별 "metrics"(주행 품질)가 더 들어 있다.
    여러 조건을 담은 models/*/eval.json은 'LABEL=파일.json#조건'으로 하나를 고른다.
    예: J8_RL=models/yeonsan_7routes_J8_RL/eval.json#scale_0.7_bus_demandsplit_seeds921-924
    (조건 이름을 모르면 #없이 주면 목록을 보여 준다.)

출력 (OUT_DIR):
    report.md        결과 요약 표 (모델별 합계, 경로별, 95% 신뢰구간, 기준 판정)
    summary.csv      모델별 합계 비율과 신뢰구간
    per_route.csv    모델 × 경로별 횟수와 비율
    fig_compare.png  모델별 완주·충돌·신호 위반·차로 갇힘 비율 (신뢰구간 막대, 충돌 5% 기준선)
    fig_routes_<LABEL>.png  경로별로 주행이 끝난 이유 (가로 누적 막대)

신뢰구간: Wilson score 95%. 평가 횟수가 적을수록 구간이 넓다.
기준 판정: 수업 평가 기준 "Collision Rate < 5%" (수업 프로젝트 평가 자료).
    충돌률 신뢰구간 상한까지 5% 미만이면 "통과", 점추정만 5% 미만이면 "경계".
"""
import csv
import json
import math
import os
import sys

OUTCOMES = [  # (키, 한국어, 색)
    ("arrival", "완주", "#2E7D5B"),
    ("collision", "충돌", "#C0392B"),
    ("red_light", "신호 위반", "#E07A2E"),
    ("wrong_lane", "차로 갇힘", "#3A6EA5"),
    ("teleported", "막힘(순간이동)", "#7B5EA7"),
    ("timeout", "시간 초과", "#8A8F98"),
]
COLLISION_LIMIT = 0.05


def wilson(k, n, z=1.96):
    """Wilson score 95% 신뢰구간 (k/n). n=0이면 (nan, nan)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def load(path):
    """결과 JSON을 읽는다. 'file.json#조건'이면 여러 조건을 담은 파일에서 그 조건만 고른다."""
    path, _, cond = path.partition("#")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    meta = d.pop("_meta", {})
    # 여러 조건을 담은 형식 (models/*/eval.json):
    #   {"map": "설명", "scale_1.0_seeds921-924": {"per_route": {...}, "total": {...}, "tags": [...]}, ...}
    conds = [k for k, v in d.items() if isinstance(v, dict) and isinstance(v.get("per_route"), dict)]
    if conds and not isinstance(d.get("per_route"), dict):
        if not cond:
            if len(conds) > 1:
                sys.exit(f"{path}에 평가 조건이 {len(conds)}개 있습니다. 'LABEL={path}#조건' 으로 하나를 고르세요:\n  "
                         + "\n  ".join(conds))
            cond = conds[0]
        if cond not in conds:
            sys.exit(f"{path}에 조건 '{cond}'가 없습니다. 있는 조건:\n  " + "\n  ".join(conds))
        meta = {**meta, "condition": cond}
        if isinstance(d.get("map"), str):
            meta["map"] = d["map"]
        d = d[cond]
    elif cond:
        sys.exit(f"{path}는 조건이 하나뿐인 형식이라 '#{cond}'를 쓸 수 없습니다.")
    # 여러 시드를 합친 형식: {"seeds": [...], "per_route": {...}, "total": {...}}
    if isinstance(d.get("per_route"), dict):
        if "seeds" in d:
            meta = {**meta, "seeds": d["seeds"]}
        d = d["per_route"]
    routes = {}
    for name, r in d.items():
        if not isinstance(r, dict):
            continue
        counts = {k: int(r.get(k, 0)) for k, _, _ in OUTCOMES}
        counts["episodes"] = int(r.get("episodes", sum(counts.values())))
        counts["insert_fail"] = int(r.get("insert_fail", 0))
        routes[name] = {"counts": counts, "metrics": r.get("metrics", {})}
    return meta, routes


def total(routes):
    t = {k: 0 for k, _, _ in OUTCOMES}
    t["episodes"] = t["insert_fail"] = 0
    for r in routes.values():
        for k in t:
            t[k] += r["counts"].get(k, 0)
    return t


def pct(x):
    return "–" if x != x else f"{100 * x:.1f}%"


def verdict(c, n):
    lo, hi = wilson(c, n)
    p = c / n if n else float("nan")
    if n == 0:
        return "평가 없음"
    if hi < COLLISION_LIMIT:
        return f"통과 (신뢰구간 상한 {pct(hi)} < 5%)"
    if p < COLLISION_LIMIT:
        return f"경계 (점추정 {pct(p)}는 5% 미만, 상한 {pct(hi)}는 5% 이상 — 평가를 늘리거나 충돌을 더 줄여야 함)"
    return f"미달 (점추정 {pct(p)} ≥ 5%)"


def mean_metric(routes, key):
    """경로별 metrics의 평균을 episode 수로 가중 평균 (있을 때만)."""
    num = den = 0.0
    for r in routes.values():
        m = r["metrics"].get(key)
        n = r["metrics"].get(key + "_n", r["counts"]["episodes"])
        if m is None or n == 0:
            continue
        num += m * n
        den += n
    return num / den if den else None


def setup_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def fig_compare(models, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    keys = [o for o in OUTCOMES if o[0] != "timeout"]
    labels = list(models)
    w = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, lab in enumerate(labels):
        t = total(models[lab][1])
        n = t["episodes"]
        xs, ys, lo, hi = [], [], [], []
        for j, (k, _, _) in enumerate(keys):
            p = t[k] / n if n else 0
            a, b = wilson(t[k], n)
            xs.append(j + (i - (len(labels) - 1) / 2) * w)
            ys.append(100 * p); lo.append(100 * (p - a)); hi.append(100 * (b - p))
        ax.bar(xs, ys, w * 0.92, label=f"{lab} (n={n})", alpha=0.9)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#222", capsize=3, lw=1)
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + 1.0, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(100 * COLLISION_LIMIT, color="#C0392B", ls="--", lw=1)
    ax.text(len(keys) - 0.5, 100 * COLLISION_LIMIT + 0.8, "충돌률 기준 5%", color="#C0392B",
            ha="right", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([kr for _, kr, _ in keys])
    ax.set_ylabel("비율 (%)  · 막대 끝 선 = 95% 신뢰구간")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.set_title("모델별 주행 결과")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_routes(label, routes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    names = list(routes)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(names) + 1.6))
    left = [0.0] * len(names)
    for k, kr, color in OUTCOMES:
        vals = []
        for nm in names:
            c = routes[nm]["counts"]
            vals.append(100 * c[k] / c["episodes"] if c["episodes"] else 0)
        ax.barh(names, vals, left=left, color=color, label=kr)
        for y, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9)
        left = [l + v for l, v in zip(left, vals)]
    for y, nm in enumerate(names):
        ax.text(101, y, f"n={routes[nm]['counts']['episodes']}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("비율 (%)")
    ax.set_title(f"{label}: 경로별로 주행이 끝난 이유", pad=30)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    models = {}
    for a in argv[1:]:
        lab, path = a.split("=", 1) if "=" in a else (os.path.splitext(os.path.basename(a))[0], a)
        models[lab] = load(path)

    # ── CSV ──
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "episodes", "insert_fail"]
                   + [x for k, _, _ in OUTCOMES for x in (k, f"{k}_rate", f"{k}_ci_lo", f"{k}_ci_hi")])
        for lab, (_, routes) in models.items():
            t = total(routes)
            row = [lab, t["episodes"], t["insert_fail"]]
            for k, _, _ in OUTCOMES:
                lo, hi = wilson(t[k], t["episodes"])
                row += [t[k], round(t[k] / t["episodes"], 4) if t["episodes"] else "", round(lo, 4), round(hi, 4)]
            w.writerow(row)
    with open(os.path.join(out_dir, "per_route.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "route", "episodes", "insert_fail"] + [k for k, _, _ in OUTCOMES]
                   + [f"{k}_rate" for k, _, _ in OUTCOMES])
        for lab, (_, routes) in models.items():
            for nm, r in routes.items():
                c = r["counts"]; n = c["episodes"]
                w.writerow([lab, nm, n, c["insert_fail"]] + [c[k] for k, _, _ in OUTCOMES]
                           + [round(c[k] / n, 4) if n else "" for k, _, _ in OUTCOMES])

    # ── 그래프 ──
    figs = []
    try:
        fig_compare(models, os.path.join(out_dir, "fig_compare.png"))
        figs.append("fig_compare.png")
        for lab, (_, routes) in models.items():
            fn = f"fig_routes_{lab}.png"
            fig_routes(lab, routes, os.path.join(out_dir, fn))
            figs.append(fn)
    except ImportError as e:
        print("[안내] matplotlib가 없어 그래프를 건너뜁니다:", e)

    # ── report.md ──
    L = ["# 주행 평가 결과", ""]
    L.append("| 모델 | 평가 횟수 | " + " | ".join(kr for _, kr, _ in OUTCOMES) + " | 충돌률 기준(5% 미만) |")
    L.append("|---|---:|" + "---|" * len(OUTCOMES) + "---|")
    for lab, (_, routes) in models.items():
        t = total(routes); n = t["episodes"]
        cells = []
        for k, _, _ in OUTCOMES:
            lo, hi = wilson(t[k], n)
            cells.append(f"{t[k]} ({pct(t[k] / n if n else float('nan'))}, {pct(lo)}~{pct(hi)})")
        L.append(f"| {lab} | {n} | " + " | ".join(cells) + f" | {verdict(t['collision'], n)} |")
    L += ["", "괄호 안: 비율, 95% 신뢰구간 (Wilson). 투입 실패(AV가 도로에 못 들어감)는 평가 횟수에서 뺐다.", ""]

    # 주행 품질 (eval_parallel 결과에만 있음)
    qual = [("travel_time_s", "완주 시 이동 시간 (s)"), ("mean_speed", "평균 속도 (m/s)"),
            ("stop_time_s", "정지 시간 (s/주행)"), ("min_gap", "최소 차간거리 평균 (m)")]
    if any(mean_metric(r, k) is not None for _, r in models.values() for k, _ in qual):
        L += ["## 주행 품질", "", "| 모델 | " + " | ".join(kr for _, kr in qual) + " |",
              "|---|" + "---:|" * len(qual)]
        for lab, (_, routes) in models.items():
            vals = [mean_metric(routes, k) for k, _ in qual]
            L.append(f"| {lab} | " + " | ".join("–" if v is None else f"{v:.1f}" for v in vals) + " |")
        L.append("")

    for lab, (meta, routes) in models.items():
        L += [f"## {lab}: 경로별", ""]
        if meta:
            L.append("조건: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
            L.append("")
        L.append("| 경로 | 평가 | " + " | ".join(kr for _, kr, _ in OUTCOMES) + " | 투입 실패 |")
        L.append("|---|---:|" + "---:|" * len(OUTCOMES) + "---:|")
        for nm, r in routes.items():
            c = r["counts"]; n = c["episodes"]
            f = lambda k: f"{c[k]} ({pct(c[k] / n)})" if n else "–"
            L.append(f"| {nm} | {n} | " + " | ".join(f(k) for k, _, _ in OUTCOMES) + f" | {c['insert_fail']} |")
        L.append("")
    if figs:
        L += ["## 그래프", ""] + [f"![{fn}]({fn})" for fn in figs] + [""]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:4 + len(models)]))   # 제목, 빈 줄, 표 머리 2줄 + 모델마다 1줄
    print(f"\n저장: {out_dir}/report.md, summary.csv, per_route.csv" + (", " + ", ".join(figs) if figs else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
