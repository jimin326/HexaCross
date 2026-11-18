"""실측 교통량 대비 시뮬레이션 교통량 검증 (레포 루트에서 실행).

    python docs/research/check_demand.py               # 3회 × 1시간
    python docs/research/check_demand.py --runs 1 --build

배경 교통만 돌린다 (AV 없음). 워밍업(기본 15분) 뒤 1시간 동안 연산교차로를
지난 차를 방향별로 세어, TRAFFIC["demand"]의 실측 OD(부산시 2025 교통조사,
이륜차·버스 제외 설정 그대로)와 비교한다.

세는 방법: SUMO vehroute 출력(차량마다 지난 edge와 edge별 통과 시각)에서
연산교차로로 들어가는 도로(정지선 edge 또는 우회전 연결로 앞) → 나가는 도로로
바뀌는 순간을 한 번의 통과로 센다. 도로 번호는 sumo_env의 _state_layout
(DATA_FORMAT v1 번호표)과 같다.

출력:
    A. 방향별 실측 / 시뮬레이션 (대/h, 오차)
    B. 진입부별 합계
    C. 막힘 지표 (teleport, 투입 대기 차량, 평균 대기 시간)
"""
import argparse
import collections
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.getcwd())  # 레포 루트에서 실행하므로 env 패키지를 찾게 함
from sumolib import checkBinary
from env import road_builder
from env.road_builder import load_od_demand
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from env.sumo_env import SumoHighwayEnv

NET_DIR = os.path.join("env", "sumo")
ORDER = SumoHighwayEnv.LEG_ORDER        # 0~5 = SW, N, W, S, E, NE


def count_movements(vehroute_path, edge_leg, stop_edges, t0, t1):
    """vehroute 출력에서 [t0, t1) 동안 연산교차로를 지난 (진입, 진출) 횟수."""
    counts = collections.Counter()
    for _, el in ET.iterparse(vehroute_path, events=("end",)):
        if el.tag != "vehicle":
            continue
        route = el.find("route")
        if route is None or not route.get("exitTimes"):
            el.clear()
            continue
        edges = route.get("edges").split()
        times = [float(x) for x in route.get("exitTimes").split()]
        last_in = None                    # (진입 번호, edge 위치)
        for i, e in enumerate(edges):
            code = edge_leg.get(e, -1)
            if 0 <= code <= 5:
                last_in = (code, i)
            elif 6 <= code <= 11 and last_in is not None:
                a, ia = last_in
                between = [edge_leg.get(x, -1) for x in edges[ia + 1:i]]
                # 정지선 edge에서 바로 나가는 도로로 (교차로 안 통과), 또는
                # 우회전 연결로(12)만 거쳐 나간 경우만 연산교차로 통과로 센다
                passed = ((i == ia + 1 and edges[ia] in stop_edges)
                          or (between and all(c == 12 for c in between)))
                if passed and t0 <= times[ia] < t1:
                    counts[(ORDER[a], ORDER[code - 6])] += 1
                last_in = None
            elif code == -1:
                last_in = None
        el.clear()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=float, default=900.0, help="세지 않는 앞부분 (s)")
    ap.add_argument("--hour", type=float, default=3600.0, help="세는 구간 길이 (s)")
    ap.add_argument("--build", action="store_true", help="도로를 새로 만든 뒤 검사")
    args = ap.parse_args()

    demand = TRAFFIC.get("demand") or {}
    legs = {k: v for k, v in (demand.get("legs") or {}).items() if v}
    if not legs:
        sys.exit("TRAFFIC['demand']['legs']가 비어 있습니다 (실측 수요가 꺼져 있음).")
    cfg = (road_builder.build(ROAD, EGO, TRAFFIC, NET_DIR) if args.build
           else os.path.join(NET_DIR, "highway.sumocfg"))
    env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                         mdp_obs=OBSERVATION, action=ACTION, reward=REWARD,
                         traffic=TRAFFIC, gui=False)
    lay = env._state_layout()
    measured = collections.Counter()
    for (o, d), vph in load_od_demand(demand).items():
        measured[(legs[o], legs[d])] += vph

    t0, t1 = args.warmup, args.warmup + args.hour
    sim = collections.Counter()
    stats = collections.defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        for run in range(args.runs):
            vr = os.path.join(tmp, f"vr{run}.xml")
            st = os.path.join(tmp, f"st{run}.xml")
            subprocess.run(
                [checkBinary("sumo"), "-c", cfg, "--step-length",
                 str(SIMULATION.get("step_length", 0.5)), "--end", str(t1),
                 "--seed", str(1000 + run), "--collision.action", "remove",
                 "--collision.check-junctions", "true", "--no-step-log", "true",
                 "--no-warnings", "true", "--vehroute-output", vr,
                 "--vehroute-output.exit-times", "true",
                 "--vehroute-output.write-unfinished", "true",
                 "--statistic-output", st, "--duration-log.statistics", "true"],
                check=True, capture_output=True)
            sim.update(count_movements(vr, lay["edge_leg"], lay["stop_edges"], t0, t1))
            root = ET.parse(st).getroot()
            veh = root.find("vehicles")
            stats["teleports"].append(int(root.find("teleports").get("total")))
            stats["waiting"].append(int(veh.get("waiting")))
            stats["loaded"].append(int(veh.get("loaded")))
            stats["inserted"].append(int(veh.get("inserted")))
            trip = root.find("vehicleTripStatistics")
            if trip is not None:
                stats["waitingTime"].append(float(trip.get("waitingTime")))
                stats["departDelay"].append(float(trip.get("departDelay")))
            print(f"run {run + 1}/{args.runs} 완료", flush=True)
    hours = args.hour / 3600.0
    sim_vph = {k: v / args.runs / hours for k, v in sim.items()}

    print(f"\n시나리오: {demand.get('scenario')} (이륜차 포함={demand.get('include_motorcycle', False)}, "
          f"버스 포함={demand.get('include_bus', False)}, 배율={demand.get('scale', 1.0)}) | "
          f"{args.runs}회 × {args.hour / 60:.0f}분, 워밍업 {args.warmup / 60:.0f}분 제외\n")
    print("### A. 방향별 교통량 (대/h)\n")
    print("| 방향 | 실측 | 시뮬레이션 | 오차 |")
    print("|---|---|---|---|")
    keys = sorted(set(measured) | set(sim_vph), key=lambda k: (ORDER.index(k[0]), ORDER.index(k[1])))
    for k in keys:
        m, s = measured.get(k, 0.0), sim_vph.get(k, 0.0)
        err = f"{(s - m) / m * 100:+.0f}%" if m else ("–" if not s else "실측 없음")
        print(f"| {k[0]}→{k[1]} | {m:.0f} | {s:.0f} | {err} |")

    print("\n### B. 진입부별 합계 (대/h)\n")
    print("| 진입부 | 실측 | 시뮬레이션 | 오차 |")
    print("|---|---|---|---|")
    for a in ORDER:
        m = sum(v for (x, _), v in measured.items() if x == a)
        s = sum(v for (x, _), v in sim_vph.items() if x == a)
        print(f"| {a} | {m:.0f} | {s:.0f} | {(s - m) / m * 100:+.0f}% |" if m else f"| {a} | 0 | {s:.0f} | – |")
    m, s = sum(measured.values()), sum(sim_vph.values())
    print(f"| 합계 | {m:.0f} | {s:.0f} | {(s - m) / m * 100:+.0f}% |")

    print("\n### C. 막힘 지표 (회당 평균, 전체 도로망)\n")
    mean = lambda k: sum(stats[k]) / len(stats[k]) if stats[k] else float("nan")   # noqa: E731
    print("| 항목 | 값 |")
    print("|---|---|")
    print(f"| teleport | {mean('teleports'):.1f} (회별 {stats['teleports']}) |")
    print(f"| 끝날 때 투입 대기 차량 | {mean('waiting'):.0f}대 (회별 {stats['waiting']}) |")
    print(f"| 투입 / 생성 | {mean('inserted'):.0f} / {mean('loaded'):.0f}대 |")
    print(f"| 차량당 평균 정지 시간 | {mean('waitingTime'):.1f}초 |")
    print(f"| 차량당 평균 투입 지연 | {mean('departDelay'):.1f}초 |")


if __name__ == "__main__":
    main()
