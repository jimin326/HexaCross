"""연산교차로 7경로 학습 설정 (train_routes.py / eval_routes.py / record_routes.py가 같이 씀).

경로 7개: 연산교차로 진입로 6개에서 하나씩 + 회전교차로를 한 바퀴 돌아 다시 연산교차로로 들어오는 경로.
에피소드마다 7개 중 하나를 같은 확률로 고른다 (규칙으로 경로를 고르지 않음).

apply()가 하는 일 (모두 환경 설정 — AV를 돕는 규칙은 넣지 않는다):
  - road_config.EGO["routes"] = ROUTES
  - mdp_config.SIMULATION["max_steps"] = 3000 (회전교차로 경로가 3 km라서, 1,500초)
  - AV 투입 실패(도로 입구가 배경 차량으로 막힘)면 reset을 다시 시도
  - LANE_PLAN_RANGE 환경변수: 목표 차로 보상(goal_lane_weight)의 정규화 거리 (보상에만 쓰임)
  - ROUTE_SET=loopB 환경변수: 7경로 대신 루프 경로 B 하나만 쓰고, 지도를 넓은 지도(yeonsan_area)로 바꾼다
    (MAX_STEPS=6000 권장, 경로 6.7 km). 경로 이름은 ego_0 하나 → N_ROUTES=1
"""
import os

ROUTES = [
    {"legs": ("S", "W", "W", "E"), "via": ["-794701342#0", "794701342#0"]},   # 연산 → 회전교차로 한 바퀴 → 다시 연산
    ("N", "S"), ("NE", "SW"), ("E", "W"), ("S", "N"), ("SW", "N"), ("W", "E"),
]
NAMES = {"ego_0": "S→W→회전교차로→W→E", "ego_1": "N→S", "ego_2": "NE→SW", "ego_3": "E→W",
         "ego_4": "S→N", "ego_5": "SW→N", "ego_6": "W→E"}


# 루프 경로 B: 연산 북쪽 진입 → 월드컵대로 남쪽 → 연수로 4거리 → 중앙천로 → 신촌로 → 회전교차로(신촌로×거제천로)
# → 거제천로 북쪽 → 세병로/반송로 → 쌍미천로 → 고분로 → 연산 동쪽 진입 → 서쪽 진출 (6,666 m)
LOOP_B = {"legs": ("N", "S", "E", "W"), "via": [
    "-26005519", "165290632", "-1116731562#0", "37402732#1", "773756904#1", "773756904#2", "773756904#3",
    "-165300755", "468972122#1", "468972122#2", "1214651036#2", "794701342", "506009346#1", "517488005", "165814510#0", "165814510#1", "-37400617#4",
    "-37400617#3", "-757668926#3", "-757668926#2", "-757668926#1", "794688376#1"]}
# 루프 경로 C: 연산 북쪽 진입 → 월드컵대로 남쪽 → 연수로 좌회전(동쪽) → 쌍미천로 좌회전(북쪽)
# → 고분로 좌회전(서쪽) → 연산 동쪽 진입 → 서쪽 진출 (회전교차로 없음, 약 3.7 km)
LOOP_C = {"legs": ("N", "S", "E", "W"), "via": [
    "-26005519", "-1307662706#0", "1116731563#1", "773756903", "37401760#0", "37401760#1", "794688376#1"]}
ROUTE_SETS = {"loopB": ([LOOP_B], {"ego_0": "루프B N→S→회전교차로→E→W"}),
              "loopC": ([LOOP_C], {"ego_0": "루프C N→S→연수로→쌍미천로→고분로→E→W"})}


def apply():
    import env.road_config as rc
    import env.mdp_config as mc
    rs = os.environ.get("ROUTE_SET")
    if rs:
        rc.EGO["routes"], names = ROUTE_SETS[rs]
        NAMES.clear(); NAMES.update(names)
        rc.ROAD["osm"]["net_file"] = "env/maps/yeonsan_area.net.xml"
        rc.ROAD["osm"]["extra_signals"] = []
    else:
        rc.EGO["routes"] = ROUTES
    mc.SIMULATION["max_steps"] = int(os.environ.get("MAX_STEPS", "3000"))
    if os.environ.get("LANE_PLAN_RANGE"):
        mc.OBSERVATION["lane_plan_range"] = float(os.environ["LANE_PLAN_RANGE"])
    _patch_reset()


def _patch_reset():
    from env.sumo_env import SumoHighwayEnv
    if getattr(SumoHighwayEnv, "_retry_patched", False):
        return
    orig = SumoHighwayEnv.reset

    def reset(self, *a, **kw):
        for k in range(10):
            try:
                return orig(self, *a, **kw)
            except RuntimeError as e:
                if "삽입 실패" not in str(e) or k == 9:
                    raise
    SumoHighwayEnv.reset = reset
    SumoHighwayEnv._retry_patched = True
