"""연산교차로 7경로 학습 설정 (train_routes.py / eval_routes.py / record_routes.py가 같이 씀).

경로 7개: 연산교차로 진입로 6개에서 하나씩 + 회전교차로를 한 바퀴 돌아 다시 연산교차로로 들어오는 경로.
에피소드마다 7개 중 하나를 같은 확률로 고른다 (규칙으로 경로를 고르지 않음).

apply()가 하는 일 (모두 환경 설정 — AV를 돕는 규칙은 넣지 않는다):
  - road_config.EGO["routes"] = ROUTES
  - mdp_config.SIMULATION["max_steps"] = 3000 (회전교차로 경로가 3 km라서, 1,500초)
  - AV 투입 실패(도로 입구가 배경 차량으로 막힘)면 reset을 다시 시도
  - LANE_PLAN_RANGE 환경변수: 목표 차로 보상(goal_lane_weight)의 정규화 거리 (보상에만 쓰임)
"""
import os

ROUTES = [
    {"legs": ("S", "W", "W", "E"), "via": ["-794701342#0", "794701342#0"]},   # 연산 → 회전교차로 한 바퀴 → 다시 연산
    ("N", "S"), ("NE", "SW"), ("E", "W"), ("S", "N"), ("SW", "N"), ("W", "E"),
]
NAMES = {"ego_0": "S→W→회전교차로→W→E", "ego_1": "N→S", "ego_2": "NE→SW", "ego_3": "E→W",
         "ego_4": "S→N", "ego_5": "SW→N", "ego_6": "W→E"}


def apply():
    import env.road_config as rc
    import env.mdp_config as mc
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
