"""모방학습 도구 공통 (expert_collect.py, dagger_collect.py): 저장소 경로·SUMO(libsumo)·7경로 설정·보상."""
import os, sys, json

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))


def setup(repo=None):
    """libsumo가 있으면 traci 대신 씀(같은 프로세스 안 SUMO, 수십 배 빠름). 7경로 설정 적용."""
    try:
        import sumo
        os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
        import libsumo
        sys.modules["traci"] = libsumo
    except ImportError:
        pass
    repo = repo or BASE
    for p in (repo, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(repo)
    import routes7
    routes7.apply()


def reward():
    from train import REWARD
    r = dict(REWARD)
    r.update(json.loads(os.environ.get("REWARD_PATCH", "{}")))
    return r
