"""7경로 PPO 학습 (train.py를 수정 없이 실행 + 7경로 설정 + 중간 저장/이어하기).

  python tools/rl/train_routes.py RUN_NAME [SEED]
환경변수:
  TIMESTEPS     총 학습 스텝 (기본 train.py 값)
  INIT_MODEL    이 정책 가중치(model.pt)에서 시작 (이어 학습)
  REWARD_PATCH  보상 덮어쓰기 JSON (예: '{"collision_penalty": 50.0}')
  CKPT_EVERY    몇 번 업데이트마다 중간 저장 (기본 5 = 10,240스텝). results/RUN_NAME/ckpt.pt가 있으면 거기서 이어서 학습
  MAX_STEPS, LANE_PLAN_RANGE  routes7.py 참고
libsumo로 빠르게 돌리려면: python -c "import sys, libsumo; sys.modules['traci']=libsumo; ..." 대신
  REPO 루트에서 SUMO_HOME을 맞추고 그대로 실행해도 된다 (traci).
"""
import json, os, re, sys
import torch
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
name = sys.argv[1]; seed = sys.argv[2] if len(sys.argv) > 2 else "0"
import routes7
routes7.apply()
from algorithms.ppo import PPO
ck = os.path.join(BASE, "results", name, "ckpt.pt")
EVERY = int(os.environ.get("CKPT_EVERY", "5"))
_learn, _update = PPO.learn, PPO.update
st = {"n": 0, "logger": None}

def learn(self, env, total_timesteps, logger=None, **kw):
    st["logger"] = logger
    if os.path.exists(ck):
        d = torch.load(ck, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(d["policy"]); self.optimizer.load_state_dict(d["optim"]); self.num_timesteps = d["steps"]
        if logger is not None:
            logger.train_rows, logger.eval_rows = d.get("train_rows", []), d.get("eval_rows", [])
        print(f"[ckpt] {d['steps']} 스텝부터 이어서 학습", flush=True)
    elif os.environ.get("INIT_MODEL"):
        self.load(os.environ["INIT_MODEL"]); print(f"[init] {os.environ['INIT_MODEL']}", flush=True)
    return _learn(self, env, total_timesteps, logger=logger, **kw)

def update(self, *a, **kw):
    out = _update(self, *a, **kw); st["n"] += 1
    if st["n"] % EVERY == 0:
        lg = st["logger"]
        torch.save({"policy": self.policy.state_dict(), "optim": self.optimizer.state_dict(), "steps": self.num_timesteps,
                    "train_rows": getattr(lg, "train_rows", []), "eval_rows": getattr(lg, "eval_rows", [])}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        if lg is not None:
            lg.save_csv()
    return out
PPO.learn, PPO.update = learn, update

src = open(os.path.join(BASE, "train.py"), encoding="utf-8").read()
src = src.replace("    seed=0,", f"    seed={seed},").replace("RUN_NAME = None", f'RUN_NAME = "{name}"')
patch = json.loads(os.environ.get("REWARD_PATCH", "{}"))
if patch:
    old = "REWARD = {**REWARD_DEFAULTS, **REWARD_OVERRIDES}"
    src = src.replace(old, old[:-1] + f", **{patch!r}}}")
if os.environ.get("TIMESTEPS"):
    src = re.sub(r"^TOTAL_TIMESTEPS = .*$", f"TOTAL_TIMESTEPS = {int(os.environ['TIMESTEPS'])}", src, flags=re.M)
os.makedirs(os.path.join(BASE, "results", name), exist_ok=True)
json.dump({"seed": seed, "reward_patch": patch, "timesteps": os.environ.get("TIMESTEPS"), "init": os.environ.get("INIT_MODEL")},
          open(os.path.join(BASE, "results", name, "run_patch.json"), "w"), indent=1)
exec(compile(src, os.path.join(BASE, "train.py"), "exec"), {"__name__": "__main__", "__file__": os.path.join(BASE, "train.py")})
