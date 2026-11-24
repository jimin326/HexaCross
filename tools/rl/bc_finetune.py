"""이미 학습한 정책(state_dict)을 모방학습 손실만으로 이어 학습한다 (PPO 없이).
손실은 residual_train.py의 모방학습 손실과 같다: bc_train.accel_loss(다음 스텝 속도) + 차로 교차 엔트로피(변경 가중치 LC_WEIGHT).

쓰임: 환경이 바뀌어 기존 정책의 습관을 먼저 고칠 때 (예: 제한속도 자르기를 없앤 뒤 "제한속도에서 계속 가속" 습관).
usage: python tools/rl/bc_finetune.py INIT.pt OUT.pt DATA1.npz [DATA2.npz ...]
환경 변수: STEPS(4000), LR(1e-4), BATCH(512), LC_WEIGHT(8), SEED(0)"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, BASE); sys.path.insert(0, HERE)
os.chdir(BASE)
import bc_train as bc_mod
from algorithms.ppo import PPO
from train import HPARAMS

init, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
seed = int(os.environ.get("SEED", "0")); torch.manual_seed(seed)
ds = [np.load(f) for f in files]
O = torch.as_tensor(np.concatenate([d["obs"] for d in ds]))
A = torch.as_tensor(np.concatenate([d["accel"] for d in ds]))
L = torch.as_tensor(np.concatenate([d["lane_cls"] for d in ds]))
agent = PPO(obs_dim=O.shape[1], act_dim=2, **{**HPARAMS, "device": "cpu", "seed": seed})
agent.policy.load_state_dict(torch.load(init, map_location="cpu", weights_only=False))
w = float(os.environ.get("LC_WEIGHT", "8")); cw = torch.tensor([w, 1.0, w])
opt = torch.optim.Adam(agent.policy.parameters(), lr=float(os.environ.get("LR", "1e-4")))
bs, n = int(os.environ.get("BATCH", "512")), int(os.environ.get("STEPS", "4000"))
print(f"data {len(O)}  steps {n}", flush=True)
for k in range(n):
    ix = torch.randint(0, len(O), (bs,))
    mu, _, lg, _ = agent.policy(O[ix])
    la = bc_mod.accel_loss(mu.squeeze(-1), A[ix], O[ix]); ll = F.cross_entropy(lg, L[ix], weight=cw)
    opt.zero_grad(); (la + ll).backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5); opt.step()
    if k % 500 == 0 or k == n - 1:
        with torch.no_grad():
            hi = O[:, 0] > 0.995
            m_hi = agent.policy(O[hi][:20000])[0].squeeze(-1).mean().item() if hi.any() else float("nan")
        print(f"step {k}  accel_loss {la.item():.4f}  lane_loss {ll.item():.4f}  제한속도에서 평균 가감속 출력 {m_hi:+.3f}", flush=True)
torch.save(agent.policy.state_dict(), out)
print("saved", out, flush=True)
