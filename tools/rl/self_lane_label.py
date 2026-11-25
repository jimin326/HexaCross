"""DAgger 데이터의 차로 정답을 모델 자신의 선택으로 바꾼다 (가감속 정답은 그대로).
긴 루프 경로에서는 SUMO getBestLanes가 '더 멀리 이어지는 차로'를 가리켜, 지금 차로로도 경로를 갈 수 있는데
'왼쪽으로 바꿔라' 정답이 많이 생긴다 (루프 C dgC3: 15%). 차로 오류가 없는 모델이면 그 선택을 정답으로 쓴다.
usage: python tools/rl/self_lane_label.py MODEL.pt IN.npz OUT.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from algorithms.ppo import PPO
from train import HPARAMS

model, src, dst = sys.argv[1:4]
d = dict(np.load(src))
a = PPO(obs_dim=d["obs"].shape[1], act_dim=2, **{**HPARAMS, "device": "cpu"})
a.policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=False))
O, L = torch.as_tensor(d["obs"], dtype=torch.float32), []
with torch.no_grad():
    for i in range(0, len(O), 8192):
        L.append(a.policy(O[i:i + 8192])[2].argmax(-1).numpy())   # 0 왼쪽 / 1 유지 / 2 오른쪽
L = np.concatenate(L)
print("before", np.bincount(d["lane_cls"], minlength=3).tolist(), "after", np.bincount(L, minlength=3).tolist())
d["lane_cls"] = L.astype(d["lane_cls"].dtype)
np.savez_compressed(dst, **d)
