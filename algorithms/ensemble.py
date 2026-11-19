"""앙상블 정책: 같은 구조(HybridActorCritic)의 신경망 여러 개를 평균해서 운전한다.

- 가감속: 구성원 μ의 평균
- 차로: 구성원 차로 확률의 로그 평균(log_softmax 평균)이 가장 큰 쪽 (결정적) 또는 그 분포에서 샘플
모방학습 모델은 같은 데이터로 학습해도 모델마다 실제 운전 결과가 크게 흔들려서, 여러 개의 평균으로 안정시킨다.

파일 형식 (torch.save): {"algorithm": "ensemble", "obs_dim": int, "act_dim": 2, "members": [state_dict, ...]}
test.py의 load_agent가 자동으로 알아본다.
"""
import numpy as np
import torch


class EnsemblePolicy:
    LANE_CMDS = (1.0, 0.0, -1.0)

    def __init__(self, members):
        self.members = members
        for m in self.members:
            m.eval()

    @staticmethod
    def build(obs_dim, act_dim, state_dicts, hparams):
        from algorithms.ppo import PPO
        ms = []
        for sd in state_dicts:
            a = PPO(obs_dim=obs_dim, act_dim=act_dim, **{**hparams, "device": "cpu"})
            a.policy.load_state_dict(sd)
            ms.append(a.policy)
        return EnsemblePolicy(ms)

    @staticmethod
    def save(path, obs_dim, act_dim, state_dicts):
        torch.save({"algorithm": "ensemble", "obs_dim": int(obs_dim), "act_dim": int(act_dim),
                    "members": list(state_dicts)}, path)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        o = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        outs = [m(o) for m in self.members]                      # (mu, std, lane_logits, value)
        mu = torch.stack([x[0].reshape(-1) for x in outs]).mean(0)
        lp = torch.stack([torch.log_softmax(x[2], -1).reshape(-1) for x in outs]).mean(0)
        if deterministic:
            accel = float(mu[0])
            lane = int(torch.argmax(lp))
        else:
            std = torch.stack([x[1].reshape(-1) for x in outs]).mean(0)
            accel = float(torch.normal(mu, std)[0])
            lane = int(torch.distributions.Categorical(logits=lp).sample())
        return np.array([accel, self.LANE_CMDS[lane]], dtype=np.float32)
