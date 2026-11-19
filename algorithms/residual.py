"""잔차 정책 (모방학습 사전학습 + 강화학습 사후학습)

기본 운전자(base) = 모방학습 앙상블 (algorithms/ensemble.py), 학습 중에는 고정.
잔차 신경망(residual) = PPO의 HybridActorCritic. 보상만 보고 PPO로 학습한다.

- 관측: 환경 관측 + 기본 운전자 행동 2개 (가감속, 차로 명령) → obs_dim + 2
- 가감속: clip(기본 가감속 + scale × 잔차 가감속, -1, 1)
- 차로: 잔차 차로 출력이 "유지(0)"면 기본 운전자의 차로 명령을 따르고,
        "왼쪽/오른쪽"이면 이번 스텝 차로 변경을 하지 않는다 (거부권). 잔차가 스스로 차로를 바꾸지는 않는다.
  → 처음(잔차 0, 거부 확률 0에 가까움)에는 기본 운전자와 똑같이 운전하고, 보상으로 고칠 점만 배운다.

파일 형식 (torch.save): {"algorithm": "residual", "obs_dim": 환경 관측 차원, "act_dim": 2, "scale": float,
                         "base": 앙상블 파일 내용(dict), "residual": state_dict}
test.py의 load_agent가 자동으로 알아본다.
"""
import numpy as np
import torch


class ResidualPolicy:
    def __init__(self, base, residual, scale):
        self.base, self.residual, self.scale = base, residual, float(scale)
        self.residual.eval()

    @staticmethod
    def augment(obs, base_action):
        return np.concatenate([np.asarray(obs, np.float32), np.asarray(base_action, np.float32)])

    @staticmethod
    def combine(base_action, res_action, scale):
        accel = float(np.clip(base_action[0] + scale * float(res_action[0]), -1.0, 1.0))
        lane = float(base_action[1]) if float(res_action[1]) == 0.0 else 0.0
        return np.array([accel, lane], dtype=np.float32)

    @staticmethod
    def init_residual(policy, log_std=-1.0, follow_bias=4.0):
        """잔차 0, 거부 확률 ≈ 0에서 시작하도록 출력층을 초기화한다."""
        with torch.no_grad():
            policy.mu_head.weight.zero_(); policy.mu_head.bias.zero_()
            policy.lane_head.weight.zero_()
            policy.lane_head.bias.copy_(torch.tensor([-follow_bias, follow_bias, -follow_bias]))
            policy.log_std.fill_(log_std)

    @staticmethod
    def build(obs_dim, act_dim, checkpoint, hparams):
        from algorithms.ensemble import EnsemblePolicy
        from algorithms.ppo import PPO
        b = checkpoint["base"]
        base = EnsemblePolicy.build(b["obs_dim"], b["act_dim"], b["members"], hparams)
        a = PPO(obs_dim=obs_dim + 2, act_dim=act_dim, **{**hparams, "device": "cpu"})
        a.policy.load_state_dict(checkpoint["residual"])
        return ResidualPolicy(base, a.policy, checkpoint["scale"])

    @staticmethod
    def save(path, obs_dim, act_dim, scale, base_checkpoint, residual_state):
        torch.save({"algorithm": "residual", "obs_dim": int(obs_dim), "act_dim": int(act_dim), "scale": float(scale),
                    "base": base_checkpoint, "residual": residual_state}, path)

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        b = self.base.predict(obs, deterministic=True)
        o = torch.as_tensor(self.augment(obs, b)).reshape(1, -1)
        mu, std, logits, _ = self.residual(o)
        if deterministic:
            r = np.array([float(mu.reshape(-1)[0]), (1.0, 0.0, -1.0)[int(logits.argmax(-1))]], np.float32)
        else:
            r = np.array([float(torch.normal(mu, std).reshape(-1)[0]),
                          (1.0, 0.0, -1.0)[int(torch.distributions.Categorical(logits=logits.reshape(-1)).sample())]], np.float32)
        return self.combine(b, r, self.scale)
