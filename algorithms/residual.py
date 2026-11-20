"""잔차 정책 (모방학습 사전학습 + 강화학습 사후학습)

기본 운전자(base) = 모방학습 앙상블 (algorithms/ensemble.py), 학습 중에는 고정.
잔차 신경망(residual) = PPO의 HybridActorCritic. 보상만 보고 PPO로 학습한다.

- 관측: 환경 관측 + 기본 운전자 행동 2개 (가감속, 차로 명령) → obs_dim + 2
- 가감속: clip(기본 가감속 + scale × 잔차 가감속, -1, 1)
- 차로: 기본은 기본 운전자의 차로 명령 그대로 (lane_veto=False, 잔차의 차로 출력은 쓰지 않음).
        lane_veto=True면 잔차 차로 출력이 "왼쪽/오른쪽"일 때 이번 스텝 차로 변경을 하지 않는다 (거부권).
        ※ 거부권을 켜고 학습하면(res1) 충돌 감점(10)이 잘못된 차로 감점(5)보다 커서 회전교차로 경로에
          꼭 필요한 차로 변경까지 막도록 배웠다 (결정적 평가 회전교차로 완주 1/30). 그래서 기본은 끔.
  → 처음(잔차 0, 거부 확률 0에 가까움)에는 기본 운전자와 똑같이 운전하고, 보상으로 고칠 점만 배운다.

파일 형식 (torch.save): {"algorithm": "residual", "obs_dim": 환경 관측 차원, "act_dim": 2, "scale": float, "lane_veto": bool,
                         "base": 앙상블 파일 내용(dict), "residual": state_dict}
test.py의 load_agent가 자동으로 알아본다.
"""
import numpy as np
import torch


class ResidualPolicy:
    def __init__(self, base, residual, scale, lane_veto=False, max_veto=0):
        self.base, self.residual, self.scale, self.lane_veto = base, residual, float(scale), bool(lane_veto)
        self.max_veto = int(max_veto)      # >0이면 거부권으로 미룰 수 있는 최대 연속 스텝 (그 뒤엔 기본 운전자 명령대로)
        self._veto_n = 0
        self.residual.eval()

    @staticmethod
    def augment(obs, base_action):
        return np.concatenate([np.asarray(obs, np.float32), np.asarray(base_action, np.float32)])

    @staticmethod
    def combine(base_action, res_action, scale, lane_veto=False):
        accel = float(np.clip(base_action[0] + scale * float(res_action[0]), -1.0, 1.0))
        lane = float(base_action[1]) if (not lane_veto or float(res_action[1]) == 0.0) else 0.0
        return np.array([accel, lane], dtype=np.float32)

    @staticmethod
    def veto_step(base_action, res_action, n, max_veto):
        """연속 거부 횟수 관리: 기본 운전자가 차로를 안 바꾸려 하면 0으로, 거부 중이면 +1.
        max_veto(>0)번 연속 거부했으면 이번에는 거부권을 쓸 수 없다 (미루기만 가능, 막지는 못함).
        반환: (이번 스텝 거부권 사용 가능 여부, 새 연속 횟수)"""
        if float(base_action[1]) == 0.0:
            return False, 0
        allowed = max_veto <= 0 or n < max_veto
        vetoing = allowed and float(res_action[1]) != 0.0
        return allowed, (n + 1 if vetoing else 0)

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
        return ResidualPolicy(base, a.policy, checkpoint["scale"], checkpoint.get("lane_veto", False), checkpoint.get("max_veto", 0))

    @staticmethod
    def save(path, obs_dim, act_dim, scale, base_checkpoint, residual_state, lane_veto=False, max_veto=0):
        torch.save({"algorithm": "residual", "obs_dim": int(obs_dim), "act_dim": int(act_dim), "scale": float(scale),
                    "lane_veto": bool(lane_veto), "max_veto": int(max_veto),
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
        if self.lane_veto and self.max_veto > 0:
            allowed, self._veto_n = self.veto_step(b, r, self._veto_n, self.max_veto)
            return self.combine(b, r, self.scale, allowed)
        return self.combine(b, r, self.scale, self.lane_veto)
