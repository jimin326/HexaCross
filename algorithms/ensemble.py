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


class EnsembleActorCritic(torch.nn.Module):
    """앙상블 전체를 PPO로 미세조정하기 위한 학습용 정책 (HybridActorCritic과 같은 인터페이스).

    - 행동 분포: 가감속 = 구성원 μ 평균 + 공통 표준편차 exp(log_std), 차로 = 구성원 log_softmax 평균
      (EnsemblePolicy의 결정적 행동과 같은 규칙이라, 학습한 것이 그대로 평가에 쓰인다)
    - 가치 V(s): 새 신경망 (구성원의 v_head는 모방학습으로 학습되지 않아 쓰지 않음)
    저장은 members()의 state_dict 목록을 EnsemblePolicy.save로 → test.py가 그대로 읽는다.
    """
    LANE_CMDS = (1.0, 0.0, -1.0)

    def __init__(self, obs_dim, state_dicts, hparams, init_log_std=-1.0, value_hidden=(256, 256)):
        super().__init__()
        from algorithms.ppo import PPO
        ms = []
        for sd in state_dicts:
            a = PPO(obs_dim=obs_dim, act_dim=2, **{**hparams, "device": "cpu"})
            a.policy.load_state_dict(sd)
            ms.append(a.policy)
        self.members_ = torch.nn.ModuleList(ms)
        layers, d = [], obs_dim
        for h in value_hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        self.v_net = torch.nn.Sequential(*layers)
        self.log_std = torch.nn.Parameter(torch.ones(1) * float(init_log_std))

    def members(self):
        return [m.state_dict() for m in self.members_]

    def forward(self, obs):
        outs = [m(obs) for m in self.members_]
        mu = torch.stack([o[0] for o in outs]).mean(0)
        logits = torch.stack([torch.log_softmax(o[2], -1) for o in outs]).mean(0)
        return mu, torch.exp(self.log_std), logits, self.v_net(obs).squeeze(-1)

    def sample(self, obs):
        mu, std, logits, value = self.forward(obs)
        nd = torch.distributions.Normal(mu, std)
        cd = torch.distributions.Categorical(logits=logits)
        accel, lane_idx = nd.sample(), cd.sample()
        log_prob = nd.log_prob(accel).sum(-1) + cd.log_prob(lane_idx)
        lane_cmd = torch.as_tensor([self.LANE_CMDS[int(i)] for i in lane_idx], device=accel.device)
        return torch.cat([accel, lane_cmd.unsqueeze(-1)], dim=-1), log_prob, value

    def det_action(self, obs):
        mu, _, logits, _ = self.forward(obs)
        lane_cmd = torch.as_tensor([self.LANE_CMDS[int(i)] for i in torch.argmax(logits, -1)], device=mu.device)
        return torch.cat([mu, lane_cmd.unsqueeze(-1)], dim=-1)

    def evaluate_actions(self, obs, actions):
        mu, std, logits, value = self.forward(obs)
        nd = torch.distributions.Normal(mu, std)
        cd = torch.distributions.Categorical(logits=logits)
        lane_idx = (1.0 - torch.round(actions[:, 1])).long().clamp(0, 2)
        log_prob = nd.log_prob(actions[:, :1]).sum(-1) + cd.log_prob(lane_idx)
        return log_prob, nd.entropy().sum(-1) + cd.entropy(), value

    def lane_entropy(self, obs):
        return torch.distributions.Categorical(logits=self.forward(obs)[2]).entropy()

    def lane_change_prob(self, obs):
        return 1.0 - torch.softmax(self.forward(obs)[2], -1)[:, 1]

    @property
    def current_std(self):
        return float(torch.exp(self.log_std).mean().item())
