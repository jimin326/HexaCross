"""앙상블 증류: 여러 BC 모델의 평균 출력(가감속 μ 평균, 차로 확률 평균)을 정답으로 단일 PPO 정책 신경망을 학습한다.
결과는 PPO.load / test.py로 바로 읽히는 state_dict 한 개.

usage: python tools/rl/distill.py ENS_FILE OUT.pt DATA1.npz [DATA2.npz ...]
환경 변수: EPOCHS(20), LR(1e-3), BATCH(1024), LOG_STD(-1.0), SEED(0)"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)


def main():
    ens, out, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    from algorithms.ppo import PPO
    from train import HPARAMS
    torch.manual_seed(int(os.environ.get("SEED", "0")))
    O = np.concatenate([np.load(p)["obs"] for p in paths])
    Ot = torch.as_tensor(O)
    members = []
    for p in open(ens).read().split():
        a = PPO(obs_dim=O.shape[1], act_dim=2, **{**HPARAMS, "device": "cpu"})
        a.policy.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)); a.policy.eval()
        members.append(a.policy)
    # 앙상블 정답 (평가 때와 같은 방식: μ 평균, log_softmax 평균 → softmax)
    mus, lps = [], []
    with torch.no_grad():
        for s in range(0, len(Ot), 8192):
            outs = [m(Ot[s:s + 8192]) for m in members]
            mus.append(torch.stack([o[0].reshape(-1) for o in outs]).mean(0))
            lps.append(torch.stack([torch.log_softmax(o[2], -1) for o in outs]).mean(0))
    T_mu = torch.cat(mus); T_p = torch.softmax(torch.cat(lps), -1)
    print(f"표본 {len(O)}, 앙상블 {len(members)}개", flush=True)
    st = PPO(obs_dim=O.shape[1], act_dim=2, **{**HPARAMS, "device": "cpu"})
    pol = st.policy
    opt = torch.optim.Adam([p for n, p in pol.named_parameters() if not n.startswith(("v_head", "log_std"))],
                           lr=float(os.environ.get("LR", "1e-3")))
    idx = np.arange(len(O)); rng = np.random.default_rng(0)
    bs = int(os.environ.get("BATCH", "1024"))
    for ep in range(int(os.environ.get("EPOCHS", "20"))):
        rng.shuffle(idx); tot = [0.0, 0.0, 0]
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            mu, _, lg, _ = pol(Ot[b])
            la = F.mse_loss(mu.reshape(-1), T_mu[b]) * 100.0
            ll = F.kl_div(torch.log_softmax(lg, -1), T_p[b], reduction="batchmean") * 10.0
            opt.zero_grad(); (la + ll).backward(); opt.step()
            tot[0] += la.item() * len(b); tot[1] += ll.item() * len(b); tot[2] += len(b)
        with torch.no_grad():
            mu, _, lg, _ = pol(Ot[:200000])
            agree = (lg.argmax(-1) == T_p[:200000].argmax(-1)).float().mean().item()
            ch = T_p[:200000].argmax(-1) != 1
            agree_ch = (lg.argmax(-1)[ch] == T_p[:200000].argmax(-1)[ch]).float().mean().item() if ch.any() else 1.0
            mae = (mu.reshape(-1) - T_mu[:200000]).abs().mean().item()
        print(f"epoch {ep + 1} 가감속 {tot[0] / tot[2]:.4f} 차로 {tot[1] / tot[2]:.4f} | 가감속 평균오차 {mae:.4f} 차로 일치 {agree:.4f} (변경 {agree_ch:.3f})", flush=True)
    with torch.no_grad():
        pol.log_std.fill_(float(os.environ.get("LOG_STD", "-1.0")))
    torch.save(pol.state_dict(), out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
