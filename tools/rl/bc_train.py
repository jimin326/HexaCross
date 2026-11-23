"""모방학습(BC): 전문가 데이터(expert_collect.py)로 PPO와 같은 정책 신경망(HybridActorCritic)을 학습한다.
가감속 = MSE(tanh μ, 전문가 가감속), 차로 = 교차 엔트로피(왼쪽/유지/오른쪽, 변경 클래스 가중치 LC_WEIGHT).
결과 state_dict는 PPO.load / vtrain INIT_MODEL로 바로 이어 학습할 수 있다.

usage: python tools/rl/bc_train.py OUT.pt DATA1.npz [DATA2.npz ...]
환경 변수: EPOCHS(30), LC_WEIGHT(30), LR(1e-3), BATCH(512), LOG_STD(-1.0), SEED(0)"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)


VMAX, ACC, DEC, DT = 13.89, 3.0, 5.0, 0.5
# 다음 스텝 속도 상한 = 차량 성능 최고속도 (env EGO["max_speed"]). 예전에는 환경이 제한속도(VMAX)에서 속도를 잘라서
# 여기서도 VMAX로 잘랐는데, 그러면 제한속도에서 가속해도 손실이 0이라 "제한속도에서 계속 가속"을 배웠다.
# 환경이 더는 자르지 않으므로(2026-11-23, 조교 답변) 손실도 차량 최고속도로 자른다.
VNEXT_MAX = float(os.environ.get("VNEXT_MAX", "27.78"))


def _vnext(a, v):
    """정규화 가감속 a → 다음 스텝 속도 (환경과 같은 식: a>=0이면 ×최대가속, 아니면 ×최대감속, 0~차량 최고속도로 자름)"""
    return torch.clamp(v + torch.where(a >= 0, a * ACC, a * DEC) * DT, 0.0, VNEXT_MAX)


def accel_loss(pred, lab, obs):
    """LOSS=speed(기본): 다음 스텝 속도(m/s) 차이의 제곱. 전문가가 완전히 멈추면 더 세게 밟아도 손해 없음(0에서 자름).
    앞차가 가깝거나(obs[9] 차간/관측거리) 정지선이 가까울 때(obs[37]) 가중치 SAFE_W배. LOSS=mse면 예전 방식."""
    if os.environ.get("LOSS", "speed") == "mse":
        return F.mse_loss(pred, lab)
    v = obs[:, 0] * VMAX
    target = _vnext(lab, v)
    m = float(os.environ.get("MARGIN", "2.0"))
    if m > 0:
        # 교사 보정: 앞차와 최소 간격보다 MARGIN m 더 떨어져 멈추도록 Krauss 안전 속도로 정답을 깎는다
        vis = float(os.environ.get("OBS_VISIBILITY", "100"))
        has = obs[:, 9] < 0.999
        gap = torch.clamp(obs[:, 9] * vis - m, min=0.0)
        vl = torch.clamp(v + obs[:, 10] * VMAX, min=0.0)
        tb = 1.0 * DEC
        vsafe = -tb + torch.sqrt(tb * tb + vl * vl + 2 * DEC * gap)
        target = torch.where(has, torch.minimum(target, vsafe), target)
    if os.environ.get("RED_CAP", "1") == "1":
        # 교사 보정: 내 신호가 적색이면 정지선 1 m 앞에서 설 수 있는 속도로 정답을 깎는다 (SUMO의 '못 서면 통과' 정답 차단)
        red = (obs[:, 31] < -0.5) & (obs[:, 37] < 0.999)
        dist = torch.clamp(obs[:, 37] ** 2 * 150.0 - 1.0, min=0.0)
        vstop = -DEC * 1.0 + torch.sqrt((DEC * 1.0) ** 2 + 2 * DEC * dist)
        target = torch.where(red, torch.minimum(target, vstop), target)
    err = (_vnext(pred, v) - target) ** 2
    near = (obs[:, 9] < 0.25) | ((obs[:, 37] < 0.4) & (obs[:, 31] < 1.0))
    w = 1.0 + (float(os.environ.get("SAFE_W", "5")) - 1.0) * near.float()
    return (w * err).mean()


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    from algorithms.ppo import PPO
    from train import HPARAMS
    torch.manual_seed(int(os.environ.get("SEED", "0"))); np.random.seed(int(os.environ.get("SEED", "0")))
    ds = [np.load(p) for p in paths]
    O = np.concatenate([d["obs"] for d in ds]); A = np.concatenate([d["accel"] for d in ds])
    L = np.concatenate([d["lane_cls"] for d in ds])
    E = np.concatenate([d["episode"].astype(np.int64) + 10 ** 9 * k for k, d in enumerate(ds)])
    # DAgger 데이터(파일 이름 dg*)의 차로 정답은 SUMO 차로 모델의 '바람'(추월 포함)이라 실제 변경과 성격이 달라서
    # LANE_FROM_DG=0(기본)이면 차로 손실에서 뺀다 (가감속 정답만 사용)
    off = tuple(x for x in os.environ.get("LANE_OFF", "").split(",") if x)   # 이 접두어로 시작하는 파일은 차로 손실에서 뺌
    lane_ok = np.concatenate([np.full(len(d["obs"]), not ((os.path.basename(p).startswith("dg") and os.environ.get("LANE_FROM_DG", "0") == "0")
                                                            or (off and os.path.basename(p).startswith(off))))
                              for p, d in zip(paths, ds)])
    eps = np.unique(E); rng = np.random.default_rng(0); rng.shuffle(eps)
    if len(eps) < 2:
        raise SystemExit("에피소드가 2개 이상 필요합니다 (검증용으로 10%를 떼어 둠)")
    val_eps = set(eps[: max(1, len(eps) // 10)].tolist())
    vm = np.array([e in val_eps for e in E])
    print(f"steps {len(O)} episodes {len(eps)} (검증 {len(val_eps)}) lane {np.bincount(L, minlength=3).tolist()}", flush=True)
    agent = PPO(obs_dim=O.shape[1], act_dim=2, **{**HPARAMS, "device": "cpu", "seed": int(os.environ.get("SEED", "0"))})   # PPO()가 시드를 다시 고정하므로 여기서 넘김
    pol = agent.policy
    w = float(os.environ.get("LC_WEIGHT", "30"))
    cw = torch.tensor([w, 1.0, w])
    opt = torch.optim.Adam([p for n, p in pol.named_parameters() if not n.startswith(("v_head", "log_std"))],
                           lr=float(os.environ.get("LR", "1e-3")))
    LW = np.ones(len(O), np.float32)          # 차로 손실 표본 가중치 (보강 표본용)
    n_aug = int(os.environ.get("AUG_LC", "0"))
    if n_aug > 0:
        # 데이터 보강: 차로를 바꾼 장면마다, 바꾸려는 쪽 차로에 가까운 뒤차(더 빠름) 또는 바로 앞차가 있게 바꾼 복사본 → 정답 "유지".
        # SUMO 전문가도 그 상황에서는 바꾸지 않는다 (막힘). 관측 블록: 왼쪽1 = [5..8], 오른쪽1 = [13..16]
        # (앞차 거리/W, 앞차 상대속도, -뒤차 거리/W, 뒤차 상대속도)
        idx = np.where((L != 1) & lane_ok & ~vm)[0]
        rr = np.random.default_rng(1)
        newO, newL = [], []
        for _ in range(n_aug):
            o = O[idx].copy()
            base = np.where(L[idx] == 0, 5, 13)
            rear = rr.random(len(idx)) < 0.7
            for j, b in enumerate(base):
                if os.environ.get("AUG_MODE") == "wide":
                    # 넓은 보강 (I8): 바꾸려는 쪽에 (가) 가까운 뒤차 0~8 m (같거나 빠름) (나) 가까운 앞차 0~3 m (느리든 빠르든)
                    #   (다) 나란히 겹친 차 (앞차 간격 0 또는 뒤차 간격 0, 속도 상관없이) → 정답 "유지".
                    #   예전 보강은 "느린 앞차"만 있어서, 옆 차로에 나란히 있는 더 빠른 차 쪽으로 바꾸다 부딪혔다 (E→W 월드컵대로 진출부).
                    k_ = rr.integers(4)
                    if k_ == 0:
                        o[j, b + 2] = -rr.uniform(0.0, 0.08); o[j, b + 3] = rr.uniform(0.0, 0.4)
                    elif k_ == 1:
                        o[j, b] = rr.uniform(0.0, 0.03); o[j, b + 1] = rr.uniform(-0.3, 0.4)
                    elif k_ == 2:
                        o[j, b] = 0.0; o[j, b + 1] = rr.uniform(-0.3, 0.4)
                    else:
                        o[j, b + 2] = -0.0; o[j, b + 3] = rr.uniform(-0.3, 0.4)
                elif rear[j]:
                    o[j, b + 2] = -rr.uniform(0.0, float(os.environ.get("AUG_REAR", "0.08")))   # 뒤차 0~8 m
                    o[j, b + 3] = rr.uniform(0.07, 0.35)           # 1~5 m/s 더 빠름
                else:
                    o[j, b] = rr.uniform(0.0, 0.03)                # 앞차 0~3 m
                    o[j, b + 1] = -rr.uniform(0.0, 0.3)            # 더 느림
            newO.append(o); newL.append(np.ones(len(idx), np.int64))
        k0 = len(O)
        O = np.concatenate([O] + newO); L = np.concatenate([L] + newL)
        A = np.concatenate([A] + [A[idx]] * n_aug)
        lane_ok = np.concatenate([lane_ok] + [np.ones(len(idx), bool)] * n_aug)
        vm = np.concatenate([vm] + [np.zeros(len(idx), bool)] * n_aug)
        LW = np.concatenate([LW, np.full(len(O) - k0, float(os.environ.get("AUG_W", os.environ.get("LC_WEIGHT", "8"))), np.float32)])
        acc_ok = np.concatenate([np.ones(k0, bool), np.zeros(len(O) - k0, bool)])
        print(f"차로 보강: 변경 장면 {len(idx)}개 × {n_aug} = {len(O) - k0}개 '유지' 표본", flush=True)
    else:
        acc_ok = np.ones(len(O), bool)
    Ot, At, Lt = torch.as_tensor(O), torch.as_tensor(A), torch.as_tensor(L)
    Mt = torch.as_tensor(lane_ok.astype(np.float32))
    LWt = torch.as_tensor(LW); AKt = torch.as_tensor(acc_ok.astype(np.float32))
    tr_idx, va_idx = np.where(~vm)[0], np.where(vm)[0]
    srng = np.random.default_rng(int(os.environ.get("SEED", "0")))   # 미니배치 순서 (검증 분할은 시드와 무관하게 고정)
    bs = int(os.environ.get("BATCH", "512"))
    for ep in range(int(os.environ.get("EPOCHS", "30"))):
        pol.train(); srng.shuffle(tr_idx); tot = [0.0, 0.0, 0]
        for s in range(0, len(tr_idx), bs):
            b = tr_idx[s:s + bs]
            mu, _, logits, _ = pol(Ot[b])
            la = accel_loss(mu.squeeze(-1)[AKt[b] > 0], At[b][AKt[b] > 0], Ot[b][AKt[b] > 0])
            sw = torch.where(LWt[b] != 1.0, LWt[b], cw[Lt[b]]) * Mt[b]
            ll = (F.cross_entropy(logits, Lt[b], reduction="none") * sw).sum() / sw.sum().clamp(min=1.0)
            loss = la + ll
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            tot[0] += la.item() * len(b); tot[1] += ll.item() * len(b); tot[2] += len(b)
        pol.eval()
        with torch.no_grad():
            mu, _, logits, _ = pol(Ot[va_idx])
            va = accel_loss(mu.squeeze(-1), At[va_idx], Ot[va_idx]).item()
            pred = logits.argmax(-1)
            vl = va_idx[lane_ok[va_idx]] if lane_ok[va_idx].any() else va_idx
            mu, _, logits, _ = pol(Ot[vl])
            pred = logits.argmax(-1)
            ch = Lt[vl] != 1
            rec = ((pred == Lt[vl]) & ch).sum().item() / max(ch.sum().item(), 1)
            fp = ((pred != 1) & ~ch).sum().item() / max((~ch).sum().item(), 1)
        print(f"epoch {ep + 1:2d} train 가감속 {tot[0] / tot[2]:.4f} 차로 {tot[1] / tot[2]:.4f} | 검증 가감속 {va:.4f} "
              f"차로변경 맞춤 {rec:.2f} 불필요한 변경 {fp:.4f}", flush=True)
    with torch.no_grad():
        pol.log_std.fill_(float(os.environ.get("LOG_STD", "-1.0")))
    torch.save(pol.state_dict(), out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
