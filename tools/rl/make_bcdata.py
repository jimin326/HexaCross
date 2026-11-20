"""PPO 미세조정 때 함께 쓸 모방학습 손실 데이터 만들기 (tools/rl/residual_train.py의 BC_DATA).

DAgger 데이터를 합치고, 차로를 바꾼 표본마다 "바꾸려는 쪽에 가까운 차가 있으면 유지" 복사본을 16개 더한다.
bc_train.py의 AUG_MODE=wide와 같은 규칙이다. bc_train.py는 이 복사본 2개에 차로 변경 가중치(LC_WEIGHT 8)를
그대로 주므로 16개면 같은 비중이 된다.

usage: python tools/rl/make_bcdata.py data/bc_ft.npz data/dgB4.npz data/dgB5.npz data/dgB6.npz data/dgB7.npz
"""
import sys
import numpy as np


def main(out, files, n_copy=16, seed=1):
    ds = [np.load(f) for f in files]
    O = np.concatenate([d["obs"] for d in ds])
    A = np.concatenate([d["accel"] for d in ds])
    L = np.concatenate([d["lane_cls"] for d in ds])
    idx = np.where(L != 1)[0]          # 차로 변경 표본 (0 = 왼쪽, 2 = 오른쪽)
    rr = np.random.default_rng(seed)
    newO = []
    for _ in range(n_copy):
        o = O[idx].copy()
        base = np.where(L[idx] == 0, 5, 13)   # 왼쪽/오른쪽 차로 관측 시작 위치: 앞차 간격, 앞차 상대속도, 뒤차 간격, 뒤차 상대속도
        k = rr.integers(4, size=len(idx))
        for j, b in enumerate(base):
            if k[j] == 0:      # 가까운 뒤차 (0~8 m, 같거나 빠름)
                o[j, b + 2] = -rr.uniform(0.0, 0.08); o[j, b + 3] = rr.uniform(0.0, 0.4)
            elif k[j] == 1:    # 가까운 앞차 (0~3 m)
                o[j, b] = rr.uniform(0.0, 0.03); o[j, b + 1] = rr.uniform(-0.3, 0.4)
            elif k[j] == 2:    # 나란히 겹친 차 (앞 간격 0)
                o[j, b] = 0.0; o[j, b + 1] = rr.uniform(-0.3, 0.4)
            else:              # 나란히 겹친 차 (뒤 간격 0)
                o[j, b + 2] = -0.0; o[j, b + 3] = rr.uniform(-0.3, 0.4)
        newO.append(o)
    O2 = np.concatenate([O] + newO)
    A2 = np.concatenate([A] + [A[idx]] * n_copy)
    L2 = np.concatenate([L] + [np.ones(len(idx), np.int64)] * n_copy)
    np.savez_compressed(out, obs=O2.astype(np.float32), accel=A2.astype(np.float32), lane_cls=L2)
    print(len(O), "→", len(O2), "차로 변경 표본", len(idx), "정답 분포", np.bincount(L2, minlength=3).tolist())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
