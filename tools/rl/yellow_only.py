"""황색 신호 장면만 뽑아 여러 번 복사한다 (모방학습 손실에서 황색 비중을 키울 때).
관측 [31]=내 신호(0 = 황색), [37]=sqrt(정지선 거리/150) < 0.999 (신호가 범위 안).
usage: python tools/rl/yellow_only.py IN.npz OUT.npz DUP [FIRST_N]
  FIRST_N: 앞의 N개만 본다 (relabel_yellow.py가 뒤에 붙인 황색 복사본을 빼고 원본만 쓸 때)"""
import sys, numpy as np
d = dict(np.load(sys.argv[1])); dup = int(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else len(d["obs"])
O = d["obs"][:n]
y = np.where((np.abs(O[:, 31]) < 0.01) & (O[:, 37] < 0.999))[0]
print("yellow", len(y), "x", dup, "=", len(y) * dup)
np.savez_compressed(sys.argv[2], **{k: np.concatenate([v[y]] * dup) for k, v in d.items()})
