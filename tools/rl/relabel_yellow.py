"""황색 신호 정답 보정: SUMO 정답은 황색에서 30~70 m 앞이어도 '계속 진행/가속'이라 적색 진입으로 이어진다.
물리 규칙으로 바꾼다 — 정지선까지 편안한 감속(≤4 m/s²)으로 설 수 있으면 멈춘다(필요 감속 ×1.15),
못 서면 그대로 통과(가감속 ≥0). 관측: [0]=속도/13.89, [31]=내 신호(+1 녹/0 황/-1 적), [37]=sqrt(정지선 거리/150).
usage: python tools/rl/relabel_yellow.py IN.npz OUT.npz [DUP=황색 표본 추가 복사 수]"""
import sys, numpy as np
VMAX, RANGE, DMAX, BCOMF, MARGIN = 13.89, 150.0, 5.0, 4.0, 2.0
d = dict(np.load(sys.argv[1])); dup = int(sys.argv[3]) if len(sys.argv) > 3 else 0
O, A = d["obs"], d["accel"].copy()
y = (np.abs(O[:, 31]) < 0.01) & (O[:, 37] < 0.999)
v = O[:, 0] * VMAX; dist = O[:, 37] ** 2 * RANGE
need = v ** 2 / (2 * np.maximum(dist - MARGIN, 0.1))
stop = y & (need <= BCOMF) & (v > 0.5)
go = y & ~stop & (v > 0.5)
A[stop] = np.clip(-(need[stop] * 1.15) / DMAX, -1.0, 0.0)
A[go] = np.maximum(A[go], 0.0)
d["accel"] = A
print(sys.argv[1], "yellow", int(y.sum()), "stop", int(stop.sum()), "go", int(go.sum()))
if dup:
    idx = np.where(y)[0]
    for k in d: d[k] = np.concatenate([d[k]] + [d[k][idx]] * dup)
np.savez_compressed(sys.argv[2], **d)
