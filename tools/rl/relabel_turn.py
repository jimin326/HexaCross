"""신호 없는 회전·합류 접근 정답 보정 (모방학습 데이터 전용, 주행 중에는 쓰지 않는다).

문제: SUMO 전문가는 신호 없는 합류부·회전교차로 입구에서 다른 도로의 차(상충 차량)를 보고 양보한다.
AV 관측에는 그 차가 없어서(공지로 끔), 전문가를 따라 하면 40~50 km/h로 들어가다 교차로 안에서 부딪힌다
(2026-11-23 평가: 충돌의 85%가 월드컵대로 → 연수로 합류부 `1768493023`).
관측만으로 안전한 행동은 "보이지 않는 차가 나타나도 설 수 있게 미리 천천히 접근"이다.

규칙 (관측만 씀):
  - 대상: 150 m 안에 신호 없음(관측 37 = 1) + 다음 교차로가 꺾임(|관측 39| ≥ 0.15, 약 27° 이상)
          + 그 교차로까지 d < 60 m (관측 40 × 300)
  - 허용 속도 v_cap(d) = sqrt(V_TURN² + 2·B·max(d − 2, 0))  (교차로 입구 V_TURN, 감속 B로 줄여 감)
  - 지금 속도가 v_cap보다 빠르면 정답 가감속 = min(정답, (v_cap − v) / 1초 를 감속 한도로 나눈 값)
관측: [0]=속도/13.89, [37]=sqrt(정지선 거리/150) (신호 없으면 1), [39]=다음 꺾는 교차로 각도/180, [40]=그 거리/300.

usage: python tools/rl/relabel_turn.py IN.npz OUT.npz [DUP=보정 표본 추가 복사 수]
"""
import sys
import numpy as np

VMAX, DMAX = 13.89, 5.0          # 관측 속도 정규화, 감속 정답 정규화 (relabel_yellow.py와 같음)
V_TURN, B, D_MAX, ANG = 4.0, 2.0, 60.0, 0.15


def relabel(d):
    O, A = d["obs"], d["accel"].copy()
    v = O[:, 0] * VMAX
    dist = O[:, 40] * 300.0
    m = (O[:, 37] > 0.999) & (np.abs(O[:, 39]) >= ANG) & (O[:, 40] < 0.999) & (dist < D_MAX)
    cap = np.sqrt(V_TURN ** 2 + 2 * B * np.maximum(dist - 2.0, 0.0))
    over = m & (v > cap + 0.5)
    want = np.clip((cap - v) / 1.0, -4.0, 0.0) / DMAX
    A[over] = np.minimum(A[over], want[over])
    return A, m, over


def main():
    d = dict(np.load(sys.argv[1]))
    dup = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    A, m, over = relabel(d)
    d["accel"] = A
    print(sys.argv[1], "신호 없는 회전 접근", int(m.sum()), "감속으로 보정", int(over.sum()))
    if dup:
        idx = np.where(over)[0]
        for k in d:
            d[k] = np.concatenate([d[k]] + [d[k][idx]] * dup)
    np.savez_compressed(sys.argv[2], **d)


if __name__ == "__main__":
    main()
