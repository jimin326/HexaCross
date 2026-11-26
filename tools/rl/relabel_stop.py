"""정지 접근 정답 보정 (모방학습 데이터 전용, 주행 중에는 쓰지 않는다).

문제: DAgger 정답 가감속은 env.step 뒤 getSpeedWithoutTraCI로 얻는다. 그런데 AV를 setSpeed로 몰고 있으면
이 값은 "SUMO 운전 모델이 골랐을 속도"가 아니라 거의 학생이 명령한 속도(약간 낮게)이고, SUMO 안전 속도보다
빠를 때만 그 안전 속도로 줄어든다 (2026-11-26 시험: 서 있는 AV에 명령 0 → 정답 0, 그냥 두면 SUMO는 1.45 m/s로 출발).
그래서 서 있거나 천천히 다가갈 때 정답이 "그대로"여서, 앞차·정지선 7~8 m 앞에 서는 버릇을 그대로 따라 했다.

규칙 (관측만 씀): 서 있는(또는 거의 선) 앞차나 적색 정지선 앞에서 **기어가는 속도(V_RULE 미만)일 때**,
    v_des = clip(K · 남은 거리, 0, V_CREEP)   남은 거리 = 앞차 간격 − GAP_VEH  또는  정지선 거리 − GAP_LINE
로 다가가게 정답을 바꾼다. 정답 가감속 = clip((v_des − v) / 0.5 s, −DEC_MAX, ACC_MAX), 정규화는 가속 3, 감속 5.
  - 남은 거리에 비례해 속도를 줄이므로 목표에 가까울수록 정답이 0에 가깝다. 목표를 넘으면(남은 거리 ≤ 0) 멈춤.
    (계단처럼 "기어가기 → 멈춤"으로 바뀌는 첫 규칙은 관측 해상도(앞차 간격 / 100 m)로 경계를 구분하지 못해
     목표를 지나서도 다시 기어가다 앞차에 닿았다: bcstop 40회 중 충돌 11, 8건이 쌍미천로 신호 앞 대기 줄)
  - V_RULE 이상에서는 원래 정답(SUMO 안전 속도로 줄인 감속)을 그대로 둔다. 다가가는 속도를 V_CREEP(2 m/s)로 묶어서
    규칙이 낸 속도가 다시 규칙 밖으로 나가지 않게 한다 (첫 판은 6 m/s까지 가속시켰다가 6 m/s 위에서 감속 정답이 없어
    신호위반·충돌이 쏟아졌다: r5 102k 120회 중 충돌 100).
관측: [0]=속도/13.89, [9]=현재 차로 앞차 간격/100 (getLeader 거리 = 범퍼 간격 − minGap 2.5 m),
      [10]=(앞차 속도 − 내 속도)/13.89, [31]=내 신호 (녹 +1, 황 0, 적 −1, 150 m 안 신호 없으면 1),
      [37]=sqrt(정지선 거리/150).

usage: python tools/rl/relabel_stop.py IN.npz OUT.npz [DUP=보정 표본 추가 복사 수]
"""
import sys
import numpy as np

VMAX, AMAX, DMAX, DT = 13.89, 3.0, 5.0, 0.5
GAP_VEH, GAP_LINE = 1.5, 1.5         # 목표: 앞차 범퍼 4.0 m (getLeader 1.5 m, 충돌 판정은 0), 정지선 1.5 m. 3.25·3.5 m는 대기 줄에서 다시 기어가다 충돌 (bcstop 40회 중 11·7, 뒤차 무작위 복사를 더한 뒤에도 3.5 m는 7, 4.0 m는 1)
ACC_MAX, DEC_MAX = 1.0, 4.5
LEAD_SLOW, V_RULE, V_CREEP, RANGE, K, AUG = 0.5, 2.5, 2.0, 25.0, 0.5, 2


def relabel(d):
    O, A0 = d["obs"], d["accel"]
    A = A0.copy()
    v = O[:, 0] * VMAX
    g = O[:, 9] * 100.0
    has_lead = O[:, 9] < 0.999
    lv = v + O[:, 10] * VMAX
    dl = O[:, 37] ** 2 * 150.0
    red = (O[:, 31] < -0.5) & (O[:, 37] < 0.999)
    # 기준 거리: 서 있는 앞차가 정지선보다 가까우면 앞차, 아니면 적색 정지선
    veh = has_lead & (lv < LEAD_SLOW) & (g < RANGE) & ~(red & (dl < g + 2.5))
    line = red & (dl < RANGE) & ~(has_lead & (g + 2.5 < dl))
    rem = np.where(veh, g - GAP_VEH, np.where(line, dl - GAP_LINE, 0.0))
    m = (veh | line) & (v < V_RULE)
    v_des = np.clip(K * rem, 0.0, V_CREEP)
    acc = np.clip((v_des - v) / DT, -DEC_MAX, ACC_MAX)
    lab = np.where(acc >= 0, acc / AMAX, acc / DMAX)
    A[m] = lab[m]
    return A, veh, line, m, rem


def main():
    d = dict(np.load(sys.argv[1]))
    dup = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    A, veh, line, m, rem = relabel(d)
    ch = m & (np.abs(A - d["accel"]) > 0.02)
    print(f"{sys.argv[1]} 앞차 정지 상황 {veh.sum()} · 적색 정지선 {line.sum()} · 규칙 적용 {m.sum()} · 정답 바뀜 {ch.sum()}")
    d["accel"] = A
    if dup > 0 and ch.any():
        idx = np.where(ch)[0]
        for k in ("obs", "accel", "lane_cls"):
            d[k] = np.concatenate([d[k]] + [d[k][idx]] * dup)
    # 목표 근처(남은 거리 0.5 m 이하)에 선 표본은 뒤차 관측(11 간격, 12 속도차)을 무작위로 바꾼 복사본을 AUG개 더한다.
    # 정답(멈춤)은 그대로. "뒤에서 빠른 차가 오면 앞으로 기어간다"는 엉뚱한 연관을 끊는다
    # (bcstop: 쌍미천로 신호 대기 줄에서 뒤차가 13 m/s로 다가오자 가속 +0.2 → 앞차 충돌).
    near = np.where(m & (rem <= 0.5) & (d["obs"][:len(m), 0] * VMAX < 0.5))[0]
    if AUG > 0 and len(near):
        rr = np.random.default_rng(3)
        Os = []
        for _ in range(AUG):
            o = d["obs"][near].copy()
            o[:, 11] = -rr.uniform(0.02, 1.0, len(near))
            o[:, 12] = rr.uniform(-0.3, 1.0, len(near))
            Os.append(o)
        d["obs"] = np.concatenate([d["obs"]] + Os)
        d["accel"] = np.concatenate([d["accel"]] + [d["accel"][near]] * AUG)
        d["lane_cls"] = np.concatenate([d["lane_cls"]] + [d["lane_cls"][near]] * AUG)
        print(f"  목표 근처 정지 표본 {len(near)} × 뒤차 무작위 {AUG}")
    np.savez_compressed(sys.argv[2], **d)


if __name__ == "__main__":
    main()
