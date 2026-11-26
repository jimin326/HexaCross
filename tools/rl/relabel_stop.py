"""정지 접근 정답 보정 (모방학습 데이터 전용, 주행 중에는 쓰지 않는다).

문제: DAgger 정답 가감속은 env.step 뒤 getSpeedWithoutTraCI로 얻는다. 그런데 AV를 setSpeed로 몰고 있으면
이 값은 "SUMO 운전 모델이 골랐을 속도"가 아니라 거의 학생이 명령한 속도(약간 낮게)이고, SUMO 안전 속도보다
빠를 때만 그 안전 속도로 줄어든다 (2026-11-26 시험: 서 있는 AV에 명령 0 → 정답 0, 그냥 두면 SUMO는 1.45 m/s로 출발).
그래서 서 있거나 천천히 다가갈 때 정답이 "그대로"여서, 앞차·정지선 7~8 m 앞에 서는 버릇을 그대로 따라 했다.

규칙 (관측만 씀): 서 있는(또는 거의 선) 앞차나 적색 정지선 앞에서, 목표 거리까지 감속 B로 설 수 있는 속도
    v_des = sqrt(2·B·max(남은 거리, 0))   남은 거리 = 앞차 간격 − GAP_VEH  또는  정지선 거리 − GAP_LINE
로 다가가게 정답을 바꾼다. 정답 가감속 = clip((v_des − v) / 0.5 s, −DEC_MAX, ACC_MAX), 정규화는 가속 3, 감속 5.
  - 남은 거리가 MARGIN 이하면 원래 정답(멈춤·감속)을 그대로 둔다.
  - 원래 정답이 더 세게 감속하는데 남은 거리가 짧을 때(안전 속도 제약)는 원래 정답을 둔다: 정답 = max(규칙, 원래)는
    남은 거리가 3 m 넘을 때만.
관측: [0]=속도/13.89, [9]=현재 차로 앞차 간격/100 (getLeader 거리 = 범퍼 간격 − minGap 2.5 m),
      [10]=(앞차 속도 − 내 속도)/13.89, [31]=내 신호 (녹 +1, 황 0, 적 −1, 150 m 안 신호 없으면 1),
      [37]=sqrt(정지선 거리/150).

usage: python tools/rl/relabel_stop.py IN.npz OUT.npz [DUP=보정 표본 추가 복사 수]
"""
import sys
import numpy as np

VMAX, AMAX, DMAX, DT = 13.89, 3.0, 5.0, 0.5
GAP_VEH, GAP_LINE = 0.75, 1.5        # 목표: 앞차 범퍼 3.25 m (getLeader 0.75 m), 정지선 1.5 m
B, ACC_MAX, DEC_MAX, MARGIN = 1.5, 1.5, 4.5, 0.3
LEAD_SLOW, V_MAX_RULE, RANGE = 0.5, 6.0, 25.0


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
    m = (veh | line) & (v < V_MAX_RULE) & (rem > MARGIN)
    v_des = np.sqrt(2 * B * np.maximum(rem, 0.0))
    acc = np.clip((v_des - v) / DT, -DEC_MAX, ACC_MAX)
    lab = np.where(acc >= 0, acc / AMAX, acc / DMAX)
    new = np.where(rem > 3.0, np.maximum(lab, A0), lab)   # 멀리 있으면 원래 정답보다 덜 서게, 가까우면 규칙대로
    A[m] = new[m]
    return A, veh, line, m


def main():
    d = dict(np.load(sys.argv[1]))
    dup = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    A, veh, line, m = relabel(d)
    ch = m & (np.abs(A - d["accel"]) > 0.02)
    print(f"{sys.argv[1]} 앞차 정지 상황 {veh.sum()} · 적색 정지선 {line.sum()} · 규칙 적용 {m.sum()} · 정답 바뀜 {ch.sum()}")
    d["accel"] = A
    if dup > 0 and ch.any():
        idx = np.where(ch)[0]
        for k in ("obs", "accel", "lane_cls"):
            d[k] = np.concatenate([d[k]] + [d[k][idx]] * dup)
    np.savez_compressed(sys.argv[2], **d)


if __name__ == "__main__":
    main()
