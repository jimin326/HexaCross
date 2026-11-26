"""정지 접근 정답 보정 (모방학습 데이터 전용, 주행 중에는 쓰지 않는다).

문제: DAgger 정답 가감속은 env.step 뒤 getSpeedWithoutTraCI로 얻는다. 그런데 AV를 setSpeed로 몰고 있으면
이 값은 "SUMO 운전 모델이 골랐을 속도"가 아니라 거의 학생이 명령한 속도(약간 낮게)이고, SUMO 안전 속도보다
빠를 때만 그 안전 속도로 줄어든다 (2026-11-26 시험: 서 있는 AV에 명령 0 → 정답 0, 그냥 두면 SUMO는 1.45 m/s로 출발).
그래서 서 있거나 천천히 다가갈 때 정답이 "그대로"여서, 앞차·정지선 7~8 m 앞에 서는 버릇을 그대로 따라 했다.

규칙 (관측만 씀): 서 있는(또는 거의 선) 앞차나 적색 정지선 앞에서 **기어가는 속도(V_RULE 미만)일 때**,
    v_des = clip(K · 남은 거리, 0, V_CREEP)   남은 거리 = 앞차 간격 − GAP_VEH  또는  정지선 거리 − GAP_LINE
로 다가가게 정답을 바꾼다. 정답 가감속 = clip((v_des − v) / 0.5 s, −DEC_MAX, ACC_MAX), 정규화는 가속 3, 감속 5.
  - 남은 거리에 비례해 속도를 줄이고, 목표 STOP_ZONE(0.6 m) 안에서는 확실히 멈춤(−3 m/s², 서 있으면 계속 제동).
    (계단처럼 "기어가기 → 멈춤"으로 바뀌는 첫 규칙은 관측 해상도(앞차 간격 / 100 m)로 경계를 구분하지 못해
     목표를 지나서도 다시 기어가다 앞차에 닿았다: bcstop 40회 중 충돌 11, 8건이 쌍미천로 신호 앞 대기 줄)
  - V_RULE 이상에서는 원래 정답(SUMO 안전 속도로 줄인 감속)을 그대로 둔다. 다가가는 속도를 V_CREEP(2 m/s)로 묶어서
    규칙이 낸 속도가 다시 규칙 밖으로 나가지 않게 한다 (첫 판은 6 m/s까지 가속시켰다가 6 m/s 위에서 감속 정답이 없어
    신호위반·충돌이 쏟아졌다: r5 102k 120회 중 충돌 100).
빠른 접근 규칙: V_RULE 이상에서 적색 정지선·더 느린 앞차까지 서는 데 필요한 감속 need = (v − 목표 속도)²/(2·남은 거리)가
  BRAKE_TRIG(1 m/s²)를 넘는데 정답이 그보다 약하면 need × 1.15로 바꾼다 (정답 버그로 이 구간 정답도 학생 명령 그대로였다).
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
LEAD_SLOW, V_RULE, V_CREEP, RANGE, K, AUG = 0.5, 2.5, 2.0, 25.0, 0.5, 4
SAT_LAB = -0.05   # 관측 속도가 13.89 m/s(제한속도 50 km/h)에서 잘린 표본은 이 값 이하 (넘어서면 속도를 알 수 없다)
STOP_ZONE, STOP_ZONE_VEH, HOLD = 0.6, 1.0, 0.6
YEL_DEC, NEAR_DUP = 4.5, 4   # 황색: 1.15배 여유를 둔 감속이 4.5 m/s² 안이면 서고(설 수 있으면 선다) 아니면 통과 / 앞차 목표 근처를 아직 움직이는 표본 복사 수
BRAKE_TRIG, BRAKE_MARGIN = 1.0, 1.15   # 빠른 접근: 목표에 서는 데 필요한 감속이 1.0 m/s² 넘으면 그 1.15배 이상 감속하라는 정답


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
    # 목표 STOP_ZONE m 안에서는 확실히 멈추라는 정답(−HOLD × 5 m/s²). 비례 정답만 쓰면 목표 근처에서 정답이
    # 0에 가까워(예: 0.06 m/s → −0.02) 신경망 오차로 계속 기어가다 앞차에 닿았다
    # (r6 102k 960회 충돌 10 중 8이 0.2 → 0.06 m/s로 줄면서 끝내 안 서고 앞차·버스에 닿음).
    # 앞차는 1.0 m(범퍼 4.0~5.0 m에서 멈춤): 0.6 m로는 버스 뒤에서 0.45 → 0.08 m/s로 계속 기어가 닿았다 (bcstop3 시드 935·1100)
    lab = np.where(rem < np.where(veh, STOP_ZONE_VEH, STOP_ZONE), np.minimum(lab, -HOLD), lab)
    A[m] = lab[m]
    # 빠른 접근 (V_RULE 이상): 적색 정지선이나 더 느린 앞차까지 남은 거리에서 서는(앞차 속도로 맞추는) 데 필요한 감속
    #   need = (v − 목표 속도)² / (2 · 남은 거리)
    # 보다 정답이 약하면 need × BRAKE_MARGIN으로 바꾼다. 정답 버그 때문에 이 구간 정답은 학생 명령 그대로라서,
    # 18 m/s로 적색 신호 84 m 앞까지 거의 안 줄이다가 신호 너머 서 있는 대기 줄을 보고서야 최대 제동 → 충돌했다
    # (r7 53k 960회 충돌 24 중 20이 x6_N_in1 출발 30~45초, 신호위반 10 중 5가 바로 앞 -88000000100#2).
    # 관측 속도는 13.89 m/s에서 잘리므로 need는 실제보다 작게 잡힌다 (안전 쪽으로 더 줄일 일은 없음).
    rl = np.maximum(dl - GAP_LINE, 0.5)
    need_l = np.where(red, v ** 2 / (2 * rl), 0.0)
    rv = np.maximum(g - GAP_VEH, 0.5)
    dv = np.maximum(v - np.maximum(lv, 0.0), 0.0)
    need_v = np.where(has_lead, dv ** 2 / (2 * rv), 0.0)
    need = np.maximum(need_l, need_v)
    fb = (~m) & (v >= V_RULE) & (need >= BRAKE_TRIG)
    lab_b = -np.minimum(need * BRAKE_MARGIN, DMAX) / DMAX
    A[fb] = np.minimum(A[fb], lab_b[fb])
    # 황색 (V_RULE 이상): 정지선까지 서는 데 필요한 감속이 YEL_DEC 안이면 서는 정답(need × 1.15), 아니면 멈칫하지 말고
    # 그대로 지나가는 정답(가감속 ≥ 0, 앞차 때문에 줄여야 할 때는 제외). 정답이 없어서 34 m 앞 황색에 줄이다 말다
    # 적색 직후 정지선을 넘었다 (bcstop3 시드 1061·1086). 관측에 황색 남은 시간이 없어서, 경계를 3.5 m/s²로 두니 황색 중간에
    # 덜 줄인 순간 "통과"로 바뀌어 다시 넘었다 (bcstop4) → 설 수 있으면 서도록 4.5 m/s²로.
    yel = (np.abs(O[:, 31]) < 0.5) & (O[:, 37] < 0.999) & (v >= V_RULE) & ~m
    need_y = v ** 2 / (2 * rl)
    ys = yel & (need_y * BRAKE_MARGIN <= YEL_DEC)
    yg = yel & ~ys & (need_v < BRAKE_TRIG)
    A[ys] = np.minimum(A[ys], -(need_y[ys] * BRAKE_MARGIN) / DMAX)
    A[yg] = np.maximum(A[yg], 0.0)
    # 속도 관측 포화: [0]은 제한속도 13.89 m/s에서 잘려 그 위 속도를 구분하지 못한다. 정답 버그로 포화 표본 일부가
    # "가속"(학생 명령 그대로)이라 19 m/s까지 올라가 적색 신호를 늦게 보고 대기 줄에 닿았다 (bcstop5 시드 923, 80 m 앞 적색에 +0.1)
    # → 포화 표본은 살짝 줄이는 정답. 제한속도 초과는 보상에서도 감점(overspeed_penalty)이다.
    sat = (O[:, 0] >= 0.995) & (A > SAT_LAB)
    A[sat] = SAT_LAB
    m = m | fb | ys | yg | sat
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
    # 범위: 앞차는 멈춤 구간 전체(1.0 m), 정지선은 0.5 m. 0.5 m로만 하니 범퍼 4.4 m에 선 뒤 95 m 뒤에 빠른 차가 나타나자
    # +0.27로 출발해 앞차에 닿았다 (bcstop4 시드 912)
    near = np.where(m & (rem <= np.where(veh, STOP_ZONE_VEH, 0.5)) & (d["obs"][:len(m), 0] * VMAX < 0.5))[0]
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
    # 앞차 목표 근처(멈춤 구간)에서 아직 움직이는 표본은 서 있는 표본보다 10배 넘게 적어 "확실히 멈춤"이 잘 안 배워졌다 → NEAR_DUP개 더 복사
    n0 = len(m)
    mv = np.where(veh & (rem < STOP_ZONE_VEH) & (d["obs"][:n0, 0] * VMAX > 0.05) & (d["obs"][:n0, 0] * VMAX < 1.5))[0]
    if NEAR_DUP > 0 and len(mv):
        for k in ("obs", "accel", "lane_cls"):
            d[k] = np.concatenate([d[k]] + [d[k][mv]] * NEAR_DUP)
        print(f"  목표 근처 움직이는 표본 {len(mv)} × {NEAR_DUP}")
    np.savez_compressed(sys.argv[2], **d)


if __name__ == "__main__":
    main()
