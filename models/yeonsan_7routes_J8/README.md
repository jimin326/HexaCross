# 연산교차로 7경로 모델 J8 (모방학습 + DAgger 7회, 앙상블 8개)

> 2026-11-20 · 브랜치 `traffic/yeonsan-capacity`

I8을 학생으로 DAgger 7차 데이터(B7, 24만 스텝)를 더 모은 뒤, 같은 방법으로 구성원 8개를 다시 학습했습니다. 학습 방법은 넓은 차로 변경 보강(`AUG_MODE=wide`)입니다.

- **지도:** 늘린 교통량 지도입니다. 주변 신호 수요 분배와 실제 버스가 들어 있습니다.
- **관측:** 진행 방향 앞 차체 감지를 켰습니다(`OBSERVATION["corridor_leader"]`, 기본값).

## 성적 (결정적 정책, 경로마다 시드당 15회)

| 교통량 (실측 대비) | 평가 | 완주 | 충돌 | 신호 위반 | 차로 갇힘 | 시간 초과 |
|---|---|---|---|---|---|---|
| **0.7 + 버스 (지금 기본값)** | 420회 (시드 921~924) | **98.1%** | **0.48%** | 0.48% | 0.95% | 0 |
| **0.85 + 버스** | 840회 (시드 921~924, 931~934) | **97.0%** | **0.60%** | 0.60% | 1.67% | 0.12% |

같은 지도, 같은 시드에서 이전 모델과 비교하면 다음과 같습니다.

| 모델 | 0.7 완주 / 충돌 (420회) | 0.85 완주 / 충돌 (840회) |
|---|---|---|
| final (이전 최종) | 95.5% / 1.9% | 90.2% / 6.4% (420회) |
| G8 | 96.7% / 0.48% | 93.7% / 3.7% |
| I8 | 97.1% / 0.48% | 96.2% / 0.48% |
| **J8** | **98.1% / 0.48%** | **97.0% / 0.60%** |

- I8보다 신호 위반(1.4%에서 0.6%)과 시간 초과(0.8%에서 0.1%)가 줄었습니다. 충돌은 I8과 차이가 없는 수준입니다(840회 중 4회와 5회).
- 남은 실패는 대부분 차로 갇힘입니다. 0.85에서 14회이고, SW→N 5회, NE→SW 4회, E→W 3회입니다.
- 원자료: `eval.json`

## 어떻게 만들었나

`models/yeonsan_7routes_I8/README.md`와 같습니다. 다른 점은 아래 두 가지입니다.

- 데이터에 DAgger B7을 더했습니다. 학생은 I8이고, 0.85 지도 워커 3개와 0.7 지도 워커 1개로 모았습니다.
- 구성원 시드는 801~808입니다.

```bash
LANE_LABEL=best N_WORKERS=4 SEED=871 python tools/rl/dagger_collect.py models/yeonsan_7routes_I8/model.pt data/dgB7 60
```

```bash
AUG_LC=2 AUG_MODE=wide SEED=801 LANE_FROM_DG=1 EPOCHS=15 LC_WEIGHT=8 python tools/rl/bc_train.py data/j801.pt \
  data/e800.npz data/dg1.npz data/dg2.npz data/dgB1.npz data/dgB2.npz data/dgB3.npz data/dgB4.npz data/dgB5.npz data/dgB6.npz data/dgB7.npz
```

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 앙상블 8개 (`{"algorithm": "ensemble", "obs_dim": 56, "act_dim": 2, "members": [...]}`) |
| `eval.json` | 위 성적표 원자료 (경로별) |

```bash
python test.py models/yeonsan_7routes_J8/model.pt
```
