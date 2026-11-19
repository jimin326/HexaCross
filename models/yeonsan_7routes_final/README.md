# 연산교차로 7경로 최종 모델 (모방학습 + DAgger, 8개 앙상블)

> 2026-11-19 · 학습·평가: 호섭 (Claude Code로 진행) · 브랜치 `train/yeonsan-6way`

연산교차로(부산 6지 신호교차로)를 지나는 경로 7개를 운전하는 모델입니다. 같은 구조의 정책 신경망 8개를 평균해서 운전합니다(앙상블).

## 성적 (처음 보는 교통 시드 6개, 경로마다 120회, 결정적 정책, 모두 840회)

| 경로 | 완주 | 충돌 | 신호 위반 | 차로 갇힘 |
|---|---|---|---|---|
| S→W→회전교차로→W→E (3.1 km) | 119 | 0 | 1 | 0 |
| N→S | 120 | 0 | 0 | 0 |
| NE→SW | 120 | 0 | 0 | 0 |
| E→W | 115 | 5 | 0 | 0 |
| S→N | 118 | 1 | 1 | 0 |
| SW→N | 111 | 9 | 0 | 0 |
| W→E | 119 | 0 | 1 | 0 |
| **합계** | **822 (97.9%)** | **15 (1.79%)** | **3 (0.36%)** | **0** |

- **시간 초과:** 0회입니다.
- **선택에 쓴 시드:** 모델을 고를 때는 시드 905~908(560회)을 썼고, 거기서는 완주 97.0%, 충돌 1.96%였습니다. 위 표는 고르는 데 쓰지 않은 시드 909~914 결과입니다.
- **목표 대비:** 완주 90% 이상은 넘었습니다. 충돌은 1.8%로, 프로젝트 평가 기준(5% 미만)은 넘었지만 팀 목표(약 1%)에는 조금 못 미칩니다. 충돌은 SW→N과 E→W에 몰려 있습니다.

### 이전 모델과 비교

| 모델 | 방법 | 완주 | 충돌 |
|---|---|---|---|
| C0 | PPO 강화학습 | 65% | 6.2% |
| BC | 모방학습 + 앙상블 증류 | 85.0% | 4.1% |
| **final** | 모방학습 + DAgger(경로 차로 정답) + 8개 앙상블 | **97.9%** | **1.79%** |

## 쓰는 법

`test.py`의 `load_agent`가 앙상블 파일을 자동으로 알아봅니다. 구현은 `algorithms/ensemble.py`에 있습니다.

```bash
python test.py models/yeonsan_7routes_final/model.pt
```

```bash
SEED=909 python tools/rl/eval_routes.py models/yeonsan_7routes_final/model.pt env/sumo_expert/w0 20 out.json
```

관측은 56개입니다(`DATA_FORMAT.md` 4절, `env/mdp_config.py` 기본값). 내용은 이렇습니다.

- 관측 거리 100 m
- 교차로 안 관측
- 노면 화살표
- 큰 틀 경로 안내

행동은 가감속과 차로 변경 2개입니다.

## 어떻게 만들었나

`models/yeonsan_7routes_BC/README.md`의 방법(전문가 데이터, 교사 보정, 위험 상황 보강)에 **DAgger 차로 정답을 바꾼 것**이 더해졌습니다.

1. **전문가 데이터:** SUMO 운전 모델이 AV를 몰게 하고, 운전자 실수(sigma)는 껐습니다. 800회, 53만 스텝을 모았습니다(`tools/rl/expert_collect.py`).
2. **DAgger 1·2회차:** 초기 BC 학생으로 모은 7만 스텝입니다. 차로 정답은 SUMO 차로 변경 모델의 "바꾸고 싶고 바꿀 수 있음"입니다.
3. **DAgger B1·B2 (이번에 추가):** 차로 정답을 **"경로를 이어 가려면 옮겨야 할 차로"**(`getBestLanes`의 오프셋)로 바꿨습니다. 그쪽이 막혀 있으면 "유지"입니다. 추월 목적 변경이 섞이지 않아서, 회전교차로 입구처럼 미리 오른쪽에 있어야 하는 곳을 배웠습니다.
   - B1: 강화학습으로 다듬은 BC 학생(RB0)으로 15만 스텝을 모았습니다.
   - B2: 그 데이터로 학습한 4개 앙상블을 학생으로 17만 스텝을 모았습니다.
   - 도구: `tools/rl/dagger_collect.py`, `LANE_LABEL=best`
4. **구성원 8개:** 위 데이터 전부(92만 스텝)로 시드만 바꿔 8개를 학습했습니다. 그중 4개는 위험 상황 보강을 넣었습니다(`AUG_LC=2`).
   - 가감속 손실은 다음 스텝 속도 오차이고, 교사 보정(앞차 2 m 여유, 적색 정지선 1 m 앞 정지)을 적용했습니다.
   - 도구: `tools/rl/bc_train.py`
5. **앙상블:** 가감속은 평균, 차로는 확률의 로그 평균에서 가장 큰 쪽을 고릅니다.
   - 증류한 단일 모델은 같은 시드에서 완주 95%, 충돌 4.3%였고, 앙상블은 완주 99%, 충돌 0.7%였습니다. 충돌을 지키려고 앙상블 그대로 씁니다.

### 해 봤지만 안 쓴 것

- **BC 모델에서 강화학습(PPO)으로 이어 학습** (모방학습 손실을 함께 걸어도)
  - 10만 스텝 저장본은 충돌이 BC의 절반(2.3%)이었습니다.
  - 하지만 회전교차로 경로를 놓쳤고, 더 학습하면 충돌이 다시 늘었습니다(6~10%).
- **노면 화살표 5칸(약간 왼쪽 / 크게 왼쪽 구분):** 회전교차로 경로가 나빠졌습니다.

## 다시 만들기

모두 저장소 루트에서 실행합니다.

```bash
N_WORKERS=4 SEED=21 python tools/rl/expert_collect.py data/e800 200
```

```bash
# 초기 학생과 DAgger 1·2 (차로 정답 wants)
python tools/rl/bc_train.py data/s0.pt data/e800.npz
```

```bash
LANE_LABEL=wants N_WORKERS=4 SEED=101 python tools/rl/dagger_collect.py data/s0.pt data/dg1 60
```

```bash
# DAgger B (차로 정답 best). 학생은 그때까지 가장 좋은 모델 (.pt, 앙상블 .pt, 또는 구성원 경로 목록 .ens)
LANE_LABEL=best N_WORKERS=4 SEED=601 python tools/rl/dagger_collect.py data/student.pt data/dgB1 60
```

```bash
# 구성원 (시드를 바꿔 8번, 절반은 AUG_LC=2 추가)
SEED=21 LANE_FROM_DG=1 EPOCHS=15 LC_WEIGHT=8 python tools/rl/bc_train.py data/m21.pt data/e800.npz data/dg1.npz data/dg2.npz data/dgB1.npz data/dgB2.npz
```

```bash
# 앙상블 파일로 묶기
python -c "import torch; from algorithms.ensemble import EnsemblePolicy; EnsemblePolicy.save('models/my/model.pt', 56, 2, [torch.load(f'data/m{s}.pt') for s in range(21, 29)])"
```

데이터 파일은 용량 때문에 저장소에 넣지 않았습니다(모두 약 20 MB, 위 명령으로 다시 만들 수 있음).

## 공지 관련 확인할 것

- **노면 화살표 관측:** 도로 표시이고, 어느 차로로 가야 하는지는 알려 주지 않습니다.
- **모방학습 방식:** 프로젝트 안내 문서는 "강화학습 / 모방학습"을 모두 허용하고, 3회차 자료에 "Expert Dataset → BC"가 있습니다. 다만 기본은 수업의 강화학습 알고리즘입니다.
- **교사와 학생:** DAgger와 전문가(교사)는 SUMO의 경로 정보를 씁니다. 학습된 AV는 위 관측만 봅니다. 주행 중 보조 규칙은 없습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `model.pt` | 앙상블 8개 (`{"algorithm": "ensemble", "members": [...]}`, 각 구성원은 PPO와 같은 `HybridActorCritic` 256×256) |
| `eval_840.json` | 위 성적표 원자료 (시드 909~914) |

주행 영상은 `docs/videos/final_7routes/`에 있습니다.
