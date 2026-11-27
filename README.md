# HexaCross

**Teaching a self-driving car to cross Busan's six-way Yeonsan junction**

We rebuilt Busan's Yeonsan junction in the SUMO traffic simulator and trained an autonomous vehicle (AV) to drive a 6.7 km loop through it. The AV first imitated an expert driver (behavior cloning, then DAgger) and was then refined with reinforcement learning (PPO). In 960 evaluation runs the final policy (RL4) reached its destination 99.38% of the time with a 0.42% collision rate.

[Project page](https://jimin326.github.io/HexaCross/) · [Dataset (Hugging Face)](https://huggingface.co/datasets/hoseob0902/HexaCross-dataset) · [Slides (PDF)](https://jimin326.github.io/HexaCross/slides/HexaCross_final.pdf) · [Driving video](https://jimin326.github.io/HexaCross/#video)

---

## 소개

부산 연산교차로(여섯 방향 도로가 만나는 신호 교차로)를 SUMO로 옮기고, 그 안에서 스스로 속도와 차로를 고르는 자율주행(AV) 정책을 학습한 프로젝트입니다.

- **도로:** OpenStreetMap 도로망을 바탕으로, 교차로 주변 차로 34개를 로드뷰와 대조해 다시 그렸습니다. 신호는 6현시 120초 주기로 설계했습니다(실제 신호 시간표는 비공개).
- **교통:** 2025년 부산시 교통조사를 따라 교차로에 시간당 4,507대, 주행 경로 주변에 시간당 4,550대를 넣었습니다.
- **AV:** 0.5초마다 56차원 관측을 보고 가감속과 차로 변경을 정합니다.
- **학습:** SUMO 운전 모델을 따라 하는 모방학습(BC → DAgger)으로 시작해, PPO와 모방 손실로 다듬었습니다.

성균관대학교 「자율주행인공지능및제어」 수업(2026년 2학기)의 팀 프로젝트입니다. 팀 이름은 와플 해적단입니다.

## 결과

최종 모델 `models/yeonsan_loopB_RL4`를 루프 B(6,666 m, 신호 13곳)에서 평가했습니다. 교통 조건 4가지 × 240회 = 960회, 결정적 정책입니다.

| 지표 | 결과 |
|---|---|
| 완주 | 954회 (99.38%) |
| 충돌 | 4회 (0.42%) |
| 신호 위반 | 2회 (0.21%) |
| 멈춘 위치 (중앙값) | 앞차 뒤 4.5 m, 정지선 앞 1.6 m |

- 조건별 결과와 학습 과정: [`models/yeonsan_loopB_RL4/README.md`](./models/yeonsan_loopB_RL4/README.md)
- 주행 데이터셋: 10,647,162 transition, 3,600 에피소드 ([Hugging Face](https://huggingface.co/datasets/hoseob0902/HexaCross-dataset), 형식은 [`DATA_FORMAT.md`](./DATA_FORMAT.md)). `models/yeonsan_loopB_RL2` 정책으로 수집했습니다.

## 실행

Python 3.12와 SUMO 1.27.1 기준입니다. SUMO는 `requirements.txt`의 `eclipse-sumo` 패키지로 함께 설치됩니다.

```bash
git clone https://github.com/jimin326/HexaCross && cd HexaCross
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# 최종 모델 평가: 루프 B, 교통 조건마다 240회
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python tools/rl/eval_parallel.py models/yeonsan_loopB_RL4/model.pt results/eval_RL4 --n 240 --workers 4 --chunk 15

# 주행 데이터 수집: 시드 11, 400 에피소드 (공개 데이터셋을 만든 명령)
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 ROAD_DETAILS=driver_styles,taxi,short_stops \
python collect_pomdp_data.py --model models/yeonsan_loopB_RL2/model.pt --episodes 400 --workers 4 --seed 11 --name loopB_RL2_s11

# 도로와 교통 흐름을 sumo-gui로 확인
python view_road.py
```

- Windows PowerShell에서는 환경 변수를 `$env:ROUTE_SET = "loopB"`처럼 먼저 지정한 뒤 `python …` 명령을 실행합니다.
- 다른 버전의 SUMO가 이미 설치되어 있으면 `requirements.txt` 주석대로 `SUMO_HOME`을 지정합니다.
- 주행 영상을 다시 만드는 방법은 [`docs/videos/loopB_RL4/README.md`](./docs/videos/loopB_RL4/README.md)에 있습니다.

## 저장소 구조

```
HexaCross/
├── env/                    # SUMO 환경
│   ├── road_config.py      #   도로 · AV · 일반 차량 설정
│   ├── mdp_config.py       #   관측 · 행동 · 보상 설정
│   ├── sumo_env.py         #   Gymnasium 환경, TraCI 제어
│   ├── maps/               #   연산교차로와 주변 도로망 (제작 과정은 maps/README.md)
│   └── demand/             #   2025년 부산시 교통조사 수요
├── algorithms/             # PPO, BC, 앙상블 · 잔차 정책
├── utils/                  # 신경망, 버퍼, 평가, 로그
├── tools/
│   ├── rl/                 #   모방학습 · DAgger · PPO 학습, 평가, 영상 녹화
│   ├── data/               #   Parquet 변환, 검사, Hugging Face 업로드
│   └── *.py                #   도로망 제작 · 검증
├── models/                 # 단계별 모델과 평가 결과 (모델마다 README)
├── checkpoints/            # 실험 중간 모델 보관
├── docs/                   # 프로젝트 페이지(GitHub Pages), 실험 기록, 도로 조사, 영상
├── collect_pomdp_data.py   # 주행 데이터셋 수집
├── train.py, train_bc.py   # 수업 기본 코드의 PPO · BC 학습
├── test.py, view_road.py   # 모델 주행 확인, 도로 확인
├── DATA_FORMAT.md          # 데이터셋 컬럼 정의
├── data/, results/         # 수업 기본 코드의 예시 데이터 · 결과 (새 결과는 git에 올리지 않음)
└── requirements.txt
```

`env/sumo/`의 SUMO 파일은 스크립트를 실행할 때마다 `env/road_config.py` 기준으로 다시 만들어집니다.

## 문서

| 문서 | 내용 |
|---|---|
| [`DATA_FORMAT.md`](./DATA_FORMAT.md) | 데이터셋 컬럼 정의, 관측 56차원, 수집 방법 |
| [`docs/project_rules.md`](./docs/project_rules.md) | 수업에서 정한 프로젝트 규칙 요약과 이 저장소의 대응 |
| [`models/yeonsan_loopB_RL4/README.md`](./models/yeonsan_loopB_RL4/README.md) | 최종 모델의 학습 방법과 평가 |
| [`docs/experiments/README.md`](./docs/experiments/README.md) | 초기 실험 기록 (7경로, 2026-11-17 ~ 11-22) |
| [`docs/experiments/loopB_log.md`](./docs/experiments/loopB_log.md) | 루프 B 학습 일지 (2026-11-21 ~ 11-27) |
| [`env/maps/README.md`](./env/maps/README.md) | 도로망을 만든 과정과 검증 |
| [`env/demand/README.md`](./env/demand/README.md) | 교통 수요 자료 |
| [`docs/research/연산교차로_도로조사.md`](./docs/research/연산교차로_도로조사.md) | 연산교차로 차로 · 신호 현장 조사 |
| [`checkpoints/README.md`](./checkpoints/README.md) | 중간 모델 이름과 실험 단계 대응 |

## 모델

| 폴더 | 설명 |
|---|---|
| `models/yeonsan_loopB_RL4` | **최종 모델.** 정지선 · 앞차에 알맞게 붙어 서도록 고친 루프 B 모델 |
| `models/yeonsan_loopB_RL2` | 가속 명령을 제한속도에서 자르지 않는 환경에서 다시 학습한 모델. 공개 데이터셋을 수집한 정책 |
| `models/yeonsan_loopB_RL3` | 평가 시드 버그를 고친 뒤 다시 학습한 모델 |
| `models/yeonsan_loopB_RL` | 루프 B를 처음 완주한 단일 신경망 모델 (sr1) |
| `models/yeonsan_loopC_RL` | 좌회전이 많은 루프 경로 C 모델 |
| `models/yeonsan_7routes_*` | 연산교차로 7경로 초기 실험 (PPO, 모방학습, DAgger 앙상블) |

## 프로젝트 규칙

수업에서 정한 규칙을 따릅니다. 주요 내용은 아래와 같고, 전체 목록은 [`docs/project_rules.md`](./docs/project_rules.md)에 있습니다.

- 주행 중에는 학습한 신경망 하나가 행동을 정합니다. 규칙이나 다른 모델이 개입하지 않습니다.
- 관측에는 차로 단위 경로 안내, 남은 신호 시간, 교차로 상충 차량 정보를 넣지 않습니다.
- 가속 명령을 제한속도에서 자르지 않습니다. 과속은 보상에서 감점합니다.
- 충돌률 5% 미만, 주행 데이터 10^7 transition 이상(모든 행에 시뮬레이터 상태 포함)이 기준입니다.

## 팀

| 이름 | 역할 |
|---|---|
| 승준 | SUMO 도로 (OpenStreetMap, 신호, 경로) |
| 지원 | 일반 차량 · 교통 구성 |
| 서영 | AV 환경 (관측 · 행동 · 보상) |
| 호섭 | 학습 · 평가 |
| 지민 | 데이터 수집 · Hugging Face · 프로젝트 페이지 · 저장소 관리 |

브랜치와 PR 규칙은 [`CONTRIBUTING.md`](./CONTRIBUTING.md)에 있습니다.

## 출처

- 수업 기본 코드와 자료: [bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving)
- 지도: © OpenStreetMap contributors
- 교통 수요: 2025년 부산시 교통조사
- 시뮬레이터: [Eclipse SUMO](https://eclipse.dev/sumo/) 1.27.1
