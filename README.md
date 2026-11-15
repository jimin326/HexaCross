# HexaCross

**Learning to Drive Through a Six-Way Signalized Intersection**

부산 연산교차로(6지 신호교차로)를 SUMO로 구현하고, PPO · Behavior Cloning으로 자율주행(AV) 정책을 학습하는 프로젝트입니다.

> 성균관대학교 「자율주행인공지능및제어」 팀 프로젝트 · 팀 **와플 해적단**

[Project Page](https://jimin326.github.io/HexaCross/) · Dataset (Hugging Face, 준비 중) · Slides (PDF, 준비 중)

---

## 팀

| 이름 | 역할 |
|---|---|
| 승준 | SUMO 도로 (OSM, 신호, 경로) |
| 지원 | 일반 차량(HV) · 교통 구성 |
| 서영 | AV 환경 (Observation · Action · Reward) |
| 호섭 | 학습 · 평가 |
| 지민 | 데이터 수집 · Hugging Face · 프로젝트 페이지 · 레포 관리 |

## 폴더 구조

```
HexaCross/
├── env/
│   ├── road_config.py          # 도로 · AV · 일반 차량 설정
│   ├── road_builder.py         # 설정을 SUMO XML 파일로 변환
│   ├── mdp_config.py           # Observation · Action · Reward · Episode 설정
│   ├── sumo_env.py             # Gymnasium 환경과 TraCI 제어
│   └── sumo/                   # 실행 때마다 자동 생성 (git에 올리지 않음)
├── algorithms/                 # ppo.py, bc.py
├── utils/                      # networks, buffer, evaluator, logger
├── view_road.py                # 도로와 교통 흐름 확인
├── train.py                    # PPO 학습
├── train_bc.py                 # BC 학습
├── collect_pomdp_data.py       # 주행 데이터셋 수집
├── test.py                     # 학습 모델 주행 평가
├── DATA_FORMAT.md              # 데이터셋 컬럼 정의
├── docs/                       # 프로젝트 페이지 (GitHub Pages)
├── project_materials/          # 수업 자료 (원본 그대로)
├── data/                       # 수집 데이터 (git 제외, Hugging Face에 업로드)
└── results/                    # 학습 결과 (git 제외)
```

## 실행

SUMO 설치는 수업 자료를 따릅니다: [Windows](./project_materials/1_SUMO_Windows_Setup.md) · [macOS](./project_materials/1_SUMO_macOS_AppleSilicon_Setup.md)

```bash
python view_road.py                               # 도로 확인
python train.py                                   # PPO 학습 → results/run_*/
python test.py results/run_xxx/model.pt           # 평가 (GUI)
python collect_pomdp_data.py --episodes 100 --workers 8   # 데이터 수집 → data/
python train_bc.py                                # BC 학습
```

`env/sumo/`의 SUMO 파일은 위 스크립트가 실행될 때마다 `env/road_config.py` 기준으로 다시 만들어집니다. 도로를 바꾸려면 `road_config.py` / `road_builder.py`를 고치세요.

---

## 협업 규칙

### 1. main에는 직접 push하지 않는다

main은 보호되어 있습니다. 모든 변경은 **브랜치 → PR → 리뷰 승인 → Squash merge** 순서로 들어갑니다.

### 2. 브랜치 이름: `역할/작업내용`

| 접두어 | 용도 | 예시 |
|---|---|---|
| `road/` | 도로 · 신호 · 경로 | `road/yeonsan-osm` |
| `traffic/` | 일반 차량 · 교통량 | `traffic/hv-routes` |
| `mdp/` | obs · action · reward | `mdp/signal-obs` |
| `train/` | 학습 · 평가 | `train/ppo-1m` |
| `data/` | 데이터 수집 | `data/collector-v1` |
| `page/` | 프로젝트 페이지 | `page/method-section` |
| `fix/` | 버그 수정 | `fix/teleport` |

### 3. 작업 흐름

```bash
git switch main
git pull
git switch -c road/yeonsan-osm
# ... 작업 ...
git add .
git commit -m "road: 연산교차로 OSM 도로 생성"
git push -u origin road/yeonsan-osm
# GitHub에서 Pull Request 생성 → 담당자 승인 → 지민이 Squash merge
git switch main
git pull
```

### 4. 커밋 메시지: `역할: 내용`

예) `mdp: obs에 신호 상태 3차원 추가`, `fix: 우회전 슬립 teleport 제거`

### 5. 리뷰

- PR은 작게, 하루 이틀 단위로 올립니다.
- PR을 만들 때 오른쪽 Reviewers에서 지민을 지정합니다. 다른 사람 담당 파일을 고쳤으면 그 사람도 추가합니다.
- 지민이 올린 PR은 다른 팀원 한 명이 승인합니다.
- `env/sumo_env.py`, `env/road_config.py`처럼 여러 명이 고치는 파일은 머지 후 단톡에 알립니다.
- obs · action · reward · 데이터 컬럼이 바뀌면 [`DATA_FORMAT.md`](./DATA_FORMAT.md)도 같이 고칩니다.

---

## 수업 자료

- [0. 프로젝트 소개](./project_materials/0_Project_Introduction.md)
- [1. SUMO 설치 — Windows](./project_materials/1_SUMO_Windows_Setup.md) · [macOS](./project_materials/1_SUMO_macOS_AppleSilicon_Setup.md)
- [2. 도로 및 POMDP 구조](./project_materials/2_Road_Environment_Construction.md)
- [3. Behavior Cloning](./project_materials/3_Model_Training_BC.md)
- [4. 강화학습](./project_materials/4_Model_Training_RL.md)
- [5. 프로젝트 준비](./project_materials/5_Project_Preparation.md)
- [6. 평가 기준](./project_materials/6_Project_Evaluation.md)
- [7. 컨설팅](./project_materials/7_Project_Consulting.md)

## 출처

수업 기본 코드: [bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving)
