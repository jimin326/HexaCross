# 데이터 포맷 (초안 v0 — 확정 전)

> 담당: 지민 · 확인: 서영(obs · reward 정의), 호섭(학습에 쓰는 컬럼)
>
> 이 문서가 확정되기 전에는 대량 수집(10^7)을 시작하지 않는다.
> 확정 후 컬럼을 바꾸면 버전을 올린다 (v1 → v2).

## 1. 저장 단위

- **1줄 = 1 transition = SUMO 1스텝 (0.5초)**
- 목표: **10^7 transition 이상**
- 수집: `collect_pomdp_data.py --workers N` (SUMO N개 병렬)

## 2. 컬럼

굵게 표시한 것은 원본 코드에 없고 우리가 추가하는 컬럼.

### 기본 (원본 `collect_pomdp_data.py`)

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `episode` | int | 에피소드 번호 | – |
| `t` | int | 에피소드 안 스텝 번호 | step |
| `observation` | float[**39**] | AV가 보는 부분관측. 원본 31 + **진입 도로별 신호 6 + 정지선 거리 1 + 목표 거리 1** (아래 3절) | 정규화 [-1, 1] |
| `action_raw` | float[2] | 정책 출력 [가감속, 차로변경] | [-1, 1] |
| `action.accel_raw` | float | 가감속 명령 | [-1, 1] |
| `action.lane_change_raw` | float | 차로변경 원시값 | [-1, 1] |
| `action.lane_change` | int | 양자화된 명령 (-1 오른쪽 / 0 유지 / +1 왼쪽) | – |
| `action.lane_change_applied` | bool | 안전 게이트를 통과해 실제로 실행됐는지 | – |
| `reward` | float | 스텝 보상 | – |
| `terminated` | bool | 충돌 · 완주 · (추가 종료조건)으로 끝남 | – |
| `truncated` | bool | 시간 초과로 끝남 | – |
| `state` | object | 시뮬레이터 절대 상태 (아래) | – |
| `detected_vehicles` | list | 시야 W 안에 감지된 차량 좌표 | m |
| `event.collided` | bool | 충돌 | – |
| `event.arrived` | bool | 목적지 도착 | – |
| `event.lane_before` / `lane_after` | int | 차로변경 전후 차로 번호 | – |
| `next_observation` / `next_state` | – | **에피소드 마지막 스텝에만 저장** (중간은 다음 줄과 같음. `iter_transitions()`가 복원) | – |

### state (절대 정보)

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `state.ego_x`, `ego_y` | float | AV 좌표 | m |
| `state.ego_speed` | float | AV 속도 | m/s |
| `state.ego_lane` | int | 차로 번호 (0 = 맨 오른쪽) | – |
| `state.ego_lane_id`, `ego_road_id` | str | SUMO 차로 id, 도로(edge) id | – |
| `state.ego_lane_pos` | float | 현재 차로 안 위치 | m |
| `state.vehicles[]` | list | 주변 차량: `id, x, y, speed, lane, lane_id, road_id, lane_pos` | m, m/s |
| **`state.signal_state`** | str | AV가 진입하는 신호등의 현재 상태 (예: `"G"`, `"y"`, `"r"`) | – |
| **`state.signal_phase`** | int | 6현시 중 현재 현시 번호 | – |
| **`state.signal_remaining`** | float | 현재 현시 남은 시간 | s |
| **`state.dist_to_stopline`** | float | AV 앞 정지선까지 거리 | m |

### event (추가)

| 컬럼 | 타입 | 내용 |
|---|---|---|
| **`event.red_light`** | bool | 적색 신호에 정지선 통과 (신호 위반) |
| **`event.wrong_lane`** | bool | 경로상 갈 수 없는 차로에 갇힘 (차로 선택 실패) |

## 3. observation 39차원 (서영 확정 필요)

신호는 "AV 앞 신호 하나"가 아니라 **다음 신호 교차로의 진입 도로마다 하나씩** column으로 넣는다.
연산교차로는 6지 교차로라 6개 도로 = 6개 column.

| 인덱스 | 내용 |
|---|---|
| 0 | 내 속도 / vmax |
| 1–20 | 5개 차로 × (앞차 거리, 앞차 속도차, 뒤차 거리, 뒤차 속도차) |
| 21–25 | 5개 차로 경로 연결성 (이 차로로 경로를 따라 갈 수 있는 거리 / lane_plan_range) |
| 26–30 | 5개 차로 전방 밀도 |
| **31** | 다음 신호 교차로 — **내 진입 도로** 신호 (녹 +1 / 황 0 / 적 −1) |
| **32–36** | 같은 교차로의 **나머지 진입 도로** 신호, 내 도로에서 시계방향 순. 도로가 6개보다 적으면(시청 삼거리) 남는 칸 −1 |
| **37** | 정지선까지 거리 / range(150m). 150m 안에 신호 없으면 31–37 전부 1 |
| **38** | 목표(경로 끝 도로)까지 남은 경로 거리 / 출발 시 경로 거리 |

`mdp_config.OBSERVATION["traffic_light"]["remaining_time"] = True`면 37 다음에 "내 신호 남은 시간" 1차원이 추가된다 (40차원).

## 4. 아직 정할 것

| 항목 | 선택지 | 메모 |
|---|---|---|
| `state.vehicles` 범위 | 전체 차량 / AV 반경 R m 안 | 전체 저장 시 1스텝 ≈ 5.4KB(gzip, 차량 ~300대 기준) → 10^7이면 약 **54GB**. 반경 제한 권장 |
| 파일 형식 | jsonl.gz + npz (원본) / Parquet | HF 업로드 · 로딩은 Parquet가 유리 |
| 샤드 크기 | 파일 하나에 N 에피소드 | HF 단일 파일 크기 제한 고려 |
| 수집 정책 | PPO / BC / 규칙 기반 / 혼합 | 정책별로 폴더 분리 |
| 신호 인코딩 | 진입 도로별 −1 / 0 / +1 (적 / 황 / 녹) | obs 31–36 (확정안) |

## 5. 메타데이터

수집 실행마다 `meta.json`을 같이 저장한다.

```json
{
  "format_version": "v0",
  "git_commit": "<collect 실행 시점 커밋 해시>",
  "policy": "ppo",
  "model": "results/run_xxx/model.pt",
  "seed": 0,
  "episodes": 1000,
  "transitions": 0,
  "step_length": 0.5,
  "sumo_version": "1.27.1",
  "created_at": "2026-11-xx"
}
```

## 6. Hugging Face 업로드 구조 (안)

```
<hf-id>/HexaCross-dataset
├── README.md            # dataset card: 컬럼, 수집 조건, 라이선스
├── v1/
│   ├── ppo/
│   │   ├── shard_0000.parquet
│   │   └── meta.json
│   └── rule/
└── sample/              # 1,000 transition 샘플 (git의 data/sample/에도 둠)
```
