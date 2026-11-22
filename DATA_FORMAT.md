# 데이터 포맷 (초안 v1 — 확정 전)

> 담당: 지민 · 확인: 서영(obs · reward 정의), 호섭(학습에 쓰는 컬럼)
>
> 이 문서가 확정되기 전에는 대량 수집(10^7)을 시작하지 않는다.
> 확정 후 컬럼을 바꾸면 버전을 올린다 (v1 → v2).

> **2026-11-22: 대량 수집은 v2(Parquet)로 정했다 (지민 결정).** 5절의 미정 항목 중 차량 범위, 파일 형식, 샤드, 수집 정책을 정했다. 정한 내용과 v1과의 차이는 10절에 있다.
>
> v1은 수업 피드백과 FAQ를 반영한 초안이다 (7절 "v0 → v1 변경").
> 컬럼은 모두 코드에 구현돼 있다 (6절 "구현 상태").

## 1. 저장 단위

- **1줄 = 1 transition = SUMO 1스텝 (0.5초)**
- 목표: **10^7 transition 이상**
- 수집: `collect_pomdp_data.py --workers N` (SUMO N개 병렬)

## 2. 원칙

1. **모든 컬럼은 숫자다.** 문자열(신호 `"G"`, SUMO id `"512187678#15_3"` 등)은 저장하지 않는다.
   범주형 값은 정수 번호나 0~1 숫자로 바꾸고, 번호표는 `meta.json`에 적는다.
2. **신호는 신호등별로 나눠서 저장한다.** 연산교차로 6개 진입부 신호를 각각 한 컬럼씩
   (한 컬럼에 여러 신호를 섞지 않음).
3. **AV가 어느 신호등 앞에 있는지 기록한다** (`state.ego_tl`).
4. **겹치는 정보는 한 번만 저장한다.** 예: 신호 색은 `state`에만 있고 `event`에는 없다.

### 번호표 (meta.json에도 같은 값을 적는다)

| 이름 | 값 |
|---|---|
| 진입부·신호등 번호 (연산교차로) | 0 = SW 중앙대로(서면) · 1 = N 중앙대로(동래) · 2 = W 월드컵대로(사직) · 3 = S 월드컵대로(남) · 4 = E 고분로 · 5 = NE 반송로 |
| 그 밖의 신호등 번호 | 6 = 시청 인근 중앙대로 남행 (연산 → 서면) · 7 = 시청 인근 중앙대로 북행 (서면 → 연산, AV가 지나는 쪽) · 8 = 시청로 · 9 = 그 밖의 신호 · -1 = 앞에 신호 없음. 6~8은 교차로 신호 그룹의 방위각 순서이고, 실제 순서는 `meta.json`의 `tl_order` |
| 신호 색 | 0 = 적색 · 0.5 = 황색 · 1 = 녹색 |
| 도로 위치 (`leg`) | 0~5 = 연산교차로로 **들어가는** 도로 (위 번호) · 6~11 = 연산교차로에서 **나가는** 도로 (번호 + 6) · 12 = 연산교차로 안 (우회전 연결로 포함) · -1 = 그 밖 (이면도로 등). 들어가는·나가는 도로는 교차로에 붙은 도로에서 같은 이름 도로를 따라 도로망 끝까지다 (예: SW는 시청 인근 교차로 너머까지). 다른 교차로 안을 지나는 동안은 들어온 도로의 번호를 이어받는다 |
| 차종 (`type`) | 0 = 승용차 계열 · 1 = 버스 (지금 배경 교통에는 0만 있음) |

## 3. 컬럼

### 기본

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `episode` | int | 에피소드 번호 | – |
| `t` | int | 에피소드 안 스텝 번호 | step |
| `observation` | float[**56**] | AV가 보는 부분관측 (아래 4절). 예전 41차원 + 노면 화살표 15 | 정규화 [-1, 1] |
| `action_raw` | float[2] | 정책 출력 [가감속, 차로변경] | [-1, 1] |
| `action.accel_raw` | float | 가감속 명령 | [-1, 1] |
| `action.lane_change_raw` | float | 차로변경 원시값 | [-1, 1] |
| `action.lane_change` | int | 양자화된 명령 (-1 오른쪽 / 0 유지 / +1 왼쪽) | – |
| `action.lane_change_applied` | bool | 안전 게이트를 통과해 실제로 실행됐는지 | – |
| `reward` | float | 스텝 보상 | – |
| `terminated` | bool | 충돌 · 완주 · 신호위반 · 차로 갇힘으로 끝남 | – |
| `truncated` | bool | 시간 초과로 끝남 | – |
| `state` | object | 시뮬레이터 절대 상태 (아래) | – |
| `detected_vehicles` | list | 관측 범위(시야 W) 안에서 감지된 차량. 원소: `x, y, rel_pos, speed, lane_offset` | m, m/s |
| `event.*` | – | 아래 event 표 | – |
| `next_observation` / `next_state` | – | **에피소드 마지막 스텝에만 저장** (중간은 다음 줄과 같음. `iter_transitions()`가 복원) | – |

### state — AV

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `state.ego_alive` | bool | AV가 시뮬레이션에 있는지 (충돌·도착 후 False) | – |
| `state.ego_x`, `ego_y` | float | AV 좌표 | m |
| `state.ego_speed` | float | AV 속도 | m/s |
| `state.ego_lane` | int | 차로 번호 (0 = 맨 오른쪽) | – |
| `state.ego_lane_pos` | float | 현재 차로 안 위치 | m |
| **`state.ego_leg`** | int | 연산교차로 기준 AV 위치 (2절 `leg` 번호표) | – |
| **`state.ego_lane_count`** | int | 지금 도로의 일반 차로 수 (버스 전용 차로 제외. 교차로 안에서는 지나는 연결 하나라 보통 1) | – |
| **`state.ego_dist_to_junction`** | float | 경로를 따라 연산교차로 입구(정지선)까지 남은 거리. 지난 뒤에는 음수. 경로가 교차로를 두 번 지나면 첫 번째 기준 | m |

### state — 신호 (연산교차로 6개 신호등, 신호등별로 한 컬럼)

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| **`state.tl_color_0` … `tl_color_5`** | float | 신호등 0~5의 현재 색 (0 / 0.5 / 1) | – |
| **`state.tl_remaining_0` … `tl_remaining_5`** | float | 신호등 0~5의 색이 바뀔 때까지 남은 시간 | s |
| **`state.ego_tl`** | int | AV가 다음에 만나는 신호등 번호 (0~5 연산, 6~8 시청 인근, 9 그 밖, -1 = 앞에 신호 없음) | – |
| **`state.ego_tl_dist`** | float | 그 신호등 정지선까지 거리. 관측 32번과 달리 150m 제한 없음 (-1 = 신호 없음) | m |

- 연산교차로 신호는 동시 녹색 6현시(가정, `env/maps/yeonsan_x6.json`의 `signal_plan_concurrent`)다. 서로 안 엇갈리는 차로끼리 함께 녹색이라, 녹색인 신호등이 동시에 2개 이상일 수 있고 한 진입부 안에서도 차로마다 색이 다를 수 있다 (예: N 2·3차로 적색, 4·5차로 녹색). `tl_color_i`는 그 진입부의 대표 링크 하나의 색이다 — 차로별 색이 필요하면 v2에서 컬럼을 나눠야 한다 (지민 확인 필요). "지금 몇 번 현시인지"는 따로 저장하지 않는다.
- 시청 인근 교차로(신호등 6~8)는 AV가 그 앞에 있을 때 `ego_tl`, `ego_tl_dist`로만 기록한다.

### state — 주변 차량

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `state.vehicles[]` | list | 차량 목록. 지금은 AV를 포함한 전체 차량 (반경 R 제한은 5절 "아직 정할 것") | – |
| ↳ `x`, `y` | float | 좌표 | m |
| ↳ `speed` | float | 속도 | m/s |
| ↳ `lane` | int | 차로 번호 (0 = 맨 오른쪽) | – |
| ↳ `lane_pos` | float | 현재 차로 안 위치 | m |
| ↳ **`leg`** | int | 도로 위치 (2절 번호표) | – |
| ↳ **`type`** | int | 차종 (2절 번호표) | – |

### event

| 컬럼 | 타입 | 내용 |
|---|---|---|
| `event.collided` | bool | 충돌 |
| `event.arrived` | bool | 목적지 도착 (충돌한 스텝은 False) |
| `event.red_light` | bool | 적색 신호에 정지선 통과 (신호 위반) |
| `event.wrong_lane` | bool | 경로상 갈 수 없는 차로에 갇힘 (차로 선택 실패) |
| `event.lane_before` / `lane_after` | int | 차로변경 전후 차로 번호 |


## 4. observation 56차원 (서영 확정 필요)

지금 코드 기준 (`env/sumo_env.py` `_get_obs`, 레이아웃 주석은 `env/mdp_config.py`).
신호는 "AV 앞 신호 하나"가 아니라 **다음 신호 교차로의 진입 도로마다 하나씩** column으로 넣는다
(연산교차로는 6지 교차로라 6개 도로 = 6개 column, 2절 원칙 2와 같은 방향).

| 인덱스 | 내용 |
|---|---|
| 0 | 내 속도 / vmax |
| 1–20 | 5개 차로(좌2, 좌1, 현재, 우1, 우2) × (앞차 거리, 앞차 속도차, 뒤차 거리, 뒤차 속도차). 나란히 겹친 차는 거리 0. 현재 차로 앞차는 교차로 안에서 내 진행 방향 앞 25 m·내 차폭 + 0.3 m 안에 있는 차체가 더 가까우면 그 차로 바꾼다 (앞 차체 감지, `OBSERVATION["corridor_leader"]`, 2026-11-20, 개수는 그대로) |
| 21–25 | 5개 차로가 있는지: 1 = 있고 달릴 수 있음, -1 = 없음·버스전용 (눈으로 보이는 정보만. 아래 공지 참고) |
| 26–30 | 5개 차로 전방 밀도 (차로 없음 = 1) |
| 31 | 다음 신호 교차로 — **내 진입 도로** 신호 (녹 +1 / 황 0 / 적 −1). 150m 안에 신호 없으면 1 |
| 32–36 | 같은 교차로의 **나머지 진입 도로** 신호, 내 도로에서 시계방향 순. 도로가 6개보다 적으면(시청 삼거리) 남는 칸 −1 |
| 37 | 정지선까지 거리: sqrt(거리 / 150m) (정지선 가까이를 더 촘촘히: 5m = 0.18, 50m = 0.58). 150m 안에 신호 없으면 31–37 전부 1 |
| 38 | 목표(경로 끝 도로)까지 남은 경로 거리 / 출발 시 경로 거리 |
| 39 | 큰 틀의 경로 안내: 다음 교차로(경로가 20° 이상 꺾이는 곳 또는 신호 교차로)에서 꺾는 각도 / 180 (−1 왼쪽 ~ +1 오른쪽, 0 직진). 300m 안에 없으면 0 |
| 40 | 그 교차로까지 거리 / 300m (없으면 1) |
| 41–55 | 5개 차로(좌2, 좌1, 현재, 우1, 우2) × 노면 화살표 [좌회전, 직진, 우회전]: 그 차로를 따라 150m 안의 첫 신호 교차로 정지선에서 갈 수 있는 방향이면 1, 아니면 0, 차로 없음·버스전용 −1. 내 경로와 무관한 도로 표시 (2026-11-19 추가) |

- 1–20의 거리 기준(관측 거리 W)은 100m다 (2026-11-19 50m → 100m: 신호 대기 줄을 보게). 교차로 안(내부 연결 차로)에서는 곧 들어갈 도로의 차로 기준으로 1–30, 41–55를 계산한다 (`internal_lane_obs`).
- 예전 41차원 모델(`models/yeonsan_7routes_C0`)을 쓰려면 `mdp_config.OBSERVATION`에서 `visibility` 50, `lane_arrows.enabled` False, `internal_lane_obs` False.

`mdp_config.OBSERVATION["traffic_light"]["remaining_time"] = True`면 37 다음에 "내 신호 남은 시간" 1차원이 추가된다 (42차원).
v1 초안(34차원)의 31 신호 색 · 32 정지선 거리 · 33 남은 시간은 31–37로 바뀌었다 (기본은 남은 시간 없음).

서영 확인 필요:
- **21–25, 39–40 (2026-11-18 바꿈, 프로젝트 공지 반영)**: 예전 21–25는 차로마다 "그 차로로 내 경로를 따라 갈 수 있는 거리"(차로 단위 경로 안내)였고, 39–40은 교차로 안 상충 차량(다른 차의 진행 방향을 이용한 도움 정보)이었다. 공지("어떤 차선으로 주행해야 하는지 등 세밀한 주행 정보 금지", "보조 규칙·도움 요소 금지")에 따라 21–25는 차로 유무만, 39–40은 큰 틀의 경로 안내(다음 교차로 회전 각도·거리)로 바꿨다. 예전 동작은 `mdp_config.OBSERVATION["lane_route_info"] = True`, `junction_foes.enabled = True`, `route_turn.enabled = False`.
- **31–36**: 진입 도로별로 칸을 나눴지만 한 칸에 세 색(−1/0/+1)을 섞었다. FAQ대로 녹·황·적을 칸으로 더 나눌지 정한다 (그러면 차원이 늘어 v2).
- **38**: 목표까지 남은 거리는 큰 틀의 경로 정보라 공지 범위 안으로 본다.
- **41–55 (노면 화살표, 2026-11-19 추가, 호섭 결정)**: 운전자가 정지선 앞에서 보는 노면 화살표다. "내 경로가 몇 차로로 가야 하는지"는 알려 주지 않는다 (AV는 39–40의 큰 틀 경로 안내와 화살표를 스스로 맞춰야 한다). 공지 "세밀한 주행 정보 금지"에 해당하는지 팀 확인 필요.
- 중앙대로 BRT 버스 전용 차로는 관측에서 "없는 차로"로 본다 (`sumo_env._ego_lane_ok`, `_approach_groups`가 버스 전용 차로를 뺀다).

## 5. 아직 정할 것

| 항목 | 선택지 | 메모 |
|---|---|---|
| `state.vehicles` 범위 R | 전체 차량 / AV 반경 R m 안 | 전체 저장 시 1스텝 ≈ 5.4KB(gzip, 차량 ~300대 기준) → 10^7이면 약 **54GB**. 반경 제한 권장 |
| `tl_remaining_*` 개수 | 6개 모두 / AV 신호(`ego_tl`) 것만 | 6개 모두 저장해도 1스텝 6칸 |
| 파일 형식 | jsonl.gz + npz (원본) / Parquet | HF 업로드 · 로딩은 Parquet가 유리. Parquet면 `tl_color_0..5`처럼 펼친 컬럼이 다루기 쉬움 |
| 샤드 크기 | 파일 하나에 N 에피소드 | HF 단일 파일 크기 제한 고려 |
| 수집 정책 | PPO / BC / 규칙 기반 / 혼합 | 정책별로 폴더 분리 |
| 신호 인코딩 | 진입 도로별 −1 / 0 / +1 (적 / 황 / 녹) | obs 31–36 (지금 코드) |
| 교통 시나리오 섞기 | 한 데이터셋에 12시·18시 섞기 / 시나리오별 폴더 | 섞을 거면 시나리오별 컬럼을 따로 두지 말고 `meta.json` + 숫자 하나로 구분 (FAQ) |

## 6. 구현 상태

| 컬럼 | 구현 위치 |
|---|---|
| `state.*` (AV, 신호, 주변 차량) | `env/sumo_env.py` `get_privileged_state` / 번호표는 `_state_layout` (도로 파일에서 한 번 만듦) |
| `detected_vehicles` | `env/sumo_env.py` `get_detected_vehicles` |
| `event.*`, `meta.json` | `collect_pomdp_data.py` (`write_meta`) |
| `event.arrived` 충돌 스텝 False | `env/sumo_env.py` `step` (PR #2) |

- 번호표는 연산교차로 OSM 도로망(`ROAD["osm"]["enabled"] = True`)에서만 만든다. 다른 도로에서는 `leg` = -1, `tl_color_*` = 0, `tl_remaining_*` = -1, `ego_dist_to_junction` = 0.
- 확인한 것 (2026-11-18): AV 경로를 끝까지 주행하며 703스텝 검사. `ego_leg` 0 → 12 → 7, `ego_tl` 7 → 0 → 9 → -1 순서, `ego_dist_to_junction` 계속 감소, 관측 31·32번과 불일치 0, 녹색 신호 동시 2개 이상 0 (당시 6현시 신호 기준 — 지금은 동시 녹색 신호라 2개 이상이 정상). 수집 속도는 v0과 같고(16 step/s) JSONL 크기는 transition당 약 36% 감소.

## 7. v0 → v1 변경

| 변경 | 이유 |
|---|---|
| `state.signal_state` (문자열 `"G"/"y"/"r"`) → `state.tl_color_0..5` (0 / 0.5 / 1) | 피드백 1 "String을 쓰지 말자", 2 "6개 신호등으로 나눠서", FAQ "신호는 숫자로, 차원을 쪼개서" |
| `state.signal_phase`, `signal_remaining` → `tl_remaining_0..5` | 신호등별로 나눔. 현시 번호는 색에서 알 수 있어 삭제 (FAQ "겹치는 정보 삭제") |
| `state.ego_tl`, `state.ego_tl_dist` 추가 (v0의 `dist_to_stopline` 대체) | 피드백 3 "어느 신호등인지 기억하는 게 필요" |
| `state.ego_lane_id`, `ego_road_id`, `vehicles[].id / lane_id / road_id` (문자열) 삭제 → `ego_leg`, `ego_lane_count`, `ego_dist_to_junction`, `vehicles[].leg`, `type` 추가 | 피드백 1. FAQ "State로 도로 구성을 예측할 수 있으면 충분" |
| `event.signal_state` (문자열) 삭제 | 피드백 1. `state.tl_color_*`와 겹침 |
| `detected_vehicles[].id` (문자열) 삭제 | 피드백 1 |
| `event.arrived`: 충돌한 스텝은 False | PR #2에서 코드 수정 (`--collision.action remove` 때문에 둘 다 True였음) |
| 4절 observation을 실제 코드 값으로 적음 (31번 인코딩, 32·33 정규화 기준) | v0은 "TBD"였음 |
| `meta.json`을 실제로 저장 (`<이름>.meta.json`) | v0은 문서에만 있고 코드가 만들지 않았음 |
| observation 34 → 41차원 (진입 도로별 신호 6칸, 목표 거리, 큰 틀의 경로 안내 2칸) · 21–25를 차로 유무만으로 | train/yeonsan-6way 브랜치 (학습 · 평가). 4절. 프로젝트 공지 반영 |

## 8. 메타데이터

수집 실행마다 `<이름>.meta.json`을 같이 저장한다 (`collect_pomdp_data.py`의 `write_meta`).

```json
{
  "format_version": "v1",
  "git_commit": "<collect 실행 시점 커밋 해시>",
  "git_dirty": false,
  "policy": "ppo",
  "model": "results/run_xxx/model.pt",
  "stochastic": false,
  "seed": 0,
  "episodes": 1000,
  "transitions": 0,
  "workers": 8,
  "step_length": 0.5,
  "sumo_version": "1.27.1",
  "created_at": "2026-11-18T12:00:00",
  "files": {"jsonl": "pomdp_xxx.jsonl.gz", "npz": "pomdp_xxx.npz"},
  "ego_routes": [["SW", "N"]],
  "leg_order": ["SW", "N", "W", "S", "E", "NE"],
  "leg_code": {"0-5": "들어가는 도로", "6-11": "나가는 도로", "12": "교차로 안", "-1": "그 밖"},
  "tl_order": ["SW", "N", "W", "S", "E", "NE",
               "cluster_12791011258_6612040590:중앙대로@30",
               "cluster_12791011258_6612040590:중앙대로@211",
               "cluster_12791011258_6612040590:시청로@301", "other"],
  "tl_color": {"red": 0.0, "yellow": 0.5, "green": 1.0},
  "vehicle_type": {"car": 0, "bus": 1},
  "signal": {"green_s": {"N": 13, "NE": 13, "E": 12, "S": 13, "SW": 18, "W": 18},
             "yellow_s": 3, "all_red_s": 2},
  "demand": {"scenario": "assumed"}
}
```

- `git_dirty`: 커밋 안 한 코드 변경이 있는 상태로 수집했는지.
- `tl_order`의 번호가 `state.ego_tl` 값이다 (0부터). `@` 뒤 숫자는 그 진입 그룹의 방위각.

- `demand`는 PR #1(실측 교통량)을 합친 뒤부터 의미가 있다. 그 전에는 가정 교통량이라 `{"scenario": "assumed"}`.

## 9. Hugging Face 업로드 구조 (안)

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

## 10. v2: 대량 수집용 Parquet (2026-11-22)

v1과 **값은 같고 저장 방식만 다르다.**
- 수집 도구: `tools/data/collect_parquet.py`
- 검사 도구: `tools/data/check_parquet.py`

### 정한 것 (5절의 미정 항목)

| 항목 | 정한 값 |
|---|---|
| 파일 형식 | Parquet (zstd 압축) 샤드. 1행 = 1 transition |
| 샤드 | 워커마다 약 25만 행(약 100 에피소드)씩. 에피소드는 샤드 경계에서 잘리지 않는다. 이름은 `shard_w{워커}_{번호}.parquet` |
| `vehicles` 범위 | AV 반경 **100 m** 안의 차량만 (관측 거리 W와 같음). AV 자신은 빼고 `ego_*` 컬럼에 둔다 |
| 수집 정책 | 루프 경로 B 최종 모델 `models/yeonsan_loopB_RL/model.pt`. 에피소드마다 결정적(μ, argmax 차로)과 확률적(정책 분포에서 샘플)을 번갈아 쓴다 → `policy_mode` 0 / 1 |
| 교통 | 기본 환경, 교통량 1.0 + 버스. 넓은 지도(`yeonsan_area`), `ROUTE_SET=loopB` |
| 규모 | 10^7 transition (약 3,900 에피소드, 약 5.5 GB) |

### 컬럼 (v1과 다른 점)

| v2 컬럼 | 타입 | v1에서 |
|---|---|---|
| `episode`, `t` | int32 | 같음. `episode`는 워커마다 1,000,000 × (워커+1)부터 |
| `worker` | int8 | 새로 (수집 워커 번호) |
| `policy_mode` | int8 | 새로 (0 결정적, 1 확률적) |
| `observation`, `next_observation` | float32[56] | `next_observation`을 **모든 행에** 저장 (v1은 에피소드 마지막 행만) |
| `action_raw` | float32[2] | 같음 |
| `accel_raw`, `lane_change_raw`, `lane_change`, `lane_change_applied` | | v1의 `action.*`를 펼침 |
| `reward`, `terminated`, `truncated` | | 같음 |
| `ego_alive` … `ego_tl_dist` (21개) | | v1의 `state.*`(주변 차량 제외)를 펼침 |
| `next_ego_alive` … `next_ego_tl_dist` (21개) | | v1의 `next_state.*`. **모든 행에** 저장 |
| `vehicles` | list<struct{x, y, speed, lane, lane_pos, leg, type}> | v1 `state.vehicles`. 반경 100 m 안만 |
| `detected_vehicles` | list<struct{x, y, rel_pos, speed, lane_offset}> | 같음 |
| `collided`, `arrived`, `red_light`, `wrong_lane`, `lane_before`, `lane_after` | | v1의 `event.*`를 펼침 |

- AV가 사라진 행(충돌·도착 직후)은 `next_ego_alive`가 False이고, 나머지 `next_ego_*`는 null이다.
- 번호표(`leg`, `ego_tl`, 신호 색)는 2절과 같다.

### 읽기

```python
import pyarrow.parquet as pq, numpy as np
tb = pq.read_table("data/v2/loopB_RL/shard_w0_0000.parquet")
obs = np.stack(tb["observation"].to_numpy(zero_copy_only=False))        # (N, 56)
act = np.stack(tb["action_raw"].to_numpy(zero_copy_only=False))         # (N, 2)
rew = tb["reward"].to_numpy(); done = tb["terminated"].to_numpy() | tb["truncated"].to_numpy()
nobs = np.stack(tb["next_observation"].to_numpy(zero_copy_only=False))
```

### 검사 (`tools/data/check_parquet.py`)

- 같은 에피소드 안에서 `next_observation[t] == observation[t+1]`, `next_ego_x[t] == ego_x[t+1]`인지
- `t`가 0부터 1씩 늘어나는지
- 마지막 행에만 `terminated` 또는 `truncated`가 있는지

시험 수집(17,935행)에서 오류는 0개였다.

### 저장 위치

- Hugging Face 데이터셋 저장소 `hoseob0902/HexaCross-dataset`(비공개)의 `v2/loopB_RL/`
- 올리는 도구: `tools/data/upload_hf.py` (수집과 동시에 다 쓴 샤드부터 올림)

### 수집 중 메모리

- 처음 버전은 워커마다 25만 행을 파이썬 객체로 들고 있다가 한꺼번에 썼다. 워커 하나에 4 GB를 넘겨 메모리 부족으로 워커가 죽었다(80만 행 손실).
- 지금은 2만 5천 행(`ROWGROUP_ROWS`)마다 열린 샤드에 row group으로 쓴다. 워커당 수백 MB다.
