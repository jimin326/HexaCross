# 데이터 포맷 (초안 v2 — 확정 전)

> 담당: 지민 · 확인: 서영(obs · reward 정의), 호섭(학습에 쓰는 컬럼)
>
> 이 문서가 확정되기 전에는 대량 수집(10^7)을 시작하지 않는다.
> 확정 후 컬럼을 바꾸면 버전을 올린다 (v2 → v3).
>
> **v2는 루프 경로 B(`ROUTE_SET=loopB`, 순환 흐름 포함) 수집을 위한 초안이다 (11절 "v1 → v2 변경").**
> v1은 연산교차로 주변만 번호표가 있어서, 루프 B 경로 6,666 m 중 3,624 m(54%)가 `leg = -1`, 연산 밖 신호 11개가 모두 "그 밖"으로 기록됐다.
> **v2에서 바꾸는 것은 `state`와 `meta`뿐이다. `observation`과 `action`은 그대로다** (모델 재학습 불필요).
> v2는 코드에 구현했고 루프 B 시험 수집으로 확인했다 (6절). 팀 확인(서영 · 호섭) 뒤 확정한다.

## 0. 수업 기본 항목 대조

`project_materials/6_Project_Evaluation.md` 2.4절 "기본 데이터 수집 항목"은 반드시 포함해야 한다. v2도 모두 유지한다.

| 수업 기본 항목 | v2 | 비고 |
|---|---|---|
| `episode`, `t` | ✅ | `episode` 번호 규칙은 8절 |
| `observation` | ✅ | 수업 19차원 → 우리 56차원 (4절, 확장 허용) |
| `action_raw`, `action.accel_raw`, `action.lane_change_raw`, `action.lane_change`, `action.lane_change_applied` | ✅ | 그대로 |
| `reward`, `terminated`, `truncated` | ✅ | 그대로 |
| `state` | ✅ | **v2에서 확장** (3절) |
| `detected_vehicles` | ✅ | 그대로 |
| `event.collided`, `event.arrived`, `event.lane_before`, `event.lane_after` | ✅ | + `red_light`, `wrong_lane` (v1부터) |
| `next_observation`, `next_state` | ✅ | 에피소드 마지막 스텝에만 저장 (수업 방식과 같음) |
| npz: `observation / action_raw / lane_change / reward / next_observation / terminated / truncated / episode / t / detected_xy / detected_offsets` | ✅ | `collect_pomdp_data.py`에 모두 있음 (2026-11-23 확인) |

## 1. 저장 단위

- **1줄 = 1 transition = SUMO 1스텝 (0.5초)**
- 목표: **10^7 transition 이상** (루프 B 1회 ≈ 2,800 스텝 → 약 3,600 에피소드)
- 수집: `collect_pomdp_data.py --workers N` (SUMO N개 병렬), 여러 PC에 나눠 수집 (8절 시드 규칙)

## 2. 원칙

1. **모든 컬럼은 숫자다.** 문자열(신호 `"G"`, SUMO id `"512187678#15_3"` 등)은 저장하지 않는다.
   범주형 값은 정수 번호나 0~1 숫자로 바꾸고, 번호표는 `meta.json`에 적는다.
2. **신호는 신호등(진입 차로 그룹)별로 나눠서 저장한다.** 한 컬럼에 여러 신호를 섞지 않는다.
   한 진입로 안에서도 차로마다 녹색 시점이 다르면 그룹별로 칸을 나눈다 (v2).
3. **AV가 어느 신호등 앞에 있는지 기록한다** (`state.ego_tl`).
4. **겹치는 정보는 한 번만 저장한다.** 다른 컬럼으로 정확히 계산되는 값은 넣지 않는다
   (예: 가속도 = 연속한 `ego_speed`의 차 ÷ 0.5초 → 저장 안 함).
5. **차로 번호는 0 = 맨 오른쪽**이다 (SUMO 차로 번호, 버스 전용 차로 포함). 현시표(`env/maps/README.md`)의 "1 = 중앙선 쪽"과 다르니 주의한다.
6. **state는 AV 정책 입력이 아니다.** `get_privileged_state()`는 수집 코드에서만 부른다. 그래서 state 확장은 FAQ의 "보조 규칙 금지", "세밀한 주행 정보 금지"와 관계없다.

### 번호표 (meta.json에도 같은 값을 적는다)

#### 도로 위치 `leg`

| 번호 | 도로 | 비고 |
|---|---|---|
| 0~5 | 연산교차로로 **들어가는** 도로: 0 SW 중앙대로(서면) · 1 N 중앙대로(동래) · 2 W 월드컵대로(사직) · 3 S 월드컵대로(남) · 4 E 고분로 · 5 NE 반송로 | v1과 같음 |
| 6~11 | 연산교차로에서 **나가는** 도로 (위 번호 + 6) | v1과 같음 |
| 12 | 연산교차로 안 (우회전 연결로 포함) | v1과 같음 |
| **13 / 14** | **연수로** (루프 방향 / 반대 방향) | v2 추가 |
| **15 / 16** | **중앙천로** (루프 방향 / 반대 방향) | v2 추가 |
| **17 / 18** | **신촌로** (루프 방향 / 반대 방향) | v2 추가 |
| **19** | **회전교차로 안** (신촌로 × 거제천로) | v2 추가 |
| **20 / 21** | **거제천로** (루프 방향 / 반대 방향) | v2 추가 |
| **22 / 23** | **쌍미천로** (루프 방향 / 반대 방향) | v2 추가 |
| **24 / 25** | **세병로** (순환 흐름 `세병로 (1)` / `세병로 (2)`) | v2 추가. AV 경로는 아니지만 순환 흐름 차량이 다님 |
| -1 | 그 밖 (이면도로 등) | |

- **루프 방향** = 루프 B에서 AV가 달리는 방향, **반대 방향** = 같은 도로의 반대 차도.
- 이름 없는 연결로(예: 월드컵대로 → 연수로 `165290632`, 33 m)와 다른 교차로 안(내부 연결)은 **들어온 도로의 번호를 이어받는다** (v1 규칙과 같음).
- 13~25는 연산교차로 번호(0~11)가 먼저 붙은 도로에는 붙이지 않는다. 예: 월드컵대로(1, 8, 9), 반송로(5), 고분로(4)는 v1 번호 그대로.

#### 신호등 번호 `ego_tl` / 신호 컬럼 번호 `i` (`tl_color_i`, `tl_remaining_i`)

| 번호 | 신호 | 비고 |
|---|---|---|
| 0~5 | 연산교차로 진입로 (`leg` 0~5와 같은 순서). 차로가 갈리는 진입로는 **그룹 A** | v1과 같음 (대표 차로를 v2에서 명시) |
| 6~8 | 시청 인근 교차로 진입 그룹 (7경로 지도에서만. 루프 B 지도에는 없음) | v1과 같음 |
| 9 | 그 밖의 신호 | v1과 같음. **v1 코드는 루프 B에서 "그 밖"을 6으로 기록했다 → v2에서 9로 고정** |
| **10~20** | **루프 B 신호 11곳** (아래 표, 경로 순서) | v2 추가 |
| **21~24** | **연산교차로 차로 그룹 B·C** (아래 표) | v2 추가 |
| -1 | 앞에 신호 없음 | v1과 같음 |

**루프 B 신호 (10~20)** — 거리는 출발점에서 경로를 따라 잰 정지선 위치 (근사)

| 번호 | 위치 | 진입 → 진출 | SUMO 신호 id | 거리 |
|---|---|---|---|---|
| 10 | 연산교차로 북쪽 상류 | 중앙대로 → 중앙대로 | `12785591052` | 445 m |
| 11 | 연수로 끝 | 연수로 → 중앙천로 | `cluster_1772229542_7222328581` | 1,977 m |
| 12 | 중앙천로 끝 (중앙대로와 만남, 가정 신호) | 중앙천로 → 신촌로 | `4633975089` | 2,881 m |
| 13 | 거제천로 | 거제천로 → 거제천로 | `cluster_3131958161_4958136016_4958136222` | 3,701 m |
| 14 | 거제천로 | 거제천로 → 거제천로 | `12508988076` | 4,191 m |
| 15 | 거제천로 | 거제천로 → 거제천로 | `6606304736` | 4,485 m |
| 16 | 거제천로 끝 | 거제천로 → 반송로 | `cluster_14158297210_1773421582` | 4,938 m |
| 17 | 반송로 끝 | 반송로 → 쌍미천로 | `cluster_14158297261_14158297263` | 5,126 m |
| 18 | 쌍미천로 | 쌍미천로 → 쌍미천로 | `7282326875` | 5,466 m |
| 19 | 쌍미천로 끝 | 쌍미천로 → 고분로 | `436818089` | 5,613 m |
| 20 | 연산교차로 서쪽 하류 | 월드컵대로 → 월드컵대로 | `cluster_3131958161_4958136016_4958136222` | 6,478 m |

- 13과 20은 같은 신호 교차로의 서로 다른 진입로다 (진입로마다 번호 하나).
- 10~20은 모두 루프가 지나는 진입로 안에서 차로별 녹색 시점이 같다 (차로 그룹 1개, 2026-11-23 지도로 확인). 그래서 진입로당 한 칸이다.
- 연산교차로는 출발 직후(N 진입, AV 차로 2·3 → `ego_tl` 23)와 도착 직전(E 진입 → `ego_tl` 4) 두 번 지난다.
- 따라서 루프 B 한 바퀴의 `ego_tl` 순서는 **10 → 23 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 4 → 20** 이다 (사이사이 -1). 지도로 확인함 (2026-11-23).
- 10~20 번호는 코드가 AV 경로를 따라가며 매긴다 (진입 edge마다 하나). 경로나 지도가 바뀌면 번호도 바뀌므로 `meta.json`의 `tl_order`로 확인한다.

**연산교차로 차로 그룹 (21~24)** — 동시 녹색 6현시에서 한 진입로 안 차로들의 녹색 시점이 다른 곳

| 진입로 | 차로 수 | 그룹 A (기존 번호 0~5) | 그룹 B | 그룹 C |
|---|---|---|---|---|
| SW 중앙대로 (`x6_SW_in1`) | 7 | 0번 칸 = 차로 0·1·2 | **21번 칸 = 차로 3·4·6** | **22번 칸 = 차로 5** |
| N 중앙대로 (`x6_N_in1`) | 5 | 1번 칸 = 차로 0·1·4 (시청 방면) | **23번 칸 = 차로 2·3** (월드컵대로 남쪽 방면, **루프 B AV가 쓰는 차로**) | – |
| S 월드컵대로 (`x6_S_in1`) | 4 | 3번 칸 = 차로 0·1 | **24번 칸 = 차로 2·3** | – |
| W, E, NE | 4 | 2 · 4 · 5번 칸 = 전 차로 | – | – |

- 차로 번호는 0 = 맨 오른쪽 (원칙 5). 그룹은 코드(`sumo_env._state_layout`)가 신호 프로그램(sumocfg 추가 파일 `highway.tls.xml`까지 반영)에서 현시마다 녹색인 링크를 비교해 나눈다. 그룹 A = 대표 링크(차로 0)가 속한 그룹, B·C는 진입로 순서(`leg_order`: SW, N, W, S, E, NE)대로 21번부터 번호를 붙인다.
- **v1의 문제:** N 진입로의 대표 링크(31)는 차로 0(시청 방면)이었다. 루프 B AV는 차로 2·3으로 출발하므로, 매 에피소드 첫 신호를 다른 그룹의 색으로 기록했다. v2는 23번 칸으로 AV 그룹을 따로 기록한다.
- SW는 버스 전용 차로(BRT)를 포함한 7차로다. 신호 프로그램상 차로 6이 일반 차로 3·4와 같은 그룹(B)으로 묶이고 차로 5만 따로(C) 켜진다. 실제 BRT 신호와 맞는지는 신호 가정(`yeonsan_x6.json`)을 만든 승준 확인이 필요하다.

#### 그 밖

| 이름 | 값 |
|---|---|
| 신호 색 | 0 = 적색 · 0.5 = 황색 · 1 = 녹색 |
| 차종 (`type`) | 0 = 승용차 계열 · 1 = 버스 |
| 방향 (`angle`) | SUMO 각도 (도). **0 = 북쪽, 90 = 동쪽, 시계방향** |

## 3. 컬럼

### 기본

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `episode` | int | 에피소드 번호 (8절: 에피소드 시드와 같은 값) | – |
| `t` | int | 에피소드 안 스텝 번호 | step |
| `observation` | float[**56**] | AV가 보는 부분관측 (4절) | 정규화 [-1, 1] |
| `action_raw` | float[2] | 정책 출력 [가감속, 차로변경] | [-1, 1] |
| `action.accel_raw` | float | 가감속 명령 | [-1, 1] |
| `action.lane_change_raw` | float | 차로변경 원시값 | [-1, 1] |
| `action.lane_change` | int | 양자화된 명령 (-1 오른쪽 / 0 유지 / +1 왼쪽) | – |
| `action.lane_change_applied` | bool | 안전 게이트를 통과해 실제로 실행됐는지 | – |
| `reward` | float | 스텝 보상 | – |
| `terminated` | bool | 충돌 · 완주 · 신호위반 · 차로 갇힘으로 끝남 | – |
| `truncated` | bool | 시간 초과로 끝남 | – |
| `state` | object | 시뮬레이터 절대 상태 (아래) | – |
| `detected_vehicles` | list | 관측 범위(시야 W = 100 m) 안에서 감지된 차량. 원소: `x, y, rel_pos, speed, lane_offset` | m, m/s |
| `event.*` | – | 아래 event 표 | – |
| `next_observation` / `next_state` | – | **에피소드 마지막 스텝에만 저장** (중간은 다음 줄과 같음. `iter_transitions()`가 복원) | – |

### state — AV

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `state.ego_alive` | bool | AV가 시뮬레이션에 있는지 (충돌·도착 후 False) | – |
| `state.ego_x`, `ego_y` | float | AV 좌표 | m |
| `state.ego_speed` | float | AV 속도 | m/s |
| **`state.ego_angle`** | float | **AV 진행 방향 (v2 추가)** | 도 (0 = 북, 시계방향) |
| `state.ego_lane` | int | 차로 번호 (0 = 맨 오른쪽) | – |
| `state.ego_lane_pos` | float | 현재 차로 안 위치 | m |
| `state.ego_leg` | int | AV 도로 위치 (2절 `leg` 번호표, **v2에서 13~25 추가**) | – |
| `state.ego_lane_count` | int | 지금 도로의 일반 차로 수 (버스 전용 차로 제외. 교차로 안에서는 보통 1) | – |
| `state.ego_dist_to_junction` | float | 경로를 따라 연산교차로 입구(정지선)까지 남은 거리. 지난 뒤에는 음수. 경로가 교차로를 두 번 지나면 첫 번째 기준 | m |

- `ego_angle`을 넣는 이유: 좌표 차이로는 정지 중(루프 B 1회 평균 약 600초 정지)에 방향을 알 수 없고, 회전교차로·교차로 안에서의 진행 방향이 필요하다.

### state — 신호 (신호등·차로 그룹별로 한 칸)

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `state.tl_color_i` | float | 신호 `i`의 현재 색 (0 / 0.5 / 1). **`i` = 0~5, 10~24** (2절 번호표. v2에서 10~24 추가) | – |
| `state.tl_remaining_i` | float | 신호 `i`의 색이 바뀔 때까지 남은 시간. **`i`는 위와 같음** | s |
| `state.ego_tl` | int | AV가 다음에 만나는 신호 번호 (2절. **AV 차로 그룹 기준**: N 진입로에서 차로 2·3이면 23) | – |
| `state.ego_tl_dist` | float | 그 신호 정지선까지 거리. 관측 32번과 달리 150 m 제한 없음 (-1 = 신호 없음) | m |

- 연산교차로 신호는 동시 녹색 6현시(가정, `env/maps/yeonsan_x6.json`의 `signal_plan_concurrent`)다. 녹색인 칸이 동시에 2개 이상일 수 있다.
- 칸 수: 색 17 + 남은 시간 17 = 34칸 (v1은 12칸). 루프 B 지도에 없는 6~9번은 칸을 만들지 않는다.
- "지금 몇 번 현시인지"는 따로 저장하지 않는다 (색에서 알 수 있음, 원칙 4).

### state — 주변 차량

| 컬럼 | 타입 | 내용 | 단위 |
|---|---|---|---|
| `state.vehicles[]` | list | **AV 반경 R m 안의 차량** (v2: 반경 제한. R은 5절에서 결정). AV 자신은 빼고 저장 | – |
| ↳ `x`, `y` | float | 좌표 | m |
| ↳ `speed` | float | 속도 | m/s |
| ↳ **`angle`** | float | **진행 방향 (v2 추가)** | 도 (0 = 북, 시계방향) |
| ↳ `lane` | int | 차로 번호 (0 = 맨 오른쪽) | – |
| ↳ `lane_pos` | float | 현재 차로 안 위치 | m |
| ↳ `leg` | int | 도로 위치 (2절 `leg` 번호표) | – |
| ↳ `type` | int | 차종 (2절) | – |

- 주변 차량은 v1에서 id(문자열)를 지웠기 때문에, 다음 줄에서 같은 차를 찾아 방향을 계산할 수 없다. 그래서 `angle`을 따로 저장한다.
- 반경 제한 이유: v1처럼 지도 전체 차량(약 300대)을 저장하면 10^7 스텝에 약 54 GB이고, 스텝마다 차량당 SUMO 호출이 6~7번이라 수집도 느려진다. 위치를 먼저 읽고 반경 밖 차량은 나머지 값을 읽지 않는다.

### event

| 컬럼 | 타입 | 내용 |
|---|---|---|
| `event.collided` | bool | 충돌 |
| `event.arrived` | bool | 목적지 도착 (충돌한 스텝은 False) |
| `event.red_light` | bool | 적색 신호에 정지선 통과 (신호 위반) |
| `event.wrong_lane` | bool | 경로상 갈 수 없는 차로에 갇힘 (차로 선택 실패) |
| `event.lane_before` / `lane_after` | int | 차로변경 전후 차로 번호 |

### v2에서 넣지 않기로 한 것

| 후보 | 넣지 않는 이유 |
|---|---|
| `ego_tl_color` (AV 앞 신호 색 한 칸) | `ego_tl` 번호로 `tl_color_i`를 찾으면 같은 값 → 원칙 4 |
| `ego_accel` (가속도) | 연속한 `ego_speed` 차 ÷ 0.5초로 정확히 계산됨 → 원칙 4 |
| `ego_route_dist` (출발 후 달린 거리) | 관측 38번(남은 거리 비율) × 경로 길이(`meta.route_length_m`)로 계산됨 → 원칙 4 |


## 4. observation 56차원 (v2 변경 없음, 서영 확정 필요)

지금 코드 기준 (`env/sumo_env.py` `_get_obs`, 레이아웃 주석은 `env/mdp_config.py`). 루프 B 모델(`models/yeonsan_loopB_RL`)도 56차원이다 (2026-11-23 확인).
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

`mdp_config.OBSERVATION["traffic_light"]["remaining_time"] = True`면 37 다음에 "내 신호 남은 시간" 1차원이 추가된다 (57차원). 루프 B 모델은 쓰지 않는다.
v1 초안(34차원)의 31 신호 색 · 32 정지선 거리 · 33 남은 시간은 31–37로 바뀌었다 (기본은 남은 시간 없음).

서영 확인 필요:
- **21–25, 39–40 (2026-11-18 바꿈, 프로젝트 공지 반영)**: 예전 21–25는 차로마다 "그 차로로 내 경로를 따라 갈 수 있는 거리"(차로 단위 경로 안내)였고, 39–40은 교차로 안 상충 차량(다른 차의 진행 방향을 이용한 도움 정보)이었다. 공지("어떤 차선으로 주행해야 하는지 등 세밀한 주행 정보 금지", "보조 규칙·도움 요소 금지")에 따라 21–25는 차로 유무만, 39–40은 큰 틀의 경로 안내(다음 교차로 회전 각도·거리)로 바꿨다. 예전 동작은 `mdp_config.OBSERVATION["lane_route_info"] = True`, `junction_foes.enabled = True`, `route_turn.enabled = False`.
- **31–36**: 진입 도로별로 칸을 나눴지만 한 칸에 세 색(−1/0/+1)을 섞었다. FAQ대로 녹·황·적을 칸으로 더 나눌지 정한다 (그러면 차원이 늘어 재학습 필요).
- **38**: 목표까지 남은 거리는 큰 틀의 경로 정보라 공지 범위 안으로 본다.
- **41–55 (노면 화살표, 2026-11-19 추가, 호섭 결정)**: 운전자가 정지선 앞에서 보는 노면 화살표다. "내 경로가 몇 차로로 가야 하는지"는 알려 주지 않는다 (AV는 39–40의 큰 틀 경로 안내와 화살표를 스스로 맞춰야 한다). 공지 "세밀한 주행 정보 금지"에 해당하는지 **컨설팅·조교 확인 필요**.
- 중앙대로 BRT 버스 전용 차로는 관측에서 "없는 차로"로 본다 (`sumo_env._ego_lane_ok`, `_approach_groups`가 버스 전용 차로를 뺀다).

## 5. 정한 것과 남은 것

| 항목 | 선택지 | 지금 (팀 확인 전) |
|---|---|---|
| **`state.vehicles` 반경 R** | 전체 차량 / AV 반경 R m 안 | **150 m** (관측 거리 100 m + 여유, `--state-radius` 기본값). 시험 수집에서 평균 47대 · 용량 약 0.9 KB / transition |
| 파일 형식 | jsonl.gz + npz (수업 기본) / + Parquet | 수집은 수업 기본 형식 그대로. **HF 업로드 때 Parquet로 변환**해 같이 올림 (HF 로딩에 유리, `tl_color_i`처럼 펼친 컬럼이 다루기 쉬움) |
| 샤드 크기 | 파일 하나에 N 에피소드 | 사람별 수집 파일 = 샤드 (8절 시드 규칙으로 겹치지 않음). **남은 것:** HF 단일 파일 크기 제한 확인 |
| 수집 정책 | PPO / BC / 규칙 기반 / 혼합 | **루프 B 모델(`models/yeonsan_loopB_RL/model.pt`), 결정적 정책** |
| 교통 시나리오 | 순환 흐름 켬 / 끔 | **켬 (4,550대/시)**. 조건은 컬럼이 아니라 `meta.json`에 (FAQ "시나리오별 칼럼은 하나로") |
| 신호 인코딩 (state) | 신호등·그룹별 0 / 0.5 / 1 | v1과 같음 (FAQ 예시와 같은 방식) |

## 6. 구현 상태

| 항목 | 구현 위치 | 상태 |
|---|---|---|
| `state.*` (AV, 신호, 주변 차량) | `env/sumo_env.py` `get_privileged_state` · `_signal_state` / 번호표 `_state_layout` → `_number_loop_roads`, `_number_signals` | ✅ `leg` 13~25, 신호 10~24, 차로 그룹, `angle`, 반경 R, "그 밖" 9 |
| `detected_vehicles` | `env/sumo_env.py` `get_detected_vehicles` | 그대로 |
| `event.*`, `meta.json` | `collect_pomdp_data.py` (`write_meta`) | ✅ 8절 meta 항목 |
| 루프 B 수집 | `collect_pomdp_data.py` | ✅ `ROUTE_SET`이 있으면 `routes7.apply()` (평가·녹화와 같은 경로·지도·MAX_STEPS) |
| 에피소드 시드 | `collect_pomdp_data.py` | ✅ 에피소드마다 `np.random.seed(episode)`, 워밍업 스냅샷 풀 기본 끔 (`--state-pool 0`) |
| 환경 난수 재현 | `env/sumo_env.py` `_child_rng()` | ✅ 차량 배치·신호 위상 난수가 전역 시드를 따름 (같은 시드 → 같은 결과) |
| libsumo | `collect_pomdp_data.py` (맨 위) | ✅ 설치돼 있으면 traci 대신 사용, `--gui`면 traci |
| `event.arrived` 충돌 스텝 False | `env/sumo_env.py` `step` (PR #2) | 그대로 |

- 번호표는 연산교차로 OSM 도로망(`ROAD["osm"]["enabled"] = True`)에서만 만든다. 다른 도로에서는 `leg` = -1, `tl_color_*` = 0, `tl_remaining_*` = -1, `ego_dist_to_junction` = 0.
- v1 확인 기록 (2026-11-18): AV 경로를 끝까지 주행하며 703스텝 검사. `ego_leg` 0 → 12 → 7, `ego_tl` 7 → 0 → 9 → -1 순서, `ego_dist_to_junction` 계속 감소, 관측 31·32번과 불일치 0. 수집 속도 16 step/s, JSONL은 transition당 약 36% 감소.
- **v2 확인 결과 (2026-11-23, 루프 B 1회, 시드 9 → 에피소드 900000, 완주 2,601스텝):**
  수업 기본 키·npz 키 모두 있음, observation 56차원, `ego_leg = -1` 0% (v1은 54%),
  `ego_leg` 순서 1 → 12 → 9 → 13 → 15 → 17 → 19 → 20 → 5 → 22 → 4 → 12 → 8,
  `ego_tl` 순서 10 → 23 → 11 → … → 19 → 4 → 20 (2절과 같음), 관측 31번과 `tl_color_{ego_tl}` 불일치 0 / 1,720스텝,
  반경 150 m 주변 차량 평균 47대(최대 157), 용량 JSONL 859 B · NPZ 73 B / transition (10^7이면 약 8.6 GB + 0.7 GB),
  속도 5.5 step/s (traci, 워커 1개).
- **libsumo (2026-11-23):** `pip install libsumo==1.27.1`이 있으면 수집 스크립트가 traci 대신 쓴다 (`USE_LIBSUMO=0`이면 traci).
  같은 시드 1회가 traci 결과와 2,601스텝 전부 같았고, 속도는 113 step/s (워커 1개, 약 20배).
  워커 8개로 16회(40,936스텝): **403 step/s** → 10^7이면 이 PC(Ryzen 5 5600X) 한 대로 약 7시간.
  워커 8개로 돌린 에피소드 900000도 워커 1개 결과와 같았다 (스텝 수 2,601, return 95.795).
  16회 전체에서도 `ego_leg = -1` 0%, `ego_tl` 순서 같음, 관측 31번과 불일치 0 / 26,847스텝, 반경 밖 차량 0대, 에피소드 번호 900000~900015 겹침 없음.

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
| observation 34 → 41 → 56차원 (진입 도로별 신호 6칸, 목표 거리, 큰 틀의 경로 안내 2칸, 노면 화살표 15칸) · 21–25를 차로 유무만으로 | train/yeonsan-6way 브랜치 (학습 · 평가). 4절. 프로젝트 공지 반영 |

## 8. 메타데이터와 시드

수집 실행마다 `<이름>.meta.json`을 같이 저장한다 (`collect_pomdp_data.py`의 `write_meta`).
수업이 요구한 형식은 아니고 팀이 추가한 것이다. 숫자로 바꾼 컬럼의 **번호표**와 **수집 조건**(동료평가 "데이터 수집 규모, 조건을 잘 제시했는가")을 담는다.

### 에피소드 번호 = 에피소드 시드 (v2)

- 에피소드마다 시드를 다시 정한다: **`episode = seed × 100000 + i`** (`seed` = `--seed`, `i` = 그 수집 안의 에피소드 순번 0, 1, 2 …).
- 사람마다 다른 `--seed`를 쓴다 (예: 지민 1, 승준 2, 지원 3, 서영 4, 호섭 5). 그러면 여러 PC의 데이터를 **번호를 고치지 않고 그대로 합칠 수 있다.**
- 같은 커밋·SUMO 1.27.1에서 그 시드로 다시 돌리면 같은 에피소드가 나온다 (6절 확인 결과). 녹화(`tools/rl/record_routes.py`)에도 그 번호를 시드로 쓴다.
- 워밍업 스냅샷 풀(`mdp_config` `warmup_state_pool` = 16)을 켜 두면, 한 프로세스가 16회를 넘긴 뒤의 에피소드는 앞 에피소드의 스냅샷을 다시 써서 시드만으로 재현되지 않는다.
  그래서 **수집은 스냅샷 풀을 끈다** (`--state-pool 0`, 기본값). 평가(`tools/rl/eval_parallel.py`)는 `--chunk`를 16 이하로 쓴다.

### 수집 분담표 (초안 — 팀 확정 전)

여러 PC로 나눠 수집할 때 **사람마다 다른 `--seed`**를 쓴다. 같은 시드를 두 사람이 쓰면 같은 에피소드를 중복 수집하고 에피소드 번호도 겹친다.
시드 번호는 여기서만 정하고, 바꾸면 이 표를 먼저 고친다.

| 사람 | `--seed` | 에피소드 수 | 에피소드 번호 | 파일 이름 (`--name`) | 상태 |
|---|---|---|---|---|---|
| 지민 | 1 | 800 | 100000 ~ 100799 | `loopB_jimin` | 대기 |
| 승준 | 2 | 800 | 200000 ~ 200799 | `loopB_seungjun` | 대기 |
| 지원 | 3 | 800 | 300000 ~ 300799 | `loopB_jiwon` | 대기 |
| 서영 | 4 | 800 | 400000 ~ 400799 | `loopB_seoyoung` | 대기 |
| 호섭 | 5 | 800 | 500000 ~ 500799 | `loopB_hoseop` | 대기 |
| (예약) | 9 | – | 900000 ~ | 시험 수집용 (2026-11-23 사용). 본 수집에 쓰지 않음 | – |

- 합계 4,000 에피소드 ≈ 1,020만 transition (루프 B 1회 평균 약 2,560스텝) → 10^7 이상.
- 1인 예상 시간: Ryzen 5 5600X · 워커 8개 · libsumo 기준 403 step/s → 800회 약 1시간 30분. PC마다 다르므로 처음 몇 회의 속도로 다시 계산한다.
- 상태 칸: 대기 → 수집 중 → 완료(파일 크기, transition 수) → 업로드. 완료하면 `meta.json`의 `episode_ids`가 위 번호와 같은지 확인한다.
- 평가 시드(`tools/rl/eval_parallel.py --seed`, 기본 905)와는 별개다.

**수집 전 확인 (모든 사람)**

1. 같은 커밋: 이 표가 들어간 브랜치(또는 머지된 main)를 받고 `git status`가 깨끗해야 한다 (`meta.json`의 `git_dirty`가 `false`).
2. SUMO 1.27.1: `python -m pip install -r requirements.txt` (sumolib · traci · libsumo · eclipse-sumo 1.27.1). 다른 버전 SUMO가 깔려 있으면 `SUMO_HOME`을 eclipse-sumo 폴더로 (requirements.txt 주석).
3. 처음 1~2분 안에 워커들이 실제로 에피소드를 끝내는지 로그를 본다.

**수집 명령** (`--seed`와 `--name`만 표대로 바꾼다)

```bash
# macOS / Linux
ROUTE_SET=loopB MAX_STEPS=6000 LANE_PLAN_RANGE=300 python collect_pomdp_data.py \
    --model models/yeonsan_loopB_RL/model.pt --episodes 800 --workers 8 --seed 1 --name loopB_jimin
```

```powershell
# Windows PowerShell
$env:ROUTE_SET="loopB"; $env:MAX_STEPS="6000"; $env:LANE_PLAN_RANGE="300"
python collect_pomdp_data.py --model models/yeonsan_loopB_RL/model.pt --episodes 800 --workers 8 --seed 1 --name loopB_jimin
```

- `--workers`는 CPU 코어 수 − 1 정도 (컴퓨터가 너무 느려지면 줄인다).
- 수집이 중간에 끊기면 **같은 시드·같은 이름으로 처음부터 다시** 돌린다 (같은 에피소드가 다시 나오므로 번호 충돌 없음). 파일은 `data/<이름>.jsonl.gz`, `.npz`, `.meta.json` 세 개다.

### meta.json 예 (v2)

```json
{
  "format_version": "v2",
  "git_commit": "<collect 실행 시점 커밋 해시>",
  "git_dirty": false,
  "policy": "ppo",
  "model": "models/yeonsan_loopB_RL/model.pt",
  "stochastic": false,
  "seed": 1,
  "episodes": 900,
  "transitions": 0,
  "workers": 8,
  "step_length": 0.5,
  "max_steps": 6000,
  "sumo_version": "1.27.1",
  "libsumo": false,
  "created_at": "2026-11-24T12:00:00",
  "files": {"jsonl": "loopB_jimin.jsonl.gz", "npz": "loopB_jimin.npz"},
  "route_set": "loopB",
  "route_length_m": 6666,
  "episode_ids": {"rule": "seed × 100000 + i (i = 0 … episodes-1)", "first": 100000, "last": 100899},
  "ego_routes": [["N", "S", "E", "W"]],
  "traffic_scale": 1.0,
  "loop_flows": {"enabled": true, "veh_per_hour": 4550,
                 "flows": [{"name": "연수로 (루프 방향)", "vph": 1000, "source": "실측"}, "..."]},
  "vehicles_radius_m": 150,
  "lane_index": "0 = 맨 오른쪽 (버스 전용 차로 포함)",
  "angle": "도, 0 = 북쪽, 90 = 동쪽, 시계방향",
  "leg_code": {"0-5": "연산교차로로 들어가는 도로 (leg_order 순)",
               "6-11": "연산교차로에서 나가는 도로 (leg_order 순 + 6)",
               "12": "연산교차로 안", "13": "연수로 (루프 방향)", "14": "연수로 (반대)",
               "...": "2절 표", "-1": "그 밖"},
  "leg_order": ["SW", "N", "W", "S", "E", "NE"],
  "tl_order": {"0": "SW", "1": "N", "...": "...", "9": "other",
               "10": "12785591052@-88000000100#3", "...": "...",
               "21": "SW 그룹 B", "22": "SW 그룹 C", "23": "N 그룹 B", "24": "S 그룹 B"},
  "tl_lane_groups": {"1": {"edge": "x6_N_in1", "lanes": [0, 1, 4]},
                     "23": {"edge": "x6_N_in1", "lanes": [2, 3]}, "...": "..."},
  "tl_color": {"red": 0.0, "yellow": 0.5, "green": 1.0},
  "vehicle_type": {"car": 0, "bus": 1},
  "signal_structure": {"type": "concurrent", "phases": ["A", "B", "C1", "C2", "D1", "D3"],
                       "cycle_s": 120.0, "yellow_s": 3, "all_red_s": 2},
  "demand": {"scenario": "normal", "scale": 1.0}
}
```

- `git_dirty`: 커밋 안 한 코드 변경이 있는 상태로 수집했는지. **본 수집은 `false`여야 한다.**
- `tl_order`의 키가 `state.ego_tl` 값이다. `tl_lane_groups`로 주변 차량이 어느 신호 칸을 받는지 찾는다 (차량의 도로·차로 → 그룹).
- v1의 `signal.green_s`(진입로별 녹색 시간)는 동시 녹색 6현시와 맞지 않아 `signal_structure`로 바꾼다.

## 9. Hugging Face 업로드 구조 (안)

```
<hf-id>/HexaCross-dataset
├── README.md            # dataset card: 컬럼, 수집 조건, 라이선스, 번호표 설명
├── v2/
│   └── loopB/
│       ├── loopB_jimin.parquet       # 사람별 수집 파일 = 샤드
│       ├── loopB_jimin.meta.json
│       └── ...
├── map/                 # env/maps/yeonsan_area.net.xml (좌표를 도로에 대조할 때)
└── sample/              # 1,000 transition 샘플 (git의 data/sample/에도 둠)
```

## 10. 일정 (안)

| 날짜 | 할 일 |
|---|---|
| 11/23 | ✅ v2 구현 · 시험 수집(1회 + 16회) · libsumo 확인 · PR |
| 11/24 오전 | 팀 확인 (서영 · 호섭, 분담표 · 반경 R) → PR 머지 → 각자 `pip install -r requirements.txt` |
| 11/24 오후 ~ 11/26 | 분담 수집 (8절 "수집 분담표": 사람별 `--seed`, 같은 커밋, SUMO 1.27.1 + libsumo 1.27.1). 루프 B 1회 평균 약 2,560스텝 → 10^7 ≈ 3,900 에피소드 |
| 11/27 | Parquet 변환 · HF 업로드 · 프로젝트 페이지 링크 |

## 11. v1 → v2 변경

| 변경 | 이유 |
|---|---|
| `leg` 13~25 추가 (루프 도로 · 방향별, 회전교차로, 세병로) | v1은 루프 B 경로의 54%(3,624 m)가 `-1`. FAQ "State 정보를 기반으로 도로의 구성을 예측할 수 있으면 충분합니다. 그게 아니라면 State 정보를 추가하세요" |
| 신호 10~20 추가 (루프 B 신호 11곳) → `tl_color_10..20`, `tl_remaining_10..20` | v1은 연산 밖 신호가 모두 "그 밖"이고 색·남은 시간이 없음. FAQ "신호 칼럼은 Dimension을 쪼개어" |
| 신호 21~24 추가 (연산교차로 N · SW · S 진입로의 차로 그룹) | 동시 녹색 6현시라 한 진입로 안에서 차로마다 녹색 시점이 다름. v1은 대표 차로 하나의 색만 기록 → N 진입로에서 루프 B AV 차로(2·3)와 다른 그룹(0)의 색을 기록했음. v1 문서 "차로별 색이 필요하면 v2에서 컬럼을 나눠야 한다"의 결정 |
| "그 밖의 신호" 번호를 9로 고정 | v1 코드는 시청 신호가 없는 지도(루프 B)에서 "그 밖"을 6으로 기록 → 문서(9)와 불일치 |
| `ego_angle`, `vehicles[].angle` 추가 | 정지 중·교차로 안 방향, 주변 차량은 id가 없어 다른 방법으로 계산 불가 |
| `state.vehicles` 반경 R 제한 | 용량(전체 저장 시 약 54 GB)과 수집 시간 |
| `episode` = `seed × 100000 + i` (에피소드별 시드) | 여러 PC 데이터를 그대로 합치기 + 시드로 에피소드 재현 |
| meta: `route_set`, `route_length_m`, `loop_flows`, `traffic_scale`, `max_steps`, `libsumo`, `vehicles_radius_m`, `lane_index`, `angle`, `tl_lane_groups`, `signal_structure` | 수집 조건과 번호표 (동료평가 "수집 규모, 조건") |
| observation 표기 41 → 56 정리 | 문서 안에서 41 / 56이 섞여 있었음. 루프 B 모델은 56차원 |
