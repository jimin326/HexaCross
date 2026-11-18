"""SUMO 고속도로 RL 환경 — Gymnasium 인터페이스 + TraCI.

┌─────────────────────────────────────────────────────────────┐
│  강화학습 루프에서 이 클래스의 역할                              │
│                                                             │
│    obs, _ = env.reset()          # 에피소드 시작, 초기 관측     │
│    obs, r, term, trunc, _ = env.step(action)  # 1스텝 진행    │
│                                                             │
│  내부적으로는 TraCI(Traffic Control Interface)라는 소켓 API로  │
│  SUMO 시뮬레이터를 원격 조종한다:                               │
│    traci.simulationStep()  → 시뮬레이션을 1스텝 진행            │
│    traci.vehicle.setSpeed() → 차량 속도 명령                   │
│    traci.vehicle.getSpeed() 등 → 상태 조회                     │
└─────────────────────────────────────────────────────────────┘

관측/행동/보상의 구체적인 정의는 configs/mdp_config.py 에 있고,
이 파일은 그 설정을 받아서 실행만 한다. (MDP를 바꿀 때 이 파일 수정 불필요)
"""
import json
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# SUMO 설치 폴더의 tools/ 안에 traci, sumolib 파이썬 모듈이 들어있다.
# pip로도 설치했지만, 버전 불일치를 피하려면 SUMO_HOME 쪽을 우선 경로에 추가.
#
# SUMO_HOME이 설정 안 돼 있으면 OS별 기본 설치 경로를 순서대로 탐색해서
# 자동으로 잡아준다 (환경변수 설정을 깜빡해도 동작하도록).
if "SUMO_HOME" not in os.environ:
    _candidates = [
        # macOS (Homebrew) — Apple Silicon / Intel
        "/opt/homebrew/opt/sumo/share/sumo",
        "/usr/local/opt/sumo/share/sumo",
        # Linux (apt)
        "/usr/share/sumo",
        # Windows (기본 인스톨러 경로)
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for _p in _candidates:
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            print(f"[sumo_env] SUMO_HOME 자동 감지: {_p}")
            break
    else:
        raise RuntimeError(
            "SUMO_HOME 환경변수가 없고 기본 경로에서도 SUMO를 찾지 못했습니다.\n"
            "  Windows: 인스톨러 재설치(SUMO_HOME 체크) 또는 환경변수 수동 설정\n"
            "  macOS  : ./setup_mac.sh 실행 또는 README_MAC.md 참고\n"
            "  Linux  : sudo apt install sumo sumo-tools 후 "
            "export SUMO_HOME=/usr/share/sumo")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
from sumolib import checkBinary


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True, route_set="train"):
        """
        Args:
            cfg_path: road_builder.build()가 만든 .sumocfg 경로
            road:     configs.road_config.ROAD
            ego:      configs.road_config.EGO (가속/감속 한계가 여기서 옴)
            mdp_sim:  configs.mdp_config.SIMULATION
            mdp_obs:  configs.mdp_config.OBSERVATION
            action:   configs.mdp_config.ACTION (행동 차원 수)
            reward:   configs.mdp_config.REWARD
            traffic:  configs.road_config.TRAFFIC (프리필/컨트롤러 정보).
                None이면 프리필 없이 flow만으로 동작 (하위호환)
            gui:      True면 sumo-gui(화면 있는 버전)로 실행
            gui_autostart: (gui=True일 때만 의미)
                True  → 창이 뜨면 시뮬레이션이 자동으로 시작
                False → 사용자가 ▶(플레이) 버튼을 누를 때까지 대기.
                        sumo-gui는 --start 옵션이 없으면 TraCI의
                        simulationStep() 호출도 플레이 전까지 블록되므로,
                        에피소드 전체가 "▶를 눌러야 시작"되는 효과가 난다.
        """
        super().__init__()
        self.cfg_path = cfg_path
        # "train": EGO["routes"], "demo": EGO["demo_routes"] (시연 영상용 긴 경로)
        self.route_set = route_set
        self.gui = gui
        self.gui_autostart = gui_autostart

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
        # 직선 도로 예제에만 있는 값 (방사형 교차로에서는 0)
        self.num_lanes = int(road.get("num_lanes", 0))
        # 시작 차선: "center"면 출발 엣지의 중앙 차선(차선수//2)으로 reset 때 결정.
        #   차선 수를 바꿔도 항상 가운데라 좌/우 변경 여지가 대칭으로 생긴다.
        #   "best"면 SUMO가 경로상 가장 적합한 차선을 고른다 (교차로 회전 경로용).
        self.ego_depart_lane = str(ego["depart_lane"]).lower()
        # 데이터셋 state(get_privileged_state)의 도로·신호 번호표용 (DATA_FORMAT v1)
        self.road_legs = list(road.get("legs", []))
        self._layout = None           # 처음 필요할 때 도로 파일에서 한 번 만든다
        self._ego_dist_ref = None     # 에피소드마다 다시 잡는다 (reset)
        # 방사형 교차로 빌더(env/junction_builder.py)가 남긴 경로/진입 도로 정보.
        # 직선 도로 예제에서는 없다 (None).
        meta_path = os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                                 "highway.meta.json")
        self.meta = None
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
        # ---- 프리필(도로 전체 사전 배치) 설정 ----
        self.traffic = traffic or {}
        pf = (self.traffic.get("prefill") or {})
        self.prefill_enabled = bool(pf.get("enabled", False))
        self.prefill_start = float(pf.get("start_offset", 80.0))
        self.prefill_end_margin = float(pf.get("end_margin", 50.0))
        self.prefill_jitter = float(pf.get("spacing_jitter", 0.3))
        # ---- 차선변경 안전 게이트 설정 (mdp_config.ACTION["lane_change_safety"]) ----
        lcs = (action.get("lane_change_safety") or {})
        self.lc_safety_enabled = bool(lcs.get("enabled", True))
        self.lc_front_min = float(lcs.get("min_front_gap", 5.0))   # m
        self.lc_rear_min = float(lcs.get("min_rear_gap", 5.0))     # m
        self.lc_front_tau = float(lcs.get("front_tau", 0.3))       # s
        self.lc_rear_tau = float(lcs.get("rear_tau", 0.5))         # s
        self.lc_spawn_protect = float(lcs.get("spawn_protect", 60.0))  # m
        self.lc_horizon = float(lcs.get("approach_horizon", 1.0))       # s
        self.road_length = float(road.get("length", 0.0))
        self.traffic_max_speed = float(self.traffic.get("max_speed", self.vmax))
        # 차선당 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
        #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
        #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
        self.prefill_spacing_fixed = (float(pf["spacing"]) if pf.get("spacing")
                                      else None)
        if pf.get("spacing"):
            self.prefill_spacing = float(pf["spacing"])
        else:
            vph = float(self.traffic.get("vehs_per_hour", 1800))
            q_lane = max(vph / 3600.0 / max(self.num_lanes, 1), 1e-6)
            self.prefill_spacing = self.traffic_max_speed / q_lane
        # 컨트롤러 타입 샘플링 준비 (rou.xml의 car_<이름> vType과 대응)
        ctrls = self.traffic.get("controllers") or {}
        if ctrls:
            names = list(ctrls.keys())
            probs = np.array([float(ctrls[n].get("probability", 1.0)) for n in names])
            self._prefill_types = [f"car_{n}" for n in names]
            self._prefill_probs = probs / probs.sum()
        else:
            self._prefill_types, self._prefill_probs = ["car"], np.array([1.0])
        self.ego_depart_speed = str(ego["depart_speed"])
        # 행동 a ∈ [-1,1] 을 실제 가속도로 변환할 때 쓰는 스케일:
        #   a = +1 → +max_accel (풀 액셀),  a = -1 → -max_decel (풀 브레이크)
        self.max_accel = float(ego["accel"])   # m/s^2
        self.max_decel = float(ego["decel"])   # m/s^2 (양수로 저장, 부호는 계산 시)

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        self.warmup_seconds = tuple(mdp_sim.get("warmup_seconds", (150.0, 300.0)))
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        # 차선 연결성 정규화 거리: "이 차선으로 경로를 따라 계속 갈 수 있는 거리"를
        # 이 값으로 나눈다. 교차로 회전 전에 차선을 미리 바꿀 수 있을 만큼 길게.
        self.lane_plan_range = float(mdp_obs.get("lane_plan_range",
                                                 mdp_obs["visibility"]))
        self.wrong_lane_steps = 0      # 경로상 막힌 차선 끝에 서 있는 연속 스텝 수
        # ---- 신호 관측 설정 (mdp_config.OBSERVATION["traffic_light"]) ----
        tl = (mdp_obs.get("traffic_light") or {})
        self.tl_obs = bool(tl.get("enabled", False))
        self.tl_range = float(tl.get("range", 150.0))
        self.tl_max_time = float(tl.get("max_time", 30.0))
        # 다음 교차로의 진입 도로별 신호 column 수 (6지 교차로 = 6)
        self.tl_columns = int(tl.get("approach_columns", 6))
        self.tl_remaining = bool(tl.get("remaining_time", False))
        self._tls_groups = {}          # 신호 id -> 진입 도로 묶음 (캐시)
        self._lane_ok = {}             # 차로 id -> 승용차가 달릴 수 있는가 (버스전용차로 = False, 캐시)
        # ---- 목표(경로 끝 도로) 진행 관련 ----
        self.goal_obs = bool(mdp_obs.get("goal_distance", True))
        # ---- 교차로 상충 차량 (내 경로와 교차로 안에서 엇갈리는 차) ----
        jf = (mdp_obs.get("junction_foes") or {})
        self.foe_obs = bool(jf.get("enabled", False))
        self.foe_range = float(jf.get("range", 60.0))
        self.foe_time = float(jf.get("time_scale", 5.0))
        self._route_total = 1.0        # 출발 위치부터 경로 끝까지 거리 (m)
        self._prev_odo = 0.0           # 직전 스텝 주행거리 (m)
        self.randomize_signal = bool(mdp_sim.get("randomize_signal_phase", True))
        # ---- 워밍업 스냅샷 풀 (OSM 도로망 전용, 학습 속도용) ----
        # 워밍업(배경 교통 150~300초 미리 흘리기)이 reset 시간의 대부분이라,
        # 워밍업이 끝난 시뮬레이션 상태를 파일로 저장해 두고 다시 불러 쓴다.
        # refresh 확률만큼은 새로 워밍업해서 풀을 계속 바꿔 다양성을 유지한다.
        self.state_pool_size = int(mdp_sim.get("warmup_state_pool", 0))
        self.state_pool_refresh = float(mdp_sim.get("warmup_state_refresh", 0.25))
        self._state_pools = {}         # 신호 계획별 스냅샷 풀
        self._state_dir = None
        self.signal_program = None     # 이번 에피소드의 신호 계획 이름
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.lane_change_duration = float(action.get("lane_change_duration", 1.0))
        self.rw = reward

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 raw ∈ [-1,1], 환경 내부에서 {-1,0,+1}로 양자화
        # PPO의 Gaussian 정책과 호환하기 위해 Box를 유지한다. 실제 적용된
        # 차선변경 명령은 info["lane_change"]에 정확한 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 관측: 31차원 (+ 신호 3차원 = 34), 범위 [-1,1].
        #   레이아웃은 env/mdp_config.py 주석 참고.
        #   시야 차선: 좌2, 좌1, 현재, 우1, 우2 (5개) — 3차선 도로에서
        #   맨 왼쪽에 있어도 오프셋 -2로 맨 오른쪽 차선까지 보인다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        self.obs_dim = 1 + n_lanes_view * 4 + n_lanes_view + n_lanes_view
        if self.tl_obs:
            # 진입 도로별 신호 × tl_columns + 정지선 거리 (+ 남은 시간, 선택)
            self.obs_dim += self.tl_columns + 1 + (1 if self.tl_remaining else 0)
        if self.goal_obs:
            self.obs_dim += 1          # 목표(경로 끝)까지 남은 거리
        if self.foe_obs:
            self.obs_dim += 2          # 상충 지점까지 거리, 도착 시간차
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    # ==================================================================
    # SUMO 프로세스 시작
    # ==================================================================
    def _start_sumo(self):
        # gui=True면 화면이 있는 sumo-gui, 아니면 화면 없는 sumo (학습용, 훨씬 빠름)
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            # 충돌한 차량을 시뮬레이션에서 제거. (기본값 teleport는 순간이동시켜버려서
            # RL 학습에는 부적합. remove로 해야 "충돌 → 에피소드 종료"가 깔끔함)
            "--collision.action", "remove",
            # 교차로 "내부"의 충돌도 검사한다. SUMO 기본값은 차선 위 충돌만
            # 잡기 때문에, 이 옵션이 없으면 ego가 신호를 무시하고 교차로에서
            # 남북 차량과 겹쳐도 충돌로 집계되지 않는다.
            "--collision.check-junctions", "true",
            "--no-warnings", "true",   # 콘솔 경고 끄기
            "--no-step-log", "true",   # 스텝 로그 끄기
            # 매 에피소드 다른 시드 → 배경 교통 패턴이 달라져서 과적합 방지
            "--seed", str(np.random.randint(0, 100000)),
        ]
        if self.gui:
            cmd += ["--quit-on-end"]      # 에피소드 종료 시 창 정리
            cmd += ["--delay", str(self.gui_delay)]
            if self.gui_autostart:
                cmd += ["--start"]        # 자동 재생
            # --start가 없으면: 창이 뜬 뒤 ▶(플레이)를 눌러야 진행된다.
            #   매 에피소드가 reset()에서 SUMO를 새로 띄우므로,
            #   에피소드마다 ▶를 눌러 시작하게 된다.
        traci.start(cmd)   # SUMO를 자식 프로세스로 띄우고 소켓 연결
        self._started = True

    # ==================================================================
    # 에피소드 초기화
    # ==================================================================
    def _prefill_traffic(self):
        """도로 전 구간에 배경차를 일정 간격(±jitter)으로 미리 배치한다.

        flow는 도로 입구에서만 차를 만들기 때문에, 프리필이 없으면
        에피소드 초반의 긴 도로가 텅 비어 "빈 도로 주행"만 잔뜩 학습된다.
        프리필은 첫 스텝부터 ego가 실제 교통류 한가운데서 시작하게 만든다.

        - 위치: start_offset ~ (길이 - end_margin), 차선별 독립 배치
        - 간격: prefill_spacing × (1 ± jitter) 무작위
        - 타입: controllers 확률 비율로 샘플링 (rou.xml의 vType과 동일)
        - 속도: departSpeed="max" → 각 개체가 "자기 허용 최고속도"로 시작.
          컨트롤러별 max_speed/speedDev 차등이 프리필에도 그대로 반영되고,
          개체별 무작위 speedFactor 때문에 숫자 속도를 주면 SUMO가
          "허용속도 초과"로 거부할 수 있는 문제도 함께 피한다.
        배치된 차들은 SUMO의 각자 컨트롤러(IDM 등)가 정상적으로 운전한다.
        """
        rng = np.random.default_rng()
        count = 0
        # 차선별 간격 배율 (없으면 전 차선 동일). 인덱스 0 = 맨 오른쪽.
        factors = (self.traffic.get("prefill", {})
                   .get("lane_spacing_factor")
                   or [1.0] * self.num_lanes)
        for lane in range(self.num_lanes):
            f = float(factors[lane]) if lane < len(factors) else 1.0
            lane_spacing = self.prefill_spacing * f
            pos = self.prefill_start + rng.uniform(0, lane_spacing)
            while pos < self.road_length - self.prefill_end_margin:
                vid = f"pre_{lane}_{count}"
                vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                try:
                    traci.vehicle.add(
                        vid, "r0", typeID=vtype,
                        departLane=str(lane),
                        departPos=f"{pos:.1f}",
                        departSpeed="max")
                    count += 1
                except traci.TraCIException:
                    pass  # 겹침 등으로 실패한 자리는 건너뜀
                step = lane_spacing * rng.uniform(
                    1.0 - self.prefill_jitter, 1.0 + self.prefill_jitter)
                pos += max(step, 10.0)
        return count

    def _prefill_radial(self, ego_edge):
        """방사형 교차로: 각 진입 도로에 배경차를 미리 배치한다.

        경로는 그 도로에서 출발하는 flow 비율대로 고르고, 간격은 도로별
        교통량(inflow)에서 계산한다. ego가 출발하는 도로는 시작 구간
        start_offset을 비워둔다.
        """
        rng = np.random.default_rng()
        groups = {}
        for rid, info in self.meta["bg_routes"].items():
            groups.setdefault(info["first_edge"], []).append((rid, info["rate"]))
        count = 0
        for edge, items in groups.items():
            e = self.meta["in_edges"][edge]
            rids = [r for r, _ in items]
            p = np.array([w for _, w in items], dtype=float)
            p = p / p.sum()
            lanes, length = int(e["lanes"]), float(e["length"])
            if self.prefill_spacing_fixed:
                spacing = self.prefill_spacing_fixed
            else:
                q_lane = max(float(e["inflow"]) / 3600.0 / max(lanes, 1), 1e-6)
                spacing = self.vmax / q_lane
            start = self.prefill_start if edge == ego_edge else 5.0
            for lane in range(lanes):
                pos = start + rng.uniform(0, spacing)
                while pos < length - self.prefill_end_margin:
                    vid = f"pre_{edge}_{lane}_{count}"
                    try:
                        traci.vehicle.add(
                            vid, str(rng.choice(rids, p=p)),
                            typeID=str(rng.choice(self._prefill_types,
                                                  p=self._prefill_probs)),
                            departLane=str(lane), departPos=f"{pos:.1f}",
                            departSpeed="max")
                        count += 1
                    except traci.TraCIException:
                        pass
                    step = spacing * rng.uniform(1.0 - self.prefill_jitter,
                                                 1.0 + self.prefill_jitter)
                    pos += max(step, 10.0)
        return count

    def _randomize_signal_phase(self):
        """모든 신호를 (주기 길이에 비례한 확률로) 임의의 위상·잔여시간에서 시작."""
        rng = np.random.default_rng()
        for tls in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls)
            prog = traci.trafficlight.getProgram(tls)
            logic = next((l for l in logic if l.programID == prog), logic[0])
            durs = np.array([p.duration for p in logic.phases], dtype=float)
            t = rng.uniform(0.0, durs.sum())          # 주기 안의 임의 시점
            idx = int(np.searchsorted(np.cumsum(durs), t, side="right"))
            idx = min(idx, len(durs) - 1)
            remaining = np.cumsum(durs)[idx] - t
            traci.trafficlight.setPhase(tls, idx)
            traci.trafficlight.setPhaseDuration(tls, max(float(remaining), 0.5))

    def _next_signal(self):
        """ego 경로상 다음 신호의 (tls_id, link_index, 거리 m, 상태 문자) 또는 None."""
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            return None
        return nxt[0] if nxt else None

    @staticmethod
    def _signal_code(ch):
        """신호 문자 → 관측값: 녹색 +1 / 황색 0 / 적색 -1."""
        return 1.0 if ch in "GgOo" else (0.0 if ch in "yYu" else -1.0)

    def _apply_signal_plan(self):
        """SIGNAL["active_plan"]에 따라 이번 에피소드의 신호 계획(SUMO 프로그램)을 고른다.

        "이름"        : 그 계획만 사용
        ["a", "b"]   : 에피소드마다 무작위
        "tod"        : SIGNAL["tod"] 시간대표에서 각 계획이 하루 중 차지하는
                       시간 비율대로 무작위 (하루 전체 신호 운영을 고르게 학습)
        """
        sp = (self.meta or {}).get("signal_plans")
        if not sp:
            self.signal_program = None
            return
        active, progs = sp.get("active"), sp["programs"]
        if active == "tod" and sp.get("tod"):
            tod = sorted(sp["tod"], key=lambda t: t["start"])
            mins = [int(t["start"][:2]) * 60 + int(t["start"][3:5]) for t in tod]
            dur = [((mins[(i + 1) % len(mins)] - m) % 1440) or 1440
                   for i, m in enumerate(mins)]
            w = {}
            for t, d in zip(tod, dur):
                w[t["plan"]] = w.get(t["plan"], 0) + d
            names = list(w)
            p = np.array([w[n] for n in names], dtype=float)
            name = names[np.random.choice(len(names), p=p / p.sum())]
        elif isinstance(active, (list, tuple)):
            name = active[np.random.randint(len(active))]
        else:
            name = active or progs[0]
        if name not in progs:
            raise ValueError(f"신호 계획 '{name}'이 없습니다 (가능: {progs})")
        traci.trafficlight.setProgram(sp["tls"], name)
        self.signal_program = name

    def _warmup_osm(self):
        """배경 교통을 미리 흘려 도로망을 채운다 (스냅샷 풀이 있으면 재사용)."""
        pool = self._state_pools.setdefault(self.signal_program, [])
        if self.state_pool_size > 0 and pool and (
                len(pool) >= self.state_pool_size
                and np.random.rand() >= self.state_pool_refresh):
            traci.simulation.loadState(pool[np.random.randint(len(pool))])
            return
        lo, hi = self.warmup_seconds
        traci.simulationStep(float(np.random.uniform(lo, hi)))
        if self.state_pool_size > 0:
            import tempfile
            if self._state_dir is None:
                self._state_dir = tempfile.mkdtemp(prefix="hexacross_state_")
            if len(pool) < self.state_pool_size:
                path = os.path.join(self._state_dir,
                                    f"{self.signal_program or 'default'}_{len(pool)}.xml.gz")
                pool.append(path)
            else:
                path = pool[np.random.randint(len(pool))]
            traci.simulation.saveState(path)

    def _junction_foe(self):
        """가장 위험한 교차로 상충 차량: (내 상충지점까지 거리 m, 도착 시간차 s) 또는 None.

        SUMO getJunctionFoes: 앞으로 지날 교차로 안에서 내 진행 경로와 엇갈리거나
        합쳐지는 경로를 가진 차량들과, 각자 상충 지점까지의 거리.
        도착 시간차 = 상대 도착 시각 − 내 도착 시각 (작을수록 같은 순간에 만남).
        상대가 이미 상충 구간 안에 있으면 상대 도착 시각 = 0.
        여러 대면 |시간차|가 가장 작은 차를 고른다.
        """
        try:
            foes = traci.vehicle.getJunctionFoes(self.ego_id, self.foe_range)
        except traci.TraCIException:
            return None
        if not foes:
            return None
        v = max(traci.vehicle.getSpeed(self.ego_id), 1.0)
        best = None
        for f in foes:
            foe_id, ego_d, foe_d, _ego_exit, foe_exit = f[0], f[1], f[2], f[3], f[4]
            if ego_d < 0 or foe_exit <= 0:      # 내가 이미 지났거나 상대가 빠져나감
                continue
            try:
                fv = max(traci.vehicle.getSpeed(foe_id), 1.0)
            except traci.TraCIException:
                continue
            dt = max(foe_d, 0.0) / fv - ego_d / v
            if best is None or abs(dt) < abs(best[1]):
                best = (float(ego_d), float(dt))
        return best

    def _ego_lane_ok(self, edge, li):
        """edge의 li번 차로가 있고 ego(승용차)가 달릴 수 있는가.
        중앙대로 BRT 버스전용차로처럼 승용차가 못 쓰는 차로는 "없는 차로"로 본다."""
        if li < 0 or li >= traci.edge.getLaneNumber(edge):
            return False
        lid = f"{edge}_{li}"
        ok = self._lane_ok.get(lid)
        if ok is None:
            allowed = traci.lane.getAllowed(lid)
            ok = ("passenger" in allowed) if allowed else ("passenger" not in traci.lane.getDisallowed(lid))
            self._lane_ok[lid] = ok
        return ok

    def _approach_groups(self, tls):
        """신호 tls가 제어하는 교차로의 "진입 도로" 묶음.

        반환: (진입 도로 목록 [(방위각, {linkIndex...}), ...] 방위각 순,
               교차로 내부 도로의 linkIndex 집합)
        교차로가 SUMO 노드 여러 개로 되어 있으면(연산교차로) 노드 사이의
        짧은 내부 도로에도 신호 링크가 있는데, 이건 진입 도로가 아니므로 뺀다.
        방위각은 교차로 중심 → 그 도로가 들어오는 쪽 (북 0°, 시계방향).
        """
        if tls in self._tls_groups:
            return self._tls_groups[tls]
        links = traci.trafficlight.getControlledLinks(tls)
        in_lanes = {i: lk[0][0] for i, lk in enumerate(links) if lk}
        juncs = {traci.edge.getToJunction(traci.lane.getEdgeID(l))
                 for l in in_lanes.values()}
        pts = [traci.junction.getPosition(j) for j in juncs]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        groups, inner = [], set()
        for i, lane in in_lanes.items():
            edge = traci.lane.getEdgeID(lane)
            if not self._ego_lane_ok(edge, int(lane.rsplit("_", 1)[1])):
                continue                           # 버스전용차로 신호는 승용차와 무관
            if traci.edge.getFromJunction(edge) in juncs:
                inner.add(i)                       # 교차로 내부 도로
                continue
            x, y = traci.lane.getShape(lane)[0]
            b = float(np.degrees(np.arctan2(x - cx, y - cy)) % 360.0)
            g = next((g for g in groups
                      if abs((g[0] - b + 180.0) % 360.0 - 180.0) < 25.0), None)
            if g is None:
                groups.append((b, {i}))
            else:
                g[1].add(i)
        groups.sort(key=lambda g: g[0])
        self._tls_groups[tls] = (groups, inner)
        return self._tls_groups[tls]

    def _next_approach_signal(self):
        """경로상 다음 "교차로 진입" 신호 (tls, link, 거리, 상태) 또는 None.
        교차로 내부 도로의 신호 링크(항상 녹색)는 건너뛴다."""
        try:
            nxt = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            return None
        for tls, link, dist, st in nxt:
            groups, inner = self._approach_groups(tls)
            if link not in inner:
                return tls, link, dist, st
        return None

    def _signal_obs(self):
        """다음 신호 교차로의 진입 도로별 신호 (레이아웃은 mdp_config 주석 참고).

        [0]            내 진입 도로(내 차로가 받는 신호)
        [1..C-1]       나머지 진입 도로, 내 도로에서 시계방향 순서
                       (교차로 도로가 C개보다 적으면 남는 칸 = -1:
                        없는 도로에서는 차가 오지 않으므로 "항상 적색"과 같다)
        [C]            정지선까지 거리 / range
        [C+1]          (remaining_time일 때만) 내 신호 색이 바뀔 때까지 남은 시간
        """
        C = self.tl_columns
        nxt = self._next_approach_signal()
        if nxt is None or nxt[2] > self.tl_range:
            return [1.0] * C + [1.0] + ([1.0] if self.tl_remaining else [])
        tls, link, dist, st = nxt
        groups, _ = self._approach_groups(tls)
        state = traci.trafficlight.getRedYellowGreenState(tls)
        mine = next((k for k, g in enumerate(groups) if link in g[1]), 0)
        cols = [self._signal_code(st)]
        for k in range(1, len(groups)):
            idx = groups[(mine + k) % len(groups)][1]
            cols.append(max(self._signal_code(state[i]) for i in idx))
        cols = (cols + [-1.0] * C)[:C]
        out = cols + [float(np.clip(dist / self.tl_range, 0.0, 1.0))]
        if self.tl_remaining:
            code = self._signal_code(st)
            out.append(float(np.clip(
                self._time_to_color_change(tls, link, code) / self.tl_max_time,
                0.0, 1.0)))
        return out

    def _time_to_color_change(self, tls, link, code):
        """ego 차선 신호의 "색"이 바뀔 때까지 남은 시간(s).

        신호 위상(phase)이 바뀌어도 ego 차선 색이 그대로일 수 있다
        (예: 동서 적색 동안 전적색 → 남북 녹색 → 남북 황색 → 전적색).
        그래서 현재 위상의 남은 시간에 "같은 색이 이어지는" 다음 위상들의
        길이를 더한다.
        """
        def color(ch):
            return 1.0 if ch in "Gg" else (0.0 if ch in "yY" else -1.0)
        remaining = (traci.trafficlight.getNextSwitch(tls)
                     - traci.simulation.getTime())
        try:
            prog = traci.trafficlight.getProgram(tls)
            logic = next(l for l in traci.trafficlight.getAllProgramLogics(tls)
                         if l.programID == prog)
            phases = logic.phases
            idx = traci.trafficlight.getPhase(tls)
            for k in range(1, len(phases)):
                p = phases[(idx + k) % len(phases)]
                if color(p.state[link]) != code:
                    break
                remaining += p.duration
        except (StopIteration, traci.TraCIException, IndexError):
            pass
        return remaining

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()
        self._ego_dist_ref = None      # 데이터셋 state의 교차로까지 거리 기준점

        # 실제 신호 현시표(road_config.SIGNAL["plans"])가 있으면 이번 에피소드 계획 선택
        self._apply_signal_plan()
        # 신호를 주기 중 임의의 시점에서 시작 (도착 타이밍 암기 방지)
        if self.randomize_signal:
            self._randomize_signal_phase()

        # ego 경로: 방사형 교차로면 EGO["routes"] 중 하나를 무작위로, 아니면 r0
        routes = []
        if self.meta:
            key = "ego_demo_routes" if self.route_set == "demo" else "ego_routes"
            routes = self.meta.get(key) or []
            if self.route_set == "demo" and not routes:
                raise RuntimeError("시연용 경로가 없습니다. road_config의 EGO['demo_routes']와 "
                                   "ROAD['osm']['enabled']를 확인하세요.")
        ego_route = str(np.random.choice(routes)) if routes else "r0"
        ego_edge = traci.route.getEdges(ego_route)[0]

        osm_layout = bool(self.meta) and self.meta.get("layout") == "osm"
        if osm_layout:
            # OSM 주변 도로망: 배경 교통을 몇 분 먼저 흘려서 도로망 전체를 채운다.
            # simulationStep(t)는 시각 t까지 한 번에 진행하므로 빠르다.
            self._warmup_osm()
        elif self.prefill_enabled:
            # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
            if self.meta:
                self._prefill_radial(ego_edge)
            else:
                self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        for _ in range(self.warmup_steps):
            traci.simulationStep()
        # 삽입에 실패해 대기 중인 프리필 차량은 지운다.
        # (예: 적색 신호 직전 위치에 최고속도로 배치된 차는 정지할 수 없어서
        #  SUMO가 삽입을 거부한다. SUMO는 한 edge에서 삽입이 막히면 그 뒤
        #  차량도 기다리게 하므로, 남겨두면 ego 삽입까지 막힌다.
        #  또 나중에 도로 한가운데서 갑자기 나타나는 것도 막는다.)
        for vid in traci.simulation.getPendingVehicles():
            if vid.startswith("pre_"):
                try:
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass

        # ego 차량 투입 — 반드시 출발 엣지의 시작점(0m)에서 출발
        depart_lane = self.ego_depart_lane
        if depart_lane == "center":
            depart_lane = str(traci.edge.getLaneNumber(ego_edge) // 2)
        self.wrong_lane_steps = 0
        # 출발 위치: OSM 도로망은 교차점에서 일정 거리 앞 (meta에 계산돼 있음)
        depart_pos = float((self.meta or {}).get("ego_depart_pos", {}).get(ego_route, 0.0))
        traci.vehicle.add(self.ego_id, ego_route, typeID="ego_type",
                          departLane=depart_lane,
                          departPos=f"{depart_pos:.1f}",
                          departSpeed=self.ego_depart_speed)

        # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 30스텝) 기다린다. 지연되어도 요청한 차선은 유지된다.
        _wait = 0
        while self.ego_id not in traci.vehicle.getIDList() and _wait < 30:
            traci.simulationStep()
            _wait += 1
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 30스텝 이상 막혀 있습니다. "
                "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요.")

        # GUI 기본 시점: ego를 화면 중앙에서 추적하고 충분히 가까이 확대한다.
        # 긴 직선도로 전체를 한 화면에 맞추는 기본 뷰는 차량이 점/삼각형처럼
        # 보이기 쉬우므로 학습/평가 관찰용으로는 이 설정이 더 직관적이다.
        if self.gui and self.ego_id in traci.vehicle.getIDList():
            try:
                view_ids = traci.gui.getIDList()
                if view_ids:
                    view_id = view_ids[0]
                    # tracking을 먼저 켜고 zoom을 마지막에 적용해야 SUMO가
                    # 차량 추적을 시작하면서 viewport를 다시 덮어쓰지 않는다.
                    if self.gui_track_ego:
                        traci.gui.trackVehicle(view_id, self.ego_id)
                    traci.gui.setZoom(view_id, self.gui_zoom)
            except traci.TraCIException:
                pass

        self.step_count = 0
        # 목표 = 경로 끝 도로(예: 중앙대로 동래 방면). 출발 위치부터 남은 경로 길이
        route = traci.vehicle.getRoute(self.ego_id)
        self._route_total = max(
            sum(traci.lane.getLength(f"{e}_0") for e in route)
            - traci.vehicle.getLanePosition(self.ego_id), 1.0)
        self._prev_odo = traci.vehicle.getDistance(self.ego_id)
        return self._get_obs(), {}

    def _goal_remaining(self) -> float:
        """목표(경로 끝)까지 남은 거리 (m). 주행거리(odometer) 기준."""
        return float(np.clip(self._route_total
                             - traci.vehicle.getDistance(self.ego_id),
                             0.0, self._route_total))

    # ==================================================================
    # 관측 만들기 (레이아웃은 env/mdp_config.py 주석 참고)
    #
    # 핵심 설계: 특정 도로에 대한 하드코딩 없이, 매 스텝 SUMO 네트워크를
    # 실시간 조회해서 "지금 내 주변"을 계산한다. 그래서 도로 구조가
    # 어떻게 바뀌어도(차선 수, 합류, 차선 감소 등) 이 코드는 그대로 동작한다.
    #
    # 사용하는 TraCI API:
    #   getLeader / getFollower : 같은 차선의 앞/뒤 차량 (edge 경계도 넘어 탐색)
    #   getNeighbors(mode)      : 옆 차선의 앞/뒤 차량.
    #       mode 비트: bit0 = 0:왼쪽 / 1:오른쪽,  bit1 = 0:후행 / 1:선행
    #   lane.getLinks           : 차선이 다음 edge로 이어지는지 (연결성 판단)
    # ==================================================================

    # (차선 오프셋, 선행차 getNeighbors mode, 후행차 mode)
    #   오프셋 +1 = 왼쪽 차선 (SUMO는 차선 index가 왼쪽으로 갈수록 커짐)
    _NEIGHBOR_MODES = {+1: (0b10, 0b00),   # 왼쪽: 선행=2, 후행=0
                       -1: (0b11, 0b01)}   # 오른쪽: 선행=3, 후행=1

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = traci.vehicle.getLeader(self.ego_id, self.visibility)
                if res is None:
                    return None
                vid, gap = res
            else:
                vid, gap = traci.vehicle.getFollower(self.ego_id, self.visibility)
                if vid == "":
                    return None
        elif abs(offset) >= 2:
            # 두 칸 이상 옆 차선: SUMO getNeighbors는 "바로 옆"만 지원하므로
            # 차선 직접 스캔(_scan_lane_neighbors — 같은 edge의 차량 목록을
            # 훑어 가장 가까운 앞/뒤를 찾음)으로 감지한다.
            front, rear = self._scan_lane_neighbors(offset)
            res = front if leader else rear
            if res is None or res[0] < 0 or res[0] > self.visibility:
                return None
            return float(res[0]), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if gap < 0 or gap > self.visibility:
            return None
        return float(gap), traci.vehicle.getSpeed(vid)

    def _lane_connectivity(self, offset: int) -> float:
        """오프셋 차선의 '경로상 연결성'을 반환.

        -1 : 그 차선이 존재하지 않음
         1 : lane_plan_range 이상 "내 경로를 따라" 계속 갈 수 있음
        0~1: (그 차선으로 경로를 따라 갈 수 있는 거리 / lane_plan_range)
             → 값이 작을수록 곧 차선을 바꿔야 한다는 경고

        SUMO의 getBestLanes는 차선마다 "차선변경 없이 경로를 따라 갈 수 있는
        거리"를 알려준다. 그래서 차선 감소/합류뿐 아니라 교차로에서
        "이 차선은 내 진행 방향(좌회전·직진 등)으로 이어지지 않는다"도
        같은 신호로 잡힌다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로(junction) 내부 통과 중
                return 1.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if not self._ego_lane_ok(edge, lane_index):
                return -1.0                   # 그런 차선이 없음 (버스전용차로 포함)
            remaining = self._route_remaining(f"{edge}_{lane_index}")
            if remaining is None:
                return 1.0
            return float(np.clip(remaining / self.lane_plan_range, 0.0, 1.0))
        except traci.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

    def _route_remaining(self, lane_id):
        """lane_id 차선으로 차선변경 없이 경로를 따라 더 갈 수 있는 거리(m)."""
        pos = traci.vehicle.getLanePosition(self.ego_id)
        for lid, length, *_ in traci.vehicle.getBestLanes(self.ego_id):
            if lid == lane_id:
                return max(float(length) - pos, 0.0)
        return None

    def _lane_density(self, offset: int) -> float:
        """오프셋 차선의 '전방 차량 밀도'를 0~1로 반환.

        밀도 = (시야 W 이내 전방 차량 수) / (그 구간의 최대 수용 대수)
               최대 수용 대수 = 사용가능거리 / density_veh_length

        인코딩 관례 (막힘 정도로 읽히도록):
            1.0 : 차선이 없거나 사실상 진입 불가  ← "꽉 찬 것"처럼 취급해서
                  에이전트가 자연스럽게 회피하게 만든다
            0.0 : 텅 빈 차선
        일반화: 차선이 시야보다 짧으면 link를 따라 다음 차선까지 이어서 센다.
        곧 끊기는 차선(link 없음)은 남은 거리만큼만 세므로 분모가 줄어
        같은 대수여도 밀도가 높게 나온다 → "끊기는 차선에 차가 몰려있음" 신호.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로 내부: 중립값
                return 0.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return 1.0                    # 차선 없음 = 꽉 참 취급
            lane_id = f"{edge}_{lane_index}"
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            remaining = traci.lane.getLength(lane_id) - ego_pos
            links = traci.lane.getLinks(lane_id)

            # 이 차선에서 앞으로 실제로 쓸 수 있는 거리
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0

            count = 0
            # (a) 현재 차선 구간의 전방 차량
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id:
                    continue
                d = traci.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            # (b) 시야가 edge 끝을 넘고 다음 차선으로 이어지면 이어서 센다
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]    # 첫 번째 연결 차선
                for vid in traci.lane.getLastStepVehicleIDs(next_lane_id):
                    if traci.vehicle.getLanePosition(vid) <= \
                            self.visibility - remaining:
                        count += 1

            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except traci.TraCIException:
            return 0.0

    def get_detected_vehicles(self):
        """현재 관측 시야에 들어온(=감지된) 배경차들의 좌표 목록을 반환.

        데이터셋 수집용 부가 정보 — 관측 벡터에는 들어가지 않는다.
        "감지"의 정의는 관측과 동일: 관측 대상 차선(obs_offsets 범위)에서
        ego 기준 종방향 ±visibility(W) 이내에 있는 차량.

        Returns: list[dict]  각 원소 =
            {"x": 절대 x, "y": 절대 y,
             "rel_pos": ego 기준 종방향 상대거리(m, 앞 +), "speed": m/s,
             "lane_offset": ego 차선 기준 오프셋(+왼쪽)}
        ego가 없으면 빈 리스트. 문자열(차량 id)은 넣지 않는다 (DATA_FORMAT v1).
        """
        out = []
        if self.ego_id not in traci.vehicle.getIDList():
            return out
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return out
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            for offset in self.obs_offsets:
                li = ego_lane + offset
                if li < 0 or li >= n_lanes:
                    continue
                for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                    if vid == self.ego_id:
                        continue
                    rel = traci.vehicle.getLanePosition(vid) - ego_pos
                    if abs(rel) > self.visibility:
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    out.append({"x": float(x), "y": float(y),
                                "rel_pos": float(rel),
                                "speed": float(traci.vehicle.getSpeed(vid)),
                                "lane_offset": int(offset)})
        except traci.TraCIException:
            pass
        return out

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim, dtype=np.float32)

        v = traci.vehicle.getSpeed(self.ego_id)
        obs = [v / self.vmax]                      # [0] 내 절대속도

        # ----- 5개 차선(좌2→좌1→현재→우1→우2)의 선행/후행 상대거리·상대속도 -----
        for offset in self.obs_offsets:
            lead = self._neighbor(offset, leader=True)
            if lead is None:
                obs += [1.0, 1.0]                  # 없음: 멀고 + 위협 없음
            else:
                gap, lv = lead
                obs += [gap / self.visibility,
                        float(np.clip((lv - v) / self.vmax, -1.0, 1.0))]

            foll = self._neighbor(offset, leader=False)
            if foll is None:
                obs += [-1.0, -1.0]                # 없음: 멀고 + 위협 없음
            else:
                gap, fv = foll
                obs += [-gap / self.visibility,    # 후행은 음수 부호
                        float(np.clip((fv - v) / self.vmax, -1.0, 1.0))]

        # ----- 5개 차선의 연결성 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_connectivity(offset))

        # ----- 5개 차선의 전방 차량 밀도 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_density(offset))

        # ----- 다음 신호 (상태, 정지선 거리, 남은 시간) -----
        if self.tl_obs:
            obs += self._signal_obs()
        if self.goal_obs:
            obs.append(self._goal_remaining() / self._route_total)
        if self.foe_obs:
            foe = self._junction_foe()
            if foe is None:
                obs += [1.0, 1.0]                  # 상충 차량 없음 = 멀고 + 시간 여유
            else:
                obs += [min(foe[0] / self.foe_range, 1.0),
                        float(np.clip(foe[1] / self.foe_time, -1.0, 1.0))]

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ==================================================================
    # 1 스텝 진행: 행동 적용 → 시뮬레이션 1스텝 → 보상/종료 판정
    # ==================================================================
    def _quantize_lane_change(self, raw: float) -> int:
        """연속 policy 출력을 실제 차선 명령 {-1,0,+1}로 변환한다.

        +1 = 왼쪽, 0 = 유지, -1 = 오른쪽. SUMO 차선 index가 왼쪽으로
        갈수록 증가하는 규약과 동일해서 해석이 직관적이다.
        """
        raw = float(np.clip(raw, -1.0, 1.0))
        if raw >= self.lane_change_threshold:
            return 1
        if raw <= -self.lane_change_threshold:
            return -1
        return 0

    def _scan_lane_neighbors(self, direction: int):
        """목표 차선을 직접 스캔해 (가장 가까운 앞차 gap·속도, 뒷차 gap·속도) 반환.

        getNeighbors의 백업. getNeighbors는 SUMO 차선변경모델의 이웃 캐시를
        읽는데, "방금 스폰된 차량"은 다음 스텝까지 이 캐시에 없어서
        도로 입구 근처에서 사각지대가 생긴다. 차선의 차량 목록을 직접
        훑으면 이미 삽입된 차량은 전부 잡힌다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + direction
            # 존재하지 않는 차선이면 조회하지 않는다.
            # (없는 lane id를 traci.lane.*로 조회하면 SUMO가 매번
            #  에러 로그를 찍으므로, 범위 검사를 먼저 해서 조용히 건너뛴다)
            if edge.startswith(":") or not self._ego_lane_ok(edge, li):
                return None, None
            lane_id = f"{edge}_{li}"
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            front, rear = None, None
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                d = traci.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - traci.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, traci.vehicle.getSpeed(vid))
                else:
                    g = -d - traci.vehicle.getLength(self.ego_id)
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            return front, rear
        except traci.TraCIException:
            return None, None

    def _lane_change_is_safe(self, direction: int) -> bool:
        """목표 차선으로 지금 이동해도 측면충돌이 나지 않는지 검사한다.

        3중 안전 규칙:
          (0) 입구 보호구간: ego가 도로 시작 spawn_protect(m) 이내면 변경 금지.
              flow가 입구(0m)에서 차를 새로 만드는데, 생성 "직전" 차량은
              어떤 API로도 미리 볼 수 없어 원천적으로 사각지대이기 때문.
          (1) 앞: gap > min_front_gap + front_tau × 내속도
              + 접근 여유: gap > (내속도 − 앞차속도)⁺ × 1s   (내가 더 빠를 때)
          (2) 뒤: gap > min_rear_gap + rear_tau × 뒤차속도
              + 접근 여유: gap > (뒤차속도 − 내속도)⁺ × 1s   (뒤가 더 빠를 때)
        이웃은 getNeighbors 결과와 차선 직접 스캔 결과 중 "더 가까운 쪽"을
        쓴다 (스폰 직후 차량 사각지대 보완).
        """
        # 입구 보호는 "경로의 첫 edge"(flow 스폰 지점)에서만 적용한다.
        # 교차로를 지난 다음 edge에서도 lane 위치가 0부터 다시 시작하므로
        # 이 조건이 없으면 교차로 직후 60m 동안 차선변경이 막힌다.
        my_pos = traci.vehicle.getLanePosition(self.ego_id)
        on_first_edge = (traci.vehicle.getRoadID(self.ego_id)
                         == traci.vehicle.getRoute(self.ego_id)[0])
        if on_first_edge and my_pos < self.lc_spawn_protect:
            return False

        v = traci.vehicle.getSpeed(self.ego_id)
        scan_front, scan_rear = self._scan_lane_neighbors(direction)

        def closer(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a if a[0] <= b[0] else b

        lead = closer(self._neighbor(direction, leader=True), scan_front)
        if lead is not None:
            gap, lv = lead
            if gap <= self.lc_front_min + self.lc_front_tau * v:
                return False
            if gap <= max(v - lv, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False

        foll = closer(self._neighbor(direction, leader=False), scan_rear)
        if foll is not None:
            gap, fv = foll
            if gap <= self.lc_rear_min + self.lc_rear_tau * fv:
                return False
            if gap <= max(fv - v, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False
        return True

    def _apply_lane_change(self, lane_change: int) -> bool:
        """가능하고 "안전한" 경우에만 차선변경을 적용하고 성공 여부를 반환.

        안전 게이트(_lane_change_is_safe)를 통과하지 못한 명령은 실행하지
        않고 False를 반환한다 → step()에서 "불가능한 명령"과 동일하게
        invalid_action_penalty가 부과된다. 즉 에이전트 입장에서
        "위험한 변경 = 실행도 안 되고 감점"이라, 측면충돌로 즉사하며
        배우는 대신 안전한 타이밍을 고르는 쪽으로 학습이 유도된다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return False
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            current = traci.vehicle.getLaneIndex(self.ego_id)
            target = current + lane_change
            if not self._ego_lane_ok(edge, target):
                return False                  # 없는 차로 / 버스전용차로
            # ★ 측면충돌 방지: 실제 이동 명령(0이 아닌 변경)에만 안전 검사
            if (lane_change != 0 and self.lc_safety_enabled
                    and not self._lane_change_is_safe(lane_change)):
                return False
            traci.vehicle.changeLane(
                self.ego_id, target, self.lane_change_duration)
            return True
        except traci.TraCIException:
            return False

    # ==================================================================
    # 데이터셋용 state (DATA_FORMAT v1). 정책 입력에는 쓰지 않는다.
    # ==================================================================
    # 연산교차로 진입부·신호등 번호 0~5 (DATA_FORMAT.md 2절 번호표와 같아야 한다)
    LEG_ORDER = ("SW", "N", "W", "S", "E", "NE")
    LEG_INSIDE = 12        # 교차로 안 (우회전 연결로 포함)
    TL_NONE = -1           # 앞에 신호 없음

    def _state_layout(self) -> dict:
        """연산교차로 기준 도로 번호와 신호 번호표를 도로 파일에서 한 번 만든다.

        edge_leg  : edge id → 0~5 연산교차로로 들어가는 도로, 6~11 나가는 도로
                    (진입부 번호 + 6), 12 교차로 안. 표에 없으면 그 밖(-1).
                    들어가는/나가는 도로는 교차로에 붙은 edge에서 같은 이름 도로를
                    따라 도로망 끝까지 이어 간다 (예: SW는 시청 인근 교차로 너머까지).
        lane_count: edge id → 승용차가 다닐 수 있는 차로 수 (버스 전용 차로 제외)
        tl_number : (신호 id, linkIndex) → ego_tl 번호
                    0~5 연산교차로, 6~ extra_signals 교차로(시청 인근)의 진입 그룹,
                    other_tl = 그 밖의 신호. 이름 순서는 tl_names (meta.json에 저장)
        """
        if self._layout is not None:
            return self._layout
        lay = {"edge_leg": {}, "lane_count": {}, "stop_edges": set(),
               "center_tls": None, "leg_link": {}, "tl_number": {},
               "tl_names": list(self.LEG_ORDER), "other_tl": len(self.LEG_ORDER)}
        self._layout = lay
        if not (self.meta and self.meta.get("layout") == "osm"):
            return lay          # 연산교차로 OSM 도로망이 아니면 번호표 없음 (-1)

        import math
        import sumolib
        net = sumolib.net.readNet(
            os.path.join(os.path.dirname(os.path.abspath(self.cfg_path)), "highway.net.xml"),
            withInternal=True)
        center_id = str(self.meta["center"])
        cx, cy = net.getNode(center_id).getCoord()
        bearings = {lg["id"]: float(lg["bearing"]) for lg in self.road_legs}
        order = [lg for lg in self.LEG_ORDER if lg in bearings]

        def bearing(px, py, ox=cx, oy=cy):
            return math.degrees(math.atan2(px - ox, py - oy)) % 360.0

        def adiff(a, b):
            return abs((a - b + 180.0) % 360.0 - 180.0)

        def leg_at(px, py):
            return min(order, key=lambda lg: adiff(bearing(px, py), bearings[lg]))

        def heading(e, at_end):
            s = e.getShape()
            (x0, y0), (x1, y1) = (s[-2], s[-1]) if at_end else (s[0], s[1])
            return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0

        def trace(e, outgoing, code, limit=5000.0):
            """같은 이름 도로를 따라 바깥으로 이어 가며 edge에 번호를 붙인다."""
            seen, total = set(), 0.0
            while e is not None and e.getID() not in seen and total < limit:
                seen.add(e.getID())
                lay["edge_leg"].setdefault(e.getID(), code)
                total += e.getLength()
                nxt = list(e.getOutgoing()) if outgoing else list(e.getIncoming())
                nxt = [n for n in nxt if n.getFunction() != "internal"
                       and n.allows("passenger") and n.getName() == e.getName()
                       and n.getID().lstrip("-") != e.getID().lstrip("-")]
                if not nxt:
                    break
                h = heading(e, at_end=outgoing)
                e = min(nxt, key=lambda n: adiff(heading(n, at_end=not outgoing), h))

        center = net.getNode(center_id)
        center_tls = self.meta.get("center_tls")
        lay["center_tls"] = center_tls
        for e in center.getIncoming():
            if e.getFunction() == "internal" or not e.allows("passenger"):
                continue
            leg = leg_at(*e.getShape()[0])
            lay["stop_edges"].add(e.getID())
            trace(e, outgoing=False, code=self.LEG_ORDER.index(leg))
            for ln in e.getLanes():
                for c in ln.getOutgoing():
                    if c.getTLSID() == center_tls:
                        lay["tl_number"][(center_tls, c.getTLLinkIndex())] = self.LEG_ORDER.index(leg)
                        lay["leg_link"].setdefault(leg, c.getTLLinkIndex())
        for e in center.getOutgoing():
            if e.getFunction() == "internal" or not e.allows("passenger"):
                continue
            far, total = e, e.getLength()     # 교차로에서 60m 이상 떨어진 점으로 방향 판정
            while total < 60.0:
                nxt = [n for n in far.getOutgoing() if n.getName() == far.getName()]
                if not nxt:
                    break
                far = nxt[0]
                total += far.getLength()
            trace(e, outgoing=True, code=6 + self.LEG_ORDER.index(leg_at(*far.getShape()[-1])))

        for e in net.getEdges(withInternal=True):
            eid = e.getID()
            lay["lane_count"][eid] = sum(1 for ln in e.getLanes() if ln.allows("passenger"))
            if eid.startswith(f":{center_id}_"):
                lay["edge_leg"][eid] = self.LEG_INSIDE          # 교차로 안
            elif "_link" in (e.getType() or "") and min(
                    math.hypot(px - cx, py - cy) for px, py in e.getShape()) < 150.0:
                lay["edge_leg"][eid] = self.LEG_INSIDE          # 우회전 연결로
        # 다른 교차로 안(내부 연결)은 들어온 도로의 번호를 이어받는다
        for e in net.getEdges(withInternal=False):
            code = lay["edge_leg"].get(e.getID())
            if code is None or code == self.LEG_INSIDE:
                continue
            for ln in e.getLanes():
                for c in ln.getOutgoing():
                    via = c.getViaLaneID()
                    if via:
                        lay["edge_leg"].setdefault(via.rsplit("_", 1)[0], code)

        # extra_signals 교차로(시청 인근)의 진입 그룹: osm_builder가 meta에 남긴 순서대로
        number = len(self.LEG_ORDER)
        for jid, groups in (self.meta.get("extra_tls") or {}).items():
            jx, jy = net.getNode(jid).getCoord()
            for e in net.getEdges(withInternal=False):
                for ln in e.getLanes():
                    for c in ln.getOutgoing():
                        if c.getTLSID() != jid:
                            continue
                        b = bearing(*e.getShape()[0], ox=jx, oy=jy)
                        k = min(range(len(groups)), key=lambda i: (
                            groups[i]["road"] != (e.getName() or ""),
                            adiff(b, groups[i]["bearing"])))
                        lay["tl_number"][(jid, c.getTLLinkIndex())] = number + k
            lay["tl_names"] += [f"{jid}:{g['road']}@{g['bearing']}" for g in groups]
            number += len(groups)
        lay["other_tl"] = number
        lay["tl_names"].append("other")
        return lay

    def _ego_dist_to_junction(self) -> float:
        """경로를 따라 연산교차로 입구(정지선)까지 남은 거리 (m). 지난 뒤에는 음수.

        에피소드 처음 부를 때 SUMO가 계산한 경로상 거리와 그때의 주행거리계를
        기준으로 잡고, 이후에는 주행거리계 증가분을 뺀다. 경로가 교차로를 두 번
        지나면 첫 번째 기준. 연산교차로 도로망이 아니거나 경로에 없으면 0.
        """
        if self._ego_dist_ref is None:
            self._ego_dist_ref = False
            stops = self._state_layout()["stop_edges"]
            route = traci.vehicle.getRoute(self.ego_id)
            stop = next((e for e in route[traci.vehicle.getRouteIndex(self.ego_id):]
                         if e in stops), None)
            if stop is not None:
                d = traci.vehicle.getDrivingDistance(
                    self.ego_id, stop, traci.lane.getLength(f"{stop}_0"))
                if d > -1e8:            # 도달 불가면 SUMO가 큰 음수(INVALID)를 준다
                    self._ego_dist_ref = (d, traci.vehicle.getDistance(self.ego_id))
        if not self._ego_dist_ref:
            return 0.0
        d0, odo0 = self._ego_dist_ref
        return float(d0 - (traci.vehicle.getDistance(self.ego_id) - odo0))

    def _signal_state(self) -> dict:
        """연산교차로 6개 신호등의 색·남은 시간과 ego 앞 신호 번호·거리 (숫자만)."""
        lay = self._state_layout()
        out = {}
        tls = lay["center_tls"]
        rgy, phases, idx, base = "", [], 0, 0.0
        if tls:
            rgy = traci.trafficlight.getRedYellowGreenState(tls)
            # 신호 프로그램(현시 목록)은 안 바뀌므로 한 번만 읽어 둔다
            prog = traci.trafficlight.getProgram(tls)
            key = ("phases", tls, prog)
            if key not in lay:
                logic = next((l for l in traci.trafficlight.getAllProgramLogics(tls)
                              if l.programID == prog), None)
                lay[key] = [(p.state, float(p.duration)) for p in logic.phases] if logic else []
            phases = lay[key]
            idx = traci.trafficlight.getPhase(tls)
            base = traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime()

        def color(ch):
            return 1.0 if ch in "Gg" else (0.5 if ch in "yY" else 0.0)

        for i, leg in enumerate(self.LEG_ORDER):
            link = lay["leg_link"].get(leg)
            if link is None or link >= len(rgy):
                out[f"tl_color_{i}"], out[f"tl_remaining_{i}"] = 0.0, -1.0
                continue
            now = color(rgy[link])
            remaining = base      # 지금 현시 남은 시간 + 같은 색이 이어지는 다음 현시들
            for k in range(1, len(phases)):
                st, dur = phases[(idx + k) % len(phases)]
                if color(st[link]) != now:
                    break
                remaining += dur
            out[f"tl_color_{i}"] = now
            out[f"tl_remaining_{i}"] = float(remaining)
        nxt = self._next_signal()
        if nxt is None:
            out["ego_tl"], out["ego_tl_dist"] = self.TL_NONE, -1.0
        else:
            tid, link, dist, _ = nxt
            out["ego_tl"] = int(lay["tl_number"].get((tid, link), lay["other_tl"]))
            out["ego_tl_dist"] = float(dist)
        return out

    def get_privileged_state(self) -> dict:
        """POMDP 데이터셋용 simulator-side state (DATA_FORMAT v1, 숫자만).

        agent가 실제로 받는 입력은 _get_obs()의 부분관측이고, 이 메서드는
        분석/지도신호용 privileged state만 수집한다. 정책 입력으로 쓰지 않는다.
        도로·신호 번호표는 _state_layout() 참고.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return {"ego_alive": False}
        try:
            lay = self._state_layout()
            edge_leg = lay["edge_leg"]
            x, y = traci.vehicle.getPosition(self.ego_id)
            road = traci.vehicle.getRoadID(self.ego_id)
            vehicles = []
            for vid in traci.vehicle.getIDList():
                try:
                    vx, vy = traci.vehicle.getPosition(vid)
                    vehicles.append({
                        "x": float(vx),
                        "y": float(vy),
                        "speed": float(traci.vehicle.getSpeed(vid)),
                        "lane": int(traci.vehicle.getLaneIndex(vid)),
                        "lane_pos": float(traci.vehicle.getLanePosition(vid)),
                        "leg": int(edge_leg.get(traci.vehicle.getRoadID(vid), -1)),
                        "type": 1 if traci.vehicle.getVehicleClass(vid) == "bus" else 0,
                    })
                except traci.TraCIException:
                    continue
            state = {
                "ego_alive": True,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_speed": float(traci.vehicle.getSpeed(self.ego_id)),
                "ego_lane": int(traci.vehicle.getLaneIndex(self.ego_id)),
                "ego_lane_pos": float(traci.vehicle.getLanePosition(self.ego_id)),
                "ego_leg": int(edge_leg.get(road, -1)),
                "ego_lane_count": int(lay["lane_count"].get(
                    road, traci.edge.getLaneNumber(road))),
                "ego_dist_to_junction": self._ego_dist_to_junction(),
            }
            state.update(self._signal_state())
            state["vehicles"] = vehicles
            return state
        except traci.TraCIException:
            return {"ego_alive": False}

    def step(self, action):
        # action = [가감속 raw, 차선변경 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        lane_change_applied = False
        lane_before = None
        signal_before = None   # 행동 직전 ego 앞의 신호 (위반 판정용)

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        if self.ego_id in traci.vehicle.getIDList():
            signal_before = self._next_signal()
            v = traci.vehicle.getSpeed(self.ego_id)
            lane_before = int(traci.vehicle.getLaneIndex(self.ego_id))

            accel_cmd = (accel_raw * self.max_accel
                         if accel_raw >= 0.0 else accel_raw * self.max_decel)
            dv = accel_cmd * self.step_length
            traci.vehicle.setSpeed(
                self.ego_id, float(np.clip(v + dv, 0.0, self.vmax)))

            lane_change_applied = self._apply_lane_change(lane_change)

        # ----- (2) 시뮬레이션 1스텝 -----
        traci.simulationStep()
        self.step_count += 1

        # ----- (3) 이벤트/관측 -----
        collided = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        # --collision.action remove라 충돌한 ego도 도착 목록(getArrivedIDList)에
        # 들어간다. 충돌한 스텝은 도착으로 치지 않는다.
        arrived = (self.ego_id in traci.simulation.getArrivedIDList()) and not collided
        alive = self.ego_id in traci.vehicle.getIDList()
        lane_after = int(traci.vehicle.getLaneIndex(self.ego_id)) if alive else None

        # ----- (3-1) 신호 위반 판정 -----
        # 직전 스텝에 앞에 있던 신호가 이번 스텝에 사라졌다 = 정지선을 넘었다.
        # 넘기 전과 넘은 뒤 모두 적색이었을 때만 위반으로 본다
        # (스텝 도중 녹색으로 바뀐 경우는 봐준다).
        red_light = False
        signal_after = self._next_signal() if alive else None
        if alive and signal_before is not None:
            tls, link, _, st_before = signal_before
            passed = signal_after is None or signal_after[0] != tls
            if passed and st_before in "rR":
                st_now = traci.trafficlight.getRedYellowGreenState(tls)[link]
                red_light = st_now in "rR"

        obs = self._get_obs()
        terminated, truncated = False, False
        wrong_lane = False
        ego_speed, ego_gap = None, None

        # ----- (4) 기존 보상 유지 -----
        if collided:
            reward = -float(self.rw["collision_penalty"])
            terminated = True
        elif arrived:
            reward = float(self.rw["arrival_bonus"])
            terminated = True
        elif not alive:
            reward = 0.0
            terminated = True
        else:
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            same_lane_leader = self._neighbor(0, leader=True)
            if same_lane_leader is not None:
                ego_gap = same_lane_leader[0]
            reward = float(self.rw["speed_weight"]) * ego_speed / self.vmax

            # ---- TTC(충돌까지 남은 시간) 감점: 앞차와 이대로 가면 몇 초 뒤 부딪히는가 ----
            # TTC = 차간거리 / (내 속도 − 앞차 속도). ttc_threshold초보다 짧아질수록
            # 감점이 커진다 (최대 ttc_penalty). 충돌(큰 감점, 드묾)보다 훨씬 자주
            # 오는 신호라 "미리 감속"을 배우게 한다.
            tp = float(self.rw.get("ttc_penalty", 0.0))
            if tp > 0.0 and same_lane_leader is not None:
                closing = ego_speed - same_lane_leader[1]
                if closing > 0.1:
                    ttc = same_lane_leader[0] / closing
                    th = float(self.rw.get("ttc_threshold", 3.0))
                    if ttc < th:
                        reward -= tp * (1.0 - ttc / th)

            # ---- 신호 제동 감점: 내 신호가 녹색이 아닌데 정지선에 서기엔 너무 빠름 ----
            # 정지선에서 서려면 필요한 감속도 a = v² / (2·거리). 이게 comfort_decel
            # 보다 크면(=지금 브레이크를 밟아도 급제동) 초과분에 비례해 감점
            # (최대 signal_brake_penalty). 드문 신호위반 감점보다 먼저, 촘촘하게
            # "적색/황색이면 미리 감속"을 가르친다.
            sbp = float(self.rw.get("signal_brake_penalty", 0.0))
            if sbp > 0.0 and self.tl_obs and ego_speed > 0.5:
                nxt = self._next_approach_signal()
                if (nxt is not None and nxt[2] <= self.tl_range
                        and self._signal_code(nxt[3]) < 1.0):
                    a_req = ego_speed ** 2 / (2.0 * max(nxt[2] - 1.0, 0.5))
                    a_c = float(self.rw.get("comfort_decel", 3.0))
                    if a_req > a_c:
                        reward -= sbp * min((a_req - a_c) / a_c, 1.0)

            # ---- 교차로 상충 감점: 상충 차량과 거의 같은 순간에 상충 지점 도착 ----
            # 내가 상충 지점에 foe_conflict_dist(m) 안으로 다가왔는데 도착 시간차가
            # foe_conflict_threshold(s)보다 작으면, 작을수록 크게 감점 (최대 foe_conflict_penalty)
            fcp = float(self.rw.get("foe_conflict_penalty", 0.0))
            if fcp > 0.0 and self.foe_obs and ego_speed > 0.5:
                foe = self._junction_foe()
                th = float(self.rw.get("foe_conflict_threshold", 2.0))
                if (foe is not None and foe[0] < float(self.rw.get("foe_conflict_dist", 30.0))
                        and abs(foe[1]) < th):
                    reward -= fcp * (1.0 - abs(foe[1]) / th)

            # ---- 목표 진행 보상: 이번 스텝에 경로를 따라 목표 쪽으로 간 거리 ----
            # (vmax로 달렸을 때 스텝당 goal_progress_weight)
            odo = traci.vehicle.getDistance(self.ego_id)
            gw = float(self.rw.get("goal_progress_weight", 0.0))
            if gw > 0.0:
                reward += gw * max(odo - self._prev_odo, 0.0) / (
                    self.vmax * self.step_length)
            self._prev_odo = odo

            # ---- 목표 차로 보상: 지금 차로가 목표 도로로 이어지지 않으면 감점 ----
            # 지금 차로로 (차로변경 없이) 경로를 따라 갈 수 있는 거리가
            # lane_plan_range보다 짧아질수록 감점이 커진다 (최대 goal_lane_weight).
            # 즉 교차로에 가까워질수록 "목표 도로로 가는 차로"에 있어야 한다.
            # 경로 끝에 다 와서 남은 거리가 짧은 것은 감점하지 않는다.
            lw = float(self.rw.get("goal_lane_weight", 0.0))
            if lw > 0.0 and not traci.vehicle.getRoadID(self.ego_id).startswith(":"):
                rem_lane = self._route_remaining(traci.vehicle.getLaneID(self.ego_id))
                if rem_lane is not None and rem_lane + 1.0 < self._goal_remaining():
                    reward -= lw * (1.0 - min(rem_lane / self.lane_plan_range, 1.0))
            # 앞차에 너무 붙으면 감점. 단, 대기 줄에 멈춰 서 있을 때
            # (속도 < close_gap_min_speed)는 2~3m 간격이 정상이므로 제외한다.
            if (ego_gap is not None and
                    ego_gap < float(self.rw["close_gap_threshold"]) and
                    ego_speed >= float(self.rw.get("close_gap_min_speed", 0.0))):
                reward -= float(self.rw["close_gap_penalty"])

            # ---- 막힘 페널티: 느린 앞차 뒤에 갇혀 있는 동안 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 env/mdp_config.py의 REWARD 주석 참고)
            # 단, 적색/황색 신호 앞에서 줄 서 있는 것은 "막힘"이 아니므로 제외
            waiting_signal = (signal_after is not None
                              and signal_after[2] <= self.tl_range
                              and signal_after[3] not in "Gg")
            bp = float(self.rw.get("blocked_penalty", 0.0))
            if (bp > 0.0 and not waiting_signal and ego_gap is not None
                    and ego_gap < float(self.rw.get("blocked_gap", 20.0))
                    and ego_speed < float(self.rw.get("blocked_speed_frac", 0.7))
                        * self.vmax):
                reward -= bp

            # ---- 차선변경 보상 항 ----
            # lane_change        : 이번 스텝에 policy가 낸 양자화 명령 (-1/0/+1)
            # lane_change_applied: 그 명령이 실제로 실행됐는지
            # 주의: _apply_lane_change는 유지 명령(0)에도 현재 차선으로
            # changeLane을 호출해 True를 반환한다(차선 고정 용도).
            # 따라서 "실제 변경"은 lane_change != 0 과 함께 판정해야 한다.
            if lane_change != 0 and lane_change_applied:
                # 실행된 변경마다 소액 감점 → 이득 없는 지그재그 억제.
                # 변경 덕에 속도 보상이 그 이상 늘면 순이득이므로,
                # "필요할 때만 바꾸는" 정책으로 수렴하게 만드는 항이다.
                reward -= float(self.rw.get("lane_change_penalty", 0.0))
            elif lane_change != 0:
                # 명령은 냈는데 실행 불가(차선 없음/교차로)였던 경우 감점
                # → 의미 없는 행동 출력을 줄이는 정칙화 (rl_action_penalty 대응)
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))

            # ---- 신호 위반: 감점 + (기본) 즉시 종료 ----
            if red_light:
                reward -= float(self.rw.get("red_light_penalty", 5.0))
                if self.rw.get("red_light_terminate", True):
                    terminated = True

            # ---- 경로 이탈: 내 경로로 이어지지 않는 차선 끝에 멈춰 섬 ----
            # (예: 직진해야 하는데 좌회전 전용 차선 끝에 도착) ego는 SUMO의
            # 자동 차선변경을 껐으므로 거기서 영원히 못 빠져나온다.
            # 2초 이상 그 상태면 감점하고 종료한다.
            if not traci.vehicle.getRoadID(self.ego_id).startswith(":"):
                rem = self._route_remaining(traci.vehicle.getLaneID(self.ego_id))
                if rem is not None and rem < 3.0 and ego_speed < 0.5:
                    self.wrong_lane_steps += 1
                else:
                    self.wrong_lane_steps = 0
            if self.wrong_lane_steps * self.step_length >= 2.0:
                wrong_lane = True
                reward -= float(self.rw.get("wrong_lane_penalty", 5.0))
                terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "collided": collided,
            "arrived": arrived,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_raw": accel_raw,
            "lane_change_raw": lane_raw,
            "lane_change": lane_change,
            "lane_change_applied": lane_change_applied,
            "lane_before": lane_before,
            "lane_after": lane_after,
            "red_light": red_light,
            "wrong_lane": wrong_lane,
            "goal_remaining": self._goal_remaining() if alive else None,
            # 다음 신호 상태 문자 ("G"/"y"/"r" 등, 범위 밖이거나 없으면 None)
            "signal_state": (signal_after[3] if signal_after is not None
                             and signal_after[2] <= self.tl_range else None),
        }
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
