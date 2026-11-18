"""경로별 주행 동영상: sumo-gui 화면을 매 스텝 캡처 → mp4.
usage: xvfb-run -a python tools/rl/record_routes.py MODEL SUMO_DIR OUT_DIR ROUTES [SEED]"""
import os, sys, json, glob, shutil, subprocess
import numpy as np, torch
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(BASE)
import routes7; routes7.apply()
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from test import load_agent
from train import REWARD
import traci
from PIL import Image, ImageDraw, ImageFont
model, sdir, out, routes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 3
NAMES = {"ego_0": "S -> W -> roundabout -> W -> E", "ego_1": "N -> S", "ego_2": "NE -> SW", "ego_3": "E -> W",
         "ego_4": "S -> N", "ego_5": "SW -> N", "ego_6": "W -> E"}
sim = {**SIMULATION, "gui_delay": 0.0, "gui_zoom": 450.0}
env = SumoHighwayEnv(cfg_path=os.path.join(sdir, "highway.sumocfg"), road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=OBSERVATION,
                     action=ACTION, reward=REWARD, traffic=TRAFFIC, gui=True)
agent = load_agent(model, env.observation_space.shape[0], 2)
font = ImageFont.load_default(size=26) if hasattr(ImageFont, "load_default") else None
os.makedirs(out, exist_ok=True)
summary = {}
for rid in routes:
    np.random.seed(seed); torch.manual_seed(seed)
    env.meta["ego_routes"] = [rid]
    obs, _ = env.reset()
    try:
        traci.gui.trackVehicle("View #0", env.ego_id)
        traci.gui.setZoom("View #0", float(os.environ.get("ZOOM", "2200")))
        traci.vehicle.highlight(env.ego_id, (255, 0, 255, 255), 7, 255, 100000, 0)
        traci.gui.setSchema("View #0", "real world") if os.environ.get("REAL") else None
    except Exception as e:
        print("view", e)
    fr = os.path.join(out, f"frames_{rid}"); shutil.rmtree(fr, ignore_errors=True); os.makedirs(fr)
    done, k, info = False, 0, {}
    while not done:
        obs, rew, te, tr, info = env.step(agent.predict(obs, deterministic=True)); done = te or tr
        f = os.path.join(fr, f"{k:05d}.png")
        traci.gui.screenshot("View #0", f, 1280, 720)
        k += 1
    traci.simulationStep()     # 마지막 캡처가 써지도록
    end = ("collision" if info["collided"] else "red light" if info.get("red_light") and te else
           "wrong lane" if info.get("wrong_lane") else "arrived" if info["arrived"] else "timeout")
    summary[rid] = {"steps": k, "end": end}
    files = sorted(glob.glob(os.path.join(fr, "*.png")))
    for i, f in enumerate(files):
        try:
            im = Image.open(f).convert("RGB")
        except Exception:
            continue
        d = ImageDraw.Draw(im)
        txt = f"{NAMES[rid]}   t = {i * 0.5:5.1f} s" + (f"   END: {end}" if i >= len(files) - 6 else "")
        d.rectangle([0, 0, im.size[0], 44], fill=(20, 24, 28))
        d.text((16, 8), txt, fill=(255, 255, 255), font=font)
        im.save(f)
    mp4 = os.path.join(out, f"{rid}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "16", "-pattern_type", "glob", "-i", os.path.join(fr, "*.png"),
                    "-vf", "scale=1280:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "veryfast", mp4], check=True)
    shutil.rmtree(fr)
    print(rid, summary[rid], round(os.path.getsize(mp4) / 1e6, 1), "MB", flush=True)
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
traci.close(False)
os._exit(0)
