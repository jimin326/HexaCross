"""수집한 Parquet 샤드 검사: 행 수, 크기, (s, a, r, s', done) 일관성, 끝난 이유.

usage: python tools/data/check_parquet.py DATA_DIR [최대 샤드 수]
- next_observation[t] == observation[t+1] (같은 에피소드 안)
- next_ego_x[t] == ego_x[t+1]
- 에피소드마다 t가 0부터 1씩 늘고, 마지막 행만 terminated 또는 truncated
"""
import glob, os, sys
import numpy as np
import pyarrow.parquet as pq

d = sys.argv[1]
fs = sorted(glob.glob(os.path.join(d, "shard_*.parquet")))
if len(sys.argv) > 2:
    fs = fs[:int(sys.argv[2])]
rows = size = bad_obs = bad_state = bad_t = bad_done = n_ep = 0
ends, modes, veh, det = {}, [0, 0], [], []
for f in fs:
    tb = pq.read_table(f)
    size += os.path.getsize(f); rows += tb.num_rows
    ep = tb["episode"].to_numpy(); t = tb["t"].to_numpy()
    obs = np.stack(tb["observation"].to_numpy(zero_copy_only=False)); nobs = np.stack(tb["next_observation"].to_numpy(zero_copy_only=False))
    ex = tb["ego_x"].to_numpy(zero_copy_only=False); nex = tb["next_ego_x"].to_numpy(zero_copy_only=False)
    done = tb["terminated"].to_numpy(zero_copy_only=False) | tb["truncated"].to_numpy(zero_copy_only=False)
    same = ep[:-1] == ep[1:]
    bad_obs += int((~np.all(np.isclose(nobs[:-1], obs[1:]), axis=1) & same).sum())
    bad_state += int(((nex[:-1] != ex[1:]) & same).sum())
    bad_t += int(((t[1:] != t[:-1] + 1) & same).sum()) + int((t[1:][~same] != 0).sum()) + int(t[0] != 0)
    last = np.r_[~same, True]
    bad_done += int((done != last).sum())
    n_ep += int(last.sum())
    pm = tb["policy_mode"].to_numpy()
    for i in np.where(last)[0]:
        e = ("arrival" if tb["arrived"][i].as_py() else "collision" if tb["collided"][i].as_py() else "red_light" if tb["red_light"][i].as_py()
             else "wrong_lane" if tb["wrong_lane"][i].as_py() else "timeout")
        k = ("det_" if pm[i] == 0 else "sto_") + e
        ends[k] = ends.get(k, 0) + 1
    modes[0] += int((pm == 0).sum()); modes[1] += int((pm == 1).sum())
    veh.append(tb["vehicles"].combine_chunks().value_lengths().to_numpy().mean())
    det.append(tb["detected_vehicles"].combine_chunks().value_lengths().to_numpy().mean())
print(f"shards {len(fs)}  rows {rows:,}  episodes {n_ep:,}  size {size/1e9:.2f} GB  ({size/max(rows,1):.0f} B/row)")
print(f"consistency errors: next_obs {bad_obs}  next_state {bad_state}  t {bad_t}  done {bad_done}")
print("policy_mode rows (det, sto):", modes)
print("episode ends:", dict(sorted(ends.items())))
print(f"vehicles within radius per row: {np.mean(veh):.1f}  detected per row: {np.mean(det):.1f}")
