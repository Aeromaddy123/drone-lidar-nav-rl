"""Record a full demo session for the video: mapping flight, then the commanded missions.
Saves every pilot step (pose, route, carrot, lidar hits) and every mapping scan.
  MUJOCO_GL=egl python record_demo.py --pilot pilot.npz --order M,3 --out rec/demo.pkl"""
import argparse, os, pickle, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

import common as C
from mission import Mission

ap = argparse.ArgumentParser()
ap.add_argument("--pilot", required=True)
ap.add_argument("--order", default="M,3")
ap.add_argument("--out", required=True)
ap.add_argument("--max_s", type=float, default=200.0)
a = ap.parse_args()

ms = Mission("scene", a.pilot)
orig_add = ms.map.add
scans = []


def add(pts):                       # keep every mapping scan with its time
  scans.append((ms.drone.time, pts.astype(np.float32)))
  orig_add(pts)


ms.map.add = add
frames, segments = [], []
for cmd in a.order.split(","):
  ok = ms.command(cmd)
  seg = dict(cmd=cmd, ok=ok, t0=ms.drone.time, path=None if ms.plan is None else ms.plan.path.copy(),
             levels=None if ms.plan is None else ms.plan.levels.copy())
  t0 = ms.drone.time
  while ok and ms.state in ("FLYING", "SURVEY", "LANDING") and ms.drone.time - t0 < a.max_s:
    ms.tick()
    hits = None
    if ms.last_lidar is not None:
      o, dirs, dist = ms.last_lidar
      k = dist < C.RAY_RANGES - 0.05
      hits = (o + dirs[k] * dist[k, None]).astype(np.float32)
    frames.append(dict(t=ms.drone.time, qpos=ms.drone.d.qpos.copy(), state=ms.state, s=ms.s,
                       carrot=getattr(ms, "carrot", None), hits=hits, cmd=cmd,
                       ctrl=ms.drone.d.ctrl.copy(), v=ms.drone.v))
  # a few seconds standing on the pad after landing
  for _ in range(100):
    ms.tick()
    frames.append(dict(t=ms.drone.time, qpos=ms.drone.d.qpos.copy(), state=ms.state, s=ms.s, carrot=None,
                       hits=None, cmd=cmd, ctrl=ms.drone.d.ctrl.copy(), v=ms.drone.v))
  seg.update(t1=ms.drone.time, end=ms.state)
  segments.append(seg)
  print(f"{cmd}: {ms.state} after {ms.drone.time - t0:.1f}s", flush=True)
  if ms.state == "CRASHED":
    break
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump(dict(frames=frames, scans=scans, segments=segments, events=ms.events, pads=ms.pads,
                 map_top=ms.map.top), open(a.out, "wb"))
print("saved", a.out, len(frames), "frames", len(scans), "scans")
