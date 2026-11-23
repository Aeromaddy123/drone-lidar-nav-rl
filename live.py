"""Live demo. A 3D window shows the forest and the drone; you type in this terminal:

    1 .. 5   fly to that checkpoint and land
    h        fly home
    m        map the area again
    r        reset the drone at home (after a crash)
    c        show / hide the lidar map cloud
    q        quit

Run (macOS needs mjpython for the window):
    .venv/bin/mjpython live.py              # maps the area first, then waits for your command
    .venv/bin/mjpython live.py --use-saved-map
"""
import argparse
import pathlib
import queue
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from mission import Mission

HERE = pathlib.Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--pilot", default=str(HERE / "pilot.npz"))
ap.add_argument("--scene", default=str(HERE / "scene"))
ap.add_argument("--use-saved-map", action="store_true")
ap.add_argument("--survey-speed", type=float, default=5.0, help="time speed-up during the mapping flight")
ap.add_argument("--speed", type=float, default=1.0, help="time speed-up during missions")
args = ap.parse_args()

ms = Mission(args.scene, args.pilot)
saved = HERE / "saved_map.npz"
cmds = queue.Queue()
show_cloud = [True]

HELP = "Type 1-5 to fly to a checkpoint, h = home, m = map again, r = reset, c = map cloud on/off, q = quit."


def reader():
  print("\n" + HELP, flush=True)
  for line in sys.stdin:
    line = line.strip()
    if line:
      cmds.put(line)


threading.Thread(target=reader, daemon=True).start()

LEVEL_RGBA = {0: (0.2, 0.9, 1.0, 1.0), 1: (1.0, 0.62, 0.15, 1.0)}


def add_geom(scn, gtype, size, pos, rgba, mat=None):
  if scn.ngeom >= scn.maxgeom:
    return None
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                      (np.eye(3) if mat is None else mat).reshape(-1), np.asarray(rgba, np.float32))
  scn.ngeom += 1
  return g


def add_line(scn, a, b, rgba, width=3.0):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                      np.asarray(rgba, np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
  scn.ngeom += 1


def draw(scn):
  scn.ngeom = 0
  # map cloud (coloured by height)
  if show_cloud[0] and ms.map.points:
    pts = np.concatenate(ms.map.points[-400:])[::3][-3500:]
    z = pts[:, 2]
    t = np.clip((z + 6) / 26, 0, 1)
    for p, tt in zip(pts, t):
      add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.09, 0.09, 0.09], p, (0.15 + 0.85 * tt, 0.9 - 0.5 * tt, 1.0 - 0.8 * tt, 0.9))
  # planned route
  if ms.plan is not None and ms.state in ("FLYING", "SURVEY", "LANDING"):
    path, lv = ms.plan.path, ms.plan.levels
    step = max(1, len(path) // 120)
    idx = list(range(0, len(path), step))
    if idx[-1] != len(path) - 1:
      idx.append(len(path) - 1)
    for i, j in zip(idx[:-1], idx[1:]):
      add_line(scn, path[i], path[j], LEVEL_RGBA[int(lv[min(j, len(lv) - 1)])], 4.0)
    if hasattr(ms, "carrot"):
      add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.18, 0, 0], ms.carrot, (1.0, 0.95, 0.2, 0.9))
  # what the pilot's lidar is touching right now
  if ms.last_lidar is not None and ms.state in ("FLYING", "SURVEY", "LANDING"):
    o, dirs, dist = ms.last_lidar
    for k in range(len(dist)):
      if dist[k] < 14.5:
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.07, 0, 0], o + dirs[k] * dist[k], (1.0, 0.2, 0.25, 1.0))
  # pads the camera has found
  for name, p in ms.pads.items():
    add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.06, 0.6, 0], [p["x"], p["y"], p["z"] + 2.0], (1, 1, 1, 0.5))


def minimap(size=300):
  surf, ground, ohag, known = ms.map.layers() if ms.map.n_points else (None,) * 4
  img = np.full((ms.map.n, ms.map.n, 3), 25, np.uint8)
  if surf is not None:
    t = np.clip((surf + 6) / 20, 0, 1)
    base = np.stack([60 + 120 * t, 90 + 110 * t, 60 + 60 * t], -1)
    base[ohag > 0.7] = base[ohag > 0.7] * 0.5 + np.array([20, 90, 30]) * 0.5
    base[~known] = 25
    img = base.astype(np.uint8)
  img = img[::-1].copy()
  n = ms.map.n
  def px(x, y):
    return int((x + 60) / 120 * (n - 1)), int((60 - y) / 120 * (n - 1))
  if ms.plan is not None and ms.state in ("FLYING", "SURVEY", "LANDING"):
    for q in ms.plan.path[::2]:
      i, j = px(q[0], q[1]); img[max(j - 1, 0):j + 1, max(i - 1, 0):i + 1] = (60, 230, 255)
  for name, p in ms.pads.items():
    i, j = px(p["x"], p["y"]); img[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3] = (255, 255, 255)
  i, j = px(*ms.drone.p[:2]); img[max(j - 3, 0):j + 4, max(i - 3, 0):i + 4] = (255, 60, 60)
  from PIL import Image
  return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def handle(c):
  c = c.lower()
  if c in ("q", "quit", "exit"):
    return False
  if c == "r":
    ms.reset()
  elif c == "c":
    show_cloud[0] = not show_cloud[0]
  elif c in ("m", "map"):
    ms.command("M")
  elif c in ("h", "home"):
    ms.command("H")
  elif c.replace("go", "").strip() in list("12345"):
    ms.command(c.replace("go", "").strip())
  else:
    print(HELP, flush=True)
  return True


with mujoco.viewer.launch_passive(ms.drone.m, ms.drone.d, show_left_ui=False, show_right_ui=False) as v:
  v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  v.cam.trackbodyid = ms.drone.bid
  v.cam.distance, v.cam.elevation, v.cam.azimuth = 16.0, -24.0, 45.0
  if args.use_saved_map and saved.exists():
    ms.load(saved)
    show_cloud[0] = False
  else:
    ms.say("Step 1: mapping the area (lidar + camera). Shown at 5x speed.")
    ms.command("M")
  last = time.time()
  frame = 0
  was = ms.state
  running = True
  while v.is_running() and running:
    while not cmds.empty():
      running = handle(cmds.get())
    now = time.time()
    dt = min(now - last, 0.1)
    last = now
    speed = args.survey_speed if ms.state == "SURVEY" else args.speed
    n = max(1, int(round(dt * speed / 0.02)))
    with v.lock():
      for _ in range(n):
        ms.tick()
      draw(v.user_scn)
      if was == "SURVEY" and ms.state == "LANDED":
        ms.save(saved)
        show_cloud[0] = False
        print("\n" + HELP, flush=True)
      if was in ("FLYING", "LANDING") and ms.state == "LANDED":
        print("\n" + HELP, flush=True)
      was = ms.state
      sp = np.linalg.norm(ms.drone.v)
      agl = ms.last_lidar[2][144] if ms.last_lidar is not None else 0.0
      target = {"H": "home"}.get(ms.target, f"checkpoint {ms.target}") if ms.target else "-"
      texts = [(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "STATE\nTARGET\nSPEED\nHEIGHT\nMAP\nPADS",
                f"{ms.state}\n{target}\n{sp:.1f} m/s\n{agl:.1f} m above ground\n"
                f"{ms.map.n_points:,} lidar points\n{', '.join(sorted(ms.pads))}")]
      frame += 1
      imgs = None
      if frame % 6 == 0:
        vp = v.viewport
        if vp is not None and vp.width > 700:
          s = 300
          imgs = [(mujoco.MjrRect(vp.width - s - 10, 10, s, s), minimap(s))]
          if ms.last_cam is not None:
            from PIL import Image
            cam = np.asarray(Image.fromarray(ms.last_cam).resize((s, s)))
            imgs.append((mujoco.MjrRect(vp.width - s - 10, 20 + s, s, s), cam))
    # overlays are handed to the window thread outside the lock (inside it they deadlock)
    v.set_texts(texts)
    if imgs:
      v.set_images(imgs)
    v.sync()
    time.sleep(max(0.0, 1 / 60 - (time.time() - now)))
