"""Mission control: mapping flight, pad finding (camera), route planning (map), flying (learned pilot), landing.
Used by the live window (live.py) and by the headless video recorder."""
import json
import pathlib

import mujoco
import numpy as np

import common as C
from flight import Drone, Pilot, mapping_dirs
from mapper import Map25D, find_pads
from planner import plan_route, survey_route

LOOK = 4.0          # carrot distance along the route (m)
CAM_PX = 320


class Mission:
  def __init__(self, scene_dir, pilot_path, camera=True, verbose=True):
    self.scene_dir = pathlib.Path(scene_dir)
    self.drone = Drone(self.scene_dir)
    self.pilot = Pilot(pilot_path)
    self.map = Map25D()
    self.scan_dirs = mapping_dirs()
    p = self.drone.p
    self.pads = {"H": dict(x=float(p[0]), y=float(p[1]), z=float(p[2]) - 0.005, n=99)}  # home: where we start
    self._pad_obs = {}
    self.state = "LANDED"
    self.at = "H"
    self.plan, self.s, self.target = None, 0.0, None
    self.ticks = 0
    self.events = []
    self.verbose = verbose
    self.last_lidar = None
    self.flight_t0 = 0.0
    self.cam = mujoco.Renderer(self.drone.m, CAM_PX, CAM_PX) if camera else None
    self.cam_id = self.drone.m.camera("down").id
    self.last_cam = None

  # ---------------------------------------------------------------- log
  def say(self, msg):
    t = self.drone.time
    self.events.append((t, msg))
    if self.verbose:
      print(f"[{t:7.1f}s] {msg}", flush=True)

  # ---------------------------------------------------------------- commands
  def command(self, name):
    name = name.strip().upper()
    if self.state in ("FLYING", "SURVEY", "LANDING"):
      if name == self.target:
        return False
    if self.state == "CRASHED":
      self.say("Crashed. Type R to reset at home.")
      return False
    if name in ("M", "MAP"):
      home = self.pads["H"]
      start = self._start_point()
      plan = survey_route([home["x"], home["y"], home["z"]])
      plan.path[0] = start
      self._go(plan, "SURVEY", "H")
      self.say("Mapping flight: climbing to 24 m, scanning the ground with the lidar and looking for pads.")
      return True
    if name not in self.pads:
      self.say(f"Checkpoint {name} is not on the map yet. Type M to map the area first.")
      return False
    if self.state == "LANDED" and self.at == name:
      self.say(f"Already on checkpoint {name}.")
      return False
    goal = self.pads[name]
    start = self._start_point()
    plan = plan_route(self.map, start, np.array([goal["x"], goal["y"], goal["z"]]))
    if plan is None:
      self.say(f"No safe route to {name} on the current map.")
      return False
    high = plan.levels.mean()
    self.say(f"Going to checkpoint {name}: route {plan.length:.0f} m, "
             f"{'mostly between the trees' if high < 0.3 else 'partly over the treetops'}.")
    self._go(plan, "FLYING", name)
    return True

  def reset(self):
    d = self.drone
    h = self.pads["H"]
    d.d.qpos[:] = d.home_qpos; d.d.qvel[:] = 0; d.armed = False
    mujoco.mj_forward(d.m, d.d)
    self.state, self.at, self.plan, self.target = "LANDED", "H", None, None
    self.say("Reset at home.")

  def _start_point(self):
    p = self.drone.p
    if self.state == "LANDED" and self.at in self.pads:
      q = self.pads[self.at]
      return np.array([q["x"], q["y"], p[2]])
    return p

  def _go(self, plan, state, target):
    self.plan, self.s, self.state, self.target = plan, 0.0, state, target
    self.flight_t0 = self.drone.time
    self.drone.armed = True

  # ---------------------------------------------------------------- one pilot step (20 ms)
  def tick(self):
    d = self.drone
    self.ticks += 1
    if self.state in ("FLYING", "SURVEY", "LANDING") and self.plan is not None:
      path, ps = self.plan.path, self.plan.ps
      p, R = d.p, d.R
      self.s, xt = C.project(np, path, ps, p, self.s)
      c1, c2, rem, fin = C.carrots(np, path, ps, self.s, LOOK)
      dists, o, dirs = d.lidar()
      self.last_lidar = (o, dirs, dists)
      self.carrot = c1
      obs = C.build_obs(np, R, p, d.v, d.omega, c1, c2, rem, fin, dists, d.last_act)
      d.step(self.pilot(obs))
      if fin > 0 and self.state != "LANDING":
        self.state = "LANDING"
      self._check_contact(path[-1])
    else:
      d.step(np.zeros(4))
    if d.armed and self.ticks % 10 == 0:            # mapping lidar, 5 Hz
      self.map.add(d.scan(self.scan_dirs))
    if self.cam is not None and d.armed and self.ticks % 25 == 0:   # down camera, 2 Hz
      self._look_for_pads()

  def _check_contact(self, goal):
    d = self.drone
    hit = d.contact_names()
    if not hit:
      return
    p, v = d.p, d.v
    speed = float(np.linalg.norm(v))
    dxy = float(np.hypot(p[0] - goal[0], p[1] - goal[1]))
    up = d.R[2, 2]
    obstacle = [h for h in hit if h.startswith(("trunk", "canopy", "rock"))]
    if obstacle or speed > 1.8 or up < np.cos(np.deg2rad(45)):
      d.armed = False
      self.state = "CRASHED"
      what = obstacle[0].rstrip("0123456789") if obstacle else "the ground"
      self.say(f"Crash! Hit {what} at {speed:.1f} m/s. Type R to reset at home.")
      return
    if self.state == "LANDING" and dxy < 0.9 and speed < 0.7 and up > np.cos(np.deg2rad(15)):
      d.armed = False
      self.state = "LANDED"
      self.at = self.target
      what = "home" if self.target == "H" else f"checkpoint {self.target}"
      self.say(f"Landed on {what} ({dxy * 100:.0f} cm from the centre) after "
               f"{d.time - self.flight_t0:.0f} s.")
      if self.plan is not None and self.plan.levels.min() == 1:   # survey finished
        self.say(f"Map done: {self.map.n_points:,} lidar points, {self.map.coverage() * 100:.0f}% of the area. "
                 f"Pads found: {', '.join(sorted(k for k in self.pads if k != 'H')) or 'none'}.")

  def _look_for_pads(self):
    d = self.drone
    self.cam.update_scene(d.d, camera=self.cam_id)
    rgb = self.cam.render()
    self.last_cam = rgb
    found = find_pads(rgb)
    if not found:
      return
    cpos = d.d.cam_xpos[self.cam_id].copy()
    cmat = d.d.cam_xmat[self.cam_id].reshape(3, 3)
    f = (CAM_PX / 2) / np.tan(np.deg2rad(d.m.cam_fovy[self.cam_id]) / 2)
    gid = np.zeros(1, np.int32)
    for name, (u, v, npx) in found.items():
      if max(abs(u - CAM_PX / 2), abs(v - CAM_PX / 2)) > 0.36 * CAM_PX:   # only trust near-vertical views
        continue
      ray = cmat @ np.array([(u - CAM_PX / 2) / f, -(v - CAM_PX / 2) / f, -1.0])
      ray /= np.linalg.norm(ray)
      dist = mujoco.mj_ray(d.m, d.d, cpos, ray, C.LIDAR_GROUPS, 1, d.bid, gid)
      if dist <= 0:
        continue
      hit = cpos + dist * ray
      o = self._pad_obs.setdefault(name, [])
      o.append(hit)
      if len(o) >= 3:
        a = np.array(o[-15:])
        mean = np.array([np.median(a[:, 0]), np.median(a[:, 1]), np.percentile(a[:, 2], 25)])
        new = name not in self.pads
        self.pads[name] = dict(x=float(mean[0]), y=float(mean[1]), z=float(mean[2]), n=len(o))
        if new:
          self.say(f"Camera found checkpoint {name} at ({mean[0]:.1f}, {mean[1]:.1f}).")

  # ---------------------------------------------------------------- save / load the map
  def save(self, path):
    np.savez_compressed(path, top=self.map.top, pads=json.dumps(self.pads))

  def load(self, path):
    z = np.load(path, allow_pickle=False)
    self.map.top = z["top"]
    self.pads.update(json.loads(str(z["pads"])))
    self.say(f"Loaded saved map ({self.map.coverage() * 100:.0f}% covered), pads: {', '.join(sorted(self.pads))}.")
