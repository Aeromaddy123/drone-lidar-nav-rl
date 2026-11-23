"""The drone in the demo world: MuJoCo physics, the same flight controller as training, real lidar rays
(mj_multiRay against the scene), the down camera, and the learned pilot (NumPy, no GPU needed)."""
import pathlib

import mujoco
import numpy as np

import common as C


class Pilot:
  """The trained PPO policy, exported to plain arrays (see export_policy.py)."""

  def __init__(self, path):
    z = np.load(path)
    self.mean, self.std = z["mean"], z["std"]
    self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["n"]))]
    self.step = int(z["step"]) if "step" in z else -1

  def __call__(self, obs):
    x = (obs - self.mean) / self.std
    for i, (w, b) in enumerate(self.layers):
      x = x @ w + b
      if i < len(self.layers) - 1:
        x = x / (1.0 + np.exp(-x))         # swish
    return np.tanh(x[:4])


class Drone:
  def __init__(self, scene_dir):
    scene_dir = pathlib.Path(scene_dir)
    self.m = mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))
    self.d = mujoco.MjData(self.m)
    self.bid, self.mass, self.J, self.Ainv = C.drone_constants(self.m)
    assert abs(self.m.opt.timestep - C.SIM_DT) < 1e-9
    self.armed = False
    self.last_act = np.zeros(4)
    self._gid = np.zeros(C.N_RAYS, np.int32)
    self._dist = np.zeros(C.N_RAYS)
    self.home_qpos = self.d.qpos.copy()
    mujoco.mj_forward(self.m, self.d)

  # ---------------------------------------------------------------- state
  @property
  def p(self):
    return self.d.qpos[:3].copy()

  @property
  def v(self):
    return self.d.qvel[:3].copy()

  @property
  def omega(self):
    return self.d.qvel[3:6].copy()

  @property
  def R(self):
    q = self.d.qpos[3:7]
    return C.quat_to_mat(np, q / np.linalg.norm(q))

  @property
  def time(self):
    return self.d.time

  def place(self, x, y, z, yaw=0.0):
    self.d.qpos[:3] = [x, y, z]
    self.d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    self.d.qvel[:] = 0
    mujoco.mj_forward(self.m, self.d)

  # ---------------------------------------------------------------- sensors
  def lidar(self):
    """Pilot lidar: 144 ring beams + 9 down + 1 up, body-fixed, same layout as training."""
    R, p = self.R, self.p
    o = p + R @ np.array([0.0, 0.0, C.LIDAR_OFFSET])
    dirs = (C.RAY_DIRS.astype(np.float64) @ R.T).reshape(-1)
    mujoco.mj_multiRay(self.m, self.d, o, dirs, C.LIDAR_GROUPS, 1, self.bid, self._gid, self._dist, None,
                       C.N_RAYS, float(C.DOWN_RANGE) + 1)
    d = np.where(self._dist < 0, C.RAY_RANGES, np.minimum(self._dist, C.RAY_RANGES))
    return d, o, dirs.reshape(-1, 3)

  def scan(self, dirs_body, max_range=80.0):
    """Mapping lidar: returns world hit points for the given body-frame directions."""
    R, p = self.R, self.p
    o = p + R @ np.array([0.0, 0.0, -0.02])
    dirs = dirs_body @ R.T
    n = len(dirs)
    gid = np.zeros(n, np.int32); dist = np.zeros(n)
    mujoco.mj_multiRay(self.m, self.d, o, dirs.reshape(-1).astype(np.float64), C.LIDAR_GROUPS, 1, self.bid, gid, dist,
                       None, n, max_range)
    ok = dist > 0
    return o + dirs[ok] * dist[ok, None]

  def contact_names(self):
    """Names of world geoms the drone is touching (empty list if none)."""
    out = []
    for i in range(self.d.ncon):
      c = self.d.contact[i]
      b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
      if (b1 == self.bid) != (b2 == self.bid):
        other = c.geom2 if b1 == self.bid else c.geom1
        out.append(self.m.geom(other).name or "geom")
    return out

  # ---------------------------------------------------------------- control
  def step(self, action):
    """One pilot step (20 ms): body-rate loop + mixer at 200 Hz, exactly as in training."""
    action = np.clip(action, -1.0, 1.0)
    if self.armed:
      thrust, rates = C.action_to_command(np, action, self.mass)
    for _ in range(C.N_SUB):
      if self.armed:
        f = C.rate_controller(np, thrust, rates, self.d.qvel[3:6], self.J, self.Ainv)
        self.d.ctrl[:] = np.clip(f, 0.0, C.MOTOR_MAX)
      else:
        self.d.ctrl[:] = 0.0
      mujoco.mj_step(self.m, self.d)
    self.last_act = action if self.armed else np.zeros(4)


def mapping_dirs(half_angle_deg=55.0, n=41):
  """Downward cone of rays for the mapping lidar (body frame)."""
  t = np.tan(np.deg2rad(half_angle_deg))
  u = np.linspace(-t, t, n)
  X, Y = np.meshgrid(u, u)
  keep = X ** 2 + Y ** 2 <= t * t
  v = np.stack([X[keep], Y[keep], -np.ones(keep.sum())], -1)
  return v / np.linalg.norm(v, axis=1, keepdims=True)
