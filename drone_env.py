"""Skydio X2 pilot: follow a route through a random forest on hilly ground and land on a pad.

Physics: the real Skydio X2 MuJoCo model (MJX), driven through a body-rate controller.
World: every episode is a new random world (hills, trees, rocks, wind), described analytically so
thousands of worlds fit on the GPU. Lidar rays and collisions are computed against that description.
"""
import re
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

import common as C

N_TREES, N_ROCKS, N_PATH = 48, 10, 30


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(ctrl_dt=C.CTRL_DT, sim_dt=C.SIM_DT, episode_length=1500, action_repeat=1,
                            impl="jax", naconmax=0, njmax=0, area=38.0, vision=False)


def x2_physics_xml() -> str:
  src = (mjx_env.MENAGERIE_PATH / "skydio_x2" / "x2.xml").read_text()
  for pat in [r'<texture[^>]*/>', r'<material name="phong3SG"[^>]*/>', r'<mesh class="x2"[^>]*/>',
              r'<geom material="phong3SG"[^>]*/>', r'<light[^>]*/>']:
    src = re.sub(pat, '', src)
  return C.fix_x2(src)


def drone_constants(m: mujoco.MjModel):
  """Mass, inertia (body frame) and motor mixer from the model itself."""
  bid = m.body("x2").id
  mass = float(m.body_subtreemass[bid])
  com = m.body_ipos[bid].copy()
  rq = np.zeros(9); mujoco.mju_quat2Mat(rq, m.body_iquat[bid]); rq = rq.reshape(3, 3)
  J = rq @ np.diag(m.body_inertia[bid]) @ rq.T
  A = np.zeros((4, 4))
  for i in range(m.nu):
    r = m.site_pos[m.actuator_trnid[i, 0]] - com
    A[:, i] = [1.0, r[1], -r[0], m.actuator_gear[i, 5]]
  assert np.linalg.cond(A) < 1e4, "motor mixer is singular"
  return bid, mass, J, np.linalg.inv(A)


# ------------------------------------------------------------------ analytic world (JAX)
def smoothstep(x):
  x = jp.clip(x, 0.0, 1.0)
  return x * x * (3 - 2 * x)


def terrain_raw(T, xy):
  ph = xy[..., None, 0] * T["F"][:, 0] + xy[..., None, 1] * T["F"][:, 1] + T["P"]
  return (T["A"] * jp.sin(ph)).sum(-1)


def terrain_h(T, xy):
  """Ground height; flattened around the two pads (pads are flat discs)."""
  h = terrain_raw(T, xy)
  for i in range(2):
    pad = T["pads"][i]
    d = jp.sqrt(((xy - pad[:2]) ** 2).sum(-1) + 1e-12)
    w = smoothstep((d - pad[3]) / 1.8)
    h = pad[2] * (1 - w) + h * w
  return h


def ray_cast(T, o, dirs, ranges):
  """Distances along each ray to terrain, trunks (vertical cylinders) and canopies/rocks (ellipsoids)."""
  # terrain: coarse march + bisection
  n0 = 16
  ts = ranges[:, None] * (jp.arange(1, n0 + 1) / n0)[None, :]
  pts = o + ts[..., None] * dirs[:, None, :]
  below = (pts[..., 2] - terrain_h(T, pts[..., :2])) < 0
  hit = below.any(-1)
  idx = jp.argmax(below, -1)
  t_hi = jp.take_along_axis(ts, idx[:, None], 1)[:, 0]
  t_lo = jp.where(idx > 0, jp.take_along_axis(ts, jp.maximum(idx - 1, 0)[:, None], 1)[:, 0], 0.0)
  for _ in range(8):
    tm = 0.5 * (t_lo + t_hi)
    pm = o + tm[:, None] * dirs
    under = (pm[:, 2] - terrain_h(T, pm[:, :2])) < 0
    t_hi = jp.where(under, tm, t_hi)
    t_lo = jp.where(under, t_lo, tm)
  t_ter = jp.where(hit, t_hi, ranges)
  t_ter = jp.where(o[2] < terrain_h(T, o[:2]), 0.0, t_ter)
  # trunks
  tr = T["trunks"]                                   # (NT,5) cx cy r z0 z1
  ox, oy = o[0] - tr[:, 0], o[1] - tr[:, 1]
  a = (dirs[:, 0] ** 2 + dirs[:, 1] ** 2)[:, None]
  b = 2 * (dirs[:, 0:1] * ox[None] + dirs[:, 1:2] * oy[None])
  c = (ox ** 2 + oy ** 2 - tr[:, 2] ** 2)[None]
  disc = b * b - 4 * a * c
  t = (-b - jp.sqrt(jp.maximum(disc, 0.0))) / (2 * a + 1e-9)
  z = o[2] + t * dirs[:, 2:3]
  ok = (disc > 0) & (t > 0) & (z >= tr[:, 3]) & (z <= tr[:, 4]) & (a > 1e-6)
  t_cyl = jp.where(ok, t, 1e6).min(-1)
  # ellipsoids (axis aligned, rx = ry)
  el = T["ells"]                                     # (NE,5) cx cy cz rxy rz
  inv = 1.0 / jp.stack([el[:, 3], el[:, 3], el[:, 4]], -1)     # (NE,3)
  oc = (o - el[:, :3]) * inv                         # (NE,3)
  a = (dirs ** 2) @ (inv ** 2).T                     # (R,NE)
  b = 2 * dirs @ (oc * inv).T
  c = (oc ** 2).sum(-1)[None] - 1.0
  disc = b * b - 4 * a * c
  t = (-b - jp.sqrt(jp.maximum(disc, 0.0))) / (2 * a)
  t = jp.where(c < 0, 0.0, t)
  t_ell = jp.where((disc > 0) & (t >= 0), t, 1e6).min(-1)
  return jp.minimum(jp.minimum(t_ter, ranges), jp.minimum(t_cyl, t_ell))


def clearance(T, p):
  """Smallest gap (m) between the drone sphere and any tree trunk, canopy or rock."""
  tr = T["trunks"]
  dxy = jp.sqrt((p[0] - tr[:, 0]) ** 2 + (p[1] - tr[:, 1]) ** 2) - tr[:, 2]
  dz = jp.maximum(jp.maximum(tr[:, 3] - p[2], p[2] - tr[:, 4]), 0.0)
  d_tr = jp.sqrt(jp.maximum(dxy, 0.0) ** 2 + dz ** 2) - jp.where((dxy < 0) & (dz <= 0), 1.0, 0.0)
  el = T["ells"]
  rad = jp.stack([el[:, 3], el[:, 3], el[:, 4]], -1)
  n = jp.sqrt((((p - el[:, :3]) / rad) ** 2).sum(-1))
  d_el = (n - 1.0) * jp.minimum(el[:, 3], el[:, 4])
  return jp.minimum(d_tr.min(), d_el.min()) - C.DRONE_R


def seg_dist_2d(pts, poly):
  """Distance from points (N,2) to a polyline (M,2)."""
  a, b = poly[:-1], poly[1:]
  ab = b - a
  t = jp.clip((((pts[:, None] - a[None]) * ab[None]).sum(-1)) / ((ab * ab).sum(-1) + 1e-9), 0, 1)
  q = a[None] + t[..., None] * ab[None]
  return jp.sqrt(((pts[:, None] - q) ** 2).sum(-1)).min(-1)


def resample(poly, n):
  seg = jp.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(-1))
  cs = jp.concatenate([jp.zeros(1), jp.cumsum(seg)])
  s = jp.linspace(0.0, cs[-1], n)
  i = jp.clip(jp.searchsorted(cs, s, side="right") - 1, 0, len(cs) - 2)
  t = (s - cs[i]) / jp.maximum(cs[i + 1] - cs[i], 1e-6)
  return poly[i] + t[:, None] * (poly[i + 1] - poly[i])


def quat_from_euler(r, p, y):
  cr, sr, cp, sp, cy, sy = jp.cos(r / 2), jp.sin(r / 2), jp.cos(p / 2), jp.sin(p / 2), jp.cos(y / 2), jp.sin(y / 2)
  return jp.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                   cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class DroneNav(mjx_env.MjxEnv):

  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_string(x2_physics_xml())
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
    self._bid, self._mass, J, Ainv = drone_constants(self._mj_model)
    self._J, self._Ainv = jp.array(J, jp.float32), jp.array(Ainv, jp.float32)
    self._dirs, self._ranges = jp.array(C.RAY_DIRS), jp.array(C.RAY_RANGES)
    self._area = float(self._config.area)

  # ---------------------------------------------------------------- world generation
  def _make_world(self, rng):
    ks = iter(jax.random.split(rng, 64))
    U = lambda shape=(), lo=0.0, hi=1.0: jax.random.uniform(next(ks), shape, minval=lo, maxval=hi)
    area = self._area
    relief = U() ** 0.7
    lam = jp.concatenate([U((3,), 18, 70), U((3,), 6, 16)])
    ang = U((6,), 0, 2 * jp.pi)
    F = (2 * jp.pi / lam)[:, None] * jp.stack([jp.cos(ang), jp.sin(ang)], -1)
    A = relief * lam * U((6,), 0.012, 0.045) * jp.array([1, 1, 1, .6, .6, .6])
    T = {"A": A, "F": F, "P": U((6,), 0, 2 * jp.pi)}
    start = U((2,), -6, 6)
    dist = U((), 6, 40)
    th = U((), 0, 2 * jp.pi)
    goal = jp.clip(start + dist * jp.array([jp.cos(th), jp.sin(th)]), -area + 3, area - 3)
    dist = jp.sqrt(((goal - start) ** 2).sum()) + 1e-3
    pad_r = 1.2
    hs, hg = terrain_raw(T, start), terrain_raw(T, goal)
    T["pads"] = jp.array([[start[0], start[1], hs, pad_r], [goal[0], goal[1], hg, pad_r]])
    # 2D route with two wiggles, like a planned path
    u = (goal - start) / dist
    perp = jp.array([-u[1], u[0]])
    wig = jp.minimum(6.0, dist / 3)
    poly = jp.stack([start, start + u * dist / 3 + perp * U((), -1, 1) * wig,
                     start + u * 2 * dist / 3 + perp * U((), -1, 1) * wig, goal])
    route = resample(poly, N_PATH)
    # trees
    dens = U((), 0.15, 1.0)
    near = (jp.arange(N_TREES) < int(N_TREES * 0.6))[:, None]      # most trees crowd the route
    ridx = jax.random.randint(next(ks), (N_TREES,), 0, N_PATH)
    txy = jp.where(near, route[ridx] + 6.0 * jax.random.normal(next(ks), (N_TREES, 2)), U((N_TREES, 2), -area, area))
    txy = jp.clip(txy, -area, area)
    act = U((N_TREES,)) < dens
    corridor = U() < 0.5
    cw = U((), 1.0, 3.0)
    r_t = U((N_TREES,), 0.15, 0.4)
    H = U((N_TREES,), 5, 13)
    rxy = U((N_TREES,), 1.2, 3.0)
    rz = rxy * U((N_TREES,), 1.0, 1.7)
    g = terrain_h(T, txy)
    cz = g + jp.maximum(H * U((N_TREES,), 0.55, 0.8), rz + U((N_TREES,), 1.2, 3.5))
    dpad = jp.sqrt(((txy[:, None] - T["pads"][None, :, :2]) ** 2).sum(-1)).min(-1)
    droute = seg_dist_2d(txy, route)
    act = act & (dpad > rxy + 2.0) & ~(corridor & (droute < rxy + cw))
    far = jp.where(act, 0.0, 500.0)
    trunks = jp.stack([txy[:, 0] + far, txy[:, 1], r_t, g - 0.5, cz], -1)
    canopies = jp.stack([txy[:, 0] + far, txy[:, 1], cz, rxy, rz], -1)
    # rocks / bushes
    rnear = (jp.arange(N_ROCKS) < N_ROCKS // 2)[:, None]
    rxyz = jp.where(rnear, route[jax.random.randint(next(ks), (N_ROCKS,), 0, N_PATH)] + 5.0 * jax.random.normal(next(ks), (N_ROCKS, 2)),
                    U((N_ROCKS, 2), -area, area))
    rr = U((N_ROCKS,), 0.4, 1.8)
    rrz = rr * U((N_ROCKS,), 0.4, 0.9)
    rg = terrain_h(T, rxyz)
    ract = (U((N_ROCKS,)) < 0.8)
    ract = ract & (jp.sqrt(((rxyz[:, None] - T["pads"][None, :, :2]) ** 2).sum(-1)).min(-1) > rr + 2.0)
    ract = ract & ~(corridor & (seg_dist_2d(rxyz, route) < rr + cw))
    rocks = jp.stack([rxyz[:, 0] + jp.where(ract, 0.0, 500.0), rxyz[:, 1], rg + rrz * U((N_ROCKS,), -0.4, 0.3), rr, rrz], -1)
    T["trunks"] = trunks
    T["ells"] = jp.concatenate([canopies, rocks], 0)
    # altitude plan
    cruise = jp.where(U() < 0.75, U((), 2.5, 7.0), U((), 9.0, 25.0))
    hover = U((), 1.5, 4.0)
    rz_ = terrain_h(T, route) + cruise
    ground_start = U() < 0.5
    sz = jp.where(ground_start, hs, hs + U((), 1.0, 6.0))
    p0 = jp.array([start[0], start[1], sz])
    path = jp.concatenate([p0[None], jp.concatenate([route, rz_[:, None]], -1),
                           jp.array([[goal[0], goal[1], hg + hover], [goal[0], goal[1], hg]])], 0)
    seg = jp.sqrt(((path[1:] - path[:-1]) ** 2).sum(-1))
    ps = jp.concatenate([jp.zeros(1), jp.cumsum(seg)])
    return T, path, ps, ground_start, p0, next(ks)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, kw = jax.random.split(rng)
    T, path, ps, ground_start, p0, k = self._make_world(kw)
    ks = jax.random.split(k, 8)
    tilt = jp.where(ground_start, 0.0, 0.17)
    q = quat_from_euler(tilt * jax.random.uniform(ks[0], (), minval=-1, maxval=1),
                        tilt * jax.random.uniform(ks[1], (), minval=-1, maxval=1),
                        jax.random.uniform(ks[2], (), minval=-jp.pi, maxval=jp.pi))
    v0 = jp.where(ground_start, 0.0, 1.0) * jax.random.uniform(ks[3], (3,), minval=-1, maxval=1)
    w0 = jp.where(ground_start, 0.0, 0.5) * jax.random.uniform(ks[4], (3,), minval=-1, maxval=1)
    qpos = jp.concatenate([p0, q])
    qvel = jp.concatenate([v0, w0])
    data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, impl="jax", naconmax=0, njmax=0)
    wang = jax.random.uniform(ks[5], (), minval=0, maxval=2 * jp.pi)
    wmag = jax.random.uniform(ks[6], (), minval=0, maxval=1.5)
    info = {
        "rng": rng, "world": T, "path": path, "ps": ps, "s": jp.zeros(()),
        "look": jax.random.uniform(ks[7], (), minval=3.0, maxval=5.5),
        "wind": jp.array([jp.cos(wang), jp.sin(wang), 0.0]) * wmag,
        "motor_scale": jax.random.uniform(jax.random.fold_in(ks[7], 1), (), minval=0.92, maxval=1.08),
        "last_act": jp.zeros(4),
    }
    # first observation without lidar (start spots are clear); keeps auto-reset cheap
    R = C.quat_to_mat(jp, q)
    c1, c2, rem, fin = C.carrots(jp, path, ps, 0.0, info["look"])
    dists = jp.concatenate([jp.full(C.N_RING, C.RING_RANGE), jp.full(9, C.DOWN_RANGE).at[0].set(p0[2] - terrain_h(T, p0[:2])),
                            jp.full(1, C.UP_RANGE)])
    obs = C.build_obs(jp, R, p0, v0, w0, c1, c2, rem, fin, dists, jp.zeros(4))
    metrics = {k_: jp.zeros(()) for k_ in ["success", "crash", "completion"]}
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  # ---------------------------------------------------------------- dynamics
  def _physics(self, data, a, motor_scale, wind):
    thrust, rates = C.action_to_command(jp, a, self._mass)
    xfrc = data.xfrc_applied.at[self._bid, :3].set(wind)

    def sub(d, _):
      f = C.rate_controller(jp, thrust, rates, d.qvel[3:6], self._J, self._Ainv)
      d = d.replace(ctrl=jp.clip(f * motor_scale, 0.0, C.MOTOR_MAX), xfrc_applied=xfrc)
      return mjx.step(self._mjx_model, d), None

    data, _ = jax.lax.scan(sub, data, None, length=self.n_substeps)
    return data

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = dict(state.info)
    T, path, ps = info["world"], info["path"], info["ps"]
    a = jp.clip(action, -1.0, 1.0)
    rng, k1, k2, k3 = jax.random.split(info["rng"], 4)
    wind = info["wind"] + 0.35 * jax.random.normal(k1, (3,)) * jp.array([1.0, 1.0, 0.5])
    data = self._physics(state.data, a, info["motor_scale"], wind)

    p, v = data.qpos[:3], data.qvel[:3]
    R = C.quat_to_mat(jp, data.qpos[3:7] / jp.linalg.norm(data.qpos[3:7]))
    up_z = R[2, 2]
    g = terrain_h(T, p[:2])
    below = p[2] <= g
    speed_pre = jp.sqrt((v * v).sum())
    impact = below & ((v[2] < -1.5) | (up_z < np.cos(np.deg2rad(35))))
    # ground support (MJX world has no floor; the ground is analytic)
    qpos = data.qpos.at[2].set(jp.where(below, g, p[2]))
    damp = jp.where(below, 0.2, 1.0)
    qvel = jp.concatenate([v[:2] * damp, jp.where(below, jp.maximum(v[2], 0.0), v[2])[None], data.qvel[3:6] * damp])
    data = data.replace(qpos=qpos, qvel=qvel)
    p, v = qpos[:3], qvel[:3]

    # route progress
    s_old = info["s"]
    s_new, xtrack = C.project(jp, path, ps, p, s_old)
    c1, c2, rem, fin = C.carrots(jp, path, ps, s_new, info["look"])

    # sensors
    o = p + R @ jp.array([0.0, 0.0, C.LIDAR_OFFSET])
    dirs_w = self._dirs @ R.T
    dists = ray_cast(T, o, dirs_w, self._ranges)
    dists = dists * (1.0 + 0.01 * jax.random.normal(k2, dists.shape))
    dists = jp.where(jax.random.uniform(k3, dists.shape) < 0.005, self._ranges, dists)
    omega = data.qvel[3:6]
    obs = C.build_obs(jp, R, p, v + 0.03 * jax.random.normal(jax.random.fold_in(k3, 1), (3,)),
                      omega + 0.02 * jax.random.normal(jax.random.fold_in(k3, 2), (3,)), c1, c2, rem, fin, dists, a)

    # outcome
    clr = clearance(T, p)
    goal = path[-1]
    dxy_goal = jp.sqrt(((p[:2] - goal[:2]) ** 2).sum())
    dxy_start = jp.sqrt(((p[:2] - T["pads"][0, :2]) ** 2).sum())
    success = (fin > 0) & below & (dxy_goal < 0.6) & (speed_pre < 0.5) & (up_z > np.cos(np.deg2rad(12))) & ~impact
    oob = (jp.abs(p[:2]).max() > self._area + 10) | (p[2] - g > 45)
    flipped = up_z < np.cos(np.deg2rad(80))
    bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | jp.isnan(obs).any()
    crash = (clr < 0) | impact | oob | flipped | bad

    # reward
    dt = self.dt
    agl = p[2] - g
    near_pads = (dxy_goal < 2.5) | (dxy_start < 2.5)
    vmax = jp.where(fin > 0, jp.clip(0.35 + 0.5 * jp.sqrt(((p - goal) ** 2).sum()), 0.35, 3.0), 3.0)
    vz_ok = 0.35 + 0.6 * jp.maximum(agl, 0.0)              # allowed sink rate shrinks near the ground
    r = 1.5 * (s_new - s_old)
    r -= dt * 0.6 * jp.minimum(xtrack, 5.0)
    r -= dt * 0.25
    r -= dt * 8.0 * jp.clip((1.5 - clr) / 1.5, 0.0, 1.0) ** 2
    r -= dt * 2.0 * jp.clip(0.7 - agl, 0.0, 0.7) * (1.0 - near_pads)
    r -= dt * 2.0 * jp.minimum(jp.maximum(speed_pre - vmax, 0.0), 3.0) ** 2
    r -= dt * 4.0 * fin * jp.minimum(jp.maximum(-v[2] - vz_ok, 0.0), 3.0) ** 2
    r -= dt * (0.3 * ((a - info["last_act"]) ** 2).sum() + 0.02 * (omega ** 2).sum() + 0.5 * (1.0 - up_z))
    r += jp.where(success, 30.0, 0.0)
    r = jp.where(crash, -25.0, r)
    r = jp.where(bad, 0.0, r)

    obs = jp.where(bad, 0.0, obs)
    data = jax.tree.map(lambda x, y: jp.where(bad, y, x), data, state.data)
    info.update(rng=rng, s=s_new, last_act=a)
    metrics = dict(state.metrics)
    metrics.update(success=success.astype(jp.float32), crash=crash.astype(jp.float32),
                   completion=jp.where(bad, 0.0, (s_new - s_old) / ps[-1]))
    done = (success | crash).astype(jp.float32)
    return mjx_env.State(data, obs, r, done, metrics, info)

  @property
  def xml_path(self) -> str:
    return ""

  @property
  def action_size(self) -> int:
    return 4

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
