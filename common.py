"""Shared definitions for the drone pilot: sensors, action mapping and the observation layout.

Used by both the GPU training env (JAX) and the live demo (NumPy + MuJoCo), so the pilot sees
exactly the same inputs in both. Pass xp=numpy or xp=jax.numpy.
"""
import numpy as np

# ---- timing / physics -------------------------------------------------------------------------
CTRL_DT = 0.02          # pilot runs at 50 Hz
SIM_DT = 0.005          # physics + rate loop at 200 Hz
N_SUB = int(round(CTRL_DT / SIM_DT))
G = 9.81
MOTOR_MAX = 13.0        # N per motor (Skydio X2 model)
DRONE_R = 0.30          # collision radius used for clearance checks (m)
LIDAR_OFFSET = 0.05     # lidar origin above the body origin (m)

# ---- action: collective thrust + body rates (like a real flight controller) -------------------
THRUST_SPAN = 0.8                       # thrust = m g (1 + 0.8 a0)  ->  0.2 g .. 1.8 g
RATE_MAX = np.array([2.5, 2.5, 1.5])    # rad/s for roll, pitch, yaw
KP_RATE = 20.0                          # rate-loop gain (1/s)

# ---- lidar --------------------------------------------------------------------------------------
RING_EL = np.deg2rad([-20.0, 0.0, 20.0])  # three scan rows (body frame)
RING_AZ = 48                              # beams per row (7.5 deg)
RING_POOL = 3                             # min over 3 beams -> 16 sectors per row
RING_RANGE = 15.0
DOWN_TILT = np.deg2rad(25.0)
DOWN_RANGE = 40.0
UP_RANGE = 15.0


def ray_dirs_body():
  """Unit ray directions in the drone body frame: 144 ring beams, 9 down beams, 1 up beam."""
  ring = []
  for r, el in enumerate(RING_EL):
    for k in range(RING_AZ):
      # rows are offset by a third of the beam spacing: a thin vertical trunk between the beams of one row
      # is caught by another row (2.5 deg effective spacing for trunks)
      az = 2 * np.pi * (k + 0.5 + r / len(RING_EL)) / RING_AZ
      ring.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
  down = [[0.0, 0.0, -1.0]]
  for k in range(8):
    a = 2 * np.pi * k / 8
    down.append([np.sin(DOWN_TILT) * np.cos(a), np.sin(DOWN_TILT) * np.sin(a), -np.cos(DOWN_TILT)])
  up = [[0.0, 0.0, 1.0]]
  dirs = np.array(ring + down + up, dtype=np.float32)
  ranges = np.array([RING_RANGE] * len(ring) + [DOWN_RANGE] * len(down) + [UP_RANGE], dtype=np.float32)
  return dirs, ranges


RAY_DIRS, RAY_RANGES = ray_dirs_body()
N_RING = len(RING_EL) * RING_AZ
N_RAYS = len(RAY_DIRS)

# ---- observation ------------------------------------------------------------------------------
OBS_SIZE = 3 + 3 + 3 + 1 + 1 + 3 + 3 + 3 + 9 + N_RING // RING_POOL + 1 + 4   # = 82


def fix_x2(src: str) -> str:
  """Skydio X2 XML -> the version we fly.
  * Propeller spin: in the Menagerie file the yaw torques of rotors 3 and 4 are swapped, which makes
    yaw torque mirror roll torque (the mixer is singular). Diagonal rotors must spin the same way.
  * No air model (drag is handled as wind disturbance during training)."""
  src = src.replace('name="thrust3" site="thrust3" gear="0 0 1 0 0  .0201"', 'name="thrust3" site="thrust3" gear="0 0 1 0 0 -.0201"')
  src = src.replace('name="thrust4" site="thrust4" gear="0 0 1 0 0 -.0201"', 'name="thrust4" site="thrust4" gear="0 0 1 0 0  .0201"')
  return src.replace('density="1.225" viscosity="1.8e-5"', 'density="0" viscosity="0"')


def drone_constants(m):
  """Mass, inertia (body frame) and motor mixer, read from the MuJoCo model."""
  import mujoco
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


LIDAR_GROUPS = np.array([1, 1, 0, 0, 0, 0], np.uint8)   # terrain, trunks, rocks, pads (0) + leaves (1)


def quat_to_mat(xp, q):
  w, x, y, z = q[0], q[1], q[2], q[3]
  return xp.stack([
      xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
      xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
      xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
  ])


def action_to_command(xp, a, mass):
  """Pilot output (-1..1) -> collective thrust (N) and desired body rates (rad/s)."""
  a = xp.clip(a, -1.0, 1.0)
  thrust = mass * G * (1.0 + THRUST_SPAN * a[0])
  rates = a[1:4] * xp.asarray(RATE_MAX)
  return thrust, rates


def rate_controller(xp, thrust, rates_des, omega, J, Ainv):
  """Body-rate PD loop + motor mixer. Returns the 4 motor thrusts (N), unclipped."""
  tau = J @ (KP_RATE * (rates_des - omega)) + xp.cross(omega, J @ omega)
  return Ainv @ xp.concatenate([xp.reshape(thrust, (1,)), tau])


def _clip_norm(xp, v, m):
  n = xp.sqrt((v * v).sum() + 1e-12)
  return v * xp.minimum(1.0, m / n) / m


def build_obs(xp, R, p, v, omega, carrot1, carrot2, remaining, final, dists, last_action):
  """R body->world rotation, p position, v world velocity, omega body rates, dists = N_RAYS lidar ranges."""
  yaw = xp.arctan2(R[1, 0], R[0, 0])
  c, s = xp.cos(yaw), xp.sin(yaw)

  def to_heading(vec):
    return xp.stack([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1], vec[2]])

  rel1 = to_heading(carrot1 - p)
  rel2 = to_heading(carrot2 - p)
  ring = dists[:N_RING].reshape(len(RING_EL), RING_AZ // RING_POOL, RING_POOL).min(-1).reshape(-1)
  down = dists[N_RING:N_RING + 9]
  up = dists[N_RING + 9:N_RING + 10]
  return xp.concatenate([
      _clip_norm(xp, rel1, 6.0),
      xp.clip(rel1, -2.0, 2.0) / 2.0,
      _clip_norm(xp, rel2, 12.0),
      xp.reshape(xp.minimum(remaining, 20.0) / 20.0, (1,)),
      xp.reshape(final, (1,)),
      xp.clip(to_heading(v) / 4.0, -2.0, 2.0),
      -R[2, :],                                   # gravity direction in body frame
      xp.clip(omega / 4.0, -2.0, 2.0),
      xp.minimum(down, DOWN_RANGE) / DOWN_RANGE,
      1.0 - xp.minimum(ring, RING_RANGE) / RING_RANGE,
      1.0 - xp.minimum(up, UP_RANGE) / UP_RANGE,
      last_action,
  ])


# ---- route following (carrot on a polyline) ---------------------------------------------------
def path_point(xp, path, ps, s):
  s = xp.clip(s, 0.0, ps[-1])
  i = xp.clip(xp.searchsorted(ps, s, side="right") - 1, 0, len(ps) - 2)
  t = (s - ps[i]) / xp.maximum(ps[i + 1] - ps[i], 1e-6)
  return path[i] + t * (path[i + 1] - path[i])


def project(xp, path, ps, p, s_old, back=1.0, ahead=6.0):
  """Advance progress s along the route; returns (s_new, cross-track distance)."""
  a, b = path[:-1], path[1:]
  ab = b - a
  l2 = (ab * ab).sum(-1) + 1e-9
  t = xp.clip(((p - a) * ab).sum(-1) / l2, 0.0, 1.0)
  q = a + t[:, None] * ab
  d = xp.sqrt(((p - q) ** 2).sum(-1))
  s_seg = ps[:-1] + t * xp.sqrt(l2)
  ok = (s_seg >= s_old - back) & (s_seg <= s_old + ahead)
  dm = xp.where(ok, d, 1e9)
  j = xp.argmin(dm)
  s_new = xp.where(ok.any(), xp.maximum(s_old, s_seg[j]), s_old)
  return s_new, xp.where(ok.any(), d[j], d.min())


def carrots(xp, path, ps, s, look):
  total = ps[-1]
  remaining = total - s
  final = (remaining <= look).astype(xp.float32) if hasattr(remaining, "astype") else float(remaining <= look)
  return path_point(xp, path, ps, s + look), path_point(xp, path, ps, s + 2 * look), remaining, final
