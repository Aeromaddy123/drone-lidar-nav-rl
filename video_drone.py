"""LinkedIn video (1080x1350, 60 fps): a drone maps a forest with lidar, learns to fly in random worlds,
then flies itself to checkpoint 3 and lands. Everything is rendered from real recordings.

  video_drone.py --start F --end F --out part.mp4     (frame range, for parallel parts)
  video_drone.py --still SECONDS --out frame.png
"""
import argparse, json, math, os, pathlib, pickle, subprocess, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
W, H, FPS = 1080, 1350, 60
INK, MUTED, ACCENT = (242, 240, 234), (190, 192, 196), (242, 115, 26)
CYAN, ORANGE, RED, GREEN = (80, 215, 255), (255, 165, 60), (255, 80, 85), (120, 225, 130)

# ----------------------------------------------------------------------------- timeline (seconds)
T_INTRO, T_MAP, T_TRAIN, T_GO, T_END = (0.0, 3.6), (3.6, 10.6), (10.6, 17.8), (17.8, 32.6), (32.6, 36.0)
DURATION = T_END[1]


def font(size, bold=False, mono=False):
  cands = (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/System/Library/Fonts/Menlo.ttc"] if mono else
           ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"])
  for f in cands:
    if os.path.exists(f):
      return ImageFont.truetype(f, size, index=1 if (bold and f.endswith("HelveticaNeue.ttc")) else 0)
  return ImageFont.load_default()


F_STEP, F_SUB, F_SMALL, F_BIG, F_TITLE, F_MONO, F_TAG = (font(27, True), font(31), font(22, True), font(54, True),
                                                       font(60, True), font(34, mono=True), font(26, True))

clip01 = lambda x: min(1.0, max(0.0, x))
def smooth(x):
  x = clip01(x); return x * x * (3 - 2 * x)
def ease(x):
  x = clip01(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def ramp(t, a, b): return smooth((t - a) / (b - a)) if b > a else float(t >= a)
def lerp(a, b, u): return a + (b - a) * u
def window(t, a, b, fade=0.35): return ramp(t, a, a + fade) * (1 - ramp(t, b - fade, b))


# ----------------------------------------------------------------------------- sky
class Sky:
  YAW = 34.0     # turn the sky so the low sun sits at ~75 deg (side light for our main camera directions)

  def __init__(self, path, width=4096):
    im = Image.open(path).convert("RGB").resize((width, width // 2), Image.LANCZOS)
    self.img = np.asarray(im).astype(np.float32)
    self.H_, self.W_ = self.img.shape[:2]
    lum = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(8))).astype(np.float32)
    v, u = np.unravel_index(np.argmax(lum), lum.shape)
    self.sun_lon = (u / self.W_ - 0.5) * 2 * math.pi + math.radians(self.YAW)
    self.sun_lat = (0.5 - v / self.H_) * math.pi

  def horizon_rgb(self, az_deg, spread=25):
    lon = math.radians(az_deg - self.YAW)
    us = ((np.linspace(lon - math.radians(spread), lon + math.radians(spread), 40) / (2 * math.pi) + 0.5) % 1.0) * (self.W_ - 1)
    v0, v1 = int(self.H_ * 0.492), int(self.H_ * 0.5)
    return self.img[v0:v1][:, us.astype(int)].reshape(-1, 3).mean(0) / 255.0

  def sun_vector(self, min_elev_deg=20):
    lat = max(self.sun_lat, math.radians(min_elev_deg))
    return np.array([math.cos(lat) * math.cos(self.sun_lon), math.cos(lat) * math.sin(self.sun_lon), math.sin(lat)])

  def render(self, cam, w, h, fovy=45.0):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); up = np.cross(r, f)
    t = math.tan(math.radians(fovy) / 2)
    xs = (np.arange(w) + 0.5 - w / 2) / (h / 2) * t
    ys = (h / 2 - np.arange(h) - 0.5) / (h / 2) * t
    d = f[None, None] + xs[None, :, None] * r[None, None] + ys[:, None, None] * up[None, None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 1], d[..., 0]) - math.radians(self.YAW)
    lat = np.arcsin(np.clip(d[..., 2], -1, 1))
    u = ((lon / (2 * math.pi) + 0.5) % 1.0) * (self.W_ - 1)
    v = np.clip(0.5 - lat / math.pi, 0, 1) * (self.H_ - 1)
    return self.img[v.astype(int), u.astype(int)]


SKY = None


def grade(rgb):
  x = np.clip(rgb / 255.0 * 1.12, 0, 1)
  x = x ** 0.92                                    # lift the shadows a little
  x = x * np.array([1.04, 1.0, 0.94])
  x = np.clip(x, 0, 1)
  x = x - 0.07 * (x - x ** 2) * (2 * x - 1)
  return np.clip(x * 255, 0, 255)


def make_cam(lookat, dist, az, el):
  c = mujoco.MjvCamera()
  c.lookat[:] = lookat; c.distance = dist; c.azimuth = az; c.elevation = el
  return c


def project(cam, X, w=W, h=H, fovy=45.0):
  az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
  f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
  r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); up = np.cross(r, f)
  pos = np.array(cam.lookat) - cam.distance * f
  d = np.asarray(X) - pos
  z = d @ f
  k = (h / 2) / math.tan(math.radians(fovy) / 2)
  return w / 2 + (d @ r) / z * k, h / 2 - (d @ up) / z * k, z


# ----------------------------------------------------------------------------- scene helpers
def add_geom(scn, gtype, size, pos, rgba, mat=None, emission=0.0):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                      (np.eye(3) if mat is None else mat).reshape(-1), np.asarray(rgba, np.float32))
  g.emission = emission
  scn.ngeom += 1


def add_line(scn, a, b, rgba, width=2.0):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
  scn.ngeom += 1


def add_tube(scn, a, b, r, rgba, emission=0.6):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(a, float), np.asarray(b, float))
  g.emission = emission
  scn.ngeom += 1


def height_rgb(z, lo=-6.0, hi=18.0):
  t = np.clip((np.asarray(z) - lo) / (hi - lo), 0, 1)[..., None]
  stops = np.array([[0.20, 0.30, 0.95], [0.10, 0.80, 0.95], [0.35, 0.95, 0.45], [1.0, 0.85, 0.20], [1.0, 0.40, 0.15]])
  x = t * (len(stops) - 1)
  i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)[..., 0]
  u = x - i[..., None]
  return stops[i] * (1 - u) + stops[i + 1] * u


class Renderer:
  """MuJoCo render + sky behind + warm grade."""

  def __init__(self, xml_path, w, h, shadowclip=0.6, fog_az=95.0, max_geom=40000):
    global SKY
    if SKY is None:
      SKY = Sky(ROOT / "assets" / "sky_qwantani_sunset_puresky.jpg")
    self.m = mujoco.MjModel.from_xml_path(str(xml_path))
    sun = SKY.sun_vector(24)
    for nm in ("sun",):
      lid = self.m.light(nm).id
      self.m.light_dir[lid] = -sun
    self.sun_id = self.m.light("sun").id
    fog = SKY.horizon_rgb(fog_az)
    self.m.vis.rgba.fog[:3] = fog * 0.95
    self.m.vis.map.shadowclip = shadowclip
    self.m.vis.global_.offwidth, self.m.vis.global_.offheight = max(w, 1080), max(h, 1350)
    self.d = mujoco.MjData(self.m)
    self.w, self.h = w, h
    self.r = mujoco.Renderer(self.m, h, w, max_geom=max_geom)
    self.far = self.m.vis.map.zfar * self.m.stat.extent
    self.sun_dir = -sun

  def aim_sun(self, xyz):
    self.m.light_pos[self.sun_id] = np.asarray(xyz) - self.sun_dir * 60.0

  def begin(self, cam):
    self.r.update_scene(self.d, cam)
    return self.r.scene

  def finish(self, cam, sky=True):
    rgb = self.r.render().astype(np.float32)
    if sky:
      self.r.enable_depth_rendering(); depth = self.r.render(); self.r.disable_depth_rendering()
      bg = depth >= self.far * 0.995
      if bg.any():
        rgb[bg] = SKY.render(cam, self.w, self.h)[bg]
    return grade(rgb)


# ----------------------------------------------------------------------------- data
class Demo:
  def __init__(self, rec_path):
    rec = pickle.load(open(rec_path, "rb"))
    self.frames = rec["frames"]; self.scans = rec["scans"]; self.segments = rec["segments"]
    self.events = rec["events"]; self.pads = rec["pads"]
    self.t = np.array([f["t"] for f in self.frames])
    self.q = np.array([f["qpos"] for f in self.frames])
    self.seg = {s["cmd"]: s for s in self.segments}
    # first-seen map at 1 m (for the painting effect)
    res, half = 1.0, 60.0
    n = int(2 * half / res)
    self.cell_t = np.full((n, n), np.inf); self.cell_z = np.full((n, n), -np.inf)
    for ts, pts in self.scans:
      if len(pts) == 0:
        continue
      i = np.clip(((pts[:, 0] + half) / res).astype(int), 0, n - 1); j = np.clip(((pts[:, 1] + half) / res).astype(int), 0, n - 1)
      np.minimum.at(self.cell_t, (j, i), ts)
      np.maximum.at(self.cell_z, (j, i), pts[:, 2])
    J, I = np.nonzero(np.isfinite(self.cell_t))
    self.cells = np.stack([-half + (I + 0.5) * res, -half + (J + 0.5) * res, self.cell_z[J, I], self.cell_t[J, I]], -1)
    self.cell_rgb = height_rgb(self.cells[:, 2])
    self.found = [(t, m.split("checkpoint ")[1].split(" ")[0]) for t, m in self.events if m.startswith("Camera found")]

  def pose(self, t):
    k = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
    u = clip01((t - self.t[k - 1]) / max(self.t[k] - self.t[k - 1], 1e-6))
    q = self.q[k - 1] * (1 - u) + self.q[k] * u
    q[3:7] /= np.linalg.norm(q[3:7])
    return q, self.frames[k]

  def scan_at(self, t):
    ts = np.array([s[0] for s in self.scans])
    k = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 1))
    return self.scans[k][1]


def set_drone(rd, q):
  rd.d.qpos[:7] = q
  rd.d.qvel[:] = 0
  mujoco.mj_kinematics(rd.m, rd.d)
  mujoco.mj_camlight(rd.m, rd.d)


# ----------------------------------------------------------------------------- 2D overlays
def overlay():
  return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def shade_bands(img, top=300, bottom=480, a=1.0):
  f = np.asarray(img).astype(np.float32)
  if top:
    band = (1 - np.linspace(0, 1, top)) ** 1.6
    f[:top, :, :3] *= (1 - 0.62 * a * band)[:, None, None]
  if bottom:
    band = np.linspace(0, 1, bottom) ** 1.6
    f[H - bottom:, :, :3] *= (1 - 0.7 * a * band)[:, None, None]
  return Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))


def step_label(dr, num, title, sub, a, y=H - 200):
  if a <= 0:
    return
  x = 64
  dr.rectangle([x, y + 8, x + 36, y + 11], fill=ACCENT + (int(255 * a),))
  dr.text((x + 52, y - 8), f"{num}   {title.upper()}", font=F_STEP, fill=INK + (int(255 * a),))
  if sub:
    dr.text((x, y + 36), sub, font=F_SUB, fill=MUTED + (int(240 * a),))


def stats(dr, cols, a, y=58):
  if a <= 0:
    return
  x = 64
  for i, (k, v, accent) in enumerate(cols):
    dr.text((x, y), k, font=F_SMALL, fill=MUTED + (int(255 * a),))
    dr.text((x, y + 26), v, font=F_BIG, fill=(ACCENT if accent else INK) + (int(255 * a),))
    x += 330


def vignette():
  yy, xx = np.mgrid[0:H, 0:W]
  r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
  return (1.0 - 0.3 * np.clip(r - 0.6, 0, 1) ** 1.5)[..., None].astype(np.float32)


VIG = None


def finalize(rgb, fade=1.0):
  global VIG
  if VIG is None:
    VIG = vignette()
  return Image.fromarray(np.clip(rgb * VIG * fade, 0, 255).astype(np.uint8)).convert("RGBA")


def minimap(demo, t_sim, path=None, size=300):
  n = 120
  img = np.full((n, n, 3), 18.0)
  shown = demo.cells[:, 3] <= t_sim
  c = demo.cells[shown]
  i = np.clip(((c[:, 0] + 60)).astype(int), 0, n - 1); j = np.clip(((c[:, 1] + 60)).astype(int), 0, n - 1)
  img[j, i] = demo.cell_rgb[shown] * 255 * 0.85
  img = img[::-1]
  im = Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.NEAREST).convert("RGBA")
  dr = ImageDraw.Draw(im)
  px = lambda x, y: ((x + 60) / 120 * size, (60 - y) / 120 * size)
  if path is not None:
    dr.line([px(p[0], p[1]) for p in path], fill=CYAN + (255,), width=3)
  for t, name in demo.found:
    if t <= t_sim:
      p = demo.pads[name]; x, y = px(p["x"], p["y"])
      dr.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 255, 255, 255), width=2)
      dr.text((x + 11, y - 13), name, font=F_TAG, fill=(255, 255, 255, 255))
  q, _ = demo.pose(t_sim)
  x, y = px(q[0], q[1])
  dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=RED + (255,))
  frame = Image.new("RGBA", (size + 4, size + 4), (255, 255, 255, 90))
  frame.paste(im, (2, 2))
  return frame


# ----------------------------------------------------------------------------- the film
class Film:
  def __init__(self):
    self.demo = Demo(ROOT / "rec" / "demo.pkl")
    self.forest = Renderer(ROOT / "scene" / "scene.xml", W, H, shadowclip=0.45, max_geom=40000)
    self.wide = None
    self.meta = json.loads((ROOT / "scene" / "world.json").read_text())
    segm, seg3 = self.demo.seg["M"], self.demo.seg["3"]
    self.map_t0, self.map_t1 = segm["t0"], segm["t1"]
    self.go_t0, self.go_t1 = seg3["t0"], seg3["t1"]
    self.route = seg3["path"]; self.levels = seg3["levels"]
    self.land_t = next((t for t, m in self.demo.events if m.startswith("Landed on checkpoint 3")), self.go_t1 - 2)
    self.land_msg = next((m for t, m in self.demo.events if m.startswith("Landed on checkpoint 3")), "")
    self._go_warp = self._make_go_warp()
    self._chase = self._make_chase()
    self.train = None

  # ---- time warps
  def map_time(self, tau):
    u = (tau - T_MAP[0]) / (T_MAP[1] - T_MAP[0])
    return lerp(self.map_t0, self.map_t1 - 1.5, ease(clip01(u)) * 0.15 + clip01(u) * 0.85)

  def _make_go_warp(self):
    """video time -> sim time for the mission: command typed, route drawn, fast cruise, real-time landing."""
    a = T_GO[0] + 2.4                      # flight starts
    b = T_GO[1] - 0.2
    taus = np.linspace(a, b, 2000)
    land_slow = self.land_t - 2.6           # sim time when we slow to 1x
    # speed profile: high during cruise, 1x for the landing; solve for the cruise speed that fits
    def total(k):
      sim = self.go_t0
      for i in range(1, len(taus)):
        sp = 1.0 if sim >= land_slow else k
        sim += sp * (taus[i] - taus[i - 1])
      return sim
    lo, hi = 0.5, 8.0
    target = self.land_t + 1.6
    for _ in range(40):
      mid = 0.5 * (lo + hi)
      lo, hi = (mid, hi) if total(mid) < target else (lo, mid)
    k = 0.5 * (lo + hi)
    sims = [self.go_t0]
    for i in range(1, len(taus)):
      sp = 1.0 if sims[-1] >= land_slow else k
      sims.append(sims[-1] + sp * (taus[i] - taus[i - 1]))
    self.go_speed = k
    return taus, np.array(sims)

  def go_time(self, tau):
    taus, sims = self._go_warp
    if tau < taus[0]:
      return self.go_t0
    return float(np.interp(tau, taus, sims))

  def _make_chase(self):
    """Smooth chase camera for the whole mission (depends only on the recording, so parts agree)."""
    ts = np.arange(self.go_t0, self.go_t1, 0.02)
    P = np.array([self.demo.pose(t)[0][:3] for t in ts])
    V = np.gradient(P, 0.02, axis=0)
    az = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
    az = np.degrees(np.unwrap(np.radians(az)))
    sp = np.linalg.norm(V[:, :2], axis=1)
    # hold heading when slow
    for i in range(1, len(az)):
      if sp[i] < 0.6:
        az[i] = az[i - 1]
    k = np.exp(-0.02 / 1.2)
    sm = az.copy()
    for i in range(1, len(sm)):
      sm[i] = k * sm[i - 1] + (1 - k) * az[i]
    for i in range(len(sm) - 2, -1, -1):
      sm[i] = k * sm[i + 1] + (1 - k) * sm[i]
    Ps = P.copy()
    for i in range(1, len(Ps)):
      Ps[i] = 0.85 * Ps[i - 1] + 0.15 * P[i]
    return ts, sm, Ps

  def chase_cam(self, t_sim, dist=7.5, el=-16.0, side=18.0):
    ts, az, Ps = self._chase
    a = float(np.interp(t_sim, ts, az))
    p = np.array([np.interp(t_sim, ts, Ps[:, k]) for k in range(3)])
    return make_cam(p + [0, 0, 0.6], dist, a + side, el)

  # ---- frame
  def frame(self, tau):
    if tau < T_INTRO[1]:
      img = self.intro(tau)
    elif tau < T_MAP[1]:
      img = self.mapping(tau)
    elif tau < T_TRAIN[1]:
      img = self.training(tau)
    elif tau < T_GO[1]:
      img = self.mission(tau)
    else:
      img = self.ending(tau)
    return img.convert("RGB")

  # ---- 0: intro
  def intro(self, tau):
    u = ease(tau / T_INTRO[1])
    home = self.demo.pads["H"]
    look = np.array([lerp(5, home["x"], u ** 0.7), lerp(5, home["y"], u ** 0.7), lerp(4, home["z"] + 0.6, u)])
    cam = make_cam(look, math.exp(lerp(math.log(140), math.log(10), u)), lerp(30, 58, u), lerp(-38, -14, u))
    q, _ = self.demo.pose(self.map_t0)
    set_drone(self.forest, q)
    self.forest.aim_sun(look)
    scn = self.forest.begin(cam)
    img = finalize(self.forest.finish(cam), fade=ramp(tau, 0, 0.5))
    img = shade_bands(img, top=420, bottom=0, a=1.0)
    ov = overlay(); dr = ImageDraw.Draw(ov)
    a = ramp(tau, 0.3, 0.9) * (1 - ramp(tau, T_INTRO[1] - 0.4, T_INTRO[1]))
    dr.text((64, 70), "A drone that maps a forest", font=F_TITLE, fill=INK + (int(255 * a),))
    dr.text((64, 142), "and flies itself to any checkpoint", font=F_TITLE, fill=INK + (int(255 * a),))
    dr.text((64, 228), "Hilly ground  ·  183 trees  ·  5 landing pads  ·  lidar + camera", font=F_SUB, fill=MUTED + (int(240 * a),))
    return Image.alpha_composite(img.convert("RGBA"), ov)

  # ---- 1: mapping
  def mapping(self, tau):
    t = self.map_time(tau)
    q, fr = self.demo.pose(t)
    u = clip01((tau - T_MAP[0]) / (T_MAP[1] - T_MAP[0]))
    look = np.array([lerp(q[0], 0, 0.55), lerp(q[1], 0, 0.55), 2.0])
    cam = make_cam(look, lerp(112, 124, u), lerp(58, 82, u), lerp(-44, -50, u))
    if tau < T_MAP[0] + 0.7:     # blend in from the intro camera
      w = ease((tau - T_MAP[0]) / 0.7)
      home = self.demo.pads["H"]
      c0 = np.array([home["x"], home["y"], home["z"] + 0.6])
      cam = make_cam(lerp(c0, look, w), math.exp(lerp(math.log(10), math.log(112), w)), 58, lerp(-14, -44, w))
    set_drone(self.forest, q)
    self.forest.aim_sun(np.array(cam.lookat))
    scn = self.forest.begin(cam)
    # painted voxel map
    tv = (tau - T_MAP[0])
    shown = self.demo.cells[:, 3] <= t
    cells, rgb = self.demo.cells[shown], self.demo.cell_rgb[shown]
    age = (t - cells[:, 3]) / max(self.map_time(tau + 1 / FPS) - t, 1e-3) / FPS     # video seconds since first seen
    for c, col, ag in zip(cells, rgb, age):
      fresh = clip01(1 - ag / 0.5)
      add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.47, 0.47, 0.06 + 0.25 * fresh], [c[0], c[1], c[2] + 0.15],
               list(np.clip(col * (1 + 0.6 * fresh), 0, 1)) + [0.92], emission=0.25 + 0.5 * fresh)
    # lidar fan from the drone to the latest scan
    pts = self.demo.scan_at(t)
    if len(pts):
      o = q[:3] - [0, 0, 0.05]
      for p in pts[:: max(1, len(pts) // 70)]:
        add_line(scn, o, p, (1.0, 0.3, 0.3, 0.5), 1.5)
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.9, 0, 0], q[:3], (1, 0.95, 0.85, 0.35), emission=1.0)
    img = finalize(self.forest.finish(cam))
    img = shade_bands(img, top=0, bottom=430)
    ov = overlay(); dr = ImageDraw.Draw(ov)
    a = window(tau, T_MAP[0], T_MAP[1])
    step_label(dr, "1", "Map the area", "Lidar paints the ground and trees. The camera spots the pads.", a)
    stats(dr, [("LIDAR POINTS", f"{self._points_at(t):,}", False), ("PADS FOUND", f"{sum(1 for x, _ in self.demo.found if x <= t)}/5", True),
               ("SPEED", f"{(self.map_t1 - self.map_t0) / (T_MAP[1] - T_MAP[0]):.0f}x", False)], a)
    # pad tags
    for tf, name in self.demo.found:
      if tf <= t:
        p = self.demo.pads[name]
        x, y, z = project(cam, [p["x"], p["y"], p["z"] + 1.0])
        k = ramp(t, tf, tf + (self.map_t1 - self.map_t0) * 0.03)
        if z > 0:
          dr.rounded_rectangle([x - 22, y - 58, x + 22, y - 16], 8, fill=(20, 22, 26, int(200 * k * a)), outline=INK + (int(255 * k * a),), width=2)
          dr.text((x, y - 37), name, font=F_TAG, fill=INK + (int(255 * k * a),), anchor="mm")
          dr.line([x, y - 16, x, y], fill=INK + (int(200 * k * a),), width=2)
    img = Image.alpha_composite(img, ov)
    mm = minimap(self.demo, t)
    mm.putalpha(Image.eval(mm.getchannel("A"), lambda v: int(v * a)))
    img.alpha_composite(mm, (W - mm.width - 48, 56 + 130))
    return img

  def _points_at(self, t):
    if not hasattr(self, "_pts_cum"):
      ts = np.array([s[0] for s in self.demo.scans]); n = np.cumsum([len(s[1]) for s in self.demo.scans])
      self._pts_cum = (ts, n)
    ts, n = self._pts_cum
    k = np.searchsorted(ts, t, side="right")
    return int(n[k - 1]) if k else 0

  # ---- 2: training montage
  def training(self, tau):
    if self.train is None:
      from montage import Montage
      self.train = Montage(ROOT)
    return self.train.frame(tau, T_TRAIN)

  # ---- 3: mission
  def mission(self, tau):
    t = self.go_time(tau)
    q, fr = self.demo.pose(t)
    cam = self.chase_cam(t)
    if tau < T_GO[0] + 2.4:     # opening: a higher view of the drone on the home pad and the route ahead
      w = ease((tau - T_GO[0]) / 2.4)
      c1 = self.chase_cam(self.go_t0, dist=26, el=-32, side=30)
      c2 = self.chase_cam(self.go_t0)
      cam = make_cam(lerp(np.array(c1.lookat), np.array(c2.lookat), w), math.exp(lerp(math.log(c1.distance), math.log(c2.distance), w)),
                     lerp(c1.azimuth, c2.azimuth, w), lerp(c1.elevation, c2.elevation, w))
    set_drone(self.forest, q)
    self.forest.aim_sun(q[:3])
    scn = self.forest.begin(cam)
    # route (drawn progressively after the command)
    draw_u = ramp(tau, T_GO[0] + 1.3, T_GO[0] + 2.3)
    path, lv = self.route, self.levels
    nshow = int(len(path) * draw_u)
    s_now = fr["s"] if fr["cmd"] == "3" else 0.0
    seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    landed_now = fr["state"] == "LANDED" and fr["cmd"] == "3"
    for i in range(max(0, nshow - 1)):
      if seg[i + 1] < s_now - 1.0 or landed_now:
        continue
      col = (0.3, 0.85, 1.0, 0.85) if lv[min(i + 1, len(lv) - 1)] == 0 else (1.0, 0.65, 0.2, 0.85)
      add_tube(scn, path[i], path[i + 1], 0.06, col, emission=0.8)
    # what the pilot's lidar touches right now
    if fr.get("hits") is not None and fr["cmd"] == "3":
      for h in fr["hits"][::2]:
        add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.06, 0, 0], h, (1.0, 0.25, 0.3, 1.0), emission=0.8)
    if fr.get("carrot") is not None and fr["cmd"] == "3" and fr["state"] != "LANDED":
      add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.16, 0, 0], fr["carrot"], (1.0, 0.95, 0.3, 0.9), emission=1.0)
    img = finalize(self.forest.finish(cam))
    img = shade_bands(img, top=300, bottom=440)
    ov = overlay(); dr = ImageDraw.Draw(ov)
    a = window(tau, T_GO[0], T_GO[1])
    # typed command
    typed = "go to 3"
    k = int(len(typed) * clip01((tau - T_GO[0] - 0.25) / 0.8))
    ca = (1 - ramp(tau, T_GO[0] + 3.2, T_GO[0] + 3.8)) * a
    if ca > 0:
      bx, by = 64, H - 330
      dr.rounded_rectangle([bx, by, bx + 420, by + 64], 10, fill=(12, 14, 18, int(215 * ca)), outline=(255, 255, 255, int(60 * ca)))
      cur = "_" if (int(tau * 3) % 2 == 0 and k < len(typed)) else ""
      dr.text((bx + 20, by + 12), "> " + typed[:k] + cur, font=F_MONO, fill=(120, 255, 160, int(255 * ca)))
    # live readout
    dists = None
    agl = q[2] - self._ground(q[0], q[1])
    v = np.linalg.norm(fr.get("v", np.zeros(3)))
    nearest = "-"
    if fr.get("hits") is not None and len(fr["hits"]) and fr["cmd"] == "3":
      hz = fr["hits"]
      above = np.array([h[2] - self._ground(h[0], h[1]) for h in hz]) > 0.5      # trees and rocks, not the ground
      if above.any():
        nearest = f"{np.linalg.norm(hz[above] - q[:3], axis=1).min():.1f} m"
    stats(dr, [("HEIGHT", f"{max(agl, 0):.1f} m", False), ("SPEED", f"{v:.1f} m/s", False), ("NEAREST OBSTACLE", nearest, True)],
          a * ramp(tau, T_GO[0] + 2.0, T_GO[0] + 2.6))
    landed = t >= self.land_t
    if not landed:
      speed_note = f"{self.go_speed:.1f}x speed" if t < self.land_t - 2.6 else "real time"
      step_label(dr, "3", "Fly to checkpoint 3", f"Planned on the lidar map. The learned pilot flies it.  ·  {speed_note}", a)
    else:
      la = ramp(t, self.land_t, self.land_t + 0.5) * a
      cm = self.land_msg.split("(")[1].split(" from")[0] if "(" in self.land_msg else ""
      step_label(dr, "3", "Landed on checkpoint 3", f"{cm} from the centre of the pad", la)
    img = Image.alpha_composite(img, ov)
    mm = minimap(self.demo, self.map_t1 + 1, path=self.route if draw_u > 0 else None, size=260)
    q2, _ = self.demo.pose(t)
    mmd = ImageDraw.Draw(mm)
    x, y = (q2[0] + 60) / 120 * 260 + 2, (60 - q2[1]) / 120 * 260 + 2
    mmd.ellipse([x - 7, y - 7, x + 7, y + 7], fill=RED + (255,), outline=(255, 255, 255, 255))
    mm.putalpha(Image.eval(mm.getchannel("A"), lambda vv: int(vv * a * ramp(tau, T_GO[0] + 1.2, T_GO[0] + 1.8))))
    img.alpha_composite(mm, (W - mm.width - 48, 220))
    return img

  def _ground(self, x, y):
    if not hasattr(self, "_H"):
      self._H = np.load(ROOT / "scene" / "heights.npy")
    i = int(np.clip((x + 60) / 0.25, 0, 480)); j = int(np.clip((y + 60) / 0.25, 0, 480))
    return float(self._H[j, i])

  # ---- 4: ending
  def ending(self, tau):
    u = ease((tau - T_END[0]) / (T_END[1] - T_END[0]))
    p = self.demo.pads["3"]
    q, _ = self.demo.pose(self.go_t1 - 0.05)
    cam = make_cam([p["x"], p["y"], p["z"] + 1.0], lerp(9, 34, u), lerp(self.chase_cam(self.go_t1 - 0.1).azimuth, self.chase_cam(self.go_t1 - 0.1).azimuth + 40, u), lerp(-18, -38, u))
    set_drone(self.forest, q)
    self.forest.aim_sun(q[:3])
    self.forest.begin(cam)
    img = finalize(self.forest.finish(cam), fade=1 - ramp(tau, T_END[1] - 0.5, T_END[1]))
    img = shade_bands(img, top=0, bottom=620)
    ov = overlay(); dr = ImageDraw.Draw(ov)
    a = ramp(tau, T_END[0] + 0.2, T_END[0] + 0.7) * (1 - ramp(tau, T_END[1] - 0.5, T_END[1]))
    lines = [("1", "Lidar map", f"{self.demo.scans and self._points_at(self.map_t1):,} points, all 5 pads found by the camera"),
             ("2", "Route planner", "between the trees, or over the canopy when it must"),
             ("3", "Learned pilot", self.train_note())]
    y = H - 470
    for i, (n_, t_, s_) in enumerate(lines):
      ai = a * ramp(tau, T_END[0] + 0.2 + 0.25 * i, T_END[0] + 0.7 + 0.25 * i)
      dr.text((64, y), f"{n_}   {t_.upper()}", font=F_STEP, fill=INK + (int(255 * ai),))
      dr.text((112, y + 38), s_, font=F_SUB, fill=MUTED + (int(240 * ai),))
      y += 110
    return Image.alpha_composite(img, ov)

  def train_note(self):
    try:
      info = json.loads((ROOT / "rec" / "train_info.json").read_text())
      return info["note"]
    except Exception:
      return "reinforcement learning (PPO) in thousands of random forests"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
  ap.add_argument("--still", type=float, default=None); ap.add_argument("--out", required=True)
  a = ap.parse_args()
  film = Film()
  if a.still is not None:
    film.frame(a.still).save(a.out); print("saved", a.out); return
  end = a.end if a.end is not None else int(DURATION * FPS)
  cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", a.out]
  ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
  for f in range(a.start, end):
    ff.stdin.write(np.asarray(film.frame(f / FPS)).tobytes())
    if f % 60 == 0:
      print("frame", f, flush=True)
  ff.stdin.close(); ff.wait()
  print("done", a.out)


if __name__ == "__main__":
  main()
