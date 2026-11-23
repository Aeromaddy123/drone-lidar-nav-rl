"""Builds the fixed demo world: 120 x 120 m of hills, a forest with clearings, rocks, a home pad and 5 numbered
landing pads, plus the Skydio X2. Writes scene/ (scene.xml, terrain.bin, textures) and scene/world.json.

Run:  python world_demo.py [--x2 path/to/menagerie/skydio_x2]
"""
import argparse, json, pathlib, shutil, struct

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import common as C

HALF = 60.0                 # world is [-60, 60]^2 m
RES = 0.25                  # hfield spacing (m)
N = int(2 * HALF / RES) + 1  # 481
PAD_R = 1.2                 # flat radius (same as training)
PAD_SIZE = 1.15             # half side of the visible pad (m)

PADS = {                    # name: (x, y, colour rgb, where)
    "H": (-44.0, -42.0, (240, 240, 235), "home meadow"),
    "1": (27.0, 33.0, (226, 47, 47), "hill top"),
    "2": (-33.0, 27.0, (255, 196, 20), "valley by the forest"),
    "3": (24.0, -20.0, (0, 190, 220), "forest clearing"),
    "4": (-4.0, 4.0, (45, 95, 235), "slope between tree clumps"),
    "5": (47.0, -47.0, (170, 70, 220), "far corner, beyond the forest"),
}


def gauss(x, y, cx, cy, s, a):
  return a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))


def terrain_raw(x, y):
  h = (2.2 * np.sin(0.065 * x + 0.3) * np.cos(0.05 * y - 0.4) + 1.3 * np.sin(0.105 * x - 0.085 * y + 1.2)
       + 0.6 * np.sin(0.21 * x + 0.17 * y + 0.5) + 0.25 * np.sin(0.47 * x - 0.36 * y + 2.0)
       + 0.12 * np.sin(0.9 * x + 0.7 * y))
  h += gauss(x, y, 27, 33, 13, 5.0)        # hill for pad 1
  h += gauss(x, y, -33, 27, 16, -3.0)      # valley for pad 2
  h += gauss(x, y, 5, 40, 10, 3.0) + gauss(x, y, -50, -5, 12, 2.5) + gauss(x, y, 40, -5, 9, 2.0)
  return h


def smoothstep(x):
  x = np.clip(x, 0, 1)
  return x * x * (3 - 2 * x)


def build_heights():
  xs = np.linspace(-HALF, HALF, N)
  X, Y = np.meshgrid(xs, xs)               # rows = y, cols = x
  H = terrain_raw(X, Y)
  pads = {}
  for k, (px, py, col, where) in PADS.items():
    pz = float(terrain_raw(px, py))
    d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    w = smoothstep((d - PAD_R - 0.3) / 2.2)
    H = pz * (1 - w) + H * w
    pads[k] = dict(x=px, y=py, z=pz, rgb=col, where=where)
  return xs, H, pads


def poisson(rng, n_try, r_min, half):
  pts = []
  for _ in range(n_try):
    p = rng.uniform(-half, half, 2)
    if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > r_min ** 2 for q in pts):
      pts.append(p)
  return np.array(pts)


def forest_mask(x, y):
  m = (np.sin(0.09 * x + 1.0) * np.cos(0.08 * y - 0.5) + 0.6 * np.sin(0.05 * x - 0.11 * y + 2.0)
       + 0.4 * np.sin(0.17 * x + 0.13 * y))
  m += 2.8 * np.exp(-((x - 24) ** 2 + (y + 20) ** 2) / (2 * 14 ** 2))     # forest around pad 3
  m += 1.0 * np.exp(-((x - 35) ** 2 + (y + 40) ** 2) / (2 * 12 ** 2))     # forest between 3 and 5
  return m


def build_trees(rng, height_at, pads):
  pts = poisson(rng, 9000, 4.2, HALF - 2)
  trees = []
  for p in pts:
    f = forest_mask(p[0], p[1])
    if not (f > 0.55 or rng.random() < 0.06):
      continue
    dpad = min(np.hypot(p[0] - q["x"], p[1] - q["y"]) for q in pads.values())
    rxy = rng.uniform(1.4, 2.8)
    clear = 6.5 if np.hypot(p[0] - 24, p[1] + 20) < 15 else 4.0
    if dpad < rxy + clear:
      continue
    g = float(height_at(p[0], p[1]))
    kind = "conifer" if rng.random() < 0.35 else "broadleaf"
    trunk_r = rng.uniform(0.18, 0.38)
    if kind == "conifer":
      rxy *= 0.75
      rz = rxy * rng.uniform(2.0, 2.6)
      cz = g + rz + rng.uniform(1.2, 2.5)
    else:
      rz = rxy * rng.uniform(1.0, 1.4)
      cz = g + max(rz + rng.uniform(1.6, 4.5), rng.uniform(5, 9))
    trees.append(dict(x=float(p[0]), y=float(p[1]), g=g, trunk_r=float(trunk_r), z1=float(cz),
                      cz=float(cz), rxy=float(rxy), rz=float(rz), kind=kind, shade=float(rng.uniform(0, 1))))
  return trees


def build_rocks(rng, height_at, pads, trees):
  rocks = []
  while len(rocks) < 34:
    p = rng.uniform(-HALF + 3, HALF - 3, 2)
    r = rng.uniform(0.5, 1.7)
    if min(np.hypot(p[0] - q["x"], p[1] - q["y"]) for q in pads.values()) < r + 3.5:
      continue
    if any(np.hypot(p[0] - t["x"], p[1] - t["y"]) < r + t["trunk_r"] + 1.0 for t in trees):
      continue
    rz = r * rng.uniform(0.45, 0.8)
    g = float(height_at(p[0], p[1]))
    rocks.append(dict(x=float(p[0]), y=float(p[1]), cz=float(g + rz * rng.uniform(-0.35, 0.1)), rxy=float(r), rz=float(rz)))
  return rocks


def ground_texture(xs, H, trees, path):
  """Bake grass/soil/rock colours from height, slope and noise into one 4096^2 texture."""
  n = 4096
  rng = np.random.default_rng(7)
  u = np.linspace(-HALF, HALF, n)
  X, Y = np.meshgrid(u, u)
  # upsample heights
  from numpy import interp
  ix = (X + HALF) / RES
  iy = (Y + HALF) / RES
  i0 = np.clip(ix.astype(int), 0, N - 2); j0 = np.clip(iy.astype(int), 0, N - 2)
  fx, fy = ix - i0, iy - j0
  Hu = (H[j0, i0] * (1 - fx) * (1 - fy) + H[j0, i0 + 1] * fx * (1 - fy) + H[j0 + 1, i0] * (1 - fx) * fy + H[j0 + 1, i0 + 1] * fx * fy)
  gy, gx = np.gradient(Hu, u[1] - u[0])
  slope = np.sqrt(gx ** 2 + gy ** 2)

  def noise(scale, seed):
    r = np.random.default_rng(seed)
    small = r.random((max(2, int(n / scale)), max(2, int(n / scale)))).astype(np.float32)
    im = Image.fromarray((small * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255.0

  n1, n2, n3 = noise(256, 1), noise(48, 2), noise(6, 3)
  hn = (Hu - Hu.min()) / (Hu.max() - Hu.min())
  grass_a = np.array([88, 122, 48], np.float32)     # sunlit grass
  grass_b = np.array([58, 92, 38], np.float32)      # darker grass
  dry = np.array([150, 140, 78], np.float32)        # dry grass
  soil = np.array([112, 88, 62], np.float32)
  rock = np.array([128, 122, 112], np.float32)
  t = np.clip(0.55 * n1 + 0.3 * n2 + 0.15 * hn, 0, 1)[..., None]
  col = grass_a * (1 - t) + grass_b * t
  dmix = np.clip((n1 - 0.62) * 4 + (hn - 0.75) * 2, 0, 1)[..., None]
  col = col * (1 - dmix * 0.6) + dry * dmix * 0.6
  smix = np.clip((slope - 0.38) * 3.0, 0, 1)[..., None]
  col = col * (1 - smix) + (soil * (1 - n2[..., None]) + rock * n2[..., None]) * smix
  # darker ground under canopies (shade + needles)
  sh = np.zeros((n, n), np.float32)
  px = lambda v: int((v + HALF) / (2 * HALF) * (n - 1))
  img = Image.new("L", (n, n), 0); dr = ImageDraw.Draw(img)
  for tr in trees:
    r = tr["rxy"] * 1.1 / (2 * HALF) * n
    cx, cy = px(tr["x"]), px(tr["y"])
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=150)
  sh = np.asarray(img.filter(ImageFilter.GaussianBlur(18)), np.float32)[..., None] / 255.0
  col = col * (1 - 0.45 * sh) + np.array([60, 50, 34], np.float32) * 0.45 * sh
  col *= (0.86 + 0.28 * n3[..., None])
  col = np.clip(col, 0, 255).astype(np.uint8)
  # MuJoCo maps the texture's first row to +y; our arrays have row 0 at -y
  Image.fromarray(col[::-1]).save(path)


def pad_texture(name, rgb, path):
  n = 512
  im = Image.new("RGB", (n, n), (34, 36, 40))
  d = ImageDraw.Draw(im)
  d.rectangle([14, 14, n - 15, n - 15], outline=rgb, width=40)
  d.ellipse([86, 86, n - 87, n - 87], outline=(235, 235, 230), width=14)
  font = None
  for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc"]:
    try:
      font = ImageFont.truetype(f, 250); break
    except OSError:
      pass
  d.text((n / 2, n / 2 + 6), name, fill=rgb if name != "H" else (240, 240, 235), font=font, anchor="mm")
  im.save(path)


def write_scene(out, xs, H, pads, trees, rocks, x2_dir):
  out.mkdir(parents=True, exist_ok=True)
  (out / "assets").mkdir(exist_ok=True)
  zmin, zmax = float(H.min()), float(H.max())   # MuJoCo re-normalises hfield data to [0,1]
  norm = ((H - zmin) / (zmax - zmin)).astype(np.float32)
  with open(out / "terrain.bin", "wb") as f:
    f.write(struct.pack("ii", N, N)); f.write(norm.tobytes())
  ground_texture(xs, H, trees, out / "assets" / "ground.png")
  for k, p in pads.items():
    pad_texture(k, p["rgb"], out / "assets" / f"pad_{k}.png")
  for f in ["X2_lowpoly.obj", "X2_lowpoly_texture_SpinningProps_1024.png"]:
    shutil.copy(x2_dir / "assets" / f, out / "assets" / f)
  shutil.copy(x2_dir / "LICENSE", out / "assets" / "LICENSE_skydio_x2_menagerie")
  x2 = C.fix_x2((x2_dir / "x2.xml").read_text())
  x2 = x2.replace('assetdir="assets"', '')
  body = x2[x2.index("<worldbody>") + len("<worldbody>"):x2.index("</worldbody>")]
  body = body.replace('<light name="spotlight" mode="targetbodycom" target="x2" pos="0 -1 2"/>', '')
  hx, hy = pads["H"]["x"], pads["H"]["y"]
  body = body.replace('<body name="x2" pos="0 0 0.1" childclass="x2">',
                      f'<body name="x2" pos="{hx} {hy} {pads["H"]["z"] + 0.045:.3f}" childclass="x2">')
  body = body.replace('<site name="imu" pos="0 0 .02"/>',
                      '<site name="imu" pos="0 0 .02"/>\n      <camera name="down" pos="0 0 -0.01" xyaxes="0 -1 0 1 0 0" fovy="90"/>'
                      '\n      <camera name="chase" pos="-6 0 2.6" xyaxes="0 -1 0 0.38 0 1" mode="track" fovy="55"/>')
  defaults = x2[x2.index("<default>"):x2.index("</default>\n  </default>") + len("</default>\n  </default>")]
  actuator = x2[x2.index("<actuator>"):x2.index("</actuator>") + len("</actuator>")]
  sensor = x2[x2.index("<sensor>"):x2.index("</sensor>") + len("</sensor>")]

  g = []
  for i, t in enumerate(trees):
    trunk_col = "0.30 0.22 0.15 1" if t["kind"] == "broadleaf" else "0.26 0.19 0.13 1"
    g.append(f'<geom name="trunk{i}" type="cylinder" size="{t["trunk_r"]:.3f} {(t["z1"] - t["g"] + 0.5) / 2:.3f}" '
             f'pos="{t["x"]:.3f} {t["y"]:.3f} {(t["z1"] + t["g"] - 0.5) / 2:.3f}" rgba="{trunk_col}" group="0"/>')
    s = t["shade"]
    # the canopy envelope is what the drone can hit (invisible, group 3); the leafy pieces inside it
    # (group 1, no collisions) are what you see and what the lidar sees
    g.append(f'<geom name="canopy{i}" type="ellipsoid" size="{t["rxy"]:.3f} {t["rxy"]:.3f} {t["rz"]:.3f}" '
             f'pos="{t["x"]:.3f} {t["y"]:.3f} {t["cz"]:.3f}" rgba="0 0 0 0" group="3"/>')
    vr = np.random.default_rng(1000 + i)
    vis = dict(contype="0", conaffinity="0", group="1", material="leaves")
    va = " ".join(f'{k}="{v}"' for k, v in vis.items())
    if t["kind"] == "conifer":
      base = np.array([0.10 + 0.05 * s, 0.24 + 0.07 * s, 0.13 + 0.03 * s])
      for k in range(6):
        f = k / 5
        zc = t["cz"] - 0.78 * t["rz"] + f * 1.5 * t["rz"]
        rr = 0.97 * t["rxy"] * np.sqrt(max(0.04, 1 - ((zc - t["cz"]) / t["rz"]) ** 2)) * (1.12 - 0.3 * f)
        rr = min(rr, 0.97 * t["rxy"])
        c = np.clip(base * (0.9 + 0.25 * f) + vr.normal(0, 0.012, 3), 0, 1)
        g.append(f'<geom type="ellipsoid" size="{rr:.3f} {rr:.3f} {0.22 * t["rz"]:.3f}" pos="{t["x"]:.3f} {t["y"]:.3f} {zc:.3f}" '
                 f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1" {va}/>')
    else:
      base = np.array([0.20 + 0.10 * s, 0.35 + 0.10 * s, 0.11 + 0.04 * s])
      c = base * 0.85
      g.append(f'<geom type="ellipsoid" size="{0.82 * t["rxy"]:.3f} {0.82 * t["rxy"]:.3f} {0.82 * t["rz"]:.3f}" '
               f'pos="{t["x"]:.3f} {t["y"]:.3f} {t["cz"]:.3f}" rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1" {va}/>')
      for k in range(8):
        u = vr.normal(size=3); u[2] = abs(u[2]) * 0.8 + 0.1 if k < 6 else -abs(u[2]) * 0.5; u /= np.linalg.norm(u)
        rb = t["rxy"] * vr.uniform(0.42, 0.52)
        cen = np.array([t["x"], t["y"], t["cz"]]) + u * np.array([t["rxy"], t["rxy"], t["rz"]]) * (1 - rb / t["rxy"]) * 0.98
        c = np.clip(base * (0.92 + 0.22 * max(u[2], 0)) + vr.normal(0, 0.015, 3), 0, 1)
        g.append(f'<geom type="ellipsoid" size="{rb:.3f} {rb:.3f} {rb * 0.9:.3f}" pos="{cen[0]:.3f} {cen[1]:.3f} {cen[2]:.3f}" '
                 f'rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1" {va}/>')
  for i, r in enumerate(rocks):
    g.append(f'<geom name="rock{i}" type="ellipsoid" size="{r["rxy"]:.3f} {r["rxy"] * 0.85:.3f} {r["rz"]:.3f}" '
             f'pos="{r["x"]:.3f} {r["y"]:.3f} {r["cz"]:.3f}" material="rock" group="0"/>')
  for k, p in pads.items():
    g.append(f'<geom name="pad_{k}" type="box" size="{PAD_SIZE} {PAD_SIZE} 0.02" pos="{p["x"]} {p["y"]} {p["z"] + 0.02:.3f}" '
             f'material="pad_{k}" group="0"/>')
  pad_mats = "\n    ".join(f'<texture name="pad_{k}" type="2d" file="assets/pad_{k}.png"/>'
                            f'<material name="pad_{k}" texture="pad_{k}" specular="0.1"/>' for k in pads)
  xml = f"""<mujoco model="drone forest">
  <compiler autolimits="true" angle="radian"/>
  <option timestep="{C.SIM_DT}" density="0" viscosity="0"/>
  <size memory="64M"/>
  <statistic center="0 0 5" extent="60"/>
  <visual>
    <global offwidth="1920" offheight="1920" azimuth="135" elevation="-25"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.0005" zfar="6" fogstart="1.3" fogend="4.5" shadowscale="0.6" shadowclip="0.6"/>
    <headlight ambient="0.32 0.30 0.28" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
    <rgba fog="0.83 0.72 0.58 1" haze="0.9 0.75 0.6 1"/>
    <scale framelength="1" framewidth="0.05"/>
  </visual>
  <default>
    {defaults[len("<default>"):]}
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.70 0.88" rgb2="0.95 0.80 0.62" width="512" height="3072"/>
    <texture name="ground" type="2d" file="assets/ground.png"/>
    <material name="ground" texture="ground" texrepeat="1 1" specular="0.05" shininess="0.1"/>
    <texture name="leaves" type="2d" builtin="flat" mark="random" random="0.35" rgb1="1 1 1" markrgb="0.55 0.6 0.5" width="256" height="256"/>
    <material name="leaves" texture="leaves" texrepeat="3 3" specular="0.05"/>
    <texture name="rock" type="2d" builtin="flat" mark="random" random="0.3" rgb1="0.52 0.50 0.47" markrgb="0.38 0.36 0.34" width="128" height="128"/>
    <material name="rock" texture="rock" texrepeat="2 2" specular="0.15"/>
    {pad_mats}
    <texture type="2d" file="assets/X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="assets/X2_lowpoly.obj"/>
    <hfield name="terrain" file="terrain.bin" size="{HALF} {HALF} {zmax - zmin:.4f} 1"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 80" dir="-0.55 -0.45 -0.70" diffuse="1.0 0.86 0.68" specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="skyfill" directional="true" pos="0 0 80" dir="0.4 0.5 -0.75" diffuse="0.18 0.22 0.30" specular="0 0 0" castshadow="false"/>
    <geom name="ground" type="hfield" hfield="terrain" pos="0 0 {zmin:.4f}" material="ground" group="0"/>
    {chr(10).join('    ' + x for x in g)}
    {body}
  </worldbody>
  {actuator}
  {sensor}
</mujoco>
"""
  (out / "scene.xml").write_text(xml)
  meta = dict(half=HALF, res=RES, n=N, zmin=zmin, zmax=zmax, pad_r=PAD_R, pad_size=PAD_SIZE,
              pads=pads, trees=trees, rocks=rocks)
  (out / "world.json").write_text(json.dumps(meta, indent=1))
  np.save(out / "heights.npy", H.astype(np.float32))
  return xml


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--x2", default=None)
  ap.add_argument("--out", default="scene")
  a = ap.parse_args()
  x2_dir = pathlib.Path(a.x2) if a.x2 else None
  if x2_dir is None:
    from mujoco_playground._src import mjx_env
    x2_dir = mjx_env.MENAGERIE_PATH / "skydio_x2"
  rng = np.random.default_rng(11)
  xs, H, pads = build_heights()

  def height_at(x, y):
    i = np.clip((x + HALF) / RES, 0, N - 1.001); j = np.clip((y + HALF) / RES, 0, N - 1.001)
    i0, j0 = int(i), int(j); fx, fy = i - i0, j - j0
    return (H[j0, i0] * (1 - fx) * (1 - fy) + H[j0, i0 + 1] * fx * (1 - fy) + H[j0 + 1, i0] * (1 - fx) * fy
            + H[j0 + 1, i0 + 1] * fx * fy)

  trees = build_trees(rng, height_at, pads)
  rocks = build_rocks(rng, height_at, pads, trees)
  write_scene(pathlib.Path(a.out), xs, H, pads, trees, rocks, x2_dir)
  print(f"trees {len(trees)}  rocks {len(rocks)}  terrain {H.min():.1f}..{H.max():.1f} m")
  for k, p in pads.items():
    print(f"pad {k}: ({p['x']:.0f}, {p['y']:.0f})  z {p['z']:.2f}  {p['where']}")


if __name__ == "__main__":
  main()
