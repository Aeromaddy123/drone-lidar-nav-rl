"""Turn the random training worlds (as saved by rollout_train.py) into one MuJoCo scene for the montage:
each world gets its own hilly ground, trees, rocks and two pads, laid out side by side; one drone body is
moved into whichever world is being rendered."""
import pathlib
import struct

import numpy as np

import common as C

SPACING = 140.0
HALF = 50.0
N = 201


def smoothstep(x):
  x = np.clip(x, 0, 1)
  return x * x * (3 - 2 * x)


def terrain_h(w, x, y):
  ph = x[..., None] * w["F"][:, 0] + y[..., None] * w["F"][:, 1] + w["P"]
  h = (w["A"] * np.sin(ph)).sum(-1)
  for pad in w["pads"]:
    d = np.sqrt((x - pad[0]) ** 2 + (y - pad[1]) ** 2)
    wt = smoothstep((d - pad[3]) / 1.8)
    h = pad[2] * (1 - wt) + h * wt
  return h


def load_worlds(npz, k=None):
  z = np.load(npz)
  n = z["qpos"].shape[0]
  out = []
  for i in range(n):
    out.append({key[2:]: z[key][i] for key in z.files if key.startswith("w_")})
  return out


def world_xml(worlds, scene_dir, x2_xml_body, x2_defaults, x2_actuator, cols=3, write=True):
  scene_dir = pathlib.Path(scene_dir)
  if not write:
    return [((k % cols) * SPACING, -(k // cols) * SPACING) for k in range(len(worlds))]
  (scene_dir / "tw").mkdir(parents=True, exist_ok=True)
  assets, geoms, offsets = [], [], []
  xs = np.linspace(-HALF, HALF, N)
  X, Y = np.meshgrid(xs, xs)
  for k, w in enumerate(worlds):
    ox, oy = (k % cols) * SPACING, -(k // cols) * SPACING
    offsets.append((ox, oy))
    Hh = terrain_h(w, X, Y)
    zmin, zmax = float(Hh.min()), float(Hh.max()) + 1e-3
    with open(scene_dir / "tw" / f"t{k}.bin", "wb") as f:
      f.write(struct.pack("ii", N, N)); f.write(((Hh - zmin) / (zmax - zmin)).astype(np.float32).tobytes())
    assets.append(f'<hfield name="t{k}" file="tw/t{k}.bin" size="{HALF} {HALF} {zmax - zmin:.4f} 1"/>')
    geoms.append(f'<geom type="hfield" hfield="t{k}" pos="{ox} {oy} {zmin:.4f}" material="ground"/>')
    rng = np.random.default_rng(k)
    for t, (cx, cy, r, z0, z1) in enumerate(w["trunks"]):
      if cx > 400:
        continue
      geoms.append(f'<geom type="cylinder" size="{r:.3f} {(z1 - z0) / 2:.3f}" pos="{cx + ox:.3f} {cy + oy:.3f} {(z1 + z0) / 2:.3f}" rgba="0.30 0.22 0.15 1"/>')
    for e, (cx, cy, cz, rxy, rz) in enumerate(w["ells"]):
      if cx > 400:
        continue
      if e < len(w["trunks"]):        # canopy: leafy blobs inside the ellipsoid the pilot trained against
        s = rng.uniform()
        base = np.array([0.20 + 0.10 * s, 0.35 + 0.10 * s, 0.11 + 0.04 * s])
        c = base * 0.85
        geoms.append(f'<geom type="ellipsoid" size="{0.84 * rxy:.3f} {0.84 * rxy:.3f} {0.84 * rz:.3f}" pos="{cx + ox:.3f} {cy + oy:.3f} {cz:.3f}" '
                     f'material="leaves" rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"/>')
        for b in range(6):
          u = rng.normal(size=3); u[2] = abs(u[2]) * 0.8 + 0.1; u /= np.linalg.norm(u)
          rb = rxy * rng.uniform(0.42, 0.52)
          cen = np.array([cx + ox, cy + oy, cz]) + u * np.array([rxy, rxy, rz]) * (1 - rb / rxy) * 0.98
          cc = np.clip(base * (0.92 + 0.22 * u[2]), 0, 1)
          geoms.append(f'<geom type="ellipsoid" size="{rb:.3f} {rb:.3f} {0.9 * rb:.3f}" pos="{cen[0]:.3f} {cen[1]:.3f} {cen[2]:.3f}" '
                       f'material="leaves" rgba="{cc[0]:.3f} {cc[1]:.3f} {cc[2]:.3f} 1"/>')
      else:
        geoms.append(f'<geom type="ellipsoid" size="{rxy:.3f} {rxy * 0.85:.3f} {rz:.3f}" pos="{cx + ox:.3f} {cy + oy:.3f} {cz:.3f}" material="rock"/>')
    for j, (px, py, pz, pr) in enumerate(w["pads"]):
      geoms.append(f'<geom type="box" size="1.15 1.15 0.02" pos="{px + ox:.3f} {py + oy:.3f} {pz + 0.02:.3f}" material="{"pad_H" if j == 0 else "pad_goal"}"/>')
  xml = f"""<mujoco model="training worlds">
  <compiler autolimits="true"/>
  <option timestep="{C.SIM_DT}" density="0" viscosity="0"/>
  <size memory="64M"/>
  <statistic center="0 0 5" extent="60"/>
  <visual>
    <global offwidth="1920" offheight="1920"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.0005" zfar="6" fogstart="1.0" fogend="3.0" shadowscale="0.6" shadowclip="0.4"/>
    <headlight ambient="0.32 0.30 0.28" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
    <rgba fog="0.83 0.72 0.58 1" haze="0.9 0.75 0.6 1"/>
  </visual>
  <default>
    {x2_defaults}
  <asset>
    <texture name="ground" type="2d" builtin="flat" mark="random" random="0.25" rgb1="0.36 0.50 0.22" markrgb="0.30 0.42 0.18" width="512" height="512"/>
    <material name="ground" texture="ground" texrepeat="40 40" specular="0.05"/>
    <texture name="leaves" type="2d" builtin="flat" mark="random" random="0.35" rgb1="1 1 1" markrgb="0.55 0.6 0.5" width="256" height="256"/>
    <material name="leaves" texture="leaves" texrepeat="3 3" specular="0.05"/>
    <texture name="rock" type="2d" builtin="flat" mark="random" random="0.3" rgb1="0.52 0.50 0.47" markrgb="0.38 0.36 0.34" width="128" height="128"/>
    <material name="rock" texture="rock" texrepeat="2 2" specular="0.15"/>
    <texture name="pad_H" type="2d" file="assets/pad_H.png"/><material name="pad_H" texture="pad_H"/>
    <texture name="pad_goal" type="2d" file="assets/pad_3.png"/><material name="pad_goal" texture="pad_goal"/>
    <texture type="2d" file="assets/X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="assets/X2_lowpoly.obj"/>
    {chr(10).join('    ' + a for a in assets)}
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 80" dir="-0.55 -0.45 -0.70" diffuse="1.0 0.86 0.68" specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="skyfill" directional="true" pos="0 0 80" dir="0.4 0.5 -0.75" diffuse="0.18 0.22 0.30" specular="0 0 0" castshadow="false"/>
{chr(10).join('    ' + g for g in geoms)}
    {x2_xml_body}
  </worldbody>
  {x2_actuator}
</mujoco>"""
  (scene_dir / "train_worlds.xml").write_text(xml)
  return offsets


def x2_parts(scene_xml_text):
  """Reuse the drone definition from the demo scene.xml."""
  s = scene_xml_text
  defaults = s[s.index("<default>") + len("<default>"):s.index("</default>\n  </default>") + len("</default>\n  </default>")]
  body = s[s.index('<body name="x2"'):]
  depth, i = 0, 0
  while True:
    a, b = body.find("<body", i), body.find("</body>", i)
    if a != -1 and a < b:
      depth += 1; i = a + 5
    else:
      depth -= 1; i = b + 7
      if depth == 0:
        break
  body = body[:i]
  act = s[s.index("<actuator>"):s.index("</actuator>") + len("</actuator>")]
  return body, defaults, act
