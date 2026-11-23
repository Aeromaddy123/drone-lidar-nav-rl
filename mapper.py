"""Builds a 2.5D map from lidar points: surface height (what the lidar hit from above), an estimate of the
bare ground under the trees, and obstacle height above ground. Also finds the coloured landing pads in
the down-camera image."""
import numpy as np
from scipy import ndimage


class Map25D:
  def __init__(self, half=60.0, res=0.5):
    self.half, self.res = half, res
    self.n = int(round(2 * half / res))
    self.top = np.full((self.n, self.n), np.nan, np.float32)      # rows = y, cols = x
    self.points = []                                                # a thinned copy for display
    self.n_points = 0

  def idx(self, x, y):
    i = np.floor((np.asarray(x) + self.half) / self.res).astype(int)
    j = np.floor((np.asarray(y) + self.half) / self.res).astype(int)
    return np.clip(i, 0, self.n - 1), np.clip(j, 0, self.n - 1)

  def center(self, i, j):
    return -self.half + (np.asarray(i) + 0.5) * self.res, -self.half + (np.asarray(j) + 0.5) * self.res

  def add(self, pts):
    if len(pts) == 0:
      return
    inside = (np.abs(pts[:, 0]) < self.half) & (np.abs(pts[:, 1]) < self.half)
    pts = pts[inside]
    i, j = self.idx(pts[:, 0], pts[:, 1])
    # several points can fall in one cell: keep the highest
    order = np.argsort(pts[:, 2])
    i, j, z = i[order], j[order], pts[order, 2]
    cur = self.top[j, i]
    self.top[j, i] = np.where(np.isnan(cur), z, np.maximum(cur, z))
    self.n_points += len(pts)
    self.points.append(pts[:: 7])

  def coverage(self):
    return float(np.isfinite(self.top).mean())

  def layers(self):
    """surface (filled), ground estimate, obstacle height above ground, known mask."""
    known = np.isfinite(self.top)
    surf = self.top.copy()
    if not known.all():   # fill small holes from the nearest known cell
      idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
      surf = surf[idx[0], idx[1]]
    # bare ground: morphological opening removes anything narrower than ~7 m (trees, rocks) but keeps
    # the shape of the hills, then a light smoothing
    w = int(7.0 / self.res) | 1
    ground = ndimage.maximum_filter(ndimage.minimum_filter(surf, size=w), size=w)
    ground = ndimage.gaussian_filter(ground, 1.0 / self.res)
    ground = np.minimum(ground, surf)
    ohag = surf - ground
    # small lidar shadows (behind trunks / canopies) count as seen if surrounded by seen cells
    near = ndimage.distance_transform_edt(~known) * self.res < 1.6
    return surf, ground, ohag, known | near

  def ground_at(self, x, y):
    _, ground, _, _ = self.layers()
    i, j = self.idx(x, y)
    return float(ground[j, i])


PAD_HUES = {"1": 0.0, "2": 46.0, "3": 189.0, "4": 224.0, "5": 280.0}


def find_pads(rgb, min_px=12):
  """Colour detection of the numbered pads in a camera image. Returns {name: (u, v, n_pixels)}."""
  x = rgb.astype(np.float32) / 255.0
  mx, mn = x.max(-1), x.min(-1)
  s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
  r, g, b = x[..., 0], x[..., 1], x[..., 2]
  d = np.maximum(mx - mn, 1e-6)
  h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
  found = {}
  for name, hue in PAD_HUES.items():
    dh = np.abs((h - hue + 180) % 360 - 180)
    mask = (dh < 14) & (s > 0.6) & (mx > 0.45)
    if mask.sum() < min_px:
      continue
    lab, nl = ndimage.label(ndimage.binary_dilation(mask, iterations=3))   # join the digit and the border
    lab = lab * mask
    sizes = ndimage.sum(mask, lab, range(1, nl + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < min_px:
      continue
    vv, uu = np.nonzero(lab == k)
    found[name] = (float(uu.mean()), float(vv.mean()), int(sizes[k - 1]))
  return found
