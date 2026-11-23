"""Route planner on the lidar map. Two flight levels: LOW (weave between trees, 3.5 m above the ground)
and HIGH (hop over the canopy). A* picks where to fly low, where to climb over, and how to get around."""
import heapq

import numpy as np
from scipy import ndimage

LOW_AGL = 3.5          # cruise height above ground when weaving between trees
HIGH_CLEAR = 4.0       # clearance above the canopy when hopping over
CLEAR_R = 2.2          # keep this far (m) from any obstacle at the low level
HOVER = 2.5            # height above a pad before the final descent
HIGH_COST = 1.35       # flying high is allowed but less preferred (more exposed, more energy)
CLIMB_COST = 1.5       # per metre of level change


class Plan:
  def __init__(self, path, levels, cells):
    self.path, self.levels, self.cells = path, levels, cells
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    self.ps = np.concatenate([[0.0], np.cumsum(seg)])

  @property
  def length(self):
    return float(self.ps[-1])


def plan_route(mp, start, goal, pad_clear=3.0):
  """start / goal: (x, y, z) of the pad surfaces. Returns a Plan (3D polyline for the pilot)."""
  surf, ground, ohag, known = mp.layers()
  res, n = mp.res, mp.n
  obst = (ohag > 0.7) | ~known
  # pads themselves are flat and free
  I, J = np.meshgrid(np.arange(n), np.arange(n))
  X, Y = mp.center(I, J)
  for p in (start, goal):
    obst &= ~(((X - p[0]) ** 2 + (Y - p[1]) ** 2) < pad_clear ** 2)
  r = int(np.ceil(CLEAR_R / res))
  disk = (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r)
  low_block = ndimage.binary_dilation(obst, structure=disk)
  for p in (start, goal):
    low_block &= ~(((X - p[0]) ** 2 + (Y - p[1]) ** 2) < 1.5 ** 2)
  ground_s = ndimage.gaussian_filter(ground, 2.0 / res)
  z_low = np.maximum(ground_s, ground) + LOW_AGL
  z_high = ndimage.gaussian_filter(ndimage.maximum_filter(surf, size=int(7.0 / res)), 1.5 / res) + HIGH_CLEAR
  z_high = np.maximum(z_high, z_low + 2.0)
  z_lv = np.stack([z_low, z_high])
  blocked = np.stack([low_block, ~known])

  si, sj = mp.idx(start[0], start[1])
  gi, gj = mp.idx(goal[0], goal[1])
  nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
  INF = 1e18
  cost = np.full((2, n, n), INF)
  parent = {}
  h = lambda i, j: res * np.hypot(i - gi, j - gj)
  start_node = (0, sj, si)
  cost[start_node] = 0.0
  pq = [(h(si, sj), 0.0, start_node)]
  goal_node = None
  while pq:
    f, c, node = heapq.heappop(pq)
    if c > cost[node]:
      continue
    lv, j, i = node
    if i == gi and j == gj and lv == 0:
      goal_node = node
      break
    # level change in place
    other = (1 - lv, j, i)
    if not blocked[other]:
      nc = c + CLIMB_COST * abs(z_lv[1, j, i] - z_lv[0, j, i])
      if nc < cost[other]:
        cost[other] = nc; parent[other] = node
        heapq.heappush(pq, (nc + h(i, j), nc, other))
    for di, dj in nbr:
      ii, jj = i + di, j + dj
      if not (0 <= ii < n and 0 <= jj < n) or blocked[lv, jj, ii]:
        continue
      step = res * (1.4142 if di and dj else 1.0)
      dz = abs(z_lv[lv, jj, ii] - z_lv[lv, j, i])
      nc = c + step * (HIGH_COST if lv else 1.0) + 0.5 * dz
      nn = (lv, jj, ii)
      if nc < cost[nn]:
        cost[nn] = nc; parent[nn] = node
        heapq.heappush(pq, (nc + h(ii, jj), nc, nn))
  if goal_node is None:
    return None
  cells = [goal_node]
  while cells[-1] in parent:
    cells.append(parent[cells[-1]])
  cells = cells[::-1]

  # line-of-sight shortcutting on each level
  def free_line(a, b):
    la, ja, ia = a; lb, jb, ib = b
    if la != lb:
      return False
    k = int(max(abs(ia - ib), abs(ja - jb))) * 2 + 1
    ii = np.round(np.linspace(ia, ib, k)).astype(int); jj = np.round(np.linspace(ja, jb, k)).astype(int)
    return not blocked[la, jj, ii].any()

  keep = [cells[0]]
  k = 0
  while k < len(cells) - 1:
    m = len(cells) - 1
    while m > k + 1 and not free_line(cells[k], cells[m]):
      m -= 1
    keep.append(cells[m]); k = m
  # dense 3D polyline: resample every 1 m, height from the level map, then make climbs early / descents late
  pts, lvls = [], []
  for a, b in zip(keep[:-1], keep[1:]):
    xa, ya = mp.center(a[2], a[1]); xb, yb = mp.center(b[2], b[1])
    L = max(np.hypot(xb - xa, yb - ya), 1e-6)
    m = max(int(L / 1.0), 1)
    for t in np.linspace(0, 1, m, endpoint=False):
      x, y = xa + t * (xb - xa), ya + t * (yb - ya)
      i, j = mp.idx(x, y)
      lv = max(a[0], b[0])
      pts.append([x, y, z_lv[lv, j, i]]); lvls.append(lv)
  pts.append([*mp.center(keep[-1][2], keep[-1][1]), z_lv[0, keep[-1][1], keep[-1][2]]]); lvls.append(0)
  pts = np.array(pts)
  pts[0, :2] = start[:2]; pts[-1, :2] = goal[:2]
  # round the corners (the pilot follows smooth curves best): moving average on the dense points,
  # then take the height again from the level map at the new positions
  xy = pts[:, :2].copy()
  for _ in range(3):
    xy[1:-1] = ndimage.uniform_filter1d(xy, 7, axis=0, mode="nearest")[1:-1]
  pts[:, :2] = xy
  for k_, (x, y) in enumerate(xy):
    i, j = mp.idx(x, y)
    pts[k_, 2] = z_lv[lvls[k_], j, i]
  z = pts[:, 2].copy()
  s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
  slope = 0.8                                  # max climb/descent per metre flown
  for i in range(len(z) - 2, -1, -1):          # climb before obstacles
    z[i] = max(z[i], z[i + 1] - slope * (s[i + 1] - s[i]))
  for i in range(1, len(z)):                   # descend after them
    z[i] = max(z[i], z[i - 1] - slope * (s[i] - s[i - 1]))
  z = ndimage.uniform_filter1d(z, 5, mode="nearest")
  z = np.maximum(z, pts[:, 2] - 0.3)
  pts[:, 2] = z
  path = np.concatenate([[start], pts, [[goal[0], goal[1], goal[2] + HOVER], goal]], 0)
  path[1, 2] = max(path[1, 2], start[2] + 2.0)
  return Plan(path, np.array([0] + lvls + [0, 0]), keep)


def survey_route(home, z_abs=24.0, lanes=(-45.0, -15.0, 15.0, 45.0), x_end=40.0):
  """Lawn-mower mapping flight at a fixed altitude with round turns, starting and ending at home."""
  def arc(c, r, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return [[c[0] + r * np.cos(t), c[1] + r * np.sin(t), z_abs] for t in a]

  pts = [list(home), [home[0], home[1], z_abs]]
  r = (lanes[1] - lanes[0]) / 2
  for k, y in enumerate(lanes):
    east = k % 2 == 0
    xs = (-x_end, x_end) if east else (x_end, -x_end)
    pts += [[xs[0], y, z_abs], [xs[1], y, z_abs]]
    if k < len(lanes) - 1:
      c = (xs[1], y + r)
      pts += arc(c, r, -np.pi / 2, np.pi / 2)[1:-1] if east else arc(c, r, -np.pi / 2, -3 * np.pi / 2)[1:-1]
  last = pts[-1]
  turn = [last[0] - 10.0 if len(lanes) % 2 == 0 else last[0] + 10.0, last[1] - 10.0, z_abs]
  pts += [turn, [home[0], home[1], z_abs], [home[0], home[1], home[2] + HOVER], list(home)]
  pts = np.array(pts, float)
  dense = [pts[0]]
  for a, b in zip(pts[:-1], pts[1:]):
    L = np.linalg.norm(b - a)
    for t in np.linspace(0, 1, max(int(L / 2.0), 1) + 1)[1:]:
      dense.append(a + t * (b - a))
  path = np.array(dense)
  # soften the remaining corners of the horizontal part (not the vertical climb / descent)
  flat = np.abs(path[:, 2] - z_abs) < 1e-6
  idx = np.nonzero(flat)[0]
  xy = path[idx, :2].copy()
  for _ in range(3):
    xy[1:-1] = ndimage.uniform_filter1d(xy, 5, axis=0, mode="nearest")[1:-1]
  path[idx, :2] = xy
  return Plan(path, np.ones(len(path), int), [])
