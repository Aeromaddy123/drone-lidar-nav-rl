"""Smoke test: shapes, hover physics, rate loop, lidar sanity and batched speed."""
import time
import jax, jax.numpy as jp, numpy as np
import drone_env, common as C

env = drone_env.DroneNav()
print("mass", env._mass, "J", np.diag(np.array(env._J)), "obs", env.observation_size, "C.OBS_SIZE", C.OBS_SIZE)
reset, step = jax.jit(env.reset), jax.jit(env.step)
s = reset(jax.random.PRNGKey(3))
print("path len", float(s.info["ps"][-1]), "start", np.array(s.info["path"][0]), "goal", np.array(s.info["path"][-1]))
print("active trees", int((np.array(s.info["world"]["trunks"][:, 0]) < 400).sum()))

# hover: zero action should hold altitude (no wind for this check)
s.info["wind"] = jp.zeros(3)
z = []
for i in range(100):
  s = step(s, jp.zeros(4)); z.append(float(s.data.qpos[2]))
print("hover z start/end", z[0], z[-1], "done", float(s.done), "reward", float(s.reward))
# roll-rate command
w = []
for i in range(5):
  s = step(s, jp.array([0.0, 0.2, 0.0, 0.0])); w.append(np.array(s.data.qvel[3:6]))
print("omega after roll cmd 0.5 rad/s:", np.round(w[-1], 3))

# lidar: drone 3 m above flat ground, compare down ray to AGL
T = s.info["world"]
p = s.data.qpos[:3]
o = p + jp.array([0, 0, C.LIDAR_OFFSET])
d = drone_env.ray_cast(T, o, jp.array(C.RAY_DIRS), jp.array(C.RAY_RANGES))
print("down ray", float(d[C.N_RING]), "true agl", float(o[2] - drone_env.terrain_h(T, o[:2])))
print("ring min/max", float(d[:C.N_RING].min()), float(d[:C.N_RING].max()), "clearance", float(drone_env.clearance(T, p)))

# batched speed
B = 4096
keys = jax.random.split(jax.random.PRNGKey(0), B)
vr, vs = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
sb = vr(keys); jax.block_until_ready(sb.obs)
act = jp.zeros((B, 4))
sb = vs(sb, act); jax.block_until_ready(sb.obs)
t0 = time.time()
for _ in range(20):
  sb = vs(sb, act)
jax.block_until_ready(sb.obs)
dt = (time.time() - t0) / 20
print(f"batched step {B} envs: {dt*1000:.1f} ms -> {B/dt:,.0f} env-steps/s per GPU")
print("crash frac after 22 hover steps", float(sb.done.mean()), "nan obs", bool(jp.isnan(sb.obs).any()))
r0 = vr(keys); jax.block_until_ready(r0.obs)
t0 = time.time(); r0 = vr(keys); jax.block_until_ready(r0.obs); print(f"batched reset: {(time.time()-t0)*1000:.1f} ms")
