"""Why does the pilot still crash? Roll out N training worlds and classify each failure."""
import argparse, pickle
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import drone_env, common as C

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--n", type=int, default=256); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
env = drone_env.DroneNav()
params = pickle.load(open(a.ckpt, "rb"))
net = ppo_networks.make_ppo_networks(env.observation_size, 4, policy_hidden_layer_sizes=(256, 256, 128),
                                     value_hidden_layer_sizes=(512, 256, 128), preprocess_observations_fn=running_statistics.normalize)
policy = ppo_networks.make_inference_fn(net)((params[0], params[1]), deterministic=True)
keys = jax.random.split(jax.random.PRNGKey(a.seed), a.n)
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
act = jax.jit(jax.vmap(lambda o, k: policy(o, k)[0]))


def diag(s):
  def one(s):
    p, v = s.data.qpos[:3], s.data.qvel[:3]
    T = s.info["world"]
    R = C.quat_to_mat(jp, s.data.qpos[3:7] / jp.linalg.norm(s.data.qpos[3:7]))
    c1, c2, rem, fin = C.carrots(jp, s.info["path"], s.info["ps"], s.info["s"], s.info["look"])
    return dict(clr=drone_env.clearance(T, p), agl=p[2] - drone_env.terrain_h(T, p[:2]), vz=v[2], speed=jp.linalg.norm(v),
                up=R[2, 2], fin=fin, rem=rem, dgoal=jp.linalg.norm(p[:2] - s.info["path"][-1, :2]))
  return jax.vmap(one)(s)


diag = jax.jit(diag)
s = reset(keys)
alive = np.ones(a.n, bool)
rows = [None] * a.n
prev = diag(s)
for t in range(1500):
  s2 = step(s, act(s.obs, keys))
  d = np.array(s2.done) > 0
  newly = d & alive
  if newly.any():
    dg = jax.tree.map(np.array, prev)           # state just before the terminal step
    dn = jax.tree.map(np.array, diag(s2))
    suc = np.array(s2.metrics["success"])
    for i in np.nonzero(newly)[0]:
      rows[i] = dict(t=t, success=bool(suc[i] > 0), **{k: float(dn[k][i]) for k in dn}, **{"pre_" + k: float(dg[k][i]) for k in dg})
  alive &= ~d
  prev = diag(s2)
  s = s2
  if not alive.any():
    break
ok = sum(1 for r in rows if r and r["success"])
cr = [r for r in rows if r and not r["success"]]
print(f"success {ok}/{a.n}, crashes {len(cr)}, timeouts {sum(1 for r in rows if r is None)}")
cause = {"tree/rock": 0, "ground impact while landing": 0, "ground impact en route": 0, "flipped/out": 0}
for r in cr:
  if r["clr"] < 0:
    cause["tree/rock"] += 1
  elif r["agl"] <= 0.02 and r["pre_fin"] > 0:
    cause["ground impact while landing"] += 1
  elif r["agl"] <= 0.02:
    cause["ground impact en route"] += 1
  else:
    cause["flipped/out"] += 1
print(cause)
land = [r for r in cr if r["agl"] <= 0.02 and r["pre_fin"] > 0]
if land:
  print("landing crashes: pre speed %.2f, pre vz %.2f, dgoal %.2f" % (np.mean([r["pre_speed"] for r in land]), np.mean([r["pre_vz"] for r in land]), np.mean([r["dgoal"] for r in land])))
er = [r for r in cr if r["agl"] <= 0.02 and r["pre_fin"] == 0]
if er:
  print("en-route ground: pre agl %.2f pre vz %.2f rem %.1f" % (np.mean([r["pre_agl"] for r in er]), np.mean([r["pre_vz"] for r in er]), np.mean([r["rem"] for r in er])))
tr = [r for r in cr if r["clr"] < 0]
if tr:
  print("tree/rock: pre speed %.2f, pre agl %.2f" % (np.mean([r["pre_speed"] for r in tr]), np.mean([r["pre_agl"] for r in tr])))
