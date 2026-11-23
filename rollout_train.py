"""Roll out a checkpoint in N fixed random training worlds (deterministic pilot). Saves trajectories + worlds
for the training montage, and prints success / crash rates.
  JAX_PLATFORMS=cpu python rollout_train.py runs/pilot1/ckpt/step_xxx.pkl out.npz --n 9 --seed 100"""
import argparse, pickle
import jax, jax.numpy as jp, numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

import drone_env

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=9); ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--steps", type=int, default=1500)
a = ap.parse_args()

env = drone_env.DroneNav()
params = pickle.load(open(a.ckpt, "rb"))
net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                     policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
                                     preprocess_observations_fn=running_statistics.normalize)
policy = ppo_networks.make_inference_fn(net)((params[0], params[1]), deterministic=True)
keys = jax.random.split(jax.random.PRNGKey(a.seed), a.n)
reset = jax.jit(jax.vmap(env.reset))
step = jax.jit(jax.vmap(env.step))
act = jax.jit(jax.vmap(lambda o, k: policy(o, k)[0]))
s = reset(keys)
qpos, done_at, outcome = [np.array(s.data.qpos)], np.full(a.n, -1), np.zeros(a.n, int)
alive = np.ones(a.n, bool)
for t in range(a.steps):
  s2 = step(s, act(s.obs, keys))
  d = np.array(s2.done) > 0
  newly = d & alive
  done_at[newly] = t + 1
  outcome[newly] = np.where(np.array(s2.metrics["success"])[newly] > 0, 1, -1)
  alive &= ~d
  s = s2                                   # finished episodes keep stepping; their poses are frozen below
  qpos.append(np.array(s.data.qpos))
  if not alive.any():
    break
qpos = np.stack(qpos, 1)
for i in range(a.n):                       # hold the last pose after the episode ended
  if done_at[i] > 0:
    qpos[i, done_at[i]:] = qpos[i, done_at[i]]
w = jax.tree.map(np.array, s.info["world"])
np.savez_compressed(a.out, qpos=qpos, done_at=done_at, outcome=outcome, path=np.array(s.info["path"]),
                    ps=np.array(s.info["ps"]), **{f"w_{k}": v for k, v in w.items()})
print(f"{a.ckpt}: success {np.mean(outcome == 1):.2f}  crash {np.mean(outcome == -1):.2f}  "
      f"timeout {np.mean(outcome == 0):.2f}  steps {qpos.shape[1]}")
