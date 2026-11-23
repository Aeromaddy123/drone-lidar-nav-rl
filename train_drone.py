"""Train the drone pilot with PPO (Brax) on all visible GPUs. Saves params at every evaluation."""
import argparse, functools, json, pathlib, pickle, time
import jax, jax.numpy as jnp
import numpy as np


def _device_put_replicated(x, devices):   # Brax 0.14 still calls this; JAX 0.10 removed it
  mesh = jax.sharding.Mesh(np.array(devices), ("d",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
  return jax.tree.map(lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * len(devices)), sharding), x)
jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

import drone_env

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--steps", type=float, default=300e6)
ap.add_argument("--evals", type=int, default=25)
ap.add_argument("--envs", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--restore", default=None)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--entropy", type=float, default=5e-3)
ap.add_argument("--lr_schedule", default="ADAPTIVE_KL")
args = ap.parse_args()

out = pathlib.Path(__file__).parent / "runs" / args.name
(out / "ckpt").mkdir(parents=True, exist_ok=True)
env = drone_env.DroneNav()
eval_env = drone_env.DroneNav()
ppo_kw = dict(num_timesteps=int(args.steps), num_evals=args.evals, num_envs=args.envs, episode_length=1500,
              action_repeat=1, learning_rate=args.lr, entropy_cost=args.entropy, discounting=0.99,
              unroll_length=20, batch_size=512, num_minibatches=32, num_updates_per_batch=4,
              normalize_observations=True, reward_scaling=1.0, clipping_epsilon=0.2, max_grad_norm=1.0,
              gae_lambda=0.95, num_eval_envs=2048, deterministic_eval=True,
              learning_rate_schedule=args.lr_schedule, desired_kl=0.01,
              learning_rate_schedule_min_lr=1e-5, learning_rate_schedule_max_lr=1e-3,
              seed=args.seed)
net_kw = dict(policy_hidden_layer_sizes=(256, 256, 128), value_hidden_layer_sizes=(512, 256, 128))
(out / "args.json").write_text(json.dumps({**vars(args), "ppo": ppo_kw, "network": net_kw,
                                           "devices": [str(d) for d in jax.devices()]}, indent=2))
t0 = time.time(); log = open(out / "progress.jsonl", "a")


def progress(step, metrics):
  rec = {"step": int(step), "minutes": round((time.time() - t0) / 60, 2)}
  rec.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/")})
  log.write(json.dumps(rec) + "\n"); log.flush()
  print(f"[{rec['minutes']:6.1f} min] step {step:>12,}  reward {rec.get('eval/episode_reward', 0):7.2f}  "
        f"success {rec.get('eval/episode_success', 0):5.3f}  crash {rec.get('eval/episode_crash', 0):5.3f}  "
        f"route {rec.get('eval/episode_completion', 0):5.3f}  len {rec.get('eval/avg_episode_length', 0):6.0f}  "
        f"sps {rec.get('eval/sps', 0):8.0f}", flush=True)


def save_params(step, make_policy, params):
  with open(out / "ckpt" / f"step_{int(step):012d}.pkl", "wb") as f:
    pickle.dump(jax.device_get(params), f)


net = functools.partial(ppo_networks.make_ppo_networks, **net_kw)
ppo.train(environment=env, eval_env=eval_env,
          wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
          network_factory=net, progress_fn=progress, policy_params_fn=save_params,
          restore_params=pickle.load(open(args.restore, "rb")) if args.restore else None, **ppo_kw)
print("TRAINING_DONE", flush=True)
