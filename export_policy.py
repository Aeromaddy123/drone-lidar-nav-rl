"""Brax PPO checkpoint (.pkl) -> plain NumPy arrays (.npz) for the live demo. Run with JAX_PLATFORMS=cpu."""
import pickle, re, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
norm, policy, _ = pickle.load(open(src, "rb"))
layers = policy["params"]
names = sorted(layers, key=lambda k: int(k.split("_")[1]))
out = {"mean": np.asarray(norm.mean, np.float32), "std": np.asarray(norm.std, np.float32), "n": len(names)}
for i, k in enumerate(names):
  out[f"w{i}"] = np.asarray(layers[k]["kernel"], np.float32)
  out[f"b{i}"] = np.asarray(layers[k]["bias"], np.float32)
m = re.search(r"step_(\d+)", src)
out["step"] = int(m.group(1)) if m else -1
np.savez(dst, **out)
print("exported", dst, "layers", [out[f"w{i}"].shape for i in range(len(names))], "step", out["step"])
