"""Generate reproducible Lorenz-63 trajectories for the first smoke benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t: float, state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float)


def generate(seed: int, n: int, dt: float, burn_in: float, sigma: float, rho: float, beta: float):
    rng = np.random.default_rng(seed)
    initial = rng.normal(0.0, 1.0, size=3)
    initial[2] += 25.0
    end = burn_in + (n - 1) * dt
    times = np.arange(0.0, end + 0.5 * dt, dt)
    sol = solve_ivp(
        lambda t, y: lorenz(t, y, sigma, rho, beta),
        (0.0, float(times[-1])),
        initial,
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    start = int(round(burn_in / dt))
    return times[start : start + n], sol.y.T[start : start + n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/lorenz63.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--burn-in", type=float, default=50.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)
    args = parser.parse_args()
    times, states = generate(args.seed, args.n, args.dt, args.burn_in, args.sigma, args.rho, args.beta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, t=times, x=states)
    meta = vars(args).copy(); meta["out"] = str(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out} with shape {states.shape}")


if __name__ == "__main__":
    main()
