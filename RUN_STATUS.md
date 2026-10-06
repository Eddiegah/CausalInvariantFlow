# Local run status

Date: 2026-10-05

## Environment

- CPython: 3.11
- Virtual environment: `.venv`
- PyTorch: 2.14.1+cpu
- CUDA: unavailable; CPU execution verified
- NumPy: 2.4.6
- SciPy: 1.17.1
- pandas: 3.0.6
- matplotlib: 3.11.2
- scikit-learn: 1.9.1
- PyYAML: 6.0.3

## Tests passed

1. **Lorenz-63 data generation**: generated 1,024 states with finite values.
2. **Flow-matching smoke training**: 2 epochs; loss decreased from `247.597621` to `201.521585`.
3. **Checkpoint load**: valid checkpoint with `config` and `state_dict` keys.
4. **ODE sampling**: generated finite samples with shape `(8, 16, 3)`.
5. **Metric checks**: empirical coverage, interval width, Brier score, and reliability bins executed successfully.
6. **Larger short benchmark**: generated 4,096 states and trained for 3 epochs; loss decreased from `216.241740` to `95.174365` to `43.751847`.

## Artifacts

- `results/lorenz63_smoke.npz`
- `results/lorenz63_smoke_model.pt`
- `results/lorenz63_medium.npz`
- `results/lorenz63_medium_model.pt`

The generated datasets and checkpoints are intentionally ignored by Git. They are local experiment artifacts, not source files.

## Interpretation

These are **pipeline smoke and scaling checks**, not publishable scientific results. The next research-grade stage must implement independent train/calibration/test trajectories, intervention grids, solver-verified physics residuals, rare-event labels, repeated seeds, proper baseline comparisons, and final figures before any manuscript claim is updated.
