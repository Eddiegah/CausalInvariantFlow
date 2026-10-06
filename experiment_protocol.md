# Experiment protocol

## Pre-registration rule

The test set, event thresholds, intervention grid, primary metrics, and model-selection rule must be frozen before final evaluation. Hyperparameters may be selected on train/validation only. Every reported mean must include repeated-seed uncertainty.

## Primary systems

### Lorenz-63

\[
\dot{x}=\sigma(y-x),\quad \dot{y}=x(\rho-z)-y,\quad \dot{z}=xy-\beta z.
\]

Default parameters: \(\sigma=10, \rho=28, \beta=8/3\). Train on a central parameter region; hold out parameter perturbations and forcing interventions. Define a rare event as entering a pre-registered high-energy or lobe-transition set. Report sensitivity to the threshold.

### Lorenz-96 or Kuramoto–Sivashinsky

Use this system to test scalability, partial observations, and spatiotemporal statistics. Report whether conclusions transfer beyond a three-dimensional attractor.

## Data generation

Use a high-accuracy reference solver. Store simulator configuration, tolerances, step size, random seed, initial state, parameters, intervention, and software version. Generate independent trajectories rather than overlapping windows from a single trajectory. Record the largest Lyapunov exponent estimate so horizons can be reported in Lyapunov-time units.

## Baselines

1. Persistence and linear extrapolation.
2. Tuned reservoir computing.
3. GRU/LSTM or neural ODE.
4. Physics-constrained reservoir.
5. Diffusion trajectory model.
6. Unconstrained conditional flow matching.
7. Physics-constrained flow matching.
8. Deep ensemble and a post-hoc calibration method.

All baselines receive matched training data and comparable hyperparameter-search budgets. Include a simple baseline even if it is not expected to win.

## Primary metrics

### Distributional forecasts

- CRPS or energy score.
- Negative log score where a valid density is available.
- Marginal and joint trajectory coverage.
- Forecast sharpness.
- Wasserstein or MMD distance for terminal and path statistics.

### Chaotic dynamics

- Valid prediction time at fixed error threshold in Lyapunov units.
- Invariant-measure error.
- Autocorrelation and power-spectrum error.
- Lyapunov-spectrum error where identifiable.
- Rollout stability and fraction of numerically invalid samples.

### Physics validity

- Governing-equation residual distribution.
- Conservation/invariant drift.
- Boundary and initial-condition violations.
- Inequality/feasibility violation rate.
- Fraction of rejected or repaired samples.

### Rare events

- Brier score and log score for event probability.
- Reliability and expected calibration error for event probability.
- Recall at fixed false-alarm rate.
- Transition-time error and interval coverage.
- Event amplitude/location error.
- Tail-weighted CRPS or energy score.

### Interventions

- Counterfactual state error.
- Event-probability error under intervention.
- Transition-time error under intervention.
- Intervention-conditional coverage.
- Decision regret if a control/action is selected from the forecast.

## Required ablations

- No physics / soft residual / projection or constrained integration.
- Flow matching versus diffusion.
- Calibrated versus uncalibrated output.
- Common-regime loss versus tail-weighted loss.
- Short versus long rollout.
- Full versus partial observation.
- In-distribution versus held-out intervention.
- One versus multiple generated samples per condition.

## Secondary active-learning experiment

Under a fixed simulator-call budget, compare random, space-filling, epistemic-uncertainty, expected-information-gain, rare-event/tail-weighted, and intervention-aware acquisition. The acquisition policy is evaluated on a fixed untouched test set. Primary outcome is improvement in tail calibration and event skill per simulator call, not training loss.

## Statistical analysis

Use at least five seeds for the core benchmark when feasible. Report mean, standard deviation, and bootstrap confidence intervals over independent trajectories or test conditions. Use paired comparisons on identical test conditions. Correct for multiple primary comparisons or pre-register one primary metric per research question. Do not report a win based on a single seed or a single trajectory.

## Stop/rollback rules

Stop claiming improvement if the method wins only on MSE but loses on calibration, physics validity, and tail score. Roll back the scope if the full architecture cannot be trained reproducibly on Lorenz-63. Move the active-learning extension to supplementary material if it requires a different data-generation or evaluation protocol.

## Expected figures

1. System diagrams and intervention semantics.
2. Calibration reliability by horizon and regime.
3. Representative generated paths with physics residual annotations.
4. Attractor/invariant-statistics comparison.
5. Rare-event probability and transition-time calibration.
6. Pareto plot: forecast score, residual, calibration, and simulator cost.
7. Ablation and failure-case matrix.

## Compute plan

Start on CPU/GPU with Lorenz-63. Scale only after a deterministic smoke test, a short training run, and a full metric pass succeed. Cache simulator outputs and never regenerate the test set during hyperparameter tuning.
