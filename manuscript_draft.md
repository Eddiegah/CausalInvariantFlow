# Manuscript draft

## Title

**Calibrated Physics-Constrained Flow Matching for Rare-Event Forecasting in Chaotic Dynamical Systems under Known Interventions**

## Abstract

Accurate point prediction in chaotic systems is fundamentally horizon-limited, but probabilistic forecasts of long-run statistics, regime transitions, and rare events remain valuable. We introduce [MODEL NAME], a conditional flow-matching framework that combines generative trajectory prediction with solver-verified physics constraints and held-out calibration. The model is evaluated under known parameter and forcing interventions on [SYSTEMS]. Unlike evaluations based only on average trajectory error, we report distributional forecast scores, Lyapunov-time scaling, invariant/statistical fidelity, physics residuals, event-probability calibration, transition-time error, and rare-path quality. We compare against tuned simple baselines, reservoir computing, neural ODEs, diffusion, unconstrained flow matching, and ensemble uncertainty baselines. [RESULTS TO BE FILLED FROM COMMITTED EXPERIMENT ARTIFACTS.] We find that [PRIMARY RESULT]. The results indicate [CONCLUSION], while exposing the limitation that [LIMITATION].

## 1. Introduction

Chaotic systems amplify small state errors, so a useful scientific forecaster should not be judged solely by whether a single trajectory remains close for arbitrarily long times. It should also represent uncertainty, preserve relevant dynamics, and estimate the probability and path of rare transitions. Prior work shows that physics-constrained recurrent models can improve long-term statistics and extreme-event prediction in selected turbulent systems [1], while diffusion models can support uncertainty quantification and event-conditioned sampling for low-dimensional chaotic dynamics [2]. Flow matching provides an efficient continuous-time generative formulation [3], but its use for calibrated, physics-verified rare-event path forecasting in chaotic systems remains insufficiently tested.

We study the following question: can a conditional flow-matching model improve the joint trade-off among distributional forecast quality, physical validity, uncertainty calibration, and rare-event skill under known interventions? Our contribution is deliberately narrower than a claim of general causal discovery. The simulator supplies intervention semantics and counterfactual ground truth; the model is evaluated on whether its conditional forecasts respond correctly.

Our contributions are:

1. A conditional generative forecasting framework with an explicitly separated flow-matching objective, physics-validity mechanism, and calibration layer.
2. A benchmark protocol that evaluates chaotic forecasts in Lyapunov units and includes distributional, physical, calibration, and rare-event metrics.
3. A controlled study of the trade-off between physics enforcement and rare-tail fidelity under held-out interventions.
4. An optional simulator-budget experiment comparing random, uncertainty-aware, and tail-aware acquisition.

## 2. Related work

[Convert the literature-gap discussion in the proposal into a 1,000–1,500 word related-work section. Keep the claims bounded by the cited papers and distinguish point forecasting, distributional forecasting, event-conditioned sampling, and causal intervention.]

## 3. Problem formulation

We consider a known simulator \(\dot{x}=f(x;\theta,u)\) and forecast a future path conditional on history, parameters, and intervention. The target is the conditional distribution \(p(x_{1:H}\mid h_0,\theta,u)\), not a single deterministic trajectory. Events are defined before training as threshold crossings or transitions between pre-specified basins.

## 4. Method

### 4.1 Conditional flow matching

[Insert implementation details: trajectory representation, base distribution, conditional path, vector-field parameterization, solver, integration tolerances, and number of generated samples.]

### 4.2 Physics enforcement

We compare: (a) no physics term; (b) soft residual penalty; (c) projection/constrained integration; and (d) post-hoc rejection/repair. Physics is evaluated independently after generation. We report residual distributions rather than only a mean penalty.

### 4.3 Calibration

Calibration uses a separate split. For each horizon and event regime, we report empirical coverage, interval/region sharpness, CRPS or energy score, and event-probability reliability. We state the exchangeability or stationarity assumptions of any conformal procedure and do not claim unconditional guarantees under arbitrary chaotic nonstationarity.

### 4.4 Known interventions

Interventions alter simulator parameters, forcing, initial conditions, or controls. We evaluate counterfactual trajectory error and event-probability error against solver-generated ground truth. We use “intervention-conditioned forecasting,” not causal discovery.

## 5. Experimental setup

### Systems

- Lorenz-63: exact low-dimensional intervention benchmark.
- Lorenz-96 or Kuramoto–Sivashinsky: spatiotemporal chaos benchmark.
- Optional extreme-event system: include only if solver verification and sufficient tail samples are available.

### Baselines

Persistence/linear extrapolation; tuned reservoir computing; GRU/LSTM or neural ODE; physics-constrained reservoir; diffusion; unconstrained flow matching; deep ensemble and Laplace/post-hoc calibration baselines.

### Splits

Use trajectory-level disjoint splits. Reserve calibration data separately. Test on held-out initial states, parameter values, intervention combinations, longer horizons, and tail-held-out regimes. Never randomly mix windows from one trajectory across train and test.

## 6. Results

This section must be written only after experiments. Required subsections:

1. Distributional forecast quality.
2. Physical validity and invariant drift.
3. Calibration by horizon and regime.
4. Rare-event probability, transition time, and path quality.
5. Intervention robustness.
6. Compute and simulator cost.
7. Ablations and failure cases.

Use the following result sentence template only after verifying the underlying files:

> Across [N] seeds and [M] test conditions, [MODEL] achieved [VALUE] versus [BASELINE] on [METRIC], with a [CONFIDENCE INTERVAL] difference. The improvement [did/did not] persist under [OOD/intervention condition].

## 7. Discussion

Interpret results separately for phase prediction, invariant/statistical fidelity, and rare-event probability. A model may lose phase accuracy after several Lyapunov times while still preserving useful distributions. Discuss whether physics constraints improve validity at the cost of suppressing valid tail paths. Explain calibration failures under nonstationarity and interventions. Do not generalize from controlled simulators to real-world deployment without a separate measurement-error and simulator-discrepancy study.

## 8. Limitations

The benchmark uses known equations and may not represent hidden variables, measurement error, model discrepancy, or unknown causal graphs. Conformal calibration assumptions may fail under strong nonstationarity. Soft physics constraints do not guarantee exact invariants. Rare-event metrics can be sensitive to threshold and sample count. Flow-matching efficiency is implementation- and solver-dependent.

## References

[1]: https://doi.org/10.1098/rspa.2021.0135 "Physics-constrained reservoir computing for chaotic flows and extreme events"
[2]: https://arxiv.org/abs/2306.07526 "User-defined Event Sampling and Uncertainty Quantification in Diffusion Models for Physical Dynamical Systems"
[3]: https://arxiv.org/abs/2210.02747 "Flow Matching for Generative Modeling"
[4]: https://arxiv.org/abs/2407.20158 "A benchmark study of machine learning for chaotic dynamical systems"
[5]: https://doi.org/10.1038/s43588-022-00376-0 "Active learning for discovering and forecasting extreme events"
