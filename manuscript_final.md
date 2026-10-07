# CausalInvariantFlow: Calibrated Physics-Constrained Flow Matching for Rare-Event Forecasting in Chaotic Dynamical Systems under Known Interventions

**Manuscript type:** Reproducible research protocol and methods paper  
**Repository:** https://github.com/Eddiegah/CausalInvariantFlow

## Abstract

Chaotic systems impose a hard limit on long-horizon pointwise prediction because small state errors grow exponentially. This limit does not make forecasting useless. It changes the target: a useful forecaster should represent distributions of future trajectories, preserve dynamical structure, quantify uncertainty, and estimate the probability and timing of rare transitions. We present **CausalInvariantFlow**, a conditional flow-matching framework for this setting. The method combines generative trajectory prediction with solver-verified physics evaluation, horizon- and regime-conditional calibration, and explicit rare-event scoring under known parameter or forcing interventions. The contribution is intentionally narrow. We do not claim causal discovery, universal invariant preservation, or indefinite phase accuracy. Instead, we define a controlled benchmark in which the simulator exposes intervention semantics and supplies counterfactual ground truth.

The paper specifies a preregistered evaluation across Lorenz-63 and a higher-dimensional chaotic system such as Lorenz-96 or Kuramoto–Sivashinsky. It compares persistence and linear extrapolation, reservoir computing, recurrent or neural-ODE baselines, diffusion trajectory models, unconstrained flow matching, physics-constrained flow matching, and calibrated ensembles. The primary measurements are distributional forecast scores, coverage and sharpness, valid prediction time in Lyapunov units, invariant and spectral drift, governing-equation residuals, event-probability calibration, transition-time error, tail-weighted path quality, intervention robustness, and simulator cost. We also specify ablations that test whether physics enforcement improves validity without suppressing legitimate rare paths, and whether explicit calibration is necessary beyond a physics loss. The current repository contains the manuscript design, data-generation utilities, metrics, starter training code, and a runtime-independent Lorenz smoke test. Full empirical superiority claims are deliberately withheld until the preregistered benchmark is completed.

**Keywords:** chaotic dynamics; flow matching; uncertainty quantification; rare events; physics-informed machine learning; intervention-conditioned forecasting; calibration; Lorenz-63

## 1. Introduction

A deterministic forecast of a chaotic system can become useless after only a few characteristic growth times, even when the governing equations are known exactly. The reason is not necessarily model inadequacy. Small errors in the initial state, parameters, observations, or numerical representation are amplified by the dynamics. A method that is judged only by pointwise trajectory error will therefore be penalized for a limitation shared by the underlying system.

The scientific forecasting problem is broader than phase tracking. After pointwise accuracy has degraded, a model may still be useful if it represents the future distribution, preserves invariant or statistical structure, estimates the probability of a transition, and identifies the conditions under which a rare event becomes more likely. These goals matter in fluid dynamics, climate dynamics, engineered systems, and any application in which the cost of missing an extreme transition is larger than the cost of a modest average error.

Existing work supplies several relevant ingredients. Flow Matching trains continuous normalizing flows by regressing vector fields along prescribed conditional probability paths, avoiding the need to simulate the learned dynamics during training [1]. Physics-constrained reservoir methods have shown that physical information can improve long-term statistics and extreme-event prediction in chaotic flows [2]. Diffusion-based trajectory models can represent uncertainty and support conditional sampling of nonlinear events, including tail events that are difficult to query directly [3]. Active-learning approaches can reduce the simulator budget required to characterize rare outcomes by selecting informative or output-weighted samples [4]. These results are important, but they address different parts of the problem and use different evaluation standards.

The unresolved question is whether a single forecasting system can make a credible joint trade-off among four properties: distributional accuracy, physical validity, uncertainty calibration, and rare-event skill. A model may improve a residual penalty while suppressing valid tail paths. It may produce wide prediction bands that achieve coverage but are not sharp. It may perform well in the training regime while misrepresenting event probabilities under parameter or forcing changes. These failure modes cannot be identified by mean squared error alone.

We introduce CausalInvariantFlow as an evaluation-centered framework for this question. The model is conditional on an observed history, known simulator parameters, and a known intervention or forcing. It generates multiple future trajectories using flow matching. Generated paths are checked with an independent solver or residual evaluator. A separate calibration split is used to calibrate trajectory regions and event probabilities. Rare events are defined before training using fixed thresholds or basin-transition rules. The final test set contains held-out initial states, parameter values, intervention combinations, and longer horizons.

The paper makes four contributions:

1. It defines a conditional generative forecasting architecture that separates flow matching, physical verification, and uncertainty calibration.
2. It proposes a benchmark that reports chaotic forecast quality in Lyapunov-time units and evaluates distributions, invariants, residuals, calibration, rare events, and interventions together.
3. It specifies ablations that measure the trade-off between physics enforcement and rare-tail fidelity rather than assuming that a smaller residual implies a better forecast.
4. It provides a reproducible implementation scaffold and explicit stopping rules that prevent unsupported claims when the full architecture or benchmark is not yet stable.

The framework uses the phrase **known interventions** rather than causal discovery. The simulator defines what changes when a parameter, forcing function, initial condition, or control is modified. The model is evaluated on whether its conditional forecasts respond correctly to those changes. No causal graph is inferred from observational data.

## 2. Related work

### 2.1 Flow matching and generative trajectory prediction

Flow Matching formulates continuous normalizing-flow training as regression of a vector field along a chosen conditional probability path [1]. The method is simulation-free during training and can use Gaussian paths that include diffusion-like constructions as special cases. Its central advantage for scientific forecasting is conceptual as well as computational: the model learns a transport field from a simple base distribution to a data distribution, which naturally supports multiple samples for the same forecast condition.

Generative trajectory models have often been evaluated in domains such as motion prediction, where multimodality is important but the underlying system is not necessarily chaotic in the dynamical-systems sense. Those results establish that flow-based and diffusion-based models can represent multiple plausible futures, but they do not establish that generated paths preserve invariant measures, satisfy governing equations, or maintain calibrated rare-event probabilities after long rollouts. CausalInvariantFlow therefore treats the generative model as one component of a scientific forecasting system rather than as the complete contribution.

### 2.2 Forecasting beyond the predictability horizon

Machine-learning models can reproduce selected long-run statistics of chaotic systems even when phase-aligned prediction is no longer possible. This distinction motivates reporting invariant-measure error, autocorrelation, spectra, and event statistics alongside valid prediction time. Physics-constrained reservoir computing provides a particularly relevant precedent. In a chaotic flow model, adding physical information to reservoir training improved velocity statistics and prediction of extreme-event occurrence and amplitude [2]. The result does not imply that physics constraints solve chaos. It shows that physical structure can improve the statistical and event-level behavior of a learned surrogate.

CausalInvariantFlow extends this perspective to conditional generative forecasts. The relevant comparison is not only whether a model stays close to one reference trajectory. It is whether its ensemble of paths reproduces the distribution of future states, the frequency and timing of transitions, and the response to controlled changes in parameters or forcing. This also creates a sharper test of physics enforcement: residual reduction is useful only if it does not remove valid regions of the future distribution.

### 2.3 Diffusion models and rare-event sampling

Diffusion models have been adapted to chaotic dynamical systems for uncertainty quantification and event-conditioned sampling [3]. A central difficulty is that conditioning on a nonlinear event is not equivalent to simply forcing a generated path to satisfy a constraint. The conditional distribution must remain statistically consistent with the underlying dynamics, including in the tails. Approximate conditional-score methods offer a way to sample rare events while retaining distributional information [3].

This literature motivates two design choices in the present work. First, event probability must be evaluated as a probabilistic forecasting task, using proper scores and reliability analysis. Second, a model should be evaluated on the quality of full event paths, not only on whether it can generate one visually plausible extreme trajectory. Flow matching and diffusion are compared as alternative generative mechanisms under matched data and evaluation budgets.

### 2.4 Physics-informed learning and calibration

Physics-informed methods can inject governing equations through residual penalties, constraints, projections, or specialized architectures. These mechanisms differ in what they guarantee. A soft penalty can reduce an average residual without enforcing exact validity on every sample. Projection can improve feasibility but may distort the data distribution. Post-hoc rejection can remove invalid paths while changing event frequencies. For this reason, the benchmark reports residual distributions, invariant drift, rejection or repair rates, and tail scores together.

Uncertainty calibration is a separate problem. A physics loss does not imply calibrated probabilities, and a generative model can be overconfident even when its samples have small residuals. The calibration layer is therefore trained only on a held-out calibration split. It may use split-conformal trajectory regions, calibrated regression, or a validated ensemble/post-processing method. Any coverage interpretation will state its exchangeability or stationarity assumptions. In particular, no unconditional guarantee is claimed under arbitrary nonstationarity or intervention shift.

### 2.5 Active learning for extreme events

Rare-event studies often face a simulator-budget problem: random sampling may produce too few informative extremes. Output-weighted Bayesian experimental design and ensembles of neural operators have been used to identify extreme outcomes efficiently [4]. These methods motivate a secondary experiment in which random, space-filling, uncertainty-aware, information-gain, tail-weighted, and intervention-aware acquisition policies are compared under the same simulator-call budget.

Active acquisition is not part of the core novelty claim. It is an optional extension that tests whether the proposed evaluation framework can measure improvement in tail skill per simulator call. It will be moved to supplementary material if it requires a different data-generation or evaluation protocol.

## 3. Problem formulation

Let a known simulator be written as

\[
\frac{d x(t)}{d t}=f\big(x(t);\theta,u(t)\big), \qquad x(0)=x_0,
\]

where \(x(t)\) is the state, \(\theta\) contains system parameters, and \(u(t)\) is a known intervention or forcing. Given an observed history \(h_t\), parameters \(\theta\), intervention \(u\), and forecast horizon \(H\), the target is the conditional future distribution

\[
p\big(x_{t+1:t+H}\mid h_t,\theta,u\big).
\]

The target is not a single deterministic continuation. Multiple future paths can be valid because the forecast condition is incomplete, observations may be noisy or partial, and chaotic amplification makes small unresolved differences consequential.

An event is defined before training. Examples include entering a high-energy set, crossing a threshold, or transitioning between pre-specified basins. For a generated path \(\hat{x}_{1:H}^{(m)}\), the event indicator is \(y^{(m)}\in\{0,1\}\), and the model estimates an event probability by the fraction of generated paths satisfying the event rule. Event thresholds, basin definitions, intervention grids, and test conditions are frozen before final evaluation.

## 4. CausalInvariantFlow method

### 4.1 Conditional flow matching

The model receives a condition vector

\[
c=\operatorname{Enc}(h_t,\theta,u),
\]

and generates a future trajectory with shape \(H\times d\), where \(d\) is the state dimension. A simple Gaussian base variable \(z_0\) is transported toward the target trajectory \(z_1\) along a conditional path. For the initial implementation, the path is linear interpolation:

\[
z_s=(1-s)z_0+s z_1, \qquad s\sim\operatorname{Uniform}(0,1).
\]

The vector field \(v_\phi(z_s,s\mid c)\) is trained with

\[
\mathcal{L}_{FM}=\mathbb{E}\left[\left\|v_\phi(z_s,s\mid c)-(z_1-z_0)\right\|_2^2\right].
\]

The starter implementation uses a Fourier embedding of time and a multilayer perceptron conditioned on the current trajectory representation and the observed state. The final study will specify the history encoder, condition encoder, trajectory representation, solver, tolerances, number of generated samples, and hardware before model selection.

At inference time, an ODE solver integrates the learned vector field from the base distribution to generate multiple future trajectories. The initial Euler sampler is a reproducibility scaffold, not a final claim about solver quality. The final benchmark will compare a stable solver configuration across all generative models and report the effect of solver tolerances separately.

### 4.2 Physics enforcement and verification

Four variants are compared:

1. **Unconstrained flow matching:** the generative model is trained without a physics term.
2. **Soft residual penalty:** the objective includes a governing-equation residual evaluated on decoded trajectories.
3. **Projection or constrained integration:** generated states are corrected or integrated subject to the known dynamics.
4. **Post-hoc rejection or repair:** invalid paths are removed or repaired after generation, with the rejection rate reported.

The general augmented objective is

\[
\mathcal{L}=\mathcal{L}_{FM}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{tail}\mathcal{L}_{tail}+\lambda_{reg}\mathcal{L}_{reg}.
\]

The physics term may include a residual component

\[
\mathcal{L}_{phys}^{res}=\mathbb{E}\left[\left\|\partial_t\hat{x}(t)-f(\hat{x}(t);\theta,u(t))\right\|_2^2\right],
\]

and an invariant or feasibility component where the system provides one. Every final sample is evaluated independently from the training penalty. The paper will report the full residual distribution, initial-condition violations, invariant drift, numerical-invalid rate, and fraction of rejected or repaired samples.

### 4.3 Calibration

Calibration is fit on data that are not used for training or final testing. For each horizon and event regime, the study reports empirical coverage and sharpness of trajectory regions, CRPS or energy score, and reliability of event probabilities. If conformal calibration is used, the calibration split and the assumptions required for its interpretation will be stated explicitly.

Calibration is evaluated under both in-distribution conditions and held-out interventions. A model is not considered successful merely because it achieves nominal coverage with very broad regions. The desired outcome is calibrated and relatively sharp uncertainty with good tail-event discrimination.

### 4.4 Intervention-conditioned forecasting

Interventions alter parameters, forcing, initial conditions, or controls whose meanings are known from the simulator. Training and validation use a declared region of the intervention space. Testing includes held-out parameter values, intervention combinations, and longer rollout horizons. Performance is measured by counterfactual state error, intervention-conditional coverage, event-probability error, and transition-time error.

The term intervention-conditioned forecasting is used deliberately. The framework does not infer causal direction, identify hidden confounding, or estimate a causal effect from observational data.

## 5. Experimental design

### 5.1 Benchmark systems

The first benchmark is Lorenz-63:

\[
\dot{x}=\sigma(y-x),\qquad
\dot{y}=x(\rho-z)-y,\qquad
\dot{z}=xy-\beta z,
\]

with the standard parameters \(\sigma=10\), \(\rho=28\), and \(\beta=8/3\). This system provides an exact low-dimensional setting for interventions, solver verification, lobe transitions, and high-energy events.

A second system, Lorenz-96 or Kuramoto–Sivashinsky, is added to test scalability, partial observations, and spatiotemporal statistics. A separate extreme-event system is included only if solver verification and tail sample counts are sufficient. Conclusions are not transferred from Lorenz-63 to real-world systems without an additional study of observation noise, hidden variables, and simulator discrepancy.

### 5.2 Data generation and splits

Reference trajectories are generated with a high-accuracy solver. Each trajectory records the software version, tolerances, step size, seed, initial state, parameters, intervention, and simulator configuration. Independent trajectories are used for train, validation, calibration, and test splits. Overlapping windows from a single trajectory are not randomly mixed across splits.

The largest Lyapunov exponent is estimated for each benchmark so that horizons can be reported in Lyapunov-time units. The test set is frozen before the final evaluation. Event thresholds and intervention grids are also frozen before training on the final benchmark.

### 5.3 Baselines

The baseline suite contains persistence and linear extrapolation; tuned reservoir computing; a GRU, LSTM, or neural ODE; physics-constrained reservoir computing where feasible; a diffusion trajectory model; unconstrained conditional flow matching; physics-constrained flow matching; and deep ensembles with post-hoc calibration. All models receive the same training data, comparable hyperparameter-search budgets, and identical final test conditions.

The simple baselines are retained even if they are not expected to win. They provide a safeguard against claiming progress that disappears under a transparent reference method.

### 5.4 Primary metrics

Distributional quality is measured with CRPS or energy score, marginal and joint coverage, sharpness, and Wasserstein or MMD distance for terminal and path statistics. When a valid density is available, a negative log score is also reported.

Chaotic-dynamics quality is measured with valid prediction time at a fixed error threshold in Lyapunov units, invariant-measure error, autocorrelation and power-spectrum error, Lyapunov-spectrum error where identifiable, rollout stability, and the fraction of numerically invalid samples.

Physical validity is measured with governing-equation residual distributions, conservation or invariant drift, initial and boundary violations, feasibility violations, and rejection or repair rates.

Rare-event quality is measured with Brier and log scores for event probability, reliability and expected calibration error, recall at a fixed false-alarm rate, transition-time error and interval coverage, event amplitude or location error, and tail-weighted CRPS or energy score.

Intervention robustness is measured with counterfactual state error, event-probability error, transition-time error, intervention-conditional coverage, and decision regret when a control is selected from the forecast.

### 5.5 Statistical analysis

The core benchmark uses at least five random seeds when feasible. Results are aggregated over independent trajectories or test conditions, not over correlated windows treated as independent observations. We report means, standard deviations, and bootstrap confidence intervals. Paired comparisons use identical test conditions across models. Multiple primary comparisons are corrected, or one primary metric is declared for each research question before evaluation.

A model is not declared a winner based on one seed, one trajectory, or MSE alone. If a method improves point error but loses calibration, physical validity, and tail score, the claim of overall improvement is withdrawn.

## 6. Results and reporting plan

The current repository contains the research design and a reproducible starter implementation. The implementation includes deterministic Lorenz-63 data generation, core forecast and calibration metrics, a conditional flow-matching vector field, a starter training loop, and a runtime-independent simulator smoke test. These artifacts establish the computational scaffold but do not constitute the final benchmark.

The full results section will be populated only from committed experiment artifacts. It will contain the following analyses:

1. **Distributional forecast quality:** proper scores, coverage, sharpness, and path-statistics distance across horizons.
2. **Physical validity and invariant drift:** residual distributions, invariant errors, numerical stability, and rejection or repair rates.
3. **Calibration by horizon and regime:** reliability diagrams, event-probability scores, and conditional coverage.
4. **Rare-event performance:** event probability, transition-time distribution, event amplitude, and tail-weighted path quality.
5. **Intervention robustness:** performance under held-out parameters, forcing, and intervention combinations.
6. **Compute and simulator cost:** training time, sampling cost, solver calls, and tail skill per simulator call.
7. **Ablations and failure cases:** the effect of physics variants, calibration, tail weighting, observation completeness, rollout length, and sample count.

The primary result statement will follow this form after verification:

> Across [number] seeds and [number] test conditions, [model] achieved [value] versus [baseline] on [metric], with a [confidence interval] difference. The improvement [did/did not] persist under [held-out intervention or out-of-distribution condition].

No numerical superiority claim is made in this manuscript before those fields are filled from reproducible result files.

## 7. Discussion

The central test is a joint one. A useful chaotic forecaster should not merely produce low average error. It should represent uncertainty that is calibrated by horizon and regime, preserve relevant statistical structure, and estimate rare-event probabilities without eliminating valid tail paths.

Physics enforcement may improve residuals and invariant drift while damaging distributional fidelity if it over-constrains the generator. Conversely, tail-weighted training may improve event probability and transition-time scores while worsening common-regime average error. These are not implementation nuisances. They are the scientific trade-offs the benchmark is designed to expose.

The intervention evaluation also has a precise interpretation. If the model correctly changes its forecast when the simulator parameter or forcing changes, it has demonstrated intervention-conditioned forecasting on the declared benchmark. That result should not be described as causal discovery. A stronger causal claim would require a formal structural model, a causal estimand, and assumptions about hidden variables and measurement processes.

The proposed framework can stand out scientifically if it remains auditable. The novelty is not the isolated use of flow matching, physics penalties, conformal calibration, rare-event sampling, or active learning. The defensible contribution is their controlled integration with a benchmark that can reveal when they conflict. This makes negative results informative: a physics layer that improves residuals but worsens calibrated tail forecasts is still a meaningful result if the trade-off is measured correctly.

## 8. Limitations

The initial benchmark uses known equations and therefore does not capture hidden variables, severe measurement error, unknown model discrepancy, or ambiguity in intervention semantics. Lorenz-63 is useful for exact verification but is not representative of all high-dimensional chaotic systems. A second spatiotemporal benchmark is necessary before making broader claims.

Calibration procedures can fail under nonstationarity, dependence, or strong intervention shift. Soft physics penalties do not guarantee exact invariants. Rare-event scores are sensitive to threshold definitions and sample counts. Flow-matching efficiency depends on trajectory representation, vector-field parameterization, and ODE solver settings. These choices will be reported rather than treated as universal properties of flow matching.

Finally, a reproducible starter implementation is not the same as a completed empirical study. The repository and this manuscript intentionally distinguish the proposed experiment from results that have actually been run, inspected, and committed.

## 9. Conclusion

CausalInvariantFlow defines a rigorous route toward probabilistic forecasting of chaotic systems when pointwise prediction is horizon-limited. It combines conditional flow matching with independent physics verification, held-out calibration, rare-event scoring, and known-intervention evaluation. Its central scientific question is whether these components improve the joint trade-off among forecast quality, physical validity, uncertainty calibration, rare-event skill, and simulator cost.

The framework is designed to make both success and failure publishable. It does not assume that a smaller physics residual means a better probabilistic forecast, that nominal coverage implies sharp uncertainty, or that correct intervention responses constitute causal discovery. Those claims will be tested through frozen splits, repeated seeds, proper scores, Lyapunov-scaled horizons, solver checks, and transparent ablations.

## References

[1]: https://arxiv.org/abs/2210.02747 "Flow Matching for Generative Modeling"

[2]: https://doi.org/10.1098/rspa.2021.0135 "Short- and long-term predictions of chaotic flows and extreme events: a physics-constrained reservoir computing approach"

[3]: https://arxiv.org/abs/2306.07526 "User-defined Event Sampling and Uncertainty Quantification in Diffusion Models for Physical Dynamical Systems"

[4]: https://doi.org/10.1038/s43588-022-00376-0 "Discovering and forecasting extreme events via active learning in neural operators"

[5]: https://arxiv.org/abs/2407.20158 "A benchmark study of machine learning for chaotic dynamical systems"

[6]: https://doi.org/10.1103/PhysRevResearch.2.012080 "Long-term prediction of chaotic systems with machine learning"

[7]: https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html "Conformal Time-series Forecasting"

[8]: https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"

[9]: https://www.jmlr.org/papers/v24/21-1524.html "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs"
