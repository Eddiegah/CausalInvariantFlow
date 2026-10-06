# Research proposal: CausalInvariantFlow

## Working title

**Calibrated Physics-Constrained Flow Matching for Rare-Event Forecasting in Chaotic Dynamical Systems under Known Interventions**

## Abstract

Chaotic systems have a finite horizon for accurate pointwise prediction, yet their long-run statistics, transition probabilities, and extreme-event behavior can remain scientifically useful. Existing work provides important but separate ingredients: flow matching enables efficient continuous normalizing-flow training; physics-informed models can improve physical residuals or long-term statistics; diffusion models can represent multimodal trajectory distributions and condition samples on rare events; and active-learning methods can discover extremes with fewer simulations. What is not established is whether these ingredients can be combined into an auditable forecasting system whose sampled trajectories are physically valid, whose uncertainty is calibrated by horizon and event regime, and whose rare-event predictions remain reliable under known parameter or forcing interventions.

We propose CausalInvariantFlow, a conditional flow-matching model with a verified physics layer and a separate calibration layer. The core paper will evaluate the model on controlled chaotic ODE/SDE systems and one spatiotemporal chaotic system. The primary claim will be narrow: compared with strong deterministic, reservoir, diffusion, unconstrained flow-matching, and ensemble baselines, the proposed method can improve the joint trade-off among distributional forecast quality, physics validity, rare-event skill, and simulator cost under held-out interventions. We will not claim causal discovery or universal invariant preservation. All empirical claims will be based on prospective splits, repeated seeds, solver verification, proper scoring rules, event-probability calibration, and transparent failure analysis.

## 1. Motivation and literature gap

Flow Matching introduced simulation-free regression of vector fields along prescribed probability paths and showed efficient continuous normalizing-flow training and sampling on image-generation tasks [1]. Later trajectory-generation work demonstrated efficient multimodal motion prediction, but those benchmarks do not establish calibrated uncertainty or long-horizon validity in chaotic scientific systems [2][3].

Chaotic forecasting has an intrinsic predictability limit. Prior scientific machine-learning studies show that learned models can reproduce long-run statistics or forecast short-term behavior, while physics-constrained reservoirs can improve velocity statistics and extreme-event prediction in particular turbulent-flow settings [4][5]. These results motivate distributional and physics-aware evaluation, not claims of indefinite pointwise prediction.

Diffusion models have also been used to quantify uncertainty and condition samples on nonlinear events in low-dimensional chaotic systems [6]. Physics-informed diffusion and neural-operator research shows that residual penalties and operator priors can improve selected PDE benchmarks, but these methods do not automatically provide calibrated intervention-conditional tail probabilities [7][8]. Conformal prediction and deep ensembles provide useful calibration baselines, but joint long-horizon calibration under chaotic rollout and intervention shift remains under-tested [9][10]. Active-learning work demonstrates that extreme events can be discovered efficiently with output-weighted acquisition and neural operators [11].

The gap is therefore not a missing isolated component. It is the absence of a controlled, auditable integration of: (i) conditional generative trajectory modeling, (ii) solver-verified physical validity, (iii) horizon- and event-conditional uncertainty calibration, and (iv) rare-event path evaluation under known interventions.

## 2. Research questions and hypotheses

**RQ1.** Does conditional flow matching improve distributional trajectory forecasts relative to deterministic and diffusion baselines at matched compute?

**RQ2.** Does physics enforcement improve long-horizon statistical fidelity and physical validity without suppressing valid rare paths?

**RQ3.** Can calibration remain acceptable under longer rollouts, held-out parameter regimes, and known interventions?

**RQ4.** Does the model forecast rare-event probability, transition time, and event amplitude better than average-error-optimized baselines?

**RQ5.** As a secondary question, can uncertainty- and tail-aware acquisition reduce the number of high-fidelity simulations required to achieve a fixed level of rare-event calibration?

Primary hypotheses:

- H1: The proposed model will improve CRPS/energy score and trajectory coverage relative to unconstrained flow matching and deterministic baselines.
- H2: Physics enforcement will reduce solver-verified residuals and invariant drift.
- H3: Physics enforcement alone will not guarantee calibration; explicit calibration will be needed.
- H4: Tail-weighted training/evaluation will improve rare-event probability and transition-time scores but may worsen common-regime average error; this trade-off will be measured rather than hidden.
- H5: The active acquisition extension will improve tail skill per simulator call relative to random and space-filling selection.

## 3. Scope boundary

The core paper will use **known intervention semantics**: the simulator exposes parameters, forcing, initial conditions, or controls whose changed values define ground-truth counterfactual rollouts. We will not infer a causal graph from observational data. We will use “intervention-conditioned” or “what-if” forecasting unless a formal structural model and counterfactual estimand are implemented.

The initial benchmark should contain:

1. Lorenz-63 or FitzHugh–Nagumo for exact, low-dimensional intervention tests.
2. Lorenz-96 or Kuramoto–Sivashinsky for spatiotemporal chaos.
3. One extreme-event system only if compute permits, such as a reduced turbulent-flow or metastable SDE benchmark.

Active acquisition is a secondary extension. Equation discovery is out of scope for the first paper because it would blur the contribution and add separate identifiability assumptions.

## 4. Formal problem

Let a simulator define

\[
\dot{x}(t)=f(x(t);\theta,u(t)), \qquad x(0)=x_0,
\]

where \(x\) is the state, \(\theta\) contains system parameters, and \(u\) is a known intervention or forcing. Given an observed history \(h_t\), parameters \(\theta\), intervention \(u\), and horizon \(H\), the model estimates

\[
p_\phi(x_{t+1:t+H}\mid h_t,\theta,u).
\]

A flow-matching model learns a vector field \(v_\phi(z,s\mid c)\) that transports a simple base distribution to the conditional future-trajectory distribution, where \(c=(h_t,\theta,u)\). The standard conditional flow-matching objective is

\[
\mathcal{L}_{FM}=\mathbb{E}_{s,z_s,c}\left[\|v_\phi(z_s,s\mid c)-u_s(z_s\mid z_1,z_0)\|_2^2\right].
\]

We add a physics term evaluated after decoding or rollout:

\[
\mathcal{L}_{phys}=\mathbb{E}\left[\|\partial_t \hat{x}-f(\hat{x};\theta,u)\|_2^2\right] + \lambda_{inv}\,\mathbb{E}[D(\mathcal{I}(\hat{x}),\mathcal{I}(x_0))],
\]

with boundary/initial and inequality terms where appropriate. The total training loss is

\[
\mathcal{L}=\mathcal{L}_{FM}+\lambda_{phys}\mathcal{L}_{phys}+\lambda_{tail}\mathcal{L}_{tail}+\lambda_{reg}\mathcal{L}_{reg}.
\]

The calibration layer is fit only on a held-out calibration split. It may use split-conformal trajectory bands, calibrated regression, or a validated ensemble/post-processing method. It must not use the final test set.

## 5. Method architecture

1. **History encoder.** Encodes a short observed state history and optional partial observations.
2. **Condition encoder.** Encodes parameters, intervention variables, forcing summaries, and a regime flag only when the flag is available at forecast time.
3. **Trajectory representation.** Uses a compact state-space or latent trajectory representation. For Lorenz-96/Kuramoto–Sivashinsky, use an operator-style encoder or Fourier features.
4. **Conditional flow-matching generator.** Generates multiple future paths from a base distribution using an ODE solver.
5. **Physics layer.** Compare soft residual training against a projection or constrained integration variant. Every final sample is checked by an independent high-fidelity solver or residual evaluator.
6. **Calibration layer.** Calibrates horizon-conditional trajectory regions and event probabilities on an untouched split.
7. **Rare-event head/evaluator.** Computes event probability, transition-time distribution, amplitude distribution, and path-level tail scores. It must not replace the full generative distribution.

## 6. Expected contribution

The defensible contribution is an evaluation-centered integration: a conditional flow-matching forecaster whose samples are verified against known dynamics, whose uncertainty is calibrated by horizon and event regime, and whose rare-event paths are evaluated under known interventions. The paper will explicitly quantify trade-offs among physical residuals, calibration, tail skill, and sampling cost.

The paper will not claim that flow matching, physics losses, conformal calibration, rare-event sampling, or active learning are individually new. It will not claim universal causal validity, exact invariant preservation from a soft loss, or indefinite chaotic phase prediction.

## References

[1]: https://arxiv.org/abs/2210.02747 "Flow Matching for Generative Modeling"
[2]: https://arxiv.org/abs/2506.08541 "TrajFlow: Multi-modal Motion Prediction via Flow Matching"
[3]: https://arxiv.org/abs/2306.03083 "MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion"
[4]: https://doi.org/10.1103/PhysRevResearch.2.012080 "Long-term prediction of chaotic systems with machine learning"
[5]: https://doi.org/10.1098/rspa.2021.0135 "Short- and long-term predictions of chaotic flows and extreme events: a physics-constrained reservoir computing approach"
[6]: https://arxiv.org/abs/2306.07526 "User-defined Event Sampling and Uncertainty Quantification in Diffusion Models for Physical Dynamical Systems"
[7]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/096347b4efc264ae7f07742fea34af1f-Abstract-Conference.html "Physics-Informed Diffusion Models"
[8]: https://www.jmlr.org/papers/v24/21-1524.html "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs"
[9]: https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html "Conformal Time-series Forecasting"
[10]: https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
[11]: https://doi.org/10.1038/s43588-022-00376-0 "Discovering and forecasting extreme events via active learning in neural operators"
