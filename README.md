# CausalInvariantFlow

**Calibrated physics-constrained flow matching for rare-event forecasting in chaotic dynamical systems under known interventions.**

[![Paper PDF](https://img.shields.io/badge/paper-PDF-b31b1b)](manuscript_final.pdf)
[![LaTeX source](https://img.shields.io/badge/LaTeX-source-008080)](manuscript_final.tex)
[![License](https://img.shields.io/badge/license-to%20be%20specified-lightgrey)](#license)

CausalInvariantFlow is a research package and reproducible benchmark scaffold for probabilistic forecasting of chaotic dynamical systems. It studies whether conditional flow matching can generate useful distributions of future trajectories while preserving known dynamical structure, calibrating uncertainty, and forecasting rare events under declared parameter or forcing interventions.

> **Research status:** This repository currently contains the methods paper, experiment protocol, starter implementation, deterministic Lorenz-63 utilities, metrics, figures, and smoke tests. It does **not** claim final empirical superiority until the preregistered benchmark is completed and its result artifacts are committed.

## Why this project matters

Long-horizon pointwise prediction in chaotic systems is fundamentally limited: small state or observation errors can grow exponentially. CausalInvariantFlow therefore evaluates forecasting as a distributional and scientific-structure problem rather than relying on mean squared trajectory error alone.

The benchmark asks:

> Can a conditional flow-matching model produce calibrated future-trajectory distributions, preserve relevant dynamical structure, and forecast rare-event paths under known interventions at competitive simulator cost?

The project treats five evaluation axes as complementary:

1. **Forecast distributions** — proper scores, coverage, sharpness, and path-statistics distance.
2. **Chaotic dynamics** — valid prediction time in Lyapunov units, invariant-measure error, spectra, and rollout stability.
3. **Physical validity** — governing-equation residuals, invariant drift, feasibility violations, and rejection or repair rates.
4. **Rare events** — event-probability scores, reliability, transition-time error, and tail-weighted path quality.
5. **Intervention robustness** — counterfactual state error, intervention-conditional coverage, and event-probability response.

No single metric defines success.

## Scope and terminology

The paper uses **known interventions** rather than causal discovery. The simulator defines the meaning of parameter, forcing, initial-condition, or control changes; the model is evaluated on whether its conditional forecasts respond correctly. The project does not infer a causal graph or claim causal identification from observational data.

The term **physics-constrained** is used operationally. Depending on the experiment, it refers to a soft governing-equation residual penalty, constrained integration or projection, or post-hoc solver verification. It does not imply exact invariant preservation by a neural network.

## Paper and project artifacts

- [`manuscript_final.pdf`](manuscript_final.pdf) — compiled 11-page methods and research-protocol paper.
- [`manuscript_final.tex`](manuscript_final.tex) — standalone LaTeX source.
- [`research_proposal.md`](research_proposal.md) — research motivation, gap, and contribution boundary.
- [`experiment_protocol.md`](experiment_protocol.md) — benchmark splits, metrics, ablations, statistical analysis, and stopping rules.
- [`RUN_STATUS.md`](RUN_STATUS.md) — local environment and reproducibility status.
- [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md) — Windows Python and PyTorch setup guidance.
- [`figures/`](figures/) — manuscript figures and the editable method-pipeline diagram.

## Repository layout

```text
.
├── manuscript_final.pdf       # Compiled paper
├── manuscript_final.tex       # LaTeX source
├── research_proposal.md       # Research proposal
├── experiment_protocol.md     # Preregistered-style protocol
├── requirements.txt           # Starter Python dependencies
├── src/
│   ├── generate_lorenz.py     # Deterministic Lorenz-63 data generation
│   ├── metrics.py              # Forecast, calibration, and rare-event metrics
│   ├── flow_matching.py        # Conditional vector field and FM loss
│   └── run_experiment.py       # Starter dataset/training/checkpoint runner
├── tests/
│   └── lorenz_smoke.js         # Runtime-independent simulator smoke test
├── figures/                    # Figures used by the LaTeX paper
└── .github/
    └── release_template.md     # Reusable GitHub release-notes template
```

## Quick start

### 1. Create an environment

The starter implementation targets Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a compatible PyTorch build for your CPU or CUDA configuration if it is not already included in your environment. Consult [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md) for Windows-specific guidance.

### 2. Run the deterministic Lorenz-63 generator

```powershell
python src/generate_lorenz.py --help
```

Use the command-line options shown by `--help` to generate trajectories with recorded seeds, parameters, and solver settings.

### 3. Run the smoke test

The runtime-independent smoke test requires Node.js:

```powershell
node tests/lorenz_smoke.js
```

### 4. Run the starter flow-matching experiment

```powershell
python src/run_experiment.py --help
```

The starter runner is intended for implementation validation and small-scale experiments. It is not, by itself, the completed preregistered benchmark.

## Reproducibility principles

The final study should preserve the following controls:

- Keep independent trajectories in train, validation, calibration, and test splits.
- Freeze event thresholds, basin definitions, intervention grids, and the final test set before evaluation.
- Report horizons in Lyapunov-time units where the benchmark supports Lyapunov-exponent estimation.
- Compare simple baselines, generative baselines, physics variants, and calibrated variants under matched data and tuning budgets.
- Aggregate over independent test conditions and repeated seeds rather than treating correlated windows as independent observations.
- Commit generated result tables, configuration files, seeds, environment information, and figure-generation scripts for every reported claim.
- Do not replace manuscript placeholders with numbers that cannot be traced to committed result artifacts.

## Planned benchmark sequence

1. Validate Lorenz-63 data generation, persistence, and linear baselines.
2. Train conditional flow matching on held-out initial states and interventions.
3. Compare unconstrained, residual-penalized, projected, and post-hoc-verified variants.
4. Add calibration and rare-event evaluation.
5. Stress-test on Lorenz-96 or Kuramoto--Sivashinsky.
6. Evaluate simulator-budgeted active acquisition only as a secondary experiment.
7. Populate the paper’s results section only from verified, committed artifacts.

## Limitations

The initial setting uses known equations and therefore does not represent hidden variables, severe measurement error, unknown model discrepancy, or ambiguous intervention semantics. Lorenz-63 provides exact low-dimensional verification but is not sufficient evidence for claims about high-dimensional or real-world systems. Calibration may fail under nonstationarity or strong intervention shift, and soft physics penalties do not guarantee exact invariants.

These limitations are part of the research design rather than hidden assumptions.

## Contributing

Contributions are welcome when they improve scientific validity, reproducibility, or clarity. Please include:

- a concise description of the change;
- the exact command and environment used to test it;
- seeds and configuration files for new experiments;
- a clear distinction between protocol changes and empirical results; and
- updates to the manuscript or experiment protocol when the scientific interpretation changes.

Avoid committing generated checkpoints, caches, or unreviewed result files. Large artifacts should be accompanied by metadata describing their origin and checksum.

## Citation

Until a citable publication record is available, cite the repository and manuscript as:

```bibtex
@misc{causalinvariantflow2026,
  title        = {CausalInvariantFlow: Calibrated Physics-Constrained Flow Matching for Rare-Event Forecasting in Chaotic Dynamical Systems under Known Interventions},
  author       = {Gah, Edmund Eric},
  year         = {2026},
  howpublished = {Research protocol and reproducible software repository},
  url          = {https://github.com/Eddiegah/CausalInvariantFlow}
}
```

## License

A project license has not yet been selected. Until a license file is added, all rights are reserved by the copyright holder. Add an explicit `LICENSE` file before distributing the code for reuse.

## Contact

For issues, reproducibility questions, or collaboration proposals, please use the repository’s GitHub issue tracker.
