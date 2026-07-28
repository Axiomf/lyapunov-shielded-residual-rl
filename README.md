# Cart-pole residual-RL study

A small, reproducible baseline for comparing:

1. nominal energy-shaping + LQR control;
2. bounded residual Soft Actor-Critic (SAC); and
3. Lyapunov-shielded bounded residual SAC.

The only modeled uncertainty is pole mass. SAC is trained with randomized pole
mass and evaluated deterministically on a wider, fixed mass grid.

## What is in the project

```text
cartpole_residual_study/
├── configs/default.yaml        # every experiment setting
├── scripts/run_study.py        # train/evaluate command
├── src/cartpole_study/
│   ├── config.py               # typed YAML loading
│   ├── plant.py                # nonlinear cart-pole dynamics
│   ├── controllers.py          # nominal and residual policies
│   ├── shield.py               # scalar Lyapunov projection
│   ├── env.py                  # Gymnasium training environment
│   └── experiment.py           # training, rollouts, diagnostics, CSV output
├── tests/                      # fast dynamics/controller checks
├── requirements.txt
└── pyproject.toml
```

Generated models and tables go under `outputs/` and are deliberately excluded
from version control.

## Quick start

Python 3.13 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
pytest
```

Run a short smoke study before committing to a full training run:

```bash
python scripts/run_study.py train --controller all --timesteps 1000
python scripts/run_study.py evaluate --controller all --quick
```

Run the configured experiment:

```bash
python scripts/run_study.py train --controller all
python scripts/run_study.py evaluate --controller all
```

Use `--seed 0` or `--mass 0.10` to isolate a case. Run
`python scripts/run_study.py --help` for all options.

## Outputs

Evaluation writes:

- `episodes.csv`: return, success, terminal displacement, effort, track
  violations, Lyapunov-decrease violations, and shield rates per rollout;
- `local_diagnostics.csv`: numerical fixed-point displacement and discrete
  closed-loop Jacobian spectral radius;
- `basin_slices.csv`: converged fraction and area estimate on the configured
  `(theta, theta_dot)` grid; and
- `summary.csv`: grouped mean and standard deviation of episode metrics.

These quantities are empirical diagnostics:

- The Jacobian is a finite-difference linearization of the sampled closed-loop
  map near a numerically estimated fixed point.
- The basin result is a finite, two-dimensional slice, not the full region of
  attraction.
- The shield enforces its configured derivative condition only where it is
  active and feasible for this model and bounded scalar input.
- No result from this project is a claim of global or unconditional safety.

## Reproducibility notes

- Configuration is centralized in `configs/default.yaml`.
- Training and evaluation seeds are explicit.
- Evaluation uses deterministic SAC actions and a frozen mass grid.
- Raw per-episode rows are retained; do not report only aggregate reward.
- Keep the three-seed results separate before pooling them.

The default training budget is intentionally moderate. Increase it only after
the smoke run and local diagnostics behave sensibly.
