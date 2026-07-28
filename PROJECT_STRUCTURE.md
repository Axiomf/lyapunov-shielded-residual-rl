# Project Structure

```text
lyapunov-shielded-residual-rl/
├── configs/
│   ├── evaluation.yaml
│   ├── nominal_controller.yaml
│   ├── plant.yaml
│   ├── sac.yaml
│   └── shield.yaml
├── report/
│   └── main.tex
├── scripts/
│   ├── evaluate.py
│   ├── make_figures.py
│   ├── smoke_test.py
│   └── train.py
├── src/
│   └── cartpole_rl/
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── basin.py
│       │   ├── fixed_points.py
│       │   ├── jacobians.py
│       │   ├── lyapunov.py
│       │   ├── mass_sweep.py
│       │   ├── metrics.py
│       │   └── rollouts.py
│       ├── controllers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── energy_shaping.py
│       │   ├── lqr.py
│       │   ├── lyapunov_shield.py
│       │   ├── nominal.py
│       │   └── residual.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── residual_cartpole.py
│       ├── plotting/
│       │   ├── __init__.py
│       │   └── figures.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── dynamics.py
│       │   ├── integrators.py
│       │   └── simulator.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── callbacks.py
│       │   └── train_sac.py
│       ├── __init__.py
│       ├── config.py
│       └── types.py
├── tests/
│   ├── test_dynamics.py
│   ├── test_environment.py
│   ├── test_integrator.py
│   ├── test_lqr.py
│   ├── test_reproducibility.py
│   └── test_shield.py
├── .gitignore
├── DEVELOPMENT_GUIDE.md
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Directory Overview

- `configs/` — YAML configuration files for the plant, controllers, shield, SAC training, and evaluation.
- `report/` — LaTeX source for the project report.
- `scripts/` — Command-line entry points for training, evaluation, smoke testing, and figure generation.
- `src/cartpole_rl/` — Main Python package containing analysis, control, environment, simulation, plotting, and training code.
- `tests/` — Automated tests for dynamics, integration, environment behavior, LQR, reproducibility, and the Lyapunov shield.

Generated directories and files such as `.venv/`, `.pytest_cache/`, `__pycache__/`, and `*.egg-info/` are intentionally omitted.
