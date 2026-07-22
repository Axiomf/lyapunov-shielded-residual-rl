# lyapunov-shielded-residual-rl
Dynamical-systems analysis of residual reinforcement learning for robust cart-pole swing-up

This project compares three controllers for a nonlinear cart-pole with uncertain pole mass:

1. a physics-only swing-up and LQR controller;
2. the same nominal controller with a learned SAC residual;
3. the residual controller with a Lyapunov safety shield.

The main outputs are stability measurements, empirical regions of attraction, robustness results across pole masses, and reproducible plots for the final report.

## Prerequisites

- Git
- Python 3.11 (Python 3.12 is also supported)
- A terminal with `make` available (recommended, but not required)

GPU support is optional. The simulator, tests, and evaluation run on CPU. A CUDA-capable GPU can reduce SAC training time.

## Quick start for teammates

Clone the repository and enter it:

```bash
git clone https://github.com/Axiomf/lyapunov-shielded-residual-rl
cd cartpole-residual-rl
```

Create and activate an isolated Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and development tools:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

If `requirements-lock.txt` is present and you need to reproduce a final experiment exactly, use it instead:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

Verify the installation:

```bash
pytest
ruff check src tests scripts
python scripts/smoke_test.py
```

The setup is complete when the tests pass and the smoke test can simulate the nominal controller at the nominal pole mass.

## Repository map

```text
cartpole-residual-rl/
├── README.md                     # Setup and working conventions
├── requirements.txt              # Direct runtime dependencies
├── requirements-dev.txt          # Runtime plus test/lint tools
├── requirements-lock.txt         # Exact versions for final reproduction
├── pyproject.toml                # Package, pytest, and Ruff configuration
├── Makefile                      # Shortcuts for common commands
├── LICENSE                      
├── .gitignore
│
├── configs/                      # Version-controlled experiment settings
│   ├── plant.yaml
│   ├── nominal_controller.yaml
│   ├── sac.yaml
│   ├── shield.yaml
│   └── evaluation.yaml
│
├── src/
│   └── cartpole_rl/              # Reusable application code
│       ├── __init__.py
│       ├── config.py
│       ├── types.py
│       ├── simulation/           # Dynamics, integration, and simulation
│       ├── controllers/          # Nominal, residual, and shielded control
│       ├── envs/                 # Gymnasium residual-learning environment
│       ├── training/             # SAC training and callbacks
│       ├── analysis/             # Fixed points, Jacobians, basins, metrics
│       └── plotting/             # Publication-ready figures
│
├── scripts/                      # Thin command-line entry points
│   ├── smoke_test.py
│   ├── train.py
│   ├── evaluate.py
│   └── make_figures.py
│
├── tests/                        # Unit and reproducibility tests
├── outputs/                      # Generated runs, models, metrics, figures
└── report/                       # LaTeX report source
    └── main.tex
```

Keep reusable logic under `src/cartpole_rl/`. Files in `scripts/` should only parse arguments, load configuration, and call package functions.

## Dependency files

The dependency files have different purposes:

| File | Purpose | Update when |
|---|---|---|
| `requirements.txt` | Direct packages needed to run the project | Adding or removing a runtime dependency |
| `requirements-dev.txt` | Includes runtime dependencies plus tests and linting | Adding a development-only tool |
| `requirements-lock.txt` | Fully resolved environment used for final runs | Freezing a verified experiment environment |
| `pyproject.toml` | Makes `src/cartpole_rl` installable and configures tools | Changing package or tool settings |

Do not manually add transitive packages to `requirements.txt`. After the environment and tests are stable, refresh the lock file with:

```bash
pip freeze > requirements-lock.txt
```

Commit the updated dependency file and explain the reason in the pull request.

## Configuration

All plant parameters, controller gains, training settings, evaluation grids, success criteria, and random seeds belong in `configs/*.yaml`. Avoid hard-coded experiment values in Python files.

The project uses these conventions everywhere:

- state order: `[x, theta, x_dot, theta_dot]`;
- `theta = 0` means the pole is upright;
- control period: `0.02 s`;
- control force is saturated to the configured limit;
- the uncertain pole mass is `mu * nominal_pole_mass`;
- training samples `mu` from the configured training range;
- evaluation uses frozen masses, initial states, and success criteria.

Each training run must save a copy of its fully resolved configuration, seed, model checkpoint, logs, and current Git commit hash in its output directory.

## Standard workflow

Activate the environment before running any command:

```bash
source .venv/bin/activate
```

Run the quality checks while developing:

```bash
make test
```

If `make` is unavailable, run:

```bash
pytest
ruff check src tests scripts
```

Run a nominal-controller smoke test before training:

```bash
make smoke
# or
python scripts/smoke_test.py --controller nominal --mu 1.0
```

Do not begin full SAC training until the nominal controller can swing up and stabilize the nominal plant.

Train the three fixed seeds:

```bash
python scripts/train.py --config configs/sac.yaml --seed 11 --output outputs/runs/residual_seed_11
python scripts/train.py --config configs/sac.yaml --seed 22 --output outputs/runs/residual_seed_22
python scripts/train.py --config configs/sac.yaml --seed 33 --output outputs/runs/residual_seed_33
```

Evaluate all controllers on the same initial states and mass grid, then generate figures:

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/make_figures.py
```

Equivalent Make targets should be available:

```bash
make install
make test
make smoke
make train
make evaluate
make figures
```

## Implementation order

Work in this order so that each layer is tested before the next depends on it:

1. Implement and test nonlinear cart-pole dynamics and RK4 integration.
2. Implement energy shaping, discrete-time LQR, and hysteretic switching.
3. Wrap the simulator as a Gymnasium environment and run the environment checker.
4. Train one pilot SAC seed, validate the logs, then train all frozen seeds.
5. Add and unit-test the scalar Lyapunov shield and its fallback behavior.
6. Freeze all experiment configuration before the final evaluation sweep.
7. Run trajectories, fixed-point analysis, finite-difference Jacobians, basin estimates, Lyapunov checks, and pole-mass sweeps.

When changing dynamics or controller behavior, add or update a focused test in the same pull request.

## Outputs and version control

Generated files live under `outputs/`:

```text
outputs/
├── models/       # Selected checkpoints
├── runs/         # Per-seed training logs and resolved configs
├── metrics/      # Evaluation CSV/NPZ files
└── figures/      # Final plots
```

Commit source code, tests, YAML configurations, final metric tables, and report figures. Do not commit virtual environments, caches, TensorBoard event streams, temporary rollouts, or large intermediate checkpoints.

Use feature branches and keep commits focused:

```bash
git switch -c feature/<short-name>
git status
git add <changed-files>
git commit -m "Describe the change"
```

Before opening a pull request, run:

```bash
pytest
ruff check src tests scripts
python scripts/smoke_test.py
```

## Reproducing the final environment

Use a clean virtual environment rather than reusing a development environment:

```bash
python3.11 -m venv .venv-clean
source .venv-clean/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e .
pytest
python scripts/smoke_test.py
```

Record the Python version, operating system, Git commit, configuration files, seeds, and command used for every final result.

## Maintainer: first-time repository bootstrap

Only use this section when creating the repository for the first time. Teammates joining an existing repository should use **Quick start for teammates** instead.

```bash
mkdir cartpole-residual-rl
cd cartpole-residual-rl
git init

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Create the directories shown in the repository map, add the dependency and configuration files, then install the package in editable mode:

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest
```

The initial commit should contain the package skeleton, dependency files, configuration templates, smoke test, and at least the basic dynamics and integration tests.

## Troubleshooting

- **`ModuleNotFoundError: cartpole_rl`**: activate the virtual environment and rerun `pip install -e .` from the repository root.
- **Wrong Python interpreter**: run `which python` on macOS/Linux or `Get-Command python` in PowerShell; it should point inside `.venv`.
- **Training results differ**: confirm the Git commit, lock file, YAML configuration, seed, device, and deterministic evaluation setting.
- **Smoke test is unstable**: verify the state order, upright-angle convention, force clipping, integration step, and controller sample time before tuning gains.
- **Large files appear in Git**: check `.gitignore` and keep intermediate checkpoints and raw logs under ignored output paths.

