# Lyapunov-Shielded Residual RL for Cart-Pole Swing-Up

This repository studies whether bounded residual reinforcement learning can make
a nominal cart-pole controller more robust to pole-mass mismatch without
damaging its local closed-loop stability.

The study compares:

1. **Nominal control** — energy shaping for swing-up, followed by LQR near the
   upright equilibrium.
2. **Residual SAC** — the nominal controller plus a bounded action correction
   learned with Soft Actor-Critic (SAC).
3. **Shielded residual SAC** — the same learned residual, filtered by a
   Lyapunov-based shield before the action reaches the plant.

SAC is trained with randomized pole mass. All three controllers are then
evaluated deterministically on the same frozen initial conditions and on a
wider, fixed pole-mass range.

> [!IMPORTANT]
> The shield and stability analyses provide empirical evidence for this plant,
> parameter range, and evaluation protocol. They do **not** establish global
> stability, unconditional constraint satisfaction, or a formal safety
> guarantee.

## What is measured

Reward is not treated as a sufficient measure of controller quality. The
evaluation also records:

- swing-up and stabilization success rate;
- displacement of the closed-loop fixed point from the upright target;
- spectral radius of a finite-difference closed-loop Jacobian;
- empirical basin geometry over the configured initial-condition grid;
- control effort and force saturation;
- cart-track violations;
- empirical Lyapunov-decrease violations; and
- shield activation and infeasibility rates.

Results are reported across three fixed training seeds. Comparisons use the same
mass grid, initial states, rollout horizon, and success criteria for every
controller.

## Prerequisites

- [Git](https://git-scm.com/)
- CPython 3.13.14
- `make` for the optional convenience targets

The simulator, tests, and evaluation run on CPU. A CUDA-capable GPU is optional
and may reduce SAC training time.

For environment details, dependency policy, Git practices, troubleshooting, and
the full reproducibility procedure, see
[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md).

## Quick start

Run all commands from the repository root.

```bash
git clone https://github.com/Axiomf/lyapunov-shielded-residual-rl.git
cd lyapunov-shielded-residual-rl
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pip check
python -m pytest
python -m ruff check src tests scripts
python scripts/smoke_test.py --controller nominal --mu 1.0
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

For an exact experiment environment, install the frozen dependencies in a clean
virtual environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

Do not start a full SAC run until the tests pass and the nominal smoke test can
swing up and stabilize the nominal plant.

## Repository structure

```text
lyapunov-shielded-residual-rl/
├── README.md                     # Setup, workflow, and conventions
├── requirements.txt              # Direct runtime dependencies
├── requirements-dev.txt          # Runtime plus test/lint tools
├── requirements-lock.txt         # Frozen environment for final experiments
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

Reusable logic belongs under `src/cartpole_rl/`. Files in `scripts/` should
only parse command-line arguments, load configuration, and call package
functions.

## Configuration and conventions

Experiment values belong in `configs/*.yaml`, not in Python source files.

| File | Contents |
| --- | --- |
| `plant.yaml` | Physical parameters, integration step, actuator limit, and track limit |
| `nominal_controller.yaml` | Energy-shaping, LQR, and switching parameters |
| `sac.yaml` | Residual bounds, training mass distribution, SAC settings, and training seeds |
| `shield.yaml` | Lyapunov test, tolerance, action projection, and fallback settings |
| `evaluation.yaml` | Frozen mass grid, initial states, horizons, success criteria, and model paths |

The project uses these conventions:

- state order: `[x, theta, x_dot, theta_dot]`;
- `theta = 0` is the upright pole configuration;
- the control period is `0.02 s`;
- commanded force is clipped to the configured actuator limit;
- actual pole mass is `mu * nominal_pole_mass`;
- training samples `mu` from the range configured in `sac.yaml`; and
- evaluation uses frozen masses, initial states, seeds, horizons, and success
  criteria from `evaluation.yaml`.

Each training run must save its fully resolved configuration, seed, model
checkpoint, training log, and Git commit hash in its output directory.

## Standard workflow

### Quality checks and smoke test

```bash
make test
make smoke
```

Without `make`:

```bash
python -m pytest
python -m ruff check src tests scripts
python scripts/smoke_test.py --controller nominal --mu 1.0
```

### Train the fixed seeds

```bash
python scripts/train.py --config configs/sac.yaml --seed 11 --output outputs/runs/residual_seed_11
python scripts/train.py --config configs/sac.yaml --seed 22 --output outputs/runs/residual_seed_22
python scripts/train.py --config configs/sac.yaml --seed 33 --output outputs/runs/residual_seed_33
```

Run the seeds separately so that a failed run does not overwrite or obscure the
others. The unshielded and shielded comparisons should use the same learned
policy; the shield is applied at control/evaluation time.

### Evaluate and plot

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/make_figures.py
```

Evaluation must use deterministic policy actions. Figure generation reads the
saved metrics and does not rerun training.

The `Makefile` also provides `install`, `test`, `smoke`, `train`, `evaluate`,
and `figures` targets. Use `make -n <target>` to inspect the commands behind a
target without executing them.

## Interpreting the stability results

- **Fixed-point displacement** measures how far the controller's numerical
  equilibrium lies from the desired upright state.
- **Jacobian spectral radius** is computed for the discrete-time closed-loop
  map at the numerical fixed point. A value below one is evidence of local
  linear asymptotic stability for that map, subject to finite-difference,
  model, and numerical error. It is not a global-stability result.
- **Empirical basin geometry** describes successful points on the sampled
  initial-condition grid. It is not a proof of the full region of attraction.
- **Lyapunov-decrease violations** count failures of the configured empirical
  decrease test. Their meaning depends on the candidate function,
  discretization, and tolerance.
- **Shield infeasibility** means no residual action considered by the shield
  satisfies its configured condition; record and evaluate the configured
  fallback separately.

Report results by controller, mass, and seed. Keep nominal constraints
(configured limits), observed violations (evaluation data), and formal
guarantees (none claimed here) clearly separated.

## Generated outputs

Generated artifacts stay under `outputs/`:

```text
outputs/
├── models/       # Selected checkpoints
├── runs/         # Per-seed training logs and resolved configs
├── metrics/      # Evaluation CSV/NPZ files
└── figures/      # Final plots
```

Commit source code, tests, YAML configurations, final metric tables, and report
figures. Keep virtual environments, caches, TensorBoard event streams,
temporary rollouts, and large intermediate checkpoints out of version control.

## Reproducing reported results

Use a clean clone, the frozen `requirements-lock.txt`, the recorded model
checkpoints, and the fixed evaluation configuration. Record the operating
system, Python version, Git commit, configuration files, seeds, device, and
exact commands.

The complete procedure is in
[Reproducing final results](DEVELOPMENT_GUIDE.md#reproducing-final-results).
