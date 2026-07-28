# Lyapunov-Shielded Residual RL for Cart-Pole Swing-Up

This repository studies whether bounded residual reinforcement learning can make
a nominal cart-pole controller more robust to pole-mass mismatch without
damaging its local closed-loop stability.

The study compares:

1. **Nominal control** â€” energy shaping for swing-up, followed by LQR near the
   upright equilibrium.
2. **Residual SAC** â€” the nominal controller plus a bounded action correction
   learned with Soft Actor-Critic (SAC).
3. **Shielded residual SAC** â€” the same learned residual, filtered by a
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
- CPython 3.11
- `make` for the convenience targets in `Makefile` (optional)

The simulator, tests, and evaluation run on CPU. A CUDA-capable GPU is optional
and may reduce SAC training time.

The dependency baseline was reviewed on 2026-07-28 for CPython 3.11. NumPy
2.4.x and SciPy 1.17.x are intentionally selected because they are the newest
release lines that support Python 3.11; their next release lines require Python
3.12 or newer.

All commands below are run from the repository root unless stated otherwise.
Using `python -m pip` and `python -m pytest` ensures that the commands use the
Python interpreter from the active virtual environment.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Axiomf/lyapunov-shielded-residual-rl.git
cd lyapunov-shielded-residual-rl
```

| Command | What it does |
| --- | --- |
| `git clone â€¦` | Downloads the repository and its Git history into a new `lyapunov-shielded-residual-rl` directory. |
| `cd lyapunov-shielded-residual-rl` | Makes that directory the current working directory so later commands can find the project files. |

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

| Command | What it does |
| --- | --- |
| `python3.11 -m venv .venv` | Creates an isolated Python 3.11 environment in `.venv` on macOS or Linux. |
| `source .venv/bin/activate` | Updates the current macOS/Linux shell to use the environment's Python and installed packages. |
| `py -3.11 -m venv .venv` | Creates the same environment with the Windows Python launcher. |
| `.venv\Scripts\Activate.ps1` | Activates the environment in Windows PowerShell. |

Activation affects only the current terminal. Run the appropriate activation
command again after opening a new terminal.

### 3. Install the project

For development:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

| Command | What it does |
| --- | --- |
| `python -m pip install --upgrade pip` | Updates the package installer inside the active virtual environment. |
| `python -m pip install -r requirements-dev.txt` | Installs the runtime dependencies plus test and lint tools listed by the project. |
| `python -m pip install -e .` | Installs `cartpole_rl` in editable mode, so changes under `src/` are available without reinstalling the package. |

For an exact experiment environment, use the lock file in a clean virtual
environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

| Command | What it does |
| --- | --- |
| `python -m pip install -r requirements-lock.txt` | Installs the versions frozen for the recorded experiments. |
| `python -m pip install --no-deps -e .` | Installs the local package without asking `pip` to resolve or change the already locked dependencies. |

Use the lock file only on a supported platform and Python version. A
`pip freeze` lock records the concrete environment in which it was produced
and may include platform-specific packages.

### 4. Verify the installation

```bash
python -m pip check
python -m pytest
python -m ruff check src tests scripts
python scripts/smoke_test.py --controller nominal --mu 1.0
```

| Command | What it does |
| --- | --- |
| `python -m pip check` | Confirms that the installed packages satisfy one another's declared version requirements. |
| `python -m pytest` | Runs the automated test suite. |
| `python -m ruff check src tests scripts` | Checks the package, tests, and command-line scripts for the lint errors configured in `pyproject.toml`. |
| `python scripts/smoke_test.py --controller nominal --mu 1.0` | Runs a short simulation with the nominal controller and nominal pole mass; `--mu 1.0` means no mass mismatch. |

Do not start a full SAC run until the tests pass and the nominal smoke test can
swing up and stabilize the nominal plant.

## Repository structure

```text
lyapunov-shielded-residual-rl/
â”œâ”€â”€ README.md                     # Setup, workflow, and conventions
â”œâ”€â”€ requirements.txt              # Direct runtime dependencies
â”œâ”€â”€ requirements-dev.txt          # Runtime plus test/lint tools
â”œâ”€â”€ requirements-lock.txt         # Frozen environment for final experiments
â”œâ”€â”€ pyproject.toml                # Package, pytest, and Ruff configuration
â”œâ”€â”€ Makefile                      # Shortcuts for common commands
â”œâ”€â”€ LICENSE
â”œâ”€â”€ .gitignore
â”‚
â”œâ”€â”€ configs/                      # Version-controlled experiment settings
â”‚   â”œâ”€â”€ plant.yaml
â”‚   â”œâ”€â”€ nominal_controller.yaml
â”‚   â”œâ”€â”€ sac.yaml
â”‚   â”œâ”€â”€ shield.yaml
â”‚   â””â”€â”€ evaluation.yaml
â”‚
â”œâ”€â”€ src/
â”‚   â””â”€â”€ cartpole_rl/              # Reusable application code
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ config.py
â”‚       â”œâ”€â”€ types.py
â”‚       â”œâ”€â”€ simulation/           # Dynamics, integration, and simulation
â”‚       â”œâ”€â”€ controllers/          # Nominal, residual, and shielded control
â”‚       â”œâ”€â”€ envs/                 # Gymnasium residual-learning environment
â”‚       â”œâ”€â”€ training/             # SAC training and callbacks
â”‚       â”œâ”€â”€ analysis/             # Fixed points, Jacobians, basins, metrics
â”‚       â””â”€â”€ plotting/             # Publication-ready figures
â”‚
â”œâ”€â”€ scripts/                      # Thin command-line entry points
â”‚   â”œâ”€â”€ smoke_test.py
â”‚   â”œâ”€â”€ train.py
â”‚   â”œâ”€â”€ evaluate.py
â”‚   â””â”€â”€ make_figures.py
â”‚
â”œâ”€â”€ tests/                        # Unit and reproducibility tests
â”œâ”€â”€ outputs/                      # Generated runs, models, metrics, figures
â””â”€â”€ report/                       # LaTeX report source
    â””â”€â”€ main.tex
```

Reusable logic belongs under `src/cartpole_rl/`. Files in `scripts/` should
only parse command-line arguments, load configuration, and call package
functions.

## Dependency files

| File | Purpose | Update it when |
| --- | --- | --- |
| `requirements.txt` | Declares direct packages needed at runtime. | A runtime dependency is added or removed. |
| `requirements-dev.txt` | Adds test, lint, and other development tools to the runtime environment. | A development-only tool changes. |
| `requirements-lock.txt` | Records the fully resolved environment used for final runs. | A new environment has been verified and intentionally frozen. |
| `pyproject.toml` | Defines the installable package and central tool configuration. | Package metadata or pytest/Ruff settings change. |

### Version policy

`requirements.txt` uses compatible ranges for development, while
`requirements-lock.txt` records the exact environment used for reported
experiments. The current ranges target these release lines:

| Area | Release line | Reason |
| --- | --- | --- |
| Numerical core | NumPy 2.4.x and SciPy 1.17.x | Newest lines that retain CPython 3.11 support. |
| RL stack | Gymnasium 1.3.x, Stable-Baselines3 2.9.x, and PyTorch 2.x | Stable-Baselines3 2.9 supports Gymnasium 1.3 and requires PyTorch 2.8 or newer; this project uses PyTorch 2.13 as its minimum baseline. |
| Data and plots | pandas 3.x, Matplotlib 3.x, and seaborn 0.13.x | Current stable release lines for metrics and publication figures. |
| Developer tools | pytest 9.x, pytest-cov 7.x, and Ruff 0.16 or newer | Current test, coverage, and lint lines; Ruff is capped below its future 1.0 release. |

These ranges are not an experiment lock. After changing either requirements
file, install into a clean CPython 3.11 environment, run the verification
commands, and then refresh `requirements-lock.txt`. A pre-existing lock file
does not update automatically.

Do not add transitive dependencies manually to `requirements.txt`. To refresh
the lock after validating a **clean** experiment environment, run:

```bash
python -m pip freeze --exclude-editable > requirements-lock.txt
```

This asks `pip` to print every non-editable installed package and uses the
shell's `>` operator to replace `requirements-lock.txt` with that list.
`--exclude-editable` prevents the machine-specific path of the local checkout
from entering the lock. Review the diff and rerun the tests before committing
it; an unrelated package installed in the environment will otherwise enter the
lock file.

## Configuration and conventions

Experiment values belong in `configs/*.yaml`, not in Python source files.

| File | Contents |
| --- | --- |
| `plant.yaml` | Physical parameters, integration step, actuator limit, and track limit. |
| `nominal_controller.yaml` | Energy-shaping, LQR, and switching parameters. |
| `sac.yaml` | Residual bounds, training mass distribution, SAC settings, and training seeds. |
| `shield.yaml` | Lyapunov test, tolerance, action projection, and fallback settings. |
| `evaluation.yaml` | Frozen mass grid, initial states, horizons, success criteria, and model paths. |

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
checkpoint, training log, and Git commit hash in its output directory. These
records distinguish code/configuration changes from stochastic variation.

## Standard workflow

### Quality checks

With `make`:

```bash
make test
```

`make test` runs the test and lint commands defined by the `test` target in
`Makefile`.

Without `make`:

```bash
python -m pytest
python -m ruff check src tests scripts
```

The first command runs behavioral and reproducibility tests. The second checks
source files for configured lint errors.

### Nominal-controller smoke test

```bash
make smoke
```

`make smoke` runs the repository's standard short nominal-controller check.
The explicit equivalent is:

```bash
python scripts/smoke_test.py --controller nominal --mu 1.0
```

Here, `--controller nominal` selects the physics-based controller and
`--mu 1.0` selects nominal pole mass.

### Train the fixed seeds

```bash
python scripts/train.py --config configs/sac.yaml --seed 11 --output outputs/runs/residual_seed_11
python scripts/train.py --config configs/sac.yaml --seed 22 --output outputs/runs/residual_seed_22
python scripts/train.py --config configs/sac.yaml --seed 33 --output outputs/runs/residual_seed_33
```

Each command trains one residual SAC policy:

- `scripts/train.py` is the training entry point;
- `--config configs/sac.yaml` loads the SAC and domain-randomization settings;
- `--seed` fixes the run's pseudorandom seed; and
- `--output` gives that seed a separate directory for checkpoints, logs, and
  resolved configuration.

Run the commands separately so that a failed seed does not overwrite or obscure
the others. The unshielded and shielded comparisons should use the same learned
policy; the shield is applied to its residual action at control/evaluation
time.

### Evaluate and plot

```bash
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/make_figures.py
```

| Command | What it does |
| --- | --- |
| `python scripts/evaluate.py --config configs/evaluation.yaml` | Evaluates the three controllers on the frozen mass/initial-state grid and writes the configured metrics; the evaluation must use deterministic policy actions. |
| `python scripts/make_figures.py` | Reads the saved metrics and creates report figures without rerunning training. |

The `Makefile` provides these convenience entry points:

| Command | What it does |
| --- | --- |
| `make install` | Installs the dependencies and editable package using the recipe in `Makefile`. |
| `make test` | Runs the repository's automated quality checks. |
| `make smoke` | Runs the nominal-controller smoke test. |
| `make train` | Runs the configured SAC training recipe. |
| `make evaluate` | Runs the frozen evaluation sweep. |
| `make figures` | Rebuilds figures from saved metrics. |

Inspect a target before using it if you need its exact underlying commands:

```bash
make -n <target>
```

`make -n <target>` prints the recipe for the selected target without executing
it. Replace `<target>` with a name such as `train` or `evaluate`.

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
  decrease test. Their meaning depends on the chosen candidate function,
  discretization, and tolerance.
- **Shield infeasibility** means no residual action considered by the shield
  satisfies its configured condition; the configured fallback must be recorded
  and evaluated separately.

Report results by controller, mass, and seed. Keep nominal constraints
(configured limits), observed violations (evaluation data), and formal
guarantees (none claimed here) clearly separated.

## Generated outputs

Generated artifacts stay under `outputs/`:

```text
outputs/
â”œâ”€â”€ models/       # Selected checkpoints
â”œâ”€â”€ runs/         # Per-seed training logs and resolved configs
â”œâ”€â”€ metrics/      # Evaluation CSV/NPZ files
â””â”€â”€ figures/      # Final plots
```

Commit source code, tests, YAML configurations, final metric tables, and report
figures. Do not commit virtual environments, caches, TensorBoard event streams,
temporary rollouts, or large intermediate checkpoints.

## Git workflow

Create a focused feature branch and inspect changes before committing:

```bash
git switch -c feature/<short-name>
git status
git add <changed-files>
git commit -m "Describe the change"
```

| Command | What it does |
| --- | --- |
| `git switch -c feature/<short-name>` | Creates and checks out a new branch; replace `<short-name>` with a concise topic. |
| `git status` | Shows the current branch and all staged, unstaged, and untracked files. |
| `git add <changed-files>` | Stages only the paths you name for the next commit. |
| `git commit -m "Describe the change"` | Records the staged snapshot with a short message. |

Before opening a pull request:

```bash
python -m pytest
python -m ruff check src tests scripts
python scripts/smoke_test.py --controller nominal --mu 1.0
```

These commands rerun, respectively, the tests, lint checks, and nominal
simulation smoke test on the exact code proposed for review.

## Reproducing final results

Start from a clean clone and create a separate environment:

```bash
python3.11 -m venv .venv-clean
source .venv-clean/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest
python scripts/smoke_test.py --controller nominal --mu 1.0
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/make_figures.py
```

| Command | What it does |
| --- | --- |
| `python3.11 -m venv .venv-clean` | Creates an environment that is separate from day-to-day development packages. |
| `source .venv-clean/bin/activate` | Activates that clean environment on macOS or Linux. |
| `python -m pip install --upgrade pip` | Updates only the clean environment's installer. |
| `python -m pip install -r requirements-lock.txt` | Restores the frozen package versions. |
| `python -m pip install --no-deps -e .` | Makes the checked-out source importable without changing locked dependencies. |
| `python -m pytest` | Verifies the clean installation. |
| `python scripts/smoke_test.py --controller nominal --mu 1.0` | Checks the baseline simulation before the expensive evaluation. |
| `python scripts/evaluate.py --config configs/evaluation.yaml` | Recomputes deterministic controller metrics using the frozen protocol. |
| `python scripts/make_figures.py` | Rebuilds figures from those metrics. |

Record the operating system, Python version, Git commit, configuration files,
model checkpoints, seeds, device, and exact commands for every reported final
result.

## Troubleshooting

- **`ModuleNotFoundError: cartpole_rl`** â€” activate the virtual environment and
  rerun `python -m pip install -e .` from the repository root.
- **Wrong Python interpreter** â€” run `which python` on macOS/Linux or
  `Get-Command python` in PowerShell. The reported path should be inside the
  active `.venv` directory.
- **PowerShell blocks activation** â€” review the current policy with
  `Get-ExecutionPolicy`. Follow your organization's security policy rather than
  weakening it globally.
- **Training results differ** â€” compare the Git commit, lock file, resolved
  YAML configuration, seed, device, software versions, and deterministic
  evaluation setting.
- **The nominal smoke test is unstable** â€” verify state order, angle convention,
  force clipping, integration step, control period, and switching logic before
  tuning gains.
- **Large files appear in Git** â€” inspect `git status`, update `.gitignore` if
  appropriate, and keep intermediate checkpoints and raw logs under ignored
  output paths.