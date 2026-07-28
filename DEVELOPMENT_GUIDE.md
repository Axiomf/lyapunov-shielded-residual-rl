# Development Guide

This guide contains the general setup, dependency-management, Git, and
troubleshooting material for the Lyapunov-Shielded Residual RL project. For the
research question, experiment protocol, configuration conventions, and standard
run commands, see [README.md](README.md).

## Working from the repository root

All commands in the project documentation assume that the current directory is
the repository root:

```bash
git clone https://github.com/Axiomf/lyapunov-shielded-residual-rl.git
cd lyapunov-shielded-residual-rl
```

`git clone` downloads the repository and its Git history.
`cd lyapunov-shielded-residual-rl` makes the new checkout the current working
directory so later commands can find the project files.

## Python environment

The dependency baseline was reviewed on 2026-07-28 for CPython 3.13.14. NumPy
2.4.x and SciPy 1.17.x are intentionally selected for the Python 3.13 baseline.

Using `python -m pip` and `python -m pytest` ensures that commands use the
interpreter from the active virtual environment.

### Create and activate a virtual environment

On macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

Activation affects only the current terminal. Activate the environment again
after opening a new terminal.

### Install for development

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

- `requirements-dev.txt` installs the runtime dependencies plus test and lint
  tools.
- Editable installation makes changes under `src/` available without
  reinstalling the package.

### Install the exact experiment environment

Use the lock file in a clean virtual environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

The first command installs the versions frozen for the recorded experiments.
`--no-deps` then installs the local package without asking `pip` to resolve or
change those versions.

A `pip freeze` lock records the concrete environment in which it was produced
and may include platform-specific packages. Use it only on a supported platform
and Python version.

## Verify the installation

```bash
python -m pip check
python -m pytest
python -m ruff check src tests scripts
python scripts/smoke_test.py --controller nominal --mu 1.0
```

| Command | Purpose |
| --- | --- |
| `python -m pip check` | Confirms installed packages satisfy their declared requirements |
| `python -m pytest` | Runs the automated test suite |
| `python -m ruff check src tests scripts` | Checks source, tests, and scripts for configured lint errors |
| `python scripts/smoke_test.py --controller nominal --mu 1.0` | Runs a short simulation with nominal control and nominal pole mass |

Do not start a full SAC run until these checks pass and the nominal controller
can swing up and stabilize the nominal plant.

## Dependency files

| File | Purpose | Update it when |
| --- | --- | --- |
| `requirements.txt` | Direct packages needed at runtime | A runtime dependency is added or removed |
| `requirements-dev.txt` | Runtime dependencies plus test, lint, and development tools | A development-only tool changes |
| `requirements-lock.txt` | Fully resolved environment used for final runs | A new environment has been verified and intentionally frozen |
| `pyproject.toml` | Installable package and central tool configuration | Package metadata or pytest/Ruff settings change |

### Version policy

`requirements.txt` uses compatible ranges for development, while
`requirements-lock.txt` records the exact environment used for reported
experiments.

| Area | Release line | Reason |
| --- | --- | --- |
| Numerical core | NumPy 2.4.x and SciPy 1.17.x | Selected for Python 3.13 compatibility |
| RL stack | Gymnasium 1.3.x, Stable-Baselines3 2.9.x, and PyTorch 2.x | Stable-Baselines3 2.9 supports Gymnasium 1.3 and requires PyTorch 2.8 or newer; this project uses PyTorch 2.13 as its minimum baseline |
| Data and plots | pandas 3.x, Matplotlib 3.x, and seaborn 0.13.x | Selected release lines for metrics and publication figures |
| Developer tools | pytest 9.x, pytest-cov 7.x, and Ruff 0.16 or newer | Test, coverage, and lint lines; Ruff is capped below its future 1.0 release |

These ranges are not an experiment lock. After changing either requirements
file, install into a clean CPython 3.13.14 environment, run the verification
commands, and refresh `requirements-lock.txt`.

Do not add transitive dependencies manually to `requirements.txt`. To refresh
the lock after validating a clean experiment environment:

```bash
python -m pip freeze --exclude-editable > requirements-lock.txt
```

`--exclude-editable` keeps the machine-specific path of the local checkout out
of the lock file. Review the diff and rerun the tests before committing it; an
unrelated package installed in the environment will otherwise enter the lock.

## Makefile shortcuts

The `Makefile` provides these convenience targets:

| Command | Purpose |
| --- | --- |
| `make install` | Installs dependencies and the editable package |
| `make test` | Runs automated quality checks |
| `make smoke` | Runs the nominal-controller smoke test |
| `make train` | Runs the configured SAC training recipe |
| `make evaluate` | Runs the frozen evaluation sweep |
| `make figures` | Rebuilds figures from saved metrics |

Inspect a target before using it when you need its exact commands:

```bash
make -n <target>
```

Replace `<target>` with a name such as `train` or `evaluate`. The `-n` flag
prints the recipe without executing it.

## Git workflow

Create a focused feature branch and inspect changes before committing:

```bash
git switch -c feature/<short-name>
git status
git add <changed-files>
git commit -m "Describe the change"
```

| Command | Purpose |
| --- | --- |
| `git switch -c feature/<short-name>` | Creates and checks out a topic branch |
| `git status` | Shows the current branch and staged, unstaged, and untracked files |
| `git add <changed-files>` | Stages only the named paths |
| `git commit -m "Describe the change"` | Records the staged snapshot |

Before opening a pull request:

```bash
python -m pytest
python -m ruff check src tests scripts
python scripts/smoke_test.py --controller nominal --mu 1.0
```

These commands rerun the tests, lint checks, and nominal simulation smoke test
on the exact code proposed for review.

Commit source code, tests, YAML configurations, final metric tables, and report
figures. Do not commit virtual environments, caches, TensorBoard event streams,
temporary rollouts, or large intermediate checkpoints.

## Reproducing final results

Start from a clean clone and create a separate environment:

```bash
git clone https://github.com/Axiomf/lyapunov-shielded-residual-rl.git
cd lyapunov-shielded-residual-rl
python3.13 -m venv .venv-clean
source .venv-clean/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest
python scripts/smoke_test.py --controller nominal --mu 1.0
python scripts/evaluate.py --config configs/evaluation.yaml
python scripts/make_figures.py
```

On Windows PowerShell, create and activate the clean environment with:

```powershell
py -3.13 -m venv .venv-clean
.venv-clean\Scripts\Activate.ps1
```

The evaluation must use deterministic policy actions, the frozen mass and
initial-state grid, the recorded model checkpoints, and the success criteria in
`configs/evaluation.yaml`.

Record the following for every reported final result:

- operating system;
- Python version;
- Git commit;
- resolved configuration files;
- model checkpoints;
- training and evaluation seeds;
- compute device; and
- exact commands.

These records distinguish code or configuration changes from stochastic and
platform variation.

## Troubleshooting

### `ModuleNotFoundError: cartpole_rl`

Activate the virtual environment and rerun:

```bash
python -m pip install -e .
```

Run it from the repository root.

### Wrong Python interpreter

Run `which python` on macOS/Linux or `Get-Command python` in PowerShell. The
reported path should be inside the active `.venv` directory.

### PowerShell blocks activation

Review the current policy with `Get-ExecutionPolicy`. Follow your
organization's security policy rather than weakening it globally.

### Training results differ

Compare the Git commit, lock file, resolved YAML configuration, seed, device,
software versions, and deterministic evaluation setting.

### The nominal smoke test is unstable

Verify state order, angle convention, force clipping, integration step, control
period, and switching logic before tuning gains.

### Large files appear in Git

Inspect `git status`, update `.gitignore` where appropriate, and keep
intermediate checkpoints and raw logs under ignored output paths.
