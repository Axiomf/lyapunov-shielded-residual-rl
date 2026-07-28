# Lyapunov-Shielded Residual RL for Cart-Pole Swing-Up

This repository studies whether bounded residual reinforcement learning can
make a nominal cart-pole controller more robust to pole-mass mismatch without
damaging its local closed-loop stability.

The study compares three controllers built from the same components:

1. **Nominal control** — energy shaping for swing-up, followed by LQR near the
   upright equilibrium.
2. **Residual SAC** — the nominal controller plus a bounded action correction
   learned with Soft Actor-Critic (SAC).
3. **Shielded residual SAC** — the same learned residual, filtered by a
   Lyapunov-based shield before the action reaches the plant.

SAC is trained with randomized pole mass. All three controllers are then
evaluated deterministically on the same frozen initial conditions and a wider,
fixed pole-mass range.

> [!IMPORTANT]
> The shield and stability analyses provide empirical evidence for this plant,
> parameter range, and evaluation protocol. They do **not** establish global
> stability, unconditional constraint satisfaction, or a formal safety
> guarantee.

## What is measured

Reward alone is not treated as a sufficient measure of controller quality. The
evaluation also records:

- swing-up and stabilization success rate;
- displacement of the closed-loop fixed point from the upright target;
- spectral radius of a finite-difference closed-loop Jacobian;
- empirical basin geometry over the configured initial-condition grid;
- control effort and force saturation;
- cart-track violations;
- empirical Lyapunov-decrease violations; and
- shield activation and infeasibility rates.

Results are reported by controller, pole mass, and training seed. Every
comparison uses the same mass grid, initial states, rollout horizon, and
success criteria.

## Prerequisites

- [Git](https://git-scm.com/)
- CPython 3.13
- [uv](https://docs.astral.sh/uv/)

The simulator, tests, and evaluation run on CPU. A CUDA-capable GPU is optional
and may reduce SAC training time.

## Quick start

Run all commands from the repository root:

```bash
git clone https://github.com/Axiomf/lyapunov-shielded-residual-rl.git
cd lyapunov-shielded-residual-rl
uv sync
uv run pytest
uv run cartpole smoke-test --config configs/experiment.yaml
```

Do not start a full SAC run until the tests pass and the nominal smoke test can
swing up and stabilize the nominal plant.

## Repository structure

```text
lyapunov-shielded-residual-rl/
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── configs/
│   └── experiment.yaml
├── src/
│   └── cartpole_rl/
│       ├── __init__.py
│       ├── config.py
│       ├── plant.py
│       ├── controllers.py
│       ├── shield.py
│       ├── environment.py
│       ├── learning.py
│       ├── evaluation.py
│       ├── plotting.py
│       └── cli.py
├── tests/
│   ├── test_plant.py
│   ├── test_controllers.py
│   ├── test_shield.py
│   ├── test_learning.py
│   └── test_pipeline.py
├── paper/
│   └── main.tex
└── runs/                       # Generated and ignored by Git
```

The project intentionally uses one configuration file, one command-line
interface, and a small set of modules with clear ownership. See
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the detailed design rules and
module contracts.

## Architecture

| Path | Responsibility |
| --- | --- |
| `configs/experiment.yaml` | Seeds, masses, controller and shield parameters, SAC settings, evaluation grids, horizons, and output options |
| `config.py` | Load, validate, and freeze the experiment configuration |
| `plant.py` | Cart-pole dynamics, integration, equilibria, physical limits, and mass mismatch |
| `controllers.py` | Nominal control, bounded residual control, and construction of the three evaluated variants |
| `shield.py` | Lyapunov filtering, activation and feasibility status, fallback behavior, and diagnostics |
| `environment.py` | RL observations, reward, termination, action normalization, and training-time mass randomization |
| `learning.py` | SAC construction, training, checkpoints, loading, and seed handling |
| `evaluation.py` | Deterministic rollouts, stability analysis, basin sampling, and reported metrics |
| `plotting.py` | Figures and summary tables generated only from saved evaluation data |
| `cli.py` | The `train`, `evaluate`, `plot`, and `smoke-test` commands |

The plant dynamics are defined once in `plant.py`; training and evaluation must
use that implementation rather than maintaining separate simulators.

## Controller composition

The three evaluated controllers are compositions of the same nominal
controller, learned residual policy, and shield:

| Controller | Control law |
| --- | --- |
| Nominal | `clip(u_nominal(x))` |
| Residual SAC | `clip(u_nominal(x) + bounded_residual(x))` |
| Shielded residual SAC | `clip(u_nominal(x) + shield(x, bounded_residual(x)))` |

Every controller exposes the same interface:

```python
action, diagnostics = controller.act(state, deterministic=True)
```

Applicable diagnostics include the nominal action, proposed and applied
residuals, shield activation, shield infeasibility, and constraint margin. This
keeps evaluation logging independent of the controller variant.

## Configuration and conventions

All experiment choices belong in `configs/experiment.yaml`; experiment
constants should not be scattered through Python modules. The configuration
contains:

```yaml
seeds: [0, 1, 2]

plant:
  nominal_pole_mass: ...

controller:
  action_limit: ...
  residual_limit: ...
  energy_shaping: ...
  lqr: ...

shield:
  enabled: true
  lyapunov_region: ...
  decrease_margin: ...
  infeasible_fallback: ...

training:
  algorithm: sac
  total_steps: ...
  mass_randomization: true
  pole_mass_range: [...]

evaluation:
  deterministic_policy: true
  pole_mass_grid: [...]
  horizon: ...
  initial_condition_grid: ...
  success_definition: ...
```

The displayed keys are a contract, not a complete parameter list. The project
also uses these conventions:

- state order: `[x, theta, x_dot, theta_dot]`;
- `theta = 0` is the upright pole configuration;
- the control period is configured explicitly;
- commanded force is clipped to the configured actuator limit;
- actual pole mass is `mu * nominal_pole_mass`;
- training samples `mu` from the configured training range; and
- evaluation uses frozen masses, initial states, seeds, horizons, and success
  criteria.

Every run validates the configuration and saves its fully resolved form beside
the results.

## Reproducing the study

### 1. Run tests and the smoke test

```bash
uv run pytest
uv run cartpole smoke-test --config configs/experiment.yaml
```

The test suite covers plant conventions and integration, controller
composition and action bounds, shield behavior and fallback handling, seeded
mass randomization, and a small end-to-end pipeline.

### 2. Train SAC

```bash
uv run cartpole train --config configs/experiment.yaml
```

Training randomizes pole mass over the configured range and freezes one
checkpoint per seed. The residual and shielded comparisons use the same learned
policy; the shield is applied at control and evaluation time.

### 3. Evaluate all controllers

```bash
uv run cartpole evaluate --config configs/experiment.yaml --run <run-name>
```

Evaluation uses deterministic policy actions and the frozen evaluation grid. It
must not retrain a policy, silently change the grid, or sample a replacement
grid.

### 4. Generate figures and tables

```bash
uv run cartpole plot --run <run-name>
```

Plotting reads saved evaluation tables. It does not train policies or run new
simulations.

## Generated runs

Generated artifacts are stored under `runs/`, which is ignored by Git. Each run
is self-describing:

```text
runs/<run-name>/
├── resolved_config.yaml
├── metadata.json
├── checkpoints/
│   ├── seed_0/
│   ├── seed_1/
│   └── seed_2/
├── tables/
│   ├── episodes.csv
│   ├── local_stability.csv
│   └── basin.csv
└── figures/
```

- `metadata.json` records the timestamp, code revision, dependency versions,
  and command used.
- `episodes.csv` preserves one row per rollout and per-seed values before
  aggregation.
- `local_stability.csv` stores the numerical fixed point, displacement, and
  Jacobian spectral radius for every controller, mass, and seed.
- `basin.csv` stores each sampled initial condition and its outcome.

Large step-by-step trajectories should be retained only when a figure or
diagnostic requires them.

## Interpreting the stability results

- **Fixed-point displacement** measures how far the controller's numerical
  equilibrium lies from the desired upright state.
- **Jacobian spectral radius** is computed for the discrete-time closed-loop
  map at the numerical fixed point. A value below one is evidence of local
  linear asymptotic stability for that map, subject to finite-difference,
  modeling, and numerical error. It is not a global-stability result.
- **Empirical basin geometry** describes successful points on the sampled
  initial-condition grid. It is not a proof of the full region of attraction.
- **Lyapunov-decrease violations** count failures of the configured empirical
  decrease test. Their meaning depends on the candidate function,
  discretization, and tolerance.
- **Shield infeasibility** means that no residual action considered by the
  shield satisfies its configured condition. The configured fallback is
  recorded and evaluated separately.

Keep configured limits, observed violations, and formal guarantees distinct
when reporting results. This study claims no global or unconditional safety
guarantee.

## Development rules

- Keep reusable logic in `src/cartpole_rl/` and argument handling in `cli.py`.
- Add experiment values to `configs/experiment.yaml`, not Python source files.
- Save raw per-seed and per-rollout measurements before aggregation.
- Keep required analysis out of notebooks.
- Generate plots and report summaries from saved tables.
- Split a module into a package only when it becomes difficult to navigate.
- Describe the shield's implemented condition, assumptions, activations, and
  failures precisely.

