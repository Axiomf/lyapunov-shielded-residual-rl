# Cart-pole residual RL starter

This is a deliberately small project skeleton for:

> Robustness and Basin Geometry of Lyapunov-Shielded Residual Reinforcement
> Learning for Cart-Pole Swing-Up

It contains the experiment structure and a small Stable-Baselines3 SAC
training adapter. The later evaluation/analysis pipeline is not implemented.

## What is implemented

- One shared state/action/transition format.
- Nonlinear cart-pole dynamics with `theta = 0` upright.
- Zero-order-held force and RK4 integration.
- Nominal discrete-time LQR computed from a finite-difference linearization.
- Energy-shaping/LQR controller with hysteresis.
- Bounded residual controller.
- Local Lyapunov shield using a simple scalar grid projection.
- Fixed plant mass during each rollout.
- Domain-randomized mass sampling between rollouts.
- Stable-Baselines3 SAC training for residual and shielded-residual control.
- Seeded replay collection and actor/configuration/metrics checkpoints.
- Dummy actors for controller and simulation smoke tests.

## What is intentionally left for you

- Replace the small energy-shaping example with the exact law you choose for
  the paper.
- Expand the short SAC interface run into a tuned training experiment.
- Finalize and tune switching, LQR, shield, reward, and SAC parameters.
- Add evaluation and analysis after training.

## Project map

```text
cartpole_starter/
├── main.py                 # Small runnable example
├── pyproject.toml
├── src/cartpole/
│   ├── actor.py            # Actor interface and dummy actors
│   ├── config.py           # Static experiment configuration
│   ├── control_math.py     # Nominal finite differences and LQR
│   ├── controllers.py      # The three controller types
│   ├── data.py             # Shared dynamic data objects
│   ├── plant.py            # Pure dynamics/RK4 functions
│   ├── shield.py           # Pure shield projection function
│   ├── simulation.py       # One-rollout orchestrator
│   └── training.py         # Stable-Baselines3 SAC adapter and mass sampling
└── tests/test_smoke.py
```

## Data flow

```text
State
  -> Controller.act(State)
  -> ControlDecision
  -> step_rk4(State, force, actual_plant)
  -> next State
  -> Transition
  -> Rollout
```

`mu` is used only to construct the actual plant at the start of a rollout.
It is never passed to the actor observation.

## Core contracts

| Function | Input | Output |
| --- | --- | --- |
| `state_derivative` | state array `(4,)`, scalar force, plant parameters | derivative array `(4,)` |
| `step_rk4` | `State`, scalar force, plant parameters | next `State` after one 0.02 s control interval |
| `normalize_state` | `State`, observation scales | clipped array `(4,)` in `[-1, 1]` |
| `Actor.act` | normalized array `(4,)`, deterministic flag | scalar action in `[-1, 1]` |
| `Controller.act` | `State` | `ControlDecision` containing the applied force and diagnostics |
| `project_with_lyapunov_shield` | state, proposed total force, nominal plant, `P`, shield settings | `ShieldResult` |
| `run_rollout` | controller, initial state, one fixed `mu`, configs | `Rollout` containing `Transition` objects |
| `train_sac` | training configuration and controller kind | `TrainingResult` with a trained actor and scalar metrics |

## Controller composition

1. `PhysicsController`
   - Swing-up law away from upright.
   - Discrete-time LQR near upright.
   - Hysteresis prevents rapid switching.

2. `ResidualController`
   - Gets the physics force.
   - Gets `a_rl` from the actor.
   - Applies `clip(u_physics + beta * a_rl, -u_max, u_max)`.

3. `ShieldedResidualController`
   - First proposes the same bounded residual action.
   - If `V(z) <= rho`, searches for the nearest scalar force satisfying the
     nominal one-step Lyapunov condition.
   - Falls back to nominal LQR if the grid search finds no feasible force.
   - Outside `V(z) <= rho`, leaves the bounded residual action unchanged.

The grid projection is intentionally easy to read. It is an approximation to
the scalar projection problem, controlled by `ShieldConfig.grid_size`. Replace
it later with a scalar optimizer if needed.

## Run

From this directory:

```bash
python -m pip install -e .
python main.py
python -m unittest discover -s tests
```

Run one short residual-SAC interface check:

```python
from cartpole.config import ExperimentConfig
from cartpole.controllers import ControllerKind
from cartpole.training import train_sac

result = train_sac(
    ControllerKind.RESIDUAL_SAC,
    ExperimentConfig(),
    seed=0,
)
print(result.metrics)
print(result.extra["checkpoint_path"])
```

## Important interpretation

The shield checks a condition on the **nominal model** only. The code does not
claim global safety, a certified region of attraction, or protection under all
mass mismatches. Later results should use terms such as “empirical basin” and
“loss of local stability.”

