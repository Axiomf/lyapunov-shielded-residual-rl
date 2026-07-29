# Lyapunov-shielded residual SAC

A small, NumPy-only implementation of the **third controller** in the
cart-pole project:

\[
u_{\text{proposed}}
= \operatorname{clip}\!\left(
u_{\text{physics}} + \beta a_{\text{RL}},
-u_{\max}, u_{\max}
\right).
\]

When the state error is inside \(V(z)=z^\top Pz\leq\rho\), the shield projects
that proposed action onto the scalar action interval satisfying

\[
V(Az+Bu)-V(z)\leq-\alpha V(z).
\]

Outside the region, the bounded residual action is unchanged. If the
constraint has no feasible action within the actuator limits, the controller
uses the nominal LQR action and records the infeasibility.

This is a **nominal, local, one-step condition**. It is not a global safety
guarantee or a certified region of attraction.

## Project structure

```text
.
├── pyproject.toml
├── README.md
├── src/lyapunov_residual_sac/
│   ├── __init__.py       public imports
│   ├── config.py         validated constants and model matrices
│   ├── controller.py     composes physics + SAC + shield
│   ├── interfaces.py     small interfaces for unfinished dependencies
│   ├── shield.py         pure projection mathematics
│   ├── stats.py          optional shield counters
│   └── stubs.py          dummy physics, LQR data, normalizer, SAC policy
└── tests/
    ├── test_controller.py
    └── test_shield.py
```

The split is intentionally plain:

- `shield.py` is the functional core. It has no knowledge of SAC or cart-pole
  code.
- `controller.py` is the OOP shell. It calls the other two controllers and
  applies the shield.
- `interfaces.py` describes the few methods your other project parts must
  provide.
- `stubs.py` is the only file containing fake project dependencies.

## Run it

Python 3.10+ is sufficient.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m lyapunov_residual_sac.stubs
```

## Replace the dummy pieces

Construct the shield from the discrete local model used by the nominal LQR:

```python
import numpy as np

from lyapunov_residual_sac import (
    ControllerConfig,
    LQRModel,
    LyapunovShield,
    ShieldConfig,
    ShieldedResidualController,
)

model = LQRModel(
    A=A_d,                         # shape (4, 4)
    B=B_d,                         # shape (4,) or (4, 1)
    P=P,                           # LQR Lyapunov/Riccati matrix
    equilibrium_state=np.zeros(4),
)

shield = LyapunovShield(
    model=model,
    config=ShieldConfig(rho=rho, alpha=alpha, u_max=10.0),
)

controller = ShieldedResidualController(
    physics=your_physics_controller,
    residual_policy=your_frozen_sac_policy,
    normalizer=your_state_normalizer,
    shield=shield,
    config=ControllerConfig(beta=3.0, u_max=10.0),
)

result = controller.act(state, deterministic=True)
plant_action = result.action
```

Your adapters need these methods:

```python
class YourPhysicsController:
    def action(self, state: np.ndarray) -> float:
        # Energy shaping / hysteretic LQR controller output.
        ...

    def lqr_action(self, state: np.ndarray) -> float:
        # Nominal local LQR output, used only as the infeasibility fallback.
        ...


class YourFrozenSACPolicy:
    def action(
        self,
        normalized_state: np.ndarray,
        deterministic: bool = True,
    ) -> float:
        # Return the dimensionless residual in [-1, 1].
        ...
```

The normalizer is simply a callable:

```python
def normalize_state(state: np.ndarray) -> np.ndarray:
    return state / observation_scales
```

Use the exact normalization and deterministic-action convention from SAC
training. The actor receives the normalized full state, not the mass ratio.

## Assumptions to check when integrating

1. `A_d` and `B_d` are zero-order-hold, discrete-time matrices for the same
   `0.02 s` control period as the plant.
2. `P`, the state order, angle convention, and equilibrium coordinates match
   the nominal LQR code.
3. `B_d` multiplies the **total scalar force** \(u\), not the residual action.
4. The equilibrium force is zero. If your model uses a nonzero equilibrium
   input, shift the action before evaluating the shield.
5. `rho` and `alpha` are experiment parameters. Do not present
   \(V(z)\leq\rho\) as a certified region unless separately proved.
6. The angle error is wrapped to \([-\pi,\pi)\) around the upright equilibrium.
7. Infeasible cases deliberately return clipped nominal LQR even if that
   fallback does not satisfy the one-step condition. The result records this.

## What each call records

`ShieldedResidualController.act(...)` returns the final action plus:

- physics action;
- raw and clipped residual action;
- unshielded proposed action;
- \(V(z)\) and the nominal model's predicted \(\Delta V\);
- whether the state was inside the shield region;
- whether projection occurred;
- whether the constraint was feasible;
- whether the LQR fallback was used;
- the admissible action interval, when one exists.

These fields make activation, projection, infeasibility, and nominal
Lyapunov-change statistics reproducible without placing logging inside the
controller.

For the empirical Lyapunov change of the actual, possibly mass-mismatched
plant, use the two observed states:

```python
from lyapunov_residual_sac import realized_lyapunov_change

delta_v_real = realized_lyapunov_change(state, next_state, model)
```

Keep this empirical value separate from `output.shield.nominal_delta_value`.
The nominal constraint does not imply that the mismatched plant has the same
one-step change.
