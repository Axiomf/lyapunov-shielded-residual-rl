# Residual SAC for the cart-pole project

This is a deliberately small implementation of controller 2:

```text
u = clip(u_physics + beta * a_RL, -u_max, u_max)
```

The SAC policy observes the normalized full state and produces one scalar
`a_RL` in `[-1, 1]`. It does not observe the pole-mass multiplier `mu`.

The code uses only NumPy and PyTorch. It does not depend on Gymnasium, a
training framework, a configuration framework, or a logging framework.

## Project structure

```text
residual_sac_cartpole/
├── residual_sac/
│   ├── config.py             # Plain dataclasses and validation
│   ├── interfaces.py         # Plant and physics-controller contracts
│   ├── environment.py        # Residual composition and normalization
│   ├── networks.py           # Actor and twin Q networks
│   ├── replay_buffer.py      # Fixed-size NumPy replay buffer
│   ├── sac.py                # SAC update and checkpoint code
│   ├── training.py           # Domain-randomized training loop
│   └── evaluation.py         # Frozen deterministic rollouts
├── examples/
│   ├── dummy_backend.py      # Runnable toy backend, not cart-pole physics
│   └── project_adapters.py   # Empty integration points for your real code
├── tests/
│   └── test_residual_sac.py
├── train.py
├── evaluate.py
└── requirements.txt
```

Classes own stateful things: the plant, controller, replay buffer, networks,
and agent. Plain functions own calculations and loops: normalization, reward,
training, and evaluation. This keeps the code modular without introducing
advanced Python machinery.

## Important placeholders

The research prompt does not specify the exact reward or the velocity
normalization scales. Consequently:

- `placeholder_reward()` in `residual_sac/environment.py` must be replaced by
  the reward used identically for all three controllers.
- `state_scale=(2.4, pi, 5.0, 10.0)` is a placeholder. Choose the velocity
  scales once, record them, and keep them frozen during training and every
  evaluation.
- `examples/dummy_backend.py` is only a toy system for checking the software
  path. It is not the cart-pole model and must not be used for research
  results.
- `examples/project_adapters.py` contains the two methods to connect to the
  completed physics/LQR code.

The environment validates `beta <= 0.3 * u_max`. With `u_max=10 N`, the
largest allowed residual scale is `beta=3 N`.

## Install and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --episodes 50
python evaluate.py --checkpoint artifacts/sac_checkpoint.pt
```

The commands above use the dummy backend. They are smoke tests, not meaningful
cart-pole experiments.

## Connecting the real physics and LQR code

1. Open `examples/project_adapters.py`.
2. Put the real plant reset/step calls in `ProjectPlant`.
3. Put the real energy-shaping/LQR controller call in
   `ProjectPhysicsController.action`.
4. In `train.py` and `evaluate.py`, replace the two `Dummy...` imports and
   constructors with the project adapters.
5. Replace `placeholder_reward` with the common experiment reward.

The required plant contract is intentionally small:

```python
state = plant.reset(initial_state, pole_mass_scale)
next_state, track_violation = plant.step(force)
```

`plant.step(force)` must perform exactly one 0.02 s zero-order-held control
interval using the real RK4 implementation. The mass scale is set only in
`reset`, so it remains fixed for the entire rollout.

The physics-controller contract is:

```python
physics_controller.reset()
u_physics = physics_controller.action(state)
```

The wrapper, not the SAC agent, adds the residual and clips the physical force.
This prevents training code from silently changing the controller equation.

## SAC conventions used here

- Squashed Gaussian actor: sample with reparameterization, then apply `tanh`.
- Twin Q critics and slowly updated target critics.
- Automatic entropy-temperature tuning with target entropy `-1` for the
  one-dimensional action.
- Time-limit truncation still bootstraps; a real terminal track violation does
  not.
- Training samples one `mu ~ Uniform(0.8, 1.2)` at rollout reset.
- Evaluation uses `tanh(actor_mean)` with no sampling and no learning.
- The checkpoint contains the actor, critics, targets, entropy coefficient,
  optimizer states, dimensions, and SAC configuration.

The actor receives only the four normalized state values. `mu` is retained in
environment `info` for experiment bookkeeping but never enters the policy.

## Scope

This package implements the unshielded residual SAC controller only. It makes
no safety or region-of-attraction claim. The later shielded controller should
wrap the proposed physical action at the environment/controller boundary; it
should not require changes to the SAC implementation.

