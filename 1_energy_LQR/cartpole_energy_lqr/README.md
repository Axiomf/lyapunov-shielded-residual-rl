# Plain cart-pole energy shaping + discrete LQR

This is a deliberately small implementation of the physics baseline for the
research project:

1. nonlinear cart-pole plant;
2. zero-order-held force and RK4 integration;
3. energy shaping for swing-up;
4. discrete-time LQR near the upright;
5. a hysteretic switch between the two modes.

It uses only NumPy and SciPy. There is no RL framework, configuration framework,
dependency injection, inheritance tree, or hidden global state.

## Project layout

```text
cartpole_energy_lqr/
├── cartpole/
│   ├── config.py       # immutable parameters and gains
│   ├── dynamics.py     # nonlinear equations and RK4/ZOH step
│   ├── energy.py       # energy-shaping law
│   ├── lqr.py          # linearization, ZOH discretization, DARE
│   └── controller.py   # stateful hysteretic hybrid controller
├── scripts/
│   └── run_swingup.py  # one deterministic rollout; writes CSV
├── tests/
│   ├── test_dynamics_and_lqr.py
│   └── test_energy_and_switch.py
├── requirements.txt
└── README.md
```

The split is intentional:

- Use **pure functions** for equations, integration, energy, and
  linear-control design. They are easy to test.
- Use a **small class** for `PhysicsController`, because hysteresis means it
  must remember its previous mode.
- Use frozen dataclasses only for named parameters and returned results.

## State and model conventions

The state is

```text
s = [x, theta, x_dot, theta_dot]
```

and `theta = 0` is upright. The pole center of mass is

```text
p_x = x + l sin(theta)
p_y =     l cos(theta)
```

With `q = [x, theta]`, the nonlinear acceleration equations are

```text
[M + m,  m l cos(theta)] [x_ddot    ]   [u - b_x x_dot + m l theta_dot^2 sin(theta)]
[m l cos(theta), J      ] [theta_ddot] = [m g l sin(theta) - b_theta theta_dot       ]
```

where `J = I_com + m l^2`. The defaults set `I_com = 0`, which is the
point-mass-pole convention. If your intended plant is a uniform rod, set its
center-of-mass inertia explicitly and use the same convention throughout the
plant, nominal model, and experiments.

## Energy shaping

The pole energy and upright target are

```text
E     = 0.5 J theta_dot^2 + m g l cos(theta)
E_ref = m g l
e_E   = E - E_ref
```

For the undamped model,

```text
E_dot = -m l x_ddot theta_dot cos(theta).
```

The swing-up controller asks for

```text
x_ddot_ref =
    k_E e_E theta_dot cos(theta)
    - k_x x
    - k_v x_dot.
```

Ignoring the cart-centering term and acceleration saturation,

```text
e_E E_dot = -m l k_E e_E^2 (theta_dot cos(theta))^2 <= 0.
```

The requested cart acceleration is converted to force using the nonlinear
nominal equations (partial feedback linearization). The force is then clipped
to the physical limit.

The exact downward rest state is a symmetry-induced equilibrium of the basic
energy law. A small deterministic kick is included so the demonstration can
start from exactly `theta = pi`, `theta_dot = 0`. For randomized experiments,
another defensible convention is to perturb the initial angle and disable the
kick. Pick one convention and keep it identical for all three controllers.

## Discrete-time LQR

`continuous_linear_model()` linearizes the nominal equations at
`s = 0, u = 0`. `scipy.signal.cont2discrete(..., method="zoh")` produces the
sampled model

```text
z[k+1] = A_d z[k] + B_d u[k]
```

at the 0.02 s control period. The discrete algebraic Riccati equation gives
`P`, and

```text
K = (R + B_d.T P B_d)^-1 B_d.T P A_d
u = -K z.
```

The controller wraps `theta` to `[-pi, pi)` before applying LQR. Its unsaturated
linear closed-loop eigenvalues are stored in
`controller.lqr.closed_loop_eigenvalues`.

`P` is useful later for defining the nominal shield candidate
`V(z) = z.T P z`, but the mere existence of `P` does not certify a nonlinear
region of attraction under saturation or mass mismatch.

## Hysteretic switching

The controller enters LQR only when all four inner thresholds are met. Once in
LQR mode, it stays there until any larger outer threshold is crossed. This
memory prevents rapid switching at a single boundary.

Call `controller.reset()` before every independent rollout. Otherwise one
rollout's final hybrid mode leaks into the next rollout.

## Run it

From the project directory:

```bash
python -m pip install -r requirements.txt
pytest -q
python -m scripts.run_swingup
python -m scripts.run_swingup --mass-multiplier 1.2 --output mass_1_2.csv
```

The example integrates the plant at 0.002 s while holding each control action
for 0.02 s. It writes a CSV containing the unwrapped and wrapped angles,
control, and active mode.

The default gains are only a transparent starting point. Tune them against
your exact physical convention and fixed training/evaluation setup. In
particular, inspect track use and force saturation rather than declaring a
swing-up successful from angle alone.

## Where later project pieces fit

Keep the physics controller unchanged and add later pieces beside it:

```text
cartpole/
├── residual.py       # u_physics + beta * actor_action
├── shield.py         # scalar projection under nominal one-step Delta V rule
├── policy.py         # one narrow adapter around the trained actor
└── metrics.py        # fixed points, Jacobians, basin and rollout metrics
```

The plant should receive only the final clipped action. Log the physics action,
raw residual, proposed combined action, shielded action, and applied action as
separate values. That separation will matter when interpreting saturation and
shield activity.

## Scope of the claims

- LQR eigenvalues describe the unsaturated nominal linearization.
- The energy calculation motivates the swing-up law under stated model
  assumptions.
- Neither item proves global safety or a nonlinear region of attraction.
- Basin results from simulation should be called an **empirical basin**.
- Under mismatch, report finite-difference local evidence and use
  **loss of local stability** only when that evidence supports it.
