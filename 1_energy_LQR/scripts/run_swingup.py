import argparse
import csv
import math
from pathlib import Path

import numpy as np

from cartpole import CartPoleParams, PhysicsController, step_zoh
from cartpole.math_utils import wrap_angle


def run(mass_multiplier, duration, output_path):
    nominal = CartPoleParams()
    plant = CartPoleParams(
        cart_mass=nominal.cart_mass,
        pole_mass=mass_multiplier * nominal.pole_mass,
        pole_com_length=nominal.pole_com_length,
        pole_inertia=nominal.pole_inertia,
        gravity=nominal.gravity,
        cart_damping=nominal.cart_damping,
        pole_damping=nominal.pole_damping,
        control_period=nominal.control_period,
        integration_step=nominal.integration_step,
        max_force=nominal.max_force,
        track_limit=nominal.track_limit,
    )

    controller = PhysicsController(nominal)
    state = np.array([0.0, math.pi, 0.0, 0.0], dtype=float)
    rows = []

    steps = round(duration / plant.control_period)
    for step in range(steps + 1):
        time = step * plant.control_period
        output = controller.action(state)
        rows.append(
            [
                time,
                *state,
                wrap_angle(state[1]),
                output.force,
                output.mode,
            ]
        )
        if step < steps:
            state = step_zoh(state, output.force, plant)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time",
                "x",
                "theta_unwrapped",
                "x_dot",
                "theta_dot",
                "theta_wrapped",
                "force",
                "mode",
            ]
        )
        writer.writerows(rows)

    final_angle = abs(wrap_angle(state[1]))
    max_position = max(abs(row[1]) for row in rows)
    balance_samples = sum(row[-1] == "balance" for row in rows)
    print(f"mass multiplier: {mass_multiplier:.3f}")
    print(f"final |theta|:  {final_angle:.4f} rad")
    print(f"final |x|:      {abs(state[0]):.4f} m")
    print(f"max |x|:        {max_position:.4f} m")
    print(f"balance time:   {balance_samples * plant.control_period:.2f} s")
    print(f"track violated: {max_position > plant.track_limit}")
    print(f"log written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass-multiplier", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--output", type=Path, default=Path("swingup_log.csv")
    )
    args = parser.parse_args()
    run(args.mass_multiplier, args.duration, args.output)


if __name__ == "__main__":
    main()
