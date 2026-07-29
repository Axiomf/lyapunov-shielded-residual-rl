"""
Replace these method bodies with calls to the completed physics/LQR project.
"""

import numpy as np

from residual_sac.interfaces import PhysicsController, Plant


class ProjectPlant(Plant):
    def __init__(self, existing_plant: object) -> None:
        self.existing_plant = existing_plant

    def reset(self, initial_state: np.ndarray, pole_mass_scale: float) -> np.ndarray:
        # TODO:
        # 1. Set plant pole mass to pole_mass_scale * nominal_pole_mass.
        # 2. Set the four-dimensional state to initial_state.
        # 3. Reset integrator/time state.
        # 4. Return a copy of the actual state.
        raise NotImplementedError("connect this method to the real plant")

    def step(self, force: float) -> tuple[np.ndarray, bool]:
        # TODO:
        # Hold force constant and call the real RK4 code for one 0.02 s
        # control interval. Return (next_state_copy, abs(x) > 2.4).
        raise NotImplementedError("connect this method to the real RK4 step")


class ProjectPhysicsController(PhysicsController):
    def __init__(self, existing_controller: object) -> None:
        self.existing_controller = existing_controller

    def reset(self) -> None:
        # TODO: reset the swing-up/LQR hysteresis mode and controller memory.
        raise NotImplementedError("connect this method to the controller reset")

    def action(self, state: np.ndarray) -> float:
        # TODO: call the existing energy-shaping/hysteretic-LQR controller.
        # Return its scalar force before the residual is added.
        raise NotImplementedError("connect this method to the physics controller")

