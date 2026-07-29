import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.dummy_backend import DummyPhysicsController, DummyPlant
from residual_sac.config import EnvironmentConfig
from residual_sac.environment import ResidualCartPoleEnvironment
from residual_sac.evaluation import evaluate_mass_grid
from residual_sac.sac import SACAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen residual SAC")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/sac_checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        default="artifacts/dummy_evaluation.json",
    )
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=500)
    arguments = parser.parse_args()

    config = EnvironmentConfig(
        beta=arguments.beta,
        max_steps=arguments.max_steps,
    )
    environment = ResidualCartPoleEnvironment(
        DummyPlant(config.control_period),
        DummyPhysicsController(config.u_max),
        config,
    )
    agent = SACAgent.from_checkpoint(
        arguments.checkpoint,
        device="cpu",
        load_optimizers=False,
    )

    mass_scales = np.linspace(0.6, 1.4, 9)
    # Dummy shared states. Replace with the project's paired evaluation set.
    initial_states = np.array(
        [
            [0.0, 0.10, 0.0, 0.0],
            [0.0, -0.10, 0.0, 0.0],
            [0.2, 0.20, 0.0, -0.1],
        ],
        dtype=np.float64,
    )
    results = evaluate_mass_grid(
        environment,
        agent,
        mass_scales,
        initial_states,
    )

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(results)} deterministic rollouts: {output_path}")


if __name__ == "__main__":
    main()

