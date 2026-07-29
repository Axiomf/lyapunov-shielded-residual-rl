from cartpole.actor import ZeroActor
from cartpole.config import ExperimentConfig
from cartpole.controllers import ControllerKind, build_controller
from cartpole.data import State
from cartpole.simulation import run_rollout


def main() -> None:
    config = ExperimentConfig()
    actor = ZeroActor()
    initial_state = State(x=0.0, theta=2.8, x_dot=0.0, theta_dot=0.0)

    for kind in ControllerKind:
        controller = build_controller(kind, config, actor)
        rollout = run_rollout(
            controller=controller,
            initial_state=initial_state,
            mu=1.0,
            config=config,
        )
        print(
            f"{kind.value:24s} "
            f"steps={len(rollout.transitions):4d} "
            f"return={rollout.total_reward:8.2f} "
            f"track_violation={rollout.track_violation}"
        )


if __name__ == "__main__":
    main()

