#!/usr/bin/env python3
"""Small command-line entry point for the complete study."""

from __future__ import annotations

import argparse
from pathlib import Path

from cartpole_study.config import load_config
from cartpole_study.experiment import CONTROLLERS, evaluate_study, train_sac


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="path to the YAML configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train residual SAC models")
    train.add_argument(
        "--controller",
        choices=("residual", "shielded", "all"),
        default="all",
    )
    train.add_argument("--seed", type=int, action="append")
    train.add_argument("--timesteps", type=int)

    evaluate = subparsers.add_parser(
        "evaluate", help="run frozen-mass deterministic evaluation"
    )
    evaluate.add_argument(
        "--controller",
        choices=(*CONTROLLERS, "all"),
        default="all",
    )
    evaluate.add_argument("--seed", type=int, action="append")
    evaluate.add_argument("--mass", type=float, action="append")
    evaluate.add_argument(
        "--quick",
        action="store_true",
        help="use a small episode count and basin grid for a smoke check",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(Path(args.config))
    seeds = tuple(args.seed or config.study.seeds)

    if args.command == "train":
        controllers = (
            ("residual", "shielded")
            if args.controller == "all"
            else (args.controller,)
        )
        for controller in controllers:
            for path in train_sac(
                config, controller, seeds, timesteps=args.timesteps
            ):
                print(path)
        return

    controllers = (
        CONTROLLERS if args.controller == "all" else (args.controller,)
    )
    paths = evaluate_study(
        config,
        controllers,
        seeds,
        masses=args.mass,
        quick=args.quick,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
