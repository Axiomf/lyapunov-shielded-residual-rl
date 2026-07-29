"""Optional aggregation of shield diagnostics."""

from dataclasses import dataclass

from .controller import ControllerOutput


@dataclass
class ShieldStats:
    """Simple counters that an evaluator may update once per control step."""

    steps: int = 0
    inside_region_steps: int = 0
    projected_steps: int = 0
    infeasible_steps: int = 0
    constraint_violation_steps: int = 0
    nominal_delta_value_sum: float = 0.0

    def update(self, output: ControllerOutput) -> None:
        """Add one controller output to the counters."""

        result = output.shield
        self.steps += 1
        self.inside_region_steps += int(result.inside_region)
        self.projected_steps += int(result.projected)
        self.infeasible_steps += int(
            result.inside_region and not result.feasible
        )
        self.constraint_violation_steps += int(
            result.inside_region and not result.constraint_satisfied
        )
        self.nominal_delta_value_sum += result.nominal_delta_value

    @staticmethod
    def _ratio(count: int, total: int) -> float:
        return count / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        """Return counts and rates in a serialization-friendly form."""

        return {
            "steps": self.steps,
            "inside_region_steps": self.inside_region_steps,
            "projected_steps": self.projected_steps,
            "infeasible_steps": self.infeasible_steps,
            "constraint_violation_steps": self.constraint_violation_steps,
            "inside_region_rate": self._ratio(
                self.inside_region_steps,
                self.steps,
            ),
            "projection_rate": self._ratio(
                self.projected_steps,
                self.inside_region_steps,
            ),
            "infeasibility_rate": self._ratio(
                self.infeasible_steps,
                self.inside_region_steps,
            ),
            "constraint_violation_rate": self._ratio(
                self.constraint_violation_steps,
                self.inside_region_steps,
            ),
            "mean_nominal_delta_value": (
                self.nominal_delta_value_sum / self.steps
                if self.steps
                else 0.0
            ),
        }

