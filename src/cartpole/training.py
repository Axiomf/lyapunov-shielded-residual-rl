"""Define the SAC-training boundary and episode-wise mass randomization.

Theoretical role
----------------
This module corresponds to the **learning and domain-randomization part** of the
project.  At the beginning of training rollout ``j``, it samples

    mu_j ~ Uniform(mu_min, mu_max),
    m_p,j = mu_j * m_p,0,

and keeps ``mu_j`` fixed while the sampled-data system evolves:

    s[k + 1] = F_h(s[k], u[k]; mu_j).

Thus training uses a distribution of fixed dynamical systems, not a pole mass
that drifts during one trajectory.  The actor observes only the normalized state
``o[k] = normalize_state(s[k])``.  It does not observe ``mu_j``, so mass mismatch
acts as an unobserved, episode-level plant parameter.

SAC optimizes the residual policy over trajectories from this training
distribution.  The selected controller still determines how the actor proposal
becomes the applied force: direct bounded addition for residual SAC, or the same
proposal followed by the nominal Lyapunov shield for shielded residual SAC.
Both agents must otherwise share rewards, initial-state sampling, plant settings,
and data conventions so their comparison remains paired.

This module intentionally does not implement SAC or stability analysis.  Domain
randomization is a robustness-training method, not a proof of robustness.  The
shield enforces only its nominal-model one-step test, and a returned frozen actor
must later be analyzed as part of the complete sampled-data closed loop.  Local
stability, realized Lyapunov change, success, and empirical-basin results are
therefore evaluation evidence rather than guarantees supplied by training.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .actor import Actor
from .config import ExperimentConfig
from .controllers import ControllerKind


@dataclass(frozen=True)
class TrainingResult:
    """Common output format expected from any future SAC implementation.

    This small data object keeps a third-party RL library behind one stable
    project interface.  Evaluation code should need the learned :class:`Actor`,
    not the library-specific model class or replay-buffer representation.

    Parameters
    ----------
    actor:
        Learned residual policy implementing ``Actor.act(observation,
        deterministic)``.  Training normally requests stochastic actions;
        frozen-policy evaluation must request deterministic actions.
    metrics:
        Final scalar training summaries, for example episode return or actor and
        critic losses.  Use stable names and the same definitions for both SAC
        controllers so results remain comparable.
    extra:
        Optional library-specific objects or metadata, such as checkpoints or
        optimizer state.  Core simulation and evaluation code should not depend
        on this dictionary.

    Notes
    -----
    ``frozen=True`` prevents replacing these three fields after construction; it
    does not mathematically freeze the parameters inside ``actor`` or make the
    two dictionaries immutable.  The SAC adapter must disable learning during
    evaluation.  Training metrics are optimization diagnostics, not evidence of
    local stability or an empirical basin.
    """

    actor: Actor
    metrics: dict[str, float]
    extra: dict[str, Any]


def sample_rollout_mass(
    rng: np.random.Generator,
    config: ExperimentConfig,
) -> float:
    """Draw one training pole-mass multiplier for a complete rollout.

    Mathematically, this returns one realization of

    ``mu ~ Uniform(config.training.mu_min, config.training.mu_max)``.

    The rollout plant then uses ``m_p = mu*m_p,0``, while the physics controller,
    LQR construction, and shield continue to use the nominal mass ``m_p,0``.
    This separation creates the model mismatch studied in the project.

    Parameters
    ----------
    rng:
        NumPy random-number generator owned by the training run.  Constructing it
        from the experiment seed makes the sequence of rollout masses
        reproducible.
    config:
        Synchronized experiment configuration containing the training interval.

    Returns
    -------
    float
        Dimensionless pole-mass multiplier for one rollout.

    Notes
    -----
    Call this function exactly once when an episode starts, then pass the returned
    value unchanged through that episode.  Calling it at every control step would
    create a drifting-parameter experiment outside this project's scope.
    Evaluation should instead use the prescribed shared mass grid, not fresh
    random draws from the narrower training interval.
    """

    # ``Generator.uniform`` returns a NumPy scalar; converting to ``float`` keeps
    # the value consistent with the scalar fields used by project data objects.
    return float(
        rng.uniform(config.training.mu_min, config.training.mu_max)
    )


def train_sac(
    controller_kind: ControllerKind,
    config: ExperimentConfig,
    seed: int,
) -> TrainingResult:
    """Train one residual actor through a future SAC-library adapter.

    This is the only intended connection point between the project and a chosen
    SAC package.  Keeping the adapter here prevents library-specific arrays,
    networks, and replay objects from changing the shared ``State``,
    ``ControlDecision``, ``Transition``, and ``Rollout`` conventions.

    In schematic mathematical form, SAC learns policy parameters ``phi`` from an
    objective based on expected discounted rewards (and an entropy term during
    training), where the expectation includes

    ``mu ~ Uniform(mu_min, mu_max)``.

    This optimization objective is distinct from the candidate Lyapunov function
    ``V(z) = z.T P z`` used by the shield.  A high SAC return does not itself
    imply Lyapunov decrease or local stability.

    Parameters
    ----------
    controller_kind:
        Must be ``RESIDUAL_SAC`` or ``SHIELDED_RESIDUAL_SAC``.  The physics
        baseline has no learned actor and is therefore rejected.
    config:
        Single source of plant, controller, observation, rollout, and training
        settings.  The two SAC variants must receive synchronized settings.
    seed:
        Independent experiment seed.  Seed the actor/critic initialization,
        stochastic action sampling, replay sampling, rollout-mass generator, and
        any SAC-library generator from this value in a reproducible way.

    Returns
    -------
    TrainingResult
        The learned actor through the small project protocol, comparable scalar
        metrics, and optional library-specific state.

    Expected adapter behavior
    -------------------------
    For each training rollout, the eventual implementation should:

    1. Draw ``mu`` once with :func:`sample_rollout_mass` and keep it fixed.
    2. Build only the actual rollout plant with ``m_p = mu*m_p,0``; keep the
       controller and shield model nominal.
    3. Give the actor only ``normalize_state(state)`` and never append ``mu``.
    4. Use stochastic actor actions during SAC data collection and keep the actor
       output in ``[-1, 1]`` before scaling it by ``beta``.
    5. Apply the selected residual or shielded-residual controller, then advance
       the nonlinear plant with the common zero-order-hold/RK4 step.
    6. Store replay values compatible with the project's time-aligned
       ``Transition`` convention: current state/action information, one shared
       reward, next state, and termination flag.
    7. Use identical rewards, initial-state distributions, horizons, and action
       conventions for both residual agents.
    8. Return an actor that performs no parameter updates when later called for
       deterministic frozen-policy evaluation.

    Notes
    -----
    For shielded residual SAC, sampled transitions and rewards must reflect the
    force actually applied after any shield projection or LQR fallback.  Logging
    should preserve the actor proposal and ``ControlDecision`` diagnostics so
    projection and infeasibility can later be measured separately.

    The final policy remains an empirical learned feedback law.  Upright fixed
    points, equilibrium bias, finite-difference closed-loop Jacobians, spectral
    radius, realized Lyapunov change, and empirical basin geometry belong to the
    evaluation pipeline and cannot be inferred from this return value alone.
    """

    # Exactly one of the three project comparators has no SAC training stage.
    if controller_kind == ControllerKind.PHYSICS:
        raise ValueError("The physics baseline is not trained with SAC.")

    # These arguments document the future adapter contract.  Deleting the local
    # names makes their intentional non-use explicit until an SAC library is wired
    # in, while the clear exception prevents accidental placeholder training.
    del config, seed
    raise NotImplementedError(
        "Connect your chosen SAC implementation at this one boundary."
    )

