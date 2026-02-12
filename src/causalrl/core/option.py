"""Options framework for temporally extended actions.

Implements the Sutton, Precup & Singh (1999) options framework with
extensions for the dual-process "habit vs. deliberation" arbitration
described in the proposal.

An Option is a temporally extended action with:
- An initiation set I(s): states where the option can start
- A policy π(s): action selection within the option
- A termination condition β(s): probability of terminating in each state

The habit arbitration mechanism decides whether to:
1. Execute a cached option (habitual, low-cost)
2. Engage the world model for forward planning
3. Engage the world model for counterfactual analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Option:
    """A temporally extended action (macro-action).

    Parameters
    ----------
    name : str
        Human-readable name for the option.
    policy : Callable[[int], int]
        Maps states to primitive actions.
    initiation_set : set[int] | None
        States where this option can be initiated. None → everywhere.
    termination_prob : Callable[[int], float]
        Maps states to termination probability β(s).
    """

    name: str
    policy: Callable[[int], int]
    initiation_set: set[int] | None = None
    termination_prob: Callable[[int], float] = field(
        default_factory=lambda: lambda s: 0.0
    )

    def can_initiate(self, state: int) -> bool:
        """Check if this option can be started in the given state."""
        if self.initiation_set is None:
            return True
        return state in self.initiation_set

    def select_action(self, state: int) -> int:
        """Select an action according to the option's internal policy."""
        return self.policy(state)

    def should_terminate(self, state: int, rng: np.random.Generator) -> bool:
        """Determine whether the option should terminate in this state."""
        return rng.random() < self.termination_prob(state)


class HabitArbitrator:
    """Decides when to use habits vs. deliberation.

    Implements the dual-process (System 1 / System 2) arbitration:
    - Low prediction error + low novelty → habitual execution (fast, cheap)
    - High prediction error or high uncertainty → engage world model

    Parameters
    ----------
    surprise_threshold : float
        Prediction error above this triggers deliberation.
    novelty_threshold : float
        State novelty above this triggers deliberation.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.5,
        novelty_threshold: float = 0.3,
    ):
        self.surprise_threshold = surprise_threshold
        self.novelty_threshold = novelty_threshold
        self._state_visits: dict[int, int] = {}

    def record_visit(self, state: int) -> None:
        """Record a state visit for novelty tracking."""
        self._state_visits[state] = self._state_visits.get(state, 0) + 1

    def get_novelty(self, state: int) -> float:
        """Compute novelty of a state (inverse of visit frequency)."""
        visits = self._state_visits.get(state, 0)
        return 1.0 / (1.0 + visits)

    def should_deliberate(
        self,
        prediction_error: float,
        state: int,
    ) -> bool:
        """Determine whether to engage deliberative reasoning.

        Returns True if the agent should use the world model for planning
        or counterfactual analysis (System 2). Returns False if habitual
        execution (System 1) is sufficient.
        """
        novelty = self.get_novelty(state)
        return (
            abs(prediction_error) > self.surprise_threshold
            or novelty > self.novelty_threshold
        )
