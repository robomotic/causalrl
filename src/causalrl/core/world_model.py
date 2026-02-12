"""World model abstraction for counterfactual simulation.

The world model serves as the "Knowledge" component in the OAK architecture.
It provides two distinct query modes:

1. **Forward planning** (predict): Simulate future trajectories under different
   actions — used for prospective decision-making.

2. **Counterfactual query** (counterfactual_query): Simulate what *would have
   happened* under an unchosen action from a known past state — used for
   retrospective credit assignment.

Per the neuroimaging evidence (Van Hoeck et al., 2013), these two modes engage
overlapping but distinct neural substrates, with counterfactual queries requiring
additional constraint checking against episodic memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    """Result of a world model query.

    Attributes
    ----------
    next_state : int
        Predicted next state.
    reward : float
        Predicted reward.
    confidence : float
        Model confidence in this prediction (0.0 to 1.0).
        Used by the OFC comparator to modulate α.
    """

    next_state: int
    reward: float
    confidence: float = 1.0


class BaseWorldModel(ABC):
    """Abstract base class for world models.

    World models learn transition dynamics T(s, a) → (s', r) from experience
    and support both forward and counterfactual queries.

    Parameters
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Size of the action space.
    learning_rate : float
        Learning rate for model updates.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate

    @abstractmethod
    def predict(self, state: int, action: int) -> Prediction:
        """Forward planning query: predict outcome of taking action in state.

        This corresponds to EFT (Episodic Future Thinking) in the neuroscience
        literature — open-ended, schema-driven simulation.

        Parameters
        ----------
        state : int
            Current state.
        action : int
            Action to simulate.

        Returns
        -------
        Prediction
            Predicted next state, reward, and confidence.
        """

    @abstractmethod
    def counterfactual_query(
        self,
        state: int,
        action_not_taken: int,
        actual_action: int,
        actual_outcome: tuple[int, float],
    ) -> Prediction:
        """Counterfactual query: what would have happened under a different action?

        This corresponds to eCFT (Episodic Counterfactual Thinking) — constrained
        by what actually happened, requiring the model to maintain both actual and
        hypothetical representations.

        The `actual_outcome` parameter provides the episodic constraint: the
        counterfactual must be consistent with the same initial state and context.

        Parameters
        ----------
        state : int
            The state where the decision was made.
        action_not_taken : int
            The alternative action to simulate.
        actual_action : int
            The action that was actually taken.
        actual_outcome : tuple[int, float]
            (next_state, reward) that actually occurred — episodic constraint.

        Returns
        -------
        Prediction
            Predicted counterfactual outcome with confidence.
        """

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> dict[str, Any]:
        """Update the world model from an observed transition.

        Parameters
        ----------
        state : int
            State before transition.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : int
            State after transition.

        Returns
        -------
        dict
            Update diagnostics (e.g., prediction error, model loss).
        """

    @abstractmethod
    def get_transition_count(self, state: int, action: int) -> int:
        """Return how many times this (state, action) has been observed.

        Used by the OFC comparator to assess model reliability for
        counterfactual queries.
        """
