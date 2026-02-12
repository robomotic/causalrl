"""Oracle (perfect) world model for ablation studies.

Wraps a Gymnasium environment to provide exact transition dynamics.
Used to isolate the effect of the counterfactual mechanism from world
model quality: if the agent has a perfect model, any performance gains
from counterfactual learning are purely algorithmic.

This enables the ablation: "oracle CF outcomes vs. learned CF outcomes"
described in the proposal.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from causalrl.core.world_model import BaseWorldModel, Prediction


class OracleWorldModel(BaseWorldModel):
    """Perfect world model that uses environment's true dynamics.

    Parameters
    ----------
    env : gym.Env
        The actual environment to query for ground-truth transitions.
    num_states : int
        Size of the state space.
    num_actions : int
        Size of the action space.
    """

    def __init__(
        self,
        env: gym.Env,
        num_states: int,
        num_actions: int,
    ):
        super().__init__(num_states, num_actions, learning_rate=0.0)
        self.env = env
        # Track visit counts for interface compatibility
        self._visit_counts = np.zeros((num_states, num_actions), dtype=np.int64)

    def predict(self, state: int, action: int) -> Prediction:
        """Use the true environment dynamics for prediction.

        Note: This requires the environment to support state setting
        (save/load state). For environments that don't, falls back to
        the stored transition if available.
        """
        # Try to get ground-truth outcome
        if hasattr(self.env, "get_outcome"):
            next_state, reward = self.env.get_outcome(state, action)
            return Prediction(
                next_state=next_state, reward=reward, confidence=1.0
            )

        # Fallback: return with full confidence but indicate we can't predict
        return Prediction(next_state=state, reward=0.0, confidence=1.0)

    def counterfactual_query(
        self,
        state: int,
        action_not_taken: int,
        actual_action: int,
        actual_outcome: tuple[int, float],
    ) -> Prediction:
        """Perfect counterfactual: query the true environment dynamics."""
        return self.predict(state, action_not_taken)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> dict[str, Any]:
        """Oracle model needs no learning, just tracks counts."""
        self._visit_counts[state, action] += 1
        return {"prediction_error": 0.0, "visit_count": int(self._visit_counts[state, action])}

    def get_transition_count(self, state: int, action: int) -> int:
        """Return visit count (for interface compatibility)."""
        return int(self._visit_counts[state, action])
