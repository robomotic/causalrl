"""Tabular world model for discrete state/action spaces.

Learns transition dynamics from experience by maintaining count-based
estimates of T(s'|s,a) and R(s,a). Supports both forward planning and
counterfactual queries.

The confidence metric is derived from visit counts: more observations
of a (state, action) pair → higher confidence in counterfactual
predictions for that pair.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.core.world_model import BaseWorldModel, Prediction


class TabularWorldModel(BaseWorldModel):
    """Count-based tabular world model.

    Maintains:
    - Transition counts N(s, a, s') for estimating P(s'|s, a)
    - Running mean rewards R̂(s, a) for estimating E[R|s, a]
    - Total visit counts N(s, a) for confidence estimation

    Parameters
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Size of the action space.
    learning_rate : float
        Exponential moving average rate for reward estimation.
        If 0, uses exact running mean instead.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.0,
    ):
        super().__init__(num_states, num_actions, learning_rate)

        # Transition counts: N(s, a, s')
        self.transition_counts = np.zeros(
            (num_states, num_actions, num_states), dtype=np.float64
        )
        # Visit counts: N(s, a)
        self.visit_counts = np.zeros(
            (num_states, num_actions), dtype=np.float64
        )
        # Running mean rewards: R̂(s, a)
        self.reward_estimates = np.zeros(
            (num_states, num_actions), dtype=np.float64
        )

    def predict(self, state: int, action: int) -> Prediction:
        """Forward planning: predict outcome of (state, action).

        Uses the learned transition model. If the action has never been
        observed from this state, returns a uniform random prediction
        with low confidence.
        """
        total = self.visit_counts[state, action]
        if total == 0:
            # No data: uniform prediction, zero confidence
            next_state = int(np.random.randint(self.num_states))
            return Prediction(next_state=next_state, reward=0.0, confidence=0.0)

        # Sample next state from learned distribution
        probs = self.transition_counts[state, action] / total
        next_state = int(np.random.choice(self.num_states, p=probs))
        reward = self.reward_estimates[state, action]

        # Confidence scales with observation count
        confidence = 1.0 - np.exp(-0.1 * total)

        return Prediction(
            next_state=next_state,
            reward=reward,
            confidence=float(confidence),
        )

    def counterfactual_query(
        self,
        state: int,
        action_not_taken: int,
        actual_action: int,
        actual_outcome: tuple[int, float],
    ) -> Prediction:
        """Counterfactual: what would have happened under action_not_taken?

        Constrained by the same starting state (episodic constraint from
        hippocampus). The confidence reflects how well the model knows the
        alternative action's dynamics from the given state.

        This query type engages more 'conflict monitoring' (per the neural
        evidence) because the model must maintain both the actual outcome
        and the hypothetical simultaneously.
        """
        prediction = self.predict(state, action_not_taken)

        # For counterfactual queries, we can potentially use the actual
        # outcome context to refine confidence. If the model is well-calibrated
        # for the actual action, it's more likely to be reliable for
        # alternatives from the same state.
        actual_count = self.visit_counts[state, actual_action]
        cf_count = self.visit_counts[state, action_not_taken]

        # Combined confidence: geometric mean of actual reliability
        # and counterfactual reliability
        if actual_count > 0 and cf_count > 0:
            actual_conf = 1.0 - np.exp(-0.1 * actual_count)
            cf_conf = 1.0 - np.exp(-0.1 * cf_count)
            prediction.confidence = float(np.sqrt(actual_conf * cf_conf))
        elif cf_count > 0:
            prediction.confidence = float(1.0 - np.exp(-0.1 * cf_count)) * 0.8
        else:
            prediction.confidence = 0.0

        return prediction

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> dict[str, Any]:
        """Update model from observed transition.

        Returns prediction error (difference between predicted and actual
        reward) as a diagnostic.
        """
        # Record transition
        self.transition_counts[state, action, next_state] += 1.0
        self.visit_counts[state, action] += 1.0

        # Update reward estimate
        old_reward = self.reward_estimates[state, action]
        if self.learning_rate > 0:
            # Exponential moving average
            self.reward_estimates[state, action] += self.learning_rate * (
                reward - old_reward
            )
        else:
            # Exact running mean
            n = self.visit_counts[state, action]
            self.reward_estimates[state, action] = (
                old_reward * (n - 1) + reward
            ) / n

        return {
            "prediction_error": reward - old_reward,
            "visit_count": int(self.visit_counts[state, action]),
        }

    def get_transition_count(self, state: int, action: int) -> int:
        """Return observation count for (state, action)."""
        return int(self.visit_counts[state, action])

    def get_transition_probs(self, state: int, action: int) -> np.ndarray:
        """Get the learned transition probability distribution P(s'|s,a)."""
        total = self.visit_counts[state, action]
        if total == 0:
            return np.ones(self.num_states) / self.num_states
        return self.transition_counts[state, action] / total
