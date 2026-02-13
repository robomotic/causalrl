"""Baseline bandit algorithms.

Implements standard epsilon-greedy with sample-average updates,
which is the textbook definition for stationary multi-armed bandits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.core.agent import BaseAgent


class EpsilonGreedyBandit(BaseAgent):
    """Standard epsilon-greedy agent using sample averages.

    Unlike StandardRLAgent (which uses constant learning rate β),
    this agent uses 1/N(a) as the step size, which guarantees
    convergence to the true mean for stationary distributions.

    Q(a) ← Q(a) + (1/N(a)) * [r - Q(a)]
    """

    def __init__(
        self,
        num_arms: int,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        # Initialize with num_states=1 for bandit
        super().__init__(
            num_states=1,
            num_actions=num_arms,
            beta=0.0,  # Unused, we use 1/N
            gamma_cf=0.0,
            epsilon=epsilon,
            discount=0.0,
            seed=seed,
        )
        self.action_counts = np.zeros(num_arms, dtype=np.int32)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update Q-value using sample average rule."""
        self.action_counts[action] += 1
        
        # Step size alpha = 1 / N(a)
        alpha = 1.0 / self.action_counts[action]
        
        # Prediction error
        delta = reward - self.q_values[state, action]
        
        # Update mean estimate
        self.q_values[state, action] += alpha * delta

        return {
            "delta_actual": delta,
            "delta_counterfactual": 0.0,
            "delta_composite": delta,
            "alpha": 0.0,
            "learning_rate": alpha,
        }
