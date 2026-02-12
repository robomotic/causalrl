"""Standard Q-learning agent — baseline without counterfactual reasoning.

This agent learns only from experienced (state, action, reward) tuples using
standard temporal-difference (TD) learning. It serves as the control condition
in ablation studies: comparing this agent to the counterfactual agent isolates
the contribution of counterfactual updates.

Update rule:
    Q(s, a) ← Q(s, a) + β · [r + γ·max_a' Q(s', a') - Q(s, a)]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.core.agent import BaseAgent


class StandardRLAgent(BaseAgent):
    """Standard Q-learning agent with no counterfactual mechanism.

    This is the α=0 baseline: it never updates unchosen actions and
    uses only the standard TD error for learning.

    Parameters
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Size of the action space.
    beta : float
        Learning rate.
    epsilon : float
        Exploration rate (ε-greedy).
    discount : float
        Discount factor.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        beta: float = 0.1,
        epsilon: float = 0.1,
        discount: float = 0.99,
        seed: int = 42,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            beta=beta,
            gamma_cf=0.0,  # no counterfactual learning
            epsilon=epsilon,
            discount=discount,
            seed=seed,
        )

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Standard Q-learning update.

        Returns
        -------
        dict
            Contains 'delta_actual' — the standard TD error.
        """
        # TD target
        next_value = 0.0 if done else self.get_state_value(next_state)
        target = reward + self.discount * next_value

        # TD error
        delta = target - self.q_values[state, action]

        # Update Q-value
        self.q_values[state, action] += self.beta * delta

        return {
            "delta_actual": delta,
            "delta_counterfactual": 0.0,
            "delta_composite": delta,
            "alpha": 0.0,
        }
