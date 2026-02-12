"""Base agent abstraction for counterfactual RL.

The agent interface is designed around the Zhang et al. (2015) framework,
where after each action selection, the agent can perform both a standard
RL update and a counterfactual update for unchosen alternatives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents.

    Agents operate in discrete action spaces. The interface supports both
    standard RL agents (which only learn from experienced transitions) and
    counterfactual agents (which also learn from simulated alternatives).

    Parameters
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Size of the action space.
    beta : float
        Learning rate for chosen option value updates.
    gamma_cf : float
        Learning rate for unchosen option value updates (counterfactual).
    epsilon : float
        Exploration rate for ε-greedy action selection.
    discount : float
        Discount factor for future rewards.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        beta: float = 0.1,
        gamma_cf: float = 0.05,
        epsilon: float = 0.1,
        discount: float = 0.99,
        seed: int = 42,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.beta = beta
        self.gamma_cf = gamma_cf
        self.epsilon = epsilon
        self.discount = discount
        self.rng = np.random.default_rng(seed)

        # Q-value table: Q(s, a)
        self.q_values = np.zeros((num_states, num_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        """Select an action using ε-greedy policy.

        Parameters
        ----------
        state : int
            Current state index.

        Returns
        -------
        int
            Selected action index.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        return int(np.argmax(self.q_values[state]))

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform a learning update after taking an action.

        Parameters
        ----------
        state : int
            State in which action was taken.
        action : int
            Action that was taken.
        reward : float
            Reward received.
        next_state : int
            Resulting state.
        done : bool
            Whether the episode terminated.

        Returns
        -------
        dict
            Update diagnostics (e.g., delta values, alpha used).
        """

    def get_value(self, state: int, action: int) -> float:
        """Get the Q-value for a state-action pair."""
        return float(self.q_values[state, action])

    def get_state_value(self, state: int) -> float:
        """Get V(s) = max_a Q(s, a)."""
        return float(np.max(self.q_values[state]))

    def reset(self) -> None:
        """Reset all learned values."""
        self.q_values[:] = 0.0
