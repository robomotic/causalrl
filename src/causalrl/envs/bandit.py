"""Multi-armed bandit environment with counterfactual feedback.

Benchmark Task 1 from the proposal: the cleanest test of counterfactual
learning. Supports:

- Partial feedback: agent sees only the chosen arm's reward (standard bandit)
- Full feedback: agent sees rewards for ALL arms (for comparison)
- Reward drift: non-stationary reward distributions for testing adaptation

The environment follows the Gymnasium API for consistency.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CounterfactualBanditEnv(gym.Env):
    """Multi-armed bandit with optional counterfactual feedback.

    Parameters
    ----------
    num_arms : int
        Number of arms/actions.
    feedback_mode : str
        "partial" — only chosen arm reward revealed.
        "full" — all arms' rewards revealed (counterfactual feedback).
    reward_drift_rate : float
        Rate at which mean rewards change per step. 0.0 → stationary.
    reward_std : float
        Standard deviation of reward noise per arm.
    reward_shift_interval : int
        If > 0, abruptly shift optimal arm every N steps (for adaptation tests).
    seed : int
        Random seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_arms: int = 10,
        feedback_mode: str = "partial",
        reward_drift_rate: float = 0.0,
        reward_std: float = 1.0,
        reward_shift_interval: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        assert feedback_mode in ("partial", "full")

        self.num_arms = num_arms
        self.feedback_mode = feedback_mode
        self.reward_drift_rate = reward_drift_rate
        self.reward_std = reward_std
        self.reward_shift_interval = reward_shift_interval

        self.action_space = spaces.Discrete(num_arms)
        self.observation_space = spaces.Discrete(1)  # single state

        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._reward_means: np.ndarray | None = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[int, dict]:
        """Reset the bandit. Regenerates reward means if seed is provided."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._reward_means = self.rng.standard_normal(self.num_arms)

        # Initialize reward means if not already done
        if self._reward_means is None:
            self._reward_means = self.rng.standard_normal(self.num_arms)
            
        self._step_count = 0

        return 0, {"reward_means": self._reward_means.copy()}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Pull an arm and receive reward.

        Returns
        -------
        observation : int
            Always 0 (single-state).
        reward : float
            Reward from the chosen arm.
        terminated : bool
            Always False (continuing task).
        truncated : bool
            Always False.
        info : dict
            Contains 'all_rewards' if feedback_mode="full",
            'optimal_action', 'is_optimal', and 'reward_means'.
        """
        assert self._reward_means is not None, "Call reset() first"

        self._step_count += 1

        # Apply reward drift
        if self.reward_drift_rate > 0:
            drift = self.rng.standard_normal(self.num_arms) * self.reward_drift_rate
            self._reward_means += drift

        # Apply abrupt shift if configured
        if (
            self.reward_shift_interval > 0
            and self._step_count % self.reward_shift_interval == 0
        ):
            self._reward_means = self.rng.standard_normal(self.num_arms)

        # Generate rewards for all arms (even if not all revealed)
        all_rewards = (
            self._reward_means
            + self.rng.standard_normal(self.num_arms) * self.reward_std
        )

        chosen_reward = float(all_rewards[action])
        optimal_action = int(np.argmax(self._reward_means))

        info: dict[str, Any] = {
            "optimal_action": optimal_action,
            "is_optimal": action == optimal_action,
            "reward_means": self._reward_means.copy(),
            "step": self._step_count,
        }

        if self.feedback_mode == "full":
            info["all_rewards"] = all_rewards.copy()

        return 0, chosen_reward, False, False, info

    def get_outcome(self, state: int, action: int) -> tuple[int, float]:
        """Oracle interface: return expected outcome for (state, action).

        Used by OracleWorldModel for perfect counterfactual predictions.
        """
        assert self._reward_means is not None
        return 0, float(self._reward_means[action])
