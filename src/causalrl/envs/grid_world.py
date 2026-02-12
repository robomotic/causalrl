"""Grid-world navigation with regret scenarios.

Benchmark Task 2 from the proposal: tests whether the agent develops
regret-sensitive policies. Features:

- Multiple goals with varying reward magnitudes
- Post-episode counterfactual feedback (agent observes unchosen goal rewards)
- Configurable stochasticity in transitions
- Foraging scenarios: guaranteed small reward vs. risky large reward

Human OFC-lesion patients fail to adjust choices based on foregone
alternatives — this environment tests the computational equivalent.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RegretGridWorldEnv(gym.Env):
    """Grid-world with counterfactual feedback and regret scenarios.

    The agent navigates a grid to reach goal positions. Upon reaching
    a goal, it observes what rewards were at alternative goals (counterfactual
    feedback), enabling regret/relief learning.

    Parameters
    ----------
    grid_size : int
        Size of the NxN grid.
    num_goals : int
        Number of goal positions with rewards.
    stochasticity : float
        Probability of random transition (0.0 → deterministic).
    max_steps : int
        Maximum episode length.
    reward_range : tuple[float, float]
        Range of goal rewards (uniform random within range).
    safe_reward : float | None
        If set, one goal always has this fixed reward (the "safe" option)
        while others are randomly drawn (risky options).
    seed : int
        Random seed.
    """

    metadata = {"render_modes": ["ansi"]}
    EMPTY = 0
    WALL = 1
    AGENT = 2
    GOAL = 3

    # Actions: 0=up, 1=right, 2=down, 3=left
    ACTION_DELTAS = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(
        self,
        grid_size: int = 5,
        num_goals: int = 3,
        stochasticity: float = 0.0,
        max_steps: int = 50,
        reward_range: tuple[float, float] = (1.0, 10.0),
        safe_reward: float | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_goals = num_goals
        self.stochasticity = stochasticity
        self.max_steps = max_steps
        self.reward_range = reward_range
        self.safe_reward = safe_reward

        # State: flattened grid position
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        self.action_space = spaces.Discrete(4)

        self.rng = np.random.default_rng(seed)
        self._agent_pos: tuple[int, int] | None = None
        self._goals: dict[tuple[int, int], float] = {}
        self._step_count = 0
        self._reached_goals: set[tuple[int, int]] = set()

    def _pos_to_state(self, pos: tuple[int, int]) -> int:
        """Convert (row, col) to flat state index."""
        return pos[0] * self.grid_size + pos[1]

    def _state_to_pos(self, state: int) -> tuple[int, int]:
        """Convert flat state index to (row, col)."""
        return divmod(state, self.grid_size)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[int, dict]:
        """Reset the grid world with new goal positions and rewards."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self._reached_goals = set()

        # Place agent at center
        center = self.grid_size // 2
        self._agent_pos = (center, center)

        # Place goals at random positions (not at agent start)
        all_positions = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) != self._agent_pos
        ]
        goal_positions = self.rng.choice(
            len(all_positions), self.num_goals, replace=False
        )

        self._goals = {}
        for i, idx in enumerate(goal_positions):
            pos = all_positions[idx]
            if self.safe_reward is not None and i == 0:
                # First goal is the "safe" option
                self._goals[pos] = self.safe_reward
            else:
                self._goals[pos] = float(
                    self.rng.uniform(*self.reward_range)
                )

        state = self._pos_to_state(self._agent_pos)
        return state, {"goals": dict(self._goals), "agent_pos": self._agent_pos}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Take a step in the grid world.

        When the agent reaches a goal, it receives that goal's reward
        and the episode ends. The info dict includes 'counterfactual_rewards'
        showing what other goals offered.
        """
        assert self._agent_pos is not None, "Call reset() first"

        self._step_count += 1

        # Apply stochasticity
        if self.rng.random() < self.stochasticity:
            action = int(self.rng.integers(4))

        # Compute new position
        dr, dc = self.ACTION_DELTAS[action]
        new_r = np.clip(self._agent_pos[0] + dr, 0, self.grid_size - 1)
        new_c = np.clip(self._agent_pos[1] + dc, 0, self.grid_size - 1)
        self._agent_pos = (int(new_r), int(new_c))

        state = self._pos_to_state(self._agent_pos)
        reward = -0.1  # small step penalty to encourage efficiency
        terminated = False
        truncated = self._step_count >= self.max_steps

        info: dict[str, Any] = {
            "agent_pos": self._agent_pos,
            "step": self._step_count,
        }

        # Check if agent reached a goal
        if self._agent_pos in self._goals:
            reward = self._goals[self._agent_pos]
            terminated = True
            self._reached_goals.add(self._agent_pos)

            # Counterfactual feedback: reveal unchosen goal rewards
            counterfactual_rewards = {
                self._pos_to_state(pos): r
                for pos, r in self._goals.items()
                if pos != self._agent_pos
            }
            info["counterfactual_rewards"] = counterfactual_rewards
            info["chosen_goal_reward"] = reward
            info["best_possible_reward"] = max(self._goals.values())
            info["regret"] = info["best_possible_reward"] - reward

        return state, reward, terminated, truncated, info

    def get_outcome(self, state: int, action: int) -> tuple[int, float]:
        """Oracle interface: return deterministic outcome for (state, action)."""
        pos = self._state_to_pos(state)
        dr, dc = self.ACTION_DELTAS[action]
        new_r = int(np.clip(pos[0] + dr, 0, self.grid_size - 1))
        new_c = int(np.clip(pos[1] + dc, 0, self.grid_size - 1))
        new_pos = (new_r, new_c)
        new_state = self._pos_to_state(new_pos)

        if new_pos in self._goals:
            return new_state, self._goals[new_pos]
        return new_state, -0.1

    def render(self) -> str | None:
        """Render the grid as ASCII."""
        if self.render_mode != "ansi":
            return None

        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U3")
        for pos, reward in self._goals.items():
            grid[pos] = f"G{reward:.0f}"
        if self._agent_pos:
            grid[self._agent_pos] = "A"
        return "\n".join(" ".join(row) for row in grid)
