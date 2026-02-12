"""Tests for benchmark environments."""

import numpy as np
import pytest

from causalrl.envs.bandit import CounterfactualBanditEnv
from causalrl.envs.grid_world import RegretGridWorldEnv


class TestCounterfactualBanditEnv:
    """Test the multi-armed bandit environment."""

    def test_reset_returns_valid_state(self):
        env = CounterfactualBanditEnv(num_arms=5, seed=42)
        state, info = env.reset()
        assert state == 0
        assert "reward_means" in info
        assert len(info["reward_means"]) == 5

    def test_partial_feedback_no_all_rewards(self):
        """Partial feedback mode doesn't reveal all arm rewards."""
        env = CounterfactualBanditEnv(feedback_mode="partial", seed=42)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "all_rewards" not in info

    def test_full_feedback_reveals_all_rewards(self):
        """Full feedback mode reveals all arm rewards."""
        env = CounterfactualBanditEnv(feedback_mode="full", seed=42)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "all_rewards" in info
        assert len(info["all_rewards"]) == 10

    def test_optimal_action_tracking(self):
        """Info should indicate whether chosen action was optimal."""
        env = CounterfactualBanditEnv(num_arms=3, seed=42)
        _, reset_info = env.reset()
        optimal = int(np.argmax(reset_info["reward_means"]))
        _, _, _, _, info = env.step(optimal)
        assert info["is_optimal"]

    def test_reward_shift(self):
        """Reward shift should change the optimal arm."""
        env = CounterfactualBanditEnv(
            num_arms=5, reward_shift_interval=10, seed=42
        )
        env.reset()

        # Force through 10 steps to trigger a shift
        rewards_before = env._reward_means.copy()
        for _ in range(10):
            env.step(0)
        rewards_after = env._reward_means.copy()
        assert not np.allclose(rewards_before, rewards_after)

    def test_oracle_interface(self):
        """get_outcome should return expected reward for an arm."""
        env = CounterfactualBanditEnv(num_arms=3, seed=42)
        env.reset()
        state, reward = env.get_outcome(0, 1)
        assert state == 0
        assert reward == pytest.approx(env._reward_means[1])


class TestRegretGridWorldEnv:
    """Test the grid-world navigation environment."""

    def test_reset_places_agent_at_center(self):
        env = RegretGridWorldEnv(grid_size=5, seed=42)
        state, info = env.reset()
        assert info["agent_pos"] == (2, 2)

    def test_goals_created(self):
        env = RegretGridWorldEnv(grid_size=5, num_goals=3, seed=42)
        _, info = env.reset()
        assert len(info["goals"]) == 3

    def test_step_returns_valid_state(self):
        env = RegretGridWorldEnv(grid_size=5, seed=42)
        env.reset()
        state, reward, term, trunc, info = env.step(0)  # move up
        assert 0 <= state < 25

    def test_reaching_goal_terminates(self):
        """Episode should end when agent reaches a goal."""
        env = RegretGridWorldEnv(grid_size=3, num_goals=4, seed=42)
        env.reset()

        # Keep stepping until we hit a goal or max steps
        done = False
        for _ in range(100):
            state, reward, term, trunc, info = env.step(
                env.action_space.sample()
            )
            if term or trunc:
                done = True
                break
        assert done

    def test_counterfactual_feedback_on_goal(self):
        """Reaching a goal should provide counterfactual reward info."""
        env = RegretGridWorldEnv(grid_size=3, num_goals=8, seed=42)
        env.reset()

        # Step until reaching a goal
        for _ in range(100):
            state, reward, term, trunc, info = env.step(
                env.action_space.sample()
            )
            if term:
                assert "counterfactual_rewards" in info
                assert "regret" in info
                break

    def test_boundary_clipping(self):
        """Agent should not move outside the grid."""
        env = RegretGridWorldEnv(grid_size=3, seed=42)
        env.reset()

        # Try to go up from top row many times
        for _ in range(10):
            env.step(0)  # up

        assert env._agent_pos[0] >= 0
