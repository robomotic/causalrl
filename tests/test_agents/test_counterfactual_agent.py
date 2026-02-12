"""Tests for the counterfactual RL agent."""

import numpy as np
import pytest

from causalrl.agents.counterfactual_rl import CounterfactualRLAgent
from causalrl.agents.standard_rl import StandardRLAgent
from causalrl.core.ofc import OFCComparator
from causalrl.world_models.tabular import TabularWorldModel


class TestCounterfactualRLAgent:
    """Test the full counterfactual learning loop."""

    def setup_method(self):
        self.num_states = 5
        self.num_actions = 3
        self.wm = TabularWorldModel(self.num_states, self.num_actions)
        self.ofc = OFCComparator(mode="fixed", alpha=0.5)
        self.agent = CounterfactualRLAgent(
            num_states=self.num_states,
            num_actions=self.num_actions,
            world_model=self.wm,
            ofc=self.ofc,
            beta=0.1,
            gamma_cf=0.05,
            epsilon=0.0,  # greedy for deterministic testing
            seed=42,
        )

    def test_update_returns_diagnostics(self):
        """Update should return full diagnostic dict."""
        info = self.agent.update(
            state=0, action=1, reward=5.0, next_state=2, done=False
        )
        assert "delta_actual" in info
        assert "delta_counterfactual" in info
        assert "delta_composite" in info
        assert "alpha" in info
        assert "cf_confidence" in info

    def test_chosen_value_updated(self):
        """Q-value of chosen action should change after update."""
        old_value = self.agent.get_value(0, 1)
        self.agent.update(state=0, action=1, reward=5.0, next_state=0, done=True)
        new_value = self.agent.get_value(0, 1)
        assert new_value != old_value

    def test_unchosen_values_updated(self):
        """Q-values of unchosen actions should change (counterfactual update)."""
        # First, give the world model some data to work with
        for _ in range(10):
            self.wm.update(state=0, action=0, reward=3.0, next_state=1)
            self.wm.update(state=0, action=2, reward=7.0, next_state=3)

        old_val_0 = self.agent.get_value(0, 0)
        old_val_2 = self.agent.get_value(0, 2)

        # Agent takes action 1 — actions 0 and 2 should get CF updates
        self.agent.update(state=0, action=1, reward=5.0, next_state=2, done=True)

        new_val_0 = self.agent.get_value(0, 0)
        new_val_2 = self.agent.get_value(0, 2)

        # Both unchosen actions should have been updated
        assert new_val_0 != old_val_0 or new_val_2 != old_val_2

    def test_world_model_learns_from_transitions(self):
        """World model should be updated on each agent step."""
        assert self.wm.get_transition_count(0, 1) == 0
        self.agent.update(state=0, action=1, reward=5.0, next_state=2, done=False)
        assert self.wm.get_transition_count(0, 1) == 1

    def test_regret_tracking(self):
        """Agent tracks regret/relief ratio."""
        # Fresh agent has 50/50 ratio
        assert self.agent.regret_ratio == 0.5

        # Update that should trigger regret (best alternative > reward)
        self.agent.q_values[0, 0] = 10.0  # alternative is much better
        self.agent.update(state=0, action=1, reward=1.0, next_state=0, done=True)

        assert self.agent.regret_ratio > 0.5


class TestStandardRLAgent:
    """Test the standard Q-learning baseline."""

    def test_no_counterfactual_in_diagnostics(self):
        """Standard agent reports zero counterfactual signal."""
        agent = StandardRLAgent(num_states=5, num_actions=3, seed=42)
        info = agent.update(state=0, action=1, reward=5.0, next_state=2, done=False)
        assert info["delta_counterfactual"] == 0.0
        assert info["alpha"] == 0.0

    def test_q_learning_convergence(self):
        """Standard Q-learning converges on a simple deterministic task."""
        agent = StandardRLAgent(
            num_states=1, num_actions=3, beta=0.1, epsilon=0.0, seed=42
        )

        # Action 2 always gives reward 10, others give 0
        for _ in range(200):
            for a in range(3):
                r = 10.0 if a == 2 else 0.0
                agent.update(state=0, action=a, reward=r, next_state=0, done=True)

        # Agent should learn that action 2 is best
        best = np.argmax(agent.q_values[0])
        assert best == 2
