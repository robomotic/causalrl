"""Tests for the tabular world model."""

import numpy as np
import pytest

from causalrl.world_models.tabular import TabularWorldModel


class TestTabularWorldModel:
    """Test count-based tabular world model."""

    def test_unseen_state_zero_confidence(self):
        """Unvisited (state, action) pairs have zero confidence."""
        wm = TabularWorldModel(num_states=10, num_actions=4)
        pred = wm.predict(state=0, action=0)
        assert pred.confidence == 0.0

    def test_confidence_increases_with_visits(self):
        """Confidence grows as more transitions are observed."""
        wm = TabularWorldModel(num_states=10, num_actions=4)

        # Record 20 transitions for (state=0, action=1)
        for _ in range(20):
            wm.update(state=0, action=1, reward=5.0, next_state=3)

        pred = wm.predict(state=0, action=1)
        assert pred.confidence > 0.8

    def test_reward_estimate_converges(self):
        """Reward estimate converges to mean of observed rewards."""
        wm = TabularWorldModel(num_states=5, num_actions=2)

        # Record 100 transitions with mean reward = 7.0
        rng = np.random.default_rng(42)
        for _ in range(100):
            r = rng.normal(7.0, 1.0)
            wm.update(state=0, action=0, reward=r, next_state=1)

        pred = wm.predict(state=0, action=0)
        assert pred.reward == pytest.approx(7.0, abs=0.3)

    def test_transition_probs_reflect_data(self):
        """Transition probabilities match observed frequencies."""
        wm = TabularWorldModel(num_states=3, num_actions=2)

        # 70% of transitions go to state 1, 30% to state 2
        for _ in range(70):
            wm.update(state=0, action=0, reward=1.0, next_state=1)
        for _ in range(30):
            wm.update(state=0, action=0, reward=1.0, next_state=2)

        probs = wm.get_transition_probs(state=0, action=0)
        assert probs[0] == pytest.approx(0.0)
        assert probs[1] == pytest.approx(0.7)
        assert probs[2] == pytest.approx(0.3)

    def test_counterfactual_query_uses_learned_model(self):
        """Counterfactual queries return predictions based on learned dynamics."""
        wm = TabularWorldModel(num_states=5, num_actions=3)

        # Train model on action=1 from state=0
        for _ in range(50):
            wm.update(state=0, action=1, reward=3.0, next_state=2)

        # Ask counterfactual: "what if I had taken action=1 instead of action=0?"
        cf = wm.counterfactual_query(
            state=0,
            action_not_taken=1,
            actual_action=0,
            actual_outcome=(4, 1.0),
        )
        assert cf.reward == pytest.approx(3.0)
        assert cf.confidence > 0

    def test_visit_counts_tracked(self):
        """Visit counts are correctly maintained."""
        wm = TabularWorldModel(num_states=3, num_actions=2)
        assert wm.get_transition_count(0, 0) == 0

        for _ in range(5):
            wm.update(state=0, action=0, reward=1.0, next_state=1)

        assert wm.get_transition_count(0, 0) == 5
        assert wm.get_transition_count(0, 1) == 0
