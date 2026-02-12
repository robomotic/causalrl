"""Tests for the composite prediction error signal."""

import numpy as np
import pytest

from causalrl.core.prediction_error import CompositeErrorSignal, ErrorSignals


class TestCompositeErrorSignal:
    """Test the core counterfactual learning mechanism."""

    def setup_method(self):
        self.computer = CompositeErrorSignal()

    def test_no_counterfactual_effect_with_alpha_zero(self):
        """With α=0, composite error equals standard TD error."""
        errors = self.computer.compute(
            reward=5.0,
            chosen_value=3.0,
            unchosen_values=np.array([8.0, 2.0]),
            alpha=0.0,
            next_state_value=4.0,
            discount=0.99,
            done=False,
        )
        # δ_actual = 5 + 0.99*4 - 3 = 5.96
        expected_actual = 5.0 + 0.99 * 4.0 - 3.0
        assert errors.delta_actual == pytest.approx(expected_actual)
        assert errors.delta_composite == pytest.approx(expected_actual)
        assert errors.alpha == 0.0

    def test_regret_signal_positive_when_missed_better(self):
        """δ_cf > 0 when the best unchosen option was better than reward."""
        errors = self.computer.compute(
            reward=3.0,
            chosen_value=2.0,
            unchosen_values=np.array([7.0, 1.0]),
            alpha=0.5,
        )
        # δ_cf = max(7, 1) - 3 = 4 → regret
        assert errors.delta_counterfactual == pytest.approx(4.0)
        assert errors.best_unchosen_value == pytest.approx(7.0)

    def test_relief_signal_negative_when_chose_best(self):
        """δ_cf < 0 when reward was better than all alternatives."""
        errors = self.computer.compute(
            reward=8.0,
            chosen_value=5.0,
            unchosen_values=np.array([3.0, 2.0]),
            alpha=0.5,
        )
        # δ_cf = max(3, 2) - 8 = -5 → relief
        assert errors.delta_counterfactual == pytest.approx(-5.0)

    def test_composite_combines_actual_and_cf(self):
        """δ_composite = δ_actual + α · δ_cf."""
        errors = self.computer.compute(
            reward=5.0,
            chosen_value=3.0,
            unchosen_values=np.array([8.0]),
            alpha=0.5,
            next_state_value=0.0,
            discount=0.99,
            done=True,
        )
        # δ_actual = 5 - 3 = 2 (done, so no bootstrap)
        # δ_cf = 8 - 5 = 3
        # δ_composite = 2 + 0.5*3 = 3.5
        assert errors.delta_actual == pytest.approx(2.0)
        assert errors.delta_counterfactual == pytest.approx(3.0)
        assert errors.delta_composite == pytest.approx(3.5)

    def test_no_unchosen_actions_no_regret(self):
        """With no alternatives, δ_cf = chosen_value - reward."""
        errors = self.computer.compute(
            reward=5.0,
            chosen_value=3.0,
            unchosen_values=np.array([]),
            alpha=0.5,
            done=True,
        )
        # No alternatives → best_unchosen = chosen_value = 3
        # δ_cf = 3 - 5 = -2
        assert errors.delta_counterfactual == pytest.approx(-2.0)

    def test_terminal_state_no_bootstrap(self):
        """At terminal states, V(s') is not used."""
        errors = self.computer.compute(
            reward=10.0,
            chosen_value=5.0,
            unchosen_values=np.array([3.0]),
            alpha=0.0,
            next_state_value=100.0,  # should be ignored
            discount=0.99,
            done=True,
        )
        assert errors.delta_actual == pytest.approx(5.0)  # 10 - 5
