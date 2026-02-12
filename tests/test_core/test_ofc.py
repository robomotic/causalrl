"""Tests for the OFC comparator (α modulation)."""

import pytest

from causalrl.core.ofc import OFCComparator


class TestOFCComparator:
    """Test OFC-based α computation."""

    def test_fixed_mode_returns_constant(self):
        """Fixed mode always returns the configured α."""
        ofc = OFCComparator(mode="fixed", alpha=0.7)
        assert ofc.compute_alpha() == pytest.approx(0.7)
        assert ofc.compute_alpha(cf_confidence=0.1) == pytest.approx(0.7)
        assert ofc.compute_alpha(cf_confidence=1.0, transition_count=1000) == pytest.approx(0.7)

    def test_adaptive_increases_with_confidence(self):
        """Adaptive α increases as counterfactual confidence grows."""
        ofc = OFCComparator(mode="adaptive", alpha_min=0.0, alpha_max=1.0)
        low = ofc.compute_alpha(cf_confidence=0.1, transition_count=5)
        high = ofc.compute_alpha(cf_confidence=0.9, transition_count=50)
        assert high > low

    def test_adaptive_increases_with_visit_count(self):
        """More visits → higher confidence → higher α."""
        ofc = OFCComparator(mode="adaptive", alpha_min=0.0, alpha_max=1.0)
        few = ofc.compute_alpha(cf_confidence=0.8, transition_count=1)
        many = ofc.compute_alpha(cf_confidence=0.8, transition_count=100)
        assert many > few

    def test_adaptive_respects_bounds(self):
        """α stays within [alpha_min, alpha_max]."""
        ofc = OFCComparator(mode="adaptive", alpha_min=0.2, alpha_max=0.8)
        low = ofc.compute_alpha(cf_confidence=0.0, transition_count=0)
        high = ofc.compute_alpha(cf_confidence=1.0, transition_count=1000)
        assert low >= 0.2
        assert high <= 0.8

    def test_zero_count_zero_alpha(self):
        """No observations → minimal α (agent shouldn't trust counterfactuals)."""
        ofc = OFCComparator(mode="adaptive", alpha_min=0.0, alpha_max=1.0)
        alpha = ofc.compute_alpha(cf_confidence=1.0, transition_count=0)
        assert alpha == pytest.approx(0.0)

    def test_invalid_mode_raises(self):
        """Invalid OFC mode should raise ValueError."""
        with pytest.raises(ValueError, match="fixed.*adaptive"):
            OFCComparator(mode="invalid")
