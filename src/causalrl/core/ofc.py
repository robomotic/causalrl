"""OFC (Orbitofrontal Cortex) Comparator — α modulation.

The OFC component determines how much weight (α) to give counterfactual
information in the composite learning signal. It acts as a gain controller:

- When the world model is confident about counterfactual predictions (high
  episodic constraint, many observations), α increases → agent learns more
  from what it didn't do.
- When counterfactuals are speculative (low model confidence, sparse data),
  α decreases → agent relies more on actual experience.

This mirrors the biological OFC's role as a comparator between actual and
counterfactual values (Coricelli et al., 2005), and the fronto-parietal
control network's adaptive weighting of β and γ (Zhang et al., 2015).

Two modes are supported:
- **Fixed**: α is a constant hyperparameter (for ablation baselines)
- **Adaptive**: α is modulated by the world model's confidence in its
  counterfactual predictions
"""

from __future__ import annotations

import numpy as np


class OFCComparator:
    """Orbitofrontal cortex comparator for counterfactual weight modulation.

    Parameters
    ----------
    mode : str
        "fixed" for constant α, "adaptive" for uncertainty-modulated α.
    alpha : float
        Base α value (used directly in fixed mode, as center in adaptive).
    alpha_min : float
        Minimum α in adaptive mode.
    alpha_max : float
        Maximum α in adaptive mode.
    uncertainty_sensitivity : float
        How strongly model confidence modulates α in adaptive mode.
        Higher values → α more responsive to confidence.
    """

    def __init__(
        self,
        mode: str = "fixed",
        alpha: float = 0.5,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        uncertainty_sensitivity: float = 1.0,
    ):
        if mode not in ("fixed", "adaptive"):
            raise ValueError(f"OFC mode must be 'fixed' or 'adaptive', got '{mode}'")
        self.mode = mode
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.uncertainty_sensitivity = uncertainty_sensitivity

    def compute_alpha(
        self,
        cf_confidence: float = 1.0,
        transition_count: int = 0,
    ) -> float:
        """Compute the counterfactual weight α.

        Parameters
        ----------
        cf_confidence : float
            World model's confidence in its counterfactual prediction (0-1).
            Only used in adaptive mode.
        transition_count : int
            Number of times the (state, action) pair has been observed.
            Higher counts → more reliable counterfactuals.

        Returns
        -------
        float
            The α value to use in the composite prediction error.
        """
        if self.mode == "fixed":
            return self.alpha

        # Adaptive mode: modulate α by counterfactual reliability
        # Combine confidence from the world model with visit-count-based
        # reliability estimate
        count_confidence = 1.0 - np.exp(-0.1 * transition_count)
        combined_confidence = cf_confidence * count_confidence

        # Scale α between alpha_min and alpha_max based on confidence
        scaled = combined_confidence ** (1.0 / self.uncertainty_sensitivity)
        alpha = self.alpha_min + scaled * (self.alpha_max - self.alpha_min)

        return float(np.clip(alpha, self.alpha_min, self.alpha_max))
