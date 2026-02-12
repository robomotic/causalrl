"""Composite prediction error signal.

Implements the dopaminergic composite learning signal from the proposal:

    δ_actual = R_obtained - V(s, o_chosen)
    δ_counterfactual = max(V(s, o_other)) - R_obtained
    δ_composite = δ_actual + α · δ_counterfactual

This mirrors Kishida et al. (2016): subsecond dopamine fluctuations in human
striatum encode superposed error signals about actual and counterfactual reward.

The composite error drives value updates for both chosen and unchosen options:
    V(s, o_chosen)   ← V(s, o_chosen)   + β · δ_composite
    V(s, o_unchosen) ← V(s, o_unchosen) + γ · δ_counterfactual
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorSignals:
    """Container for all prediction error components.

    Attributes
    ----------
    delta_actual : float
        Standard TD error: R - V(s, a_chosen).
    delta_counterfactual : float
        Opportunity cost: max(V(s, a_other)) - R.
        Positive → regret (missed a better option).
        Negative → relief (chose better than best alternative).
    delta_composite : float
        Combined signal: δ_actual + α · δ_counterfactual.
    alpha : float
        Counterfactual weight used for this computation.
    best_unchosen_value : float
        Value of the best unchosen alternative.
    """

    delta_actual: float
    delta_counterfactual: float
    delta_composite: float
    alpha: float
    best_unchosen_value: float


class CompositeErrorSignal:
    """Computes the composite dopaminergic prediction error.

    This is the core learning mechanism of counterfactual RL: each real
    experience generates both a standard TD error (what did I get vs. what
    I expected?) and a counterfactual error (what did I get vs. what I
    could have gotten?).

    The two are combined with weight α, which can be fixed or dynamically
    modulated by the OFC comparator based on counterfactual reliability.
    """

    def compute(
        self,
        reward: float,
        chosen_value: float,
        unchosen_values: np.ndarray,
        alpha: float,
        next_state_value: float = 0.0,
        discount: float = 0.99,
        done: bool = False,
    ) -> ErrorSignals:
        """Compute the composite prediction error.

        Parameters
        ----------
        reward : float
            Reward actually received.
        chosen_value : float
            Current Q(s, a_chosen) — value of the action taken.
        unchosen_values : np.ndarray
            Q(s, a_i) for all unchosen alternatives.
        alpha : float
            Counterfactual weight from the OFC comparator.
        next_state_value : float
            V(s') = max_a Q(s', a) for bootstrapping (0 if terminal).
        discount : float
            Discount factor γ.
        done : bool
            Whether the episode terminated.

        Returns
        -------
        ErrorSignals
            All components of the composite error.
        """
        # Standard TD error: δ_actual = r + γ·V(s') - V(s, a_chosen)
        target = reward + (0.0 if done else discount * next_state_value)
        delta_actual = target - chosen_value

        # Counterfactual error: δ_cf = max(V(s, a_other)) - R
        # Positive → regret (best alternative was better than what I got)
        # Negative → relief (what I got was better than best alternative)
        if len(unchosen_values) > 0:
            best_unchosen = float(np.max(unchosen_values))
        else:
            best_unchosen = chosen_value  # no alternatives → no regret
        delta_counterfactual = best_unchosen - reward

        # Composite: δ_composite = δ_actual + α · δ_cf
        delta_composite = delta_actual + alpha * delta_counterfactual

        return ErrorSignals(
            delta_actual=delta_actual,
            delta_counterfactual=delta_counterfactual,
            delta_composite=delta_composite,
            alpha=alpha,
            best_unchosen_value=best_unchosen,
        )
