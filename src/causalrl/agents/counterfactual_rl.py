"""Counterfactual RL agent — the core implementation.

Implements the Zhang et al. (2015) hybrid learning framework extended
for the OAK architecture. After each action:

1. Standard TD update on the chosen action (as in Q-learning)
2. World model query for each unchosen alternative
3. OFC comparator computes α based on counterfactual reliability
4. Composite prediction error δ_composite = δ_actual + α·δ_cf
5. Both chosen and unchosen option values are updated

This agent learns from what it did AND what it didn't do,
implementing the dopaminergic composite learning signal described
in Kishida et al. (2016).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.core.agent import BaseAgent
from causalrl.core.world_model import BaseWorldModel
from causalrl.core.prediction_error import CompositeErrorSignal, ErrorSignals
from causalrl.core.ofc import OFCComparator


class CounterfactualRLAgent(BaseAgent):
    """RL agent with counterfactual learning via composite prediction errors.

    Parameters
    ----------
    num_states : int
        Size of the state space.
    num_actions : int
        Size of the action space.
    world_model : BaseWorldModel
        World model for generating counterfactual predictions.
    ofc : OFCComparator
        OFC comparator for α modulation.
    beta : float
        Learning rate for chosen option updates.
    gamma_cf : float
        Learning rate for unchosen option updates.
    epsilon : float
        Exploration rate (ε-greedy).
    discount : float
        Discount factor.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        world_model: BaseWorldModel,
        ofc: OFCComparator | None = None,
        beta: float = 0.1,
        gamma_cf: float = 0.05,
        epsilon: float = 0.1,
        discount: float = 0.99,
        seed: int = 42,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            beta=beta,
            gamma_cf=gamma_cf,
            epsilon=epsilon,
            discount=discount,
            seed=seed,
        )
        self.world_model = world_model
        self.ofc = ofc or OFCComparator(mode="fixed", alpha=0.5)
        self.error_computer = CompositeErrorSignal()

        # Diagnostics tracking
        self._step_count = 0
        self._regret_history: list[float] = []
        self._relief_history: list[float] = []

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform composite update: actual + counterfactual.

        Parameters
        ----------
        ... (existing params)
        info : dict
            Optional dictionary that may contain 'all_rewards' for 
            full-feedback environments.
        """
        self._step_count += 1

        # Step 1: Update world model
        # If full feedback is available, update world model for ALL actions
        if info and "all_rewards" in info:
            all_rewards = info["all_rewards"]
            for a, r in enumerate(all_rewards):
                wm_info = self.world_model.update(state, a, r, next_state)
        else:
            # Standard partial feedback update
            wm_info = self.world_model.update(state, action, reward, next_state)

        # Step 2: Query counterfactual outcomes for unchosen actions
        unchosen_actions = [a for a in range(self.num_actions) if a != action]
        cf_predictions = {}
        for alt_action in unchosen_actions:
            cf_pred = self.world_model.counterfactual_query(
                state=state,
                action_not_taken=alt_action,
                actual_action=action,
                actual_outcome=(next_state, reward),
            )
            cf_predictions[alt_action] = cf_pred

        # Step 3: Compute α via OFC comparator
        # Use mean confidence across counterfactual predictions
        if cf_predictions:
            mean_cf_confidence = np.mean(
                [p.confidence for p in cf_predictions.values()]
            )
        else:
            mean_cf_confidence = 0.0

        alpha = self.ofc.compute_alpha(
            cf_confidence=mean_cf_confidence,
            transition_count=self.world_model.get_transition_count(state, action),
        )

        # Step 4: Compute composite prediction error
        chosen_value = self.q_values[state, action]
        unchosen_values = np.array(
            [self.q_values[state, a] for a in unchosen_actions]
        )
        next_value = 0.0 if done else self.get_state_value(next_state)

        errors = self.error_computer.compute(
            reward=reward,
            chosen_value=chosen_value,
            unchosen_values=unchosen_values,
            alpha=alpha,
            next_state_value=next_value,
            discount=self.discount,
            done=done,
        )

        # Step 5: Update chosen option value with composite error
        self.q_values[state, action] += self.beta * errors.delta_composite

        # Step 6: Update unchosen option values with counterfactual error
        # Each unchosen action gets updated based on its simulated outcome
        for alt_action in unchosen_actions:
            cf_pred = cf_predictions[alt_action]
            # Counterfactual-specific error for this alternative
            cf_reward = cf_pred.reward
            cf_delta = cf_reward - self.q_values[state, alt_action]
            # Weight by confidence and γ_cf
            self.q_values[state, alt_action] += (
                self.gamma_cf * cf_pred.confidence * cf_delta
            )

        # Track regret/relief for diagnostics
        if errors.delta_counterfactual > 0:
            self._regret_history.append(errors.delta_counterfactual)
        else:
            self._relief_history.append(abs(errors.delta_counterfactual))

        return {
            "delta_actual": errors.delta_actual,
            "delta_counterfactual": errors.delta_counterfactual,
            "delta_composite": errors.delta_composite,
            "alpha": alpha,
            "cf_confidence": float(mean_cf_confidence),
            "regret": errors.delta_counterfactual > 0,
            "world_model": wm_info,
            "step": self._step_count,
        }

    @property
    def mean_regret(self) -> float:
        """Average regret signal magnitude over history."""
        if not self._regret_history:
            return 0.0
        return float(np.mean(self._regret_history))

    @property
    def mean_relief(self) -> float:
        """Average relief signal magnitude over history."""
        if not self._relief_history:
            return 0.0
        return float(np.mean(self._relief_history))

    @property
    def regret_ratio(self) -> float:
        """Fraction of updates where agent experienced regret (vs. relief)."""
        total = len(self._regret_history) + len(self._relief_history)
        if total == 0:
            return 0.5
        return len(self._regret_history) / total
