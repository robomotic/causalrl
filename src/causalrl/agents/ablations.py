"""Ablation agent variants for controlled experiments.

Provides systematic ablations described in the proposal:
- α=0: No counterfactual weight (equivalent to standard RL)
- Fixed α: Constant counterfactual weight
- Adaptive α: OFC-modulated based on epistemic uncertainty
- Oracle world model: Perfect counterfactual outcomes
- Learned world model: Noisy counterfactual outcomes

These isolate the contribution of individual architectural components.
"""

from __future__ import annotations

from causalrl.agents.counterfactual_rl import CounterfactualRLAgent
from causalrl.core.ofc import OFCComparator
from causalrl.core.world_model import BaseWorldModel


def create_no_cf_agent(
    num_states: int,
    num_actions: int,
    world_model: BaseWorldModel,
    **kwargs,
) -> CounterfactualRLAgent:
    """Create agent with α=0 — no counterfactual learning.

    Uses the CF agent infrastructure but with the OFC locked to α=0,
    so no counterfactual signal contributes to learning. This is
    functionally equivalent to StandardRLAgent but allows comparing
    the same code path with/without CF.
    """
    ofc = OFCComparator(mode="fixed", alpha=0.0)
    return CounterfactualRLAgent(
        num_states=num_states,
        num_actions=num_actions,
        world_model=world_model,
        ofc=ofc,
        **kwargs,
    )


def create_fixed_alpha_agent(
    num_states: int,
    num_actions: int,
    world_model: BaseWorldModel,
    alpha: float = 0.5,
    **kwargs,
) -> CounterfactualRLAgent:
    """Create agent with fixed α — constant counterfactual weight."""
    ofc = OFCComparator(mode="fixed", alpha=alpha)
    return CounterfactualRLAgent(
        num_states=num_states,
        num_actions=num_actions,
        world_model=world_model,
        ofc=ofc,
        **kwargs,
    )


def create_adaptive_alpha_agent(
    num_states: int,
    num_actions: int,
    world_model: BaseWorldModel,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    uncertainty_sensitivity: float = 1.0,
    **kwargs,
) -> CounterfactualRLAgent:
    """Create agent with adaptive α — OFC-modulated by uncertainty."""
    ofc = OFCComparator(
        mode="adaptive",
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        uncertainty_sensitivity=uncertainty_sensitivity,
    )
    return CounterfactualRLAgent(
        num_states=num_states,
        num_actions=num_actions,
        world_model=world_model,
        ofc=ofc,
        **kwargs,
    )
