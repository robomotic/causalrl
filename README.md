# Counterfactual RL for the OAK Architecture

**Implementing counterfactual reinforcement learning inspired by human developmental cognition and dopaminergic composite learning signals.**

This project implements the counterfactual RL framework described in the [PDP Proposal for Openmind Research Institute](paper/PDP_Proposal_OpenMind.pdf), which integrates Pearl's Ladder of Causality with the OAK (Options And Knowledge) architecture through neurally-grounded composite prediction errors.

## Core Idea

Standard RL agents learn only from what they experience. **Counterfactual RL agents also learn from what they *didn't* do**, by:

1. Using a world model to simulate "what would have happened" under unchosen actions
2. Computing a **composite prediction error** that combines standard TD error with opportunity cost (regret/relief)
3. Updating values for both chosen and unchosen options simultaneously

This mirrors the dopaminergic composite error signals observed in human striatum (Kishida et al., 2016).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Agent Loop                     │
│                                                  │
│  State ──► Action Selection (ε-greedy)           │
│              │                                   │
│              ▼                                   │
│  Environment Step → reward, next_state           │
│              │                                   │
│              ├──► World Model Update             │
│              │                                   │
│              ├──► Counterfactual Queries          │
│              │    (simulate unchosen actions)     │
│              │                                   │
│              ├──► OFC Comparator (compute α)     │
│              │                                   │
│              └──► Composite Error Signal         │
│                   δ = δ_actual + α·δ_cf          │
│                        │                         │
│              ┌─────────┴──────────┐              │
│              ▼                    ▼              │
│    Update Q(chosen)     Update Q(unchosen)       │
└─────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Train a counterfactual agent on a 10-arm bandit
python scripts/train.py --env bandit --agent counterfactual --episodes 2000

# Train with a config file
python scripts/train.py --config configs/bandit_comparison.yaml

# Run tests
pytest tests/ -v
```

## Project Structure

| Module | Description |
|--------|-------------|
| `core/prediction_error.py` | Composite δ = δ_actual + α·δ_cf |
| `core/ofc.py` | OFC comparator (adaptive α modulation) |
| `core/world_model.py` | Forward planning + counterfactual queries |
| `agents/counterfactual_rl.py` | Full CF agent (Zhang et al. framework) |
| `agents/standard_rl.py` | Q-learning baseline (α=0) |
| `envs/bandit.py` | Multi-armed bandit with feedback modes |
| `envs/grid_world.py` | Grid navigation with regret scenarios |
| `evaluation/metrics.py` | Sample efficiency, adaptation speed |

## References

- Zhang, S., Maddox, W. T., & Glass, B. D. (2015). *Reinforcement Learning and Counterfactual Reasoning*. Topics in Cognitive Science.
- Kishida, K. T., et al. (2016). *Subsecond dopamine fluctuations in human striatum encode superposed error signals about actual and counterfactual reward*. PNAS.
- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
