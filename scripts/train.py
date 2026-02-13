"""Training script for counterfactual RL experiments.

Usage:
    python scripts/train.py --config configs/bandit_comparison.yaml
    python scripts/train.py --env bandit --agent counterfactual --episodes 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import trange

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causalrl.agents.standard_rl import StandardRLAgent
from causalrl.agents.counterfactual_rl import CounterfactualRLAgent
from causalrl.agents.ablations import (
    create_no_cf_agent,
    create_fixed_alpha_agent,
    create_adaptive_alpha_agent,
)
from causalrl.core.ofc import OFCComparator
from causalrl.envs.bandit import CounterfactualBanditEnv
from causalrl.envs.grid_world import RegretGridWorldEnv
from causalrl.evaluation.metrics import EpisodeRecord, ExperimentMetrics
from causalrl.utils.config import ExperimentConfig
from causalrl.world_models.tabular import TabularWorldModel
from causalrl.world_models.oracle import OracleWorldModel


def create_env(config: ExperimentConfig):
    """Create environment from config."""
    if config.env.type == "bandit":
        return CounterfactualBanditEnv(
            num_arms=config.env.num_arms,
            feedback_mode=config.env.feedback_mode,
            reward_drift_rate=config.env.reward_drift_rate,
            reward_std=config.env.reward_std,
            reward_shift_interval=config.env.reward_shift_interval,
            seed=config.seed,
        )
    elif config.env.type == "grid_world":
        return RegretGridWorldEnv(
            grid_size=config.env.grid_size,
            num_goals=config.env.num_goals,
            stochasticity=config.env.stochasticity,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown env type: {config.env.type}")


def create_agent(config: ExperimentConfig, env):
    """Create agent from config."""
    if config.env.type == "bandit":
        num_states = 1
        num_actions = config.env.num_arms
    else:
        num_states = config.env.grid_size ** 2
        num_actions = 4

    if config.agent.type == "standard":
        return StandardRLAgent(
            num_states=num_states,
            num_actions=num_actions,
            beta=config.agent.beta,
            epsilon=config.agent.epsilon,
            discount=config.agent.discount,
            seed=config.seed,
        )
    elif config.agent.type == "counterfactual":
        # Create world model
        wm_type = config.agent.world_model.type
        if wm_type == "tabular":
            wm = TabularWorldModel(
                num_states, 
                num_actions, 
                learning_rate=config.agent.world_model.learning_rate
            )
        elif wm_type == "oracle":
            wm = OracleWorldModel(env, num_states, num_actions)
        else:
            raise ValueError(f"Unknown world model type: {wm_type}")

        ofc = OFCComparator(
            mode=config.agent.ofc.mode,
            alpha=config.agent.ofc.alpha,
            alpha_min=config.agent.ofc.alpha_min,
            alpha_max=config.agent.ofc.alpha_max,
            uncertainty_sensitivity=config.agent.ofc.uncertainty_sensitivity,
        )

        return CounterfactualRLAgent(
            num_states=num_states,
            num_actions=num_actions,
            world_model=wm,
            ofc=ofc,
            beta=config.agent.beta,
            gamma_cf=config.agent.gamma,
            epsilon=config.agent.epsilon,
            discount=config.agent.discount,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown agent type: {config.agent.type}")


def run_bandit_episode(agent, env, episode_num: int) -> EpisodeRecord:
    """Run a single bandit episode (1 step per episode)."""
    state, info = env.reset()
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    update_info = agent.update(state, action, reward, next_state, terminated, info)

    return EpisodeRecord(
        episode=episode_num,
        total_reward=reward,
        steps=1,
        optimal_actions=int(info.get("is_optimal", False)),
        total_actions=1,
        mean_delta_actual=update_info.get("delta_actual", 0.0),
        mean_delta_cf=update_info.get("delta_counterfactual", 0.0),
        mean_alpha=update_info.get("alpha", 0.0),
        regret_count=int(update_info.get("delta_counterfactual", 0) > 0),
        relief_count=int(update_info.get("delta_counterfactual", 0) < 0),
    )


def run_gridworld_episode(agent, env, episode_num: int) -> EpisodeRecord:
    """Run a single grid-world episode."""
    state, info = env.reset()
    total_reward = 0.0
    steps = 0
    deltas_actual = []
    deltas_cf = []
    alphas = []
    regret_count = 0
    relief_count = 0

    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        update_info = agent.update(state, action, reward, next_state, terminated or truncated, info)

        total_reward += reward
        steps += 1
        deltas_actual.append(update_info.get("delta_actual", 0.0))
        deltas_cf.append(update_info.get("delta_counterfactual", 0.0))
        alphas.append(update_info.get("alpha", 0.0))
        if update_info.get("delta_counterfactual", 0) > 0:
            regret_count += 1
        elif update_info.get("delta_counterfactual", 0) < 0:
            relief_count += 1

        state = next_state
        done = terminated or truncated

    return EpisodeRecord(
        episode=episode_num,
        total_reward=total_reward,
        steps=steps,
        mean_delta_actual=float(np.mean(deltas_actual)) if deltas_actual else 0.0,
        mean_delta_cf=float(np.mean(deltas_cf)) if deltas_cf else 0.0,
        mean_alpha=float(np.mean(alphas)) if alphas else 0.0,
        regret_count=regret_count,
        relief_count=relief_count,
    )


def train(config: ExperimentConfig) -> ExperimentMetrics:
    """Run a full training experiment.

    Returns
    -------
    ExperimentMetrics
        Complete metrics for the experiment.
    """
    env = create_env(config)
    agent = create_agent(config, env)

    metrics = ExperimentMetrics(agent_name=f"{config.agent.type}_{config.name}")

    run_episode = (
        run_bandit_episode if config.env.type == "bandit" else run_gridworld_episode
    )

    for ep in trange(config.num_episodes, desc=config.name, leave=False):
        record = run_episode(agent, env, ep)
        metrics.add_episode(record)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train counterfactual RL agents")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--env", type=str, default="bandit", choices=["bandit", "grid_world"])
    parser.add_argument("--agent", type=str, default="counterfactual",
                        choices=["standard", "counterfactual"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        # Allow CLI overrides
        if args.agent != "counterfactual": # only override if user explicitly set it
             config.agent.type = args.agent
        if args.episodes != 1000:
             config.num_episodes = args.episodes
    else:
        config = ExperimentConfig(
            name=f"{args.agent}_{args.env}",
            seed=args.seed,
            num_episodes=args.episodes,
        )
        config.env.type = args.env
        config.agent.type = args.agent

    print(f"Training: {config.name}")
    print(f"  Agent: {config.agent.type}")
    print(f"  Environment: {config.env.type}")
    print(f"  Episodes: {config.num_episodes}")

    metrics = train(config)

    # Save results
    output_dir = Path(args.output) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    # Save full rewards for plotting
    rewards_path = output_dir / "rewards.json"
    with open(rewards_path, "w") as f:
        json.dump(metrics.rewards.tolist(), f)

    print(f"\nResults saved to {output_dir}")
    print(f"  Mean reward: {np.mean(metrics.rewards):.3f}")
    if len(metrics.rewards) >= 50:
        print(f"  Final reward (last 50): {np.mean(metrics.rewards[-50:]):.3f}")


if __name__ == "__main__":
    main()
