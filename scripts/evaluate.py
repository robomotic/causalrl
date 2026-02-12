"""Evaluation and plotting script for causalrl.

This script parses rewards from results directories and generates
comparison plots using the evaluation modules.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causalrl.evaluation.metrics import ExperimentMetrics, EpisodeRecord
from causalrl.evaluation.plots import plot_learning_curves

def load_metrics_from_rewards(rewards_path: Path, agent_name: str) -> ExperimentMetrics:
    """Create an ExperimentMetrics object from a rewards JSON file."""
    with open(rewards_path) as f:
        rewards = json.load(f)
    
    metrics = ExperimentMetrics(agent_name=agent_name)
    for i, r in enumerate(rewards):
        metrics.add_episode(EpisodeRecord(episode=i, total_reward=r, steps=1))
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--output", type=str, default="results/comparison.png", help="Path to save the plot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist.")
        return

    results_map = {}
    # Search recursively for rewards.json files
    for rewards_file in results_dir.rglob("rewards.json"):
        # The parent directory name is usually the experiment name
        # The grandparent directory name is often the agent type
        # Let's combine them for a unique label
        agent_type = rewards_file.parent.parent.name
        exp_name = rewards_file.parent.name
        label = f"{agent_type}_{exp_name}"
        
        results_map[label] = load_metrics_from_rewards(rewards_file, label)

    if not results_map:
        print("No valid results found. Ensure you ran train.py with full metrics saving.")
        return

    print(f"Generating plot for {len(results_map)} agents...")
    plot_learning_curves(results_map, window=50, save_path=args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
