"""Parameter sweep for Counterfactual RL on Multi-Armed Bandits.

This script runs experiments over a range of alpha and gamma_cf values
to visualize their impact on learning performance.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import subprocess
import json
from tqdm import tqdm

def run_experiment(alpha, gamma_cf, episodes=2000, seed=42):
    name = f"alpha_{alpha}_gamma_{gamma_cf}"
    output_dir = Path("results/sweep") / name
    
    cmd = [
        "python3", "scripts/train.py",
        "--env", "bandit",
        "--agent", "counterfactual",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output", "results/sweep"
    ]
    
    # We need to pass these specific params. 
    # Current train.py doesn't have CLI flags for alpha/gamma directly 
    # but we can use a temporary config or I'll just add them to train.py.
    # Actually, easiest is to write a temporary yaml.
    
    config = {
        "name": name,
        "seed": seed,
        "num_episodes": episodes,
        "env": {
            "type": "bandit",
            "num_arms": 10,
            "feedback_mode": "full"
        },
        "agent": {
            "type": "counterfactual",
            "beta": 0.1,
            "gamma": gamma_cf,
            "epsilon": 0.1,
            "world_model": {"type": "tabular"},
            "ofc": {
                "mode": "fixed",
                "alpha": alpha
            }
        }
    }
    
    config_path = output_dir / "config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    subprocess.run(["python3", "scripts/train.py", "--config", str(config_path), "--output", "results/sweep"], check=True)

def main():
    alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
    gammas = [0.01, 0.05, 0.1]
    
    print("Starting parameter sweep...")
    for a in alphas:
        for g in gammas:
            print(f"Running alpha={a}, gamma={g}")
            run_experiment(a, g)
            
    print("Sweep complete. Run python3 scripts/evaluate.py --results_dir results/sweep to see results.")

if __name__ == "__main__":
    main()
