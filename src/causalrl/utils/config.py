"""Configuration management for experiments."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class WorldModelConfig:
    """Configuration for the world model component."""

    type: str = "tabular"  # "tabular", "neural", "oracle"
    learning_rate: float = 0.1
    # Neural world model specific
    hidden_size: int = 128
    num_layers: int = 2


@dataclass
class OFCConfig:
    """Configuration for the OFC comparator (α controller)."""

    mode: str = "fixed"  # "fixed" or "adaptive"
    alpha: float = 0.5  # counterfactual weight (fixed mode)
    alpha_min: float = 0.0  # min α (adaptive mode)
    alpha_max: float = 1.0  # max α (adaptive mode)
    uncertainty_sensitivity: float = 1.0  # how strongly uncertainty modulates α


@dataclass
class AgentConfig:
    """Configuration for the counterfactual RL agent."""

    type: str = "counterfactual"  # "standard", "counterfactual", "ablation"
    beta: float = 0.1  # learning rate for chosen option (β)
    gamma: float = 0.05  # learning rate for unchosen options (γ)
    epsilon: float = 0.1  # exploration rate (ε-greedy)
    discount: float = 0.99  # discount factor
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    ofc: OFCConfig = field(default_factory=OFCConfig)


@dataclass
class EnvConfig:
    """Configuration for the environment."""

    type: str = "bandit"  # "bandit", "grid_world"
    # Bandit specific
    num_arms: int = 10
    feedback_mode: str = "partial"  # "partial" or "full"
    reward_drift_rate: float = 0.0  # 0.0 = stationary
    # Grid-world specific
    grid_size: int = 5
    num_goals: int = 3
    stochasticity: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    seed: int = 42
    num_episodes: int = 1000
    eval_interval: int = 100
    log_dir: str = "results"
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Recursively construct config from a dictionary."""
        if "agent" in data:
            agent_data = data["agent"]
            if "world_model" in agent_data:
                agent_data["world_model"] = WorldModelConfig(**agent_data["world_model"])
            if "ofc" in agent_data:
                agent_data["ofc"] = OFCConfig(**agent_data["ofc"])
            data["agent"] = AgentConfig(**agent_data)
        if "env" in data:
            data["env"] = EnvConfig(**data["env"])
        return cls(**data)
