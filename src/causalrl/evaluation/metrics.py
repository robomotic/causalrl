"""Evaluation metrics for counterfactual RL experiments.

Implements the quantitative metrics from the proposal:
1. Sample Efficiency — episodes to reach 90% optimal performance
2. Adaptation Speed — episodes to recover after environmental shift
3. Regret Sensitivity — value modulation by opportunity cost
4. World Model Quality — counterfactual vs. forward prediction accuracy
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeRecord:
    """Record of a single episode."""

    episode: int
    total_reward: float
    steps: int
    optimal_actions: int = 0
    total_actions: int = 0
    mean_delta_actual: float = 0.0
    mean_delta_cf: float = 0.0
    mean_alpha: float = 0.0
    regret_count: int = 0
    relief_count: int = 0


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for a full experiment run."""

    agent_name: str
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def add_episode(self, record: EpisodeRecord) -> None:
        """Add an episode record."""
        self.episodes.append(record)

    @property
    def rewards(self) -> np.ndarray:
        """Array of total rewards per episode."""
        return np.array([e.total_reward for e in self.episodes])

    @property
    def optimal_action_rate(self) -> np.ndarray:
        """Array of optimal action rates per episode."""
        return np.array([
            e.optimal_actions / max(e.total_actions, 1) for e in self.episodes
        ])

    def sample_efficiency(
        self,
        optimal_reward: float,
        threshold: float = 0.9,
        window: int = 50,
    ) -> int | None:
        """Episodes to reach threshold fraction of optimal performance.

        Parameters
        ----------
        optimal_reward : float
            The theoretically optimal reward per episode.
        threshold : float
            Fraction of optimal to consider "learned" (default 90%).
        window : int
            Smoothing window for reward moving average.

        Returns
        -------
        int | None
            Episode number where threshold was reached, or None if never.
        """
        rewards = self.rewards
        if len(rewards) < window:
            return None

        target = threshold * optimal_reward
        smoothed = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )

        indices = np.where(smoothed >= target)[0]
        if len(indices) == 0:
            return None
        return int(indices[0]) + window

    def adaptation_speed(
        self,
        shift_episode: int,
        optimal_reward_post_shift: float,
        threshold: float = 0.9,
        window: int = 50,
    ) -> int | None:
        """Episodes after a shift to recover to threshold performance.

        Parameters
        ----------
        shift_episode : int
            Episode at which the environment changed.
        optimal_reward_post_shift : float
            Optimal reward after the environmental shift.
        threshold : float
            Recovery threshold.
        window : int
            Smoothing window.

        Returns
        -------
        int | None
            Episodes after shift to recovery, or None.
        """
        rewards = self.rewards[shift_episode:]
        if len(rewards) < window:
            return None

        target = threshold * optimal_reward_post_shift
        smoothed = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )

        indices = np.where(smoothed >= target)[0]
        if len(indices) == 0:
            return None
        return int(indices[0]) + window

    def regret_sensitivity_score(self) -> float:
        """Measure how strongly regret/relief signals modulate learning.

        Returns ratio of regret events to total events. Values near 0.5
        indicate balanced regret/relief; values near 0 or 1 indicate
        strongly biased value functions.
        """
        total_regret = sum(e.regret_count for e in self.episodes)
        total_relief = sum(e.relief_count for e in self.episodes)
        total = total_regret + total_relief
        if total == 0:
            return 0.5
        return total_regret / total

    def mean_counterfactual_weight(self) -> float:
        """Average α used across all episodes."""
        alphas = [e.mean_alpha for e in self.episodes if e.mean_alpha > 0]
        return float(np.mean(alphas)) if alphas else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "total_episodes": len(self.episodes),
            "mean_reward": float(np.mean(self.rewards)) if self.episodes else 0.0,
            "final_reward_mean": float(np.mean(self.rewards[-50:])) if len(self.episodes) >= 50 else 0.0,
            "regret_sensitivity": self.regret_sensitivity_score(),
            "mean_alpha": self.mean_counterfactual_weight(),
        }
