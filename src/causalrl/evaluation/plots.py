"""Visualization tools for experiment results.

Generates publication-quality plots for:
- Learning curves (standard vs. counterfactual agents)
- Adaptation response curves (pre/post environment shift)
- Ablation comparison charts
- Regret/relief signal distributions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causalrl.evaluation.metrics import ExperimentMetrics


def set_style() -> None:
    """Set publication-quality matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
    })


def plot_learning_curves(
    results: dict[str, ExperimentMetrics],
    window: int = 50,
    title: str = "Learning Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot smoothed learning curves for multiple agents.

    Parameters
    ----------
    results : dict[str, ExperimentMetrics]
        Mapping from agent name to its metrics.
    window : int
        Smoothing window size.
    title : str
        Plot title.
    save_path : str | Path | None
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    set_style()
    fig, ax = plt.subplots()

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(results)))

    for (name, metrics), color in zip(results.items(), colors):
        rewards = metrics.rewards
        if len(rewards) < window:
            ax.plot(rewards, label=name, color=color)
        else:
            smoothed = np.convolve(
                rewards, np.ones(window) / window, mode="valid"
            )
            episodes = np.arange(window - 1, len(rewards))
            ax.plot(episodes, smoothed, label=name, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_adaptation(
    results: dict[str, ExperimentMetrics],
    shift_episode: int,
    window: int = 50,
    context_episodes: int = 200,
    title: str = "Adaptation After Environmental Shift",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot adaptation response around an environmental shift.

    Parameters
    ----------
    results : dict[str, ExperimentMetrics]
        Agent metrics.
    shift_episode : int
        Episode where the environment shifted.
    context_episodes : int
        Number of episodes before/after shift to show.
    """
    set_style()
    fig, ax = plt.subplots()

    start = max(0, shift_episode - context_episodes)
    end = shift_episode + context_episodes
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(results)))

    for (name, metrics), color in zip(results.items(), colors):
        rewards = metrics.rewards[start:end]
        if len(rewards) < window:
            continue
        smoothed = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        episodes = np.arange(start + window - 1, start + len(rewards))
        ax.plot(episodes, smoothed, label=name, color=color, linewidth=2)

    ax.axvline(shift_episode, color="red", linestyle="--", alpha=0.7, label="Env shift")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_ablation_comparison(
    results: dict[str, ExperimentMetrics],
    optimal_reward: float,
    title: str = "Ablation Study: Sample Efficiency",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing sample efficiency across ablation variants.

    Parameters
    ----------
    results : dict[str, ExperimentMetrics]
        Agent metrics (each key is an ablation variant).
    optimal_reward : float
        Theoretical optimal reward for efficiency computation.
    """
    set_style()
    fig, ax = plt.subplots()

    names = []
    efficiencies = []

    for name, metrics in results.items():
        eff = metrics.sample_efficiency(optimal_reward)
        names.append(name)
        efficiencies.append(eff if eff is not None else len(metrics.episodes))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, efficiencies, color=colors)

    ax.set_ylabel("Episodes to 90% Optimal")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")

    # Add value labels
    for bar, val in zip(bars, efficiencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
