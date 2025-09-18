from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from beaverfe import BeaverPipeline
from beaverfe.auto_parameters.shared import evaluate_model


def evaluate_transformations(
    transformations: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    model,
    scoring: str,
    cv: Union[int, Callable] = None,
    groups: Optional[np.ndarray] = None,
    plot_file: Optional[str] = "performance_evolution.png",
) -> List[Dict]:
    """
    Evaluates a model by incrementally applying transformations and
    plots the evolution of the score.

    Parameters
    ----------
    transformations : list of dict
        List of transformations in Beaver format.
    X : np.ndarray
        Input features.
    y : np.ndarray
        Labels.
    model : estimator
        Model to evaluate.
    scoring : str
        Evaluation metric.
    cv : int or callable, optional
        Cross-validation strategy.
    groups : np.ndarray, optional
        Groups for cross-validation if applicable.
    plot_file : str, optional
        Path where the plot will be saved. Default is "performance_evolution.png".

    Returns
    -------
    scores : list of dict
        List with transformation names and corresponding scores.
    """
    scores = []
    transformations_applied = []

    # Baseline evaluation
    baseline_score = evaluate_model(X, y, model, scoring, cv, groups)
    scores.append({"name": "Baseline", "score": baseline_score})

    # Incremental evaluation
    for transformation in transformations:
        transformations_applied.append(transformation)
        pipe = BeaverPipeline(transformations_applied)
        score = evaluate_model(X, y, model, scoring, cv, groups, pipe)
        scores.append({"name": transformation["name"], "score": score})

    # Plot score evolution (enumerated labels)
    _plot_scores(scores, plot_file)

    return scores


def _plot_scores(scores: List[Dict], plot_file: Optional[str]) -> None:
    """Generates and saves the score evolution plot with enumerated labels."""
    names = [s["name"] for s in scores]
    values = [s["score"] for s in scores]

    # Enumerate names to avoid duplicates
    display_names = [f"{i} - {name}" for i, name in enumerate(names)]

    x = np.arange(len(values))

    plt.figure(figsize=(10, 6))
    plt.plot(x, values, marker="o", linestyle="-", linewidth=2)
    plt.title("Score evolution by transformation")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.xticks(x, display_names, rotation=45)

    if plot_file:
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        print(f"Plot saved to {plot_file}")

    plt.close()
