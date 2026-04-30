import numpy as np
import matplotlib.pyplot as plt

from primitive_manifold_planner.planning.transitions import TransitionCandidate
from primitive_manifold_planner.visualization import plot_transition_candidates_2d


def test_plot_transition_candidates_2d_accepts_valid_candidates():
    fig, ax = plt.subplots()

    candidates = [
        TransitionCandidate(
            point=np.array([1.0, 1.0]),
            residual_norm=0.0,
            score=1.0,
            seed_used=None,
        ),
        TransitionCandidate(
            point=np.array([2.0, -1.0]),
            residual_norm=0.0,
            score=2.0,
            seed_used=None,
        ),
    ]

    plot_transition_candidates_2d(
        ax=ax,
        candidates=candidates,
        selected_point=np.array([1.0, 1.0]),
        annotate=False,
    )

    plt.close(fig)