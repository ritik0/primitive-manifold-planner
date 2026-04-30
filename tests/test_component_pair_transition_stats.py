import numpy as np

from primitive_manifold_planner.planners.transition_manager import (
    TransitionCandidate,
    TransitionManager,
)


def test_transition_manager_prefers_historically_successful_candidate_for_component_pair():
    manager = TransitionManager()
    src = ("left_table", 1.0, "0")
    dst = ("transfer_zone", 1.0, "switch")

    candidate_a = TransitionCandidate(
        src=src,
        dst=dst,
        transition_point=np.array([0.0, 0.0]),
        score=1.0,
        candidate_index=0,
        metadata={"source_component": "0", "target_component": "switch"},
    )
    candidate_b = TransitionCandidate(
        src=src,
        dst=dst,
        transition_point=np.array([0.0, 0.0]),
        score=1.0,
        candidate_index=1,
        metadata={"source_component": "0", "target_component": "switch"},
    )
    manager.add_candidates([candidate_a, candidate_b])

    manager.record_candidate_attempt(src, dst, candidate_index=1, success=True)
    manager.record_candidate_attempt(src, dst, candidate_index=1, success=True)
    manager.record_candidate_attempt(src, dst, candidate_index=0, success=False)

    ranked = manager.rank_candidates(
        src=src,
        dst=dst,
        current_x=np.array([0.0, 0.0]),
    )

    assert ranked[0].candidate_index == 1


def test_transition_manager_tracks_attempt_stats_per_component_pair():
    manager = TransitionManager()
    src = ("left_table", 1.0, "0")
    dst = ("transfer_zone", 1.0, "switch")

    manager.record_candidate_attempt(src, dst, candidate_index=2, success=True)
    manager.record_candidate_attempt(src, dst, candidate_index=2, success=False)

    stats = manager.get_candidate_attempt_stats(src, dst, candidate_index=2)

    assert stats.successes == 1
    assert stats.failures == 1
