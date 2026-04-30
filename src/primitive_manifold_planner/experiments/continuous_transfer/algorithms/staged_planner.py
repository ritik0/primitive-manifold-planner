"""Thin staged-planner shell that delegates Example 65 work to existing stage modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.stage_graph import StageGraph


@dataclass(frozen=True)
class StagedPlannerDelegates:
    """Stage-specific callables used by the lightweight staged shell."""

    run_left_stage: Callable[[], object]
    build_left_stage_failure: Callable[[], object]
    run_middle_stage: Callable[[object], object]
    bridge_middle_support: Callable[[object], int]
    select_right_stage_inputs: Callable[[object], object]
    run_right_stage: Callable[[object, object], object]
    finalize_route: Callable[[object, int, object, object], object]


@dataclass
class StagedPlannerShell:
    """Minimal orchestrator around a fixed staged-planning sequence."""

    stage_graph: StageGraph
    delegates: StagedPlannerDelegates

    def run(self):
        self._validate_stage_graph()
        left_result = self.delegates.run_left_stage()
        if len(left_result) == 0:
            return self.delegates.build_left_stage_failure()

        middle_result = self.delegates.run_middle_stage(left_result)
        bridge_rounds = int(self.delegates.bridge_middle_support(middle_result))
        right_stage_inputs = self.delegates.select_right_stage_inputs(middle_result)
        right_result = self.delegates.run_right_stage(middle_result, right_stage_inputs)
        return self.delegates.finalize_route(middle_result, bridge_rounds, right_stage_inputs, right_result)

    def _validate_stage_graph(self) -> None:
        stage_ids = self.stage_graph.stage_ids()
        if stage_ids != ["left", "family", "right"]:
            raise ValueError("Example 65 staged shell expects stages ordered as left -> family -> right.")
        if not self.stage_graph.has_transition("left", "family"):
            raise ValueError("Example 65 staged shell requires a left -> family transition.")
        if not self.stage_graph.has_transition("family", "right"):
            raise ValueError("Example 65 staged shell requires a family -> right transition.")
