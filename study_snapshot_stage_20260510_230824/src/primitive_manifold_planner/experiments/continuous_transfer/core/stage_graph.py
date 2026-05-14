"""Lightweight staged-planning graph structures for Example 65 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageNode:
    """One stage in a fixed staged-planning sequence."""

    stage_id: str
    label: str
    stage_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionConstraint:
    """A lightweight directed transition between two stages."""

    source_stage_id: str
    target_stage_id: str
    transition_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageGraph:
    """Minimal fixed-stage graph used by the staged planner shell."""

    nodes: list[StageNode]
    transitions: list[TransitionConstraint]

    def stage_ids(self) -> list[str]:
        return [str(node.stage_id) for node in self.nodes]

    def outgoing(self, stage_id: str) -> list[TransitionConstraint]:
        return [
            transition
            for transition in self.transitions
            if str(transition.source_stage_id) == str(stage_id)
        ]

    def incoming(self, stage_id: str) -> list[TransitionConstraint]:
        return [
            transition
            for transition in self.transitions
            if str(transition.target_stage_id) == str(stage_id)
        ]

    def has_transition(self, source_stage_id: str, target_stage_id: str) -> bool:
        return any(
            str(transition.source_stage_id) == str(source_stage_id)
            and str(transition.target_stage_id) == str(target_stage_id)
            for transition in self.transitions
        )
