"""Small declarative scene loader for Example 65 continuous-transfer benchmarks."""

from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from primitive_manifold_planner.families.standard import SphereFamily

from ..family_definition import ContinuousMaskedPlaneFamily, build_bounds, build_family_obstacles
from ..support import sphere_point


@dataclass(frozen=True)
class ContinuousTransferScene:
    """Materialized scene objects built from a declarative Example 65 description."""

    description: dict[str, Any]
    left_support: SphereFamily
    transfer_family: ContinuousMaskedPlaneFamily
    right_support: SphereFamily
    start_q: np.ndarray
    goal_q: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


def parse_scene_description(scene_description: dict[str, Any] | str) -> dict[str, Any]:
    """Parse a plain dict or a small JSON/Python-literal string scene description."""

    if isinstance(scene_description, dict):
        return copy.deepcopy(scene_description)
    text = str(scene_description).strip()
    if len(text) == 0:
        raise ValueError("Scene description string must not be empty.")
    if text[0] in "{[":
        return json.loads(text)
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed scene description must be a dictionary.")
    return copy.deepcopy(parsed)


def default_example_65_scene_description(obstacle_profile: str = "none") -> dict[str, Any]:
    """Return the current Example 65 sphere-family-sphere scene as declarative data."""

    profile_name = "none" if obstacle_profile is None else str(obstacle_profile)
    return {
        "scene_name": "example_65_continuous_transfer_family",
        "left_support": {
            "family": "sphere",
            "name": "left_support_3d",
            "center": [-2.15, -0.55, 0.48],
            "radius": 1.05,
        },
        "transfer_family": {
            "family": "continuous_masked_plane",
            "name": "transfer_foliation_3d",
            "base_point": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "lambda_min": -0.55,
            "lambda_max": 0.55,
            "plane_offsets": [round(float(value), 2) for value in np.linspace(-0.54, 0.54, 10)],
            "half_u": 0.95,
            "half_v": 2.15,
            "nominal_lambda": 0.0,
            "obstacle_profile": profile_name,
        },
        "right_support": {
            "family": "sphere",
            "name": "right_support_3d",
            "center": [2.15, 0.55, 0.48],
            "radius": 1.05,
        },
        "start": {
            "type": "sphere_point",
            "support": "left_support",
            "radius": 1.05,
            "azimuth_deg": 0.0,
            "elevation_deg": -90.0,
        },
        "goal": {
            "type": "sphere_point",
            "support": "right_support",
            "radius": 1.05,
            "azimuth_deg": 0.0,
            "elevation_deg": 90.0,
        },
        "bounds": {
            "family_count": 3,
        },
    }


def build_continuous_transfer_scene(
    scene_description: dict[str, Any] | str,
) -> ContinuousTransferScene:
    """Build the current continuous-transfer scene objects from declarative data."""

    description = parse_scene_description(scene_description)
    left_spec = dict(description["left_support"])
    transfer_spec = dict(description["transfer_family"])
    right_spec = dict(description["right_support"])
    start_spec = dict(description["start"])
    goal_spec = dict(description["goal"])
    bounds_spec = dict(description.get("bounds", {}))

    left_radius = float(left_spec["radius"])
    right_radius = float(right_spec["radius"])

    left_support = SphereFamily(
        name=str(left_spec["name"]),
        center=np.asarray(left_spec["center"], dtype=float),
        radii={left_radius: left_radius},
    )
    transfer_family = ContinuousMaskedPlaneFamily(
        name=str(transfer_spec["name"]),
        base_point=np.asarray(transfer_spec["base_point"], dtype=float),
        normal=np.asarray(transfer_spec["normal"], dtype=float),
        lambda_min=float(transfer_spec["lambda_min"]),
        lambda_max=float(transfer_spec["lambda_max"]),
        half_u=float(transfer_spec["half_u"]),
        half_v=float(transfer_spec["half_v"]),
        nominal_lambda=float(transfer_spec.get("nominal_lambda", 0.0)),
        obstacles=build_family_obstacles(
            str(transfer_spec.get("obstacle_profile", "none")),
            half_u=float(transfer_spec["half_u"]),
            half_v=float(transfer_spec["half_v"]),
        ),
    )
    right_support = SphereFamily(
        name=str(right_spec["name"]),
        center=np.asarray(right_spec["center"], dtype=float),
        radii={right_radius: right_radius},
    )

    start_q = _resolve_sphere_point(start_spec, left_support=left_support, right_support=right_support)
    goal_q = _resolve_sphere_point(goal_spec, left_support=left_support, right_support=right_support)
    bounds_min, bounds_max = build_bounds(int(bounds_spec.get("family_count", 3)))

    return ContinuousTransferScene(
        description=description,
        left_support=left_support,
        transfer_family=transfer_family,
        right_support=right_support,
        start_q=np.asarray(start_q, dtype=float),
        goal_q=np.asarray(goal_q, dtype=float),
        bounds_min=np.asarray(bounds_min, dtype=float),
        bounds_max=np.asarray(bounds_max, dtype=float),
    )


def _resolve_sphere_point(
    point_spec: dict[str, Any],
    *,
    left_support: SphereFamily,
    right_support: SphereFamily,
) -> np.ndarray:
    if str(point_spec.get("type")) != "sphere_point":
        raise ValueError("Only sphere_point start/goal specifications are supported in this loader.")
    support_name = str(point_spec.get("support"))
    if support_name == "left_support":
        family = left_support
    elif support_name == "right_support":
        family = right_support
    else:
        raise ValueError(f"Unknown support reference '{support_name}' in scene description.")
    return sphere_point(
        family.center,
        float(point_spec["radius"]),
        azimuth_deg=float(point_spec["azimuth_deg"]),
        elevation_deg=float(point_spec["elevation_deg"]),
    )
