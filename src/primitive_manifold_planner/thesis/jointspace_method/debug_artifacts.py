from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


@dataclass(frozen=True)
class CspaceDebugArtifact:
    """Serializable debug snapshot for one certified C-space route.

    ``dense_theta_path`` is the source of truth. ``dense_fk_trace`` should be
    FK(dense_theta_path), used only for visualization/debugging.
    """

    output_dir: Path
    dense_theta_path: np.ndarray
    dense_fk_trace: np.ndarray
    stage_labels: list[str]
    constraint_residuals: np.ndarray
    joint_steps: np.ndarray
    lambda_labels: np.ndarray | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _write_plots(base: Path, artifact: CspaceDebugArtifact) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (base / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return

    theta_path = np.asarray(artifact.dense_theta_path, dtype=float)
    residuals = np.asarray(artifact.constraint_residuals, dtype=float)
    labels = list(artifact.stage_labels)
    if len(theta_path) > 0:
        # Plot theta components directly to inspect the certified dense joint route.
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(theta_path[:, 0], label="theta0")
        ax.plot(theta_path[:, 1], label="theta1")
        ax.plot(theta_path[:, 2], label="theta2")
        ax.set_xlabel("waypoint index")
        ax.set_ylabel("theta [rad]")
        ax.set_title("Dense theta trajectory")
        ax.legend()
        fig.tight_layout()
        fig.savefig(base / "theta_vs_index.png", dpi=160)
        plt.close(fig)

        fig = plt.figure(figsize=(6.5, 5.8))
        ax3 = fig.add_subplot(111, projection="3d")
        ax3.plot(theta_path[:, 0], theta_path[:, 1], theta_path[:, 2], color="#111827", linewidth=1.5)
        colors = {"left": "#f97316", "plane": "#2563eb", "family": "#2563eb", "right": "#16a34a"}
        for stage in dict.fromkeys(labels):
            indices = [idx for idx, label in enumerate(labels) if str(label) == str(stage)]
            if len(indices) >= 2:
                segment = theta_path[np.asarray(indices, dtype=int)]
                # Stage-colored segments show which active manifold labels the route uses.
                ax3.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=colors.get(str(stage), "#dc2626"), linewidth=2.5, label=f"{stage} segment")
        ax3.scatter(theta_path[0, 0], theta_path[0, 1], theta_path[0, 2], color="black", s=42, label="start")
        ax3.scatter(theta_path[-1, 0], theta_path[-1, 1], theta_path[-1, 2], color="gold", s=52, label="goal")
        for key, color, label in (
            ("selected_left_plane_transition_index", "#ef4444", "left-plane transition"),
            ("selected_plane_right_transition_index", "#14b8a6", "plane-right transition"),
            ("selected_left_family_transition_index", "#ef4444", "left-family transition"),
            ("selected_family_right_transition_index", "#14b8a6", "family-right transition"),
        ):
            idx = int(artifact.summary.get(key, -1))
            if 0 <= idx < len(theta_path):
                # Transition markers identify shared configurations between stages.
                ax3.scatter(theta_path[idx, 0], theta_path[idx, 1], theta_path[idx, 2], color=color, s=60, label=label)
        ax3.set_xlabel("theta0")
        ax3.set_ylabel("theta1")
        ax3.set_zlabel("theta2")
        ax3.set_title("C-space dense path")
        ax3.legend()
        fig.tight_layout()
        fig.savefig(base / "cspace_3d_path.png", dpi=160)
        plt.close(fig)

    if len(residuals) > 0:
        # Residual plot checks active-manifold error along the whole route.
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(residuals, color="#455a64")
        ax.set_xlabel("waypoint index")
        ax.set_ylabel("active constraint residual")
        ax.set_title("Constraint residual along dense theta path")
        fig.tight_layout()
        fig.savefig(base / "residual_vs_index.png", dpi=160)
        plt.close(fig)


def save_cspace_debug_artifacts(artifact: CspaceDebugArtifact) -> Path:
    """Save generic C-space debug artifacts for a certified theta route.

    The saved arrays preserve the dense theta path, its FK trace, per-waypoint
    residuals, joint steps, stage labels, optional lambda labels, and plots.
    """

    base = Path(artifact.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    theta_path = np.asarray(artifact.dense_theta_path, dtype=float)
    fk_trace = np.asarray(artifact.dense_fk_trace, dtype=float)
    residuals = np.asarray(artifact.constraint_residuals, dtype=float)
    joint_steps = np.asarray(artifact.joint_steps, dtype=float)
    labels = list(artifact.stage_labels)

    # Dense theta route is the executable source; FK trace is derived from it.
    np.save(base / "dense_theta_path.npy", theta_path)
    np.save(base / "dense_fk_trace.npy", fk_trace)
    # Residuals and joint steps are the lightweight certification audit trail.
    np.save(base / "constraint_residuals.npy", residuals)
    np.save(base / "joint_steps.npy", joint_steps)
    (base / "dense_stage_labels.txt").write_text("\n".join(labels), encoding="utf-8")
    (base / "dense_stage_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    if artifact.lambda_labels is not None:
        lambda_labels = np.asarray(artifact.lambda_labels, dtype=float)
        np.save(base / "dense_lambda_labels.npy", lambda_labels)
        lambda_json = [None if not np.isfinite(float(value)) else float(value) for value in lambda_labels]
        (base / "dense_lambda_labels.json").write_text(json.dumps(lambda_json, indent=2), encoding="utf-8")

    summary = {"created_at": datetime.now().isoformat(timespec="seconds"), **dict(artifact.summary)}
    (base / "cspace_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _write_plots(base, artifact)
    return base
