#!/usr/bin/env python3
"""
Export tracking motion clips from the Unitree G1 reinforcement-learning workspace.

This script mirrors the logic used by ``TrackingPolicyRaw`` in
``sim2sim/real_g1/src/policy.py`` to:

  * load full-length motion trajectories from ``.npz`` bundles,
  * remap joint orders into the dataset (policy) ordering, and
  * emit a JSON structure compatible with the web demo.

Example:
    python export_tracking_motions.py \\
        --config /Users/axell/MiniLocal/soft-contact/sim2sim/real_g1/config/tracking_29.yaml \\
        --repo-root /Users/axell/MiniLocal/soft-contact/sim2sim/real_g1 \\
        --output /Users/axell/MiniLocal/GentleHumanoidWeb/public/examples/checkpoints/g1/motions.json

Dependencies:
    - numpy
    - pyyaml

SciPy is *not* required; the quaternion helpers are implemented with NumPy only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml

JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

JOINT_NAMES_23 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint"
]


def mapping_joints(data: np.ndarray, target: Iterable[str]) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    target = list(target)
    cols = data.shape[1]
    if cols == len(target):
        return data.astype(np.float32, copy=True)
    if cols == len(JOINT_NAMES_29):
        source = JOINT_NAMES_29
    elif cols == len(JOINT_NAMES_23):
        source = JOINT_NAMES_23
    else:
        raise ValueError(f"Unsupported joint dimension {cols}")
    index = {name: i for i, name in enumerate(source)}
    remapped = np.zeros((data.shape[0], len(target)), dtype=np.float32)
    for j, name in enumerate(target):
        i = index.get(name)
        if i is not None:
            remapped[:, j] = data[:, i]
    return remapped


def slice_interval(array: np.ndarray, start: int, end: int) -> np.ndarray:
    stop = None if end is None or end == -1 else end
    return array[start:stop]


def load_motion_sequence(base: Path,
                         entry: Dict[str, object],
                         dataset_joint_names: Iterable[str]) -> Dict[str, list]:
    npz_path = (base / str(entry["path"])).resolve()
    data = np.load(npz_path, allow_pickle=True)

    start = int(entry.get("start", 0))
    end = entry.get("end", None)
    joint_pos = slice_interval(data["dof_pos"], start, end)
    root_pos = slice_interval(data["root_pos"], start, end)
    root_rot_xyzw = slice_interval(data["root_rot"], start, end)
    root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

    joint_pos = mapping_joints(joint_pos, dataset_joint_names)
    root_pos = np.asarray(root_pos, dtype=np.float32)
    if root_pos.ndim == 3:
        root_pos = root_pos[:, 0, :]
    root_quat = np.asarray(root_quat, dtype=np.float32)

    return {
        "joint_pos": joint_pos.tolist(),
        "root_quat": root_quat.tolist(),
        "root_pos": root_pos.tolist()
    }


def load_motion_clip(entry: Dict[str, object],
                     dataset_joint_names: Iterable[str]) -> Dict[str, list]:
    joint_pos = np.asarray(entry["joint_pos"], dtype=np.float32).reshape(1, -1)
    joint_pos = mapping_joints(joint_pos, dataset_joint_names)
    root_quat = np.asarray(entry["root_quat"], dtype=np.float32).reshape(1, 4)
    root_pos = np.asarray(entry["root_pos"], dtype=np.float32).reshape(1, 3)

    return {
        "joint_pos": joint_pos.tolist(),
        "root_quat": root_quat.tolist(),
        "root_pos": root_pos.tolist()
    }


def export_motions(config_path: Path, repo_root: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    dataset_joint_names = config["dataset_joint_names"]

    repo_root = repo_root.resolve()
    motions: Dict[str, Dict[str, list]] = {}

    for motion in config.get("motions", []):
        name = motion["name"]
        motions[name] = load_motion_sequence(repo_root, motion, dataset_joint_names)

    for clip in config.get("motion_clips", []):
        name = clip["name"]
        motions[name] = load_motion_clip(clip, dataset_joint_names)

    if "default" not in motions:
        raise ValueError("Generated motions do not include a 'default' clip.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(motions, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tracking motions to JSON.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to tracking_X.yaml (e.g. tracking_29.yaml)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root used to resolve relative asset paths. Defaults to config.parent.parent."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the exported motions."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    repo_root = (args.repo_root or config_path.parent.parent).resolve()
    output_path = args.output.resolve()

    export_motions(config_path, repo_root, output_path)
    print(f"Wrote motions to {output_path}")


if __name__ == "__main__":
    main()
