#!/usr/bin/env python3
"""
Convert motion .npz files into per-clip JSON files and update a motion index.

Example:
  python3 scripts/add_motion_clips.py \\
    --policy public/examples/checkpoints/g1/tracking_policy.json \\
    --index public/examples/checkpoints/g1/motions.json \\
    /path/to/05_13_stageii.npz /path/to/55_01_stageii.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


INDEX_FORMAT = "tracking-motion-index-v1"
DEFAULT_SUFFIX = "_stageii"
DEFAULT_MAX_FRAMES = 120 * 50

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


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    return cleaned or "motion"


def derive_motion_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith(DEFAULT_SUFFIX):
        stem = stem[: -len(DEFAULT_SUFFIX)]
    return sanitize_name(stem)


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


def load_policy_dataset_joint_names(policy_path: Path) -> List[str]:
    config = json.loads(policy_path.read_text())
    tracking = config.get("tracking") or {}
    names = tracking.get("dataset_joint_names") or config.get("policy_joint_names")
    if not names:
        raise ValueError("Policy config missing tracking.dataset_joint_names or policy_joint_names")
    return list(names)


def to_clip(npz_path: Path, dataset_joint_names: List[str], max_frames: int) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        joint_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)

        if joint_pos.shape[0] > max_frames:
            joint_pos = joint_pos[:max_frames]
            root_pos = root_pos[:max_frames]
            root_rot_xyzw = root_rot_xyzw[:max_frames]

        joint_names = data.get("joint_names", None)
        if joint_names is not None:
            src_names = joint_names.tolist()
            target_names = list(dataset_joint_names)
            if src_names != target_names:
                name_to_idx = {n: i for i, n in enumerate(src_names)}
                remap = np.zeros((joint_pos.shape[0], len(target_names)), dtype=np.float32)
                for i, n in enumerate(target_names):
                    j = name_to_idx.get(n)
                    if j is not None:
                        remap[:, i] = joint_pos[:, j]
                joint_pos = remap
        else:
            joint_pos = mapping_joints(joint_pos, dataset_joint_names)

        if root_pos.ndim == 3:
            root_pos = root_pos[:, 0, :]

        root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

        return {
            "joint_pos": joint_pos.tolist(),
            "root_quat": root_quat.tolist(),
            "root_pos": root_pos.tolist()
        }


def load_or_init_index(index_path: Path) -> Tuple[dict, List[dict]]:
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get("format") != INDEX_FORMAT:
            raise ValueError(f"Unsupported index format in {index_path}")
        motions = index.get("motions", [])
        if not isinstance(motions, list):
            raise ValueError(f"Index motions should be a list in {index_path}")
        return index, motions
    return {"format": INDEX_FORMAT, "base_path": "./motions", "motions": []}, []


def resolve_motions_dir(index_path: Path, index: dict, override: Path | None) -> Path:
    if override:
        return override
    base_path = index.get("base_path") or "./motions"
    if isinstance(base_path, str) and (base_path.startswith("http://") or base_path.startswith("https://")):
        raise ValueError("Index base_path is a URL. Provide --motions-dir for output.")
    return (index_path.parent / Path(base_path)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .npz motions to JSON clips and update an index.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("public/examples/checkpoints/g1/tracking_policy.json"),
        help="Policy JSON path used to read dataset_joint_names."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("public/examples/checkpoints/g1/motions.json"),
        help="Motion index JSON path."
    )
    parser.add_argument(
        "--motions-dir",
        type=Path,
        default=None,
        help="Output directory for per-motion JSON files."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum frames per clip."
    )
    parser.add_argument(
        "motions",
        nargs="+",
        help="One or more .npz motion paths."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_joint_names = load_policy_dataset_joint_names(args.policy)
    index, motions = load_or_init_index(args.index)
    motions_dir = resolve_motions_dir(args.index, index, args.motions_dir)
    motions_dir.mkdir(parents=True, exist_ok=True)

    existing = {entry.get("name") for entry in motions if isinstance(entry, dict)}
    added = 0
    skipped = 0

    for motion_path in args.motions:
        path = Path(motion_path)
        if not path.exists():
            raise FileNotFoundError(path)
        name = derive_motion_name(path)
        if name in existing:
            skipped += 1
            continue
        clip = to_clip(path, dataset_joint_names, args.max_frames)
        out_path = motions_dir / f"{name}.json"
        out_path.write_text(json.dumps(clip, ensure_ascii=False, indent=2))
        motions.append({"name": name, "file": f"{name}.json"})
        existing.add(name)
        added += 1

    index["motions"] = motions
    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"Added {added} motion(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
