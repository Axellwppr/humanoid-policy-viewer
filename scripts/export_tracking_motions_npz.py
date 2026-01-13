#!/usr/bin/env python3
"""
Export tracking motions from .npz files into an index + per-motion JSON files.

Rules:
  - Use tracking_raw.yaml as the source of motion entries.
  - Keep only one motion per base name (e.g. aiming1_subject1 -> aiming1).
  - Cap each motion length to 120s * 50Hz = 6000 frames.
  - Each motion file contains joint_pos, root_quat (wxyz), root_pos.
"""
#   python3 /home/axell/Desktop/tmp/GentleHumanoidWeb/scripts/export_tracking_motions_npz.py \
#     --config /home/axell/Desktop/gt_sim2real/config/tracking_raw.yaml \
#     --repo-root /home/axell/Desktop/gt_sim2real \
#     --output /home/axell/Desktop/tmp/GentleHumanoidWeb/public/examples/checkpoints/g1/motions.json \
#     --motions-dir /home/axell/Desktop/tmp/GentleHumanoidWeb/public/examples/checkpoints/g1/motions

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import yaml


MAX_FRAMES = 120 * 50
INDEX_FORMAT = "tracking-motion-index-v1"

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


def base_name(name: str) -> str:
    if "_subject" in name:
        return name.split("_subject", 1)[0]
    return name


def resolve_path(base: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def load_motion_sequence(npz_path: Path,
                         start: int,
                         end: int,
                         dataset_joint_names: Iterable[str]) -> Dict[str, list]:
    data = np.load(npz_path, allow_pickle=True)

    joint_pos = slice_interval(data["dof_pos"], start, end)
    root_pos = slice_interval(data["root_pos"], start, end)
    root_rot_xyzw = slice_interval(data["root_rot"], start, end)

    if joint_pos.shape[0] > MAX_FRAMES:
        joint_pos = joint_pos[:MAX_FRAMES]
        root_pos = root_pos[:MAX_FRAMES]
        root_rot_xyzw = root_rot_xyzw[:MAX_FRAMES]

    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)

    joint_names = data.get("joint_names", None)
    if joint_names is not None:
        src_names = joint_names.tolist()
        target_names = list(dataset_joint_names)
        if src_names != target_names:
            name_to_idx = {n: i for i, n in enumerate(src_names)}
            remap = np.zeros((joint_pos.shape[0], len(target_names)), dtype=np.float32)
            for i, n in enumerate(target_names):
                j = name_to_idx.get(n, None)
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


def write_motion_file(motions_dir: Path, name: str, clip: Dict[str, list]) -> None:
    motion_path = motions_dir / f"{name}.json"
    motion_path.write_text(json.dumps(clip, ensure_ascii=False, indent=2))


def resolve_base_path(output_path: Path, motions_dir: Path) -> str:
    index_root = output_path.parent.resolve()
    motions_dir = motions_dir.resolve()
    try:
        rel = motions_dir.relative_to(index_root)
    except ValueError:
        return motions_dir.as_posix()
    rel_str = rel.as_posix()
    return f"./{rel_str}" if rel_str else "."


def export_motions(config_path: Path,
                   repo_root: Path,
                   output_path: Path,
                   motions_dir: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    dataset_joint_names = config["dataset_joint_names"]

    repo_root = repo_root.resolve()
    motions_dir.mkdir(parents=True, exist_ok=True)
    seen_base = set()
    index_entries = []
    default_present = False

    for motion in config.get("motions", []):
        name = motion["name"]
        base = base_name(name)
        if base in seen_base:
            continue
        seen_base.add(base)

        path = resolve_path(repo_root, str(motion["path"]))
        t0 = int(motion.get("start", 0))
        t1 = int(motion.get("end", -1))
        clip = load_motion_sequence(path, t0, t1, dataset_joint_names)
        write_motion_file(motions_dir, name, clip)
        index_entries.append({"name": name, "file": f"{name}.json"})
        if name == "default":
            default_present = True

    for clip in config.get("motion_clips", []):
        name = clip["name"]
        clip_data = load_motion_clip(clip, dataset_joint_names)
        write_motion_file(motions_dir, name, clip_data)
        index_entries.append({"name": name, "file": f"{name}.json"})
        if name == "default":
            default_present = True

    if not default_present:
        raise ValueError("Generated motions do not include a 'default' clip.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = resolve_base_path(output_path, motions_dir)
    index = {
        "format": INDEX_FORMAT,
        "base_path": base_path,
        "motions": index_entries
    }
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(f"Wrote motion index to {output_path} (count={len(index_entries)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tracking motions to JSON.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to tracking_raw.yaml."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repository root used to resolve relative motion paths."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the motion index."
    )
    parser.add_argument(
        "--motions-dir",
        type=Path,
        default=None,
        help="Directory for per-motion JSON files (default: output/motions)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motions_dir = args.motions_dir or (args.output.parent / "motions")
    export_motions(args.config, args.repo_root, args.output, motions_dir)


if __name__ == "__main__":
    main()
