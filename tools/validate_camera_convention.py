#!/usr/bin/env python3
"""
Validate metadata camera conventions for LVSM datasets.

This script performs:
1. Numeric integrity checks (rotation validity, determinant, translation scale, intrinsics drift)
2. Candidate convention scoring (w2c/c2w assumptions, axis flips, translation scaling)
3. Visualization export (top-down camera trajectory with forward arrows)

Usage:
  python tools/validate_camera_convention.py \
      --repo-root . \
      --official-dir preprocessed_data/test \
      --custom-dir preprocessed_data/test_waymo \
      --out-dir experiments/camera_validation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import matplotlib
import plotly.graph_objects as go
import plotly.io as pio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SceneData:
    scene_name: str
    metadata_path: Path
    image_size_wh: Tuple[int, int]
    intrinsics: np.ndarray  # (N, 4) as fx, fy, cx, cy in pixels
    w2c: np.ndarray  # (N, 4, 4)
    image_paths: List[Path]


def resolve_image_path(image_rel: Path, repo_root: Path, dataset_dir: Path) -> Path:
    image_abs = (repo_root / image_rel).resolve()
    if image_abs.exists():
        return image_abs

    # Fallback: if metadata was copied from another split, repair split component.
    # Example: preprocessed_data/test/images/... while running on test_waymo.
    rel_parts = list(image_rel.parts)
    if len(rel_parts) >= 3 and rel_parts[0] == "preprocessed_data" and rel_parts[2] == "images":
        repaired_rel = Path("preprocessed_data") / dataset_dir.name / "images" / Path(*rel_parts[3:])
        repaired_abs = (repo_root / repaired_rel).resolve()
        if repaired_abs.exists():
            return repaired_abs
        raise FileNotFoundError(
            f"Image path not found from metadata: {image_abs}; fallback also missing: {repaired_abs}"
        )

    raise FileNotFoundError(f"Image path not found from metadata: {image_abs}")


def load_scene(metadata_path: Path, repo_root: Path, dataset_dir: Path) -> SceneData:
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload["frames"]
    if not frames:
        raise ValueError(f"No frames in {metadata_path}")

    intrinsics = np.array([fr["fxfycxcy"] for fr in frames], dtype=np.float64)
    w2c = np.array([fr["w2c"] for fr in frames], dtype=np.float64)
    image_paths = []
    for fr in frames:
        image_paths.append(resolve_image_path(Path(fr["image_path"]), repo_root, dataset_dir))

    with Image.open(image_paths[0]) as im:
        w, h = im.size

    return SceneData(
        scene_name=payload.get("scene_name", metadata_path.stem),
        metadata_path=metadata_path,
        image_size_wh=(w, h),
        intrinsics=intrinsics,
        w2c=w2c,
        image_paths=image_paths,
    )


def rotation_metrics(rot: np.ndarray) -> Dict[str, float]:
    eye = np.eye(3, dtype=np.float64)
    ortho_err = np.linalg.norm(rot @ np.transpose(rot, (0, 2, 1)) - eye[None], axis=(1, 2))
    det = np.linalg.det(rot)
    return {
        "orthonormality_mean": float(np.mean(ortho_err)),
        "orthonormality_max": float(np.max(ortho_err)),
        "det_mean": float(np.mean(det)),
        "det_min": float(np.min(det)),
        "det_max": float(np.max(det)),
        "det_abs_dev_mean": float(np.mean(np.abs(det - 1.0))),
    }


def camera_motion_metrics(c2w: np.ndarray) -> Dict[str, float]:
    centers = c2w[:, :3, 3]
    if len(centers) < 2:
        return {
            "median_step": 0.0,
            "mean_step": 0.0,
            "max_step": 0.0,
            "path_length": 0.0,
            "bbox_diag": 0.0,
            "jerk_mean": 0.0,
        }

    diffs = centers[1:] - centers[:-1]
    step = np.linalg.norm(diffs, axis=1)

    if len(diffs) >= 2:
        jerk = np.linalg.norm(diffs[1:] - diffs[:-1], axis=1)
        jerk_mean = float(np.mean(jerk))
    else:
        jerk_mean = 0.0

    bbox_min = np.min(centers, axis=0)
    bbox_max = np.max(centers, axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)

    return {
        "median_step": float(np.median(step)),
        "mean_step": float(np.mean(step)),
        "max_step": float(np.max(step)),
        "path_length": float(np.sum(step)),
        "bbox_diag": float(bbox_diag),
        "jerk_mean": jerk_mean,
    }


def intrinsics_metrics(intr: np.ndarray, image_wh: Tuple[int, int]) -> Dict[str, float]:
    w, h = image_wh
    fx, fy, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]
    fxn, fyn = fx / max(w, 1), fy / max(h, 1)
    cxn, cyn = cx / max(w, 1), cy / max(h, 1)

    return {
        "fx_mean": float(np.mean(fx)),
        "fy_mean": float(np.mean(fy)),
        "cx_mean": float(np.mean(cx)),
        "cy_mean": float(np.mean(cy)),
        "fx_std": float(np.std(fx)),
        "fy_std": float(np.std(fy)),
        "cx_std": float(np.std(cx)),
        "cy_std": float(np.std(cy)),
        "fx_norm_mean": float(np.mean(fxn)),
        "fy_norm_mean": float(np.mean(fyn)),
        "cx_norm_mean": float(np.mean(cxn)),
        "cy_norm_mean": float(np.mean(cyn)),
        "principal_center_offset_norm": float(
            np.mean(np.sqrt((cxn - 0.5) ** 2 + (cyn - 0.5) ** 2))
        ),
    }


def make_k(fxfycxcy: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = fxfycxcy
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k


def invert_batch(mats: np.ndarray) -> np.ndarray:
    return np.linalg.inv(mats)


def apply_c2w_local_axis_flip(c2w: np.ndarray, flip_y: bool, flip_z: bool) -> np.ndarray:
    s = np.eye(4, dtype=np.float64)
    s[1, 1] = -1.0 if flip_y else 1.0
    s[2, 2] = -1.0 if flip_z else 1.0
    return c2w @ s[None]


def scaled_translation(c2w: np.ndarray, scale: float) -> np.ndarray:
    out = c2w.copy()
    out[:, :3, 3] *= scale
    return out


def build_candidates(raw_w2c: np.ndarray) -> Dict[str, np.ndarray]:
    c2w_default = invert_batch(raw_w2c)
    candidates = {
        "default_w2c_to_c2w": c2w_default,
        "raw_as_c2w": raw_w2c.copy(),
        "default_flip_yz": apply_c2w_local_axis_flip(c2w_default, flip_y=True, flip_z=True),
        "default_flip_y": apply_c2w_local_axis_flip(c2w_default, flip_y=True, flip_z=False),
        "default_flip_z": apply_c2w_local_axis_flip(c2w_default, flip_y=False, flip_z=True),
    }
    return candidates


def project_points(w2c: np.ndarray, k: np.ndarray, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = w2c[:3, :3]
    t = w2c[:3, 3:4]
    points_cam = (r @ points_world.T + t).T
    z = points_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(points_world), 2), dtype=np.float64)
    uv[valid, 0] = k[0, 0] * (points_cam[valid, 0] / z[valid]) + k[0, 2]
    uv[valid, 1] = k[1, 1] * (points_cam[valid, 1] / z[valid]) + k[1, 2]
    return uv, valid


def make_reference_world_points(c2w0: np.ndarray, median_step: float) -> np.ndarray:
    c = c2w0[:3, 3]
    right = c2w0[:3, 0]
    down = c2w0[:3, 1]
    forward = c2w0[:3, 2]

    d = max(1.0, 20.0 * median_step)
    pts = []
    xs = np.linspace(-0.7, 0.7, 9)
    ys = np.linspace(-0.45, 0.45, 7)
    for x in xs:
        for y in ys:
            pts.append(c + forward * d + right * (x * d) + down * (y * d))
    return np.array(pts, dtype=np.float64)


def candidate_reprojection_score(c2w: np.ndarray, intrinsics: np.ndarray, image_wh: Tuple[int, int]) -> Dict[str, float]:
    w, h = image_wh
    motion = camera_motion_metrics(c2w)
    pts_world = make_reference_world_points(c2w[0], motion["median_step"])

    frame_count = len(c2w)
    max_eval = min(frame_count, 12)

    in_bounds_scores = []
    depth_valid_scores = []

    for i in range(max_eval):
        w2c_i = np.linalg.inv(c2w[i])
        k_i = make_k(intrinsics[i])
        uv, z_valid = project_points(w2c_i, k_i, pts_world)
        in_bounds = (
            z_valid
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < h)
        )
        depth_valid_scores.append(float(np.mean(z_valid)))
        in_bounds_scores.append(float(np.mean(in_bounds)))

    depth_valid = float(np.mean(depth_valid_scores))
    in_bounds = float(np.mean(in_bounds_scores))

    # A small regularizer that prefers plausible trajectory smoothness.
    jerk = motion["jerk_mean"]
    mean_step = motion["mean_step"]
    smoothness = float(np.exp(-jerk / (mean_step + 1e-6))) if mean_step > 0 else 0.0

    total = 0.55 * in_bounds + 0.35 * depth_valid + 0.10 * smoothness

    return {
        "score_total": float(total),
        "score_in_bounds": in_bounds,
        "score_depth_valid": depth_valid,
        "score_smoothness": smoothness,
    }


def draw_topdown_trajectory(c2w: np.ndarray, out_path: Path, title: str) -> None:
    centers = c2w[:, :3, 3]
    forwards = c2w[:, :3, 2]

    # Top-down plane uses X and Z.
    xz = centers[:, [0, 2]]
    min_v = np.min(xz, axis=0)
    max_v = np.max(xz, axis=0)
    span = np.maximum(max_v - min_v, 1e-6)

    canvas_h, canvas_w = 900, 1200
    margin = 80
    scale_x = (canvas_w - 2 * margin) / span[0]
    scale_y = (canvas_h - 2 * margin) / span[1]
    scale = min(scale_x, scale_y)

    def world_to_canvas(p: np.ndarray) -> Tuple[int, int]:
        px = int((p[0] - min_v[0]) * scale + margin)
        py = int((p[1] - min_v[1]) * scale + margin)
        # Image y grows down; invert for nicer top-down view.
        py = canvas_h - py
        return px, py

    img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 250

    pts = np.array([world_to_canvas(p) for p in xz], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 85, 220), 2, lineType=cv2.LINE_AA)

    for i, (center, fwd) in enumerate(zip(centers, forwards)):
        p0 = np.array([center[0], center[2]], dtype=np.float64)
        f2 = np.array([fwd[0], fwd[2]], dtype=np.float64)
        norm = np.linalg.norm(f2)
        if norm < 1e-8:
            continue
        f2 = f2 / norm

        p1 = p0 + f2 * max(span) * 0.04
        c0 = world_to_canvas(p0)
        c1 = world_to_canvas(p1)
        cv2.circle(img, c0, 3, (40, 40, 40), -1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(img, c0, c1, (20, 140, 20), 1, line_type=cv2.LINE_AA, tipLength=0.3)

        if i in (0, len(centers) - 1):
            cv2.putText(
                img,
                str(i),
                (c0[0] + 6, c0[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(img, title, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def load_texture_rgb(image_path: Path, max_tex_width: int = 128) -> np.ndarray:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if w > max_tex_width:
            new_w = max_tex_width
            new_h = max(8, int(round(h * new_w / max(w, 1))))
            rgb = rgb.resize((new_w, new_h), resample=Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def load_texture_rgba_strings(image_path: Path, max_tex_width: int = 96) -> np.ndarray:
    tex = load_texture_rgb(image_path, max_tex_width=max_tex_width)
    alpha = np.full((*tex.shape[:2], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([tex, alpha], axis=2)
    return np.array(
        [
            [f"rgba({int(px[0])},{int(px[1])},{int(px[2])},{px[3] / 255.0:.4f})" for px in row]
            for row in rgba
        ],
        dtype=object,
    )


def camera_axes(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = c2w[:3, 3]
    right = c2w[:3, 0]
    down = c2w[:3, 1]
    forward = c2w[:3, 2]
    return center, right, down, forward


def compute_visible_plane(c2w: np.ndarray, fxfycxcy: np.ndarray, image_wh: Tuple[int, int], depth: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width_px, height_px = image_wh
    fx, fy = fxfycxcy[0], fxfycxcy[1]
    center, right, down, forward = camera_axes(c2w)

    plane_center = center + forward * depth
    plane_half_w = depth * width_px / max(fx, 1e-8) * 0.5
    plane_half_h = depth * height_px / max(fy, 1e-8) * 0.5

    top_left = plane_center - right * plane_half_w - down * plane_half_h
    top_right = plane_center + right * plane_half_w - down * plane_half_h
    bottom_right = plane_center + right * plane_half_w + down * plane_half_h
    bottom_left = plane_center - right * plane_half_w + down * plane_half_h
    return top_left, top_right, bottom_right, bottom_left


def plot_textured_camera_plane(ax, c2w: np.ndarray, fxfycxcy: np.ndarray, image_wh: Tuple[int, int], image_path: Path, depth: float, label: str, color: Tuple[float, float, float]) -> None:
    center, right, down, forward = camera_axes(c2w)
    top_left, top_right, bottom_right, bottom_left = compute_visible_plane(c2w, fxfycxcy, image_wh, depth)

    tex = load_texture_rgb(image_path)
    tex_h, tex_w = tex.shape[:2]
    if tex_w < 2 or tex_h < 2:
        return

    u = np.linspace(0.0, 1.0, tex_w)
    v = np.linspace(0.0, 1.0, tex_h)
    uu, vv = np.meshgrid(u, v)
    plane = (
        top_left[None, None, :] * (1.0 - uu[..., None]) * (1.0 - vv[..., None])
        + top_right[None, None, :] * uu[..., None] * (1.0 - vv[..., None])
        + bottom_right[None, None, :] * uu[..., None] * vv[..., None]
        + bottom_left[None, None, :] * (1.0 - uu[..., None]) * vv[..., None]
    )

    rgba = np.concatenate([tex.astype(np.float32) / 255.0, np.ones((*tex.shape[:2], 1), dtype=np.float32)], axis=2)
    ax.plot_surface(
        plane[:, :, 0],
        plane[:, :, 1],
        plane[:, :, 2],
        facecolors=rgba,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    corners = [top_left, top_right, bottom_right, bottom_left, top_left]
    ax.plot([p[0] for p in corners], [p[1] for p in corners], [p[2] for p in corners], color=color, linewidth=1.5)
    ax.plot([center[0], top_left[0]], [center[1], top_left[1]], [center[2], top_left[2]], color=color, linewidth=0.8, alpha=0.7)
    ax.plot([center[0], top_right[0]], [center[1], top_right[1]], [center[2], top_right[2]], color=color, linewidth=0.8, alpha=0.7)
    ax.plot([center[0], bottom_left[0]], [center[1], bottom_left[1]], [center[2], bottom_left[2]], color=color, linewidth=0.8, alpha=0.7)
    ax.plot([center[0], bottom_right[0]], [center[1], bottom_right[1]], [center[2], bottom_right[2]], color=color, linewidth=0.8, alpha=0.7)

    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=16)
    label_text = f"{label}\nfx={fxfycxcy[0]:.1f} fy={fxfycxcy[1]:.1f}\ncx={fxfycxcy[2]:.1f} cy={fxfycxcy[3]:.1f}"
    ax.text(center[0], center[1], center[2], label_text, fontsize=7, color=color)

    # small axis triad to emphasize camera orientation
    axis_len = depth * 0.35
    ax.plot([center[0], center[0] + right[0] * axis_len], [center[1], center[1] + right[1] * axis_len], [center[2], center[2] + right[2] * axis_len], color=(1.0, 0.2, 0.2), linewidth=1.2)
    ax.plot([center[0], center[0] + down[0] * axis_len], [center[1], center[1] + down[1] * axis_len], [center[2], center[2] + down[2] * axis_len], color=(0.2, 1.0, 0.2), linewidth=1.2)
    ax.plot([center[0], center[0] + forward[0] * axis_len], [center[1], center[1] + forward[1] * axis_len], [center[2], center[2] + forward[2] * axis_len], color=(0.2, 0.4, 1.0), linewidth=1.2)


def scene_bbox_from_poses(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = c2w[:, :3, 3]
    bbox_min = np.min(centers, axis=0)
    bbox_max = np.max(centers, axis=0)
    span = bbox_max - bbox_min
    pad = np.maximum(span * 0.2, 1.0)
    return bbox_min - pad, bbox_max + pad


def plane_grid_from_corners(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_right: np.ndarray,
    bottom_left: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    u = np.linspace(0.0, 1.0, cols)
    v = np.linspace(0.0, 1.0, rows)
    uu, vv = np.meshgrid(u, v)
    plane = (
        top_left[None, None, :] * (1.0 - uu[..., None]) * (1.0 - vv[..., None])
        + top_right[None, None, :] * uu[..., None] * (1.0 - vv[..., None])
        + bottom_right[None, None, :] * uu[..., None] * vv[..., None]
        + bottom_left[None, None, :] * (1.0 - uu[..., None]) * vv[..., None]
    )
    return plane


def mesh_indices(rows: int, cols: int) -> Tuple[List[int], List[int], List[int]]:
    ii: List[int] = []
    jj: List[int] = []
    kk: List[int] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v0 = r * cols + c
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1
            ii.extend([v0, v1])
            jj.extend([v2, v2])
            kk.extend([v1, v3])
    return ii, jj, kk


def make_textured_plane_trace(
    c2w: np.ndarray,
    fxfycxcy: np.ndarray,
    image_wh: Tuple[int, int],
    image_path: Path,
    depth: float,
    name: str,
    visible: bool,
    rows: int = 18,
    cols: int = 18,
):
    top_left, top_right, bottom_right, bottom_left = compute_visible_plane(c2w, fxfycxcy, image_wh, depth)
    plane = plane_grid_from_corners(top_left, top_right, bottom_right, bottom_left, rows, cols)
    tex = load_texture_rgba_strings(image_path, max_tex_width=cols)
    ii, jj, kk = mesh_indices(rows, cols)

    return go.Mesh3d(
        x=plane[:, :, 0].reshape(-1),
        y=plane[:, :, 1].reshape(-1),
        z=plane[:, :, 2].reshape(-1),
        i=ii,
        j=jj,
        k=kk,
        vertexcolor=tex.reshape(-1),
        flatshading=True,
        showscale=False,
        name=name,
        visible=visible,
        hoverinfo="skip",
    )


def make_camera_traces(c2w: np.ndarray, intrinsics: np.ndarray, image_paths: List[Path], image_wh: Tuple[int, int], candidate_name: str, max_frames: int = 8, visible: bool = False) -> List[object]:
    indices = np.linspace(0, len(c2w) - 1, num=min(max_frames, len(c2w)), dtype=int)
    centers = c2w[:, :3, 3]
    motion = camera_motion_metrics(c2w)
    plane_depth = max(0.5, motion["median_step"] * 4.0)

    traces: List[object] = []
    traces.append(
        go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=5),
            marker=dict(size=3, color="#1f77b4"),
            name=f"{candidate_name} trajectory",
            visible=visible,
            hoverinfo="skip",
        )
    )

    for idx in indices:
        center, right, down, forward = camera_axes(c2w[idx])
        fx, fy, cx, cy = intrinsics[idx]
        top_left, top_right, bottom_right, bottom_left = compute_visible_plane(c2w[idx], intrinsics[idx], image_wh, plane_depth)

        traces.append(
            go.Scatter3d(
                x=[center[0], top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]],
                y=[center[1], top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]],
                z=[center[2], top_left[2], top_right[2], bottom_right[2], bottom_left[2], top_left[2]],
                mode="lines",
                line=dict(color="rgba(50,50,50,0.55)", width=2),
                name=f"{candidate_name} frustum {idx:03d}",
                visible=visible,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                text=[f"{idx:03d}<br>fx={fx:.1f} fy={fy:.1f}<br>cx={cx:.1f} cy={cy:.1f}"],
                textposition="top center",
                marker=dict(size=5, color="crimson"),
                name=f"{candidate_name} pose {idx:03d}",
                visible=visible,
            )
        )
        traces.append(
            make_textured_plane_trace(
                c2w[idx],
                intrinsics[idx],
                image_wh,
                image_paths[idx],
                plane_depth,
                f"{candidate_name} image {idx:03d}",
                visible=visible,
            )
        )
        axis_len = plane_depth * 0.35
        traces.append(
            go.Scatter3d(
                x=[center[0], center[0] + right[0] * axis_len],
                y=[center[1], center[1] + right[1] * axis_len],
                z=[center[2], center[2] + right[2] * axis_len],
                mode="lines",
                line=dict(color="red", width=4),
                name=f"{candidate_name} x-axis {idx:03d}",
                visible=visible,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[center[0], center[0] + down[0] * axis_len],
                y=[center[1], center[1] + down[1] * axis_len],
                z=[center[2], center[2] + down[2] * axis_len],
                mode="lines",
                line=dict(color="green", width=4),
                name=f"{candidate_name} y-axis {idx:03d}",
                visible=visible,
                hoverinfo="skip",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[center[0], center[0] + forward[0] * axis_len],
                y=[center[1], center[1] + forward[1] * axis_len],
                z=[center[2], center[2] + forward[2] * axis_len],
                mode="lines",
                line=dict(color="blue", width=4),
                name=f"{candidate_name} z-axis {idx:03d}",
                visible=visible,
                hoverinfo="skip",
            )
        )

    return traces


def render_interactive_scene(scene: SceneData, c2w_map: Dict[str, np.ndarray], out_path: Path, title: str) -> None:
    candidate_names = list(c2w_map.keys())
    traces: List[object] = []
    candidate_trace_ranges: Dict[str, Tuple[int, int]] = {}

    for candidate_name in candidate_names:
        start = len(traces)
        visible = candidate_name == candidate_names[0]
        traces.extend(
            make_camera_traces(
                c2w_map[candidate_name],
                scene.intrinsics,
                scene.image_paths,
                scene.image_size_wh,
                candidate_name,
                visible=visible,
            )
        )
        candidate_trace_ranges[candidate_name] = (start, len(traces))

    buttons = []
    for candidate_name in candidate_names:
        vis = [False] * len(traces)
        start, end = candidate_trace_ranges[candidate_name]
        for i in range(start, end):
            vis[i] = True
        buttons.append(
            dict(
                label=candidate_name,
                method="update",
                args=[{"visible": vis}, {"title": f"{title} | {candidate_name}"}],
            )
        )

    centers = c2w_map[candidate_names[0]][:, :3, 3]
    bbox_min, bbox_max = scene_bbox_from_poses(c2w_map[candidate_names[0]])
    centers_mid = np.mean(centers, axis=0)
    span = np.maximum(bbox_max - bbox_min, 1e-6)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} | {candidate_names[0]}",
        template="plotly_white",
        width=1500,
        height=1050,
        margin=dict(l=0, r=0, t=70, b=0),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.02,
                y=0.98,
                buttons=buttons,
                showactive=True,
            )
        ],
        scene=dict(
            xaxis=dict(title="X", range=[centers_mid[0] - span[0] * 0.65, centers_mid[0] + span[0] * 0.65]),
            yaxis=dict(title="Y", range=[centers_mid[1] - span[1] * 0.65, centers_mid[1] + span[1] * 0.65]),
            zaxis=dict(title="Z", range=[centers_mid[2] - span[2] * 0.65, centers_mid[2] + span[2] * 0.65]),
            aspectmode="manual",
            aspectratio=dict(x=max(span[0], 1e-6), y=max(span[1], 1e-6), z=max(span[2], 1e-6)),
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
        ),
        legend=dict(itemsizing="constant"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_path), include_plotlyjs="cdn", full_html=True, auto_open=False)


def render_3d_scene(scene: SceneData, c2w: np.ndarray, out_path: Path, title: str, max_frames: int = 8) -> None:
    indices = np.linspace(0, len(c2w) - 1, num=min(max_frames, len(c2w)), dtype=int)
    centers = c2w[:, :3, 3]
    motion = camera_motion_metrics(c2w)
    plane_depth = max(0.5, motion["median_step"] * 4.0)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")

    line_color = "#1f77b4"
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=line_color, linewidth=2.0, alpha=0.85)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color=line_color, s=18, alpha=0.85)

    for rank, idx in enumerate(indices):
        color = plt.cm.tab10(rank % 10)[:3]
        label = f"{idx:03d}"
        plot_textured_camera_plane(
            ax,
            c2w[idx],
            scene.intrinsics[idx],
            scene.image_size_wh,
            scene.image_paths[idx],
            plane_depth,
            label,
            color,
        )

    bbox_min, bbox_max = scene_bbox_from_poses(c2w)
    centers_mid = np.mean(centers, axis=0)
    span = bbox_max - bbox_min
    span = np.maximum(span, 1e-6)
    ax.set_xlim(centers_mid[0] - span[0] * 0.6, centers_mid[0] + span[0] * 0.6)
    ax.set_ylim(centers_mid[1] - span[1] * 0.6, centers_mid[1] + span[1] * 0.6)
    ax.set_zlim(centers_mid[2] - span[2] * 0.6, centers_mid[2] + span[2] * 0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    try:
        ax.set_box_aspect(span)
    except Exception:
        pass
    ax.view_init(elev=20, azim=-60)
    ax.grid(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def recommend_translation_scale(candidate_median_step: float, reference_median_step: float | None) -> Dict[str, float]:
    if reference_median_step is None or reference_median_step <= 0:
        return {"recommended_scale": 1.0, "distance_log": 0.0}

    scales = [1.0, 0.1, 0.01, 0.001]
    best_scale = 1.0
    best_dist = float("inf")
    for s in scales:
        d = abs(np.log((candidate_median_step * s + 1e-12) / (reference_median_step + 1e-12)))
        if d < best_dist:
            best_dist = d
            best_scale = s
    return {"recommended_scale": float(best_scale), "distance_log": float(best_dist)}


def evaluate_scene(
    scene: SceneData,
    out_dir: Path,
    reference_median_step: float | None = None,
    render_3d_candidates: List[str] | None = None,
    render_interactive: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rot_metrics = rotation_metrics(scene.w2c[:, :3, :3])
    intr_metrics = intrinsics_metrics(scene.intrinsics, scene.image_size_wh)

    candidates = build_candidates(scene.w2c)
    candidate_results = {}

    for name, c2w in candidates.items():
        motion = camera_motion_metrics(c2w)
        score = candidate_reprojection_score(c2w, scene.intrinsics, scene.image_size_wh)
        candidate_results[name] = {
            "motion": motion,
            "score": score,
        }

        vis_path = out_dir / f"{scene.scene_name}__{name}__topdown.png"
        draw_topdown_trajectory(c2w, vis_path, f"{scene.scene_name} | {name}")

    ranked = sorted(
        candidate_results.items(),
        key=lambda kv: kv[1]["score"]["score_total"],
        reverse=True,
    )

    best_candidate_name = ranked[0][0]
    best_median_step = candidate_results[best_candidate_name]["motion"]["median_step"]
    scale_hint = recommend_translation_scale(best_median_step, reference_median_step)

    candidate_names_to_render = [best_candidate_name, "default_w2c_to_c2w"] + (render_3d_candidates or [])
    candidate_names_to_render = list(dict.fromkeys(candidate_names_to_render))
    for candidate_name in candidate_names_to_render:
        if candidate_name not in candidates:
            continue
        render_3d_scene(
            scene,
            candidates[candidate_name],
            out_dir / f"{scene.scene_name}__{candidate_name}__3d.png",
            f"{scene.scene_name} | {candidate_name} | textured frustums",
        )

    if render_interactive:
        html_path = out_dir / f"{scene.scene_name}__interactive.html"
        interactive_candidates = {
            "default_w2c_to_c2w": candidates["default_w2c_to_c2w"],
            best_candidate_name: candidates[best_candidate_name],
        }
        render_interactive_scene(
            scene,
            interactive_candidates,
            html_path,
            f"{scene.scene_name} | interactive camera viewer",
        )

    return {
        "scene_name": scene.scene_name,
        "metadata_path": str(scene.metadata_path),
        "image_size_wh": list(scene.image_size_wh),
        "num_frames": int(scene.w2c.shape[0]),
        "raw_w2c_rotation": raw_rot_metrics,
        "intrinsics": intr_metrics,
        "candidates": candidate_results,
        "best_candidate": best_candidate_name,
        "best_score": ranked[0][1]["score"],
        "translation_scale_hint": scale_hint,
        "ranking": [name for name, _ in ranked],
    }


def load_dataset_scenes(dataset_dir: Path, repo_root: Path, limit_scenes: int | None) -> List[SceneData]:
    metadata_dir = dataset_dir / "metadata"
    files = sorted(metadata_dir.glob("*.json"))
    if limit_scenes is not None:
        files = files[:limit_scenes]

    scenes = []
    for fp in files:
        try:
            scenes.append(load_scene(fp, repo_root, dataset_dir))
        except Exception as e:
            print(f"[WARN] Failed to load {fp}: {e}")
    return scenes


def summarize_dataset(results: List[Dict[str, object]]) -> Dict[str, object]:
    if not results:
        return {"num_scenes": 0}

    best_candidates = [r["best_candidate"] for r in results]
    score_totals = [r["best_score"]["score_total"] for r in results]
    median_steps = [
        r["candidates"][r["best_candidate"]]["motion"]["median_step"]
        for r in results
    ]

    counts: Dict[str, int] = {}
    for c in best_candidates:
        counts[c] = counts.get(c, 0) + 1

    return {
        "num_scenes": len(results),
        "best_candidate_histogram": counts,
        "score_total_mean": float(np.mean(score_totals)),
        "score_total_std": float(np.std(score_totals)),
        "best_median_step_mean": float(np.mean(median_steps)),
        "best_median_step_std": float(np.std(median_steps)),
    }


def run_dataset(
    dataset_name: str,
    dataset_dir: Path,
    repo_root: Path,
    out_root: Path,
    limit_scenes: int | None,
    reference_median_step: float | None = None,
    max_3d_scenes: int | None = None,
    interactive: bool = False,
) -> Dict[str, object]:
    scenes = load_dataset_scenes(dataset_dir, repo_root, limit_scenes)
    print(f"Loaded {len(scenes)} scenes from {dataset_name}")

    per_scene = []
    scene_out = out_root / dataset_name / "scenes"
    scene_out.mkdir(parents=True, exist_ok=True)

    render_3d_candidates = ["default_w2c_to_c2w"]

    for idx, scene in enumerate(scenes, start=1):
        print(f"  [{idx}/{len(scenes)}] Evaluating {scene.scene_name}")
        scene_render_candidates = render_3d_candidates if (max_3d_scenes is None or idx <= max_3d_scenes) else []
        result = evaluate_scene(
            scene,
            scene_out,
            reference_median_step=reference_median_step,
            render_3d_candidates=scene_render_candidates,
            render_interactive=interactive,
        )
        per_scene.append(result)

    summary = summarize_dataset(per_scene)

    payload = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "summary": summary,
        "scenes": per_scene,
    }

    out_json = out_root / dataset_name / "summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def compare_reports(official: Dict[str, object], custom: Dict[str, object]) -> Dict[str, object]:
    o_sum = official.get("summary", {})
    c_sum = custom.get("summary", {})

    return {
        "official_num_scenes": o_sum.get("num_scenes", 0),
        "custom_num_scenes": c_sum.get("num_scenes", 0),
        "official_best_candidate_histogram": o_sum.get("best_candidate_histogram", {}),
        "custom_best_candidate_histogram": c_sum.get("best_candidate_histogram", {}),
        "official_best_median_step_mean": o_sum.get("best_median_step_mean", None),
        "custom_best_median_step_mean": c_sum.get("best_median_step_mean", None),
        "step_mean_ratio_custom_over_official": (
            (c_sum.get("best_median_step_mean", 0.0) + 1e-12)
            / (o_sum.get("best_median_step_mean", 0.0) + 1e-12)
            if o_sum.get("best_median_step_mean", 0.0) is not None
            else None
        ),
        "official_score_total_mean": o_sum.get("score_total_mean", None),
        "custom_score_total_mean": c_sum.get("score_total_mean", None),
    }


def make_single_dataset_comparison(
    official: Dict[str, object] | None,
    custom: Dict[str, object] | None,
) -> Dict[str, object]:
    if official is not None and custom is not None:
        return compare_reports(official, custom)
    if official is not None:
        summary = official.get("summary", {})
        return {
            "official_num_scenes": summary.get("num_scenes", 0),
            "custom_num_scenes": 0,
            "official_best_candidate_histogram": summary.get("best_candidate_histogram", {}),
            "custom_best_candidate_histogram": {},
            "official_best_median_step_mean": summary.get("best_median_step_mean", None),
            "custom_best_median_step_mean": None,
            "step_mean_ratio_custom_over_official": None,
            "official_score_total_mean": summary.get("score_total_mean", None),
            "custom_score_total_mean": None,
        }
    if custom is not None:
        summary = custom.get("summary", {})
        return {
            "official_num_scenes": 0,
            "custom_num_scenes": summary.get("num_scenes", 0),
            "official_best_candidate_histogram": {},
            "custom_best_candidate_histogram": summary.get("best_candidate_histogram", {}),
            "official_best_median_step_mean": None,
            "custom_best_median_step_mean": summary.get("best_median_step_mean", None),
            "step_mean_ratio_custom_over_official": None,
            "official_score_total_mean": None,
            "custom_score_total_mean": summary.get("score_total_mean", None),
        }
    return {
        "official_num_scenes": 0,
        "custom_num_scenes": 0,
        "official_best_candidate_histogram": {},
        "custom_best_candidate_histogram": {},
        "official_best_median_step_mean": None,
        "custom_best_median_step_mean": None,
        "step_mean_ratio_custom_over_official": None,
        "official_score_total_mean": None,
        "custom_score_total_mean": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LVSM camera convention on metadata datasets.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--official-dir", type=Path, default=Path("preprocessed_data/test"))
    parser.add_argument("--custom-dir", type=Path, default=Path("preprocessed_data/test_waymo"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/camera_validation"))
    parser.add_argument(
        "--official-limit-scenes",
        type=int,
        default=20,
        help="Limit official scenes for speed (default 20). Use 0 or negative for all.",
    )
    parser.add_argument(
        "--custom-limit-scenes",
        type=int,
        default=0,
        help="Limit custom scenes for speed (default all). Use 0 or negative for all.",
    )
    parser.add_argument(
        "--max-3d-scenes",
        type=int,
        default=3,
        help="Render 3D textured visualizations for only the first N scenes per dataset. Use 0 or negative for all.",
    )
    parser.add_argument(
        "--interactive-html",
        action="store_true",
        help="Also write interactive HTML viewers for each evaluated scene.",
    )
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Skip the official reference dataset and run only the custom dataset.",
    )
    parser.add_argument(
        "--skip-custom",
        action="store_true",
        help="Skip the custom dataset and run only the official reference dataset.",
    )
    return parser.parse_args()


def normalize_limit(v: int) -> int | None:
    return None if v <= 0 else v


def normalize_scene_limit(v: int) -> int | None:
    return None if v <= 0 else v


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    official_dir = (repo_root / args.official_dir).resolve()
    custom_dir = (repo_root / args.custom_dir).resolve()
    out_root = (repo_root / args.out_dir).resolve()

    print("=== LVSM Camera Convention Validator ===")
    print(f"repo_root: {repo_root}")
    print(f"official_dir: {official_dir}")
    print(f"custom_dir: {custom_dir}")
    print(f"out_dir: {out_root}")

    official_ref_step = None
    official = None
    if not args.skip_official:
        official = run_dataset(
            dataset_name="official_test",
            dataset_dir=official_dir,
            repo_root=repo_root,
            out_root=out_root,
            limit_scenes=normalize_limit(args.official_limit_scenes),
            max_3d_scenes=normalize_scene_limit(args.max_3d_scenes),
            interactive=args.interactive_html,
        )

        official_scenes = official.get("scenes", [])
        if official_scenes:
            step_vals = []
            for s in official_scenes:
                cand = s.get("candidates", {}).get("default_w2c_to_c2w", {})
                motion = cand.get("motion", {})
                st = motion.get("median_step", None)
                if isinstance(st, (float, int)) and st > 0:
                    step_vals.append(float(st))
            if step_vals:
                official_ref_step = float(np.median(step_vals))
                print(f"Reference median step from official default convention: {official_ref_step:.6g}")

    custom = None
    if not args.skip_custom:
        custom = run_dataset(
            dataset_name="custom_test_waymo",
            dataset_dir=custom_dir,
            repo_root=repo_root,
            out_root=out_root,
            limit_scenes=normalize_limit(args.custom_limit_scenes),
            reference_median_step=official_ref_step,
            max_3d_scenes=normalize_scene_limit(args.max_3d_scenes),
            interactive=args.interactive_html,
        )

    comparison = make_single_dataset_comparison(official, custom)
    out_cmp = out_root / "comparison_summary.json"
    with out_cmp.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Comparison Summary ===")
    print(json.dumps(comparison, indent=2))
    print(f"\nWrote comparison report: {out_cmp}")


if __name__ == "__main__":
    main()
