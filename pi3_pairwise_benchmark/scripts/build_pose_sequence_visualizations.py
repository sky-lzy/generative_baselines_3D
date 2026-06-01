#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from pi3_pairwise_benchmark.io import load_manifest, safe_sequence_name


COLORS = {
    "gt": "#285ad8",
    "pi3": "#d84a28",
    "vwm": "#148a5a",
}


def _camera_center(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w[:3, 3], dtype=np.float64)


def _tum_to_c2w(tum_poses: np.ndarray) -> np.ndarray:
    out = []
    for row in tum_poses:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 3] = row[:3]
        quat_wxyz = row[3:]
        mat[:3, :3] = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
        out.append(mat)
    return np.stack(out, axis=0)


def _similarity_align_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0:
        return pred
    aligned = (gt[0] @ np.linalg.inv(pred[0]) @ pred).copy()
    gt_centers = np.stack([_camera_center(x) for x in gt])
    pr_centers = np.stack([_camera_center(x) for x in aligned])
    gt_span = np.linalg.norm(gt_centers[-1] - gt_centers[0])
    pr_span = np.linalg.norm(pr_centers[-1] - pr_centers[0])
    scale = gt_span / pr_span if pr_span > 1e-8 else 1.0
    origin = pr_centers[0]
    for idx in range(len(aligned)):
        aligned[idx, :3, 3] = gt_centers[0] + scale * (_camera_center(aligned[idx]) - origin)
    return aligned


def _frustum_points(c2w: np.ndarray, size: float) -> list[np.ndarray]:
    pts_cam = [
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([-0.6, -0.4, 1.0, 1.0]) * size,
        np.array([0.6, -0.4, 1.0, 1.0]) * size,
        np.array([0.6, 0.4, 1.0, 1.0]) * size,
        np.array([-0.6, 0.4, 1.0, 1.0]) * size,
    ]
    pts_cam[0][3] = 1.0
    for pt in pts_cam[1:]:
        pt[3] = 1.0
    return [(c2w @ pt)[:3] for pt in pts_cam]


def _draw_frustum(ax, c2w: np.ndarray, color: str, size: float, alpha: float = 1.0) -> None:
    pts = _frustum_points(c2w, size)
    for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
        seg = np.stack([pts[i], pts[j]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, alpha=alpha, linewidth=1.0)


def _set_equal_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.max(np.linalg.norm(points - center, axis=1))), 1e-3)
    for setter, val in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(val - radius, val + radius)


def _contact_sheet(image_paths: list[str], labels: list[str], output_path: Path, max_images: int = 12) -> None:
    if len(image_paths) > max_images:
        idxs = np.linspace(0, len(image_paths) - 1, max_images).round().astype(int).tolist()
        image_paths = [image_paths[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    thumbs = []
    for path, label in zip(image_paths, labels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((180, 120), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (190, 150), "white")
        canvas.paste(image, ((190 - image.width) // 2, 6))
        ImageDraw.Draw(canvas).text((6, 130), label[:28], fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (190 * len(thumbs), 150), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (190 * idx, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _render_case(
    name: str,
    image_paths: list[str],
    labels: list[str],
    gt_c2w: np.ndarray,
    pred_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, str]:
    case_dir = output_dir / safe_sequence_name(name)
    case_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = case_dir / "inputs.jpg"
    _contact_sheet(image_paths, labels, inputs_path)

    preds = {}
    for method, path in pred_paths.items():
        if path and path.is_file():
            pred = np.load(path)
            if pred.shape[-2:] == (3, 4):
                tmp = np.tile(np.eye(4), (pred.shape[0], 1, 1))
                tmp[:, :3, :4] = pred
                pred = tmp
            preds[method] = _similarity_align_to_gt(pred, gt_c2w)

    gt_centers = np.stack([_camera_center(x) for x in gt_c2w])
    frustum_size = max(np.linalg.norm(gt_centers[-1] - gt_centers[0]), 1.0) * 0.035
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], color=COLORS["gt"], marker="o", label="GT")
    stride = max(len(gt_c2w) // 12, 1)
    for c2w in gt_c2w[::stride]:
        _draw_frustum(ax, c2w, COLORS["gt"], frustum_size, alpha=0.25)
    points = [gt_centers]
    for method, pred in preds.items():
        centers = np.stack([_camera_center(x) for x in pred])
        color = COLORS.get(method, "#444444")
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, marker="^", label=method)
        for c2w in pred[:: max(len(pred) // 12, 1)]:
            _draw_frustum(ax, c2w, color, frustum_size, alpha=0.85)
        points.append(centers)
    _set_equal_axes(ax, np.concatenate(points, axis=0))
    ax.set_title(name)
    ax.legend(loc="upper left")
    panel_path = case_dir / "poses.png"
    fig.tight_layout()
    fig.savefig(panel_path)
    plt.close(fig)
    return {"name": name, "inputs": str(inputs_path), "panel": str(panel_path)}


def _build_re10k(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = load_manifest(args.re10k_manifest, require_images=True)[: args.max_re10k]
    out = []
    for row in rows:
        seq = safe_sequence_name(row.sequence_name)
        out.append(
            _render_case(
                f"Re10K ordered {row.sequence_name}",
                row.image_paths,
                [f"idx {idx} frame {fid}" for idx, fid in enumerate(row.image_ids)],
                row.gt_c2w,
                {
                    "pi3": args.pi3_re10k_dir / seq / "pred_c2w.npy",
                    "vwm": args.vwm_re10k_dir / seq / "pred_c2w.npy",
                },
                args.output_dir,
            )
        )
    return out


def _build_tum_window(args: argparse.Namespace, name: str, start: int, pred_dir: Path) -> dict[str, str]:
    sys.path.insert(0, str(args.pi3_root))
    from relpose.evo_utils import load_traj

    seq = args.tum_sequence
    image_dir = args.tum_root / seq / "rgb_90"
    image_paths = [str(p) for p in sorted(image_dir.glob("*.png"))][start : start + 50]
    gt_tum, _ = load_traj(str(args.tum_root / seq / "groundtruth_90.txt"), traj_format="tum", skip=start, num_frames=50)
    gt_c2w = _tum_to_c2w(gt_tum)
    pi3_pred = args.pi3_tum_dir / seq / "pred_poses.npy"
    if pi3_pred.is_file():
        full = np.load(pi3_pred)
        sliced = args.output_dir / "_tmp" / f"pi3_{name}.npy"
        sliced.parent.mkdir(parents=True, exist_ok=True)
        np.save(sliced, full[start : start + 50])
        pi3_pred = sliced
    return _render_case(
        f"TUM {name} {seq}",
        image_paths,
        [f"frame {start + i}" for i in range(len(image_paths))],
        gt_c2w,
        {
            "pi3": pi3_pred,
            "vwm": pred_dir / "tum" / seq / "pred_poses.npy",
        },
        args.output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static pose visualizations for Pi3/VWM sequence outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--re10k-manifest", type=Path, required=True)
    parser.add_argument("--pi3-re10k-dir", type=Path, required=True)
    parser.add_argument("--vwm-re10k-dir", type=Path, required=True)
    parser.add_argument("--max-re10k", type=int, default=3)
    parser.add_argument("--pi3-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Pi3")
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--tum-sequence", default="rgbd_dataset_freiburg3_sitting_halfsphere")
    parser.add_argument("--pi3-tum-dir", type=Path, required=True)
    parser.add_argument("--vwm-tum-first50-dir", type=Path, required=True)
    parser.add_argument("--vwm-tum-last50-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _build_re10k(args)
    records.append(_build_tum_window(args, "first50", 0, args.vwm_tum_first50_dir))
    records.append(_build_tum_window(args, "last50", 40, args.vwm_tum_last50_dir))

    cards = []
    for rec in records:
        rel_inputs = Path(rec["inputs"]).relative_to(args.output_dir)
        rel_panel = Path(rec["panel"]).relative_to(args.output_dir)
        cards.append(
            f"<section><h2>{html.escape(rec['name'])}</h2>"
            f"<img src='{html.escape(str(rel_inputs))}'><img src='{html.escape(str(rel_panel))}'></section>"
        )
    index = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Pose visualizations</title>"
        "<style>body{font-family:sans-serif;margin:24px}section{margin-bottom:36px}"
        "img{max-width:100%;display:block;margin:10px 0;border:1px solid #ddd}</style></head><body>"
        "<h1>Pi3 / VWM Pose Visualizations</h1>"
        "<p>Predictions are similarity-aligned to GT for visualization only.</p>"
        + "\n".join(cards)
        + "</body></html>"
    )
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
