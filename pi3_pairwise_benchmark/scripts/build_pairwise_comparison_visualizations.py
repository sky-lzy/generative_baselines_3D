#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from pi3_pairwise_benchmark.io import ManifestRow, load_manifest, safe_sequence_name


COLORS = {
    "gt": "#285ad8",
    "pi3": "#d84a28",
    "ours": "#148a5a",
}
PLY_COLORS = {
    "gt": (40, 90, 220),
    "pi3": (220, 80, 40),
    "ours": (20, 138, 90),
}


def _camera_center(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w[:3, 3], dtype=np.float64)


def _align_pred_pair_to_gt(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    aligned = (gt_c2w[0] @ np.linalg.inv(pred_c2w[0]) @ pred_c2w).copy()
    gt_c0, gt_c1 = _camera_center(gt_c2w[0]), _camera_center(gt_c2w[1])
    pr_c0, pr_c1 = _camera_center(aligned[0]), _camera_center(aligned[1])
    gt_len = np.linalg.norm(gt_c1 - gt_c0)
    pr_len = np.linalg.norm(pr_c1 - pr_c0)
    scale = gt_len / pr_len if pr_len > 1e-8 else 1.0
    for idx in range(len(aligned)):
        aligned[idx, :3, 3] = gt_c0 + scale * (_camera_center(aligned[idx]) - pr_c0)
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
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, alpha=alpha, linewidth=1.1)


def _set_equal_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.max(np.linalg.norm(points - center, axis=1))), 1e-3)
    for setter, val in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(val - radius, val + radius)


def _make_contact_sheet(image_paths: list[str], labels: list[str], output_path: Path) -> None:
    thumbs = []
    for path, label in zip(image_paths, labels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((260, 170), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (270, 205), "white")
        canvas.paste(image, ((270 - image.width) // 2, 8))
        ImageDraw.Draw(canvas).text((8, 182), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (270 * len(thumbs), 205), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (270 * idx, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _load_prediction(pred_dir: Path | None, row: ManifestRow, pair: tuple[int, int]) -> tuple[np.ndarray, dict[str, float]] | None:
    if pred_dir is None:
        return None
    pair_dir = pred_dir / safe_sequence_name(row.sequence_name) / f"pair_{pair[0]:02d}_{pair[1]:02d}"
    if not (pair_dir / "pred_c2w.npy").is_file():
        return None
    metrics = {}
    if (pair_dir / "metrics.json").is_file():
        metrics = json.loads((pair_dir / "metrics.json").read_text(encoding="utf-8"))
    return np.load(pair_dir / "pred_c2w.npy"), metrics


def _optional_thumbnails(pair_dir: Path | None) -> list[Path]:
    if pair_dir is None or not pair_dir.is_dir():
        return []
    patterns = ["*depth*.png", "*point*.png", "*points*.png", "*map*.png"]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(pair_dir.glob(pattern)))
    return out[:4]


def _write_ply(path: Path, cameras: list[tuple[str, np.ndarray]], frustum_size: float) -> None:
    vertices: list[tuple[float, float, float, int, int, int]] = []
    edges: list[tuple[int, int]] = []
    for label, c2w in cameras:
        color = PLY_COLORS[label]
        start = len(vertices)
        for pt in _frustum_points(c2w, frustum_size):
            vertices.append((float(pt[0]), float(pt[1]), float(pt[2]), *color))
        for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            edges.append((start + i, start + j))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for vertex in vertices:
            f.write("%f %f %f %d %d %d\n" % vertex)
        for edge in edges:
            f.write("%d %d\n" % edge)


def _render_pair(
    row: ManifestRow,
    pair: tuple[int, int],
    pi3_dir: Path,
    ours_dir: Path | None,
    output_dir: Path,
) -> dict[str, str]:
    seq_dir = output_dir / safe_sequence_name(row.sequence_name)
    seq_dir.mkdir(parents=True, exist_ok=True)
    gt_pair_c2w = row.gt_c2w[list(pair)]
    gt_centers = np.stack([_camera_center(c2w) for c2w in row.gt_c2w])
    frustum_size = max(np.linalg.norm(gt_centers[-1] - gt_centers[0]), 1.0) * 0.035

    pi3_pred = _load_prediction(pi3_dir, row, pair)
    ours_pred = _load_prediction(ours_dir, row, pair)
    if pi3_pred is None:
        raise FileNotFoundError(f"Missing Pi3 prediction for {row.sequence_name} pair {pair}")

    pi3_c2w, pi3_metrics = pi3_pred
    pi3_aligned = _align_pred_pair_to_gt(pi3_c2w, gt_pair_c2w)
    ours_aligned = None
    ours_metrics = {}
    if ours_pred is not None:
        ours_c2w, ours_metrics = ours_pred
        ours_aligned = _align_pred_pair_to_gt(ours_c2w, gt_pair_c2w)

    fig = plt.figure(figsize=(13, 5), dpi=140)
    for panel_idx, (title, pred, color, metrics) in enumerate(
        [
            ("GT + Pi3", pi3_aligned, COLORS["pi3"], pi3_metrics),
            ("GT + ours", ours_aligned, COLORS["ours"], ours_metrics),
        ],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
        ax.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], color=COLORS["gt"], marker="o", label="GT")
        for c2w in row.gt_c2w:
            _draw_frustum(ax, c2w, COLORS["gt"], frustum_size, alpha=0.2)
        points = [gt_centers]
        if pred is not None:
            pred_centers = np.stack([_camera_center(x) for x in pred])
            ax.plot(pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2], color=color, marker="^", label=title)
            for c2w in pred:
                _draw_frustum(ax, c2w, color, frustum_size, alpha=0.95)
            points.append(pred_centers)
        metric_text = f"Auc30={metrics.get('Auc_30', float('nan')):.2f}" if metrics else "missing"
        ax.set_title(f"{title} | {metric_text}")
        _set_equal_axes(ax, np.concatenate(points, axis=0))
        ax.legend(loc="upper left")
    panel_path = seq_dir / f"pair_{pair[0]:02d}_{pair[1]:02d}_comparison.png"
    fig.tight_layout()
    fig.savefig(panel_path)
    plt.close(fig)

    inputs_path = seq_dir / f"pair_{pair[0]:02d}_{pair[1]:02d}_inputs.jpg"
    _make_contact_sheet(
        [row.image_paths[pair[0]], row.image_paths[pair[1]]],
        [f"idx {pair[0]} / frame {row.image_ids[pair[0]]}", f"idx {pair[1]} / frame {row.image_ids[pair[1]]}"],
        inputs_path,
    )
    ply_cameras = [("gt", c2w) for c2w in row.gt_c2w] + [("pi3", c2w) for c2w in pi3_aligned]
    if ours_aligned is not None:
        ply_cameras.extend(("ours", c2w) for c2w in ours_aligned)
    ply_path = seq_dir / f"pair_{pair[0]:02d}_{pair[1]:02d}_comparison.ply"
    _write_ply(ply_path, ply_cameras, frustum_size)

    return {
        "sequence": row.sequence_name,
        "pair": f"{pair[0]}-{pair[1]}",
        "inputs": str(inputs_path),
        "panel": str(panel_path),
        "ply": str(ply_path),
        "pi3_auc30": f"{pi3_metrics.get('Auc_30', float('nan')):.2f}",
        "ours_auc30": f"{ours_metrics.get('Auc_30', float('nan')):.2f}" if ours_metrics else "missing",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static Pi3-vs-ours pairwise comparison visualizations.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pi3-dir", type=Path, required=True)
    parser.add_argument("--ours-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--pairs", default="0,1;0,9;4,8")
    args = parser.parse_args()

    rows = load_manifest(args.manifest, require_images=True)[: args.max_sequences]
    pairs = []
    for item in args.pairs.split(";"):
        left, right = item.split(",")
        pairs.append((int(left), int(right)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in rows:
        for pair in pairs:
            records.append(_render_pair(row, pair, args.pi3_dir, args.ours_dir, args.output_dir))

    sections = []
    for rec in records:
        rel_inputs = Path(rec["inputs"]).relative_to(args.output_dir)
        rel_panel = Path(rec["panel"]).relative_to(args.output_dir)
        rel_ply = Path(rec["ply"]).relative_to(args.output_dir)
        sections.append(
            f"""
            <section>
              <h2>{html.escape(rec['sequence'])} pair {html.escape(rec['pair'])}</h2>
              <p>Pi3 Auc_30={html.escape(rec['pi3_auc30'])} | ours Auc_30={html.escape(rec['ours_auc30'])} | <a href="{rel_ply}">PLY frustums</a></p>
              <img src="{rel_inputs}" alt="input images">
              <img src="{rel_panel}" alt="pose comparison">
            </section>
            """
        )
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pairwise Pose Comparison</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    section {{ border-top: 1px solid #d7dde5; padding: 18px 0; }}
    img {{ max-width: 100%; display: block; margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Pairwise Pose Comparison</h1>
  <p>Predictions are Sim(3)-aligned to GT for visualization only. Metrics remain Pi3 relative-pose metrics.</p>
  {''.join(sections)}
</body>
</html>
"""
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "Open index.html in a browser. PLY files contain GT, Pi3, and ours camera frustums when available.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} comparison records to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
