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


def _camera_center(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w[:3, 3], dtype=np.float64)


def _align_pred_pair_to_gt(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """Gauge-align a two-view prediction for visualization only.

    Pi3 relative-pose metrics are gauge/scale insensitive. For visual inspection,
    anchor predicted camera 0 to GT camera 0 and scale the predicted baseline to
    the GT baseline length.
    """
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
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    for i, j in edges:
        seg = np.stack([pts[i], pts[j]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, alpha=alpha, linewidth=1.2)


def _set_equal_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.max(np.linalg.norm(points - center, axis=1))), 1e-3)
    for setter, val in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(val - radius, val + radius)


def _make_contact_sheet(image_paths: list[str], labels: list[str], output_path: Path) -> None:
    thumbs = []
    for path, label in zip(image_paths, labels):
        image = Image.open(path).convert("RGB")
        image.thumbnail((220, 150), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (230, 180), "white")
        x = (230 - image.width) // 2
        canvas.paste(image, (x, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 158), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (230 * len(thumbs), 180), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (230 * idx, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_ply(path: Path, cameras: list[tuple[str, np.ndarray]], frustum_size: float) -> None:
    colors = {
        "gt": (40, 90, 220),
        "pred": (220, 80, 40),
    }
    vertices: list[tuple[float, float, float, int, int, int]] = []
    edges: list[tuple[int, int]] = []
    for label, c2w in cameras:
        color = colors[label]
        start = len(vertices)
        pts = _frustum_points(c2w, frustum_size)
        for pt in pts:
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


def _load_pair(pi3_2view_dir: Path, row: ManifestRow, pair: tuple[int, int]) -> tuple[np.ndarray, dict[str, float]]:
    pair_dir = pi3_2view_dir / safe_sequence_name(row.sequence_name) / f"pair_{pair[0]:02d}_{pair[1]:02d}"
    pred_c2w = np.load(pair_dir / "pred_c2w.npy")
    metrics = json.loads((pair_dir / "metrics.json").read_text(encoding="utf-8"))
    return pred_c2w, metrics


def _render_sequence(
    row: ManifestRow,
    pi3_2view_dir: Path,
    output_dir: Path,
    pairs: list[tuple[int, int]],
) -> list[dict[str, str]]:
    seq_dir = output_dir / safe_sequence_name(row.sequence_name)
    seq_dir.mkdir(parents=True, exist_ok=True)
    gt_centers = np.stack([_camera_center(c2w) for c2w in row.gt_c2w])
    frustum_size = max(np.linalg.norm(gt_centers[-1] - gt_centers[0]), 1.0) * 0.035
    records = []

    _make_contact_sheet(
        row.image_paths,
        [f"idx {i} / frame {frame_id}" for i, frame_id in enumerate(row.image_ids)],
        seq_dir / "inputs_10view.jpg",
    )

    for pair in pairs:
        pred_c2w, metrics = _load_pair(pi3_2view_dir, row, pair)
        gt_pair_c2w = row.gt_c2w[list(pair)]
        pred_aligned = _align_pred_pair_to_gt(pred_c2w, gt_pair_c2w)

        fig = plt.figure(figsize=(12, 5), dpi=140)
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], color="#285ad8", marker="o", label="GT 10-view")
        ax.plot(
            [_camera_center(pred_aligned[0])[0], _camera_center(pred_aligned[1])[0]],
            [_camera_center(pred_aligned[0])[1], _camera_center(pred_aligned[1])[1]],
            [_camera_center(pred_aligned[0])[2], _camera_center(pred_aligned[1])[2]],
            color="#d84a28",
            marker="^",
            label="Pi3 2-view pred aligned",
        )
        for c2w in row.gt_c2w:
            _draw_frustum(ax, c2w, "#285ad8", frustum_size, alpha=0.35)
        for c2w in pred_aligned:
            _draw_frustum(ax, c2w, "#d84a28", frustum_size, alpha=0.95)
        all_points = np.concatenate([gt_centers, np.stack([_camera_center(x) for x in pred_aligned])], axis=0)
        _set_equal_axes(ax, all_points)
        ax.set_title(f"{row.sequence_name} pair {pair[0]}-{pair[1]}")
        ax.legend(loc="upper left")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis("off")
        text = "\n".join(
            [
                "Pi3 2-view metrics",
                f"Racc_5: {metrics['Racc_5']:.1f}",
                f"Tacc_5: {metrics['Tacc_5']:.1f}",
                f"Auc_5: {metrics['Auc_5']:.2f}",
                f"Auc_30: {metrics['Auc_30']:.2f}",
                "",
                "Prediction is Sim(3)-aligned",
                "to GT for visualization only.",
            ]
        )
        ax2.text(0.02, 0.95, text, va="top", family="monospace", fontsize=11)
        png_path = seq_dir / f"pair_{pair[0]:02d}_{pair[1]:02d}_frustums.png"
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

        image_sheet = seq_dir / f"pair_{pair[0]:02d}_{pair[1]:02d}_inputs.jpg"
        _make_contact_sheet(
            [row.image_paths[pair[0]], row.image_paths[pair[1]]],
            [f"idx {pair[0]} / frame {row.image_ids[pair[0]]}", f"idx {pair[1]} / frame {row.image_ids[pair[1]]}"],
            image_sheet,
        )
        ply_path = seq_dir / f"pair_{pair[0]:02d}_{pair[1]:02d}_frustums.ply"
        cameras = [("gt", c2w) for c2w in row.gt_c2w] + [("pred", c2w) for c2w in pred_aligned]
        _write_ply(ply_path, cameras, frustum_size)
        records.append(
            {
                "sequence": row.sequence_name,
                "pair": f"{pair[0]}-{pair[1]}",
                "panel": str(png_path),
                "inputs": str(image_sheet),
                "ply": str(ply_path),
                "auc30": f"{metrics['Auc_30']:.2f}",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static Pi3 smoke visualizations from saved 2-view outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pi3-2view-dir", type=Path, required=True)
    parser.add_argument("--pi3-10view-csv", type=Path, required=True)
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
    records: list[dict[str, str]] = []
    for row in rows:
        records.extend(_render_sequence(row, args.pi3_2view_dir, args.output_dir, pairs))

    pi3_10 = args.pi3_10view_csv.read_text(encoding="utf-8").strip()
    pi3_2 = json.loads((args.pi3_2view_dir / "summary.json").read_text(encoding="utf-8"))
    cards = []
    for rec in records:
        rel_panel = Path(rec["panel"]).relative_to(args.output_dir)
        rel_inputs = Path(rec["inputs"]).relative_to(args.output_dir)
        rel_ply = Path(rec["ply"]).relative_to(args.output_dir)
        cards.append(
            f"""
            <section>
              <h2>{html.escape(rec['sequence'])} pair {html.escape(rec['pair'])}</h2>
              <p>Auc_30={html.escape(rec['auc30'])} | <a href="{rel_ply}">PLY frustums</a></p>
              <img src="{rel_inputs}" alt="input images">
              <img src="{rel_panel}" alt="camera frustum visualization">
            </section>
            """
        )
    index = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pi3 Re10K Smoke Visualizations</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    pre {{ background: #f3f4f6; padding: 12px; overflow-x: auto; }}
    section {{ border-top: 1px solid #d7dde5; padding: 18px 0; }}
    img {{ max-width: 100%; display: block; margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Pi3 Re10K Smoke Visualizations</h1>
  <p>Source run: 12967573. These are qualitative checks for the 3-sequence smoke run, not final benchmark averages.</p>
  <h2>Pi3 10-view metrics</h2>
  <pre>{html.escape(pi3_10)}</pre>
  <h2>Pi3 2-view summary</h2>
  <pre>{html.escape(json.dumps(pi3_2, indent=2, sort_keys=True))}</pre>
  {''.join(cards)}
</body>
</html>
"""
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")
    readme = f"""# Pi3 Re10K Smoke Visualizations

Source run: `12967573`.

Open `index.html` in a browser. Each selected pair includes:

- the two input images,
- a static PNG with the full GT 10-view trajectory and the Pi3 2-view predicted pair,
- a PLY file with GT and predicted camera frustums for external 3D inspection.

The predicted pair is Sim(3)-aligned to GT for visualization only. Quantitative
metrics are still computed by the benchmark scripts using the Pi3 relative-pose
angular metrics.

Pi3 10-view and 2-view smoke numbers are strong sanity evidence because pose
errors are very small on these sequences. Caveat: this is a 3-sequence smoke
run, not a final dataset average.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
