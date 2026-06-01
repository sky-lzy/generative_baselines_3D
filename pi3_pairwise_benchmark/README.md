# Pi3 Pairwise Benchmark

This folder is isolated from the existing sparse-view and generative-baseline
code. It is for Re10K pose benchmarking with these modes:

1. `pi3_10view`: official Pi3 relpose reproduction, all 10 images at once.
2. `pi3_2view`: same 10-image samples, but Pi3 sees only one image pair per run.
3. `ours_2view`: same image pairs for `video_world_model`.
4. `pi3_50view`: official Pi3 relpose evaluation on 50 continuous frames per
   sequence, still using one inference per sequence and Pi3's aggregate
   unordered pair metric.
5. `ours_50view`: `video_world_model` inference once on the same 50 continuous
   frames, evaluated with the same all-pairs relative-pose metric.
6. `geo4d_50view` / `geo4d_2view`: official Geo4D code/checkpoints used only
   to produce predicted camera stacks, then scored by the same Pi3 relative-pose
   metric code.

The shared metric is Pi3's relative-pose angular metric over world-to-camera
poses: `Racc_5/15/30`, `Tacc_5/15/30`, and `Auc_5/15/30`.

Pi3 paper benchmark inventory and exact reproduction tracking:

```text
pi3_pairwise_benchmark/docs/pi3_paper_benchmark_reproduction.md
```

## Setup

```bash
cd /path/to/generative_baselines_3D
python pi3_pairwise_benchmark/scripts/bootstrap_pi3.py
```

The bootstrap clones `yyfz/Pi3` branch `evaluation` into `external/Pi3`.

Geo4D setup uses the official repository:

```text
pi3_pairwise_benchmark/external/Geo4D
```

Current Geo4D checkout: `f9188ba3ed497bb63d347df5347f172bb91cfc16`.
Official checkpoints are stored at:

```text
pi3_pairwise_benchmark/external/Geo4D/checkpoints/geo4d/vae.ckpt
pi3_pairwise_benchmark/external/Geo4D/checkpoints/geo4d/model.ckpt
```

The benchmark adapter includes only environment compatibility shims for missing
optional imports and OpenCLIP/PyTorch API differences. It does not use Geo4D's
own evaluation metrics; predicted `c2w` poses are evaluated by the local
Pi3-style metric code in `pi3_pairwise_benchmark/metrics.py`.

## Validation

```bash
PYTHONPATH=pi3_pairwise_benchmark \
python -m unittest discover -s pi3_pairwise_benchmark/tests -v
```

## Slurm Runs

Smoke run on 3 official Re10K seed-42 sequences:

```bash
sbatch --parsable pi3_pairwise_benchmark/slurm/re10k_pose_smoke_gpu.sbatch
```

The smoke script prepares Pi3-format Re10K samples from the local
`video_world_model` mp4/txt tree, runs Pi3 official 10-view Re10K relpose eval,
materializes a manifest, and runs Pi3 as a 2-view baseline over all 45 pairs per
sequence.

50-view smoke run:

```bash
sbatch --parsable pi3_pairwise_benchmark/slurm/re10k_pose_50view_smoke_gpu.sbatch
```

For 50-view reporting, the model runs once per sequence on 50 continuous frames.
Metrics follow Pi3's 10-view convention: evaluate all unordered relative pairs,
so each sequence contributes `C(50,2)=1225` pose pairs to the aggregate.

Run `video_world_model` on an existing 50-view manifest:

```bash
sbatch --parsable \
  --export=ALL,MANIFEST=/path/to/manifest_50view.jsonl,OUTPUT_DIR=/path/to/ours_50view_s40 \
  pi3_pairwise_benchmark/slurm/ours_50view_existing_gpu.sbatch
```

For 2-view sparse input, use deterministic sampled pairs. The default remains
`all` pairs for backwards compatibility; the long-baseline setting is:

```bash
python pi3_pairwise_benchmark/scripts/run_pi3_2view_pairs.py \
  --manifest /path/to/manifest_50view.jsonl \
  --output-dir /path/to/pi3_2view_long \
  --pretrained-model-name-or-path pi3_pairwise_benchmark/external/Pi3/checkpoints/Pi3 \
  --pair-policy long \
  --pairs-per-sequence 50 \
  --long-gap-min 30
```

Full run:

```bash
sbatch --parsable pi3_pairwise_benchmark/slurm/re10k_pose_full_gpu.sbatch
```

If the Slurm environment needs activation, pass `ENV_SETUP`, for example:

```bash
sbatch --parsable --export=ALL,ENV_SETUP='source ~/.bashrc && conda activate pi3' \
  pi3_pairwise_benchmark/slurm/re10k_pose_smoke_gpu.sbatch
```

## Current Integration State

The metric and artifact contracts are implemented and tested. The Pi3 official
10-view runner and Pi3 2-view runner are provided as integration scripts around
the Pi3 evaluation branch.

Current Pi3 Re10K smoke source: Slurm job `12967573`.

- Pi3 official 10-view smoke metrics on 3 Re10K sequences:
  `Racc_5=100.0`, `Tacc_5=100.0`, `Auc_5=90.37`,
  `Auc_30=98.40`.
- Pi3 2-view fair-baseline smoke metrics over the same 3 sequences and all
  135 image pairs:
  `Racc_5=100.0`, `Tacc_5=97.78`, `Auc_5=82.07`,
  `Auc_30=96.99`.

Interpretation: these are strong sanity evidence for the Pi3 setup because the
pose errors are very small in both the official 10-view path and pairwise
2-view path. Caveat: this is still only a 3-sequence smoke run, not a final
dataset average.

Qualitative visualization artifact:

```text
pi3_pairwise_benchmark/outputs/pi3_visualizations_12967573/index.html
```

This HTML includes selected input pairs, GT 10-view trajectories, Pi3 2-view
predicted camera frustums aligned to GT for visualization only, and PLY files
for external 3D inspection. The source 10-view Pi3 run only saved aggregate
metrics, not a point cloud/reconstruction artifact, so the current visualization
uses the saved Pi3 2-view pose predictions and manifest GT poses.

The `ours_2view` script is a complete pairwise evaluation harness for
`video_world_model`. The adapter builds a two-view first/last conditioning clip
and returns a `(2,4,4)` `c2w` stack for one image pair.

Current 50-view smoke source: Slurm job `13135511`.

- Pi3 official 50-view smoke metrics on 3 Re10K sequences, evaluated over all
  `C(50,2)` unordered pairs per sequence:
  `Racc_5=100.0`, `Tacc_5=96.71`, `Auc_5=81.12`,
  `Auc_30=96.63`.
- Pi3 2-view long-baseline smoke metrics over 20 deterministic long-baseline
  pairs per sequence, 60 pairs total:
  `Racc_5=100.0`, `Tacc_5=100.0`, `Auc_5=94.67`,
  `Auc_30=99.11`.
- VWM 2-view long-baseline smoke metrics over the same 60 pairs:
  `Racc_5=96.67`, `Tacc_5=46.67`, `Auc_5=22.33`,
  `Auc_30=81.72`.
- VWM true 50-view smoke metrics on the same 3 sequences, one inference per
  50-frame stack and evaluated over all `C(50,2)` unordered pairs per sequence:
  `Racc_5=100.0`, `Tacc_5=65.36`, `Auc_5=38.53`,
  `Racc_15=100.0`, `Tacc_15=93.33`, `Auc_15=66.82`,
  `Racc_30=100.0`, `Tacc_30=100.0`, `Auc_30=83.25`.
  Output:
  `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke_50view/13135511/outputs/ours_50view_s40`.
- Geo4D 50-view smoke metrics on the same 3 sequences, one inference per
  50-frame stack and evaluated over all `C(50,2)` unordered pairs per sequence:
  corrected Geo4D TUM quaternion parsing gives
  `Racc_5=100.0`, `Tacc_5=0.87`, `Auc_5=0.26`,
  `Racc_15=100.0`, `Tacc_15=22.56`, `Auc_15=7.10`,
  `Racc_30=100.0`, `Tacc_30=75.27`, `Auc_30=31.85`.
  Output:
  `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke_50view/13135511/outputs/geo4d_50view_s5`.
- Geo4D 2-view long-baseline smoke metrics over the same 60 pairs as Pi3/VWM:
  corrected Geo4D TUM quaternion parsing gives
  `Racc_5=100.0`, `Tacc_5=6.67`, `Auc_5=2.00`,
  `Racc_15=100.0`, `Tacc_15=36.67`, `Auc_15=19.00`,
  `Racc_30=100.0`, `Tacc_30=71.67`, `Auc_30=35.22`.
  Output:
  `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke_50view/13135511/outputs/geo4d_2view_long20_s5`.

Geo4D 2-view is implemented as "two unique images" input by padding each pair
to a 16-frame clip, because the official Geo4D inference path expects video
clips. The metric still evaluates only the two endpoint cameras.

Important interpretation note: Geo4D's official pose evaluation uses evo
trajectory metrics with `align=True` and `correct_scale=True`. On the same
three 50-frame Re10K smoke sequences, the saved Geo4D trajectories have small
official-eval-style aligned errors: mean `ATE=0.037`, mean `RPE_trans=0.030`,
and mean `RPE_rot=0.127 deg`. The low Pi3 `Tacc_5` is therefore mostly a
metric/convention mismatch: Pi3's relative-pose translation-angle metric scores
raw local-frame translation directions, while Geo4D's reported camera-pose
setup aligns the predicted trajectory before evaluation.

Geo4D Pi3-protocol diagnostics:

```text
pi3_pairwise_benchmark/outputs/geo4d_pi3_protocol_13183762/geo4d_pi3_protocol_diagnostic.md
pi3_pairwise_benchmark/outputs/geo4d_2view_pi3_protocol_13183763/geo4d_2view_pi3_protocol_diagnostic.md
```

The fixed camera-basis sweep finds `diag(1,-1,-1)` as the best proper
signed-permutation convention transform, improving Geo4D 50-view raw Pi3
`Auc_30` from `31.85` to `44.20`. GT-assisted scale-only, SE3, and Sim3 center
alignment do not change Pi3 translation-angle metrics, as expected from the
metric's global similarity invariance. A more unfair GT-assisted continuous
camera-basis oracle reaches `Auc_30=64.01` with one global basis, and
`Auc_30=79.65` with a separately fitted basis per sequence; both remain below
Pi3 50-view `Auc_30=96.63`.

For the 2-view long-baseline setting, Geo4D raw corrected Pi3 `Auc_30=35.22`
is below VWM's `Auc_30=81.72`. The best fixed convention transform reaches
`Auc_30=54.11`. A very unfair per-sequence GT-assisted camera-basis oracle
reaches `Auc_30=85.33`, which should be read only as an upper-bound diagnostic,
not as a benchmark result.

## OOD Dataset Manifests

DL3DV-Evaluation uses the local COLMAP `images.bin`/`cameras.bin` annotations
under the Gaussian Splat folders. COLMAP poses are read as world-to-camera and
converted to the shared manifest contract:

```text
pi3_pairwise_benchmark/outputs/datasets/dl3dv_eval/manifest_50view_50seq.jsonl
```

Processed zero-shot 7Scenes and TUM use `gt_cameras.npz` where `extrinsics`
are camera-to-world; RGB frames are extracted from `gt_rgb.mp4` into
`outputs/datasets/zeroshot_frames`:

```text
pi3_pairwise_benchmark/outputs/datasets/7scenes/manifest_50view_50seq.jsonl
pi3_pairwise_benchmark/outputs/datasets/tum/manifest_50view_50seq.jsonl
```

Current counts: DL3DV-Eval has 50 50-frame sequences, 7Scenes has 50, and TUM
has 20. Bonn is present in `processed_zeroshot` but is not included for Pi3
pose metrics because that processed copy has no `gt_cameras.npz`.

Raw TUM RGB-D is also supported through timestamp association between
`rgb.txt` and `groundtruth.txt`; this is separate from the 20-row processed
zero-shot TUM subset:

```text
pi3_pairwise_benchmark/outputs/datasets/tum_rgbd/manifest_50view_50seq.jsonl
```

OOD 50-view results use one inference per 50-frame sequence and all
`C(50,2)=1225` relative-pose pairs per sequence:

```text
DL3DV-Eval Pi3, 50 sequences:   Racc_5=98.66, Tacc_5=96.10, Auc_30=96.27
DL3DV-Eval VWM, 50 sequences:   Racc_5=77.02, Tacc_5=27.08, Auc_30=67.55
DL3DV-Eval Geo4D, 50 sequences: Racc_5=64.64, Tacc_5=13.34, Auc_30=49.24
7Scenes Pi3, 50 sequences:      Racc_5=89.46, Tacc_5=61.22, Auc_30=80.15
7Scenes VWM, 50 sequences:      Racc_5=29.23, Tacc_5=6.38,  Auc_30=37.29
7Scenes Geo4D, 50 sequences:    Racc_5=34.59, Tacc_5=4.24,  Auc_30=28.63
TUM RGB-D Pi3, 50 sequences:    Racc_5=99.84, Tacc_5=45.27, Auc_30=61.82
TUM RGB-D VWM, 50 sequences:    Racc_5=54.56, Tacc_5=6.57,  Auc_30=29.17
TUM RGB-D Geo4D, 50 sequences:  Racc_5=55.21, Tacc_5=1.52,  Auc_30=12.53
```

OOD raw-vs-oracle tables are written to:

```text
pi3_pairwise_benchmark/outputs/ood_50view_raw_vs_oracle_tables.md
pi3_pairwise_benchmark/outputs/ood_50view_raw_vs_oracle_tables.json
pi3_pairwise_benchmark/outputs/tum_rgbd_raw_vs_oracle_tables.md
pi3_pairwise_benchmark/outputs/tum_rgbd_raw_vs_oracle_tables.json
```

Geo4D's own aligned ATE/RPE metric can also be run on saved predictions:

```bash
/n/home01/rcai/.conda/envs/psamp/bin/python \
  pi3_pairwise_benchmark/scripts/eval_geo4d_saved_with_official_metrics.py \
  --manifest pi3_pairwise_benchmark/outputs/datasets/tum_rgbd/manifest_50view_50seq.jsonl \
  --prediction-dir pi3_pairwise_benchmark/outputs/datasets/tum_rgbd/geo4d_50view_s50_s5 \
  --output-dir pi3_pairwise_benchmark/outputs/datasets/tum_rgbd/geo4d_50view_s50_s5/official_geo4d_metrics
```

The current saved TUM RGB-D 50-frame windows score `ATE=0.10393`,
`RPE trans=0.03259`, `RPE rot=2.02924` under Geo4D's own `eval_metrics`
implementation. Geo4D Table 2 reports TUM-dynamics `ATE=0.073`,
`RPE trans=0.020`, `RPE rot=0.635`, but that uses first 90 frames with temporal
stride 3, so this is a sanity check rather than a strict reproduction.
Report:

```text
pi3_pairwise_benchmark/outputs/datasets/tum_rgbd/geo4d_50view_s50_s5/official_geo4d_metrics/geo4d_official_metrics_summary.md
```

These include 50-sequence 50-view rows plus 50-sequence 2-view long-baseline
rows. The 2-view rows use 2 deterministic long-baseline pairs per sequence, so
each method/dataset has 100 evaluated pairs.

The first OOD Geo4D runs appeared stalled during model load; diagnostics showed
the slow section was OpenCLIP checkpoint `load_state_dict`. The Slurm wrapper
now uses a per-job `MPLCONFIGDIR` and the adapter emits repeated faulthandler
stack dumps during Geo4D model load. DL3DV-Eval Geo4D and 7Scenes Geo4D both
completed after that diagnostic patch.

Current 50-sequence Re10K expansion source:

```text
/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke_50view/13198546/outputs/manifest_50view.jsonl
```

Official Pi3 50-view on these 50 sequences: `Racc_5=100.00`,
`Tacc_5=79.54`, `Auc_30=87.14` over `61250` relative pairs. Saved Pi3 cameras
are written under `pi3_50view_save`, and Pi3 2-view long-baseline over 100
pairs is complete with `Tacc_5=89.00`, `Auc_30=91.37`.

Re10K 50-sequence results completed so far:

```text
Pi3 50-view:       Racc_5=100.00, Tacc_5=79.54, Auc_30=87.14, pairs=61250
VWM 50-view:       Racc_5=97.16,  Tacc_5=22.56, Auc_30=68.87, pairs=61250
Geo4D 50-view:     Racc_5=98.98,  Tacc_5=1.13,  Auc_30=25.30, pairs=61250
Pi3 2-view long:   Racc_5=100.00, Tacc_5=89.00, Auc_30=91.37, pairs=100
VWM 2-view long:   Racc_5=94.00,  Tacc_5=23.00, Auc_30=62.30, pairs=100
Geo4D 2-view long: Racc_5=69.00,  Tacc_5=4.00,  Auc_30=23.40, pairs=100
```

Raw-vs-oracle tables for these same saved predictions are written to:

```text
pi3_pairwise_benchmark/outputs/re10k_50seq_raw_vs_oracle_tables.md
pi3_pairwise_benchmark/outputs/re10k_50seq_raw_vs_oracle_tables.json
```

The oracle rows apply a GT-assisted per-sequence SO3 camera-basis fit from
predicted relative translation directions to GT directions. They are an
upper-bound diagnostic, not a fair benchmark result.

50-view long-baseline comparison visualization:

```text
pi3_pairwise_benchmark/outputs/pi3_ours_50view_long_comparison_13135511/index.html
```

Comparison visualizations across Pi3 and ours can be built from saved pairwise
prediction directories:

```bash
PYTHONPATH=pi3_pairwise_benchmark \
python pi3_pairwise_benchmark/scripts/build_pairwise_comparison_visualizations.py \
  --manifest /n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke/12967573/outputs/manifest.jsonl \
  --pi3-dir /n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke/12967573/outputs/pi3_2view \
  --ours-dir /n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke/12967573/outputs/ours_2view_seq1_s40_pairs45_retry \
  --output-dir pi3_pairwise_benchmark/outputs/pi3_ours_comparison_12967573
```
