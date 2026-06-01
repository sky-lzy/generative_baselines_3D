# Pi3 Paper Benchmark Reproduction Plan

This document tracks the exact Pi3 paper benchmarks before we adapt the same
settings to VWM and Geo4D. It intentionally separates paper numbers, official
code paths, dataset/input settings, local readiness, and our reproduction
status.

Sources:

- Paper: <https://arxiv.org/abs/2507.13347>
- Official code checkout used locally:
  `pi3_pairwise_benchmark/external/Pi3`
- Local checkpoint:
  `pi3_pairwise_benchmark/external/Pi3/checkpoints/Pi3/model.safetensors`
- Official eval overview:
  `pi3_pairwise_benchmark/external/Pi3/README.md`

## Current Status

Pi3's cached checkpoint is present locally. The official `external/Pi3/data`
folder is not populated, so exact paper reproduction still requires preparing
or symlinking the datasets into Pi3's expected layouts. Re10K is the furthest
along because we already have a conversion path from the local VWM-format Re10K
copy into Pi3's expected Re10K layout.

Existing reproduction evidence is only a smoke check, not a paper-scale
reproduction:

| Task | Dataset | Paper Pi3 number | Our current reproduction |
|---|---|---:|---:|
| Camera pose angular | Re10K, 10 random views | RRA30 99.99 / RTA30 95.62 / AUC30 85.90 | 3-sequence smoke: RRA30 100.00 / RTA30 100.00 / AUC30 98.40 |

The custom 50-view / 2-view tables already produced in this repo are useful
for our benchmark design, but they are not Pi3-paper reproduction runs.

## 2026-05-28 Official Reproduction Runs

User-requested dataset scope: RealEstate10K, Sintel, TUM-dynamics, 7-Scenes,
DTU, ETH3D, Bonn, KITTI.

Submitted official Pi3 runs:

| Slurm job | Official Pi3 task | Datasets | Status | Output root |
|---|---|---|---|---|
| `16657106` -> `16701810` | `relpose/eval_angle.py` | RealEstate10K | completed after one-sequence local mp4 metadata repair | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/re10k_10view_official_20260528` |
| `16660386` | `relpose/eval_dist.py` | Sintel, TUM-dynamics | failed: missing `rootutils` dependency | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/relpose_distance_sintel_tum_20260528` |
| `16660433` | `videodepth/infer.py` + `videodepth/eval.py` | Sintel, Bonn, KITTI | failed: missing `rootutils` dependency | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/videodepth_sintel_bonn_kitti_20260528` |
| `16660484` | `monodepth/infer.py` + `monodepth/eval.py` | Sintel, Bonn, KITTI | failed: missing `rootutils` dependency | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/monodepth_sintel_bonn_kitti_20260528` |
| `16661246` | `relpose/eval_dist.py` | Sintel, TUM-dynamics | completed; matches paper closely | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/relpose_distance_sintel_tum_20260528` |
| `16661314` -> `16667888` | `videodepth/infer.py` + `videodepth/eval.py` | Sintel, Bonn, KITTI | completed after import-path retry | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/videodepth_sintel_bonn_kitti_20260528` |
| `16661409` | `monodepth/infer.py` + `monodepth/eval.py` | Sintel, Bonn, KITTI | completed; matches paper closely | `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/monodepth_sintel_bonn_kitti_20260528` |

Current blockers for exact point-map reproduction:

| Dataset | Pi3 paper task | Local path checked | Blocker |
|---|---|---|---|
| 7-Scenes | `mv_recon/eval.py` | `/n/netscratch/kempner_rcai_lab/Everyone/datasets/7scenes` | Pi3 class expects `frame-XXXXXX.depth.proj.png`; local tree has `frame-XXXXXX.depth.png` and no `.depth.proj.png`. Need the Spann3R/CUT3R-style projected-depth preparation or confirmation that symlinking is valid. |
| DTU | `mv_recon/eval.py` | `/n/netscratch/kempner_rcai_lab/Everyone/datasets/dtu` | Pi3 class expects per-scan `depths/` and `binary_masks/`; local tree currently shows `images/`, `cams/`, and `pair.txt` only. |
| ETH3D | `mv_recon/eval.py` | `/n/netscratch/kempner_rcai_lab/Everyone/datasets/eth3d` | Pi3 class expects `images/custom_undistorted`, `ground_truth_depth/custom_undistorted`, and `custom_undistorted_cam`; local path has permission-denied `ground_truth_depth` and no visible `custom_undistorted_cam`. |

I am not submitting the point-map jobs until those layout issues are resolved,
because a failed run would not be an exact reproduction.

## Official Evaluation Code Map

| Paper task | Official script | Config | Main data config |
|---|---|---|---|
| Camera pose, angular metrics | `relpose/eval_angle.py` | `configs/evaluation/relpose-angular.yaml` | `configs/data/relpose-angular.yaml` |
| Camera pose, distance metrics | `relpose/eval_dist.py` | `configs/evaluation/relpose-distance.yaml` | `configs/data/relpose-distance.yaml` |
| Point map / multi-view reconstruction | `mv_recon/eval.py` | `configs/evaluation/mv_recon.yaml` | `configs/data/mv_recon.yaml` |
| Video depth | `videodepth/infer.py`, `videodepth/eval.py` | `configs/evaluation/videodepth.yaml` | `configs/data/depth.yaml` |
| Monocular depth | `monodepth/infer.py`, `monodepth/eval.py` | `configs/evaluation/monodepth.yaml` | `configs/data/depth.yaml` |

## Camera Pose: Angular Metrics

Paper table: Table 1. Appendix Table 9 also reports tighter thresholds on
Re10K.

Protocol from Appendix A.5 and official config:

- datasets: RealEstate10K and CO3Dv2
- input: 10 randomly sampled images per sequence, seed 42 in official config
- pair construction: all unordered pairs among the 10 images
- metric: relative rotation accuracy, relative translation accuracy, and AUC
  over angular errors; Table 1 reports threshold 30 degrees
- official config: `load_img_size=512`, `no_crop=False`

| Dataset | Type | Scene/content | Input setting | Official code dataset key | Paper Pi3 result |
|---|---|---|---|---|---|
| RealEstate10K | real video-derived multiview | mostly static real estate scenes; indoor/outdoor; camera trajectory from video | sparse/disconnected 10-view sample from each video sequence; all 45 pairs | `Re10K`; `datasets/sequences/re10k_test_1719.txt`; `Re10K_relpose_seq-id-map_seed42.json` | RRA30 99.99 / RTA30 95.62 / AUC30 85.90 |
| CO3Dv2 | object-centric multiview image/video captures | real object-centric scenes, mostly static object/camera captures; seen during training | sparse/disconnected 10-view sample per sequence; all 45 pairs | `CO3Dv2`; all categories; `CO3Dv2_relpose_seq-id-map_seed42.json` | RRA30 99.05 / RTA30 97.33 / AUC30 88.41 |

Re10K tighter threshold paper row for Pi3:

| Threshold | RRA | RTA | AUC |
|---:|---:|---:|---:|
| 1 deg | 85.19 | 27.57 | 24.87 |
| 3 deg | 97.56 | 65.57 | 47.28 |
| 5 deg | 98.83 | 78.32 | 58.63 |
| 10 deg | 99.63 | 88.69 | 72.11 |
| 15 deg | 99.86 | 92.02 | 78.39 |

Local exact reproduction status:

- Re10K: smoke output:
  `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/smoke/12967573/outputs/pi3_10view/Re10K-metric.csv`
- Re10K full exact reproduction completed on 2026-05-28:
  - failed attempt: `16656897`, missing `python` on batch `PATH`
  - prep attempt: `16657106`, completed 1718/1719 sequences, then exited before
    eval because one local mp4 failed metadata extraction
  - one-sequence CPU repair: `16701393`, completed after adding a metadata
    fallback in the local Re10K preparation script
  - eval-only resume: `16701810`, completed on `kempner` A100
  - output root:
    `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/re10k_10view_official_20260528`
  - setting manifest:
    `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/re10k_10view_official_20260528/outputs/manifest.jsonl`
  - metric CSV:
    `/n/netscratch/kempner_rcai_lab/Lab/video_model/test-time/pi3_pairwise_benchmark/pi3_paper_repro/re10k_10view_official_20260528/outputs/pi3_10view/Re10K-metric.csv`
  - reproduced metrics:
    `Racc_5=98.812`, `Tacc_5=80.013`, `Auc_5=62.748`,
    `Racc_15=99.841`, `Tacc_15=92.436`, `Auc_15=80.151`,
    `Racc_30=99.997`, `Tacc_30=95.645`, `Auc_30=87.307`
  - interpretation: matches the paper row closely on `Racc_30/Tacc_30`
    (`99.99/95.62`) and is slightly higher than the documented paper `Auc_30`
    (`85.90`).
- CO3Dv2: not run locally yet. Needs data in Pi3 layout:
  `data/co3dv2/data` and `data/co3dv2/co3d_v2_annotations`.

## Camera Pose: Distance Metrics

Paper table: Table 1.

Protocol from Appendix A.5 and official config:

- datasets: Sintel, TUM-dynamics, ScanNetv2
- input: continuous sequence frames, `pose_eval_stride=1`
- metric: ATE, RPE translation, RPE rotation
- alignment: predicted trajectories are Sim(3)-aligned to GT before metrics
- official config: `load_img_size=512`, `no_crop=False`

| Dataset | Type | Scene/content | Input setting | Official code dataset key | Paper Pi3 result |
|---|---|---|---|---|---|
| Sintel | synthetic video | dynamic synthetic outdoor/movie scenes | continuous sequence frames from 14 listed sequences | `sintel`; `data/sintel/training/final/{seq}` and `camdata_left/{seq}` | ATE 0.074 / RPE-t 0.040 / RPE-r 0.282 |
| TUM-dynamics | real RGB-D video | dynamic indoor RGB-D sequences | prepared `rgb_90`, `depth`, `groundtruth_90.txt`; continuous 90-frame trajectory | `tum`; `data/tum/{seq}` | ATE 0.014 / RPE-t 0.009 / RPE-r 0.312 |
| ScanNetv2 | real RGB-D video | static indoor room scans; seen/related train distribution | prepared `color_90`, `depth_90`, `pose_90.txt`; continuous 90-frame trajectory | `scannetv2`; `data/scannetv2/{seq}` | ATE 0.031 / RPE-t 0.013 / RPE-r 0.347 |

Local exact reproduction status:

- none complete.
- We ran a separate custom TUM RGB-D Pi3-style angular benchmark, but that is
  not the paper's TUM-dynamics distance-metric protocol.

## Point Map / Multi-View Reconstruction

Paper tables: Table 2, Table 3, Appendix Table 10.

Protocol from paper and official code:

- predicted point maps are aligned to GT using Umeyama Sim(3), then refined
  with ICP
- metrics: Accuracy, Completion, Normal Consistency; Appendix reports Chamfer
  Distance as mean of Accuracy and Completion
- official config: `load_img_size=518`, `no_crop=True`

| Dataset | Type | Scene/content | Input setting | Official code dataset key | Paper Pi3 result |
|---|---|---|---|---|---|
| 7-Scenes sparse | RGB-D video-derived multiview | static indoor scenes | stride keyframes, `kf_every=200` | `7scenes-sparse` | Acc mean/med 0.047/0.029; Comp 0.075/0.049; NC 0.742/0.841 |
| 7-Scenes dense | RGB-D video-derived multiview | static indoor scenes | stride keyframes, `kf_every=40` | `7scenes-dense` | Acc 0.016/0.007; Comp 0.022/0.011; NC 0.689/0.792 |
| NRGBD sparse | RGB-D video-derived multiview | static indoor neural RGB-D reconstruction sequences | stride keyframes, `kf_every=500` | `NRGBD-sparse` | Acc 0.026/0.015; Comp 0.028/0.014; NC 0.916/0.992 |
| NRGBD dense | RGB-D video-derived multiview | static indoor neural RGB-D reconstruction sequences | stride keyframes, `kf_every=100` | `NRGBD-dense` | Acc 0.015/0.008; Comp 0.013/0.005; NC 0.898/0.987 |
| DTU | object-centric MVS images | static object/tabletop captures | stride keyframes, `kf_every=5` | `DTU` | Acc 1.198/0.646; Comp 1.849/0.607; NC 0.678/0.768 |
| ETH3D | scene-level MVS images/videos | static indoor/outdoor scenes | stride keyframes, `kf_every=5` | `ETH3D` | Acc 0.194/0.131; Comp 0.210/0.128; NC 0.883/0.969 |

Appendix CD row for Pi3:

| Dataset setting | CD mean | CD med |
|---|---:|---:|
| 7-Scenes sparse | 0.061 | 0.039 |
| 7-Scenes dense | 0.019 | 0.009 |
| NRGBD sparse | 0.026 | 0.013 |
| NRGBD dense | 0.013 | 0.006 |
| DTU | 1.472 | 0.626 |
| ETH3D | 0.199 | 0.128 |

Local exact reproduction status:

- none complete.
- We have custom Pi3 angular pose runs on processed 7Scenes, but those are not
  the paper's point-map/ICP reconstruction protocol.

## Video Depth

Paper table: Table 4.

Protocol from paper and official code:

- datasets: Sintel, Bonn, KITTI
- input: continuous video sequences
- inference script writes one predicted depth per frame
- evaluation aligns each video sequence to GT with scale and shift in the
  default official config (`align=scale&shift`)
- metrics: Abs Rel and depth accuracy threshold; FPS reported on KITTI using
  one A800 GPU
- official config: `load_img_size=512`, `no_crop=True`

| Dataset | Type | Scene/content | Input setting | Official code dataset key | Paper Pi3 result |
|---|---|---|---|---|---|
| Sintel | synthetic video | dynamic synthetic scenes | continuous listed sequences | `sintel` | Abs Rel 0.233 / accuracy 0.664 |
| Bonn | real RGB-D video | dynamic indoor RGB-D sequences with people/objects | `rgb_110` and `depth_110` prepared sequences | `bonn` | Abs Rel 0.049 / accuracy 0.975 |
| KITTI | real driving video | outdoor autonomous driving | gathered validation video sequences | `kitti` | Abs Rel 0.038 / accuracy 0.986; FPS 57.4 |

Local exact reproduction status:

- none complete.

## Monocular Depth

Paper table: Table 5. Appendix Table 11 repeats the same Pi3 row while adding
Depth Anything V2.

Protocol from paper and official code:

- input: each image/depth map evaluated independently
- video datasets are flattened through their sequences; NYU-v2 is a mono image
  set
- official default uses median-scale alignment
- metrics: Abs Rel and depth accuracy threshold
- official config: `load_img_size=512`, `no_crop=True`

| Dataset | Type | Scene/content | Input setting | Official code dataset key | Paper Pi3 result |
|---|---|---|---|---|---|
| Sintel | synthetic video frames | dynamic synthetic scenes | per-frame monocular depth | `sintel` | Abs Rel 0.277 / accuracy 0.614 |
| Bonn | real RGB-D video frames | dynamic indoor RGB-D | per-frame monocular depth | `bonn` | Abs Rel 0.044 / accuracy 0.976 |
| KITTI | real driving frames | outdoor autonomous driving | per-frame monocular depth | `kitti` | Abs Rel 0.060 / accuracy 0.971 |
| NYU-v2 | real RGB-D images | static indoor scenes | per-image monocular depth | `nyu-v2` | Abs Rel 0.054 / accuracy 0.956 |

Local exact reproduction status:

- none complete.

## Reproduction Priority

1. Re10K angular pose full official reproduction.
   - Reason: easiest local path; code and checkpoint already validated.
   - Exact target: Table 1 and Table 9 Re10K Pi3 row.
   - Existing wrapper:
     `pi3_pairwise_benchmark/slurm/re10k_pose_full_gpu.sbatch`
   - Must run with `RUN_OURS=0` to avoid VWM pairwise inference in the same
     job.

2. CO3Dv2 angular pose.
   - Reason: completes angular pose table.
   - Blocker: prepare/symlink CO3Dv2 official data and annotations into Pi3
     layout.

3. Sintel/TUM-dynamics/ScanNet distance pose.
   - Reason: validates trajectory alignment and evo-style pose metrics.
   - Blocker: prepare official MonST3R/CUT3R-style folders, especially TUM
     `groundtruth_90.txt` and ScanNet `pose_90.txt`.

4. Video depth and monodepth.
   - Reason: checks whether Pi3's depth eval path is reproducible before VWM
     adaptation.
   - Blocker: prepare Sintel/Bonn/KITTI/NYU-v2 depth folders exactly as Pi3
     expects.

5. Point-map reconstruction.
   - Reason: most expensive and most different from VWM's current pose-centric
     benchmark.
   - Blocker: prepare 7-Scenes, NRGBD, DTU, ETH3D pointcloud/depth caches and
     verify Open3D/ICP output.

## Exact Reproduction Criteria

A run should only be called an exact Pi3 paper reproduction if all of the
following hold:

- uses `yyfz233/Pi3` or the local mirror of that checkpoint
- uses the official Pi3 eval script for that task without metric changes
- uses the official Pi3 sampling file or regenerates it with the documented
  seed/config
- uses the same dataset split and preprocessing expected by Pi3's config
- reports the same metric family and alignment mode as the paper table
- records Slurm job id, command, output directory, and final CSV/JSON metrics

Runs that change sequence count, image count, frame stride, pair sampling, or
alignment are useful diagnostics, but should be reported separately from the
paper reproduction table.
