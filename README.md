# standardized_eval — home-field depth + pose evaluation, reproducible

Self-contained, intuitive reproduction of home-field evaluation:
**our video-diffusion models** (RGB → camera pose + depth) vs **PI3 / DA3 / Geo4D**, scored with
the **π³ metric code verbatim** (evo Sim(3) ATE/RPE for pose; scale/affine/LADS-aligned AbsRel/δ1
for depth) on **deterministic, byte-identical frames** for every method.

**Verified reproduction**:

- *Evaluation:* re-scoring the campaign's existing predictions through this codebase produced
**byte-identical CSVs** for ours (pose sintel, depth-ablation sintel, RE10K-50, ScanNet++ pose)
and baselines (PI3 sintel depth, DA3 sintel pose + depth).
- *Inference:* a fresh PI3 run on bonn (inference → scoring, GPU) reproduced the campaign CSVs
**byte-for-byte**.

```
standardized_eval/
├── runner.py               # THE entry point — expands the enabled matrix into commands
├── configs/
│   ├── config.yaml         # enable/disable methods x datasets x tasks; run options
│   ├── paths.yaml          # every data/code root + where preds/results are written
│   └── methods.yaml        # method registry (ckpt/algorithm/resolution/scale flags)
├── inference/              # campaign bridges, one folder per method family
│   ├── ours/               #   original path plus opt-in v2/shared-K/BA components
│   ├── pi3/                #   run_pi3_c50.py (official Pi3, first-50 frames), run_pi3_re10k50.py
│   └── da3_geo4d/          #   run_baseline_pi3.py, run_baseline_scannetpp.py
├── evaluation/             # campaign scorers + verbatim π³ metric code
│   ├── pi3_metrics/        #   md5-identical copy of Pi3_depthpose relpose/evo_utils.py + utils/depth.py
│   ├── eval_ours_pi3.py    #   pose (+π³ scale-aligned depth) for ours
│   ├── eval_baseline_pi3.py#   same for DA3/Geo4D
│   ├── score_depth_hf.py   #   headline depth: scale / affine-LSQ / LADS(L1) per-video alignment
│   ├── eval_re10k_dist50.py, eval_scannetpp_pose.py, align_ablation.py
├── tools/apply_patches.py  # archival path-shim helper, auditable + idempotent
├── preds/  results/  logs/ # outputs (roots configurable in paths.yaml)
├── nvs/                    # ADDITIVE: home-field NVS (two-view first+last) eval — see nvs/README.md
└── MANIFEST.md             # source → copy map with md5s
```

## Quickstart

Everything runs in the `test2` env:

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/standardized_eval
mamba run -n test2 python3 runner.py run.dry_run=true      # see the exact commands, run nothing
mamba run -n test2 python3 runner.py                        # run the enabled matrix on gpu 0
mamba run -n test2 python3 runner.py run.gpu=2              # ... on gpu 2
mamba run -n test2 python3 runner.py run.stages=[evaluation] # score existing preds only
mamba run -n test2 python3 runner.py select.methods.da3=false select.datasets.scannetv2=false
```

What runs is controlled entirely by `configs/config.yaml`:

- `select.methods.*: true|false` — which methods (tags from `configs/methods.yaml`)
- `select.datasets.*` — sintel / tum / scannetv2 / bonn / kitti (home-field),
re10k50 / scannetpp (zero-shot)
- `select.tasks.pose|depth`
- `run.stages` — `[inference, evaluation]`, or either alone
- `run.only_method`, `run.only_dataset` — optional exact selectors for one cell
- `run.skip_existing` — a step is skipped iff its result CSV already exists **and contains real
values** (empty/header-only CSVs never count)
- every key is CLI-overridable via OmegaConf dotlist (see examples above)

pose = sintel, tum, scannetv2, re10k50, scannetpp. 

depth = sintel, bonn, kitti, scannetpp.  

## NVS (novel-view synthesis) evaluation — `nvs/`

A self-contained, additive module reproducing the home-field **NVS** evaluation (two-view
first+last conditioning, PSNR on the generated interior frames) on **dl3dv / spatialvid /
mip (Mip-NeRF360) / re10k_c50** — center-50 consecutive frames per scene. Own runner + configs,
same conventions as this benchmark (OmegaConf matrix, count-aware skips, marker CSVs, logs/,
PASS/FAIL summary); nothing above is imported or modified. Evaluation-stage reproduction is
verified to float64 exactness against the July campaign. See `nvs/README.md`.

```bash
mamba run -n test2 python3 nvs/run_nvs.py run.dry_run=true   # the exact commands
mamba run -n test2 python3 nvs/run_nvs.py                    # run the enabled NVS matrix
```

## Optional S5bF components

The requested cumulative variants are registered for Sintel, Bonn, TUM, and
ScanNet++. They are opt-in, so adding them does not replace the benchmark's
historical default selection:

| method key | fine-tuned depth VAE v2 | one predicted shared K | verified sparse BA |
|---|---:|---:|---:|
| `s5bF_original` | no | no | no |
| `s5bF_depth_v2` | yes | no | no |
| `s5bF_depth_v2_shared_k` | yes | yes | no |
| `s5bF_depth_v2_shared_k_ba` | yes | yes | yes |

Run one method/dataset cell with the registered settings:

```bash
python3 runner.py run.only_method=s5bF_depth_v2_shared_k_ba \
  run.only_dataset=sintel run.stop_on_error=true
```

`run.only_method` and `run.only_dataset` select the requested cell even though
the component methods are disabled in the default matrix. The same components
can be enabled directly on either OURS inference entrypoint:

```bash
python inference/ours/run_ours_pi3.py ... \
  --depth_vae_ckpt /path/to/depth-vae-v2.ckpt \
  --shared_camera_intrinsics \
  --bundle_adjust
```

`--bundle_adjust` requires both the v2 depth checkpoint and
`--shared_camera_intrinsics`. With no component flags, the original inference
behavior is preserved.

The v2 decoder returns depth and its learned confidence. Confidence is saved
as `pred_depth_confidence.npy`, but it is not used to filter BA observations or
depth-evaluation pixels.

Shared-intrinsics recovery consumes predicted Plücker rays only. The verified
BA holds the shared intrinsics and dense decoder depth fixed, fixes frame 0 as
the gauge, and jointly optimizes the remaining camera poses and sparse 3D track
positions. The exact verified BA core requires Python 3.10 and is checksum
validated before use.

`bonn`/`kitti` pose CSVs carry the π³ scale-aligned depth columns (that's how the harness scores
them); the headline depth numbers are the LADS rows in `results/depth_ablation/`.

## Evaluating the normalization arms G and G2

G and G2 change the depth representation (log map instead of signed disparity) and the ray
channels (Plücker moments / s). The scorers dispatch on a per-scene `norm_meta.json` that
inference stamps into every preds dir, so the right inverse is applied automatically — the
legacy path is byte-for-byte unchanged when no meta is present.

```bash
# inference + scoring, one arm x one dataset (methods norm_G_ext / norm_G2_ext):
python runner.py run.only_method=norm_G_ext run.only_dataset=sintel run.stop_on_error=true
```

Scoring is self-contained: `vendor/` carries byte-identical copies of the three external
modules the decode needs (`eval_common_v3`, `geo4d_eval`, `datasets/_geometry_builder` +
`_crop_utils`), used only as a fallback when no VWM checkout / GEN3D_ROOT is present.
Verified: re-scoring the campaign's saved predictions with this branch reproduces the
published CSVs byte-identically (G sintel pose+depth, G2 bonn 3-alignment, and F as the
legacy-path regression check). Inference still requires the VWM repo (model code).

For NVS, `nvs/configs/nvs_methods.yaml` has config-ready entries `norm_G_24k` / `norm_G2_24k`
(same parallax pattern as `s5bC_12500`, `ray_encoding=moment`); no NVS reference numbers exist
for these arms yet.

## Where results land


| what                               | file                                                   |
| ---------------------------------- | ------------------------------------------------------ |
| pose ATE/RPE (ours + baselines)    | `results/pose/<tag>_<ds>.csv` (`AVERAGE(meanseq)` row) |
| headline depth (scale/affine/LADS) | `results/depth_ablation/<tag>_<ds>.csv` (row `lads`)   |
| RE10K-50 zero-shot pose            | `results/re10k50/<tag>_re10k.csv`                      |
| ScanNet++ zero-shot depth / pose   | `results/scannetpp_depth                               |


Ours home-field tags carry the `__contig_first` suffix (50 contiguous native frames).

## Reusing the campaign's existing predictions

Point the preds roots at the original dirs and run evaluation-only — no GPU needed:

```bash
mamba run -n test2 python3 runner.py run.stages=[evaluation] \
  preds.ours=$VWM/eval_pi3/preds_hf \
  preds.pi3=$VWM/eval_pi3/preds_hf_pi3 \
  preds.baselines=$VWM/eval_pi3/preds_baselines_hf
# VWM = .../world_model_4d/video_world_model_new
```

## Adding a new method 

1. Ours checkpoint: add a block to `configs/methods.yaml` (copy any 5B entry; set `tag` + `ckpt`;
  add `scale_flags` **only** for global-metric-trained models, e.g. D). Enable it in
   `config.yaml` → done: the runner covers all datasets/tasks automatically.
2. A brand-new external baseline: add an inference script under `inference/<name>/` that writes
  the standard preds layout (`<preds>/<name>/<dataset>/<seq>/pred_c2w.npy` +
   `pred_depth_metric.npy` + `frames.json`, using the same frame indices — see
   `run_baseline_pi3.py` for the frame-parity pattern), then add a `jobs_for_`* branch in
   `runner.py` (~20 lines, mirror `jobs_for_da3_geo4d`). The scorers need no changes.

## For agents (parallelization etc.)

- `run.dry_run=true` prints every command with its env — lift them into sbatch/queue harnesses
verbatim. One runner per GPU with disjoint `select` splits is the simplest scale-out.
- All child-process env is set by the runner (`VWM_REPO`, `PI3_ROOT`, `PI3_METRICS`,
`GEN3D_ROOT`, `GEO4D_DIR`, `RE10K_`*, `HF_HOME`, `CUDA_VISIBLE_DEVICES`, `INFER_DIT_BF16`);
child cwd is `paths.vwm_repo`.
- Long-pole sharding: `inference/ours/run_ours_pi3.py --seqs a,b,c` restricts to a seq subset and
skips already-complete seqs — disjoint shards may write the same preds dir; score once.
- Gotchas (learned the hard way): DA3 needs compute capability ≥ 9.0 (H100/H200). Geo4D is
diffusion-based (slowest baseline). PI3's SVD needs cuSOLVER workspace — it dies with
`CUSOLVER_STATUS_INTERNAL_ERROR` on a GPU whose memory is almost fully reserved by another
process. `mamba run … python3 -` (stdin heredoc) swallows the child's stdout — always run
script *files*.

## Verification policy

Every file under `inference/` and `evaluation/` retains its campaign behavior
except the documented path shims and optional OURS component integration. The
π³ metric files remain **md5-identical** to `pi3_eval/evaluation/Pi3_depthpose`.
`MANIFEST.md` maps each campaign copy to its source.
