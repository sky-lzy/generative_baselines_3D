# standardized_eval — home-field depth + pose evaluation, reproducible

Self-contained, intuitive reproduction of the JULY3 home-field evaluation campaign:
**our video-diffusion models** (RGB → camera pose + depth) vs **PI3 / DA3 / Geo4D**, scored with
the **π³ metric code verbatim** (evo Sim(3) ATE/RPE for pose; scale/affine/LADS-aligned AbsRel/δ1
for depth) on **deterministic, byte-identical frames** for every method.

**Verified reproduction.** Both halves were checked against the original campaign artifacts:
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
├── inference/              # verbatim inference scripts, one folder per method family
│   ├── ours/               #   run_ours_pi3.py (5 home-field ds + RE10K-50), run_eval_scannetpp.py
│   ├── pi3/                #   run_pi3_c50.py (official Pi3, first-50 frames), run_pi3_re10k50.py
│   └── da3_geo4d/          #   run_baseline_pi3.py, run_baseline_scannetpp.py
├── evaluation/             # verbatim scorers + the π³ metric code
│   ├── pi3_metrics/        #   md5-identical copy of Pi3_depthpose relpose/evo_utils.py + utils/depth.py
│   ├── eval_ours_pi3.py    #   pose (+π³ scale-aligned depth) for ours
│   ├── eval_baseline_pi3.py#   same for DA3/Geo4D
│   ├── score_depth_hf.py   #   headline depth: scale / affine-LSQ / LADS(L1) per-video alignment
│   ├── eval_re10k_dist50.py, eval_scannetpp_pose.py, align_ablation.py
├── tools/apply_patches.py  # the ONLY diffs vs the source repo (path shims), auditable + idempotent
├── preds/  results/  logs/ # outputs (roots configurable in paths.yaml)
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
- `run.skip_existing` — a step is skipped iff its result CSV already exists **and contains real
  values** (empty/header-only CSVs never count)
- every key is CLI-overridable via OmegaConf dotlist (see examples above)

Task ↔ dataset semantics (mirrors the campaign):
pose = sintel, tum, scannetv2, re10k50, scannetpp · depth = sintel, bonn, kitti, scannetpp.
`bonn`/`kitti` pose CSVs carry the π³ scale-aligned depth columns (that's how the harness scores
them); the headline depth numbers are the LADS rows in `results/depth_ablation/`.

## Where results land

| what | file |
|---|---|
| pose ATE/RPE (ours + baselines) | `results/pose/<tag>_<ds>.csv` (`AVERAGE(meanseq)` row) |
| headline depth (scale/affine/LADS) | `results/depth_ablation/<tag>_<ds>.csv` (row `lads`) |
| RE10K-50 zero-shot pose | `results/re10k50/<tag>_re10k.csv` |
| ScanNet++ zero-shot depth / pose | `results/scannetpp_depth|scannetpp_pose/<tag>_*.csv` |

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

## Adding a new method (humans)

1. Ours checkpoint: add a block to `configs/methods.yaml` (copy any 5B entry; set `tag` + `ckpt`;
   add `scale_flags` **only** for global-metric-trained models, e.g. D). Enable it in
   `config.yaml` → done: the runner covers all datasets/tasks automatically.
2. A brand-new external baseline: add an inference script under `inference/<name>/` that writes
   the standard preds layout (`<preds>/<name>/<dataset>/<seq>/pred_c2w.npy` +
   `pred_depth_metric.npy` + `frames.json`, using the same frame indices — see
   `run_baseline_pi3.py` for the frame-parity pattern), then add a `jobs_for_*` branch in
   `runner.py` (~20 lines, mirror `jobs_for_da3_geo4d`). The scorers need no changes.

## For agents (parallelization etc.)

- `run.dry_run=true` prints every command with its env — lift them into sbatch/queue harnesses
  verbatim. One runner per GPU with disjoint `select` splits is the simplest scale-out.
- All child-process env is set by the runner (`VWM_REPO`, `PI3_ROOT`, `PI3_METRICS`,
  `GEN3D_ROOT`, `GEO4D_DIR`, `RE10K_*`, `HF_HOME`, `CUDA_VISIBLE_DEVICES`, `INFER_DIT_BF16`);
  child cwd is `paths.vwm_repo`.
- Long-pole sharding: `inference/ours/run_ours_pi3.py --seqs a,b,c` restricts to a seq subset and
  skips already-complete seqs — disjoint shards may write the same preds dir; score once.
- Gotchas (learned the hard way): DA3 needs compute capability ≥ 9.0 (H100/H200). Geo4D is
  diffusion-based (slowest baseline). PI3's SVD needs cuSOLVER workspace — it dies with
  `CUSOLVER_STATUS_INTERNAL_ERROR` on a GPU whose memory is almost fully reserved by another
  process. `mamba run … python3 -` (stdin heredoc) swallows the child's stdout — always run
  script *files*.

## Verbatim policy

Every file under `inference/` and `evaluation/` is a byte-copy of the campaign script it
reproduces, **except** the path shims applied by `tools/apply_patches.py` (repo discovery via
`VWM_REPO` env instead of `__file__`, hydra `initialize_config_dir`, and env-overridable data
roots — originals as defaults). Re-run it any time (idempotent); `MANIFEST.md` maps each copy to
its source. The π³ metric files in `evaluation/pi3_metrics/` are **md5-identical** to
`pi3_eval/evaluation/Pi3_depthpose` — the metric code is untouched.
