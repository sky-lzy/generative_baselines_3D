# nvs/ — home-field NVS evaluation, reproducible

Self-contained reproduction of the home-field **novel-view synthesis** evaluation (July
center-50 campaign): condition **our video-diffusion models** on **two views (first + last
frame)** plus the full camera raymap sequence, generate the 48 interior frames, and score
**PSNR against ground-truth frames** on 4 scene sets. Fully additive to this repo — nothing in
the depth+pose benchmark is imported or modified.

**Protocol (the shared setting):**

- **Datasets** — center-50 *consecutive* frames per test video (small-baseline, matches
  training clip structure), reconfusion-format scene dirs under `paths.scenes_root`:
  `dl3dv` (DL3DV-Evaluation, 20 scenes) · `spatialvid` (10) · `mip` (Mip-NeRF360, 7) ·
  `re10k_c50` (RE10K, 10). Built by `vwm_repo/eval_svc/build_homefield_*.py` (pose convention
  verified: relative-extrinsic max-err ≈ 0 vs the known-good SVC-official scene sets).
- **Conditioning** — `ncf=2`: RGB of frames 0 and T−1 + Plücker raymaps of ALL frames
  (scale-normalized per the model's training scheme); the model generates the interior frames.
  (`ncf=1` = single-view extension, opt-in via `run.ncf=[2,1]`.)
- **Metric** — per-scene mean PSNR over the **generated interior frames only** (frames 0/T−1
  are inputs), computed by the inference engine on raw tensors *before* mp4 compression and
  saved per scene (`sample_XXXXX/rgb_metrics.json`); dataset score = unweighted scene mean.
- **Sampling** — 40 diffusion steps, seed 42, hist_guidance 1.0, no augmentations,
  `INFER_DIT_BF16=1` (bf16 DiT), 288×384 (5B) / 240×320 (1.3B).

The inference command is **field-for-field identical** to the campaign runner
(`vwm_repo/eval_svc/hf_homefield.sbatch`); the engine is the source repo's
`scripts/inference_single_gpu_ray_to_rgb_depth_eval.py` with the `svc_scenes` loader,
invoked with `cwd=vwm_repo` — no model/loader code is duplicated here.

**Verified reproduction** (evaluation stage, 2026-07-27): re-scoring the campaign's existing
predictions through `evaluation/score_nvs_psnr.py` reproduced the campaign aggregates
(`final_stats.json` / `REPORT_HOMEFIELD_CENTER50.html`) to **float64 exactness**
(max |diff| = 3.6e-15 dB) on all 16 checked cells — `s5bF_20000` and `paper_1p3b` ×
4 datasets × {ncf2, ncf1}:

| PSNR (dB) | dl3dv n2 | spatialvid n2 | mip n2 | re10k n2 | dl3dv n1 | spatialvid n1 | mip n1 | re10k n1 |
|---|---|---|---|---|---|---|---|---|
| s5bF_20000 | 14.43 | 15.03 | 11.84 | 22.64 | 12.00 | 13.07 | 11.01 | 17.24 |
| paper_1p3b | 12.45 | 13.62 | 11.57 | 20.82 | 10.98 | 12.69 | 11.00 | 14.75 |

The scorer hard-fails on any >1e-6 dB mismatch vs a complete cell's `final_stats.json`, and a
cell missing any scene never writes a CSV (count-aware, same convention as the depth+pose
runner's marker checks). Inference reproduction: the command is verbatim and seeded (seed 42);
bitwise-identical samples additionally require the same GPU architecture/dtype kernels
(campaign: H100/H200, bf16).

```
nvs/
├── run_nvs.py                    # THE entry point — expands the enabled matrix into commands
├── configs/
│   ├── nvs_config.yaml           # enable/disable methods x datasets; ncf protocol; run options
│   ├── nvs_paths.yaml            # vwm repo + scene roots + where preds/results are written
│   └── nvs_methods.yaml          # method registry (ckpt/algorithm/resolution/scale overrides)
├── inference/run_nvs_infer.py    # campaign-verbatim command builder (count-aware SKIP/REDO)
├── evaluation/score_nvs_psnr.py  # per-scene CSV + AVERAGE from the engine's rgb_metrics.json
├── preds/  results/  logs/       # outputs (roots configurable in nvs_paths.yaml)
```

## Quickstart

Everything runs in the `test2` env; inference needs one 80GB GPU (H100/H200):

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga/world_model_4d/standardized_eval
mamba run -n test2 python3 nvs/run_nvs.py run.dry_run=true   # see the exact commands, run nothing
mamba run -n test2 python3 nvs/run_nvs.py                    # run the enabled matrix on gpu 0
mamba run -n test2 python3 nvs/run_nvs.py run.gpu=2          # ... on gpu 2
mamba run -n test2 python3 nvs/run_nvs.py run.stages=[evaluation]  # score existing preds only
mamba run -n test2 python3 nvs/run_nvs.py run.only_method=s5bF_20000 run.only_dataset=dl3dv
# re-score the original campaign predictions (no GPU):
mamba run -n test2 python3 nvs/run_nvs.py run.stages=[evaluation] \
    preds.nvs=<vwm_repo>/eval_svc/hf_runs run.ncf=[2,1]
```

What runs is controlled entirely by `nvs/configs/nvs_config.yaml`:

- `select.methods.*: true|false` — which checkpoints (tags from `nvs_methods.yaml`)
- `select.datasets.*` — dl3dv / spatialvid / mip / re10k_c50
- `run.ncf` — `[2]` = the shared two-view first+last protocol; `[2,1]` adds single-view
- `run.stages` — `[inference, evaluation]`, or either alone
- `run.only_method`, `run.only_dataset` — optional exact selectors for one cell
- `run.skip_existing` — a step is skipped iff its result CSV already exists with real values;
  inference itself is count-aware (complete = `final_stats.json` count == #scene dirs; a
  partial cell from an interrupted run is re-run, never skipped)
- every key is CLI-overridable via OmegaConf dotlist (see examples above)

Outputs: `nvs/preds/<tag>__<ds>__ncf<k>/sample_XXXXX/{pred_rgb.mp4, gt_rgb.mp4,
rgb_metrics.json, ...}` and `nvs/results/<tag>__<ds>__ncf<k>.csv` (`scene,psnr` rows +
`AVERAGE`; sample index = sorted scene-dir order).

## Evaluating YOUR model

1. Add a block to `nvs/configs/nvs_methods.yaml` (copy `s5bF_20000`, change tag + ckpt).
   `kind: ours_5b` covers any checkpoint of the `wan_t2v_ray_depth_mot_concat_5b` algorithm at
   288×384; override `algorithm`/`height`/`width` per entry if yours differs.
2. **Match your training raymap-scale convention** via `dataset_overrides` — the test-time
   conditioning must be encoded the way the model was trained or the input is
   out-of-distribution: per-clip (default, no key) · global-metric
   (`scale_mode=global_metric global_metric_scale=10.0`) · parallax companded-origin (see the
   `s5bC_12500` entry) · fixed P90 constant (`mutual_scale=<const>`).
3. Enable it: `select.methods.<tag>=true` (or `run.only_method=<tag>`) and run. GPU cost:
   ~47 two-view scenes ≈ a few hours on one H100 (≈4 min/scene at 40 steps).

Historical note: the full campaign matrix (these two models + the 5 rebuttal-1.3B ablations +
SVC/SEVA + AetherV1 baselines) lives in `vwm_repo/eval_svc/REPORT_HOMEFIELD_CENTER50.html`;
baseline numbers were produced by `eval_svc/svc_homefield.sbatch` (SEVA's own `demo.py
img2img`) and `eval_svc/run_aether_nvs.py`. The `paper_1p3b` checkpoint was purged from
netscratch after the campaign; its predictions still re-score (table above).
