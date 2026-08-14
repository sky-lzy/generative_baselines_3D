# Fair NVS benchmark — run log (2026-08-14)

Ours (F 5B, MoT 1.3B) vs SEVA on 128 RealEstate10K scenes, four conditioning regimes, with the
input pixels, the camera, the codec and the scorer all held identical. This file records what was
run, what was verified, and what was found wrong along the way — including the mistakes, because a
benchmark whose failures are undocumented is not one anybody should trust.

## Models

| tag | what | checkpoint | encoding |
|---|---|---|---|
| `F_5b` | 5B flagship | `outputs/s5b_de_jul15/s5bF_2500.ckpt` | default per-clip max-abs raymap normalization |
| `mot13b_full` | 1.3B MoT, full RE10K train | `mot13b_re10k_full_38755539/checkpoints/last.ckpt` | **raw** (`raymap_normalize=none`), 240×320, n_frames=50 |
| `seva` | Stable Virtual Camera | HF `stabilityai/stable-virtual-camera` | — |

`mot13b_full`'s convention was read from its own `.hydra/overrides.yaml`, not assumed:
`dataset.raymap_normalize=none`, `scale_preserving=false`, `height=240 width=320`,
`raymap_height=30 raymap_width=40`, `n_frames=50`, algorithm
`wan_t2v_ray_depth_mot_concat_rebuttal`. Feeding it the default max-abs normalization would be a
pure train/test encoding mismatch.

## Benchmarks (run in this order)

| phase | set | inputs | scored frames | scale sweep |
|---|---|---|---|---|
| p1 | `re10k128_50f` | frames 0, 49 | 1…48 | none (both at own default) |
| p2 | `re10k128_4dim` | frame 0 | 10,20,…,60 | 3 per side |
| p3 | `re10k128_4dim` | frames 0, 69 | 10,20,…,60 | 3 per side |
| p4 | `re10k128_50f` | frame 0 | 1…49 | 5 per side |

## Verified before spending GPU hours

**Geometry parity** (`evaluation/verify_geometry_parity.py`, run through SEVA's own
`seva.eval.transform_img_and_K`):

```
source K @ 384x288: fx 253.650  cx 192.000 cy 144.000     FOV 74.2478° x 59.1682°
SEVA @ 576x768    : fx 507.299  cx 384.000 cy 288.000     FOV 74.2478° x 59.1682°   (exactly 2x)
F_5b  @ 384x288   : fx 253.650  cx 192.000                FOV 74.2478° x 59.1682°
mot13b@ 320x240   : fx 211.375  cx 160.000                FOV 74.2478° x 59.1682°
crop: no-op (4:3 in, 4:3 out)     relative trajectories: max|diff| 0.0e+00
```

All three methods see the same scene content. `transform_img_and_K` selects its pixel-vs-normalized
K branch by testing whether cx,cy ∈ [0,1]; our cx = 192 is in pixels, so the unnormalized branch is
the one taken — confirmed, not inferred.

**Scorer parity** (`evaluation/test_score_fair.py`): prediction == GT scores PSNR 120.000 /
SSIM 1.000000 / LPIPS 0.000000 through *both* readers; the two reader paths differ by
max|Δ| = 0.000e+00 on byte-identical content.

**Scene set** (`data/build_fair_scenes.py --verify_only`): 128 scenes, cx,cy = 192.0,144.0,
pose drift 0.00e+00 vs source, ALL CHECKS PASSED.

## Sampler tuning (stage 0, 10 held-out scenes)

Neither side's guidance had ever been tuned on this data. Ours enters as

```
flow = flow_cond*(1 + lang + hist) − lang*flow_no_lang − hist*flow_no_hist
```

so `--hist_guidance` is an effective CFG scale of **(1 + hist)** on the conditioning axis, and the
engine default of 1.0 means every prior campaign number was sampled at guidance 2.0 — at twice the
model calls, since the negative branch is a second forward pass. Grid: hist ∈ {0, 0.5, 1, 2, 3},
plus a lang ∈ {0, 2} probe (expected inert: `svc_scenes` emits an empty caption). SEVA gets an
equally deep `--cfg` grid bracketing both its 2.0 default and the 6.0 its docs prescribe for
single-view RE10K. Tuning one side and pinning the other is exactly the swept-vs-unswept comparison
this benchmark exists to avoid.

## Bugs found and fixed

1. **`moment_scale_mult` clamp on raw-gauge models.** `svc_scenes.py` clamped the scaled moments to
   [−1,1] unconditionally. That is the *normalized* encoding's training band; on raw COLMAP-gauge
   moments it hard-clips real geometry, so every swept cell on `ov1`/`mot13b_full` would have
   measured progressive destruction of the moment channels rather than a scale change. `mult=1.0`
   short-circuits, so past unswept runs were unaffected.
2. **`--extra --save_raw` never worked.** argparse reads a value starting with `-` as a missing
   argument; every ours-side job died instantly (`expected one argument`, six EXIT 2 logs at
   02:05:45). All passthroughs are now the single-token `--extra=--flag=value` form.
3. **`--max_samples` default 50.** Carried into the shard path explicitly — this silently evaluated
   50 of 128 scenes once before and exited rc=0.

## Cluster notes (for the next person)

* `kempner_requeue` **cannot** appear in a multipartition submission — partitions are assigned
  round-robin, not as a union.
* Accounts must match the partition: `kempner_*` → `kempner_ydu_lab`, FASRC → `ydu_lab`.
* `kempner_h200_priority` rejects our group; `kempner_h100` projected a 11:52 start and `seas_gpu`
  10:14, both past the deadline — the preemptible requeue partitions backfill far faster.
* `rtx6000pro` (RTX PRO 6000 Blackwell, 96 GB) is ~120 idle GPUs on the requeue partitions and was
  being excluded by a constraint listing only `h100|h200|a100`.
* **One core per job.** The engine's DataLoader is in-process; FASRC flagged our earlier 12-core
  jobs as ~92% wasted (167 CPU-hours). Every job here is `--cpus-per-task=2` with BLAS pinned.
* Startup is I/O-dominated: 9.5 GB (SEVA) / 17 GB (5B) read from netscratch per job, ~5 min. Shard
  sizes are chosen so that load is amortised over ~50 min of sampling.
