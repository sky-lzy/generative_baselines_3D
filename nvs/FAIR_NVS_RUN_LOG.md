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

## Reliability log (the requeue partitions are hostile by design)

| event | count | handling |
|---|---|---|
| PREEMPTION | 87+ | `--requeue` + per-scene `--resume`; automatic, no loss |
| TIMEOUT | 5+ | **not** auto-requeued — needed `topup_fair.py`; wall clock raised 1.5h → 2h |
| CUDA device busy | 2 | transient node contention; top-up resubmits |
| `lang_guidance > 0` | 2 | **not runnable** in this inference path (text encoder not instantiated) |

Two top-up bugs worth remembering, both found by watching what it actually did:

1. **Cell-level gating was too coarse.** A dead shard hid behind nine healthy ones until the whole
   cell went idle. On a deadline that is an hour lost for nothing. Now judged per shard.
2. **Naive shard-level gating resubmitted finished shards.** Cell counts cannot distinguish "this
   shard died" from "this shard is done and the cell is merely early", so the first shard-level pass
   queued 84 jobs, most of them no-ops at 5–11 min of weight loading each — and the 25-minute
   cooldown would have repeated it. `shard_missing()` now recomputes the shard's own assigned scenes.

**Shard count changed mid-run** (8 → per-kind 4/3/10) when shards were sized from measured
throughput. This is safe for coverage because sample indices are GLOBAL and `--resume` is per-scene:
any partition that spans 0..127 completes the cell, and the 4-way partition does. It does mean shard
*geometry* is not comparable between the early and later jobs of the same cell, which is why the 80
already-queued no-op shards were left to run rather than cancelled on a geometry test — a wrong
`scancel` would have cost far more than ~10 GPU-h of redundant startup.

## Results (128 scenes, each method at its own PSNR-best scale)

| benchmark | Ours F (5B) | Ours MoT (1.3B) | SEVA | F vs SEVA |
|---|---|---|---|---|
| 50-frame, 2 view | 23.068 / .7744 / .0821 | 17.790 / .5782 / .2872 | **25.250 / .8378 / .0813** | −2.182 dB, p=4.6e-15 |
| 4DiM, 2 view | 20.562 / .6883 / .1117 | 15.136 / .5125 / .5269 | **24.295 / .8255 / .0819** | −3.733 dB, p=1.6e-40 |
| 4DiM, 1 view | **15.957 / .5184 / .2640** | 12.338 / .4707 / .9176 | 15.729 / .4971 / .2764 | +0.228 dB, p=0.30 (tie) |
| 50-frame, 1 view | 17.220 / .5561 / **.2176** | 13.352 / .4906 / .8114 | 17.218 / **.5645** / .2365 | +0.002 dB, p=0.99 (tie) |

**The pattern is structural, not noise.** With two input views SEVA wins decisively (−2.2 and −3.7 dB,
p < 1e-14, we take 14/128 and 4/128 scenes). With one input view the two are statistically
indistinguishable (+0.23 dB p=0.30 with 69/128 scenes; +0.002 dB p=0.99 with 64/128 — a literal
coin flip), and our LPIPS is slightly better in both single-view cells.

That is what SEVA's re-anchoring predicts. `chunk_strategy="nearest-gt"` re-injects the GT input
frames into every chunk, so with 2 inputs it evaluates at ~9.5% conditioning density against our 4%;
with 1 input that advantage largely evaporates. The gap is a conditioning-density difference, not a
resolution or scoring artefact — the geometry and scorer parity checks above rule those out.

### PSNR rewards under-travel in the single-view regime

`travel_ratio` at each method's PSNR-best scale (1.0 = frame k lands at camera k):

| cell | ours F | SEVA |
|---|---|---|
| 50-frame 2 view | 0.996 (19/144 exact) | 1.008 (**96/144**) |
| 4DiM 2 view | 0.974 (3/18) | 1.006 (13/18) |
| 4DiM 1 view | 0.668 | **0.381** |
| 50-frame 1 view | 0.551 | 0.832 |

Two readings, both worth keeping:

1. **Two-view: both are geometrically correct, but SEVA is far more precise per frame** — 96 of 144
   frames land exactly against our 19. That, not any setup asymmetry, is the −2.2 dB.
2. **Single-view: BOTH methods' PSNR-optimal scale under-travels badly.** Staying near the input
   frame is a good L2 hedge when the target is 60 frames away, so the PSNR-best scale is *not* the
   geometrically faithful one. On 4DiM ours is closer to correct (0.668 vs 0.381) while scoring
   marginally higher; on 50-frame it is the other way round (0.551 vs 0.832). Any single-view claim
   from PSNR alone should be read with this in mind.

### Caveat: SEVA's sweep is not complete on two cells

Sweep depth achieved per side — 4DiM 1-view 5 vs 3, 50-frame 1-view 5 vs 1. Both of our single-view
"ties" therefore compare our best-of-5 against SEVA's best-of-3 and best-of-1. **SEVA's numbers in
those two cells are lower bounds**, and the ties could move against us when its remaining scales
finish. The two-view cells are like-for-like (1 vs 1 and 5 vs 5) and are final.
