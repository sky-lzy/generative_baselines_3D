# Gen3R / Gen3C on the fair NVS benchmark

`evaluation/score_nvs_fair.py` already supports both (`--method gen3r|gen3c`); what was
missing was a way to tell a *bad model* from *malformed predictions*. That is what
`verify_baseline_preds.py` is for.

## Scoring an existing cell

```bash
python nvs/evaluation/score_nvs_fair.py \
  --run_dir  <preds>/<tag> \
  --method   gen3c \
  --scenes_root <svc_bench>/scenes_fair/re10k128_50f \
  --split 2 --metric_size 256 \
  --out_csv  nvs/results_fair/<tag>.csv \
  --require_complete
```

`--require_complete` matters: without it the scorer averages whatever scenes parsed and a
half-broken cell yields a plausible-looking number.

## Before trusting a number, verify the predictions

```bash
python nvs/baselines/verify_baseline_preds.py --method gen3c --run_dir <preds>/<tag> \
    --split 2 --n_frames 50 --expect_scenes 128
```

It re-implements the scorer's read path exactly (same filenames, shapes, dtype, index
range) and reports per scene whether it would be **scored or silently dropped** — no GPU,
no model, no ground truth needed. Exit 0 = clean, 1 = at least one scene would be dropped
or the scene count is wrong.

Verified against the real 128-scene Gen3R cell (`fair50f_2view`): 128/128 valid. Negative
controls all caught: wrong reader (`no pred_rgb.npy` ×128), out-of-range ids under
`--split 1` (`ids [49,50] exceed Gen3R's 49 slots`), and a short 5-scene cell.

## File contracts (from score_nvs_fair.py)

| method | file | shape / dtype | frames |
|---|---|---|---|
| gen3c | `pred_rgb.npy` | `(T,H,W,3)` **uint8** | clip's own T; scorer indexes by CLIP frame id |
| gen3r | `pred_rgb_560.npy` | `(T,H,W,3)` | **exactly 49** (0..48) |

## Why Gen3C scores low — read this before concluding the model is broken

1. **Conditioning asymmetry (the big one).** Gen3C is seeded from **one** view, while ours
   and SEVA get the two `ncf2` conditioning views. Multi-frame seeding needs supplied
   depth, and two independent MoGe depths do not share a scale. Its score is therefore a
   **lower bound and not a like-for-like comparison.** `provenance.json` records `n_seed`;
   the verifier prints the distribution and warns when it is all-1.
2. **121-frame window.** Gen3C must generate 121 frames; the runner resamples our camera
   path across that window and slices the original indices back out. An off-by-one in that
   slice shifts every frame and tanks PSNR while leaving the images looking fine — compare
   `pred_rgb.npy[i]` against `gt_rgb.mp4` frame `i` visually before trusting a low score.
3. **Coordinate frame.** Camera-conditioned baselines expect OpenCV convention. Scenes
   built for Aria carry a device-frame correction (`T_fix`); applying it twice, or not at
   all, produces a plausible-looking but wrong trajectory.
4. **Range/dtype.** The scorer divides by 255, so `pred_rgb.npy` must be uint8. float in
   [0,1] silently scores as near-black — the verifier rejects this explicitly.

## Known-good reference

Gen3R, 128 scenes, 50f/ncf2, padded to 48 views: **19.1986 dB / 0.6191 / 0.2048**.
The 47-view figure (19.0204) is *not* comparable to 48-view methods.
