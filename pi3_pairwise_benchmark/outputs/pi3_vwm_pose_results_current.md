# Pi3/VWM Pose Results

Saved on 2026-05-28. Updated with Pi3 TUM first/last-50 references on 2026-05-29.

| Dataset / setting | Method | Scope | Metric format | @5 deg | @15 deg | @30 deg | Distance metric |
|---|---|---|---|---:|---:|---:|---:|
| Re10K 10 random views | Pi3 paper | full paper | `Racc / Tacc / Auc` | `98.83 / 78.32 / 58.63` | `99.86 / 92.02 / 78.39` | `99.99 / 95.62 / 85.90` | n/a |
| Re10K 10 random views | Pi3 reproduced | 1719 seq | `Racc / Tacc / Auc` | `98.812 / 80.013 / 62.748` | `99.841 / 92.436 / 80.151` | `99.997 / 95.645 / 87.307` | n/a |
| Re10K 10 random views | VWM `320x240`, 40 steps | 3 seq smoke | `Racc / Tacc / Auc` | `46.67 / 5.93 / 0.74` | `91.85 / 43.70 / 17.23` | `100.0 / 65.93 / 36.05` | n/a |
| Re10K 10 random views | VWM `512x384` crop, 4 steps | 3 seq smoke | `Racc / Tacc / Auc` | `11.11 / 4.44 / 0.00` | `71.11 / 48.89 / 11.85` | `95.56 / 73.33 / 38.07` | n/a |
| Re10K 10 random views | VWM `512x384` crop, 40 steps | 3 seq smoke | `Racc / Tacc / Auc` | `34.81 / 4.44 / 0.44` | `93.33 / 45.19 / 15.65` | `100.0 / 88.89 / 42.69` | n/a |
| Re10K 10 views sorted by frame id | Pi3 saved-camera companion | 3 seq smoke | `Racc / Tacc / Auc` | `100.0 / 100.0 / 90.37` | `100.0 / 100.0 / 96.79` | `100.0 / 100.0 / 98.40` | n/a |
| Re10K 10 views sorted by frame id | VWM `512x384` crop, 40 steps | 3 seq smoke | `Racc / Tacc / Auc` | `44.44 / 2.96 / 0.59` | `99.26 / 72.59 / 28.94` | `100.0 / 100.0 / 60.91` | n/a |
| Sintel `alley_2` relpose-distance | Pi3 paper | full set | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.074 / 0.040 / 0.282` |
| Sintel `alley_2` relpose-distance | Pi3 reproduced | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0123 / 0.0106 / 0.0705` |
| Sintel `alley_2` relpose-distance | VWM `320x240`, 40 steps | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0458 / 0.0111 / 1.284` |
| Sintel `alley_2` relpose-distance | VWM `512x384` crop, 4 steps | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0563 / 0.0082 / 1.945` |
| Sintel `alley_2` relpose-distance | VWM `512x384` crop, 40 steps | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0628 / 0.0098 / 2.239` |
| TUM `sitting_halfsphere` 90 frames | Pi3 paper | full set | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.014 / 0.009 / 0.312` |
| TUM `sitting_halfsphere` 90 frames | Pi3 reproduced | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0095 / 0.0067 / 0.357` |
| TUM `sitting_halfsphere` 90 frames | VWM `320x240`, 40 steps | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0786 / 0.0100 / 1.549` |
| TUM `sitting_halfsphere` 90 frames | VWM `512x384` crop, 4 steps | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0679 / 0.0102 / 4.682` |
| TUM `sitting_halfsphere` 90 frames | VWM `512x384` crop, 40 steps | same seq | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0692 / 0.0101 / 2.019` |
| TUM `sitting_halfsphere`, first 50 frames | Pi3 reference | same window | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0088 / 0.0068 / 0.388` |
| TUM `sitting_halfsphere`, first 50 frames | VWM `512x384` crop, 40 steps | same window | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0263 / 0.0112 / 3.882` |
| TUM `sitting_halfsphere`, last 50 frames | Pi3 reference | same window | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0070 / 0.0062 / 0.345` |
| TUM `sitting_halfsphere`, last 50 frames | VWM `512x384` crop, 40 steps | same window | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0293 / 0.0112 / 3.285` |
| TUM `sitting_halfsphere`, first/last 50 mean | Pi3 reference | mean of 2 windows | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0079 / 0.0065 / 0.367` |
| TUM `sitting_halfsphere`, first/last 50 mean | VWM `512x384` crop, 40 steps | mean of 2 windows | `ATE / RPE-t / RPE-r` | n/a | n/a | n/a | `0.0278 / 0.0112 / 3.584` |

Notes:
- Sintel/TUM metrics are computed with Pi3's `eval_metrics()` path using `align=True` and `correct_scale=True`.
- Re10K VWM rows are 3-sequence smoke runs with 135 pairs; Pi3 reproduced is the full 1719-sequence official eval.
- For Sintel/TUM, Pi3 reproduced values in the table are same-sequence values for fair comparison; the paper values are full-set aggregate rows.

Visualization artifact:
`sparse_view/pi3_pairwise_benchmark/outputs/pose_visualizations_adjusted_20260528/index.html`
