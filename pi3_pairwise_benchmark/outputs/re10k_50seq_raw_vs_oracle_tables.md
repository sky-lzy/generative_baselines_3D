# Re10K Raw vs Oracle Tables

Oracle rows apply a GT-assisted per-sequence SO3 camera-basis rotation fit from predicted relative translation directions to GT directions. This is an upper-bound diagnostic, not a fair benchmark.

## Raw Pi3 Metrics

| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50-view | Pi3 | 100.00 | 79.54 | 58.70 | 100.00 | 93.33 | 79.22 | 100.00 | 95.87 | 87.14 | 61250 |
| 50-view | Ours/VWM | 97.16 | 22.56 | 9.88 | 100.00 | 83.80 | 43.66 | 100.00 | 97.29 | 68.87 | 61250 |
| 50-view | Geo4D | 98.98 | 1.13 | 0.45 | 100.00 | 23.87 | 8.13 | 100.00 | 56.75 | 25.30 | 61250 |
| 2-view long | Pi3 | 100.00 | 89.00 | 70.00 | 100.00 | 94.00 | 85.40 | 100.00 | 98.00 | 91.37 | 100 |
| 2-view long | Ours/VWM | 94.00 | 23.00 | 9.00 | 98.00 | 70.00 | 38.47 | 99.00 | 94.00 | 62.30 | 100 |
| 2-view long | Geo4D | 69.00 | 4.00 | 1.20 | 91.00 | 23.00 | 9.93 | 97.00 | 47.00 | 23.40 | 100 |

## Oracle Per-Sequence Camera-Basis SO3

| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50-view | Pi3 | 97.55 | 85.33 | 68.56 | 99.46 | 94.56 | 84.11 | 100.00 | 96.78 | 90.01 | 61250 |
| 50-view | Ours/VWM | 90.36 | 74.27 | 37.76 | 99.58 | 97.59 | 72.71 | 100.00 | 99.63 | 85.90 | 61250 |
| 50-view | Geo4D | 85.05 | 34.13 | 12.98 | 99.25 | 71.72 | 41.74 | 100.00 | 87.60 | 61.82 | 61250 |
| 2-view long | Pi3 | 73.00 | 98.00 | 63.80 | 95.00 | 100.00 | 79.47 | 98.00 | 100.00 | 88.63 | 100 |
| 2-view long | Ours/VWM | 52.00 | 60.00 | 14.20 | 81.00 | 90.00 | 47.07 | 95.00 | 96.00 | 66.80 | 100 |
| 2-view long | Geo4D | 47.00 | 72.00 | 20.80 | 70.00 | 90.00 | 42.67 | 91.00 | 98.00 | 60.87 | 100 |
