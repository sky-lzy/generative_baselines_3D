# TUM RGB-D Raw vs Oracle Tables

TUM RGB-D rows use 50 raw RGB-D trajectory windows with 50 frames each. 50-view rows evaluate all C(50,2)=1225 relative-pose pairs per sequence. 2-view rows use 2 deterministic long-baseline pairs per sequence; Geo4D has 99 valid pairs because one pair hit an OpenCV PnP degeneracy. Oracle rows use a GT-assisted per-sequence SO3 camera-basis fit from predicted relative translation directions to GT directions.

## Raw Pi3 Metrics

| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TUM RGB-D 50-view | Pi3 | 99.84 | 45.27 | 28.31 | 100.00 | 66.66 | 48.78 | 100.00 | 80.75 | 61.82 | 61250 |
| TUM RGB-D 50-view | Ours/VWM | 54.56 | 6.57 | 1.31 | 84.16 | 32.40 | 11.81 | 95.48 | 58.87 | 29.17 | 61250 |
| TUM RGB-D 50-view | Geo4D | 55.21 | 1.52 | 0.25 | 80.71 | 12.16 | 3.97 | 90.84 | 30.89 | 12.53 | 61250 |
| TUM RGB-D 2-view long | Pi3 | 97.00 | 48.00 | 32.20 | 99.00 | 69.00 | 53.40 | 99.00 | 87.00 | 67.70 | 100 |
| TUM RGB-D 2-view long | Ours/VWM | 64.00 | 8.00 | 2.40 | 77.00 | 28.00 | 11.87 | 83.00 | 59.00 | 26.87 | 100 |
| TUM RGB-D 2-view long | Geo4D | 40.40 | 0.00 | 0.00 | 58.59 | 3.03 | 0.20 | 71.72 | 12.12 | 3.30 | 99 |

## Oracle Per-Sequence Camera-Basis SO3

| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TUM RGB-D 50-view | Pi3 | 82.35 | 47.18 | 29.43 | 91.09 | 69.83 | 49.71 | 95.81 | 83.69 | 62.62 | 61250 |
| TUM RGB-D 50-view | Ours/VWM | 43.21 | 12.17 | 2.16 | 73.81 | 43.83 | 15.54 | 87.76 | 66.84 | 33.08 | 61250 |
| TUM RGB-D 50-view | Geo4D | 40.06 | 3.04 | 0.41 | 67.50 | 20.48 | 5.04 | 80.16 | 46.38 | 15.67 | 61250 |
| TUM RGB-D 2-view long | Pi3 | 63.00 | 68.00 | 38.60 | 77.00 | 84.00 | 54.93 | 88.00 | 98.00 | 66.47 | 100 |
| TUM RGB-D 2-view long | Ours/VWM | 34.00 | 26.00 | 3.40 | 61.00 | 60.00 | 20.93 | 66.00 | 82.00 | 36.60 | 100 |
| TUM RGB-D 2-view long | Geo4D | 21.21 | 27.27 | 2.63 | 54.55 | 61.62 | 16.63 | 64.65 | 85.86 | 32.19 | 99 |
