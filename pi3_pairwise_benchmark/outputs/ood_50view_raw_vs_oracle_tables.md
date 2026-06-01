# OOD Raw vs Oracle Tables

OOD rows include 50-sequence 50-view runs with all C(50,2)=1225 relative-pose pairs per sequence, plus 50-sequence 2-view long-baseline runs with 100 sampled pairs per dataset/method. Oracle rows use a GT-assisted per-sequence SO3 camera-basis fit from predicted relative translation directions to GT directions.

## Raw Pi3 Metrics

| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DL3DV-Eval 50-view | Pi3 | 98.66 | 96.10 | 89.11 | 99.46 | 97.75 | 94.62 | 100.00 | 98.03 | 96.27 | 61250 |
| DL3DV-Eval 50-view | Ours/VWM | 77.02 | 27.08 | 8.43 | 96.88 | 82.74 | 44.43 | 98.64 | 94.50 | 67.55 | 61250 |
| DL3DV-Eval 50-view | Geo4D | 64.64 | 13.34 | 3.61 | 92.35 | 58.08 | 25.30 | 98.23 | 84.16 | 49.24 | 61250 |
| 7Scenes 50-view | Pi3 | 89.46 | 61.22 | 30.80 | 99.91 | 91.09 | 65.52 | 100.00 | 96.61 | 80.15 | 61250 |
| 7Scenes 50-view | Ours/VWM | 29.23 | 6.38 | 0.84 | 84.14 | 40.53 | 13.06 | 98.93 | 77.71 | 37.29 | 61250 |
| 7Scenes 50-view | Geo4D | 34.59 | 4.24 | 0.86 | 82.59 | 28.19 | 10.29 | 98.21 | 61.67 | 28.63 | 61250 |
| DL3DV-Eval 2-view long | Pi3 | 94.00 | 94.00 | 85.80 | 97.00 | 97.00 | 92.20 | 99.00 | 97.00 | 94.60 | 100 |
| DL3DV-Eval 2-view long | Ours/VWM | 61.00 | 44.00 | 12.00 | 89.00 | 87.00 | 51.60 | 95.00 | 90.00 | 70.13 | 100 |
| DL3DV-Eval 2-view long | Geo4D | 9.00 | 0.00 | 0.00 | 27.00 | 2.00 | 0.40 | 53.00 | 11.00 | 1.53 | 100 |
| 7Scenes 2-view long | Pi3 | 76.00 | 45.00 | 22.00 | 94.00 | 88.00 | 58.53 | 94.00 | 91.00 | 74.17 | 100 |
| 7Scenes 2-view long | Ours/VWM | 31.00 | 15.00 | 0.80 | 70.00 | 52.00 | 18.93 | 81.00 | 73.00 | 40.00 | 100 |
| 7Scenes 2-view long | Geo4D | 3.00 | 0.00 | 0.00 | 25.00 | 1.00 | 0.00 | 52.00 | 12.00 | 0.50 | 100 |

## Oracle Per-Sequence Camera-Basis SO3

| Setting | Method | Racc_5 | Tacc_5 | Auc_5 | Racc_15 | Tacc_15 | Auc_15 | Racc_30 | Tacc_30 | Auc_30 | Pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DL3DV-Eval 50-view | Pi3 | 98.59 | 96.54 | 90.47 | 99.68 | 99.32 | 95.54 | 100.00 | 99.78 | 97.57 | 61250 |
| DL3DV-Eval 50-view | Ours/VWM | 72.85 | 57.41 | 18.23 | 96.92 | 89.39 | 55.83 | 98.38 | 96.27 | 74.75 | 61250 |
| DL3DV-Eval 50-view | Geo4D | 57.13 | 38.34 | 7.52 | 90.92 | 79.12 | 37.69 | 97.99 | 92.01 | 61.32 | 61250 |
| 7Scenes 50-view | Pi3 | 83.90 | 66.60 | 32.23 | 98.18 | 91.68 | 65.78 | 100.00 | 97.02 | 80.35 | 61250 |
| 7Scenes 50-view | Ours/VWM | 24.05 | 12.20 | 0.94 | 79.41 | 54.40 | 16.84 | 96.64 | 82.85 | 42.57 | 61250 |
| 7Scenes 50-view | Geo4D | 23.18 | 9.42 | 0.66 | 67.38 | 43.45 | 11.35 | 90.34 | 72.57 | 32.48 | 61250 |
| DL3DV-Eval 2-view long | Pi3 | 66.00 | 96.00 | 49.40 | 85.00 | 98.00 | 68.87 | 95.00 | 100.00 | 78.97 | 100 |
| DL3DV-Eval 2-view long | Ours/VWM | 18.00 | 80.00 | 8.40 | 47.00 | 96.00 | 27.13 | 68.00 | 96.00 | 43.57 | 100 |
| DL3DV-Eval 2-view long | Geo4D | 2.00 | 72.00 | 0.80 | 21.00 | 96.00 | 6.07 | 50.00 | 100.00 | 21.10 | 100 |
| 7Scenes 2-view long | Pi3 | 50.00 | 84.00 | 24.60 | 75.00 | 92.00 | 53.73 | 82.00 | 96.00 | 66.30 | 100 |
| 7Scenes 2-view long | Ours/VWM | 7.00 | 50.00 | 1.00 | 43.00 | 76.00 | 16.87 | 66.00 | 92.00 | 37.20 | 100 |
| 7Scenes 2-view long | Geo4D | 1.00 | 32.00 | 0.00 | 12.00 | 74.00 | 3.07 | 42.00 | 94.00 | 14.13 | 100 |
