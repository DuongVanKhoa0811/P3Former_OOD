# DOCs

## Point-level OOD baselines on DSO

Test split (2,625 frames; PQ 46.50, mIoU 48.66; 1.008B ID / 15.9M OOD points
= 1.55% OOD):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 86.26 | 13.76 | 51.15  |
| MaxLogit | 92.08 | 34.24 | 42.70  |
| ODIN     | 86.53 | 19.51 | 69.63  |
| Energy   | 92.36 | 35.47 | 42.55  |
| Entropy  | 87.84 | 20.23 | 50.40  |
| Group MSP      | 89.33 | 14.50 | 45.99  |
| Group MaxLogit | 92.08 | 34.24 | 42.70  |
| Group ODIN     | 22.40 |  0.92 | 96.21  |
| Group Energy   | 92.35 | 35.01 | 42.56  |
| Group Entropy  | 89.67 | 18.91 | 46.41  |
| GN MSP         | 92.35 | 18.08 | 34.11  |
| GN MaxLogit    | 93.54 | 35.99 | 33.18  |
| GN ODIN        | 65.68 |  6.10 | 96.09  |
| GN Energy      | 93.79 | 36.53 | 32.81  |
| GN Entropy     | 92.11 | 13.70 | 34.11  |

Test + Cetran (3,605 frames; PQ 46.19, mIoU 47.94; 1.261B ID / 23.6M OOD points
= 1.84% OOD):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 87.76 | 18.06 | 45.76  |
| MaxLogit | 92.67 | 35.90 | 38.12  |
| ODIN     | 89.36 | 27.19 | 55.64  |
| Energy   | 92.88 | 35.55 | 37.94  |
| Entropy  | 89.38 | 26.12 | 44.90  |
| Group MSP      | 89.97 | 16.30 | 40.66  |
| Group MaxLogit | 92.67 | 35.90 | 38.12  |
| Group ODIN     | 24.43 |  1.11 | 96.08  |
| Group Energy   | 92.88 | 36.03 | 37.95  |
| Group Entropy  | 90.45 | 21.85 | 40.78  |
| GN MSP         | 92.07 | 19.81 | 39.89  |
| GN MaxLogit    | 93.75 | 36.28 | 30.36  |
| GN ODIN        | 70.95 |  8.60 | 94.82  |
| GN Energy      | 93.94 | 36.31 | 29.96  |
| GN Entropy     | 91.75 | 14.49 | 39.89  |

## Point-level OOD baselines on SemanticKITTI

Val (4071 scans; PQ 60.32, PQ† 62.99, mIoU 62.49 vs 62.63 / 66.25 / 66.77 for the official
checkpoint; same 476.8M ID / 9.4M OOD points):

| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 87.25 | 12.79 | 41.39  |
| MaxLogit | 90.03 | 32.50 | 44.38  |
| ODIN     | 91.46 | 26.94 | 38.33  |
| Energy   | 90.20 | 33.26 | 44.47  |
| Entropy  | 88.33 | 17.40 | 40.99  |
| Group MSP      | 90.65 | 17.26 | 36.16  |
| Group MaxLogit | 90.03 | 32.50 | 44.38  |
| Group ODIN     | 27.45 |  1.20 | 94.44  |
| Group Energy   | 90.39 | 35.17 | 44.37  |
| Group Entropy  | 91.03 | 22.14 | 35.74  |
| GN MSP         | 71.81 | 12.70 | 89.49  |
| GN MaxLogit    | 87.88 | 28.51 | 52.50  |
| GN ODIN        | 88.47 | 12.27 | 44.20  |
| GN Energy      | 88.14 | 30.65 | 52.56  |
| GN Entropy     | 71.86 | 12.49 | 89.49  |
