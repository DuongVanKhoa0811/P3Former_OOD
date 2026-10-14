# 2026-09-15 — Random two-group partitions of the 24 classes (offline sweep)

Full record of the `DOCs.md` entry of the same date ([DOCs.md](../../DOCs.md)). Paths are relative to the repo root.

`tools/sweep_bipartitions.py` scores random 2-group partitions of the 24 classes straight
from a logits dump, without running the model: the smaller group's size is drawn uniformly
from 1..12, then its classes at random; the vehicle-vs-rest split (`s0.1.2.3.4` =
`p_v_hgcno`) is always included as a cross-check. Names are the train ids of the smaller
group (`partitions.tsv` lists the class names). Same score formulas as `ood_scores.py`,
same metric formulas as `_OODPointMetric`; `bipartitions.log` is read by
`summarize_hierarchy_ablation.py`. Several dump directories are evaluated as one split.

- **Splits.** `dso_infos_test_cetran.pkl` is exactly test + Cetran (same frames, same order),
  so only the held-out test split needed a new dump:
  `p3former_2xb1_3x_dso_ood_dump_test.py` (2,625 frames, 55 GB, written to
  `/mnt/sandisk/khoadv/...` and symlinked as `work_dirs/p3former_2xb1_3x_dso_ood_dump/logits_test`;
  ~25 min, ~103 GB RAM). Passing `logits_test` + `logits` gives test + Cetran, `logits_test`
  alone the test split.
- **Backend.** `--backend numpy` (default) uses CPU workers (Cetran: ~1 h with 8 workers);
  `--backend torch --device cuda:N` does the same two passes on a GPU: Cetran 13 min, test
  54 min, test + Cetran 67 min. TF32 matmuls are switched off (they shift the group masses by
  ~4e-4 and FPR@95 by up to 57 points) and CUDA `bincount` is replaced by a sort-based count
  (~340 ms vs ~1 ms per block in torch 1.10).
- **Bins.** 2^16 bins, log-spaced towards both ends of every score range (`--bin-eps`;
  `--bin-eps 0` = the equal-width bins of the online metric). Equal-width bins at 2^16 cannot
  separate very confident OOD points from the ID mass: on the test split more than 5% of the
  OOD points sit within 7.6e-6 of full confidence and FPR@95 read 100% for 43/500 splits.
  With log-spaced bins the flat and Group rows reproduce the online metric to <= 0.05 on all
  three splits (vehicle-vs-rest Group MSP on Cetran: 96.00/47.23/18.42 offline = online). One
  exception, Group MSP FPR@95 of vehicle-vs-rest on test (67.10 offline, 68.02 online), is
  the online value's own bin limit: for two groups Group Entropy is a monotone function of
  Group MSP, so both must give the same FPR@95 — 67.07 / 67.10 offline, 67.12 / 68.02 online. GN
  MSP / GN Entropy stay approximate — they pile up at an interior value, -(1 - prior) — e.g.
  test vehicle-vs-rest GN MSP FPR@95 98.38 offline vs 76.99 online; GN Energy is exact.

```bash
# held-out test dump (once), then the sweeps; long jobs: start them with nohup setsid ... &
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_dso_ood_dump_test.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
W=work_dirs/p3former_2xb1_3x_dso_ood_dump
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits                                                        # Cetran      -> $W/bipartitions
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test --out-dir $W/bipartitions_test                    # test
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test $W/logits --out-dir $W/bipartitions_test_cetran   # test + Cetran
python tools/summarize_hierarchy_ablation.py --family group --exclude odin --plot $W/bipartitions_test_cetran/top4_group.png $W/bipartitions_test_cetran/bipartitions.log
```

Results (500 partitions, seed 0; family group, ODIN excluded — the GN family is not ranked
offline, see 2026-09-24; Group MSP as AUROC/AP/FPR@95, improvement = mean dAUROC + mean dAP
- mean dFPR@95):

| split (flat MSP; flat Energy) | above flat | best split | its Group MSP | improvement |
| --- | --- | --- | --- | --- |
| Cetran (90.42/28.27/32.32; 93.14/40.33/26.35) | 61 | `s1.3.17` {bicycle, truck, gate} | 96.84/54.34/17.13 | +27.69 |
| test (86.26/13.75/51.15; 92.36/35.47/42.56) | 136 | `s16` {overhead-bridge} | 94.21/39.75/29.06 | +34.78 |
| test + Cetran (87.76/18.06/45.76; 92.88/35.54/37.95) | 99 | `s16` {overhead-bridge} | 94.04/40.45/28.52 | +27.42 |

- **The Cetran winners do not transfer.** On test + Cetran `s1.3.17` drops to #45 (+7.38,
  91.70/26.56/41.73) and vehicle-vs-rest (`p_v_hgcno`, #12 on Cetran with +21.37) to #142
  (-3.17, 90.07/25.30/54.59); on test alone they fall below flat (-0.27 and -10.88; online
  test `p_v_hgcno` Group MSP 87.21/17.23/68.02 vs flat MSP 86.26/13.76/51.15). Rank
  correlation of the improvement: Cetran~test +0.61, test~test+Cetran +0.97 (test has 4x the
  points). 32 splits beat flat on both Cetran and test.
- **test + Cetran top 5:** `s16` +27.42, `s2.16` {motorcycle, overhead-bridge} +26.24
  (94.16/37.98/27.90), `s4.16` {bus, overhead-bridge} +23.50, `s2.3.4.5.6.16` +21.01,
  `s4.5.6.16` +19.39. Overhead-bridge alone is the best split on test and test + Cetran but
  only #40 on Cetran (+7.03); single classes above flat on test + Cetran: overhead-bridge
  +27.4, gate +10.1, truck +4.0, perimeter-barrier +1.2 (Cetran: truck +25.2, gate +17.5,
  overhead-bridge +7.0, bicycle +5.8). Building, drain, unpaved-road, bus and overhead-bridge
  are over-represented in the top-20 smaller groups; paved-road, sidewalk, terrain, trunks
  and vegetation never appear there on any split.
- **Most robust splits** (largest worst-case improvement over Cetran and test):
  `s0.1.3.5.10.16` {car, bicycle, truck, person, unpaved-road, overhead-bridge} +17.87 / +19.04
  (test + Cetran +18.66, 93.52/33.19/33.63) and `s2.3.4.5.6.16` {motorcycle, truck, bus, person,
  rider, overhead-bridge} +16.94 / +22.23 (test + Cetran +21.01, 93.84/34.65/31.91).
- **Reading.** A two-group Group-MSP flags a point when its mass is split between the two
  groups, so the best split isolates the classes the OOD points are confused with and
  merges every within-ID confusion. Which classes those are depends on the scenes — truck /
  gate on Cetran, overhead-bridge (and building / drain / unpaved-road) on the test routes —
  so a split picked on one split is tuned to its OOD objects. The gain is in the MSP /
  Entropy scores; Group Energy moves by < 1 point everywhere.

Caveat: these are selections among 500 random splits on the evaluation data itself — report
a split only after confirming it online (`class_groups_variants`) on data it was not picked on.
