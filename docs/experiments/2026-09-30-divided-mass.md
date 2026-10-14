# 2026-09-30 — Why some two-group splits beat flat: divided mass

Full record of the `DOCs.md` entry of the same date ([DOCs.md](../../DOCs.md)). Paths are relative to the repo root.

For a split A | B, Group MSP is m − 1, where m = min(P_A, P_B) is the *divided mass*. A point is *divided* at δ when m ≥ δ; at δ = 0.05 that means 0.05 ≤ P_A ≤ 0.95. So the OOD / ID divided shares at δ are Group MSP's TPR / FPR at the threshold δ − 1. Flat MSP has the same reading through u = 1 − max p, and u ≥ m for every point. `tools/divided_mass.py` histograms m and u from the logit dumps for two sets of splits, then joins the sweep's metric deltas:
- the 24 single-class splits. The 500-split sweep had missed {unpaved-road}, {sidewalk} and {building}, so all 24 are now also scored with `sweep_bipartitions.py --subsets singletons`;
- the 500 sweep splits.

The spec and plan are `docs/superpowers/{specs,plans}/2026-09-*-divided-mass-resemblance*`.

```bash
W=work_dirs/p3former_2xb1_3x_dso_ood_dump
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits --subsets singletons --out-dir $W/singletons
python tools/sweep_bipartitions.py --backend torch --device cuda:1 $W/logits_test --subsets singletons --out-dir $W/singletons_test
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test $W/logits --subsets singletons --out-dir $W/singletons_test_cetran
python tools/divided_mass.py --backend torch --device cuda:0        # -> $W/divided_mass/ (~7 min)
python tools/plot_divided_mass.py && python tools/plot_divided_mass.py --threshold 0.001
```

**Checks** (`summary.md`): all 21 PASS.
- Flat MSP from the u histograms matches the sweep within 0.004 AUROC, 0.003 AP and 0.15 FPR@95.
- Group MSP matches the sweep within 0.006 AUROC, 0.052 AP and 0.27 FPR@95 on the 482 / 453 / 460 well-conditioned splits (δ95 ≥ 1e-5).
- The 21 single-class splits scored by both sweeps agree exactly.

The first run caught two defects before these checks passed: the u histograms were clipped at 0.5, and the float32 effect below.

**Float32 ties in the implemented Group MSP, and the online bins.** The sweep ranks points by the float32 score −max(P_A, P_B), which cannot separate points with m < 2⁻²⁴ ≈ 6e-8. The online metric of `test.py` is coarser still. It puts every score into 2^20 equal-width bins between the scores' minimum and maximum; Group MSP spans about 0.5, so every point with m below ≈ 4.8e-7 falls into one bin.
- On 21 / 50 / 43 splits, catching 95 % of the OOD points needs a threshold below m ≈ 1e-5. For those splits the implemented FPR@95 depends on float32 rounding, and online also on the binning.
- For example, {bicycle} on Test reads 89.20 in the sweep, 100 under a float32 emulation, and **80.43 exactly**. The `exact_*` columns hold the float32-free values.
- {gate} on Test is a robust split with δ95 = 4.0e-7, which lies below the online resolution. Online, its FPR@95 lands near 42 or at 100, depending on where the bins fall, so it cannot be confirmed online as things stand. {gate} on Test + Cetran (δ95 = 7.1e-7) is also affected, but less: an emulation of the online bins from `histograms.npz` gives FPR@95 ≈ 39.5, against 34.0 exact, and it stays robust.
- The same emulation gives 67.19 for `p_v_hgcno`'s Group MSP FPR@95 on test, not the online 68.02. So the 09-15 entry's attribution of that gap to the online bin limit is unsupported.
- Scoring Group MSP as log min(P_A, P_B) would fix this, since it ranks identically in exact arithmetic. So would log-spaced online bins.

**Table A — single-class splits at δ = 0.05.** The rows are every class that beats flat on some set, plus {vegetation}.
- *div*: the % of points divided.
- *prec*: the share of the divided points that are OOD. Flat MSP's share of *uncertain* points (u ≥ δ) that are OOD is 11.0 / 6.2 / 7.3 % on Cetran / Test / Test + Cetran.
- *sel*: selectivity, the OOD retention over the ID retention. Retention is the fraction of flat MSP's uncertain points that stay divided.
- *δ95*: the depth at which 95 % of the OOD points are divided. The ID div there equals the exact FPR@95 (flat MSP: 32.4 / 51.2 / 45.9).
- *imp*: the improvement.

| set | split | OOD div % | ID div % | prec % | sel | δ95 | ID div % at δ95 | imp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cetran | truck | 15.6 | 0.285 | 62.4 | 13.4 | 5.3e-5 | 18.4 | +25.2 |
| Cetran | gate | 11.4 | 0.099 | 77.8 | 28.2 | 8.7e-6 | 21.7 | +17.5 |
| Cetran | overhead-bridge | 11.1 | 0.098 | 77.4 | 27.6 | 8.0e-6 | 33.0 | +7.0 |
| Cetran | bicycle | 1.65 | 0.037 | 57.3 | 10.8 | 1.0e-6 | 28.5 | +5.8 |
| Cetran | building | 52.3 | 2.80 | 36.2 | 4.6 | 3.2e-4 | 32.1 | +4.3 |
| Cetran | bus | 1.37 | 0.015 | 73.6 | 22.5 | 4.0e-6 | 31.7 | +0.3 |
| Cetran | perimeter-barrier | 48.3 | 8.61 | 14.6 | 1.37 | 1.5e-4 | 26.1 | −11.7 |
| Cetran | vegetation | 33.0 | 7.41 | 11.9 | 1.09 | 1.2e-4 | 58.0 | −40.4 |
| Test | overhead-bridge | 18.2 | 0.142 | 66.9 | 30.8 | 3.6e-6 | 29.1 | +34.8 |
| Test | building | 51.4 | 2.56 | 24.0 | 4.8 | 1.5e-4 | 37.4 | +16.3 |
| Test | gate | 0.53 | 0.072 | 10.3 | 1.76 | 4.0e-7 | 41.5 | +6.6 |
| Test | perimeter-barrier | 24.6 | 1.52 | 20.4 | 3.9 | 3.3e-5 | 44.6 | +5.5 |
| Test | truck | 6.42 | 0.254 | 28.5 | 6.1 | 2.8e-6 | 57.9 | −2.9 |
| Test | bus | 0.81 | 0.096 | 11.8 | 2.04 | 2.9e-7 | 59.8 | −6.1 |
| Test | vegetation | 38.5 | 12.0 | 4.8 | 0.77 | 3.6e-4 | 70.5 | −28.6 |
| Test | bicycle | 0.48 | 0.067 | 10.1 | 1.72 | 9.3e-10 | 80.4 | −40.1 |
| Test + Cetran | overhead-bridge | 15.9 | 0.133 | 69.0 | 28.2 | 4.6e-6 | 28.5 | +27.4 |
| Test + Cetran | building | 51.7 | 2.61 | 27.0 | 4.7 | 2.0e-4 | 33.8 | +13.3 |
| Test + Cetran | gate | 4.07 | 0.078 | 49.5 | 12.4 | 7.1e-7 | 34.0 | +10.1 |
| Test + Cetran | truck | 9.42 | 0.260 | 40.4 | 8.6 | 4.6e-6 | 47.6 | +4.0 |
| Test + Cetran | perimeter-barrier | 32.3 | 2.94 | 17.1 | 2.6 | 4.8e-5 | 37.4 | +1.2 |
| Test + Cetran | bus | 0.99 | 0.079 | 19.0 | 2.96 | 4.8e-7 | 53.9 | −7.0 |
| Test + Cetran | vegetation | 36.7 | 11.0 | 5.8 | 0.78 | 2.6e-4 | 70.3 | −35.8 |
| Test + Cetran | bicycle | 0.86 | 0.061 | 20.8 | 3.3 | 2.7e-9 | 77.3 | −38.8 |

**Table B — Spearman ρ at δ = 0.05**, over the 24 single-class / all 503 splits. Precision and the log OOD/ID ratio rank the splits nearly identically; they differ only through the +0.5 continuity correction.

| set | prec ~ ΔAP | prec ~ ΔAUROC | prec ~ improvement | OOD div ~ ΔFPR@95 | ID div ~ ΔAP |
| --- | --- | --- | --- | --- | --- |
| Cetran | 0.63 / 0.87 | 0.43 / 0.77 | 0.45 / 0.72 | −0.43 / −0.28 | −0.37 / −0.41 |
| Test | 0.81 / 0.95 | 0.54 / 0.80 | 0.51 / 0.72 | −0.68 / −0.47 | −0.15 / −0.52 |
| Test + Cetran | 0.82 / 0.95 | 0.62 / 0.80 | 0.53 / 0.74 | −0.66 / −0.44 | −0.26 / −0.52 |

The largest |ρ| with improvement over δ:
- All splits: precision at δ = 0.3, with 0.79 / 0.80 / 0.84.
- Single-class splits: precision at δ = 0.3 on Cetran (0.71), and the OOD div % at δ = 1e-3 on Test (0.63) and on Test + Cetran (0.62).

**Table C — robust splits** (improvement > 0 on all three sets). There are now **33**, because {building} joined {overhead-bridge} and {gate}. The table shows the top 10 by worst-set improvement, with the divided shares on Test + Cetran at δ = 0.05; flat MSP's precision there is 7.3 %.

| split | classes (A) | imp. Cetran / Test / T+C | OOD div % | ID div % | prec % |
| --- | --- | --- | --- | --- | --- |
| `s0.1.3.5.10.16` | car, bicycle, truck, person, unpaved-road, overhead-bridge | +17.9 / +19.0 / +18.7 | 27.7 | 0.57 | 47.5 |
| `s2.3.4.5.6.16` | motorcycle, truck, bus, person, rider, overhead-bridge | +16.9 / +22.2 / +21.0 | 26.4 | 0.50 | 49.5 |
| `s0.2.5.16.17.18.19` | car, motorcycle, person, overhead-bridge, gate, pole-like-object, drain | +19.7 / +15.1 / +17.3 | 28.6 | 0.74 | 41.8 |
| `s2.3.10.12.17.18.19` | motorcycle, truck, unpaved-road, building, gate, pole-like-object, drain | +14.5 / +19.4 / +18.0 | 58.8 | 3.23 | 25.4 |
| `s10.12.17.18.19` | unpaved-road, building, gate, pole-like-object, drain | +12.4 / +20.1 / +17.9 | 56.5 | 2.99 | 26.1 |
| `s0.1.2.3.4.6.7.10.16.17.18` | car, bicycle, motorcycle, truck, bus, rider, traffic-sign, unpaved-road, overhead-bridge, gate, pole-like-object | +21.9 / +12.3 / +14.8 | 36.0 | 0.76 | 46.8 |
| `s3.16.18` | truck, overhead-bridge, pole-like-object | +14.9 / +12.2 / +12.7 | 28.5 | 0.54 | 49.7 |
| `s7.13.16.17.23` | traffic-sign, window, overhead-bridge, gate, obscurant | +11.8 / +12.0 / +12.2 | 25.8 | 0.75 | 39.1 |
| `s10.12.17.19` | unpaved-road, building, gate, drain | +10.8 / +21.9 / +18.3 | 55.2 | 2.83 | 26.7 |
| `s2.16` | motorcycle, overhead-bridge | +10.0 / +32.0 / +26.2 | 16.2 | 0.19 | 61.9 |

29 of the 33 contain overhead-bridge, gate or building. 29 contain at least one class that has **no ID point in Cetran**.

**Figures** (in `$W/divided_mass/`):
- `bubble_singletons_0.05` and `bubble_singletons_0.001`: ID vs OOD divided share of the 24 single-class splits, with flat MSP's star and its iso-ratio line;
- `bubble_robust_0.05`: the robust splits over the grey sweep, with the top 8 numbered and keyed;
- `rho_vs_threshold`: the ρ of each statistic against each metric delta, over δ.

**Reading.**
- **Mechanism.** Group MSP flags only divided points, so a split trades OOD uncertainty for ID uncertainty. It beats flat when it keeps much more of flat MSP's uncertain OOD points than of its uncertain ID points.
  - At δ = 0.05 every single-class winner does: selectivity 1.8–31 and precision 10–78 %, against flat's 6–11 %.
  - The strong winners (improvement > +10) reach a selectivity of 4.7–31 and a precision of 24–78 %.
  - {vegetation} is the clearest loser, with a selectivity of 0.77–1.09: it absorbs OOD uncertainty as fast as ID uncertainty.
- **A clean ratio is necessary, not sufficient.** Necessary: among all 503 splits, no split with improvement > 0 has a selectivity below 1 at δ = 0.05, on any set. Not sufficient: several losers also have a clean ratio, such as {bicycle} and {bus} on Test (selectivity 1.7–2.0), but they divide under 1 % of the OOD points. FPR@95 is decided by the depth δ95, because the ID divided share there *is* the exact FPR@95.
  - {gate} and {bicycle} on Test look alike at δ = 0.05: OOD div 0.53 vs 0.48 %, precision 10.3 vs 10.1 %.
  - Their depths differ 400-fold: δ95 = 4.0e-7 vs 9.3e-10. The other 99.5 % of the OOD points are confidently non-bicycle.
  - So their ID div at δ95 is 41.5 vs 80.4 %, against flat's 51.2, and their improvements are +6.6 vs −40.1.
- **Which statistic predicts what.**
  - Divided precision (the absolute-count view) is the best single predictor over all splits. At δ = 0.05, ρ is 0.87–0.95 with ΔAP and 0.72–0.74 with improvement, rising to 0.79–0.84 at δ = 0.3 (`rho_vs_threshold`).
  - The ID divided % alone is a weak, negative predictor.
  - Over the single-class splits, the OOD divided share tracks ΔFPR@95, with ρ −0.43 / −0.68 / −0.66.
- **Robust splits come in two kinds** (Table C). Both reach a divided precision 3.5–8.5× flat's.
  - Splits led by overhead-bridge divide about 16–36 % of the OOD points at an ID cost of only 0.2–0.8 %, with a precision of 39–62 %.
  - Splits built on building, gate and drain divide 55–59 % of the OOD points at an ID cost of 2.8–3.2 %, with a precision of about 26 %.
- **Cetran vs Test: the OOD population shifts.**
  - Cetran's best single-class splits are truck, gate, overhead-bridge and bicycle. Their ID divided share is tiny on both sets: truck 0.29 → 0.25 %, gate 0.10 → 0.07 %, bicycle 0.04 → 0.07 %. On Cetran that is because it has no ID point of these classes; on Test it holds even though those ID points exist.
  - What changes is the OOD side. Test's OOD points lean toward these classes far less: truck 15.6 → 6.4 %, gate 11.4 → 0.53 %, bicycle 1.65 → 0.48 %. The ID:OOD ratio also doubles, from 33× to 63×.
  - So truck and bicycle do not transfer, and gate weakens from +17.5 to +6.6. Overhead-bridge's OOD share grows (11.1 → 18.2 %), and it becomes the best split on Test.
  - This is the poor transfer noted on 2026-09-15: a split picked on Cetran is tuned to Cetran's OOD objects.

**Caveat.** Divided shares are ROC operating points of Group MSP, so their link to the metric deltas is partly by construction. The explanation rests on the comparison with flat MSP (selectivity, precision) and on δ95. The splits are still selected on the evaluation data, so any split must be confirmed online before it is reported.
