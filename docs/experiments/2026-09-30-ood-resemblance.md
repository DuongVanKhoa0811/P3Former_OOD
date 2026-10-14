# 2026-09-30 — Do the best splits group the classes the OOD points resemble? (feature space)

Full record of the `DOCs.md` entry of the same date ([DOCs.md](../../DOCs.md)). Paths are relative to the repo root.

The question from 09-29: which ID classes do the OOD points look like in the features the classifier reads? And does the best split "put the classes that the OOD objects resemble on one side and everything else on the other", as the 09-15 reading suggested?

*Re-run 2026-10-04 after a sampling fix (`84521b0`).* The kNN bank and the ID queries are now two halves of one shuffled weighted draw. Before, the queries took the low-weight leftovers of the draw, which inflated the ID feature-divided shares by about 0.5 point on Test (0.18–0.76 over seeds 0–2) and not measurably on the other sets. No conclusion changed. Seeds 1 and 2 were added as a noise check.

**Features.** `pe_features` is the 256-d per-point input of the semantic classifier (`sem_preds = pe_features @ sem_queriesᵀ`, bias-free). "Appearance" is `pe_features` minus the positional embedding added just before it. `tools/extract_point_features.py` captures both by wrapping `_P3FormerHead.init_inputs` on the model instance.
- Per frame it keeps up to 64 ID points per class and 512 OOD points, each weighted to its (frame, class) stratum.
- Cetran: 980 frames, 1,252,847 samples, 501,760 of them OOD. Test: 2,625 frames, 2,995,232 samples, 1,034,359 OOD.
- Checks: the captured features reproduce the classifier's logits (`feat @ W^T` in float64) within 0.0073 / 0.0077. At the sampled points, the run's logits, labels and OOD flags equal those of the logit dumps; the largest logit difference is 0.0.

**Measure** (`tools/ood_class_resemblance.py`): a cosine kNN (k = 10) against a class-balanced bank of ID samples. Per class, one weighted draw (∝ sample weight) is shuffled and split into the bank (up to 4,000) and disjoint ID queries (up to 2,000).
- r_OOD(c) is the share of class c among the OOD points' neighbours; 4.17 % means no preference.
- r_ID(c) is the same share around the ID points of the other classes, and contrast = r_OOD / r_ID.
- Per split A | B, with A the smaller group:
  - R_A = Σ r_OOD(c) over the measured classes of A, and E_A = R_A over its no-preference value ("together": the neighbours fall in A);
  - *feature-divided*: the k neighbours hold classes of both sides ("cut through"), with its log OOD/ID ratio;
  - the *literal* test: the splits grouped by where the k most-resembled classes sit.
- **Cetran has no ID point of 10 classes**: bicycle, motorcycle, truck, rider, unpaved-road, window, overhead-bridge, gate, drain and obscurant. Against its own bank (the default) they cannot be measured, so they are reported as NaN; that includes truck and gate, its two best splits. `--reference test_cetran` measures every set's OOD points against the Test + Cetran bank instead.

```bash
W=work_dirs/p3former_2xb1_3x_dso_ood_dump; CFG=configs/p3former/p3former_2xb1_3x_dso_ood.py; CKPT=work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
python tools/extract_point_features.py $CFG $CKPT --ann dso_infos_cetran.pkl --out-dir $W/features_cetran --check-dump $W/logits     # ~6 min, GPU
python tools/extract_point_features.py $CFG $CKPT --ann dso_infos_test.pkl --out-dir $W/features_test --check-dump $W/logits_test  # ~7 min
python tools/ood_class_resemblance.py --device cuda:0                                               # -> $W/resemblance/ (~1.5 min)
python tools/ood_class_resemblance.py --device cuda:0 --seed 1 --out-dir $W/resemblance_seed1      # and --seed 2 -> resemblance_seed2
python tools/ood_class_resemblance.py --device cuda:0 --k 50 --out-dir $W/resemblance_k50
python tools/ood_class_resemblance.py --device cuda:0 --reference test_cetran --out-dir $W/resemblance_xref
python tools/plot_ood_class_resemblance.py [$W/resemblance_xref] [--space appearance]
python tools/plot_feature_tsne.py [--space appearance]
```

**Table A — what the OOD points resemble**: the top 8 by r_OOD (full space), as r_OOD % / r_ID %, with the improvement of {c} | rest in brackets.

| rank | Cetran, own bank | Cetran, T+C bank | Test | Test + Cetran |
| --- | --- | --- | --- | --- |
| 1 | person 31.5 / 0.16 (−7.1) | gate 29.1 / 0.88 (+17.5) | building 26.0 / 0.61 (+16.3) | building 19.3 / 0.47 (+13.3) |
| 2 | perimeter-barrier 26.6 / 1.05 (−11.7) | perimeter-barrier 16.3 / 1.74 (−11.7) | perimeter-barrier 15.6 / 3.18 (+5.5) | gate 16.5 / 0.88 (+10.1) |
| 3 | other-barrier 11.6 / 1.41 (−13.8) | person 12.1 / 0.10 (−7.1) | gate 9.0 / 0.73 (+6.6) | perimeter-barrier 13.4 / 1.74 (+1.2) |
| 4 | building 7.8 / 0.30 (+4.3) | other-barrier 7.7 / 0.82 (−13.8) | other-barrier 5.2 / 0.94 (−1.3) | person 6.0 / 0.10 (−46.7) |
| 5 | terrain 5.6 / 0.93 (−52.4) | truck 5.1 / 0.19 (+25.2) | sidewalk 4.3 / 1.01 (−23.5) | other-barrier 5.8 / 0.82 (−5.2) |
| 6 | pole-like-object 3.3 / 0.14 (−18.0) | sidewalk 4.4 / 2.52 (−52.2) | traffic-sign 4.2 / 0.29 (−9.1) | sidewalk 4.1 / 2.52 (−29.9) |
| 7 | sidewalk 2.9 / 10.05 (−52.2) | building 4.4 / 0.47 (+4.3) | drain 4.2 / 0.99 (−24.2) | bus 3.4 / 0.08 (−7.0) |
| 8 | traffic-cone 2.7 / 0.30 (−29.6) | traffic-sign 3.9 / 0.19 (−5.2) | bus 4.1 / 0.08 (−6.1) | trunks 3.3 / 8.32 (−43.7) |

**Table B — class level**: Spearman ρ over the classes, full / appearance space. For Cetran with its own bank, ρ is over the 14 measured classes.

| run | set | r_OOD ~ improvement | r_OOD ~ OOD div % | contrast ~ improvement |
| --- | --- | --- | --- | --- |
| k = 10 | Cetran (own bank) | 0.34 / 0.51 | 0.24 / 0.24 | 0.75 / 0.83 |
| k = 10 | Cetran (T+C bank) | 0.61 / 0.69 | 0.30 / 0.33 | 0.70 / 0.73 |
| k = 10 | Test | 0.76 / 0.92 | 0.62 / 0.64 | 0.24 / 0.28 |
| k = 10 | Test + Cetran | 0.58 / 0.79 | 0.53 / 0.54 | 0.28 / 0.35 |
| k = 50 | Cetran (own bank) | 0.32 / 0.47 | 0.31 / 0.23 | 0.67 / 0.85 |
| k = 50 | Test | 0.73 / 0.91 | 0.61 / 0.63 | 0.22 / 0.30 |
| k = 50 | Test + Cetran | 0.61 / 0.80 | 0.53 / 0.52 | 0.35 / 0.37 |

**Table C — split level**: ρ with the improvement over all 503 splits, full / appearance.

| set | together: R_A | together: E_A | cut through: OOD feature-divided | cut through: log ratio |
| --- | --- | --- | --- | --- |
| Cetran (own bank) | 0.19 / 0.24 | 0.23 / 0.31 | 0.05 / 0.21 | 0.62 / 0.70 |
| Cetran (T+C bank) | 0.34 / 0.36 | 0.38 / 0.42 | 0.36 / 0.38 | 0.60 / 0.63 |
| Test | 0.42 / 0.47 | 0.48 / 0.59 | 0.33 / 0.42 | 0.48 / 0.53 |
| Test + Cetran | 0.40 / 0.44 | 0.47 / 0.55 | 0.34 / 0.42 | 0.60 / 0.64 |

**Seed noise** (seeds 0–2, k = 10, full / appearance):
- ρ(r_OOD, improvement) spans 0.73–0.76 / 0.90–0.92 on Test, 0.58–0.61 / 0.79–0.81 on Test + Cetran, and 0.29–0.34 / 0.43–0.53 on Cetran's own bank.
- The contrast moves most, by up to 0.12.
- The split-level E_A and log-ratio ρ move by at most 0.03.
- The top three classes are the same for every seed.

**Literal reading** (`placement.tsv`, full space): the splits grouped by where the 2 (3) most-resembled classes sit. Each cell gives the median improvement, the % of splits with improvement > 0, and n.

| set | classes | small side (all in A) | large side (all in B) | apart |
| --- | --- | --- | --- | --- |
| Cetran (own bank) | person, perimeter-barrier | −12.3, 2 % (n = 41) | −15.7, 17 % (276) | −12.7, 8 % (186) |
| Cetran (T+C bank) | gate, perimeter-barrier | −15.45, 0 % (n = 44) | −20.4, 11 % (265) | −10.4, 16 % (194) |
| Test | building, perimeter-barrier | −2.7, 45 % (40) | −18.3, 12 % (269) | −2.8, 45 % (194) |
| Test | + gate | −5.8, 33 % (12) | −21.5, 11 % (194) | −4.5, 37 % (297) |
| Test + Cetran | building, gate | −2.6, 29 % (41) | −18.3, 15 % (256) | −7.3, 24 % (206) |
| Test + Cetran | + perimeter-barrier | −11.6, 0 % (12) | −25.1, 9 % (194) | −6.1, 28 % (297) |

**Table D — the Part 1 robust splits in the features** (Test + Cetran, full space; the log ratio in the appearance space in brackets).

| split | R_A % | E_A | OOD feat-div % | ID feat-div % | log ratio | imp |
| --- | --- | --- | --- | --- | --- | --- |
| `s0.1.3.5.10.16` | 14.4 | 0.58 | 33.4 | 16.1 | 0.32 (0.45) | +18.7 |
| `s2.3.4.5.6.16` | 15.3 | 0.61 | 33.6 | 4.5 | 0.88 (1.08) | +21.0 |
| `s0.2.5.16.17.18.19` | 30.3 | 1.04 | 57.9 | 13.9 | 0.62 (0.68) | +17.3 |
| `s2.3.10.12.17.18.19` | 46.2 | 1.58 | 68.1 | 27.2 | 0.40 (0.41) | +18.0 |
| `s10.12.17.18.19` | 42.2 | 2.03 | 66.3 | 26.5 | 0.40 (0.41) | +17.9 |
| `s0.1.2.3.4.6.7.10.16.17.18` | 35.0 | 0.76 | 64.1 | 22.2 | 0.46 (0.47) | +14.8 |
| `s3.16.18` | 6.2 | 0.49 | 17.5 | 5.2 | 0.53 (0.90) | +12.7 |
| `s7.13.16.17.23` | 24.4 | 1.17 | 54.0 | 12.0 | 0.65 (0.74) | +12.2 |
| `s10.12.17.19` | 40.3 | 2.42 | 64.2 | 24.4 | 0.42 (0.41) | +18.3 |
| `s2.16` | 2.3 | 0.28 | 6.4 | 3.0 | 0.33 (0.84) | +26.2 |

**Figures** (`$W/resemblance/`, and the same names in `$W/resemblance_xref/`):
- `profile_{full,appearance}`: r_OOD and r_ID per class, with the appearance-space r_OOD as rings;
- `bubble_feature_singletons_{full,appearance}`: OOD vs ID feature-divided share of the single-class splits;
- `hypothesis_{full,appearance}`: improvement against R_A, the OOD feature-divided share and the log ratio;
- `tsne_{full,appearance}`, in `$W/resemblance/` only: t-SNE of the feature samples, with one fixed colour per class and OOD in black.

**Reading.**
- **What the OOD points resemble: mostly structures.**
  - On Test: building (26 %), perimeter-barrier (16 %) and gate (9 %).
  - On Test + Cetran: building, gate and perimeter-barrier (19 / 17 / 13 %).
  - On Cetran against the full bank: gate leads (29 %), then perimeter-barrier and person (16 / 12 %). Truck is fifth (5.1 %), with a contrast of 28.

  The resemblance is spread: the top class takes 19–31 % of the neighbours, and the top three about half to 70 %.
- **At the class level, resemblance matches the winning splits, with one exception.**
  - On Test, the three most-resembled classes are exactly the three positive single-class splits besides overhead-bridge.
  - Over the classes, ρ(r_OOD, improvement) is 0.76 on Test, 0.58 on Test + Cetran and 0.61 on Cetran with the full bank. It is only 0.34 over Cetran's own 14 measured classes, which lack truck and gate.
  - r_OOD also tracks Part 1's OOD divided share (ρ 0.62 / 0.53 / 0.30): OOD points that neighbour class c in the features also carry classifier mass on c.
  - **Overhead-bridge is the exception.** It is the best split on Test and Test + Cetran (+34.8 / +27.4), yet its r_OOD is only 1.4 %, below no preference. In the appearance space it rises to 6.0 %, and 22 % of the OOD points are feature-divided on it, against 2.6 % of the ID points. So the positional part of `pe_features` hides this resemblance from the kNN, while the linear classifier still puts overhead-bridge mass on 18 % of the OOD points (Part 1).
- **The appearance space**, without the positional embedding, strengthens the class-level alignment with the improvement: ρ(r_OOD, improvement) is 0.92 / 0.79 / 0.69 on Test / Test + Cetran / Cetran (full bank), against 0.76 / 0.58 / 0.61. The top classes stay the same.
- **At the split level, the literal hypothesis gets mixed support.**
  - Putting the two or three most-resembled classes together on the small side is about as good as separating them on Test (45 vs 45 % of the splits positive). On Test + Cetran it is better for the top two (29 vs 24 %) but worse for the top three (0 vs 28 %). On Cetran it is worse for the top two (0–2 vs 8–16 %) and mixed for the top three (0–7 vs 6–13 %).
  - What consistently hurts on Test and Test + Cetran is lumping those classes with everything else: the large side has a median improvement of −18 to −25.
  - An example from Test: {overhead-bridge} scores +34.8 and {building} +16.3, but {building, overhead-bridge} only +14.3, although its E_A (3.3) is ten times {overhead-bridge}'s (0.34).
  - Over all splits, "together" (E_A) predicts as well as the "cut through" log ratio on Test and Test + Cetran (0.47–0.59 against 0.48–0.64). On Cetran it is weak (0.23 with its own bank, 0.38 with the full bank), while the log ratio holds (0.62 / 0.60). The log ratio is the more consistent predictor: ρ 0.48–0.70 on every set, in both spaces and with either bank.
  - The top robust splits (Table D) have E_A from 0.28 to 2.4, so "togetherness" is not what they share. Their OOD points are feature-divided 2–7.5× more often than their ID points. Over all 33 robust splits on Test + Cetran, the median log ratio is 0.62, against 0.23 for the other splits, though it goes down to 0.11.
- **The hypothesis, refined.** The best split cuts *through* the OOD points' neighbourhoods. It puts one (or a few) of the classes the OOD points lean toward against the rest. The OOD mass, spread over several resembled classes, then straddles the boundary, while each ID class's usual confusers stay on its own side. This is Part 1's divided precision seen from the features. The feature ratio ranks the splits less well than the logit-space precision does (ρ 0.48–0.70 vs 0.72–0.84), so the neighbourhoods explain part of the effect, not all of it.
- **k = 50.** The r_OOD ranking barely moves: its rank ρ with k = 10 is 0.955–0.991.
  - The top three classes are the same on Test and Test + Cetran in both spaces, and on Cetran's own bank in the full space.
  - In Cetran's appearance space the third class changes, from building to other-barrier.
  - The headline ρ of Tables B and C move by at most 0.08. At the class level that is about the seed noise; at the split level it is up to twice the noise.
- **t-SNE** (`tsne_{full,appearance}`; 300 ID points per class and 1,000 OOD points per set). This view is qualitative: t-SNE keeps neighbourhoods, not distances.
  - In the appearance space, the ID classes form clean clusters. The OOD points gather in a region of their own, which borders building, window, perimeter-barrier, gate and other-barrier on Test and Test + Cetran, and perimeter-barrier, other-barrier and person on Cetran. This is the kNN picture of Table A.
  - In the full space, position splits every class into several clusters, and the OOD points spread among them, mostly next to the barriers, building and window. On Cetran, the Stop points form small tight clusters of their own.
- **Caveats.**
  - The test is correlational only, over splits selected on the evaluation data.
  - The kNN reads neighbourhoods, while the classifier is linear on the same features, and overhead-bridge shows the two can disagree.
  - Cetran's own-bank numbers cover only its 14 measured classes.
  - Differences in ρ smaller than the seed noise (up to 0.12 at the class level, 0.03 at the split level) are not meaningful.
