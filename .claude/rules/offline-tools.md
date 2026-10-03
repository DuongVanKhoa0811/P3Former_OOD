# Offline analysis tools (`tools/`)

These tools read test.py logs, the logit dumps, or the feature samples. Only `extract_point_features.py` runs the model. Each tool's module docstring lists its inputs, its outputs and a run command.

- `make_dso_hierarchy_variants.py`: generates the `configs/p3former/hier/` configs. They cover every set partition of the six base groups plus one split hierarchy, 203 hierarchies in 17 batches, on the Cetran split.
- `summarize_hierarchy_ablation.py LOG... --family group|gn`: prints the per-hierarchy Δ vs flat and `improvement` = mean ΔAUROC + mean ΔAP − mean ΔFPR@95, with an optional `--plot`. It refuses `--family gn` on sweep logs.
- `sweep_bipartitions.py DUMP_DIR...`: scores random two-group class partitions from the logit dumps.
  - Several directories are evaluated as one split; the test dump plus the Cetran dump equal `dso_infos_test_cetran.pkl`.
  - Use `--backend torch --device cuda:N` for the large splits.
  - Its GN MSP / GN Entropy rows are approximate, so don't rank the GN family from it.
  - `--subsets singletons` (or a JSON list of subsets) scores explicit splits instead of random ones, e.g. all 24 single-class splits. It requires `--out-dir`, so the random sweep's `bipartitions/` is never overwritten.
- `plot_ood_score_distributions.py DUMP_DIR`: plots ID vs OOD Group-MSP densities per group.
- `divided_mass.py [--backend torch --device cuda:N]`: for every split scored by the sweeps (`bipartitions*/`, `singletons*/`), histograms the divided mass m = min(P_A, P_B) from the logit dumps, and flat MSP's u = 1 − max p. It writes `divided_mass/`:
  - `histograms.npz` and one `<set>.tsv` per set, holding the divided shares, precision, selectivity, δ95 and the joined metric deltas at each δ;
  - `flat.tsv`, `robust.tsv` and `rho.tsv`;
  - `summary.md`, with consistency checks against the sweep logs and the float32-tie section.

  It takes ~7 min on a GPU.
- `plot_divided_mass.py [DIVIDED_DIR] [--threshold δ] [--top N]`: writes `bubble_singletons_<δ>`, `bubble_robust_<δ>` (top N ≤ 10 keyed by rank) and `rho_vs_threshold` into `divided_mass/`.
- `extract_point_features.py CONFIG CKPT --ann PKL --out-dir DIR [--check-dump LOGITS_DIR]`: runs the model once on a GPU (~25 GB) and saves weighted per-frame samples of `pe_features`, the semantic classifier's input, together with the positional embedding.
  - Per frame it keeps up to 64 ID points per class and 512 OOD points.
  - It refuses a non-empty `--out-dir`.
  - `--check-dump` verifies the recomputed logits against a logit dump.
  - Large outputs go on `/mnt/sandisk` (`features_{cetran,test}`).
- `ood_class_resemblance.py [--k 10] [--reference SET] [--chunk N] [--out-dir DIR]`: kNN resemblance of the OOD samples to each ID class, plus per-split statistics. It writes `resemblance/` (`profile_*`, `splits_*`, `placement.tsv`, `rho.tsv` and `summary.md`).
  - Per class: r_OOD, r_ID and the contrast.
  - Per split: R_A, E_A, the feature-divided shares and the literal placement test.
  - A class with no ID samples in a set is left out of that set's bank and reported as NaN, not 0.
  - `--reference test_cetran` measures every set against the Test + Cetran bank.
- `plot_ood_class_resemblance.py [RES_DIR] [--space full|appearance]`: the profile, feature-bubble and hypothesis figures.
- `plot_feature_tsne.py [--space full|appearance]`: a t-SNE of the feature samples, with one fixed colour per class (`CLASS_COLOURS`) and OOD in black.
