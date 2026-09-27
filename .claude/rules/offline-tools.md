# Offline analysis tools (`tools/`)

These read test.py logs or logit dumps; none of them runs the model.

- `make_dso_hierarchy_variants.py`: generates the `configs/p3former/hier/` configs. They cover every set partition of the six base groups plus one split hierarchy, 203 hierarchies in 17 batches, on the Cetran split.
- `summarize_hierarchy_ablation.py LOG... --family group|gn`: prints the per-hierarchy Δ vs flat and `improvement` = mean ΔAUROC + mean ΔAP − mean ΔFPR@95, with an optional `--plot`. It refuses `--family gn` on sweep logs.
- `sweep_bipartitions.py DUMP_DIR...`: scores random two-group class partitions from the logit dumps. Several directories are evaluated as one split, and the test dump plus the Cetran dump equal `dso_infos_test_cetran.pkl`. Use `--backend torch --device cuda:N` for the large splits. Its GN MSP / GN Entropy rows are approximate, so don't rank the GN family from it.
- `plot_ood_score_distributions.py DUMP_DIR`: plots ID vs OOD Group-MSP densities per group.
