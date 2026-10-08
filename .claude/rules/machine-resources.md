# Running on this machine

- **Shared hardware.** 2× RTX 6000 Ada (46 GB each) and 251 GB RAM, shared with another user. Check `nvidia-smi` first and run one eval per GPU. An OOD eval needs ~17 GB of GPU memory for SemanticKITTI and ~25 GB for DSO; a GPU OOM shows up as spconv `cuda execution failed with error 2`.
- **Evaluator RAM.** `_OODPointMetric` holds every score of every valid point until the end, about (4 × #scores + 1) bytes per point. DSO test + Cetran (1.28B points) with 15 scores needs ≈ 80 GB. Each `class_groups_variants` entry adds 10 scores, which is why the hierarchy batches have 12 hierarchies each and run on the smaller Cetran split.
- **Disk.** `/` is ~95% full. Large dumps live on `/mnt/sandisk/khoadv/` and are symlinked into `work_dirs/` (e.g. `work_dirs/p3former_2xb1_3x_dso_ood_dump/logits_test`).
- **Long jobs.** Claude Code background tasks are killed when the session ends. Start multi-hour jobs detached: `nohup setsid bash -c "<cmd> > <log> 2>&1" < /dev/null > /dev/null 2>&1 &`.
- **Another server.** When these GPUs are busy, work can move to another server. `tools/copy_to_server.sh` copies what git does not track (DSO data and pkls, checkpoints, result logs, or any `--path`) to the clone there; `--help` lists the tiers and the one-time setup. The server's address goes on the command line, never into the repo, which is public.
