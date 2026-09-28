# Running on this machine

- **Shared hardware.** 2× RTX 6000 Ada (46 GB each) and 251 GB RAM, shared with another user. Check `nvidia-smi` first and run one eval per GPU. An OOD eval needs ~17 GB of GPU memory for SemanticKITTI and ~25 GB for DSO; a GPU OOM shows up as spconv `cuda execution failed with error 2`.
- **Evaluator RAM.** `_OODPointMetric` holds every score of every valid point until the end, about (4 × #scores + 1) bytes per point. DSO test + Cetran (1.28B points) with the five flat scores needs ≈ 27 GB, and each extra score adds ≈ 5 GB.
- **Disk.** `/` is ~95% full. Put large outputs (dumps, feature caches) on `/mnt/sandisk/khoadv/` and symlink them into `work_dirs/`.
- **Long jobs.** Claude Code background tasks are killed when the session ends. Start multi-hour jobs detached: `nohup setsid bash -c "<cmd> > <log> 2>&1" < /dev/null > /dev/null 2>&1 &`.
