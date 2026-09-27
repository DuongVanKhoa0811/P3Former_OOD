---
paths:
  - "p3former/**/*.py"
  - "evaluation/**/*.py"
  - "tools/**/*.py"
  - "tests/**/*.py"
---

# torch 1.10 CUDA pitfalls on these GPUs

- CUDA matmuls default to TF32; set `torch.backends.cuda.matmul.allow_tf32 = False` when probabilities must be exact.
- CUDA `torch.bincount` is extremely slow when most values share one bin. `torch_counts` in `tools/sweep_bipartitions.py` is a sort-based replacement.
