---
paths:
  - "p3former/**/*.py"
  - "evaluation/**/*.py"
  - "tools/**/*.py"
  - "tests/**/*.py"
---

# torch 1.10 CUDA pitfalls on these GPUs

- CUDA matmuls default to TF32; set `torch.backends.cuda.matmul.allow_tf32 = False` when results must be exact (probabilities, covariances).
- CUDA `torch.bincount` is extremely slow when most values share one bin; count with a sort-based method instead.
