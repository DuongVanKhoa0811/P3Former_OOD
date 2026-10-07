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
- cuSOLVER-backed ops (`torch.linalg.cholesky`, `cholesky_ex`, `inv` / `torch.inverse`, `eigh`) raise `CUSOLVER_STATUS_INTERNAL_ERROR` on CUDA tensors, in float32 and float64: the GPUs are sm_89 and torch 1.10.1+cu111 is built up to sm_86. Run factorizations on the CPU. Matmul, `torch.triangular_solve` and `logsumexp` work on the GPU.
