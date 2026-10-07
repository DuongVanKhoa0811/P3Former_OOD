"""OCCUQ density score from fitted class-conditional Gaussians.

The score is OCCUQ's log q(z) = logsumexp_c [log N(z; mu_c, Sigma_c) +
log pi_c], with empirical class priors (official code
tools/gmm_utils.py:17-44 and 291-299 at commit 2aa5429).

The emitted OOD score is asinh(-log q):
- higher means more OOD;
- it is strictly increasing in -log q, so the exact AUROC, AP and FPR@95
  are those of -log q;
- the compression keeps the metric's equal-width histogram bins fine when a
  few far-off points have a very large -log q.

Files come from tools/fit_occuq_gmm.py.
"""
import contextlib
import hashlib
import math
import os.path as osp
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn

GMM_KEYS = ('means', 'prec_chol', 'log_det_prec', 'log_prior')


@contextlib.contextmanager
def no_tf32():
    """Disable TF32 CUDA matmuls inside the block, then restore the previous
    value, even after an exception (torch 1.10 enables TF32 by default)."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def fingerprint(modules: Iterable[Tuple[str, nn.Module]]) -> str:
    """SHA-256 over the state_dict entries of the named modules, sorted by
    full name, as float32 bytes. Spectral norm's weight_orig, weight_u and
    weight_v are state_dict entries, so they count."""
    entries = {}
    for prefix, module in modules:
        for name, tensor in module.state_dict().items():
            entries[f'{prefix}.{name}'] = tensor
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(name.encode())
        digest.update(entries[name].detach().float().cpu().contiguous()
                      .numpy().tobytes())
    return digest.hexdigest()


def load_gmm(path: str, features: str, expected_fingerprint: str,
             device) -> Dict[str, torch.Tensor]:
    """Load a tools/fit_occuq_gmm.py file as float32 tensors on ``device``.
    Refuses a file fitted on other features or for other weights."""
    if not osp.isfile(path):
        raise FileNotFoundError(
            f'{path} not found: fit it with tools/fit_occuq_gmm.py CONFIG '
            f'CHECKPOINT --features {features} --out {path}')
    data = torch.load(path, map_location='cpu')
    if data['features'] != features:
        raise ValueError(f'{path} holds Gaussians of {data["features"]} '
                         f'features, not {features}')
    if data['fingerprint'] != expected_fingerprint:
        raise ValueError(
            f'{path} was fitted for other weights (fingerprint '
            f'{data["fingerprint"][:12]}, this model '
            f'{expected_fingerprint[:12]}): refit it for this checkpoint')
    return {
        key: data[key].to(device=device, dtype=torch.float32)
        for key in GMM_KEYS
    }


def log_density(z: torch.Tensor, gmm: Dict[str, torch.Tensor],
                chunk: int = 65536) -> torch.Tensor:
    """log q(z) for voxel features z [V, D], in the dtype of gmm['means'].

    Centred before the product, chunked over voxels, TF32 off."""
    means, prec, log_det, log_prior = (gmm[key] for key in GMM_KEYS)
    num_classes, dim = means.shape
    const = 0.5 * dim * math.log(2 * math.pi)
    out = []
    with no_tf32():
        for start in range(0, z.shape[0], chunk):
            zc = z[start:start + chunk].to(means.dtype)
            per_class = zc.new_empty((zc.shape[0], num_classes))
            for c in range(num_classes):
                y = (zc - means[c]) @ prec[c]
                per_class[:, c] = log_det[c] - const - 0.5 * y.pow(2).sum(1)
            out.append(torch.logsumexp(per_class + log_prior, dim=1))
    return torch.cat(out) if out else z.new_zeros((0, ), dtype=means.dtype)


def density_score(z: torch.Tensor, gmm: Dict[str, torch.Tensor],
                  chunk: int = 65536) -> torch.Tensor:
    """asinh(-log q(z)) per voxel: higher = more OOD."""
    log_q = log_density(z, gmm, chunk)
    if not bool(torch.isfinite(log_q).all()):
        raise FloatingPointError('non-finite OCCUQ log-density: check the '
                                 'Gaussian file and the features')
    return torch.asinh(-log_q)


def point_density_scores(voxel_features: torch.Tensor,
                         gmm: Dict[str, torch.Tensor],
                         point2voxel_map: torch.Tensor,
                         chunk: int = 65536) -> np.ndarray:
    """Per-voxel density scores projected to points: float32 numpy [N], as
    point_ood_scores does for the flat scores."""
    scores = density_score(voxel_features.detach(), gmm, chunk)
    return scores[point2voxel_map.long()].cpu().numpy().astype(np.float32)
