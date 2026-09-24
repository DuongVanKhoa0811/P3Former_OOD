#!/usr/bin/env python
"""Score random two-group class partitions offline from a logits dump.

Every split of the 24 DSO classes into two groups A | B is a candidate
hierarchy for the Group / Group-Normalised scores. There are 2^23 - 1 of
them, so ``--num`` (default 500) are sampled and scored from the per-point
logits written by ``_OODLogitsDumpMetric`` (config
``p3former_2xb1_3x_dso_ood_dump.py``) -- no GPU pass. Sampling is
stratified by size: |A| is drawn uniformly from 1..12, then A uniformly
among the subsets of that size, so single-class-vs-rest splits are as
likely as balanced ones; duplicates are dropped. The vehicle-vs-rest
partition (``p_v_hgcno`` of the 2026-08-25 ablation) is always included as
a cross-check against the online numbers. A partition is named after the
ids of its smaller group, ``s<id>.<id>...`` (``s0.1.2.3.4`` = vehicle vs
rest); ``partitions.tsv`` lists the class names behind every name.

Scores follow p3former/utils/ood_scores.py exactly (``group_*`` / ``gn_*``
for msp, odin, energy, entropy; the MaxLogit variants are skipped since
the summariser always drops them). Metrics use the histogram formulas of
_OODPointMetric over the empirical range of every score (a first pass
over the dump records it) and the float16 logits of the dump. The
``--bins`` bins (2^16; 2^20 online) are log-spaced towards both ends of
each score's range (``--bin-eps``, see :func:`bin_index`), where the scores
pile up: equal-width bins, at 2^16, cannot separate very confident OOD
points from the ID mass and FPR@95 saturates at 100%. The flat, group_*
and gn_energy rows then agree with the online metric to ~0.05; gn_msp and
gn_entropy pile up at an interior value, -(1 - prior), which these bins do
not refine, and are approximate (FPR@95 can be off by many points).

Several dump directories are evaluated as one split: the Cetran dump plus
the held-out test dump (``p3former_2xb1_3x_dso_ood_dump_test.py``) is
exactly ``dso_infos_test_cetran.pkl``. ``--backend numpy`` (default) runs
on CPU worker processes; ``--backend torch --device cuda:0`` runs the same
two passes on a GPU, which the larger splits need (test + Cetran has 1.3B
points: hours on CPU, minutes on a GPU). The model is never run either way.

Outputs in ``--out-dir`` (default ``<dump_dir>/../bipartitions``; required
with several dump directories):
    bipartitions.log   ``method | AUROC | AP | FPR@95`` rows -- the flat
                       scores plus ``<name>_group_*`` / ``<name>_gn_*`` --
                       in the format tools/summarize_hierarchy_ablation.py
                       reads (deltas vs flat, improvement ranking, --plot)
    partitions.tsv / partitions.json   class composition of every name

Run from the repo root (on CPU the 980-frame Cetran dump takes about an
hour with 8 workers and ~25 GB RAM):
    python tools/sweep_bipartitions.py work_dirs/p3former_2xb1_3x_dso_ood_dump/logits
    python tools/summarize_hierarchy_ablation.py --family group --exclude odin \\
        work_dirs/p3former_2xb1_3x_dso_ood_dump/bipartitions/bipartitions.log
    # test + Cetran, on a GPU
    python tools/sweep_bipartitions.py --backend torch --device cuda:0 \\
        work_dirs/p3former_2xb1_3x_dso_ood_dump/logits_test \\
        work_dirs/p3former_2xb1_3x_dso_ood_dump/logits \\
        --out-dir work_dirs/p3former_2xb1_3x_dso_ood_dump/bipartitions_test_cetran
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool

# Each worker runs its own matmuls; keep BLAS from oversubscribing the box
# (must be set before numpy is imported).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from evaluation.functional.ood_eval import metrics_from_histograms  # noqa: E402
from summarize_hierarchy_ablation import (_improvement, parse_logs,  # noqa: E402
                                          summarise)

# Train ids of the 24-class DSO model (datasets/dso_dataset.py METAINFO).
CLASSES = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'rider',
           'traffic-sign', 'traffic-cone', 'paved-road', 'unpaved-road',
           'sidewalk', 'building', 'window', 'perimeter-barrier',
           'other-barrier', 'overhead-bridge', 'gate', 'pole-like-object',
           'drain', 'terrain', 'trunks', 'vegetation', 'obscurant')
NUM_CLASSES = len(CLASSES)
REFERENCE = (0, 1, 2, 3, 4)  # vehicle vs rest = p_v_hgcno

FLAT_KEYS = ('msp', 'maxlogit', 'odin', 'energy', 'entropy')
HIER_KEYS = ('group_msp', 'group_odin', 'group_energy', 'group_entropy',
             'gn_msp', 'gn_odin', 'gn_energy', 'gn_entropy')
ODIN_T = 1000.0
EPS = 1e-12  # p3former.utils.ood_scores._EPS
BIN_EPS = 1e-7  # finest histogram bin, as a fraction of the score range


# --------------------------------------------------------------- partitions
def canonical(subset):
    """The smaller side of the bipartition (the side holding class 0 when
    both have 12 classes), as a sorted tuple of train ids."""
    subset = tuple(sorted(int(c) for c in subset))
    if len(set(subset)) != len(subset) or not subset:
        raise ValueError(f'invalid subset {subset}')
    complement = tuple(c for c in range(NUM_CLASSES) if c not in subset)
    if not complement:
        raise ValueError('a bipartition needs two non-empty groups')
    if len(subset) > len(complement) or (len(subset) == len(complement)
                                         and 0 not in subset):
        subset = complement
    return subset


def partition_name(subset):
    return 's' + '.'.join(str(c) for c in subset)


def sample_bipartitions(num, seed=0):
    """``num`` distinct bipartitions: the reference first, then random ones
    with the smaller side's size uniform in 1..12."""
    rng = np.random.RandomState(seed)
    subsets = [canonical(REFERENCE)]
    seen = set(subsets)
    while len(subsets) < num:
        size = rng.randint(1, NUM_CLASSES // 2 + 1)
        subset = canonical(rng.choice(NUM_CLASSES, size, replace=False))
        if subset not in seen:
            seen.add(subset)
            subsets.append(subset)
    return subsets


def subsets_to_mask(subsets):
    mask = np.zeros((len(subsets), NUM_CLASSES), dtype=bool)
    for i, subset in enumerate(subsets):
        mask[i, list(subset)] = True
    return mask


# ------------------------------------------------------------------- scores
def load_frame(path):
    """(logits float32 [N, C], ood bool [N]) over the valid points."""
    with np.load(path) as data:
        valid = data['valid']
        return data['logits'][valid].astype(np.float32), data['ood'][valid]


def base_quantities(z):
    """Per-point quantities shared by every score: max logit ``m``, softmax
    ``p`` and its log, ODIN softmax ``pT``, ``e64`` = exp(z - m) in float64
    (group log-sum-exps without underflow) and the flat log-sum-exp."""
    m = z.max(axis=1)
    zc = z - m[:, None]
    lse = np.log(np.exp(zc).sum(axis=1))
    logp = zc - lse[:, None]
    zT = z / ODIN_T
    zT -= zT.max(axis=1, keepdims=True)
    pT = np.exp(zT)
    pT /= pT.sum(axis=1, keepdims=True)
    return dict(m=m, p=np.exp(logp), logp=logp, pT=pT,
                e64=np.exp(zc.astype(np.float64)), lse_all=m + lse)


def flat_scores(q):
    return OrderedDict([
        ('msp', -q['p'].max(axis=1)),
        ('maxlogit', -q['m']),
        ('odin', -q['pT'].max(axis=1)),
        ('energy', -q['lse_all']),
        ('entropy', -(q['p'] * q['logp']).sum(axis=1)),
    ])


def bipartition_scores(q, a_mask):
    """Group / GN scores of every bipartition in ``a_mask`` [c, C] (True =
    class in group A), each [c, N] float32 (one row per bipartition, so a
    row slice of points is contiguous), higher = more OOD."""
    c = a_mask.shape[0]
    M = a_mask.astype(np.float32)  # [c, C]
    M2 = np.concatenate([M, 1.0 - M], axis=0)  # A rows, then B rows
    P2 = M2 @ q['p'].T
    PT2 = M2 @ q['pT'].T
    lse2 = (np.log(M2.astype(np.float64) @ q['e64'].T)
            + q['m'].astype(np.float64)[None, :]).astype(np.float32)
    PA, PB = P2[:c], P2[c:]
    PTA, PTB = PT2[:c], PT2[c:]
    LA, LB = lse2[:c], lse2[c:]
    kA = a_mask.sum(axis=1).astype(np.float32)[:, None]
    kB = NUM_CLASSES - kA
    prior_A, prior_B = kA / NUM_CLASSES, kB / NUM_CLASSES
    log_kA, log_kB = np.log(kA), np.log(kB)
    PAc, PBc = np.maximum(PA, EPS), np.maximum(PB, EPS)
    qA, qB = np.maximum(PA - prior_A, 0.0), np.maximum(PB - prior_B, 0.0)
    qTA, qTB = np.maximum(PTA - prior_A, 0.0), np.maximum(PTB - prior_B, 0.0)
    return OrderedDict([
        ('group_msp', -np.maximum(PA, PB)),
        ('group_odin', -np.maximum(PTA, PTB)),
        ('group_energy', -np.maximum(LA, LB)),
        ('group_entropy', -(PAc * np.log(PAc) + PBc * np.log(PBc))),
        ('gn_msp', -np.maximum(qA, qB)),
        ('gn_odin', -np.maximum(qTA, qTB)),
        ('gn_energy', -np.maximum(LA - log_kA, LB - log_kB)),
        ('gn_entropy', -((qA + EPS) * np.log(qA + EPS)
                         + (qB + EPS) * np.log(qB + EPS))),
    ])


def bin_index(s, lo, hi, bins, eps=BIN_EPS):
    """Histogram bin of every score between ``lo`` and ``hi`` (scalars or,
    for a [c, N] block, per-row arrays).

    ``eps = 0``: equal-width bins, exactly as _OODPointMetric bins them.
    ``eps > 0``: bins are log-spaced towards both ends of the range. With
    t = (s - lo) / (hi - lo) and g(x) = log1p(x / eps) / log1p(1 / eps),
    the bin is (g(t) + 1 - g(1 - t)) / 2 -- monotone in s, so AUROC / AP /
    FPR@95 stay valid -- and a bin is ~5e-4 * (eps + distance to the nearer
    end) wide at 2^16 bins. Scores pile up at the ends: most ID points, and
    on the held-out test split more than 5% of the OOD points, lie within
    ~1e-5 of the minimum of the probability-based scores (less than one
    equal-width bin at 2^16 bins: FPR@95 then reads 100%), while the
    clipped GN scores pile up just below their maximum of exactly 0.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    hi = np.where(hi <= lo, lo + 1.0, hi)  # constant score: one bin
    span = hi - lo
    if s.ndim == 2:
        lo, span = lo.reshape(-1, 1), span.reshape(-1, 1)
    b = s.astype(np.float64)
    b -= lo
    if eps == 0:
        b *= (bins - 1) / span  # (s - smin) * scale, as online
    else:
        b /= span  # t; a true division, so that t = 1 exactly at s = hi
        np.clip(b, 0.0, 1.0, out=b)
        far = 1.0 - b
        b /= eps
        np.log1p(b, out=b)
        far /= eps
        np.log1p(far, out=far)
        b -= far  # log1p(t / eps) - log1p((1 - t) / eps)
        b *= 0.5 / np.log1p(1.0 / eps)
        b += 0.5
        b *= bins - 1
    b = b.astype(np.int64)
    np.clip(b, 0, bins - 1, out=b)
    return b


# ------------------------------------------------------------ torch backend
# The same scores and binning as above on a torch device, line for line.
# torch is imported on demand so that the numpy backend does not need it.
torch = None


def require_torch():
    global torch
    if torch is None:
        import torch as _torch
        # CUDA matmuls default to TF32 on Ampere and newer (torch 1.7-1.11):
        # a relative error of ~1e-4 in the group masses, which scrambles the
        # ranking of confident points whose uncertainty 1 - max P_g is ~1e-5
        # (FPR@95 moved by up to 57 points). Full float32 agrees with numpy
        # to ~5e-7.
        _torch.backends.cuda.matmul.allow_tf32 = False
        torch = _torch
    return torch


def torch_base_quantities(z):
    m = z.max(dim=1).values
    zc = z - m[:, None]
    lse = torch.log(torch.exp(zc).sum(dim=1))
    logp = zc - lse[:, None]
    zT = z / ODIN_T
    zT = zT - zT.max(dim=1, keepdim=True).values
    pT = torch.exp(zT)
    pT = pT / pT.sum(dim=1, keepdim=True)
    return dict(m=m, p=torch.exp(logp), logp=logp, pT=pT,
                e64=torch.exp(zc.double()), lse_all=m + lse)


def torch_flat_scores(q):
    return OrderedDict([
        ('msp', -q['p'].max(dim=1).values),
        ('maxlogit', -q['m']),
        ('odin', -q['pT'].max(dim=1).values),
        ('energy', -q['lse_all']),
        ('entropy', -(q['p'] * q['logp']).sum(dim=1)),
    ])


def torch_bipartition_scores(q, a_mask):
    """:func:`bipartition_scores` for a bool tensor ``a_mask`` [c, C]."""
    c = a_mask.shape[0]
    M = a_mask.float()
    M2 = torch.cat([M, 1.0 - M], dim=0)  # A rows, then B rows
    P2 = M2 @ q['p'].T
    PT2 = M2 @ q['pT'].T
    lse2 = (torch.log(M2.double() @ q['e64'].T)
            + q['m'].double()[None, :]).float()
    PA, PB = P2[:c], P2[c:]
    PTA, PTB = PT2[:c], PT2[c:]
    LA, LB = lse2[:c], lse2[c:]
    kA = a_mask.sum(dim=1).float()[:, None]
    kB = NUM_CLASSES - kA
    prior_A, prior_B = kA / NUM_CLASSES, kB / NUM_CLASSES
    log_kA, log_kB = torch.log(kA), torch.log(kB)
    PAc, PBc = PA.clamp_min(EPS), PB.clamp_min(EPS)
    qA, qB = (PA - prior_A).clamp_min(0.0), (PB - prior_B).clamp_min(0.0)
    qTA = (PTA - prior_A).clamp_min(0.0)
    qTB = (PTB - prior_B).clamp_min(0.0)
    return OrderedDict([
        ('group_msp', -torch.maximum(PA, PB)),
        ('group_odin', -torch.maximum(PTA, PTB)),
        ('group_energy', -torch.maximum(LA, LB)),
        ('group_entropy', -(PAc * torch.log(PAc) + PBc * torch.log(PBc))),
        ('gn_msp', -torch.maximum(qA, qB)),
        ('gn_odin', -torch.maximum(qTA, qTB)),
        ('gn_energy', -torch.maximum(LA - log_kA, LB - log_kB)),
        ('gn_entropy', -((qA + EPS) * torch.log(qA + EPS)
                         + (qB + EPS) * torch.log(qB + EPS))),
    ])


def torch_bin_index(s, lo, hi, bins, eps=BIN_EPS):
    """:func:`bin_index` with float64 tensors ``lo`` / ``hi``."""
    hi = torch.where(hi <= lo, lo + 1.0, hi)
    span = hi - lo
    if s.dim() == 2:
        lo, span = lo.reshape(-1, 1), span.reshape(-1, 1)
    b = s.double()
    b -= lo
    if eps == 0:
        # tensor / tensor: ``scalar / tensor`` is evaluated as reciprocal *
        # scalar, one ulp away from numpy's division
        b *= torch.full_like(span, bins - 1) / span
    else:
        b /= span
        b.clamp_(0.0, 1.0)
        far = 1.0 - b
        b /= eps
        b.log1p_()
        far /= eps
        far.log1p_()
        b -= far
        b *= 0.5 / float(np.log1p(1.0 / eps))
        b += 0.5
        b *= bins - 1
    b = b.long()
    b.clamp_(0, bins - 1)
    return b


def torch_counts(index, size):
    """``torch.bincount(index, minlength=size)`` for values in [0, size).

    The CUDA bincount kernel serialises on hot bins, and nearly all ID
    points share one bin: ~350 ms for a block of 64 partitions (torch
    1.10), against ~1 ms for this sort-based count.
    """
    if not index.is_cuda:
        return torch.bincount(index, minlength=size)
    dtype = torch.int32 if size < 2**31 else torch.int64
    values, counts = torch.unique(index.to(dtype), return_counts=True)
    out = torch.zeros(size, dtype=torch.int64, device=index.device)
    out[values.long()] = counts
    return out


def _prefetch(files, depth=8):
    """Yield ``load_frame(path)`` in order, read ahead by a few threads."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = deque()
        for path in files:
            pending.append(pool.submit(load_frame, path))
            if len(pending) >= depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _torch_passes(files, a_mask, bins, eps, chunk, device):
    """Both passes (score ranges, then histograms) in one process on
    ``device``; returns numpy (hist, flat_hist, counts) like the workers."""
    require_torch()
    dev = torch.device(device)
    mask = torch.from_numpy(a_mask).to(dev)
    num = mask.shape[0]
    f64 = dict(dtype=torch.float64, device=dev)

    t0 = time.time()
    lo = {k: torch.full((num, ), np.inf, **f64) for k in HIER_KEYS}
    hi = {k: torch.full((num, ), -np.inf, **f64) for k in HIER_KEYS}
    lo.update({k: np.inf for k in FLAT_KEYS})
    hi.update({k: -np.inf for k in FLAT_KEYS})
    for i, (z, _) in enumerate(_prefetch(files)):
        if z.shape[0]:
            q = torch_base_quantities(torch.from_numpy(z).to(dev))
            for key, s in torch_flat_scores(q).items():
                lo[key] = min(lo[key], float(s.min()))
                hi[key] = max(hi[key], float(s.max()))
            for j0 in range(0, num, chunk):
                block = slice(j0, j0 + chunk)
                for key, s in torch_bipartition_scores(q, mask[block]).items():
                    lo[key][block] = torch.minimum(lo[key][block],
                                                   s.amin(dim=1).double())
                    hi[key][block] = torch.maximum(hi[key][block],
                                                   s.amax(dim=1).double())
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f'  scan: {i + 1}/{len(files)} frames '
                  f'({time.time() - t0:.0f} s)', flush=True)
    for key in FLAT_KEYS:
        lo[key] = torch.tensor(lo[key], **f64)
        hi[key] = torch.tensor(hi[key], **f64)

    hist = {k: torch.zeros((num, 2, bins), dtype=torch.int64, device=dev)
            for k in HIER_KEYS}
    flat_hist = {k: torch.zeros((2, bins), dtype=torch.int64, device=dev)
                 for k in FLAT_KEYS}
    counts = np.zeros(2, dtype=np.int64)
    for i, (z, ood) in enumerate(_prefetch(files)):
        if z.shape[0]:
            order = np.argsort(ood, kind='stable')  # ID first, OOD last
            n_id = int(ood.size - ood.sum())
            counts += [n_id, ood.size - n_id]
            q = torch_base_quantities(torch.from_numpy(z[order]).to(dev))
            for key, s in torch_flat_scores(q).items():
                b = torch_bin_index(s, lo[key], hi[key], bins, eps)
                flat_hist[key][0] += torch_counts(b[:n_id], bins)
                flat_hist[key][1] += torch_counts(b[n_id:], bins)
            for j0 in range(0, num, chunk):
                block = slice(j0, j0 + chunk)
                c = mask[block].shape[0]
                offsets = (torch.arange(c, device=dev) * bins)[:, None]
                for key, s in torch_bipartition_scores(q, mask[block]).items():
                    b = torch_bin_index(s, lo[key][block], hi[key][block],
                                        bins, eps)
                    b += offsets  # one count for the whole block
                    h = hist[key]
                    h[block, 0] += torch_counts(
                        b[:, :n_id].reshape(-1), c * bins).view(c, bins)
                    h[block, 1] += torch_counts(
                        b[:, n_id:].reshape(-1), c * bins).view(c, bins)
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f'  score: {i + 1}/{len(files)} frames '
                  f'({time.time() - t0:.0f} s)', flush=True)
    return ({k: v.cpu().numpy() for k, v in hist.items()},
            {k: v.cpu().numpy() for k, v in flat_hist.items()}, counts)


# ------------------------------------------------------------------ workers
def _scan_shard(args):
    """Empirical (min, max) of every flat score and of every hierarchy
    score per bipartition -- the histogram ranges, as online."""
    files, a_mask, chunk, shard = args
    num = a_mask.shape[0]
    lo = {k: np.full(num, np.inf) for k in HIER_KEYS}
    hi = {k: np.full(num, -np.inf) for k in HIER_KEYS}
    lo.update({k: np.inf for k in FLAT_KEYS})
    hi.update({k: -np.inf for k in FLAT_KEYS})
    for i, path in enumerate(files):
        z, _ = load_frame(path)
        if z.shape[0] == 0:
            continue
        q = base_quantities(z)
        for key, s in flat_scores(q).items():
            lo[key] = min(lo[key], float(s.min()))
            hi[key] = max(hi[key], float(s.max()))
        for j0 in range(0, num, chunk):
            block = slice(j0, j0 + chunk)
            for key, s in bipartition_scores(q, a_mask[block]).items():
                np.minimum(lo[key][block], s.min(axis=1), out=lo[key][block])
                np.maximum(hi[key][block], s.max(axis=1), out=hi[key][block])
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f'  scan shard {shard}: {i + 1}/{len(files)} frames',
                  flush=True)
    return lo, hi


def _score_shard(args):
    files, a_mask, ranges, bins, eps, chunk, shard = args
    num = a_mask.shape[0]
    hist = {k: np.zeros((num, 2, bins), dtype=np.int32) for k in HIER_KEYS}
    flat_hist = {k: np.zeros((2, bins), dtype=np.int64) for k in FLAT_KEYS}
    counts = np.zeros(2, dtype=np.int64)
    for i, path in enumerate(files):
        z, ood = load_frame(path)
        if z.shape[0] == 0:
            continue
        # ID points first, OOD points last: each label is then a
        # contiguous slice of every score row.
        order = np.argsort(ood, kind='stable')
        z, ood = z[order], ood[order]
        n_id = int(ood.size - ood.sum())
        parts = (slice(0, n_id), slice(n_id, None))  # ID, OOD
        counts += [n_id, ood.size - n_id]
        q = base_quantities(z)
        for key, s in flat_scores(q).items():
            b = bin_index(s, *ranges[key], bins, eps)
            for lab in (0, 1):
                flat_hist[key][lab] += np.bincount(b[parts[lab]],
                                                   minlength=bins)
        for j0 in range(0, num, chunk):
            block = a_mask[j0:j0 + chunk]
            for key, s in bipartition_scores(q, block).items():
                lo, hi = ranges[key]
                b = bin_index(s, lo[j0:j0 + chunk], hi[j0:j0 + chunk], bins,
                              eps)
                h = hist[key]
                for j in range(block.shape[0]):
                    for lab in (0, 1):
                        h[j0 + j, lab] += np.bincount(
                            b[j, parts[lab]], minlength=bins).astype(np.int32)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f'  shard {shard}: {i + 1}/{len(files)} frames', flush=True)
    return hist, flat_hist, counts


# --------------------------------------------------------------------- main
def _numpy_passes(files, a_mask, bins, eps, chunk, workers):
    """Both passes on CPU worker processes, each over a shard of frames."""
    workers = max(1, min(workers, len(files)))
    shards = [files[i::workers] for i in range(workers)]
    t0 = time.time()
    with Pool(workers) as pool:
        scans = pool.map(_scan_shard, [(shard, a_mask, chunk, i)
                                       for i, shard in enumerate(shards)])
    ranges = {}
    for key in FLAT_KEYS:
        ranges[key] = (min(lo[key] for lo, _ in scans),
                       max(hi[key] for _, hi in scans))
    for key in HIER_KEYS:
        ranges[key] = (np.minimum.reduce([lo[key] for lo, _ in scans]),
                       np.maximum.reduce([hi[key] for _, hi in scans]))
    print(f'score ranges scanned in {time.time() - t0:.0f} s', flush=True)

    hist = {k: np.zeros((a_mask.shape[0], 2, bins), dtype=np.int64)
            for k in HIER_KEYS}
    flat_hist = {k: np.zeros((2, bins), dtype=np.int64) for k in FLAT_KEYS}
    counts = np.zeros(2, dtype=np.int64)
    jobs = [(shard, a_mask, ranges, bins, eps, chunk, i)
            for i, shard in enumerate(shards)]
    with Pool(workers) as pool:
        for part_hist, part_flat, part_counts in pool.imap_unordered(
                _score_shard, jobs):
            for k in HIER_KEYS:
                hist[k] += part_hist[k]
            for k in FLAT_KEYS:
                flat_hist[k] += part_flat[k]
            counts += part_counts
    return hist, flat_hist, counts


def sweep(dump_dirs, out_dir, num=500, seed=0, bins=2**16, workers=8,
          chunk=64, top=10, rank_exclude=('odin', ), backend='numpy',
          device='cuda:0', bin_eps=BIN_EPS):
    """Score ``num`` bipartitions over the frames of one or several dump
    directories (evaluated together as one split)."""
    if isinstance(dump_dirs, str):
        dump_dirs = [dump_dirs]
    files = []
    for dump_dir in dump_dirs:
        found = sorted(glob.glob(osp.join(dump_dir, '*.npz')))
        if not found:
            raise FileNotFoundError(f'no .npz dumps in {dump_dir}')
        files += found
    dump_dir = ' + '.join(dump_dirs)
    subsets = sample_bipartitions(num, seed)
    names = [partition_name(s) for s in subsets]
    a_mask = subsets_to_mask(subsets)
    sizes = np.bincount(a_mask.sum(axis=1), minlength=NUM_CLASSES // 2 + 1)
    print(f'{len(files)} frames in {dump_dir}; {len(subsets)} bipartitions '
          f'(seed {seed}), smaller-group sizes 1..{NUM_CLASSES // 2}: '
          + ' '.join(str(n) for n in sizes[1:]))

    t0 = time.time()
    if backend == 'torch':
        hist, flat_hist, counts = _torch_passes(files, a_mask, bins, bin_eps,
                                                chunk, device)
    elif backend == 'numpy':
        hist, flat_hist, counts = _numpy_passes(files, a_mask, bins, bin_eps,
                                                chunk, workers)
    else:
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend}")
    n_id, n_ood = (int(c) for c in counts)
    print(f'scored in {time.time() - t0:.0f} s: {n_id} ID / {n_ood} OOD '
          'points', flush=True)

    rows = OrderedDict()
    for key in FLAT_KEYS:
        rows[key] = metrics_from_histograms(flat_hist[key][1],
                                            flat_hist[key][0])
    for i, name in enumerate(names):
        for key in HIER_KEYS:
            rows[f'{name}_{key}'] = metrics_from_histograms(
                hist[key][i, 1], hist[key][i, 0])

    os.makedirs(out_dir, exist_ok=True)
    log_path = osp.join(out_dir, 'bipartitions.log')
    write_log(log_path, rows, n_id, n_ood, out_dir, dump_dir, num, seed,
              bins, bin_eps)
    write_partitions(out_dir, names, subsets)
    print(f'wrote {log_path}')
    print_ranking(log_path, names, subsets, top, rank_exclude)
    return rows


def write_log(path, rows, n_id, n_ood, out_dir, dump_dir, num, seed, bins,
              bin_eps):
    with open(path, 'w') as fh:
        fh.write(f"work_dir = '{out_dir}'\n")
        fh.write(f'# offline bipartition sweep: {num} two-group partitions '
                 f'(seed {seed}) of the {NUM_CLASSES} classes, scored from '
                 f'{dump_dir} with {bins} histogram bins '
                 + (f'log-spaced towards both ends (eps {bin_eps:g})'
                    if bin_eps > 0 else 'of equal width') + '\n')
        fh.write(f'Point-level OOD evaluation: {n_id} ID points, {n_ood} '
                 f'OOD points ({n_ood / max(n_id + n_ood, 1):.4%} OOD)\n')
        header = f'{"method":>10} | {"AUROC":>8} | {"AP":>8} | {"FPR@95":>8}'
        fh.write(header + '\n' + '-' * len(header) + '\n')
        for method, m in rows.items():
            fh.write(f'{method:>10} | {100 * m["auroc"]:8.2f} | '
                     f'{100 * m["ap"]:8.2f} | {100 * m["fpr95"]:8.2f}\n')


def describe(subset):
    other = [c for c in range(NUM_CLASSES) if c not in subset]
    return (', '.join(CLASSES[c] for c in subset),
            ', '.join(CLASSES[c] for c in other))


def write_partitions(out_dir, names, subsets):
    with open(osp.join(out_dir, 'partitions.tsv'), 'w') as fh:
        fh.write('name\tsize_A\tgroup_A\tgroup_B\n')
        for name, subset in zip(names, subsets):
            group_a, group_b = describe(subset)
            fh.write(f'{name}\t{len(subset)}\t{group_a}\t{group_b}\n')
    with open(osp.join(out_dir, 'partitions.json'), 'w') as fh:
        json.dump([
            dict(name=name, A=list(subset),
                 B=[c for c in range(NUM_CLASSES) if c not in subset])
            for name, subset in zip(names, subsets)
        ], fh, indent=1)


def print_ranking(log_path, names, subsets, top, exclude):
    """Console preview of the summariser's ranking, with class names."""
    composition = {name: describe(subset)[0]
                   for name, subset in zip(names, subsets)}
    rows = parse_logs([log_path])
    exclude = set(exclude) | {'maxlogit'}
    flat, summary = summarise(rows, exclude=exclude)
    print('flat reference (AUROC/AP/FPR@95): ' + '  '.join(
        f'{k} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}' for k, v in flat.items()))
    reference = partition_name(canonical(REFERENCE))
    for key in ('group_msp', 'gn_msp'):
        m = rows[f'{reference}_{key}']
        print(f'reference {reference} (vehicle vs rest) {key}: '
              f'{m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}')
    # Group family only: the offline GN MSP / GN Entropy rows are approximate
    # (see the module docstring), so no GN ranking is published here and
    # summarize_hierarchy_ablation.py refuses --family gn on this log.
    for fam in ('group', ):
        order = sorted((h for h in summary if fam in summary[h]),
                       key=lambda h: -_improvement(summary[h][fam]['mean']))
        n_pos = sum(_improvement(summary[h][fam]['mean']) > 0 for h in order)
        print(f'\ntop {top} of {len(order)} bipartitions, family {fam} '
              f'(improvement = mean dAUROC + mean dAP - mean dFPR@95 over '
              f'the baselines except {", ".join(sorted(exclude))}; '
              f'{n_pos} above 0):')
        print(f'| rank | name | improvement | {fam}_msp AUROC/AP/FPR@95 | '
              'smaller group |')
        print('| --- | --- | --- | --- | --- |')
        for rank, name in enumerate(order[:top], start=1):
            m = rows[f'{name}_{fam}_msp']
            print(f'| {rank} | {name} | '
                  f'{_improvement(summary[name][fam]["mean"]):+.2f} | '
                  f'{m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f} | {composition[name]} |')
    print('\n(no GN ranking: the offline GN MSP / GN Entropy rows are '
          'approximate; rank the GN family from test.py logs)')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump_dirs', nargs='+',
                    help='one or several directories of _OODLogitsDumpMetric '
                    'npzs, evaluated together as one split')
    ap.add_argument('--backend', choices=['numpy', 'torch'], default='numpy',
                    help='numpy: CPU worker processes; torch: one process '
                    'on --device')
    ap.add_argument('--device', default='cuda:0',
                    help='torch device of --backend torch')
    ap.add_argument('--num', type=int, default=500,
                    help='number of bipartitions (incl. vehicle vs rest)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bins', type=int, default=2**16,
                    help='histogram bins per score (online metric: 2^20)')
    ap.add_argument('--bin-eps', type=float, default=BIN_EPS,
                    help='bins are log-spaced towards both ends of every '
                    'score range, down to eps of the range; 0 = equal-width '
                    'bins exactly as the online metric, which at 2^16 bins '
                    'cannot resolve very confident OOD points (see '
                    'bin_index)')
    ap.add_argument('--workers', type=int,
                    default=min(8, os.cpu_count() or 1),
                    help='worker processes (each holds ~4 bytes x 16 x '
                    '--num x --bins of histograms)')
    ap.add_argument('--chunk', type=int, default=64,
                    help='bipartitions scored per matmul block')
    ap.add_argument('--out-dir', default=None,
                    help='default: <dump_dir>/../bipartitions (required '
                    'with several dump directories)')
    ap.add_argument('--top', type=int, default=10,
                    help='rows of the console ranking per family')
    ap.add_argument('--rank-exclude', default='odin',
                    help='comma-separated baselines left out of the console '
                    'ranking (maxlogit always is)')
    args = ap.parse_args()
    if args.out_dir is None and len(args.dump_dirs) > 1:
        ap.error('--out-dir is required with several dump directories')
    out_dir = args.out_dir or osp.join(
        osp.dirname(osp.normpath(args.dump_dirs[0])), 'bipartitions')
    sweep(args.dump_dirs, out_dir, num=args.num, seed=args.seed,
          bins=args.bins, workers=args.workers, chunk=args.chunk,
          top=args.top,
          rank_exclude=[b for b in args.rank_exclude.split(',') if b],
          backend=args.backend, device=args.device, bin_eps=args.bin_eps)


if __name__ == '__main__':
    main()
