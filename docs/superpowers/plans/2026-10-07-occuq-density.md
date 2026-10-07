# OCCUQ Density Scores vs Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port OCCUQ's feature-density OOD score to P3Former on DSO in three variants (`pe`, A, C), and decide with a like-for-like `improvement` score whether it beats the 33 robust grouping splits.

**Architecture:**
- **Head:** a new `_OCCUQHead` (OCCUQ's spectrally normalized residual MLP) sits next to `sem_queries` in `_P3FormerHead`, fed by `pe_features`.
- **Variants:** variant A trains it on the frozen model. Variant C fine-tunes everything with it attached. `pe` uses `pe_features` directly, with no head.
- **Fitting:** a tool fits one Gaussian per class on the training split from running float64 sums.
- **Scoring:** `predict` scores asinh(−log q) per point next to the flat OOD scores.
- **Comparison:** a second tool compares the `test.py` logs with grouping's logged robust splits.

**Tech Stack:** Python 3.8, torch 1.10.1+cu111, mmengine 0.7.4, mmdet3d 1.1.0, spconv, the repo's plain-function test files (pytest-compatible).

**Spec:** `docs/superpowers/specs/2026-10-07-occuq-density-design.md`

## Global Constraints

- **Interpreter.** Always `/home/khoadv/miniconda3/envs/p3former/bin/python`, run from the repo root `/home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD`. Bare `python` is a different env. `dist_train.sh` calls bare `python`, so prefix it with `PATH=/home/khoadv/miniconda3/envs/p3former/bin:$PATH`.
- **Branch.** All work is on `ood-baselines/occuq`; nothing goes to `flat` or `grouping`.
- **Torch 1.10 APIs.**
  - There is no `torch.linalg.solve_triangular`; use `torch.triangular_solve`.
  - `torch.linalg.cholesky_ex` exists.
  - CUDA matmuls default to TF32. Disable it **only** around the density computation, with `no_tf32()`.
  - CUDA `torch.bincount` is slow on skewed labels; count on the CPU.
- **Protocol:**
  - OOD is raw 17 (Stop) and 28 (Others).
  - ID means the mapped label ≠ 24.
  - `_OODPointMetric` evaluates only the keys in `score_keys`, whose default is the five flat scores. New keys must be listed.
- **Sets:** `cetran` (`dso_infos_cetran.pkl`, 980 frames), `test` (`dso_infos_test.pkl`, 2,625) and `test_cetran` (`dso_infos_test_cetran.pkl`, 3,605). Training split: `dso_infos_train.pkl`, 8,474 frames.
- **Base checkpoint:** `work_dirs/p3former_2xb1_3x_dso/epoch_36.pth`.
- **Fitting constants:**
  - 24 classes; ignore index 24.
  - Minimum 2,560 voxels per class.
  - `--check` tolerance 0.05 nats.
  - `score_chunk` 65,536.
  - Regularization: every class covariance gets the same diagonal, a ridge of 1e-6 × the mean pooled ID variance plus one jitter shared by all classes: the first of `[0, 2.2e-308] + [10**e for e in range(-308, 0)]` for which every class factors (amended after the final review; see the end of this plan).
- **Training recipe for A and C:**
  - `load_from` the base checkpoint.
  - AdamW, lr 2e-4, weight decay 0.01, no backbone multiplier, no clipping.
  - 9 epochs: `LinearLR` warmup for 500 iterations from ⅓, then `CosineAnnealingLR` to `eta_min=2e-7` with `convert_to_iter_based=True`.
  - `val_interval=9`, and `CheckpointHook(interval=9, max_keep_ckpts=1)`.
  - 2 GPUs × batch 1, seeds 0, 1 and 2.
- **Storage.** `/` is 99% full. `work_dirs/p3former_2xb1_3x_dso_occuq_{pe,a,c,compare}` are symlinks into `/mnt/sandisk/khoadv/occuq/`.
- **Machine.**
  - The GPUs are shared: run `nvidia-smi` first, and run one evaluation per GPU (~25 GB).
  - Start long jobs detached: `nohup setsid bash -c "<cmd> > <log> 2>&1" < /dev/null > /dev/null 2>&1 &`.
- **Tests.**
  - A module docstring with the run command.
  - `sys.path.insert(0, <repo root>)`.
  - Plain `test_*` functions that end with `print('PASS <name>')`.
  - A `__main__` block that calls every test and prints `ALL TESTS PASSED`.
- **Commits.** Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
  ```

## Review Focus

1. **The preprocessor's flag under freezing.** Variant A must leave the data preprocessor in training mode, because it builds `voxel_semantic_mask` only then; otherwise the head's loss crashes. Pinned by `test_variant_a_freezes_the_real_model` (Task 5).
2. **Validation before any Gaussians exist.** The end-of-training validation of A and C must not ask for `density`, since no Gaussian file exists yet. Pinned by `test_training_configs` (Task 6): the validation evaluator is PQ-only.
3. **A Gaussian file used with other weights** (a seed mix-up, or the `pe` file with a C checkpoint) must be refused, not silently scored. Pinned by `test_fingerprint_and_load_gmm_guards` (Task 3) and `test_predict_emits_density_scores_and_checks_the_fingerprint` (Task 4).
4. **Sweep rows that disagree.** A robust split listed in both the bipartition and the singleton logs with different rows, or a mismatch with `robust.tsv`, must stop the comparison. Pinned by `test_merge_rows_rejects_conflicts` and `test_compare_rejects_a_robust_tsv_mismatch` (Task 8).
5. **Point labels and voxel maps that don't line up.** A voxel without points, a length mismatch, or a label outside 0–24 must raise, not mislabel voxels. Pinned by `test_majority_vote_rejects_empty_voxels_and_length_mismatch` and `test_out_of_range_label_is_rejected` (Task 2).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `p3former/decode_heads/occuq_head.py` (new) | `_OCCUQHead`: the head, a plain `nn.Module` |
| `p3former/utils/gmm_fit.py` (new) | pure fitting math: voxel majority labels, running float64 sums, jitter search, `finalize` |
| `p3former/utils/gmm_density.py` (new) | pure scoring: `no_tf32`, `fingerprint`, `load_gmm`, `log_density`, `density_score`, `point_density_scores` |
| `p3former/utils/freeze.py` (new) | `freeze_all_but`, `eval_all_but` |
| `p3former/decode_heads/p3former_head.py` (modify) | `occuq_cfg`; OCCUQ forward, losses and density scores |
| `p3former/segmentors/p3former.py` (modify) | variant A freezing: `requires_grad`, `train()`, no-grad features |
| `configs/p3former/p3former_2xb1_3x_dso_occuq_{pe,a,c}.py` (new) | the three variants |
| `tools/fit_occuq_gmm.py` (new) | runs the model over the training split and writes the Gaussian file |
| `tools/compare_occuq_grouping.py` (new) | reads logs only; writes the verdict report |
| `tests/test_occuq_head.py`, `tests/test_fit_occuq_gmm.py`, `tests/test_gmm_density.py`, `tests/test_p3former_head_occuq.py`, `tests/test_occuq_freeze.py`, `tests/test_occuq_configs.py`, `tests/test_compare_occuq_grouping.py` (new) | one test file per unit |
| `DOCs.md`, `.claude/CLAUDE.md`, `.claude/rules/{architecture/ood-pipeline,configs,commands,testing}.md` (modify) | results and instructions (Task 12) |

The spec named four test files. The head integration, the freezing and the configs get their own files here, so that each task writes its file once.

---

### Task 1: The OCCUQ head module

**Files:**
- Create: `p3former/decode_heads/occuq_head.py`
- Test: `tests/test_occuq_head.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_OCCUQHead(in_channels: int = 256, num_classes: int = 25, num_blocks: int = 4)`, whose `forward(x: Tensor[V, C]) -> (logits: Tensor[V, num_classes], feature: Tensor[V, C])`. It has the attributes `input_proj`, `blocks` (a ModuleList) and `classifier`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_occuq_head.py`:

```python
"""Tests for the OCCUQ head (p3former/decode_heads/occuq_head.py).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_occuq_head.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.decode_heads.occuq_head import _OCCUQHead  # noqa: E402


def test_shapes():
    head = _OCCUQHead(in_channels=32, num_classes=5)
    logits, feature = head(torch.randn(7, 32))
    assert logits.shape == (7, 5)
    assert feature.shape == (7, 32)
    print('PASS test_shapes')


def test_layout_mirrors_occuq():
    head = _OCCUQHead(in_channels=32, num_classes=5)
    keys = set(head.state_dict())
    # OCCUQ's widened 1x1 conv: no bias, no spectral normalisation
    assert {k for k in keys if k.startswith('input_proj.')} == {
        'input_proj.weight'}
    assert len(head.blocks) == 4
    for prefix in [f'blocks.{i}.' for i in range(4)] + ['classifier.']:
        for name in ('weight_orig', 'weight_u', 'weight_v', 'bias'):
            assert prefix + name in keys, prefix + name
    print('PASS test_layout_mirrors_occuq')


def test_spectral_norm_is_one_after_power_iterations():
    torch.manual_seed(0)
    head = _OCCUQHead(in_channels=32, num_classes=5).train()
    x = torch.randn(64, 32)
    for _ in range(100):  # one power iteration per training-mode forward
        head(x)
    for layer in list(head.blocks) + [head.classifier]:
        sigma = torch.linalg.svdvals(layer.weight.detach())[0].item()
        assert abs(sigma - 1.0) < 1e-2, sigma
    print('PASS test_spectral_norm_is_one_after_power_iterations')


def test_feature_is_last_block_output():
    torch.manual_seed(0)
    head = _OCCUQHead(in_channels=16, num_classes=3).eval()
    x = torch.randn(5, 16)
    with torch.no_grad():
        logits, feature = head(x)
        h = head.input_proj(x)
        for block in head.blocks:
            h = h + torch.relu(block(h))
        assert torch.allclose(feature, h, atol=1e-6)
        assert torch.allclose(logits, head.classifier(h), atol=1e-6)
    print('PASS test_feature_is_last_block_output')


if __name__ == '__main__':
    test_shapes()
    test_layout_mirrors_occuq()
    test_spectral_norm_is_one_after_power_iterations()
    test_feature_is_last_block_output()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_occuq_head.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'p3former.decode_heads.occuq_head'`.

- [ ] **Step 3: Write the implementation**

Create `p3former/decode_heads/occuq_head.py`:

```python
"""OCCUQ uncertainty head (Heidrich, Beemelmanns et al., ICRA 2025).

Mirrors ``MLPHeadv5`` of the official code (``occ_head.py:199-233`` at
commit 2aa5429):
- a bias-free linear map without spectral normalisation (SurroundOcc's
  widened 1x1 conv);
- four residual blocks ``x + ReLU(SN-Linear(x))``;
- a spectrally normalised linear classifier.

Spectral normalisation is ``torch.nn.utils.spectral_norm`` with its
defaults (sigma = 1, one power iteration). It stays active at inference,
using the stored u and v. The Gaussians of the density score are fitted on
the output of the last block (spec
docs/superpowers/specs/2026-10-07-occuq-density-design.md).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class _OCCUQHead(nn.Module):
    """Per-voxel OCCUQ head on ``pe_features``.

    Args:
        in_channels: feature width (P3Former's ``embed_dims``, 256).
        num_classes: classifier outputs (25 for DSO, ignore channel
            included).
        num_blocks: residual spectrally normalised blocks (4, as OCCUQ).
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 25,
                 num_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.blocks = nn.ModuleList(
            spectral_norm(nn.Linear(in_channels, in_channels))
            for _ in range(num_blocks))
        self.classifier = spectral_norm(nn.Linear(in_channels, num_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[V, C] features -> (logits [V, num_classes], feature [V, C])."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + F.relu(block(x))
        return self.classifier(x), x
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_occuq_head.py`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add p3former/decode_heads/occuq_head.py tests/test_occuq_head.py
git commit -q -F - <<'EOF'
Add the OCCUQ head (spectrally normalised residual MLP)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 2: Fitting the Gaussians from running sums

**Files:**
- Create: `p3former/utils/gmm_fit.py`
- Test: `tests/test_fit_occuq_gmm.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `JITTERS: List[float]`.
  - `voxel_majority_labels(point_labels: Tensor[N], point2voxel_map: Tensor[N], num_voxels: int, num_labels: int) -> Tensor[V] int64`.
  - `GaussianStats(num_classes: int, dim: int, device='cpu')`:
    - attributes `count` [C], `sum` [C, D] and `outer` [C, D, D], all float64;
    - `update(features: Tensor[V, D], labels: Tensor[V], ignore_index: int) -> None`.
  - `cholesky_with_jitter(cov: Tensor[D, D] float64) -> (Tensor[D, D], float)`.
  - `finalize(stats: GaussianStats, min_count: int) -> dict`, with keys `means` [C, D], `covs` [C, D, D], `prec_chol` [C, D, D], `log_det_prec` [C], `log_prior` [C], `counts` [C] and `jitter` [C], all float64.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fit_occuq_gmm.py`:

```python
"""Tests for p3former/utils/gmm_fit.py (OCCUQ Gaussians from running sums).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_fit_occuq_gmm.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.utils.gmm_fit import (JITTERS, GaussianStats,  # noqa: E402
                                    cholesky_with_jitter, finalize,
                                    voxel_majority_labels)


def _data(n=600, dim=4, seed=0):
    """Features of 3 classes plus ignore (label 3)."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n, ), generator=g)
    feats = torch.randn(n, dim, generator=g) * 2 + labels[:, None].float()
    return feats, labels


def test_running_sums_match_torch_cov():
    feats, labels = _data()
    stats = GaussianStats(num_classes=3, dim=4)
    for chunk in torch.split(torch.randperm(len(labels)), 97):
        stats.update(feats[chunk], labels[chunk], ignore_index=3)
    gmm = finalize(stats, min_count=10)
    eye = torch.eye(4, dtype=torch.float64)
    for c in range(3):
        zc = feats[labels == c].double()
        assert torch.allclose(gmm['means'][c], zc.mean(0), atol=1e-10)
        assert torch.allclose(gmm['covs'][c], torch.cov(zc.t()), atol=1e-10)
        assert int(gmm['counts'][c]) == len(zc)
        prec = gmm['prec_chol'][c]  # P = L^-T whitens the class
        assert torch.allclose(prec.t() @ gmm['covs'][c] @ prec, eye,
                              atol=1e-8)
        assert torch.isclose(gmm['log_det_prec'][c],
                             -0.5 * torch.logdet(gmm['covs'][c]), atol=1e-10)
    counts = torch.tensor([(labels == c).sum() for c in range(3)],
                          dtype=torch.float64)
    assert torch.allclose(gmm['log_prior'], torch.log(counts / counts.sum()))
    assert (gmm['jitter'] == 0).all()
    print('PASS test_running_sums_match_torch_cov')


def test_ignore_label_never_reaches_the_sums():
    feats, labels = _data()
    clean = GaussianStats(3, 4)
    clean.update(feats[labels != 3], labels[labels != 3], ignore_index=3)
    poisoned = feats.clone()
    poisoned[labels == 3] = 1e6  # ignored voxels, e.g. Stop / Others
    mixed = GaussianStats(3, 4)
    mixed.update(poisoned, labels, ignore_index=3)
    for name in ('count', 'sum', 'outer'):
        assert torch.equal(getattr(clean, name), getattr(mixed, name)), name
    print('PASS test_ignore_label_never_reaches_the_sums')


def test_out_of_range_label_is_rejected():
    stats = GaussianStats(3, 4)
    try:
        stats.update(torch.zeros(2, 4), torch.tensor([0, 7]), ignore_index=3)
    except ValueError as err:
        assert 'labels must be in' in str(err)
    else:
        raise AssertionError('label 7 was accepted')
    print('PASS test_out_of_range_label_is_rejected')


def test_majority_vote():
    p2v = torch.tensor([0, 0, 0, 1, 1, 2])
    labels = torch.tensor([5, 5, 2, 3, 1, 4])
    voxel = voxel_majority_labels(labels, p2v, num_voxels=3, num_labels=25)
    # voxel 0: 5 wins 2-1; voxel 1: 3 and 1 tie -> lowest (1); voxel 2: 4
    assert voxel.tolist() == [5, 1, 4]
    print('PASS test_majority_vote')


def test_majority_vote_rejects_empty_voxels_and_length_mismatch():
    labels = torch.tensor([1, 2])
    for p2v, num_voxels in ((torch.tensor([0, 2]), 3),  # voxel 1 is empty
                            (torch.tensor([0]), 1)):  # 2 labels, 1 entry
        try:
            voxel_majority_labels(labels, p2v, num_voxels, 25)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted p2v={p2v.tolist()}, '
                                 f'V={num_voxels}')
    print('PASS test_majority_vote_rejects_empty_voxels_and_length_mismatch')


def test_jitter_zero_for_positive_definite():
    a = torch.randn(5, 5, dtype=torch.float64)
    cov = a @ a.t() + 0.1 * torch.eye(5, dtype=torch.float64)
    _, jitter = cholesky_with_jitter(cov)
    assert jitter == 0.0
    print('PASS test_jitter_zero_for_positive_definite')


def test_jitter_is_smallest_working_value():
    cov = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, -1e-3],
                                  dtype=torch.float64))
    chol, jitter = cholesky_with_jitter(cov)
    # 1e-3 only lifts the last eigenvalue to 0, which is not positive
    assert jitter == 1e-2, jitter
    assert JITTERS.index(jitter) == JITTERS.index(1e-3) + 1
    eye = torch.eye(5, dtype=torch.float64)
    assert torch.allclose(chol @ chol.t(), cov + jitter * eye, atol=1e-12)
    print('PASS test_jitter_is_smallest_working_value')


def test_min_count_error_names_the_class():
    feats, labels = _data(n=60)
    stats = GaussianStats(3, 4)
    stats.update(feats, labels, ignore_index=3)
    try:
        finalize(stats, min_count=1000)
    except ValueError as err:
        assert 'fewer than 1000' in str(err) and '(0, ' in str(err)
    else:
        raise AssertionError('no error for too few voxels')
    print('PASS test_min_count_error_names_the_class')


if __name__ == '__main__':
    test_running_sums_match_torch_cov()
    test_ignore_label_never_reaches_the_sums()
    test_out_of_range_label_is_rejected()
    test_majority_vote()
    test_majority_vote_rejects_empty_voxels_and_length_mismatch()
    test_jitter_zero_for_positive_definite()
    test_jitter_is_smallest_working_value()
    test_min_count_error_names_the_class()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_fit_occuq_gmm.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'p3former.utils.gmm_fit'`.

- [ ] **Step 3: Write the implementation**

Create `p3former/utils/gmm_fit.py`:

```python
"""Class-conditional Gaussians for the OCCUQ density score, from streamed
voxel features.

OCCUQ (official code ``tools/gmm_utils.py:163-220`` at commit 2aa5429)
stores up to 200M float32 feature vectors per class and calls
``torch.cov``. Here each class keeps running float64 sums instead: the
count, sum z and sum z z^T. So every training voxel is used, with 12.6 MB
of state for 24 classes x 256 dimensions.

``finalize`` turns the sums into, per class:
- the mean, and the N-1 covariance (``torch.cov``'s convention);
- the precision Cholesky factor P = L^-T, found with OCCUQ's jitter search
  (per class here);
- its log-determinant and the log class prior.

Pure functions; ``tools/fit_occuq_gmm.py`` runs the model and feeds them.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# OCCUQ's jitter list (tools/gmm_utils.py:13-14), tried in this order.
JITTERS = [0.0, torch.finfo(torch.float64).tiny] + [
    10.0**e for e in range(-308, 0)
]


def voxel_majority_labels(point_labels: torch.Tensor,
                          point2voxel_map: torch.Tensor, num_voxels: int,
                          num_labels: int) -> torch.Tensor:
    """Label of each voxel, by majority vote of its points' labels.

    Args:
        point_labels: [N] mapped labels in [0, num_labels).
        point2voxel_map: [N] voxel index of each point (any int/float dtype).
        num_voxels: number of voxels V.
        num_labels: number of label values (25 for DSO: 24 classes +
            ignore).

    Returns:
        [V] int64. Ties go to the lowest label (``argmax`` returns the first
        maximum).
    """
    if point_labels.shape != point2voxel_map.shape:
        raise ValueError(f'{point_labels.shape[0]} point labels but '
                         f'{point2voxel_map.shape[0]} point-to-voxel '
                         'entries')
    votes = torch.zeros(num_voxels, num_labels, device=point_labels.device)
    votes.index_add_(0, point2voxel_map.long(),
                     F.one_hot(point_labels.long(), num_labels).float())
    empty = (votes.sum(dim=1) == 0).nonzero().flatten()
    if len(empty):
        raise ValueError(f'{len(empty)} voxels have no points (first: '
                         f'{empty[:5].tolist()}): point2voxel_map does not '
                         'match the voxel features')
    return votes.argmax(dim=1)


class GaussianStats:
    """Running per-class count, sum z and sum z z^T, in float64.

    Args:
        num_classes: classes with a Gaussian (labels 0..num_classes-1).
        dim: feature dimension D.
        device: where the sums live (the features' device).
    """

    def __init__(self, num_classes: int, dim: int, device='cpu') -> None:
        kw = dict(dtype=torch.float64, device=device)
        self.count = torch.zeros(num_classes, **kw)
        self.sum = torch.zeros(num_classes, dim, **kw)
        self.outer = torch.zeros(num_classes, dim, dim, **kw)

    def update(self, features: torch.Tensor, labels: torch.Tensor,
               ignore_index: int) -> None:
        """Add voxels (features [V, D], labels [V]); skip ``ignore_index``."""
        if features.shape[1] != self.sum.shape[1]:
            raise ValueError(f'features have {features.shape[1]} dimensions, '
                             f'the sums {self.sum.shape[1]}')
        keep = labels != ignore_index
        z = features[keep].double()
        y = labels[keep].long()
        num_classes = self.count.shape[0]
        if len(y) and (int(y.min()) < 0 or int(y.max()) >= num_classes):
            raise ValueError(f'labels must be in [0, {num_classes}) or '
                             f'{ignore_index}, got {int(y.min())}..'
                             f'{int(y.max())}')
        for c in torch.unique(y).tolist():
            zc = z[y == c]
            self.count[c] += zc.shape[0]
            self.sum[c] += zc.sum(dim=0)
            self.outer[c] += zc.t() @ zc


def cholesky_with_jitter(cov: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Lower Cholesky factor of ``cov + jitter * I`` (float64), with the
    first jitter of ``JITTERS`` that makes it succeed."""
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    for jitter in JITTERS:
        chol, info = torch.linalg.cholesky_ex(cov + jitter * eye)
        if int(info) == 0 and bool(torch.isfinite(chol).all()):
            return chol, jitter
    raise ValueError('the covariance is not positive definite even with '
                     f'jitter {JITTERS[-1]}')


def finalize(stats: GaussianStats, min_count: int) -> Dict[str, torch.Tensor]:
    """Turn the running sums into Gaussians, all float64.

    Returns a dict with:
    - means [C, D];
    - covs [C, D, D], with the N-1 correction and no jitter;
    - prec_chol [C, D, D]: P = L^-T of cov + jitter, so (z - mu) P has
      identity covariance;
    - log_det_prec [C]: sum log diag P, which is -0.5 log det Sigma;
    - log_prior [C]: log(n_c / sum n);
    - counts [C] and jitter [C].
    """
    if min_count < 2:
        raise ValueError('min_count must be at least 2 (N-1 covariance)')
    n = stats.count
    too_few = [(c, int(n[c])) for c in range(len(n)) if n[c] < min_count]
    if too_few:
        raise ValueError(f'classes with fewer than {min_count} voxels '
                         f'(class, count): {too_few}')
    means = stats.sum / n[:, None]
    covs = (stats.outer - n[:, None, None] * means[:, :, None] *
            means[:, None, :]) / (n[:, None, None] - 1)
    covs = 0.5 * (covs + covs.transpose(1, 2))
    prec_chol, log_det_prec, jitters = [], [], []
    for c in range(len(n)):
        chol, jitter = cholesky_with_jitter(covs[c])
        eye = torch.eye(chol.shape[0], dtype=chol.dtype, device=chol.device)
        chol_inv = torch.triangular_solve(eye, chol, upper=False).solution
        prec = chol_inv.t().contiguous()
        prec_chol.append(prec)
        log_det_prec.append(torch.log(torch.diagonal(prec)).sum())
        jitters.append(jitter)
    return dict(
        means=means,
        covs=covs,
        prec_chol=torch.stack(prec_chol),
        log_det_prec=torch.stack(log_det_prec),
        log_prior=torch.log(n / n.sum()),
        counts=n.clone(),
        jitter=torch.tensor(jitters, dtype=torch.float64))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_fit_occuq_gmm.py`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add p3former/utils/gmm_fit.py tests/test_fit_occuq_gmm.py
git commit -q -F - <<'EOF'
Add OCCUQ Gaussian fitting from running float64 sums

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 3: The density score

**Files:**
- Create: `p3former/utils/gmm_density.py`
- Test: `tests/test_gmm_density.py`

**Interfaces:**
- Consumes: `GaussianStats` and `finalize` from Task 2, in the tests only.
- Produces:
  - `GMM_KEYS = ('means', 'prec_chol', 'log_det_prec', 'log_prior')`.
  - `no_tf32()`, a context manager.
  - `fingerprint(modules: Iterable[Tuple[str, nn.Module]]) -> str`.
  - `load_gmm(path: str, features: str, expected_fingerprint: str, device) -> Dict[str, Tensor]`: float32 tensors for `GMM_KEYS`.
  - `log_density(z: Tensor[V, D], gmm: dict, chunk: int = 65536) -> Tensor[V]`, computed in `gmm['means'].dtype`.
  - `density_score(z, gmm, chunk=65536) -> Tensor[V]`: asinh(−log q).
  - `point_density_scores(voxel_features: Tensor[V, D], gmm: dict, point2voxel_map: Tensor[N], chunk=65536) -> np.ndarray[N] float32`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gmm_density.py`:

```python
"""Tests for p3former/utils/gmm_density.py (OCCUQ density score).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_gmm_density.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.functional.ood_eval import binary_ood_metrics  # noqa: E402
from p3former.utils.gmm_density import (GMM_KEYS, density_score,  # noqa: E402
                                        fingerprint, load_gmm, log_density,
                                        no_tf32, point_density_scores)
from p3former.utils.gmm_fit import GaussianStats, finalize  # noqa: E402
# tests/ is on sys.path (the script's directory, or pytest's insertion);
# "from tests..." would hit the site-packages "tests" package instead
from test_ood_eval import _ref_metrics  # noqa: E402


def _fitted(dim=3, seed=0):
    """Two well-separated 3-d Gaussians fitted from samples (float64)."""
    g = torch.Generator().manual_seed(seed)
    stats = GaussianStats(2, dim)
    for c, shift in enumerate((0.0, 6.0)):
        z = torch.randn(500, dim, generator=g, dtype=torch.float64)
        stats.update(z * (1 + c) + shift, torch.full((500, ), c),
                     ignore_index=2)
    return finalize(stats, min_count=10)


def test_log_density_matches_multivariate_normal():
    gmm = _fitted()
    z = torch.randn(20, 3, dtype=torch.float64) * 4
    mvn = torch.distributions.MultivariateNormal(
        gmm['means'], covariance_matrix=gmm['covs'])
    expected = torch.logsumexp(
        mvn.log_prob(z[:, None, :]) + gmm['log_prior'], dim=1)
    assert torch.allclose(log_density(z, gmm, chunk=7), expected, atol=1e-9)
    print('PASS test_log_density_matches_multivariate_normal')


def test_far_points_score_higher():
    gmm = _fitted()
    near = gmm['means'][0][None]
    far = near + 50.0
    scores = density_score(torch.cat([near, far]), gmm)
    assert scores[1] > scores[0]
    print('PASS test_far_points_score_higher')


def test_no_tf32_restores_the_flag():
    for initial in (True, False):
        torch.backends.cuda.matmul.allow_tf32 = initial
        try:
            with no_tf32():
                assert torch.backends.cuda.matmul.allow_tf32 is False
                raise RuntimeError('boom')
        except RuntimeError:
            pass
        assert torch.backends.cuda.matmul.allow_tf32 is initial
    torch.backends.cuda.matmul.allow_tf32 = True  # torch 1.10's default
    print('PASS test_no_tf32_restores_the_flag')


def test_asinh_keeps_binned_metrics_exact_on_heavy_tails():
    rng = np.random.RandomState(0)
    s = np.concatenate([rng.normal(0, 1, 20000), rng.normal(1.5, 1, 1000),
                        np.full(5, 1e12)])  # five far-off ID points
    y = np.concatenate([np.zeros(20000, bool), np.ones(1000, bool),
                        np.zeros(5, bool)])
    exact = _ref_metrics(s, y)
    binned = binary_ood_metrics([np.arcsinh(s)], [y])
    for got, ref in zip((binned['auroc'], binned['ap'], binned['fpr95']),
                        exact):
        assert abs(got - ref) < 1e-3, (got, ref)
    raw = binary_ood_metrics([s], [y])  # why the score is compressed
    assert abs(raw['auroc'] - exact[0]) > 0.05
    print('PASS test_asinh_keeps_binned_metrics_exact_on_heavy_tails')


def test_fingerprint_and_load_gmm_guards():
    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    fp_a = fingerprint([('m', a)])
    assert fp_a == fingerprint([('m', a)])
    assert fp_a != fingerprint([('m', b)])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'gmm.pth')
        torch.save(dict(_fitted(), features='pe', fingerprint=fp_a), path)
        loaded = load_gmm(path, 'pe', fp_a, 'cpu')
        assert set(loaded) == set(GMM_KEYS)
        assert loaded['means'].dtype == torch.float32
        for features, fp in (('pe', fingerprint([('m', b)])),
                             ('head', fp_a)):
            try:
                load_gmm(path, features, fp, 'cpu')
            except ValueError:
                pass
            else:
                raise AssertionError(f'accepted features={features}')
        try:
            load_gmm(os.path.join(tmp, 'missing.pth'), 'pe', fp_a, 'cpu')
        except FileNotFoundError as err:
            assert 'fit_occuq_gmm.py' in str(err)
        else:
            raise AssertionError('a missing file was accepted')
    print('PASS test_fingerprint_and_load_gmm_guards')


def test_point_projection_and_non_finite():
    gmm = {k: v.float() for k, v in _fitted().items() if k in GMM_KEYS}
    voxels = torch.tensor([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]])
    p2v = torch.tensor([1, 0, 1])
    pts = point_density_scores(voxels, gmm, p2v)
    assert pts.dtype == np.float32 and pts.shape == (3, )
    assert np.allclose(pts, density_score(voxels, gmm)[p2v].numpy())
    bad = dict(gmm, means=gmm['means'] * float('nan'))
    try:
        density_score(voxels, bad)
    except FloatingPointError:
        pass
    else:
        raise AssertionError('a NaN density was accepted')
    print('PASS test_point_projection_and_non_finite')


if __name__ == '__main__':
    test_log_density_matches_multivariate_normal()
    test_far_points_score_higher()
    test_no_tf32_restores_the_flag()
    test_asinh_keeps_binned_metrics_exact_on_heavy_tails()
    test_fingerprint_and_load_gmm_guards()
    test_point_projection_and_non_finite()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_gmm_density.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'p3former.utils.gmm_density'`.

- [ ] **Step 3: Write the implementation**

Create `p3former/utils/gmm_density.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_gmm_density.py tests/test_ood_eval.py`
Expected: `6 passed` from the new file, plus the existing `test_ood_eval.py` tests still passing.

- [ ] **Step 5: Commit**

```bash
git add p3former/utils/gmm_density.py tests/test_gmm_density.py
git commit -q -F - <<'EOF'
Add the OCCUQ density score

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 4: Wire OCCUQ into `_P3FormerHead`

**Files:**
- Modify: `p3former/decode_heads/p3former_head.py`
  - imports (line 15);
  - `__init__` signature (lines 192-219) and the block after `self.sem_queries` (line 256);
  - `forward` (lines 408-434);
  - `loss` (lines 436-454);
  - `predict` (lines 701-734).
- Test: `tests/test_p3former_head_occuq.py`

**Interfaces:**
- Consumes:
  - `_OCCUQHead` (Task 1);
  - `fingerprint`, `load_gmm` and `point_density_scores` (Task 3);
  - `GaussianStats` and `finalize` (Task 2), in the tests only.
- Produces, on `_P3FormerHead`:
  - **Constructor:** a new argument `occuq_cfg=None`. Its keys are listed in `OCCUQ_CFG_KEYS`: `head`, `freeze_base`, `loss_weight`, `gmm_file`, `pe_gmm_file` and `score_chunk`.
  - **Attributes:** `occuq_cfg`, and `occuq_head`, which is an `_OCCUQHead` or `None`.
  - **`forward(features, voxel_coors)`** now returns 5 items. The 5th, `occuq_out`, is `None` without `occuq_cfg`; otherwise it is `dict(pe=List[Tensor[V_b, C]], logits=List[Tensor[V_b, K]] | None, feature=List[Tensor[V_b, C]] | None)`.
  - **`extract_pe_features(features, voxel_coors) -> List[Tensor[V_b, C]]`**, which doesn't run the decoder.
  - **`occuq_losses(logits: List[Tensor], batch_data_samples) -> dict`**, with the keys `loss_occuq_ce` and `loss_occuq_lovasz`.
  - **`density_scores(occuq_out, batch_idx, point2voxel_map) -> Dict[str, np.ndarray]`**, with the keys `density` and/or `density_pe`.
  - **`density_fingerprint(source: str) -> str`**, where `source` is `'pe'` or `'head'`.
  - **`predict`** adds the density keys to each sample's OOD score dict.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_p3former_head_occuq.py`:

```python
"""Tests for the OCCUQ integration in _P3FormerHead (occuq_cfg).

Builds a tiny head on the CPU (about 1 s).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_p3former_head_occuq.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.registry import init_default_scope  # noqa: E402

init_default_scope('mmdet3d')
import mmdet.models  # noqa: E402,F401  registers the mmdet.* losses
import mmdet3d.models  # noqa: E402,F401  registers LovaszLoss
from mmdet3d.registry import MODELS  # noqa: E402
from mmdet3d.structures import Det3DDataSample, PointData  # noqa: E402

import p3former.decode_heads.p3former_head  # noqa: E402,F401
from p3former.utils.gmm_fit import GaussianStats, finalize  # noqa: E402

NUM_VOXELS = 30


def build_head(occuq_cfg=None, seed=0):
    torch.manual_seed(seed)
    return MODELS.build(
        dict(
            type='_P3FormerHead',
            num_classes=5,
            num_queries=4,
            embed_dims=16,
            thing_class=[0, 1],
            stuff_class=[2, 3],
            ignore_index=4,
            num_decoder_layers=1,
            cls_channels=(16, 16, 5),
            mask_channels=(16, 16, 16, 16, 16),
            point_cloud_range=[0, -3.14159265359, -3, 50, 3.14159265359, 13],
            loss_mask=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                           gamma=2.0, alpha=0.25, reduction='mean',
                           loss_weight=1.0),
            loss_dice=dict(type='mmdet.DiceLoss', loss_weight=2.0),
            loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                          gamma=4.0, alpha=0.25, loss_weight=1.0),
            ood_cfg=dict(num_ood_logits=4),
            occuq_cfg=occuq_cfg))


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    coors = torch.cat([
        torch.zeros(NUM_VOXELS, 1, dtype=torch.int32),
        torch.randint(0, 30, (NUM_VOXELS, 3), generator=g, dtype=torch.int32)
    ], 1)
    return torch.randn(NUM_VOXELS, 16, generator=g), coors


def test_existing_outputs_unchanged_by_the_occuq_head():
    base = build_head(None)
    occuq = build_head(dict(head=True))
    missing, unexpected = occuq.load_state_dict(base.state_dict(),
                                                strict=False)
    assert missing and all(k.startswith('occuq_head.') for k in missing)
    assert not unexpected
    base.eval()
    occuq.eval()
    feats, coors = inputs()
    with torch.no_grad():
        a = base.forward(feats.clone(), coors)
        b = occuq.forward(feats.clone(), coors)
        pe = occuq.extract_pe_features(feats.clone(), coors)
    assert len(a) == len(b) == 5 and a[4] is None
    for x, y in zip(a[3], b[3]):  # sem_preds
        assert torch.equal(x, y)
    for x, y in zip(a[1][-1], b[1][-1]):  # last layer's mask predictions
        assert torch.equal(x, y)
    assert b[4]['logits'][0].shape == (NUM_VOXELS, 5)
    assert b[4]['feature'][0].shape == (NUM_VOXELS, 16)
    assert torch.equal(b[4]['pe'][0], pe[0])
    print('PASS test_existing_outputs_unchanged_by_the_occuq_head')


def test_freeze_base_loss_skips_the_decoder():
    head = build_head(dict(freeze_base=True))
    feats, coors = inputs()
    sample = Det3DDataSample()
    sample.gt_pts_seg = PointData(
        voxel_semantic_mask=torch.randint(0, 5, (NUM_VOXELS, )))
    losses = head.loss(
        dict(features=feats, voxels=dict(voxel_coors=coors)), [sample],
        train_cfg=None)
    assert set(losses) == {'loss_occuq_ce', 'loss_occuq_lovasz'}
    sum(v.mean() for v in losses.values()).backward()
    with_grad = {n for n, p in head.named_parameters() if p.grad is not None}
    assert any(n.startswith('occuq_head.') for n in with_grad)
    decoder = ('transformer_decoder.', 'fc_mask.', 'fc_cls.', 'sem_queries.',
               'queries.')
    assert not any(n.startswith(decoder) for n in with_grad), with_grad
    print('PASS test_freeze_base_loss_skips_the_decoder')


def test_predict_emits_density_scores_and_checks_the_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        files = dict(head=os.path.join(tmp, 'gmm_head.pth'),
                     pe=os.path.join(tmp, 'gmm_pe.pth'))
        cfg = dict(head=True, gmm_file=files['head'],
                   pe_gmm_file=files['pe'])
        head = build_head(cfg).eval()
        feats, coors = inputs()
        with torch.no_grad():
            out = head.forward(feats.clone(), coors)[4]
        for source, key in (('head', 'feature'), ('pe', 'pe')):
            stats = GaussianStats(2, 16)
            stats.update(out[key][0], torch.arange(NUM_VOXELS) % 2,
                         ignore_index=2)
            torch.save(
                dict(finalize(stats, min_count=2), features=source,
                     fingerprint=head.density_fingerprint(source)),
                files[source])
        sample = Det3DDataSample()
        sample.gt_pts_seg = PointData(
            point2voxel_map=torch.randint(0, NUM_VOXELS, (50, )))
        with torch.no_grad():
            _, _, scores = head.predict(
                dict(features=feats.clone(), voxels=dict(voxel_coors=coors)),
                [sample])
        for key in ('density', 'density_pe'):
            assert scores[0][key].shape == (50, ), key
            assert np.isfinite(scores[0][key]).all(), key
        # a head with other weights must refuse these Gaussians
        other = build_head(cfg, seed=1).eval()
        try:
            with torch.no_grad():
                other.predict(
                    dict(features=feats.clone(),
                         voxels=dict(voxel_coors=coors)), [sample])
        except ValueError as err:
            assert 'fitted for other weights' in str(err)
        else:
            raise AssertionError('Gaussians of other weights were accepted')
    print('PASS test_predict_emits_density_scores_and_checks_the_fingerprint')


def test_invalid_occuq_cfg_is_rejected():
    for cfg in (dict(freeze_base=True, head=False),
                dict(head=False, gmm_file='x.pth'),
                dict(unknown_key=1)):
        try:
            build_head(cfg)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted occuq_cfg={cfg}')
    print('PASS test_invalid_occuq_cfg_is_rejected')


if __name__ == '__main__':
    test_existing_outputs_unchanged_by_the_occuq_head()
    test_freeze_base_loss_skips_the_decoder()
    test_predict_emits_density_scores_and_checks_the_fingerprint()
    test_invalid_occuq_cfg_is_rejected()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_p3former_head_occuq.py`
Expected: FAIL. `_P3FormerHead.__init__() got an unexpected keyword argument 'occuq_cfg'`, raised as a `TypeError` while building the head.

- [ ] **Step 3: Write the implementation**

In `p3former/decode_heads/p3former_head.py`:

(a) Below `from p3former.utils.ood_scores import point_ood_scores` (line 15), add:

```python
from p3former.decode_heads.occuq_head import _OCCUQHead
from p3former.utils.gmm_density import (fingerprint, load_gmm,
                                        point_density_scores)

# decode_head.occuq_cfg keys (spec 2026-10-07-occuq-density-design.md)
OCCUQ_CFG_KEYS = ('head', 'freeze_base', 'loss_weight', 'gmm_file',
                  'pe_gmm_file', 'score_chunk')
# the layers that produce pe_features (pe_type='mpe'); the density
# fingerprint covers them
MPE_MODULES = ('polar_proj', 'polar_norm', 'cart_proj', 'cart_norm',
               'pe_conv')
```

(b) In the `__init__` signature, replace

```python
                 point_cloud_range=[],
                 ood_cfg=None):
```

with

```python
                 point_cloud_range=[],
                 ood_cfg=None,
                 occuq_cfg=None):
```

(c) After the `if use_sem_loss:` block, which ends with `self.sem_queries = nn.Conv3d(...)` (line 256), and before `# build assigner`, insert:

```python
        # OCCUQ (spec 2026-10-07): a spectrally normalised head next to
        # sem_queries, whose weights also seed the stuff queries
        self.occuq_cfg = occuq_cfg
        self.occuq_head = None
        # source ('head' / 'pe') -> Gaussian tensors, loaded on the first
        # predict; a plain dict, so it never enters a checkpoint
        self._occuq_gmms = {}
        if occuq_cfg is not None:
            self._check_occuq_cfg(occuq_cfg, ood_cfg, use_sem_loss, pe_type)
            if occuq_cfg.get('head', True):
                self.occuq_head = _OCCUQHead(embed_dims, num_classes)
```

(d) In `forward`, replace

```python
        queries, features, mpe, sem_preds = self.init_inputs(
            feature_split, voxel_coor_split, batch_size)
```

with

```python
        queries, features, mpe, sem_preds = self.init_inputs(
            feature_split, voxel_coor_split, batch_size)
        occuq_out = self.occuq_forward(features)
```

and replace its last line

```python
        return class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds
```

with

```python
        return (class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer,
                sem_preds, occuq_out)
```

(e) Replace the whole `loss` method with:

```python
    def loss(self, batch_inputs, batch_data_samples, train_cfg):
        if self.occuq_cfg is not None and self.occuq_cfg.get(
                'freeze_base', False):
            # OCCUQ variant A: only the OCCUQ head trains, so the decoder,
            # the matching and the panoptic losses are skipped
            pe_features = self.extract_pe_features(
                batch_inputs['features'],
                batch_inputs['voxels']['voxel_coors'])
            logits = [self.occuq_head(f)[0] for f in pe_features]
            return self.occuq_losses(logits, batch_data_samples)
        class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, sem_preds, occuq_out = self.forward(batch_inputs['features'], batch_inputs['voxels']['voxel_coors'])
        cls_targets_buffer, mask_targets_buffer, label_weights_buffer = self.bipartite_matching(class_preds_buffer, mask_preds_buffer, pos_mask_preds_buffer, batch_data_samples)
        losses = dict()
        for i in range(self.num_decoder_layers+1):
            losses.update(self.loss_single_layer(class_preds_buffer[i], mask_preds_buffer[i], pos_mask_preds_buffer[i],
                                                cls_targets_buffer[i], mask_targets_buffer[i], label_weights_buffer[i], i))
        if self.use_sem_loss:
            gt_semantic_segs = [
                data_sample.gt_pts_seg.voxel_semantic_mask
                for data_sample in batch_data_samples
            ]
            seg_label = torch.cat(gt_semantic_segs)
            sem_preds = torch.cat(sem_preds, dim=0)
            losses['loss_ce'] = self.loss_ce(
                sem_preds, seg_label, ignore_index=self.ignore_index)
            losses['loss_lovasz'] = self.loss_lovasz(
                sem_preds, seg_label, ignore_index=self.ignore_index)
        if occuq_out is not None and occuq_out['logits'] is not None:
            losses.update(
                self.occuq_losses(occuq_out['logits'], batch_data_samples))
        return losses

    @staticmethod
    def _check_occuq_cfg(occuq_cfg, ood_cfg, use_sem_loss, pe_type):
        """Reject occuq_cfg combinations that cannot work."""
        unknown = sorted(set(occuq_cfg) - set(OCCUQ_CFG_KEYS))
        if unknown:
            raise KeyError(f'unknown occuq_cfg keys {unknown}; expected '
                           f'{OCCUQ_CFG_KEYS}')
        head = occuq_cfg.get('head', True)
        if pe_type != 'mpe':
            raise ValueError("occuq_cfg needs pe_type='mpe': the density "
                             'fingerprint covers the MPE layers')
        if head and not use_sem_loss:
            raise ValueError('occuq_cfg.head needs use_sem_loss=True: the '
                             'OCCUQ head reuses its CE and Lovasz losses')
        if occuq_cfg.get('freeze_base', False) and not head:
            raise ValueError('occuq_cfg.freeze_base needs head=True: '
                             'nothing else would train')
        if occuq_cfg.get('gmm_file') and not head:
            raise ValueError('occuq_cfg.gmm_file needs head=True: density '
                             'is computed on the OCCUQ head feature')
        if (occuq_cfg.get('gmm_file')
                or occuq_cfg.get('pe_gmm_file')) and ood_cfg is None:
            raise ValueError('occuq_cfg Gaussian files need ood_cfg: the '
                             'density scores are emitted with the OOD scores')

    def occuq_forward(self, pe_features):
        """OCCUQ outputs per sample: dict(pe, logits, feature), with logits
        and feature None without the head; None without occuq_cfg."""
        if self.occuq_cfg is None:
            return None
        out = dict(pe=pe_features, logits=None, feature=None)
        if self.occuq_head is not None:
            heads = [self.occuq_head(f) for f in pe_features]
            out['logits'] = [logits for logits, _ in heads]
            out['feature'] = [feature for _, feature in heads]
        return out

    def extract_pe_features(self, features, voxel_coors):
        """Per-sample pe_features (the semantic branch's input), computed
        as in forward but without the decoder."""
        batch_size = int(voxel_coors[:, 0].max().item()) + 1
        feature_split = [
            features[voxel_coors[:, 0] == i] for i in range(batch_size)
        ]
        coor_split = [
            voxel_coors[voxel_coors[:, 0] == i] for i in range(batch_size)
        ]
        pe_features, _ = self.mpe(feature_split, coor_split, batch_size)
        return pe_features

    def occuq_losses(self, logits, batch_data_samples):
        """The OCCUQ head's CE and Lovasz losses on the voxel labels."""
        seg_label = torch.cat([
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ])
        logits = torch.cat(logits, dim=0)
        weight = self.occuq_cfg.get('loss_weight', 1.0)
        return dict(
            loss_occuq_ce=weight * self.loss_ce(
                logits, seg_label, ignore_index=self.ignore_index),
            loss_occuq_lovasz=weight * self.loss_lovasz(
                logits, seg_label, ignore_index=self.ignore_index))

    def density_scores(self, occuq_out, batch_idx, point2voxel_map):
        """OCCUQ density scores of one sample, for the configured Gaussian
        files: 'density' (OCCUQ head feature) and/or 'density_pe'
        (pe_features), as per-point float32 arrays."""
        scores = {}
        if occuq_out is None:
            return scores
        chunk = self.occuq_cfg.get('score_chunk', 65536)
        for key, source, file_key in (('density', 'head', 'gmm_file'),
                                      ('density_pe', 'pe', 'pe_gmm_file')):
            path = self.occuq_cfg.get(file_key)
            if not path:
                continue
            feats = occuq_out['feature' if source == 'head' else 'pe'][
                batch_idx]
            if source not in self._occuq_gmms:
                self._occuq_gmms[source] = load_gmm(
                    path, source, self.density_fingerprint(source),
                    feats.device)
            scores[key] = point_density_scores(
                feats, self._occuq_gmms[source], point2voxel_map, chunk)
        return scores

    def density_fingerprint(self, source):
        """SHA-256 of the weights that produce the density feature of
        ``source``: the MPE layers for 'pe', plus the OCCUQ head for
        'head'."""
        names = list(MPE_MODULES)
        if source == 'head':
            names.append('occuq_head')
        return fingerprint((name, getattr(self, name)) for name in names)
```

(f) In `predict`, replace the first line

```python
        class_preds_buffer, mask_preds_buffer, _, sem_preds = self.forward(batch_inputs['features'], batch_inputs['voxels']['voxel_coors'])
```

with

```python
        class_preds_buffer, mask_preds_buffer, _, sem_preds, occuq_out = self.forward(batch_inputs['features'], batch_inputs['voxels']['voxel_coors'])
```

and replace the `pts_ood_scores.append(...)` block inside `if self.ood_cfg is not None:` with

```python
                scores = point_ood_scores(
                    voxel_logits,
                    point2voxel_map,
                    odin_temperature=self.ood_cfg.get(
                        'odin_temperature', 1000.0),
                    energy_temperature=self.ood_cfg.get(
                        'energy_temperature', 1.0))
                scores.update(
                    self.density_scores(occuq_out, batch_idx,
                                        point2voxel_map))
                pts_ood_scores.append(scores)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_p3former_head_occuq.py tests/test_ood_scores.py tests/test_ood_metric.py`
Expected: all pass, `4 passed` from the new file.

- [ ] **Step 5: Commit**

```bash
git add p3former/decode_heads/p3former_head.py tests/test_p3former_head_occuq.py
git commit -q -F - <<'EOF'
Wire the OCCUQ head and density scores into _P3FormerHead

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 5: Freeze everything but the OCCUQ head (variant A)

**Files:**
- Create: `p3former/utils/freeze.py`
- Modify: `p3former/segmentors/p3former.py` (whole class)
- Test: `tests/test_occuq_freeze.py`

**Interfaces:**
- Consumes: `_P3FormerHead.occuq_cfg` and `.occuq_head` (Task 4).
- Produces:
  - `freeze_all_but(model: nn.Module, keep: nn.Module) -> None`.
  - `eval_all_but(model: nn.Module, keep: nn.Module) -> None`.
  - `_P3Former.freeze_base: bool`.
  - `_P3Former.train(mode=True)`: with `freeze_base`, the voxel encoder, the backbone and the decode head outside `occuq_head` stay in eval mode, and the data preprocessor keeps `mode`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_occuq_freeze.py`:

```python
"""Tests for OCCUQ variant A's freezing (p3former/utils/freeze.py and
_P3Former). Builds the real DSO model on the CPU (about 3 s).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_occuq_freeze.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from p3former.decode_heads.occuq_head import _OCCUQHead  # noqa: E402
from p3former.utils.freeze import eval_all_but, freeze_all_but  # noqa: E402


def test_helpers_on_a_toy_model():
    keep = _OCCUQHead(in_channels=8, num_classes=3)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8),
                                torch.nn.BatchNorm1d(8), keep)
    freeze_all_but(model, keep)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith('2.') for n in trainable)
    model.train()
    eval_all_but(model, keep)
    assert not model[0].training and not model[1].training
    assert all(m.training for m in keep.modules())
    before = model[1].running_mean.clone()
    model(torch.randn(16, 8))
    assert torch.equal(model[1].running_mean, before)  # BN stats fixed
    print('PASS test_helpers_on_a_toy_model')


def test_variant_a_freezes_the_real_model():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    cfg = Config.fromfile(
        os.path.join(ROOT, 'configs/p3former/p3former_2xb1_3x_dso_ood.py'))
    cfg.model.decode_head.occuq_cfg = dict(freeze_base=True)
    init_default_scope('mmdet3d')
    from mmdet3d.registry import MODELS
    model = MODELS.build(cfg.model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith('decode_head.occuq_head.') for n in trainable)
    model.train()
    # the preprocessor builds voxel_semantic_mask only in training mode
    assert model.data_preprocessor.training
    occuq = set(model.decode_head.occuq_head.modules())
    for name, module in model.named_modules():
        if module in occuq:
            assert module.training, name
        elif name.startswith(('voxel_encoder', 'backbone', 'decode_head')):
            assert not module.training, name
    model.eval()
    assert not any(m.training for m in model.modules())
    print('PASS test_variant_a_freezes_the_real_model')


if __name__ == '__main__':
    test_helpers_on_a_toy_model()
    test_variant_a_freezes_the_real_model()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_occuq_freeze.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'p3former.utils.freeze'`.

- [ ] **Step 3: Write the implementation**

Create `p3former/utils/freeze.py`:

```python
"""Freeze all of a model except one submodule (OCCUQ variant A)."""
import torch.nn as nn


def freeze_all_but(model: nn.Module, keep: nn.Module) -> None:
    """Set requires_grad=False on every parameter of ``model`` outside
    ``keep``, and True on the parameters inside it."""
    keep_ids = {id(p) for p in keep.parameters()}
    for param in model.parameters():
        param.requires_grad_(id(param) in keep_ids)


def eval_all_but(model: nn.Module, keep: nn.Module) -> None:
    """Put every module of ``model`` outside ``keep`` in eval mode, and
    leave ``keep``'s modules as they are.

    Call it after ``model.train()``: BatchNorm statistics and dropout of the
    frozen part then stay fixed. The flag is set module by module, because
    ``Module.eval()`` would recurse into ``keep``."""
    keep_ids = {id(m) for m in keep.modules()}
    for module in model.modules():
        if id(module) not in keep_ids:
            module.training = False
```

Replace `p3former/segmentors/p3former.py` with:

```python
import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.segmentors.cylinder3d import Cylinder3D
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

from p3former.utils.freeze import eval_all_but, freeze_all_but


@MODELS.register_module()
class _P3Former(Cylinder3D):
    """P3Former."""

    # OCCUQ variant A (decode_head.occuq_cfg.freeze_base): only the OCCUQ
    # head trains (spec docs/superpowers/specs/2026-10-07-occuq-density-design.md)
    freeze_base = False

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(voxel_encoder=voxel_encoder,
                        backbone=backbone,
                        decode_head=decode_head,
                        neck=neck,
                        auxiliary_head=auxiliary_head,
                        loss_regularization=loss_regularization,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg,
                        data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)
        occuq_cfg = getattr(self.decode_head, 'occuq_cfg', None)
        if occuq_cfg is not None and occuq_cfg.get('freeze_base', False):
            self.freeze_base = True
            freeze_all_but(self, self.decode_head.occuq_head)

    def train(self, mode: bool = True):
        """As nn.Module.train.

        With freeze_base, the voxel encoder, the backbone and the decode
        head outside the OCCUQ head stay in eval mode (fixed BatchNorm
        statistics). The data preprocessor keeps ``mode``, because it builds
        the voxel labels only in training mode."""
        super().train(mode)
        if self.freeze_base:
            for part in (self.voxel_encoder, self.backbone,
                         self.decode_head):
                eval_all_but(part, self.decode_head.occuq_head)
        return self

    def loss(self, batch_inputs_dict,batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone; frozen in OCCUQ variant A
        if self.freeze_base:
            with torch.no_grad():
                x = self.extract_feat(batch_inputs_dict)
        else:
            x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        losses = dict()
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        pts_semantic_preds, pts_instance_preds, pts_ood_scores = \
            self.decode_head.predict(batch_inputs_dict, batch_data_samples)
        return self.postprocess_result(pts_semantic_preds,
                                       pts_instance_preds,
                                       batch_data_samples,
                                       pts_ood_scores)

    def postprocess_result(self, pts_semantic_preds, pts_instance_preds,
                           batch_data_samples, pts_ood_scores=None):
        for i in range(len(pts_semantic_preds)):
            seg_data = {'pts_semantic_mask': pts_semantic_preds[i],
                        'pts_instance_mask': pts_instance_preds[i]}
            if pts_ood_scores is not None:
                for key, value in pts_ood_scores[i].items():
                    seg_data[f'ood_{key}'] = value
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**seg_data)})
        return batch_data_samples
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_occuq_freeze.py tests/test_p3former_head_occuq.py`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add p3former/utils/freeze.py p3former/segmentors/p3former.py tests/test_occuq_freeze.py
git commit -q -F - <<'EOF'
Freeze everything but the OCCUQ head for variant A

The data preprocessor keeps its training flag: it builds the voxel labels
the OCCUQ head's loss needs only in training mode.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 6: The OCCUQ configs

**Files:**
- Create: `configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py`, `configs/p3former/p3former_2xb1_3x_dso_occuq_a.py`, `configs/p3former/p3former_2xb1_3x_dso_occuq_c.py`
- Test: `tests/test_occuq_configs.py`

**Interfaces:**
- Consumes: `occuq_cfg` (Task 4) and `freeze_base` (Task 5).
- Produces:
  - **Configs:** `test_evaluator[1].score_keys` is the five flat keys plus `density_pe` (pe) or `density` (A, C). `val_evaluator` is PQ-only.
  - **Gaussian file:** set per run with `--cfg-options model.decode_head.occuq_cfg.gmm_file=...`, or `pe_gmm_file=...` for pe.

- [ ] **Step 1: Write the failing test**

Create `tests/test_occuq_configs.py`:

```python
"""Tests for the OCCUQ configs (configs/p3former/p3former_2xb1_3x_dso_occuq_*.py).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_occuq_configs.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmengine.config import Config  # noqa: E402

FLAT = ('msp', 'maxlogit', 'odin', 'energy', 'entropy')


def load(variant):
    return Config.fromfile(
        os.path.join(ROOT, 'configs/p3former/'
                     f'p3former_2xb1_3x_dso_occuq_{variant}.py'))


def test_pe_config():
    cfg = load('pe')
    assert dict(cfg.model.decode_head.occuq_cfg) == dict(head=False)
    assert tuple(cfg.test_evaluator[1].score_keys) == FLAT + ('density_pe', )
    assert cfg.test_evaluator[0].learning_map_inv[24] == 0
    print('PASS test_pe_config')


def test_training_configs():
    for variant, freeze in (('a', True), ('c', False)):
        cfg = load(variant)
        assert cfg.model.decode_head.occuq_cfg.freeze_base is freeze
        assert tuple(cfg.test_evaluator[1].score_keys) == FLAT + ('density', )
        assert cfg.test_evaluator[0].learning_map_inv[24] == 0
        # end-of-training validation runs before any Gaussians exist
        assert [m.type for m in cfg.val_evaluator] == ['_PanopticSegMetric']
        assert cfg.load_from == 'work_dirs/p3former_2xb1_3x_dso/epoch_36.pth'
        opt = cfg.optim_wrapper.optimizer
        assert (opt.type, opt.lr, opt.weight_decay) == ('AdamW', 2e-4, 0.01)
        assert cfg.train_cfg.max_epochs == 9
        assert cfg.train_cfg.val_interval == 9
        warmup, cosine = cfg.param_scheduler
        assert (warmup.type, warmup.end, warmup.by_epoch) == ('LinearLR', 500,
                                                              False)
        assert abs(warmup.start_factor - 1 / 3) < 1e-12
        assert (cosine.type, cosine.T_max, cosine.eta_min) == (
            'CosineAnnealingLR', 9, 2e-7)
        assert cosine.convert_to_iter_based is True
        assert cfg.default_hooks.checkpoint.interval == 9
        assert cfg.default_hooks.checkpoint.max_keep_ckpts == 1
        assert cfg.randomness.seed == 0
        assert cfg.train_dataloader.batch_size == 1
    print('PASS test_training_configs')


if __name__ == '__main__':
    test_pe_config()
    test_training_configs()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_occuq_configs.py`
Expected: FAIL with `FileNotFoundError` for `p3former_2xb1_3x_dso_occuq_pe.py`.

- [ ] **Step 3: Write the configs**

Create `configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py`:

```python
_base_ = ['./p3former_2xb1_3x_dso_ood.py']

# OCCUQ, variant pe (docs/superpowers/specs/2026-10-07-occuq-density-design.md).
# Class-conditional Gaussians on the base checkpoint's pe_features, scored as
# density_pe. It is the signal check before any training, and the ablation
# without the spectrally normalised head. Evaluation only.
# Fit the Gaussians:
#   python tools/fit_occuq_gmm.py configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py \
#       work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe \
#       --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth --check 5
# then pass the file to test.py:
#   --cfg-options model.decode_head.occuq_cfg.pe_gmm_file=work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth
model = dict(decode_head=dict(occuq_cfg=dict(head=False)))

pq_metric = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    stuff_class_inds=[
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ],
    min_num_points=50,
    id_offset=2**16,
    dataset_type='dso',
    learning_map_inv={{_base_.learning_map_inv}})
val_evaluator = [pq_metric]
test_evaluator = [
    pq_metric,
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24,
        score_keys=('msp', 'maxlogit', 'odin', 'energy', 'entropy',
                    'density_pe')),
]
```

Create `configs/p3former/p3former_2xb1_3x_dso_occuq_a.py`:

```python
_base_ = ['./p3former_2xb1_3x_dso_ood.py']

# OCCUQ, variant A (docs/superpowers/specs/2026-10-07-occuq-density-design.md).
# The OCCUQ head (_OCCUQHead, OCCUQ's spectrally normalised residual MLP) is
# trained alone on top of the frozen base model, with OCCUQ's fine-tuning
# recipe scaled to P3Former: a quarter of the base schedule's epochs and
# learning rate, cosine with a 500-iteration warmup.
# Train (for seeds 1 and 2, add --cfg-options randomness.seed=N):
#   bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso_occuq_a.py 2 \
#       --work-dir work_dirs/p3former_2xb1_3x_dso_occuq_a/seed0
# Then fit the Gaussians (tools/fit_occuq_gmm.py --features head) and pass
# the file to test.py:
#   --cfg-options model.decode_head.occuq_cfg.gmm_file=<seed dir>/gmm_head.pth
model = dict(decode_head=dict(occuq_cfg=dict(freeze_base=True)))

load_from = 'work_dirs/p3former_2xb1_3x_dso/epoch_36.pth'
randomness = dict(seed=0)

optim_wrapper = dict(optimizer=dict(lr=2e-4))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=9, val_interval=9)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=9,
        eta_min=2e-7,
        by_epoch=True,
        begin=0,
        end=9,
        convert_to_iter_based=True),
]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=9, max_keep_ckpts=1))

pq_metric = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    stuff_class_inds=[
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ],
    min_num_points=50,
    id_offset=2**16,
    dataset_type='dso',
    learning_map_inv={{_base_.learning_map_inv}})
# validation at the end of training: no Gaussians exist yet, so PQ only
val_evaluator = [pq_metric]
test_evaluator = [
    pq_metric,
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[17, 28],
        seg_offset=2**16,
        ignore_index=24,
        score_keys=('msp', 'maxlogit', 'odin', 'energy', 'entropy',
                    'density')),
]
```

Create `configs/p3former/p3former_2xb1_3x_dso_occuq_c.py`:

```python
_base_ = ['./p3former_2xb1_3x_dso_occuq_a.py']

# OCCUQ, variant C (docs/superpowers/specs/2026-10-07-occuq-density-design.md).
# As variant A, but the whole model is fine-tuned end to end with the OCCUQ
# head attached, as OCCUQ does: P3Former's losses plus the head's CE and
# Lovasz. The commands are variant A's, with this config and
# work_dirs/p3former_2xb1_3x_dso_occuq_c/seed<N>.
model = dict(decode_head=dict(occuq_cfg=dict(freeze_base=False)))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_occuq_configs.py`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py configs/p3former/p3former_2xb1_3x_dso_occuq_a.py configs/p3former/p3former_2xb1_3x_dso_occuq_c.py tests/test_occuq_configs.py
git commit -q -F - <<'EOF'
Add the OCCUQ configs (pe, A, C)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 7: The fitting tool

**Files:**
- Create: `tools/fit_occuq_gmm.py`

**Interfaces:**
- Consumes:
  - `GaussianStats`, `finalize` and `voxel_majority_labels` (Task 2);
  - `GMM_KEYS` and `log_density` (Task 3);
  - `extract_pe_features`, `occuq_head` and `density_fingerprint` of the head (Task 4);
  - the configs (Task 6).
- Produces:
  - **CLI:** `tools/fit_occuq_gmm.py CONFIG CKPT --features head|pe --out FILE [--ann dso_infos_train.pkl] [--check N] [--limit N] [--workers 4] [--device cuda:0]`.
  - **The file:** `finalize`'s keys plus `features`, `fingerprint`, `checkpoint`, `config`, `ann_file`, `frames` and `class_names`, and `train_accuracy` for `--features head`. `load_gmm` (Task 3) reads it.

Its pure parts are tested in Tasks 2 and 3. The tool itself is exercised end to end in Task 9.

- [ ] **Step 1: Write the tool**

Create `tools/fit_occuq_gmm.py`:

```python
#!/usr/bin/env python
"""Fit OCCUQ's class-conditional Gaussians on the training split.

Runs a P3Former checkpoint over the training split (eval mode, test
pipeline, batch 1) and fits one full-covariance Gaussian per ID class on a
per-voxel feature:
- --features pe: pe_features, the semantic branch's input;
- --features head: the output of the OCCUQ head's last block.

Each voxel is one sample, labelled by the majority vote of its points'
mapped labels. Voxels labelled 24 are skipped: that is ignore, which
includes the OOD classes Stop and Others. The statistics are running
float64 sums, so every voxel is used (p3former/utils/gmm_fit.py).

The output is a torch.save'd dict of plain tensors, plus the fingerprint of
the weights that produced the features; _P3FormerHead refuses the file for
other weights. With --check N, the float32 GPU log-density of N frames must
match float64 on the CPU within 0.05 nats, or nothing is saved. Spec:
docs/superpowers/specs/2026-10-07-occuq-density-design.md.

Run from the repo root (one GPU; ~15-30 min for the 8,474 training frames):
    CUDA_VISIBLE_DEVICES=0 python tools/fit_occuq_gmm.py \\
        configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py \\
        work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe \\
        --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth --check 5
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from p3former.utils.gmm_density import GMM_KEYS, log_density  # noqa: E402
from p3former.utils.gmm_fit import (GaussianStats, finalize,  # noqa: E402
                                    voxel_majority_labels)

NUM_CLASSES = 24  # DSO ID classes
IGNORE_INDEX = 24  # mapped label of ignored points, incl. the OOD classes
MIN_COUNT = 2560  # 10 x 256 voxels per class
CHECK_TOL = 0.05  # nats: float32 GPU log-density vs float64 CPU


def build(config, ann, workers):
    """(cfg, model, dataloader) as test.py builds them, batch size 1, on
    ``ann``; the weights are loaded by ``load_weights``."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(config)  # also imports cfg.custom_imports
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    cfg.test_dataloader.dataset.dataset.ann_file = ann
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = workers
    model = MODELS.build(cfg.model)
    return cfg, model, Runner.build_dataloader(cfg.test_dataloader)


def load_weights(model, checkpoint, features):
    """Load ``checkpoint``. Every weight but the OCCUQ head's must be in it,
    and the OCCUQ head's too for --features head."""
    state = torch.load(checkpoint, map_location='cpu')
    state = state.get('state_dict', state)
    missing, _ = model.load_state_dict(state, strict=False)
    head_missing = [k for k in missing if k.startswith('decode_head.occuq_head.')]
    other_missing = [k for k in missing if k not in head_missing]
    if other_missing:
        raise KeyError(f'{len(other_missing)} model weights are missing from '
                       f'{checkpoint}, e.g. {other_missing[:3]}')
    if features == 'head' and head_missing:
        raise ValueError(f'{checkpoint} has no trained OCCUQ head: fit '
                         '--features head on a variant A or C checkpoint')


def frame_voxels(model, data, features):
    """One frame (batch size 1). Returns the voxel features [V, D], the
    voxel labels [V] (majority vote of the points' mapped labels) and the
    OCCUQ head's logits [V, K] (None for --features pe)."""
    data = model.data_preprocessor(data, False)
    inputs, sample = data['inputs'], data['data_samples'][0]
    x = model.extract_feat(inputs)
    head = model.decode_head
    (pe, ) = head.extract_pe_features(x.features,
                                      inputs['voxels']['voxel_coors'])
    logits, feats = (None, pe) if features == 'pe' else head.occuq_head(pe)
    point_labels = torch.as_tensor(
        np.asarray(sample.eval_ann_info['pts_semantic_mask']).astype(np.int64),
        device=feats.device)
    labels = voxel_majority_labels(point_labels,
                                   sample.gt_pts_seg.point2voxel_map,
                                   feats.shape[0], IGNORE_INDEX + 1)
    return feats, labels, logits


def check_precision(model, loader, features, gmm, frames, device):
    """Largest |log q| difference over the first ``frames`` frames between
    the production path (float32 on ``device``, TF32 off) and float64 on
    the CPU. Raises above CHECK_TOL."""
    gmm32 = {k: gmm[k].to(device=device, dtype=torch.float32) for k in GMM_KEYS}
    gmm64 = {k: gmm[k].to(dtype=torch.float64).cpu() for k in GMM_KEYS}
    worst = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= frames:
                break
            feats, _, _ = frame_voxels(model, data, features)
            fast = log_density(feats.float(), gmm32).double().cpu()
            exact = log_density(feats.double().cpu(), gmm64)
            worst = max(worst, float((fast - exact).abs().max()))
    if worst > CHECK_TOL:
        raise RuntimeError(f'the float32 log-density differs from float64 by '
                           f'{worst:.4g} nats (> {CHECK_TOL}); nothing saved')
    return worst


def report(result):
    """Print the per-class voxel counts, jitters and training accuracy."""
    acc = result.get('train_accuracy')
    print(f"{'class':>18} | {'voxels':>12} | {'jitter':>8} | {'train acc':>9}")
    for c, name in enumerate(result['class_names']):
        a = f'{100 * float(acc[c]):8.2f}%' if acc is not None else '        -'
        print(f"{name:>18} | {int(result['counts'][c]):12d} | "
              f"{float(result['jitter'][c]):8.1e} | {a}")


def fit(config, checkpoint, out, features, ann='dso_infos_train.pkl',
        check=0, limit=0, workers=4, device='cuda:0'):
    """Fit, check and save the Gaussians; returns the saved dict."""
    if osp.exists(out):
        raise FileExistsError(f'{out} exists: delete it or choose another --out')
    cfg, model, loader = build(config, ann, workers)
    if features == 'head' and model.decode_head.occuq_head is None:
        raise ValueError('--features head needs a config whose '
                         'decode_head.occuq_cfg builds the OCCUQ head')
    load_weights(model, checkpoint, features)
    model.to(device).eval()
    stats = None
    correct = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    frames, t0 = 0, time.time()
    with torch.no_grad():
        for data in loader:
            if limit and frames >= limit:
                break
            feats, labels, logits = frame_voxels(model, data, features)
            if stats is None:
                stats = GaussianStats(NUM_CLASSES, feats.shape[1],
                                      feats.device)
            stats.update(feats, labels, IGNORE_INDEX)
            if logits is not None:
                valid = labels != IGNORE_INDEX
                y = labels[valid].cpu()  # CPU: CUDA bincount is slow here
                pred = logits[valid, :NUM_CLASSES].argmax(dim=1).cpu()
                total += torch.bincount(y, minlength=NUM_CLASSES).double()
                correct += torch.bincount(y[pred == y],
                                          minlength=NUM_CLASSES).double()
            frames += 1
            if frames % 500 == 0:
                print(f'{frames} frames ({time.time() - t0:.0f} s)', flush=True)
    if stats is None:
        raise ValueError(f'no frames in {ann}')
    print(f'{frames} frames in {time.time() - t0:.0f} s', flush=True)
    result = {k: v.cpu() for k, v in finalize(stats, MIN_COUNT).items()}
    result.update(
        features=features,
        fingerprint=model.decode_head.density_fingerprint(features),
        checkpoint=checkpoint, config=config, ann_file=ann, frames=frames,
        class_names=list(cfg.get('class_names',
                                 [str(c) for c in range(NUM_CLASSES)])))
    if features == 'head':
        result['train_accuracy'] = correct / total.clamp(min=1)
    report(result)
    if check:
        worst = check_precision(model, loader, features, result, check, device)
        print(f'--check {check}: max |log q float32 GPU - float64 CPU| = '
              f'{worst:.4g} nats')
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    torch.save(result, out)
    print(f'saved {out}')
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='an OCCUQ config (as test.py)')
    ap.add_argument('checkpoint', help='checkpoint file')
    ap.add_argument('--features', choices=('head', 'pe'), required=True)
    ap.add_argument('--out', required=True,
                    help='output .pth; refuses to overwrite')
    ap.add_argument('--ann', default='dso_infos_train.pkl',
                    help='info pkl of the split to fit on')
    ap.add_argument('--check', type=int, default=0, metavar='N',
                    help='compare float32 GPU and float64 CPU log-densities '
                    'on N frames before saving')
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many frames (0 = all; smoke runs)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    fit(args.config, args.checkpoint, args.out, args.features, ann=args.ann,
        check=args.check, limit=args.limit, workers=args.workers,
        device=args.device)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Check that the tool parses and the suite still passes**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python tools/fit_occuq_gmm.py --help && /home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests`
Expected: the usage text, then every test passing.

- [ ] **Step 3: Commit**

```bash
git add tools/fit_occuq_gmm.py
git commit -q -F - <<'EOF'
Add tools/fit_occuq_gmm.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 8: The comparison tool

**Files:**
- Create: `tools/compare_occuq_grouping.py`
- Test: `tests/test_compare_occuq_grouping.py`

**Interfaces:**
- Consumes: logs only.
  - Grouping: `<grouping-dir>/divided_mass/robust.tsv`, and `<grouping-dir>/{bipartitions,singletons}{,_test,_test_cetran}/bipartitions.log`.
  - OCCUQ: `<run-dir>/<set>/<timestamp>/<timestamp>.log`, as `test.py --work-dir <run-dir>/<set>` writes them.
- Produces:
  - `SETS`, `SWEEP_DIRS`;
  - `parse_rows(path) -> Dict[str, (auroc, ap, fpr95)]`;
  - `merge_rows(a, b, where)`;
  - `reference(rows, where)`;
  - `improvement(metrics, ref) -> float`;
  - `verdict(values, best) -> str`;
  - `compare(grouping_dir, pe_dir, a_dirs, c_dirs) -> str` (markdown);
  - CLI: `--grouping-dir`, `--pe`, `--a`, `--c`, `--out`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare_occuq_grouping.py`:

```python
"""Tests for tools/compare_occuq_grouping.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_compare_occuq_grouping.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.compare_occuq_grouping import (SWEEP_DIRS, compare,  # noqa: E402
                                          improvement, merge_rows,
                                          parse_rows, reference, verdict)

FLAT_ROWS = {'msp': (88.0, 18.0, 45.0), 'energy': (93.0, 36.0, 38.0),
             'entropy': (89.0, 27.0, 44.0)}  # R = (90.0, 27.0, 42.33)
SPLITS = {  # group_msp, group_energy, group_entropy
    'sA': ((93.0, 40.0, 30.0), (93.0, 36.0, 38.0), (93.0, 40.0, 30.0)),
    'sB': ((91.0, 30.0, 40.0), (93.0, 36.0, 38.0), (91.0, 30.0, 40.0)),
}
GROUP_KEYS = ('group_msp', 'group_energy', 'group_entropy')
PREFIX = '2026/10/07 12:00:00 - mmengine - INFO - '


def _write_log(path, rows, prefix='', extra=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f'{prefix}    method |    AUROC |       AP |   FPR@95\n')
        for name, (a, p, f) in rows.items():
            fh.write(f'{prefix}{name:>10} | {a:8.2f} | {p:8.2f} | {f:8.2f}\n')
        fh.write(extra)


def _run(run_dir, set_name, key, density, pq=0.4619):
    rows = dict(FLAT_ROWS, **{key: density})
    extra = (f'{PREFIX}Epoch(test) [10/10]    pq: {pq:.4f}  '
             'pq_dagger: 0.5000  miou: 0.4794\n')
    _write_log(os.path.join(run_dir, set_name, '20261007_120000',
                            '20261007_120000.log'), rows, PREFIX, extra)


def _layout(tmp, tsv_improvement=None):
    grouping = os.path.join(tmp, 'dump')
    ref = reference(FLAT_ROWS, 'synthetic')
    header = ('name\tsize_A\tgroup_A\timprovement_cetran\timprovement_test\t'
              'improvement_test_cetran\tworst')
    lines = [header]
    for name, scores in SPLITS.items():
        value = (sum(improvement(m, ref) for m in scores) / 3
                 if tsv_improvement is None else tsv_improvement)
        lines.append(f'{name}\t1\tx' + f'\t{value:.4f}' * 4)
    os.makedirs(os.path.join(grouping, 'divided_mass'))
    with open(os.path.join(grouping, 'divided_mass', 'robust.tsv'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    for bip, sing in SWEEP_DIRS.values():
        # sA in the bipartition log, sB in the singleton log, sA in both
        for sub, names in ((bip, ('sA', )), (sing, ('sA', 'sB'))):
            rows = dict(FLAT_ROWS)
            for name in names:
                rows.update({f'{name}_{k}': m
                             for k, m in zip(GROUP_KEYS, SPLITS[name])})
            _write_log(os.path.join(grouping, sub, 'bipartitions.log'), rows)
    pe = os.path.join(tmp, 'pe')
    a_dirs = [os.path.join(tmp, f'a{s}') for s in range(3)]
    c_dirs = [os.path.join(tmp, f'c{s}') for s in range(3)]
    for set_name in SWEEP_DIRS:
        _run(pe, set_name, 'density_pe', (92.0, 30.0, 40.0))  # I = 7.33
        for d, auroc in zip(a_dirs, (94.0, 93.5, 94.5)):  # I = 31.0 +- 0.5
            _run(d, set_name, 'density', (auroc, 40.0, 28.33))
        for d, auroc in zip(c_dirs, (93.5, 92.5, 93.0)):  # around 28.33
            _run(d, set_name, 'density', (auroc, 40.0, 30.0), pq=0.4500)
    return grouping, pe, a_dirs, c_dirs


def test_parse_rows_with_the_mmengine_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'x.log')
        rows = {'energy': (92.88, 35.55, 37.94),
                's0.1.3.5.10.16_group_msp': (94.04, 40.45, 28.52)}
        # a confusion-matrix row, as the panoptic metric logs it
        matrix = f'{PREFIX}2832\t|\t1260\t|\t3605\t|\t0\t|\t0\n'
        _write_log(path, rows, prefix=PREFIX, extra=matrix)
        assert parse_rows(path) == rows
    print('PASS test_parse_rows_with_the_mmengine_prefix')


def test_improvement_reproduces_the_spec_example():
    # s16 on test + Cetran, with the sweep's offline flat rows (spec)
    rows = {'msp': (87.76, 18.06, 45.76), 'energy': (92.88, 35.54, 37.95),
            'entropy': (89.38, 26.11, 44.91)}
    ref = reference(rows, 'test_cetran')
    values = [improvement(m, ref) for m in ((94.04, 40.45, 28.52),
                                            (92.90, 36.40, 37.97),
                                            (94.06, 40.45, 28.43))]
    for got, want in zip(values, (32.27, 17.63, 32.38)):
        assert abs(got - want) < 0.01, (got, want)
    assert abs(sum(values) / 3 - 27.42) < 0.01
    print('PASS test_improvement_reproduces_the_spec_example')


def test_verdict_rule():
    assert verdict([30.0, 31.0, 32.0], best=29.0) == 'better'
    assert verdict([30.0, 31.0, 32.0], best=33.0) == 'worse'
    assert verdict([30.0, 31.0, 32.0], best=31.0) == 'too close to call'
    assert verdict([40.0], best=31.0) == 'better'
    print('PASS test_verdict_rule')


def test_merge_rows_rejects_conflicts():
    a = {'s16_group_msp': (94.04, 40.45, 28.52)}
    assert merge_rows(a, dict(a), 'x') == a
    try:
        merge_rows(a, {'s16_group_msp': (94.04, 40.95, 28.52)}, 'x')
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting duplicate rows were accepted')
    print('PASS test_merge_rows_rejects_conflicts')


def test_compare_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        report = compare(*_layout(tmp))
    assert '| pe | worse | worse | worse |' in report
    assert '| A | better | better | better |' in report
    assert '| C | too close to call | too close to call | too close to call |' in report
    assert '`sA` (group_msp)' in report
    assert 'PQ 45.00' in report and 'base model PQ 46.19' in report
    print('PASS test_compare_end_to_end')


def test_compare_rejects_a_robust_tsv_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            compare(*_layout(tmp, tsv_improvement=99.0))
        except ValueError as err:
            assert 'robust.tsv' in str(err)
        else:
            raise AssertionError('a robust.tsv mismatch was accepted')
    print('PASS test_compare_rejects_a_robust_tsv_mismatch')


if __name__ == '__main__':
    test_parse_rows_with_the_mmengine_prefix()
    test_improvement_reproduces_the_spec_example()
    test_verdict_rule()
    test_merge_rows_rejects_conflicts()
    test_compare_end_to_end()
    test_compare_rejects_a_robust_tsv_mismatch()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_compare_occuq_grouping.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.compare_occuq_grouping'`.

- [ ] **Step 3: Write the tool**

Create `tools/compare_occuq_grouping.py`:

```python
#!/usr/bin/env python
"""Compare OCCUQ density scores with the robust grouping splits.

Reads logs only:
- grouping's 33 robust splits (divided_mass/robust.tsv), with their
  per-score rows from the bipartition-sweep logs;
- the test.py logs of the OCCUQ runs.

Every score x is reduced to a single-score improvement against the mean R
of the flat MSP, Energy and Entropy rows of its estimator:
    I(x) = (AUROC_x - R_AUROC) + (AP_x - R_AP) - (FPR@95_x - R_FPR@95)
R comes from the sweep logs for grouping, and from the base model's online
flat rows (the pe runs) for every OCCUQ variant.

A split counts with its best single Group score (like-for-like).
robust.tsv's averaged improvement is shown alongside and must be reproduced
by the rows. Verdict per variant and set, against the best split:
- better if every run is above it;
- worse if every run is below it;
- too close to call otherwise.

Spec: docs/superpowers/specs/2026-10-07-occuq-density-design.md.

Run from the repo root:
    python tools/compare_occuq_grouping.py \\
        --pe work_dirs/p3former_2xb1_3x_dso_occuq_pe \\
        --a work_dirs/p3former_2xb1_3x_dso_occuq_a/seed{0,1,2} \\
        --c work_dirs/p3former_2xb1_3x_dso_occuq_c/seed{0,1,2} \\
        --out work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md
"""
import argparse
import csv
import glob
import os
import os.path as osp
import re
import statistics
from typing import Dict, List, Sequence, Tuple

Metrics = Tuple[float, float, float]  # AUROC, AP, FPR@95 in %

SETS = ('cetran', 'test', 'test_cetran')
SET_TITLES = {'cetran': 'Cetran', 'test': 'test', 'test_cetran': 'test + Cetran'}
SWEEP_DIRS = {
    'cetran': ('bipartitions', 'singletons'),
    'test': ('bipartitions_test', 'singletons_test'),
    'test_cetran': ('bipartitions_test_cetran', 'singletons_test_cetran'),
}
FLAT = ('msp', 'energy', 'entropy')
GROUP_SCORES = ('group_msp', 'group_energy', 'group_entropy')
NUMBER = r'(-?\d+(?:\.\d+)?)'
# a name starting with a letter, so the numeric rows the panoptic metric
# also logs with '|' separators (a confusion matrix) never match
ROW = re.compile(
    rf'([A-Za-z_][\w.]*)\s*\|\s*{NUMBER}\s*\|\s*{NUMBER}\s*\|\s*{NUMBER}\s*$')
EPOCH_TEST = re.compile(r'Epoch\(test\).*?\bpq: ([\d.]+).*?\bmiou: ([\d.]+)')
CONSISTENCY_TOL = 0.02
DUPLICATE_TOL = 0.011  # the rows are printed with 2 decimals


def parse_rows(path: str) -> Dict[str, Metrics]:
    """The ``method | AUROC | AP | FPR@95`` rows of a test.py or sweep log."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            m = ROW.search(line)
            if m:
                rows[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                    float(m.group(4)))
    return rows


def merge_rows(a: Dict[str, Metrics], b: Dict[str, Metrics],
               where: str) -> Dict[str, Metrics]:
    """Union of two row dicts; a name present in both must agree."""
    out = dict(a)
    for name, metrics in b.items():
        if name in out and max(
                abs(x - y) for x, y in zip(out[name], metrics)) > DUPLICATE_TOL:
            raise ValueError(f'{name} differs between the logs of {where}: '
                             f'{out[name]} vs {metrics}')
        out[name] = metrics
    return out


def reference(rows: Dict[str, Metrics], where: str) -> Metrics:
    """R: the mean of the flat MSP, Energy and Entropy rows."""
    missing = [k for k in FLAT if k not in rows]
    if missing:
        raise KeyError(f'flat rows {missing} are missing in {where}')
    return tuple(
        sum(rows[k][i] for k in FLAT) / len(FLAT) for i in range(3))


def improvement(m: Metrics, ref: Metrics) -> float:
    """I = (AUROC - R_AUROC) + (AP - R_AP) - (FPR@95 - R_FPR@95)."""
    return (m[0] - ref[0]) + (m[1] - ref[1]) - (m[2] - ref[2])


def read_robust(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def grouping_table(robust, rows, set_name) -> List[Dict]:
    """Per robust split on ``set_name``: its best single Group score and
    like-for-like value, and robust.tsv's averaged improvement, which the
    rows must reproduce."""
    ref = reference(rows, f'the sweep logs of {set_name}')
    table = []
    for split in robust:
        name = split['name']
        scores = {}
        for key in GROUP_SCORES:
            row = f'{name}_{key}'
            if row not in rows:
                raise KeyError(f'{row} is missing from the sweep logs of '
                               f'{set_name}')
            scores[key] = rows[row]
        values = {k: improvement(m, ref) for k, m in scores.items()}
        averaged = sum(values.values()) / len(values)
        logged = float(split[f'improvement_{set_name}'])
        if abs(averaged - logged) > CONSISTENCY_TOL:
            raise ValueError(f'{name} on {set_name}: the rows give an '
                             f'averaged improvement of {averaged:.4f}, '
                             f'robust.tsv {logged:.4f}')
        best = max(values, key=values.get)
        table.append(dict(name=name, best_score=best, metrics=scores[best],
                          value=values[best], averaged=logged))
    return table


def _has_rows(path: str) -> bool:
    with open(path) as fh:
        return any(ROW.search(line) for line in fh)


def find_log(run_dir: str, set_name: str) -> str:
    """The newest test.py log with an OOD table under run_dir/set_name."""
    logs = [p for p in sorted(glob.glob(osp.join(run_dir, set_name, '*',
                                                 '*.log'))) if _has_rows(p)]
    if not logs:
        raise FileNotFoundError(f'no test.py log with an OOD table in '
                                f'{run_dir}/{set_name}/*/')
    return logs[-1]  # timestamp directories sort chronologically


def occuq_run(run_dir: str, set_name: str, key: str) -> Dict:
    """One OCCUQ evaluation: its density row, flat Energy row, flat
    reference, and PQ and mIoU in %."""
    path = find_log(run_dir, set_name)
    rows = parse_rows(path)
    if key not in rows:
        raise KeyError(f'the {key} row is missing in {path}')
    pq = miou = None
    with open(path) as fh:
        for line in fh:
            m = EPOCH_TEST.search(line)
            if m:
                pq, miou = 100 * float(m.group(1)), 100 * float(m.group(2))
    return dict(path=path, density=rows[key], energy=rows['energy'],
                reference=reference(rows, path), pq=pq, miou=miou)


def verdict(values: Sequence[float], best: float) -> str:
    """better if every value is above ``best``, worse if every value is
    below it, too close to call otherwise."""
    if all(v > best for v in values):
        return 'better'
    if all(v < best for v in values):
        return 'worse'
    return 'too close to call'


def rank(value: float, table: List[Dict]) -> int:
    """How many robust splits ``value`` beats (like-for-like)."""
    return sum(value > row['value'] for row in table)


def _fmt(m: Metrics) -> str:
    return f'{m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f}'


def _mean(runs, key) -> Metrics:
    return tuple(statistics.mean(r[key][i] for r in runs) for i in range(3))


def render_set(set_name, table, pe, variants, ref) -> Tuple[str, Dict]:
    """The markdown section of one set, and its verdict per variant."""
    best = max(table, key=lambda row: row['value'])
    n = len(table)
    lines = [
        f'## {SET_TITLES[set_name]}', '',
        "R (mean of the base model's flat MSP, Energy and Entropy): "
        f'{ref[0]:.2f} / {ref[1]:.2f} / {ref[2]:.2f}', '',
        '| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged '
        'improvement | splits beaten | verdict |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |'
    ]
    energy_i = improvement(pe['energy'], ref)
    lines.append(f"| flat Energy (base model) | {_fmt(pe['energy'])} | "
                 f'{energy_i:.2f} | | {rank(energy_i, table)}/{n} | |')
    lines.append(f"| best robust split `{best['name']}` "
                 f"({best['best_score']}) | {_fmt(best['metrics'])} | "
                 f"{best['value']:.2f} | {best['averaged']:.2f} | | |")
    verdicts = {}
    for label, runs in variants.items():
        values = [improvement(r['density'], ref) for r in runs]
        mean = statistics.mean(values)
        spread = (f' [{min(values):.2f}, {max(values):.2f}]'
                  if len(values) > 1 else '')
        verdicts[label] = verdict(values, best['value'])
        lines.append(f'| OCCUQ {label} ({len(runs)} run'
                     f'{"s" if len(runs) > 1 else ""}) | '
                     f"{_fmt(_mean(runs, 'density'))} | {mean:.2f}{spread} | "
                     f'| {rank(mean, table)}/{n} | {verdicts[label]} |')
    c_runs = variants['C']
    if all(r['pq'] is not None for r in c_runs) and pe['pq'] is not None:
        lines += [
            '', f"C after fine-tuning (mean over runs): PQ "
            f"{statistics.mean(r['pq'] for r in c_runs):.2f}, mIoU "
            f"{statistics.mean(r['miou'] for r in c_runs):.2f}, own flat "
            f"Energy {_fmt(_mean(c_runs, 'energy'))}; base model PQ "
            f"{pe['pq']:.2f}, mIoU {pe['miou']:.2f}."
        ]
    return '\n'.join(lines), verdicts


def compare(grouping_dir: str, pe_dir: str, a_dirs: Sequence[str],
            c_dirs: Sequence[str]) -> str:
    """The markdown report."""
    robust = read_robust(osp.join(grouping_dir, 'divided_mass', 'robust.tsv'))
    sections, summary = [], {}
    for set_name in SETS:
        rows = {}
        for sub in SWEEP_DIRS[set_name]:
            rows = merge_rows(
                rows, parse_rows(osp.join(grouping_dir, sub,
                                          'bipartitions.log')), set_name)
        table = grouping_table(robust, rows, set_name)
        pe = occuq_run(pe_dir, set_name, 'density_pe')
        variants = {
            'pe': [pe],
            'A': [occuq_run(d, set_name, 'density') for d in a_dirs],
            'C': [occuq_run(d, set_name, 'density') for d in c_dirs],
        }
        text, summary[set_name] = render_set(set_name, table, pe, variants,
                                             pe['reference'])
        sections.append(text)
    head = [
        '# OCCUQ vs the robust grouping splits', '',
        'Like-for-like single-score improvement I = (AUROC - R_AUROC) + '
        '(AP - R_AP) - (FPR@95 - R_FPR@95); a split counts with its best '
        'Group score. Verdict against the best robust split: better if '
        'every run is above it, worse if every run is below it, too close '
        'to call otherwise.', '',
        '| variant | ' + ' | '.join(SET_TITLES[s] for s in SETS) + ' |',
        '| --- |' + ' --- |' * len(SETS)
    ]
    for label in ('pe', 'A', 'C'):
        head.append(f'| {label} | ' +
                    ' | '.join(summary[s][label] for s in SETS) + ' |')
    tail = [
        '## Caveats', '',
        '- The robust splits were selected on these same sets, and their '
        'numbers are offline (bipartition sweep); some of their FPR@95 values '
        'are limited by float32 ties (divided_mass/summary.md).',
        "- OCCUQ's settings were fixed in advance, and its numbers come from "
        'the online metric.', '- Both points favour grouping.'
    ]
    return '\n\n'.join(['\n'.join(head)] + sections + ['\n'.join(tail)]) + '\n'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grouping-dir',
                    default='work_dirs/p3former_2xb1_3x_dso_ood_dump',
                    help='holds divided_mass/robust.tsv and the sweep logs')
    ap.add_argument('--pe', required=True,
                    help='run dir of the pe evaluations (<set>/ subdirs)')
    ap.add_argument('--a', nargs='+', required=True,
                    help='run dirs of the variant A seeds')
    ap.add_argument('--c', nargs='+', required=True,
                    help='run dirs of the variant C seeds')
    ap.add_argument('--out', required=True, help='markdown report')
    args = ap.parse_args()
    report = compare(args.grouping_dir, args.pe, args.a, args.c)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(report)
    print(report)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests/test_compare_occuq_grouping.py`
Expected: `6 passed`.

- [ ] **Step 5: Check the grouping side on the real logs**

Run:
```bash
/home/khoadv/miniconda3/envs/p3former/bin/python -c "
import sys; sys.path.insert(0, '.')
from tools.compare_occuq_grouping import *
d = 'work_dirs/p3former_2xb1_3x_dso_ood_dump'
robust = read_robust(d + '/divided_mass/robust.tsv')
for s in SETS:
    rows = {}
    for sub in SWEEP_DIRS[s]:
        rows = merge_rows(rows, parse_rows(f'{d}/{sub}/bipartitions.log'), s)
    t = grouping_table(robust, rows, s)
    b = max(t, key=lambda r: r['value'])
    print(s, len(t), b['name'], b['best_score'], round(b['value'], 2))
"
```
Expected: three lines, each with 33 splits and no exception. That means all 33 robust splits reproduce `robust.tsv` within 0.02 on every set.

- [ ] **Step 6: Commit**

```bash
git add tools/compare_occuq_grouping.py tests/test_compare_occuq_grouping.py
git commit -q -F - <<'EOF'
Add tools/compare_occuq_grouping.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

### Task 9: Storage and the signal check (variant `pe`)

**Files:** none tracked. The run outputs go under `/mnt/sandisk/khoadv/occuq/`.

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces:
  - `work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth`;
  - `work_dirs/p3former_2xb1_3x_dso_occuq_pe/{cetran,test,test_cetran}/<ts>/<ts>.log`, with `density_pe` rows;
  - the script `/mnt/sandisk/khoadv/occuq/run_variant.sh`, which Tasks 10–11 use.

- [ ] **Step 1: Create the storage and the symlinks**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD
mkdir -p /mnt/sandisk/khoadv/occuq/{pe,a,c,compare}
for v in pe a c compare; do ln -s /mnt/sandisk/khoadv/occuq/$v work_dirs/p3former_2xb1_3x_dso_occuq_$v; done
ls -la work_dirs | grep occuq
```
Expected: four symlinks into `/mnt/sandisk/khoadv/occuq/`.

- [ ] **Step 2: Check the GPUs, then run a smoke fit on 20 frames**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
CUDA_VISIBLE_DEVICES=0 /home/khoadv/miniconda3/envs/p3former/bin/python tools/fit_occuq_gmm.py configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/smoke_gmm_pe.pth --limit 20
```
Expected: `20 frames in ... s` and `saved the running sums to .../smoke_gmm_pe.sums.pth`. Then either `ValueError: classes with fewer than 2560 voxels (class, count): [...]`, because 20 frames lack the rare classes, or `saved .../smoke_gmm_pe.pth`. Both prove the loop works end to end. Then delete the smoke files: `rm -f work_dirs/p3former_2xb1_3x_dso_occuq_pe/smoke_gmm_pe.pth work_dirs/p3former_2xb1_3x_dso_occuq_pe/smoke_gmm_pe.sums.pth`.

- [ ] **Step 3: Write the run script**

```bash
cat > /mnt/sandisk/khoadv/occuq/run_variant.sh <<'EOF'
#!/usr/bin/env bash
# OCCUQ runs (docs/superpowers/plans/2026-10-07-occuq-density.md).
# Usage: run_variant.sh pe | run_variant.sh a|c SEED...
# Never pass --resume to an OCCUQ training run: with CheckpointHook(interval=9) there is no checkpoint before epoch 9, so --resume would train from random weights; rerun the seed instead.
set -euo pipefail
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD
PY=/home/khoadv/miniconda3/envs/p3former/bin/python
export PATH=/home/khoadv/miniconda3/envs/p3former/bin:$PATH
V=$1; shift
CFG=configs/p3former/p3former_2xb1_3x_dso_occuq_$V.py
OUT=work_dirs/p3former_2xb1_3x_dso_occuq_$V
fit() {  # fit CKPT FEATURES GMM_FILE
  if [ -f $3 ]; then return 0; fi
  local SUMS=${3%.pth}.sums.pth  # the pass saves its sums there before finalizing
  if [ -f $SUMS ]; then  # an earlier fit failed after its pass: finalize the saved sums
    CUDA_VISIBLE_DEVICES=0 $PY tools/fit_occuq_gmm.py $CFG $1 --features $2 --out $3 --check 5 --from-sums $SUMS
  else
    CUDA_VISIBLE_DEVICES=0 $PY tools/fit_occuq_gmm.py $CFG $1 --features $2 --out $3 --check 5
  fi
}
evaluate() {  # evaluate CKPT RUN_DIR KEY OPTION=GMM_FILE
  for S in cetran test test_cetran; do
    # a set whose run directory already holds a test.py log with the KEY row is done
    if grep -qs " $3 | " $2/$S/*/*.log; then continue; fi
    CUDA_VISIBLE_DEVICES=0 $PY test.py $CFG $1 --work-dir $2/$S \
      --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_$S.pkl model.decode_head.occuq_cfg.$4
  done
}
# Steps whose output exists are skipped, so a rerun after a failure resumes.
if [ "$V" = pe ]; then
  CKPT=work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
  fit $CKPT pe $OUT/gmm_pe.pth
  evaluate $CKPT $OUT density_pe pe_gmm_file=$OUT/gmm_pe.pth
  exit 0
fi
for N in "$@"; do
  [ -f $OUT/seed$N/epoch_9.pth ] || CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh $CFG 2 --work-dir $OUT/seed$N --cfg-options randomness.seed=$N
  fit $OUT/seed$N/epoch_9.pth head $OUT/seed$N/gmm_head.pth
  evaluate $OUT/seed$N/epoch_9.pth $OUT/seed$N density gmm_file=$OUT/seed$N/gmm_head.pth
done
EOF
chmod +x /mnt/sandisk/khoadv/occuq/run_variant.sh
```

- [ ] **Step 4: Fit the `pe` Gaussians, detached (about 15–30 min)**

```bash
nohup setsid bash -c "CUDA_VISIBLE_DEVICES=0 /home/khoadv/miniconda3/envs/p3former/bin/python tools/fit_occuq_gmm.py configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth --check 5 > /mnt/sandisk/khoadv/occuq/pe/fit.out 2>&1" < /dev/null > /dev/null 2>&1 &
```
Wait for it to finish, checking with `tail -3 /mnt/sandisk/khoadv/occuq/pe/fit.out`. Expected in `fit.out`:
- `saved the running sums to .../gmm_pe.sums.pth`, right after the pass;
- the per-class table, where every class has at least 2,560 voxels;
- the regularization lines, read with `grep "^ridge \|^pooled variance" /mnt/sandisk/khoadv/occuq/pe/fit.out`:
  - the ridge, 1e-06 × the mean pooled variance, and the shared jitter, expected 0;
  - the number of dimensions whose pooled variance is exactly 0: expect at least one, including dim 190 (the dead `pe_conv` LayerNorm channel). If there is none, stop and report it: the features differ from the ones the final review measured.
- `--check 5: max |log q float32 GPU - float64 CPU| = <x> nats` with x < 0.05;
- `saved .../gmm_pe.pth`.

If the fit fails after the pass, `run_variant.sh pe` (Step 6) finalizes the saved sums with `--from-sums` instead of redoing the pass.

- [ ] **Step 5: Run a smoke evaluation on the 5-frame mini split**

```bash
CUDA_VISIBLE_DEVICES=0 /home/khoadv/miniconda3/envs/p3former/bin/python test.py configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir /mnt/sandisk/khoadv/occuq/pe_smoke --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl model.decode_head.occuq_cfg.pe_gmm_file=work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth
grep -h "density_pe " /mnt/sandisk/khoadv/occuq/pe_smoke/*/*.log
```
Expected: a `density_pe` row with finite values. Then remove the smoke run: `rm -r /mnt/sandisk/khoadv/occuq/pe_smoke`.

A and C get no mini smoke evaluation, because their head Gaussians need the full training split. Tasks 4 and 6 test that path. `run_variant.sh` also stops at the first failing evaluation, and a rerun resumes without retraining.

- [ ] **Step 6: Evaluate on the three sets, detached (about 35 min)**

`run_variant.sh pe` skips the fit, because `gmm_pe.pth` exists.

```bash
nohup setsid bash -c "/mnt/sandisk/khoadv/occuq/run_variant.sh pe > /mnt/sandisk/khoadv/occuq/pe/run.out 2>&1" < /dev/null > /dev/null 2>&1 &
```
Expected: `run.out` shows three `test.py` runs that each end with a `Epoch(test)` metrics line.

- [ ] **Step 7: Verify the outputs**

```bash
grep -h "density_pe \|    energy \|pq: " work_dirs/p3former_2xb1_3x_dso_occuq_pe/*/*/*.log | cut -c1-200
```
Expected, per set:
- a finite `density_pe` row;
- flat Energy equal to the logged baselines: 92.36 / 35.47 / 42.55 on test and 92.88 / 35.55 / 37.94 on test + Cetran;
- PQ 0.4650 on test and 0.4619 on test + Cetran.

- [ ] **Step 8: Checkpoint with the user**

Report the `density_pe` rows next to flat Energy on the three sets, and ask whether to start the training runs (Tasks 10–11, about 21 h on both GPUs). The spec runs every variant regardless of the outcome, but this is the cheap point to stop if something looks wrong.

---

### Task 10: Variant A runs (seeds 0–2)

**Files:** none tracked; outputs go under `work_dirs/p3former_2xb1_3x_dso_occuq_a/`.

**Interfaces:**
- Consumes: `run_variant.sh` (Task 9) and the `_occuq_a` config (Task 6).
- Produces: `seed{0,1,2}/epoch_9.pth`, `seed{0,1,2}/gmm_head.pth` and `seed{0,1,2}/{cetran,test,test_cetran}/<ts>/<ts>.log`.

- [ ] **Step 1: Run a smoke training on the 5-frame mini split**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD
PATH=/home/khoadv/miniconda3/envs/p3former/bin:$PATH CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso_occuq_a.py 2 --work-dir work_dirs/p3former_2xb1_3x_dso_occuq_a/smoke --cfg-options train_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl val_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl train_cfg.max_epochs=1 train_cfg.val_interval=1 default_hooks.logger.interval=1
grep -m2 "Epoch(train)" work_dirs/p3former_2xb1_3x_dso_occuq_a/smoke/*/*.log | cut -c1-300
```
Expected:
- The train lines show `lr` ≈ 6.67e-05 (⅓ of 2e-4, the start of the warmup) and only `decode.loss_occuq_ce` and `decode.loss_occuq_lovasz`, with no `loss_mask` or `loss_dice`.
- `epoch_1.pth` is written.
- A PQ table is printed.

- [ ] **Step 2: Check that the frozen weights did not move**

```bash
/home/khoadv/miniconda3/envs/p3former/bin/python -c "
import torch
a = torch.load('work_dirs/p3former_2xb1_3x_dso/epoch_36.pth', map_location='cpu')['state_dict']
b = torch.load('work_dirs/p3former_2xb1_3x_dso_occuq_a/smoke/epoch_1.pth', map_location='cpu')['state_dict']
print('changed base tensors:', [k for k in a if not torch.equal(a[k], b[k])])
print('new modules:', sorted({k.split('.')[1] for k in b if k not in a}))
"
```
Expected: `changed base tensors: []` (this covers BatchNorm statistics and `num_batches_tracked`) and `new modules: ['occuq_head']`.

Before removing the smoke run, smoke the head-fit path on its checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /home/khoadv/miniconda3/envs/p3former/bin/python tools/fit_occuq_gmm.py configs/p3former/p3former_2xb1_3x_dso_occuq_a.py work_dirs/p3former_2xb1_3x_dso_occuq_a/smoke/epoch_1.pth --features head --out /mnt/sandisk/khoadv/occuq/a/smoke_gmm_head.pth --limit 20
```
Expected: `20 frames in ... s` and `saved the running sums to /mnt/sandisk/khoadv/occuq/a/smoke_gmm_head.sums.pth`. Then either the min-count `ValueError` or `saved /mnt/sandisk/khoadv/occuq/a/smoke_gmm_head.pth`.

Then delete the smoke outputs: `rm -r /mnt/sandisk/khoadv/occuq/a/smoke; rm -f /mnt/sandisk/khoadv/occuq/a/smoke_gmm_head.pth /mnt/sandisk/khoadv/occuq/a/smoke_gmm_head.sums.pth`.

- [ ] **Step 3: Launch the three seeds, detached (about 9 h)**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nohup setsid bash -c "/mnt/sandisk/khoadv/occuq/run_variant.sh a 0 1 2 > /mnt/sandisk/khoadv/occuq/a/run.out 2>&1" < /dev/null > /dev/null 2>&1 &
```

- [ ] **Step 4: Verify each seed when the job ends**

```bash
tail -5 /mnt/sandisk/khoadv/occuq/a/run.out
grep -h "train acc" -A 25 /mnt/sandisk/khoadv/occuq/a/run.out | head -30
grep -h "density \|    energy \|pq: " work_dirs/p3former_2xb1_3x_dso_occuq_a/seed*/*/*/*.log | cut -c1-200
```
Expected:
- **Fits:** every fit passes `--check 5`.
- **Head accuracy:** the training accuracy shows a working head: most classes above 50%, with road, building and vegetation above 90%.
- **Density rows:** every seed and set has a finite `density` row.
- **Frozen-path check:** the flat rows and PQ equal Task 9's on the same set.

---

### Task 11: Variant C runs (seeds 0–2)

**Files:** none tracked; outputs go under `work_dirs/p3former_2xb1_3x_dso_occuq_c/`.

**Interfaces:**
- Consumes: `run_variant.sh` (Task 9) and the `_occuq_c` config (Task 6).
- Produces: the same layout as Task 10, under `_occuq_c`.

- [ ] **Step 1: Run a smoke training on the mini split**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD
PATH=/home/khoadv/miniconda3/envs/p3former/bin:$PATH CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso_occuq_c.py 2 --work-dir work_dirs/p3former_2xb1_3x_dso_occuq_c/smoke --cfg-options train_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl val_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl train_cfg.max_epochs=1 train_cfg.val_interval=1 default_hooks.logger.interval=1
grep -m1 "Epoch(train)" work_dirs/p3former_2xb1_3x_dso_occuq_c/smoke/*/*.log | tr ' ' '\n' | grep "loss"
```
Expected:
- The train line holds P3Former's losses (`decode.loss_mask_*`, `decode.loss_dice_*`, `decode.loss_cls_*`, `decode.loss_ce`, `decode.loss_lovasz`) **and** `decode.loss_occuq_ce` and `decode.loss_occuq_lovasz`.
- `epoch_1.pth` is written.

Then `rm -r /mnt/sandisk/khoadv/occuq/c/smoke`.

- [ ] **Step 2: Launch the three seeds, detached (about 18 h)**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nohup setsid bash -c "/mnt/sandisk/khoadv/occuq/run_variant.sh c 0 1 2 > /mnt/sandisk/khoadv/occuq/c/run.out 2>&1" < /dev/null > /dev/null 2>&1 &
```

- [ ] **Step 3: Verify each seed when the job ends**

```bash
tail -5 /mnt/sandisk/khoadv/occuq/c/run.out
grep -h "density \|pq: " work_dirs/p3former_2xb1_3x_dso_occuq_c/seed*/*/*/*.log | cut -c1-200
```
Expected:
- every fit passes `--check 5`;
- every seed and set has a finite `density` row;
- PQ stays within about 1 point of the base model's: 46.50 on test and 46.19 on test + Cetran.

Report a larger PQ drop to the user. The comparison still runs, and the report shows PQ.

---

### Task 12: Comparison, results log and instructions

**Files:**
- Modify: `DOCs.md` (append an entry)
- Modify: `.claude/CLAUDE.md`, `.claude/rules/architecture/ood-pipeline.md`, `.claude/rules/configs.md`, `.claude/rules/commands.md`, `.claude/rules/testing.md`

**Interfaces:**
- Consumes: the outputs of Tasks 9–11, and `tools/compare_occuq_grouping.py` (Task 8).
- Produces: `work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md` and the documentation.

- [ ] **Step 1: Run the comparison**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD
/home/khoadv/miniconda3/envs/p3former/bin/python tools/compare_occuq_grouping.py \
  --pe work_dirs/p3former_2xb1_3x_dso_occuq_pe \
  --a work_dirs/p3former_2xb1_3x_dso_occuq_a/seed0 work_dirs/p3former_2xb1_3x_dso_occuq_a/seed1 work_dirs/p3former_2xb1_3x_dso_occuq_a/seed2 \
  --c work_dirs/p3former_2xb1_3x_dso_occuq_c/seed0 work_dirs/p3former_2xb1_3x_dso_occuq_c/seed1 work_dirs/p3former_2xb1_3x_dso_occuq_c/seed2 \
  --out work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md
```
Expected: the report prints, with a verdict table (pe / A / C × Cetran / test / test + Cetran) and one section per set.

- [ ] **Step 2: Append the `DOCs.md` entry**

Append, at the bottom of `DOCs.md`, an entry titled `## YYYY-MM-DD — OCCUQ density scores vs the robust grouping splits (DSO)`, with the current date. It holds:
1. One paragraph:
   - links: the spec, this plan, the branch `ood-baselines/occuq`;
   - what pe, A and C are;
   - the verdict rule (like-for-like `improvement` against the best of the 33 robust splits; better / worse / too close to call over the seeds).
2. The commands. The content of `/mnt/sandisk/khoadv/occuq/run_variant.sh` in a bash block, followed by the three invocations: `run_variant.sh pe`, `run_variant.sh a 0 1 2` and `run_variant.sh c 0 1 2`.
3. The comparison command from Step 1.
4. The verdict table and the three set tables, copied verbatim from `occuq_vs_grouping.md`.
5. Two or three sentences reading the result. Base them only on the copied tables:
   - which variant, if any, beats grouping and on which sets;
   - whether the spectrally normalized head matters (A vs pe);
   - C's PQ cost.

- [ ] **Step 3: Run the whole suite**

Run: `/home/khoadv/miniconda3/envs/p3former/bin/python -m pytest -q tests`
Expected: every test passes (39 existing plus 42 new, so 81, after the final-review amendments). Note the summary line's count and time for Step 4.

- [ ] **Step 4: Update the instructions on this branch**

In `.claude/CLAUDE.md`:
- Replace the sentence `This is branch \`ood-baselines/flat\`, the base of the OOD research branches; ...` with:

  `This is branch \`ood-baselines/occuq\`: the flat baselines of \`ood-baselines/flat\` plus OCCUQ's feature-density score (spec \`docs/superpowers/specs/2026-10-07-occuq-density-design.md\`). \`rules/branches.md\` lists the OOD branches and how changes move between them.`
- Replace the section `### Next step: OCCUQ on \`ood-baselines/occuq\`` with an `### OCCUQ (done YYYY-MM-DD)` section, using the current date. It has three bullets: what was built, the verdict table copied from the report, and the date of the `DOCs.md` entry.
- Update the date of the `## Progress log (updated ...)` heading.

Append to `.claude/rules/architecture/ood-pipeline.md`:

```markdown
## OCCUQ density scores (branch `ood-baselines/occuq`)

- `_P3FormerHead(occuq_cfg=...)` builds `_OCCUQHead` (`p3former/decode_heads/occuq_head.py`) on `pe_features`, next to `sem_queries`. The keys are `head`, `freeze_base`, `loss_weight`, `gmm_file`, `pe_gmm_file` and `score_chunk`.
- With `freeze_base` (variant A), `_P3Former` freezes everything else and keeps it in eval mode, except the data preprocessor: it builds the voxel labels only in training mode.
- `tools/fit_occuq_gmm.py` fits one Gaussian per ID class on the training split (`p3former/utils/gmm_fit.py`). Every class covariance gets the same diagonal, a ridge of 1e-6 × the mean pooled variance plus a shared jitter, because `pe_features` has a dead dimension. The pass saves its sums to `<out stem>.sums.pth`, and `--from-sums` finalizes them without a new pass.
- `predict` scores `density` and `density_pe` = asinh(−log q) with `p3former/utils/gmm_density.py`, with TF32 off only there, and stores them as `ood_density` and `ood_density_pe`. List them in `_OODPointMetric`'s `score_keys`.
- A Gaussian file carries a fingerprint of the weights that produced its feature, and `predict` refuses a file fitted for other weights.
```

Append to the variants list of `.claude/rules/configs.md`:

```markdown
- `_occuq_pe`, `_occuq_a` and `_occuq_c` (branch `ood-baselines/occuq`): OCCUQ's density score.
  - `pe` is evaluation only. `a` trains the OCCUQ head on the frozen model. `c` fine-tunes end to end.
  - Each run passes its Gaussian file with `--cfg-options model.decode_head.occuq_cfg.gmm_file=...` (`pe_gmm_file` for `pe`).
  - Their validation evaluator is PQ-only, because the Gaussians are fitted after training.
```

Append to `.claude/rules/commands.md`:

````markdown
OCCUQ (branch `ood-baselines/occuq`):

```bash
python tools/fit_occuq_gmm.py <occuq config> <checkpoint> --features head|pe --out <file> --check 5
python tools/fit_occuq_gmm.py ... --from-sums <file stem>.sums.pth   # finalize the sums of a fit that failed after its pass
python tools/compare_occuq_grouping.py --pe <dir> --a <seed dirs> --c <seed dirs> --out <report.md>
/mnt/sandisk/khoadv/occuq/run_variant.sh pe | a 0 1 2 | c 0 1 2   # fit + evaluate (+ train)
```
````

In `.claude/rules/testing.md`, replace `# 39 tests, ~5 s.` with the count and time from Step 3.

- [ ] **Step 5: Commit**

```bash
git add DOCs.md .claude/CLAUDE.md .claude/rules/architecture/ood-pipeline.md .claude/rules/configs.md .claude/rules/commands.md .claude/rules/testing.md
git commit -q -F - <<'EOF'
Log the OCCUQ comparison and update the branch instructions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011c7SVFbsayG16DKorLVYtF
EOF
```

---

## Amendments after the final code review (2026-10-07)

The final whole-branch review found a dead dimension in the base checkpoint's `pe_features`. Channel 190 of the `decode_head.pe_conv.1` LayerNorm has γ = 0.0291 and β = −0.3372, so it is always 0 after the ReLU.
- Every class covariance is then singular, and Task 2's per-class jitter search picks 2.2e-308.
- The precision factor reaches ~6.7e153, which is +inf in float32, so the float32 log-density is NaN.
- Task 7's `--check` still passed, because `max(worst, nan)` keeps `worst`.

So the shipped code differs from the code blocks of Tasks 2, 3 and 7:

1. **`finalize` (`p3former/utils/gmm_fit.py`).**
   - Every class covariance gets the same diagonal, (ridge + jitter)·I. The ridge is `ridge_rel` (default 1e-6) × the mean pooled ID variance; the jitter is one value for all classes, as in OCCUQ.
   - `cholesky_with_jitter` takes a batch and returns the first jitter for which every matrix factors.
   - Output: `jitter` [C] is the whole diagonal added, identical across classes; `ridge` (0-d) and `pooled_var` [D] are new; `covs` stays raw. Constant features raise an error.
2. **`tools/fit_occuq_gmm.py`.**
   - `--check` fails on any non-finite log-density, naming the frame, and compares with `not (diff <= 0.05)` (`compare_log_densities`).
   - Before saving, the float32 casts of `means`, `prec_chol`, `log_det_prec` and `log_prior` must be finite (`check_float32_finite`, which names the classes and the regularization).
   - The report adds the ridge, the shared jitter, the minimum pooled variance and the zero-variance dimensions.
   - Right after the pass, the raw sums go to `<out stem>.sums.pth`, and a pass refuses to overwrite them. `--from-sums` finalizes them without a pass, and refuses sums of other features or weights.
3. **Tests.**
   - `test_fit_occuq_gmm.py` covers the shared diagonal, a dead dimension, constant features and the batched jitter search.
   - `test_gmm_density.py` compares with `covs + jitter·I`.
   - The new `tests/test_fit_occuq_tool.py` covers the tool's helpers and, on a fake model, `fit`'s save-then-resume flow.
   - The suite has 81 tests.
4. **Spec.**
   - The regularization and its reason, the Output and `--check` text, and the error list.
   - A correction: about 1.4% of Stop and Others points lie in voxels whose majority label is an ID class, so those voxels do enter the sums.
5. **This plan.**
   - The Global Constraints regularization line.
   - `run_variant.sh` (Task 9 Step 3) finalizes a fit's saved sums with `--from-sums`, skips sets that already have the score's row, and warns against `--resume`.
   - Task 9 Steps 2 and 4 and Task 10 Step 2 (a new head-fit smoke run) expect and clean up the sums file; Step 4 also checks the regularization lines.
   - Task 12's test count and its rule snippets.
