# Point-level OOD Baselines (MSP / MaxLogit / ODIN / Energy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four post-hoc point-level OOD scores (MSP, MaxLogit, ODIN T=1000/ε=0, Energy T=1) to P3Former inference and evaluate them (AUROC / AP / FPR@95) on SemanticKITTI val with `other-structure` (raw 52) and `other-object` (raw 99) as OOD classes.

**Architecture:** Scores are pure functions of the head's auxiliary semantic-branch logits (`sem_preds`, per-voxel, already computed and discarded at predict time); they are projected to points via the existing `point2voxel_map` and stored as extra `pred_pts_seg` keys. A new mmengine metric derives point-level OOD ground truth from the raw panoptic labels already present in `eval_ann_info` and computes histogram-based AUROC/AP/FPR@95 in numpy. Spec: `docs/superpowers/specs/2026-08-12-ood-baselines-design.md`.

**Tech Stack:** Python 3.8, torch 1.10.1, mmengine 0.7.4, mmdet3d 1.1.0, numpy. No new dependencies (no sklearn/scipy).

## Global Constraints

- Work on branch `ood-baselines` **in this checkout** (`/home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD`). Do NOT use a separate worktree: `data/`, `checkpoint/`, and `work_dirs/` are untracked local resources the e2e run needs.
- All commands run from the repo root. Use the env python explicitly: `ENVPY=/home/khoadv/miniconda3/envs/p3former/bin/python` (conda's solver plugin prints a harmless entry-point error; ignore it, or bypass conda entirely with this path).
- Python 3.8 syntax only: `typing.Dict/List/Optional`, no `X | Y` unions, no `list[str]`.
- Score convention everywhere: **higher score = more OOD** (REL Eq. 2).
- Hyperparameters (from `docs/others/baselines/OOD_Baseline.pdf`): ODIN temperature **1000.0**, ε **= 0** (no input perturbation); Energy temperature **1.0**. MSP/MaxLogit have none.
- OOD logits = first **19** channels of the aux semantic branch (channel 19 is the never-supervised "unlabeled" slot). Taken from config (`num_ood_logits`), not hardcoded.
- GT derivation constants: `ood_raw_ids=[52, 99]`, `seg_offset=2**16`, `ignore_index=19`.
- GPU: `CUDA_VISIBLE_DEVICES=0` only — GPU 1 is running a training job. Do not touch GPU 1.
- Existing configs must be bit-for-bit unaffected: every new model/config knob defaults to "off" (`ood_cfg=None`).
- Tests follow the repo convention (see `tests/test_dso_panoptic_eval.py`): pytest-style functions + `if __name__ == '__main__':` runner with `print('PASS ...')` lines, run directly with python, `sys.path.insert` of repo root at top.
- Commit after each task, on `ood-baselines`. Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_018UW5JFvkKxZv5uKouGtwBq`.
- `DOCs.md` has uncommitted local edits that are NOT ours. Append to the file but never `git add DOCs.md` unless `git diff DOCs.md` shows only our appended section; otherwise leave it unstaged and flag it to the user at the end.

---

### Task 1: OOD score functions (`p3former/utils/ood_scores.py`)

**Files:**
- Create: `p3former/utils/ood_scores.py`
- Test: `tests/test_ood_scores.py`

**Interfaces:**
- Consumes: nothing (pure torch/numpy).
- Produces (used by Task 4 head glue and Task 5 config):
  - `OOD_SCORE_KEYS = ('msp', 'maxlogit', 'odin', 'energy')` — canonical method order.
  - `compute_ood_scores(logits: torch.Tensor[V, C], odin_temperature: float = 1000.0, energy_temperature: float = 1.0) -> Dict[str, torch.Tensor[V]]`
  - `point_ood_scores(voxel_logits: torch.Tensor[V, C], point2voxel_map: torch.Tensor[N], odin_temperature: float = 1000.0, energy_temperature: float = 1.0) -> Dict[str, np.ndarray[N] float32]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ood_scores.py`:

```python
"""Tests for p3former/utils/ood_scores.py (point-level OOD baseline scores).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_scores.py
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.utils.ood_scores import (OOD_SCORE_KEYS, compute_ood_scores,
                                       point_ood_scores)


def test_score_keys_and_shapes():
    logits = torch.randn(7, 19)
    scores = compute_ood_scores(logits)
    assert tuple(scores.keys()) == OOD_SCORE_KEYS == ('msp', 'maxlogit',
                                                      'odin', 'energy')
    for key in OOD_SCORE_KEYS:
        assert scores[key].shape == (7, ), key
    print('PASS test_score_keys_and_shapes')


def test_known_values_two_classes():
    # One voxel, two classes, logits [2, 0]. Higher score = more OOD.
    logits = torch.tensor([[2.0, 0.0]])
    scores = compute_ood_scores(logits, odin_temperature=1000.0,
                                energy_temperature=1.0)
    p_max = math.exp(2.0) / (math.exp(2.0) + 1.0)
    assert torch.allclose(scores['msp'], torch.tensor([-p_max]), atol=1e-6)
    assert torch.allclose(scores['maxlogit'], torch.tensor([-2.0]))
    p_max_T = math.exp(2.0 / 1000) / (math.exp(2.0 / 1000) + 1.0)
    assert torch.allclose(scores['odin'], torch.tensor([-p_max_T]), atol=1e-7)
    energy = -math.log(math.exp(2.0) + 1.0)
    assert torch.allclose(scores['energy'], torch.tensor([energy]), atol=1e-6)
    print('PASS test_known_values_two_classes')


def test_uniform_logits_are_most_ood():
    # Voxel 0: confident one-hot-ish logits. Voxel 1: flat logits.
    # Every method must rate the flat voxel as MORE OOD (higher score).
    confident = torch.full((1, 19), -5.0)
    confident[0, 3] = 10.0
    flat = torch.zeros(1, 19)
    scores = compute_ood_scores(torch.cat([confident, flat]))
    for key in OOD_SCORE_KEYS:
        assert scores[key][1] > scores[key][0], key
    # Exact values for the flat voxel: softmax = 1/19 each.
    assert torch.allclose(scores['msp'][1], torch.tensor(-1.0 / 19), atol=1e-6)
    assert torch.allclose(scores['maxlogit'][1], torch.tensor(0.0))
    assert torch.allclose(scores['energy'][1],
                          torch.tensor(-math.log(19.0)), atol=1e-6)
    print('PASS test_uniform_logits_are_most_ood')


def test_energy_temperature():
    logits = torch.randn(5, 19)
    scores = compute_ood_scores(logits, energy_temperature=2.0)
    expected = -2.0 * torch.logsumexp(logits / 2.0, dim=1)
    assert torch.allclose(scores['energy'], expected, atol=1e-6)
    print('PASS test_energy_temperature')


def test_odin_is_temperature_scaled_msp():
    logits = torch.randn(6, 19)
    scores = compute_ood_scores(logits, odin_temperature=1000.0)
    expected = -torch.softmax(logits / 1000.0, dim=1).max(dim=1).values
    assert torch.allclose(scores['odin'], expected, atol=1e-8)
    print('PASS test_odin_is_temperature_scaled_msp')


def test_numerical_stability_huge_logits():
    logits = torch.tensor([[1e4, -1e4, 0.0] + [0.0] * 16])
    scores = compute_ood_scores(logits)
    for key in OOD_SCORE_KEYS:
        assert torch.isfinite(scores[key]).all(), key
    print('PASS test_numerical_stability_huge_logits')


def test_point_projection():
    voxel_logits = torch.tensor([
        [10.0, 0.0],   # voxel 0: confident
        [0.0, 0.0],    # voxel 1: flat
        [0.0, 5.0],    # voxel 2: confident
    ])
    point2voxel_map = torch.tensor([0, 0, 2, 1], dtype=torch.int64)
    pts = point_ood_scores(voxel_logits, point2voxel_map)
    vox = compute_ood_scores(voxel_logits)
    for key in OOD_SCORE_KEYS:
        assert isinstance(pts[key], np.ndarray)
        assert pts[key].dtype == np.float32
        assert pts[key].shape == (4, )
        expected = vox[key][torch.tensor([0, 0, 2, 1])].numpy()
        assert np.allclose(pts[key], expected, atol=1e-6), key
    # the flat voxel's point is the most OOD point
    for key in OOD_SCORE_KEYS:
        assert pts[key].argmax() == 3, key
    # float map (as produced by dynamic_scatter_3d) must also work
    pts2 = point_ood_scores(voxel_logits, point2voxel_map.float())
    assert np.allclose(pts2['msp'], pts['msp'])
    print('PASS test_point_projection')


if __name__ == '__main__':
    test_score_keys_and_shapes()
    test_known_values_two_classes()
    test_uniform_logits_are_most_ood()
    test_energy_temperature()
    test_odin_is_temperature_scaled_msp()
    test_numerical_stability_huge_logits()
    test_point_projection()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_scores.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'p3former.utils'`

- [ ] **Step 3: Write minimal implementation**

Create `p3former/utils/ood_scores.py` (new directory `p3former/utils/` — NO `__init__.py`, the repo uses namespace packages throughout):

```python
"""Post-hoc point-level OOD scores computed from semantic logits.

All scores follow the convention: higher score = more OOD (REL, ISPRS 2026,
Eq. 2). Hyperparameters follow docs/others/baselines/OOD_Baseline.pdf:
ODIN uses temperature 1000 with epsilon = 0 (no input perturbation), the
Energy score uses temperature 1.
"""
from typing import Dict

import numpy as np
import torch

# Canonical method order; evaluation tables and pred_pts_seg keys follow it.
OOD_SCORE_KEYS = ('msp', 'maxlogit', 'odin', 'energy')


def compute_ood_scores(logits: torch.Tensor,
                       odin_temperature: float = 1000.0,
                       energy_temperature: float = 1.0
                       ) -> Dict[str, torch.Tensor]:
    """Compute all OOD scores from classification logits.

    Args:
        logits: [V, C] float tensor of per-voxel (or per-point) class logits.
        odin_temperature: softmax temperature for ODIN (T=1000, eps=0).
        energy_temperature: temperature for the free-energy score (T=1).

    Returns:
        dict with keys ``OOD_SCORE_KEYS``, each a [V] tensor,
        higher = more OOD.
    """
    if logits.dim() != 2:
        raise ValueError(f'expected [V, C] logits, got shape '
                         f'{tuple(logits.shape)}')
    logits = logits.float()
    scores = dict()
    scores['msp'] = -torch.softmax(logits, dim=1).max(dim=1).values
    scores['maxlogit'] = -logits.max(dim=1).values
    scores['odin'] = -torch.softmax(logits / odin_temperature,
                                    dim=1).max(dim=1).values
    scores['energy'] = -energy_temperature * torch.logsumexp(
        logits / energy_temperature, dim=1)
    return scores


def point_ood_scores(voxel_logits: torch.Tensor,
                     point2voxel_map: torch.Tensor,
                     odin_temperature: float = 1000.0,
                     energy_temperature: float = 1.0
                     ) -> Dict[str, np.ndarray]:
    """Compute voxel OOD scores and project them to points.

    Args:
        voxel_logits: [V, C] per-voxel class logits.
        point2voxel_map: [N] index of each point's voxel (any int/float
            dtype; cast to long, same as the panoptic projection).

    Returns:
        dict with keys ``OOD_SCORE_KEYS``, each a float32 numpy array [N].
    """
    voxel_scores = compute_ood_scores(voxel_logits, odin_temperature,
                                      energy_temperature)
    p2v = point2voxel_map.long()
    return {
        key: value[p2v].detach().cpu().numpy().astype(np.float32)
        for key, value in voxel_scores.items()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_scores.py`
Expected: 7 `PASS` lines + `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add p3former/utils/ood_scores.py tests/test_ood_scores.py
git commit -m "Add MSP/MaxLogit/ODIN/Energy OOD score functions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018UW5JFvkKxZv5uKouGtwBq"
```

---

### Task 2: Histogram-based OOD metrics (`evaluation/functional/ood_eval.py`)

**Files:**
- Create: `evaluation/functional/ood_eval.py`
- Test: `tests/test_ood_eval.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces (used by Task 3 metric):
  - `binary_ood_metrics(scores_chunks: List[np.ndarray], labels_chunks: List[np.ndarray], num_bins: int = 2**20) -> Dict[str, float]` with keys `'auroc'`, `'ap'`, `'fpr95'`. `labels_chunks` are bool arrays, True = OOD = positive.
  - `ood_point_eval(scores: Dict[str, List[np.ndarray]], labels: List[np.ndarray], logger=None) -> Dict[str, float]` with flat keys `f'{method}_AUROC'`, `f'{method}_AP'`, `f'{method}_FPR95'` (percent values, 4 decimals), logging one table row per method plus ID/OOD counts.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ood_eval.py`. The reference implementations are exact
(sort-based, tie-aware) and live in the test file only:

```python
"""Tests for evaluation/functional/ood_eval.py (AUROC / AP / FPR@95).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_eval.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.functional.ood_eval import binary_ood_metrics, ood_point_eval


def _ref_metrics(scores, labels):
    """Exact tie-aware reference: group by unique score descending."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = labels.sum()
    n_neg = (~labels).sum()
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    # boundaries of equal-score groups in descending order
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    ends = np.r_[starts[1:], len(s)]
    tp = fp = 0.0
    tps, fps = [], []
    for a, b in zip(starts, ends):
        tp += y[a:b].sum()
        fp += (~y[a:b]).sum()
        tps.append(tp)
        fps.append(fp)
    tps = np.asarray(tps)
    fps = np.asarray(fps)
    tpr = tps / n_pos
    fpr = fps / n_neg
    auroc = np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr])
    precision = tps / (tps + fps)
    ap = float(np.sum(np.diff(np.r_[0.0, tpr]) * precision))
    k = int(np.searchsorted(tpr, 0.95, side='left'))
    fpr95 = float(fpr[min(k, len(fpr) - 1)])
    return float(auroc), ap, fpr95


def test_perfect_separation():
    scores = [np.array([0.0, 1.0, 10.0, 11.0])]
    labels = [np.array([False, False, True, True])]
    m = binary_ood_metrics(scores, labels)
    assert abs(m['auroc'] - 1.0) < 1e-9, m
    assert abs(m['ap'] - 1.0) < 1e-9, m
    assert abs(m['fpr95'] - 0.0) < 1e-9, m
    print('PASS test_perfect_separation')


def test_inverted_separation():
    scores = [np.array([10.0, 11.0, 0.0, 1.0])]
    labels = [np.array([False, False, True, True])]
    m = binary_ood_metrics(scores, labels)
    assert abs(m['auroc'] - 0.0) < 1e-9, m
    assert abs(m['fpr95'] - 1.0) < 1e-9, m
    print('PASS test_inverted_separation')


def test_all_tied_scores():
    # Every point has the same score -> AUROC 0.5 by tie convention.
    scores = [np.zeros(10)]
    labels = [np.array([True] * 3 + [False] * 7)]
    m = binary_ood_metrics(scores, labels)
    assert abs(m['auroc'] - 0.5) < 1e-9, m
    assert abs(m['ap'] - 0.3) < 1e-9, m  # precision = prevalence at one cut
    assert abs(m['fpr95'] - 1.0) < 1e-9, m
    print('PASS test_all_tied_scores')


def test_matches_exact_reference_on_grid_scores():
    # Scores on a coarse grid: distinct values never share a histogram bin,
    # equal values always do -> histogram result must match the exact
    # reference to float precision.
    rng = np.random.RandomState(0)
    for trial in range(5):
        scores = rng.randint(0, 200, size=3000).astype(np.float64) / 10.0
        labels = rng.rand(3000) < 0.15
        if not labels.any() or labels.all():
            continue
        m = binary_ood_metrics([scores], [labels])
        auroc, ap, fpr95 = _ref_metrics(scores, labels)
        assert abs(m['auroc'] - auroc) < 1e-9, (trial, m['auroc'], auroc)
        assert abs(m['ap'] - ap) < 1e-9, (trial, m['ap'], ap)
        assert abs(m['fpr95'] - fpr95) < 1e-9, (trial, m['fpr95'], fpr95)
    print('PASS test_matches_exact_reference_on_grid_scores')


def test_matches_reference_on_random_floats():
    # Continuous scores: allow tiny histogram quantization error.
    rng = np.random.RandomState(1)
    scores = rng.randn(20000)
    labels = rng.rand(20000) < 0.1
    m = binary_ood_metrics([scores], [labels])
    auroc, ap, fpr95 = _ref_metrics(scores, labels)
    assert abs(m['auroc'] - auroc) < 1e-3
    assert abs(m['ap'] - ap) < 1e-3
    assert abs(m['fpr95'] - fpr95) < 1e-3
    print('PASS test_matches_reference_on_random_floats')


def test_chunked_equals_concatenated():
    rng = np.random.RandomState(2)
    scores = rng.randn(9000)
    labels = rng.rand(9000) < 0.2
    whole = binary_ood_metrics([scores], [labels])
    chunks = binary_ood_metrics(
        [scores[:100], scores[100:5000], scores[5000:]],
        [labels[:100], labels[100:5000], labels[5000:]])
    for key in ('auroc', 'ap', 'fpr95'):
        assert abs(whole[key] - chunks[key]) < 1e-12, key
    print('PASS test_chunked_equals_concatenated')


def test_rejects_degenerate_and_nan_inputs():
    try:
        binary_ood_metrics([np.array([1.0, 2.0])],
                           [np.array([False, False])])
        raise AssertionError('expected ValueError for no positives')
    except ValueError:
        pass
    try:
        binary_ood_metrics([np.array([np.nan, 2.0])],
                           [np.array([True, False])])
        raise AssertionError('expected ValueError for NaN scores')
    except ValueError:
        pass
    print('PASS test_rejects_degenerate_and_nan_inputs')


def test_ood_point_eval_table():
    rng = np.random.RandomState(3)
    labels = [rng.rand(1000) < 0.1, rng.rand(800) < 0.1]
    # 'good' separates well, 'bad' is random
    scores = {
        'good': [lab + rng.randn(len(lab)) * 0.1 for lab in labels],
        'bad': [rng.randn(len(lab)) for lab in labels],
    }
    result = ood_point_eval(scores, labels, logger=None)
    assert set(result.keys()) == {
        'good_AUROC', 'good_AP', 'good_FPR95',
        'bad_AUROC', 'bad_AP', 'bad_FPR95'}
    assert result['good_AUROC'] > 99.0          # percent scale
    assert 40.0 < result['bad_AUROC'] < 60.0
    assert result['good_FPR95'] < 5.0
    print('PASS test_ood_point_eval_table')


if __name__ == '__main__':
    test_perfect_separation()
    test_inverted_separation()
    test_all_tied_scores()
    test_matches_exact_reference_on_grid_scores()
    test_matches_reference_on_random_floats()
    test_chunked_equals_concatenated()
    test_rejects_degenerate_and_nan_inputs()
    test_ood_point_eval_table()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_eval.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.functional.ood_eval'`

- [ ] **Step 3: Write minimal implementation**

Create `evaluation/functional/ood_eval.py`:

```python
"""Point-level OOD detection metrics (AUROC, AP, FPR@95).

Histogram-based so that ~5*10^8 points fit in memory without giant sorts:
scores are bucketed into ``num_bins`` equal-width bins between the global
min and max; all three metrics are computed tie-aware from the two
(ID / OOD) histograms. Convention: higher score = more OOD; OOD is the
positive class.
"""
from typing import Dict, List

import numpy as np


def binary_ood_metrics(scores_chunks: List[np.ndarray],
                       labels_chunks: List[np.ndarray],
                       num_bins: int = 2**20) -> Dict[str, float]:
    """Compute AUROC / AP / FPR@95TPR for one score over chunked data.

    Args:
        scores_chunks: list of 1-D float arrays (higher = more OOD).
        labels_chunks: matching list of 1-D bool arrays (True = OOD).
        num_bins: histogram resolution.

    Returns:
        dict with float keys ``auroc``, ``ap``, ``fpr95`` in [0, 1].
    """
    if len(scores_chunks) != len(labels_chunks):
        raise ValueError('scores and labels chunk counts differ')
    smin, smax = np.inf, -np.inf
    for s in scores_chunks:
        if s.size == 0:
            continue
        if np.isnan(s).any():
            raise ValueError('NaN OOD scores')
        smin = min(smin, float(s.min()))
        smax = max(smax, float(s.max()))
    if not (np.isfinite(smin) and np.isfinite(smax)):
        raise ValueError('no finite OOD scores')
    if smax <= smin:
        smax = smin + 1.0  # all scores identical: one occupied bin
    scale = (num_bins - 1) / (smax - smin)

    hist_pos = np.zeros(num_bins, dtype=np.int64)
    hist_neg = np.zeros(num_bins, dtype=np.int64)
    for s, y in zip(scores_chunks, labels_chunks):
        if s.shape != y.shape:
            raise ValueError('scores/labels shape mismatch')
        if s.size == 0:
            continue
        y = y.astype(bool)
        b = ((s.astype(np.float64) - smin) * scale).astype(np.int64)
        np.clip(b, 0, num_bins - 1, out=b)
        hist_pos += np.bincount(b[y], minlength=num_bins)
        hist_neg += np.bincount(b[~y], minlength=num_bins)

    n_pos = int(hist_pos.sum())
    n_neg = int(hist_neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f'need both OOD and ID points, got {n_pos} OOD / {n_neg} ID')

    # AUROC, tie-aware: every positive scores above the negatives in
    # strictly lower bins and gets half credit against same-bin negatives.
    neg_strictly_below = np.cumsum(hist_neg) - hist_neg
    auroc = float(
        (hist_pos * (neg_strictly_below + 0.5 * hist_neg)).sum() /
        (float(n_pos) * float(n_neg)))

    # Descending-threshold cumulative counts (one cut per bin).
    hp = hist_pos[::-1].astype(np.float64)
    hn = hist_neg[::-1].astype(np.float64)
    tp = np.cumsum(hp)
    fp = np.cumsum(hn)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # AP, sklearn-style step integration with ties grouped per bin.
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    contributes = hp > 0
    ap = float((hp[contributes] / n_pos * precision[contributes]).sum())

    # FPR at the loosest threshold reaching TPR >= 0.95.
    k = int(np.searchsorted(tpr, 0.95, side='left'))
    k = min(k, num_bins - 1)
    fpr95 = float(fpr[k])

    return dict(auroc=auroc, ap=ap, fpr95=fpr95)


def ood_point_eval(scores: Dict[str, List[np.ndarray]],
                   labels: List[np.ndarray],
                   logger=None) -> Dict[str, float]:
    """Evaluate several score methods and log a result table.

    Args:
        scores: method name -> list of per-scan score arrays.
        labels: list of per-scan bool arrays (True = OOD).
        logger: optional ``logging.Logger``-like; ``print`` if None.

    Returns:
        flat dict ``{method}_AUROC`` / ``{method}_AP`` / ``{method}_FPR95``
        in percent, rounded to 4 decimals.
    """

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    n_ood = int(sum(int(lab.sum()) for lab in labels))
    n_id = int(sum(int((~lab.astype(bool)).sum()) for lab in labels))
    _log(f'Point-level OOD evaluation: {n_id} ID points, '
         f'{n_ood} OOD points ({n_ood / max(n_id + n_ood, 1):.4%} OOD)')

    results = dict()
    header = f'{"method":>10} | {"AUROC":>8} | {"AP":>8} | {"FPR@95":>8}'
    _log(header)
    _log('-' * len(header))
    for method, chunks in scores.items():
        m = binary_ood_metrics(chunks, labels)
        results[f'{method}_AUROC'] = round(100.0 * m['auroc'], 4)
        results[f'{method}_AP'] = round(100.0 * m['ap'], 4)
        results[f'{method}_FPR95'] = round(100.0 * m['fpr95'], 4)
        _log(f'{method:>10} | {results[f"{method}_AUROC"]:8.2f} | '
             f'{results[f"{method}_AP"]:8.2f} | '
             f'{results[f"{method}_FPR95"]:8.2f}')
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_eval.py`
Expected: 8 `PASS` lines + `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add evaluation/functional/ood_eval.py tests/test_ood_eval.py
git commit -m "Add histogram-based point-level OOD metrics (AUROC/AP/FPR95)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018UW5JFvkKxZv5uKouGtwBq"
```

---

### Task 3: `_OODPointMetric` (`evaluation/metrics/ood_metric.py`)

**Files:**
- Create: `evaluation/metrics/ood_metric.py`
- Test: `tests/test_ood_metric.py`

**Interfaces:**
- Consumes: `ood_point_eval` from Task 2 (exact signature above).
- Produces (used by Task 5 config): mmdet3d-registered metric
  `_OODPointMetric(ood_raw_ids=(52, 99), seg_offset=2**16, ignore_index=19, score_keys=('msp', 'maxlogit', 'odin', 'energy'), collect_device='cpu', prefix=None)`.
  Reads `data_sample['pred_pts_seg'][f'ood_{key}']` (per-point float arrays, Task 4 produces them) and `data_sample['eval_ann_info']` (`pts_instance_mask` = raw panoptic label, `pts_semantic_mask` = mapped train ids 0–19 — both already provided by the existing pipeline). Returns percent metrics `{method}_AUROC/_AP/_FPR95`, logged under prefix `ood/`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ood_metric.py`:

```python
"""Tests for evaluation/metrics/ood_metric.py (_OODPointMetric).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_metric.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.ood_metric import _OODPointMetric

# Raw semantic ids: 10 (car -> train 0), 30 (person -> train 5),
# 52 (other-structure -> OOD), 99 (other-object -> OOD),
# 0 (unlabeled -> excluded), 1 (outlier -> excluded).
RAW_SEM = np.array([10, 30, 52, 99, 0, 1, 10, 52], dtype=np.int64)
MAPPED = np.array([0, 5, 19, 19, 19, 19, 0, 19], dtype=np.int64)
INSTANCE_BITS = np.array([7, 3, 0, 0, 0, 0, 8, 0], dtype=np.int64)
RAW_PANOPTIC = (INSTANCE_BITS << 16) | RAW_SEM
# expected: OOD mask [F,F,T,T,-,-,F,T], excluded points 4 and 5


def _sample(scores):
    pred = {f'ood_{key}': np.asarray(val, dtype=np.float32)
            for key, val in scores.items()}
    return {
        'pred_pts_seg': pred,
        'eval_ann_info': {
            'pts_instance_mask': RAW_PANOPTIC.copy(),
            'pts_semantic_mask': MAPPED.copy(),
        },
    }


def test_gt_derivation_and_perfect_scores():
    # OOD points (2, 3, 7) get high scores; excluded points (4, 5) get the
    # HIGHEST scores of all -- they must be dropped, else AUROC < 1.
    good = [0.0, 0.1, 5.0, 6.0, 99.0, 98.0, 0.2, 7.0]
    bad = [-s for s in good]
    metric = _OODPointMetric(
        score_keys=('msp', 'maxlogit', 'odin', 'energy'))
    metric.process({}, [_sample({'msp': good, 'maxlogit': bad,
                                 'odin': good, 'energy': good})])
    assert len(metric.results) == 1
    labels, scores = metric.results[0]
    assert labels.tolist() == [False, False, True, True, False, True]
    assert scores['msp'].shape == (6, )
    results = metric.compute_metrics(metric.results)
    assert results['msp_AUROC'] == 100.0, results
    assert results['msp_AP'] == 100.0
    assert results['msp_FPR95'] == 0.0
    assert results['maxlogit_AUROC'] == 0.0
    assert results['odin_AUROC'] == 100.0
    assert results['energy_AUROC'] == 100.0
    print('PASS test_gt_derivation_and_perfect_scores')


def test_accumulates_across_scans():
    metric = _OODPointMetric(score_keys=('msp', ))
    s = [0.0, 0.1, 5.0, 6.0, 9.0, 9.0, 0.2, 7.0]
    metric.process({}, [_sample({'msp': s})])
    metric.process({}, [_sample({'msp': s})])
    assert len(metric.results) == 2
    results = metric.compute_metrics(metric.results)
    assert results['msp_AUROC'] == 100.0
    print('PASS test_accumulates_across_scans')


def test_missing_score_key_message():
    metric = _OODPointMetric(score_keys=('msp', ))
    try:
        metric.process({}, [{
            'pred_pts_seg': {},
            'eval_ann_info': {
                'pts_instance_mask': RAW_PANOPTIC.copy(),
                'pts_semantic_mask': MAPPED.copy(),
            },
        }])
        raise AssertionError('expected KeyError')
    except KeyError as e:
        assert 'ood_cfg' in str(e), e
    print('PASS test_missing_score_key_message')


def test_length_mismatch_rejected():
    metric = _OODPointMetric(score_keys=('msp', ))
    try:
        metric.process({}, [_sample({'msp': [0.0, 1.0]})])
        raise AssertionError('expected ValueError')
    except ValueError:
        pass
    print('PASS test_length_mismatch_rejected')


if __name__ == '__main__':
    test_gt_derivation_and_perfect_scores()
    test_accumulates_across_scans()
    test_missing_score_key_message()
    test_length_mismatch_rejected()
    print('ALL TESTS PASSED')
```

Note the expected label order in the first test: points 4 and 5 (raw 0/1)
are dropped, so the 8-point scan yields 6 kept points
`[F, F, T, T, F, T]` (indices 0, 1, 2, 3, 6, 7).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_metric.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.metrics.ood_metric'`

- [ ] **Step 3: Write minimal implementation**

Create `evaluation/metrics/ood_metric.py`:

```python
"""Point-level OOD detection metric for SemanticKITTI-style datasets.

Ground truth is derived per point from the raw panoptic label kept in
``eval_ann_info['pts_instance_mask']`` (raw semantic id = label %
``seg_offset``): raw ids in ``ood_raw_ids`` are OOD (positive), points
whose mapped train label is not ``ignore_index`` are ID (negative), and
everything else (SemanticKITTI raw 0 ``unlabeled`` / 1 ``outlier``) is
excluded. Scores are read from ``pred_pts_seg['ood_<key>']`` as emitted by
``_P3FormerHead`` when ``ood_cfg`` is set.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS

from ..functional.ood_eval import ood_point_eval


@METRICS.register_module()
class _OODPointMetric(BaseMetric):
    """Point-level OOD AUROC / AP / FPR@95 over the whole split."""

    default_prefix = 'ood'

    def __init__(self,
                 ood_raw_ids: Sequence[int] = (52, 99),
                 seg_offset: int = 2**16,
                 ignore_index: int = 19,
                 score_keys: Sequence[str] = ('msp', 'maxlogit', 'odin',
                                              'energy'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ood_raw_ids = np.asarray(list(ood_raw_ids), dtype=np.int64)
        self.seg_offset = seg_offset
        self.ignore_index = ignore_index
        self.score_keys = tuple(score_keys)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_pts_seg']
            eval_ann = data_sample['eval_ann_info']
            raw_panoptic = np.asarray(eval_ann['pts_instance_mask'])
            raw_sem = raw_panoptic % self.seg_offset
            mapped = np.asarray(eval_ann['pts_semantic_mask'])
            ood = np.isin(raw_sem, self.ood_raw_ids)
            valid = (mapped != self.ignore_index) | ood

            scores = dict()
            for key in self.score_keys:
                pred_key = f'ood_{key}'
                if pred_key not in pred:
                    raise KeyError(
                        f"'{pred_key}' missing from pred_pts_seg. Set "
                        "model.decode_head.ood_cfg in the config so "
                        "_P3FormerHead emits OOD scores.")
                value = pred[pred_key]
                if hasattr(value, 'detach'):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value)
                if value.shape[0] != mapped.shape[0]:
                    raise ValueError(
                        f'{pred_key} has {value.shape[0]} points but the '
                        f'ground truth has {mapped.shape[0]}')
                scores[key] = value[valid].astype(np.float32)
            self.results.append((ood[valid], scores))

    def compute_metrics(self, results: List[tuple]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        labels = [labels for labels, _ in results]
        scores = {
            key: [scan_scores[key] for _, scan_scores in results]
            for key in self.score_keys
        }
        return ood_point_eval(scores, labels, logger=logger)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_metric.py`
Expected: 4 `PASS` lines + `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics/ood_metric.py tests/test_ood_metric.py
git commit -m "Add _OODPointMetric for point-level OOD evaluation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018UW5JFvkKxZv5uKouGtwBq"
```

---

### Task 4: Emit OOD scores from the model (`_P3FormerHead` + `_P3Former`)

**Files:**
- Modify: `p3former/decode_heads/p3former_head.py` (constructor ~line 190; `predict` ~line 692)
- Modify: `p3former/segmentors/p3former.py` (`predict` + `postprocess_result`, lines 59–70)

**Interfaces:**
- Consumes: `point_ood_scores` from Task 1.
- Produces (used by Task 5 config, read by Task 3 metric):
  - `_P3FormerHead.__init__(..., ood_cfg=None)` — dict with keys `num_ood_logits` (int, default `num_classes - 1`), `odin_temperature` (float, default 1000.0), `energy_temperature` (float, default 1.0). `None` = today's behavior exactly.
  - `_P3FormerHead.predict` returns a 3-tuple `(pts_semantic_preds, pts_instance_preds, pts_ood_scores)` where `pts_ood_scores` is `None` when `ood_cfg is None`, else a list (one dict per sample) of `{key: float32 np.ndarray[N]}` with keys `('msp', 'maxlogit', 'odin', 'energy')`.
  - `_P3Former.postprocess_result(..., pts_ood_scores=None)` stores each score as `pred_pts_seg['ood_<key>']`.

- [ ] **Step 1: Modify `_P3FormerHead.__init__`**

In `p3former/decode_heads/p3former_head.py`, add the import at the top of
the file (after the existing imports, ~line 13):

```python
from p3former.utils.ood_scores import point_ood_scores
```

Add `ood_cfg=None` to the `__init__` signature after
`point_cloud_range=[]`:

```python
                 point_cloud_range=[],
                 ood_cfg=None):
```

And store it right after `self.num_queries = num_queries` (~line 232),
guarding the aux-branch requirement:

```python
        self.ood_cfg = ood_cfg
        if self.ood_cfg is not None:
            assert use_sem_loss, (
                'decode_head.ood_cfg requires use_sem_loss=True: OOD scores '
                'are computed from the auxiliary semantic branch (sem_queries)')
```

- [ ] **Step 2: Modify `_P3FormerHead.predict`**

Replace the whole `predict` method (currently lines 692–710) with:

```python
    def predict(self, batch_inputs, batch_data_samples):
        class_preds_buffer, mask_preds_buffer, _, sem_preds = self.forward(batch_inputs['features'], batch_inputs['voxels']['voxel_coors'])
        semantic_preds, instance_ids = self.generate_panoptic_results(class_preds_buffer[-1], mask_preds_buffer[-1])
        semantic_preds = torch.cat(semantic_preds)
        instance_ids = torch.cat(instance_ids)
        pts_semantic_preds = []
        pts_instance_preds = []
        pts_ood_scores = [] if self.ood_cfg is not None else None
        coors = batch_inputs['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            semantic_sample = semantic_preds[coors[:, 0] == batch_idx]
            instance_sample = instance_ids[coors[:, 0] == batch_idx]
            point2voxel_map = batch_data_samples[
                batch_idx].gt_pts_seg.point2voxel_map.long()
            point_semantic_sample = semantic_sample[point2voxel_map]
            point_instance_sample = instance_sample[point2voxel_map]
            pts_semantic_preds.append(point_semantic_sample.cpu().numpy())
            pts_instance_preds.append(point_instance_sample.cpu().numpy())
            if self.ood_cfg is not None:
                # sem_preds[b] rows follow the same per-sample voxel order
                # as the panoptic projection above.
                num_logits = self.ood_cfg.get('num_ood_logits',
                                              self.num_classes - 1)
                voxel_logits = sem_preds[batch_idx][:, :num_logits]
                pts_ood_scores.append(
                    point_ood_scores(
                        voxel_logits,
                        point2voxel_map,
                        odin_temperature=self.ood_cfg.get(
                            'odin_temperature', 1000.0),
                        energy_temperature=self.ood_cfg.get(
                            'energy_temperature', 1.0)))

        return pts_semantic_preds, pts_instance_preds, pts_ood_scores
```

- [ ] **Step 3: Modify `_P3Former.predict` and `postprocess_result`**

In `p3former/segmentors/p3former.py`, replace both methods (lines 59–70)
with:

```python
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

- [ ] **Step 4: Verify the modules still import and expose the new knob**

Run:

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python -c "
import inspect
import p3former.decode_heads.p3former_head as head_mod
import p3former.segmentors.p3former as seg_mod
sig = inspect.signature(head_mod._P3FormerHead.__init__)
assert 'ood_cfg' in sig.parameters and sig.parameters['ood_cfg'].default is None
sig2 = inspect.signature(seg_mod._P3Former.postprocess_result)
assert sig2.parameters['pts_ood_scores'].default is None
print('OK: ood_cfg wired, defaults off')"
```

Expected: `OK: ood_cfg wired, defaults off` (a mmcv/mmdet3d import warning
storm before it is normal). The real behavioral verification is the smoke
run in Task 5 — no isolated unit test exists for this glue by design; the
projection math it delegates to was tested in Task 1.

- [ ] **Step 5: Re-run all three test files (regression check)**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && \
/home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_scores.py && \
/home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_eval.py && \
/home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_metric.py
```

Expected: three `ALL TESTS PASSED`.

- [ ] **Step 6: Commit**

```bash
git add p3former/decode_heads/p3former_head.py p3former/segmentors/p3former.py
git commit -m "Emit point-level OOD scores from P3Former predict via ood_cfg

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018UW5JFvkKxZv5uKouGtwBq"
```

---

### Task 5: OOD config, smoke run, full val evaluation, DOCs

**Files:**
- Create: `configs/p3former/p3former_8xb2_3x_semantickitti_ood.py`
- Create (data, not committed): `data/semantickitti/semantickitti_infos_mini.pkl`
- Modify: `DOCs.md` (append a dated section; see Global Constraints about staging)

**Interfaces:**
- Consumes: `ood_cfg` head knob (Task 4), `_OODPointMetric` (Task 3).
- Produces: the runnable evaluation entry point:
  `CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_8xb2_3x_semantickitti_ood.py checkpoint/semantickitti_val_62.6.pth`

- [ ] **Step 1: Write the config**

Create `configs/p3former/p3former_8xb2_3x_semantickitti_ood.py`. The
evaluator list must restate the `_PanopticSegMetric` entry (base-config
variables like `learning_map_inv` cannot be referenced from a child
config, so its dict is copied verbatim), and `custom_imports` must restate
the full base list plus the new metric module:

```python
_base_ = ['./p3former_8xb2_3x_semantickitti.py']

# Post-hoc point-level OOD scoring (MSP / MaxLogit / ODIN / Energy) from
# the auxiliary semantic branch. Protocol and hyperparameters:
# docs/superpowers/specs/2026-08-12-ood-baselines-design.md
model = dict(
    decode_head=dict(
        ood_cfg=dict(
            num_ood_logits=19,
            odin_temperature=1000.0,
            energy_temperature=1.0)))

learning_map_inv = {  # copied from the base dataset config
    0: 10,
    1: 11,
    2: 15,
    3: 18,
    4: 20,
    5: 30,
    6: 31,
    7: 32,
    8: 40,
    9: 44,
    10: 48,
    11: 49,
    12: 50,
    13: 51,
    14: 70,
    15: 71,
    16: 72,
    17: 80,
    18: 81,
    19: 0
}

val_evaluator = [
    dict(
        type='_PanopticSegMetric',
        thing_class_inds=[0, 1, 2, 3, 4, 5, 6, 7],
        stuff_class_inds=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        min_num_points=50,
        id_offset=2**16,
        dataset_type='semantickitti',
        learning_map_inv=learning_map_inv),
    dict(
        type='_OODPointMetric',
        ood_raw_ids=[52, 99],
        seg_offset=2**16,
        ignore_index=19),
]
test_evaluator = val_evaluator

custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head',
        'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'evaluation.metrics.ood_metric',
        'datasets.semantickitti_dataset',
        'datasets.transforms.loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
```

- [ ] **Step 2: Create the 20-scan mini split for smoke testing**

The `_submit` config already expects `semantickitti_infos_mini.pkl`; it
does not exist yet. Create it from the val infos:

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && /home/khoadv/miniconda3/envs/p3former/bin/python -c "
import pickle
with open('data/semantickitti/semantickitti_infos_val.pkl', 'rb') as f:
    infos = pickle.load(f)
print('keys:', sorted(infos.keys()))
assert 'data_list' in infos, infos.keys()
infos['data_list'] = infos['data_list'][:20]
with open('data/semantickitti/semantickitti_infos_mini.pkl', 'wb') as f:
    pickle.dump(infos, f)
print('wrote', len(infos['data_list']), 'scans')"
```

Expected: `keys: ['data_list', 'metainfo']` (or similar including
`data_list`) then `wrote 20 scans`. If `data_list` is missing, STOP and
inspect the pkl structure before proceeding — do not guess.

- [ ] **Step 3: Smoke run on the mini split (GPU 0, ~1 min)**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && CUDA_VISIBLE_DEVICES=0 /home/khoadv/miniconda3/envs/p3former/bin/python test.py \
  configs/p3former/p3former_8xb2_3x_semantickitti_ood.py \
  checkpoint/semantickitti_val_62.6.pth \
  --cfg-options test_dataloader.dataset.dataset.ann_file=semantickitti_infos_mini.pkl \
  2>&1 | tail -40
```

Expected in the output:
- the usual PQ table (values on 20 scans are arbitrary — it just must not crash), and
- the OOD table: four rows (`msp`, `maxlogit`, `odin`, `energy`) with finite AUROC/AP/FPR@95 values and a logged ID/OOD point count line with OOD fraction roughly in the 0.1–2% range.

If the OOD table shows AUROC exactly 50.0 for every method, or the metric
raises "need both OOD and ID points", debug before the full run (likely
the mini scans contain no 52/99 points — pick a different slice, e.g.
`[1000:1020]`, and repeat).

- [ ] **Step 4: Full val evaluation (GPU 0, ~7 min)**

```bash
cd /home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD && CUDA_VISIBLE_DEVICES=0 /home/khoadv/miniconda3/envs/p3former/bin/python test.py \
  configs/p3former/p3former_8xb2_3x_semantickitti_ood.py \
  checkpoint/semantickitti_val_62.6.pth \
  2>&1 | tail -60
```

Expected:
- PQ table **identical** to the 2026-08-05 baseline run: PQ 62.63, RQ 72.42, SQ 76.17, mIoU 66.77 (proves the panoptic path is untouched);
- OOD table over 4071 scans with plausible values — REL Table 3 reports MaxLogit at AUROC 84.9 / FPR@95 47.6 / AP 53.9 on a Mask4Former backbone, so same order of magnitude, NOT the same numbers;
- all four methods distinct and finite. MSP and ODIN will be correlated (ODIN is temperature-scaled MSP with ε=0) but not identical.

If PQ differs from 62.63 in any digit: STOP, the panoptic path was
perturbed — bisect Task 4's edits before proceeding.

- [ ] **Step 5: Log results in DOCs.md**

Append a section to `DOCs.md` (adjust the numbers to the actual run):

````markdown
## 2026-08-12 — Point-level OOD baselines (MSP / MaxLogit / ODIN / Energy)

Post-hoc OOD scoring from the aux semantic branch; OOD classes = raw 52
(other-structure) + 99 (other-object); raw 0/1 excluded. Spec:
`docs/superpowers/specs/2026-08-12-ood-baselines-design.md`.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_8xb2_3x_semantickitti_ood.py checkpoint/semantickitti_val_62.6.pth
```

Results (val, 4071 scans, official checkpoint; PQ table unchanged at 62.63):

| method   | AUROC | AP | FPR@95 |
| -------- | ----- | -- | ------ |
| MSP      | XX.XX | XX.XX | XX.XX |
| MaxLogit | XX.XX | XX.XX | XX.XX |
| ODIN     | XX.XX | XX.XX | XX.XX |
| Energy   | XX.XX | XX.XX | XX.XX |

ODIN = temperature-scaled MSP (T=1000, ε=0 per docs/others/baselines/OOD_Baseline.pdf);
Energy uses T=1. Smoke test: add
`--cfg-options test_dataloader.dataset.dataset.ann_file=semantickitti_infos_mini.pkl`.
````

Replace `XX.XX` with the real numbers from Step 4's output before saving.

- [ ] **Step 6: Commit**

```bash
git add configs/p3former/p3former_8xb2_3x_semantickitti_ood.py
git diff DOCs.md   # inspect: only our appended section? then also: git add DOCs.md
git commit -m "Add OOD evaluation config for SemanticKITTI val

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018UW5JFvkKxZv5uKouGtwBq"
```

If `git diff DOCs.md` shows pre-existing edits that are not ours, commit
only the config and report the DOCs.md situation to the user.

---

## Verification checklist (whole feature)

- [ ] `tests/test_ood_scores.py`, `tests/test_ood_eval.py`, `tests/test_ood_metric.py` all print `ALL TESTS PASSED`.
- [ ] Full val run prints the unchanged PQ table (62.63) AND the four-row OOD table.
- [ ] `git log --oneline` on `ood-baselines` shows one commit per task.
- [ ] DOCs.md carries the commands + results table.
- [ ] The plain config `p3former_8xb2_3x_semantickitti.py` still works (no `ood_cfg`, dict evaluator): quick check is that Task 4's defaults keep `pts_ood_scores=None` and `postprocess_result` only adds keys when it is not None.
