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
