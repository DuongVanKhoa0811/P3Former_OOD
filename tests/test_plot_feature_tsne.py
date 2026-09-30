"""Tests for tools/plot_feature_tsne.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_feature_tsne.py
"""
import os
import re
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import plot_feature_tsne as tsne  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402
from make_dso_hierarchy_variants import GROUPS  # noqa: E402


def test_class_colours():
    assert list(tsne.CLASS_COLOURS) == list(sb.CLASSES)
    colours = [c.lower() for c in tsne.CLASS_COLOURS.values()]
    assert len(set(colours)) == sb.NUM_CLASSES
    assert all(re.fullmatch(r'#[0-9a-f]{6}', c) for c in colours)
    assert tsne.OOD_COLOUR.lower() not in colours
    covered = sorted(c for _, ids in GROUPS.values() for c in ids)
    assert covered == list(range(sb.NUM_CLASSES))  # every class has a family
    print('test_class_colours passed')


def test_tsne_figure_smoke():
    rng = np.random.RandomState(0)
    n = 240
    label = np.repeat([0, 12, 16, 24], n // 4)
    ood = label == 24
    raw = np.where(ood, np.where(np.arange(n) % 2 == 0, 17, 28), 1)
    feat = rng.randn(n, 8) + np.where(ood[:, None], 2.0, label[:, None] / 8.0)
    samples = dict(label=label, ood=ood, raw=raw, weight=np.ones(n))
    idx = tsne.tsne_sample(samples, 40, 50, np.random.default_rng(0))
    assert len(idx) == 3 * 40 + 50 and len(set(idx.tolist())) == len(idx)
    xy = tsne.embed(feat[idx].astype(np.float32), seed=0)
    assert xy.shape == (len(idx), 2) and np.all(np.isfinite(xy))
    panels = OrderedDict((name, (xy, label[idx], raw[idx], ood[idx]))
                         for name in ('Cetran', 'Test', 'Test + Cetran'))
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, 'tsne_full')
        tsne.tsne_figure(panels, stem)
        for ext in ('.pdf', '.png'):
            assert os.path.getsize(stem + ext) > 1000
    print('test_tsne_figure_smoke passed')


def test_label_position_picks_the_main_cluster():
    rng = np.random.RandomState(0)
    # Class 0: dense cluster near (0, 0) with 30 points, sparse cluster near (10, 10) with 10 points
    cluster_0_main = 0.1 * rng.randn(30, 2)
    cluster_0_sparse = (10, 10) + 0.1 * rng.randn(10, 2)
    class_0 = np.vstack([cluster_0_main, cluster_0_sparse])
    # Class 1: 30 points near (5, -5)
    class_1 = (5, -5) + 0.1 * rng.randn(30, 2)
    xy = np.vstack([class_0, class_1])
    members = np.arange(40)  # indices of class 0 (30 + 10)
    label = tsne.label_position(xy, members)
    # Assert label lies within 1.0 of (0, 0), where the main cluster is
    assert np.linalg.norm(label) < 1.0, f'label {label} is too far from (0, 0)'
    # Assert label is one of class 0's points
    assert np.any(np.all(np.abs(class_0 - label) < 0.15, axis=1)), \
        f'label {label} is not in class 0'
    print('test_label_position_picks_the_main_cluster passed')


if __name__ == '__main__':
    test_class_colours()
    test_tsne_figure_smoke()
    test_label_position_picks_the_main_cluster()
    print('ALL TESTS PASSED')
