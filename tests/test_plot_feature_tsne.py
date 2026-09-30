"""Tests for tools/plot_feature_tsne.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_feature_tsne.py
"""
import os
import re
import sys
import tempfile
from collections import OrderedDict

import matplotlib.pyplot as plt
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


def test_class_names_stay_inside_the_figure():
    # Long names at the far left and far right of each of three panels (the
    # real layout): centred, they would run past the figure edge.
    rng = np.random.RandomState(0)
    n = 40
    left = sb.CLASSES.index('perimeter-barrier')
    right = sb.CLASSES.index('overhead-bridge')
    xy = np.vstack([np.column_stack([-50 + rng.randn(n), rng.randn(n)]),
                    np.column_stack([50 + rng.randn(n), rng.randn(n)])])
    label = np.repeat([left, right], n)
    ood = np.zeros(2 * n, bool)
    raw = np.ones(2 * n, int)
    panels = OrderedDict((name, (xy, label, raw, ood))
                         for name in ('Cetran', 'Test', 'Test + Cetran'))
    captured = {}
    save = tsne.pdm.save
    tsne.pdm.save = lambda fig, stem: captured.setdefault('fig', fig)
    try:
        with plt.rc_context(tsne.pdm.STYLE):
            tsne.tsne_figure(panels, 'unused')
            fig = captured['fig']
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            names = [t for ax in fig.axes for t in ax.texts]
            assert len(names) == 6
            for t in names:
                box = t.get_window_extent(renderer)
                assert (box.x0 >= fig.bbox.x0 - 1
                        and box.x1 <= fig.bbox.x1 + 1), (t.get_text(), box)
    finally:
        tsne.pdm.save = save
        plt.close('all')
    print('test_class_names_stay_inside_the_figure passed')


def test_label_position_picks_the_main_cluster():
    rng = np.random.RandomState(0)
    # Class 0, first block: 20 points at (10, 10) + 0.1*randn (contaminated, listed FIRST)
    cluster_0_contaminated = (10, 10) + 0.1 * rng.randn(20, 2)
    # Class 0, second block: 20 points at (0, 0) + 0.1*randn (clean main cluster)
    cluster_0_main = 0.1 * rng.randn(20, 2)
    class_0 = np.vstack([cluster_0_contaminated, cluster_0_main])
    # Class 1: 60 points at (10, 10) + 0.5*randn (surrounding the contaminated cluster)
    class_1 = (10, 10) + 0.5 * rng.randn(60, 2)
    xy = np.vstack([class_0, class_1])
    members = np.arange(40)  # indices of class 0 (20 + 20)
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
    test_class_names_stay_inside_the_figure()
    test_label_position_picks_the_main_cluster()
    print('ALL TESTS PASSED')
