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


def test_class_names_stay_inside_the_figure():
    import matplotlib.patches
    import plot_divided_mass as pdm  # noqa: E402

    rng = np.random.RandomState(0)
    # One class at far left, one at far right with a long name
    xy_left = np.hstack([
        -50 + 0.1 * rng.randn(50, 1),
        rng.randn(50, 1)
    ])
    xy_right = np.hstack([
        50 + 0.1 * rng.randn(50, 1),
        rng.randn(50, 1)
    ])
    xy = np.vstack([xy_left, xy_right])
    label = np.concatenate([np.zeros(50), np.ones(50)])
    raw = np.ones(100)
    ood = np.zeros(100, dtype=bool)

    panels = OrderedDict([('Panel', (xy, label, raw, ood))])

    # Monkeypatch pdm.save to capture the figure
    saved_figs = []
    original_save = pdm.save
    def mock_save(fig, stem):
        saved_figs.append(fig)
    pdm.save = mock_save

    try:
        tsne.tsne_figure(panels, 'dummy_stem')
        assert len(saved_figs) == 1, 'pdm.save was not called'
        fig = saved_figs[0]
        fig.canvas.draw()

        # Check that all text extents are within figure bounds
        fig_bbox = fig.bbox
        for text in fig.texts:
            extent = text.get_window_extent(renderer=fig.canvas.get_renderer())
            assert extent.xmin >= fig_bbox.xmin - 1, \
                f'Text "{text.get_text()}" extends left of figure (xmin={extent.xmin}, fig={fig_bbox.xmin})'
            assert extent.xmax <= fig_bbox.xmax + 1, \
                f'Text "{text.get_text()}" extends right of figure (xmax={extent.xmax}, fig={fig_bbox.xmax})'
    finally:
        pdm.save = original_save

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
