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


if __name__ == '__main__':
    test_class_colours()
    test_tsne_figure_smoke()
    print('ALL TESTS PASSED')
