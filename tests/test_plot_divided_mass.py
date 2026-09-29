"""Tests for tools/plot_divided_mass.py (smoke tests on synthetic tables).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_divided_mass.py
"""
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

SCRIPT = os.path.join(_REPO_ROOT, 'tools', 'plot_divided_mass.py')


def _fake_divided_dir(tmp):
    """The tables of tools/divided_mass.py that the figures read, with one
    split dividing no ID point at delta = 0.05."""
    rng = np.random.RandomState(0)
    subsets = sb.singleton_subsets() + [(0, 1), (2, 5, 16)]
    for key in dm.SETS:
        rows = []
        for s in subsets:
            row = OrderedDict([
                ('name', sb.partition_name(s)), ('size_A', len(s)),
                ('group_A', ', '.join(sb.CLASSES[c] for c in s))])
            for d in dm.DELTAS:
                k = dm.delta_key(d)
                row[f'ood_div@{k}'] = float(10**rng.uniform(-1, 1.5))
                row[f'id_div@{k}'] = float(10**rng.uniform(-2, 1))
            robust = s in [(16, ), (2, 5, 16)]
            row['improvement'] = float(rng.uniform(1, 35) if robust
                                       else rng.uniform(-40, 35))
            row['robust'] = int(robust)
            rows.append(row)
        rows[3]['id_div@0.05'] = 0.0  # {truck}: drawn on the axis floor
        dm.write_tsv(os.path.join(tmp, f'{key}.tsv'), rows)
    dm.write_tsv(os.path.join(tmp, 'flat.tsv'), [
        OrderedDict([('set', spec['label']), ('id_unc@0.05', 3.0),
                     ('ood_unc@0.05', 40.0)]) for spec in dm.SETS.values()])
    dm.write_tsv(os.path.join(tmp, 'rho.tsv'), [
        OrderedDict([('set', key), ('population', pop),
                     ('delta', dm.delta_key(d)), ('statistic', s),
                     ('target', t), ('rho', float(rng.uniform(-1, 1))),
                     ('n', 24)])
        for key in dm.SETS for pop in ('single', 'all') for d in dm.DELTAS
        for t in dm.TARGETS for s in dm.STATISTICS])


def test_figures_are_written():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        proc = subprocess.run([sys.executable, SCRIPT, tmp, '--top', '2'],
                              capture_output=True, text=True, cwd=_REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
        for name in ('bubble_singletons_0.05', 'bubble_robust_0.05',
                     'rho_vs_threshold'):
            for ext in ('.pdf', '.png'):
                path = os.path.join(tmp, name + ext)
                assert os.path.getsize(path) > 1000, path
        proc = subprocess.run([sys.executable, SCRIPT, tmp, '--threshold',
                               '0.07'], capture_output=True, text=True,
                              cwd=_REPO_ROOT)
        assert proc.returncode == 2 and '--threshold' in proc.stderr
    print('test_figures_are_written passed')


def test_labels_do_not_overlap_when_there_is_room():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    import plot_divided_mass as pdm
    fig, ax = plt.subplots(figsize=(6, 5))
    xy = [(1, 1), (3, 30), (10, 3), (30, 10), (1, 30), (30, 1)]
    points = [dict(x=x, y=y, value=5 + 3 * i, text=f'class {i}')
              for i, (x, y) in enumerate(xy)]
    items = pdm.bubble_panel(ax, points, (0.3, 100), (0.3, 100))
    pdm.place_labels(ax, items)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [pdm._box(Text.get_window_extent(t, renderer)) for t in ax.texts]
    assert len(boxes) == len(points)
    clashes = sum(pdm._overlap(a, b) > 0
                  for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert clashes == 0, clashes
    plt.close(fig)
    print('test_labels_do_not_overlap_when_there_is_room passed')


if __name__ == '__main__':
    test_figures_are_written()
    test_labels_do_not_overlap_when_there_is_room()
    print('ALL TESTS PASSED')
