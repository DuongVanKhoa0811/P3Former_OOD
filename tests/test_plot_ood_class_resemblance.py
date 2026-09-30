"""Tests for tools/plot_ood_class_resemblance.py (synthetic tables).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_ood_class_resemblance.py
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
import ood_class_resemblance as res  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402


def test_figures_are_written():
    rng = np.random.RandomState(0)
    subsets = sb.singleton_subsets() + [(12, 16), (0, 1, 2, 3, 4)]
    with tempfile.TemporaryDirectory() as tmp:
        rho = []
        for key in res.SETS:
            for space in res.SPACES:
                shares = 100.0 * rng.dirichlet(np.ones(sb.NUM_CLASSES))
                dm.write_tsv(os.path.join(tmp, f'profile_{key}_{space}.tsv'), [
                    OrderedDict([
                        ('class', name), ('bank', 40), ('queries', 20),
                        # class 5 ('person') is absent from the bank on
                        # every set/space, same as a real excluded class:
                        # NaN r_ood/r_id, a normal, finite improvement.
                        ('r_ood', float('nan') if c == 5
                         else float(shares[c])),
                        ('r_id', float('nan') if c == 5
                         else float(rng.uniform(0, 8))),
                        ('contrast', float(rng.uniform(0, 5))),
                        ('feat_div_ood', float(rng.uniform(0.1, 60))),
                        ('feat_div_id', 0.0 if c == 3 else float(rng.uniform(0.1, 10))),
                        ('ood_div', float(rng.uniform(0, 50))),
                        ('id_div', float(rng.uniform(0, 5))),
                        ('improvement', float(rng.uniform(-40, 35)))])
                    for c, name in enumerate(sb.CLASSES)])
                dm.write_tsv(os.path.join(tmp, f'splits_{key}_{space}.tsv'), [
                    OrderedDict([
                        ('name', sb.partition_name(s)), ('size_A', len(s)),
                        ('group_A', 'x'), ('R_A', float(rng.uniform(0, 60))),
                        ('E_A', float(rng.uniform(0, 5))),
                        ('feat_div_ood', float(rng.uniform(0, 60))),
                        ('feat_div_id', float(rng.uniform(0, 10))),
                        ('feat_log_ratio', float(rng.uniform(-1, 2))),
                        ('ood_div', 1.0), ('id_div', 1.0),
                        ('improvement', float(rng.uniform(-40, 35))),
                        ('robust', int(s == (16, )))]) for s in subsets])
                for population, pairs in (('classes', res.CLASS_PAIRS),
                                          ('splits', res.SPLIT_PAIRS)):
                    rho += [OrderedDict([
                        ('set', key), ('space', space), ('k', 10),
                        ('population', population), ('x', x), ('y', y),
                        ('rho', float(rng.uniform(-1, 1))), ('n', 24)])
                        for x, y in pairs]
        dm.write_tsv(os.path.join(tmp, 'rho.tsv'), rho)
        script = os.path.join(_REPO_ROOT, 'tools',
                              'plot_ood_class_resemblance.py')
        for space in res.SPACES:
            proc = subprocess.run([sys.executable, script, tmp, '--space',
                                   space], capture_output=True, text=True,
                                  cwd=_REPO_ROOT)
            assert proc.returncode == 0, proc.stderr
            for name in ('profile', 'bubble_feature_singletons', 'hypothesis'):
                for ext in ('.pdf', '.png'):
                    path = os.path.join(tmp, f'{name}_{space}{ext}')
                    assert os.path.getsize(path) > 1000, path
    print('test_figures_are_written passed')


def test_missing_res_dir_names_the_producer():
    """S3: plot_ood_class_resemblance.load on a directory with none of
    tools/ood_class_resemblance.py's output must name that tool, not raise
    a bare FileNotFoundError from deep inside read_tsv's open()."""
    import plot_ood_class_resemblance as pcr
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pcr.load(tmp, 'full')
        except FileNotFoundError as err:
            assert 'run tools/ood_class_resemblance.py' in str(err), err
        else:
            raise AssertionError('missing resemblance directory accepted')
    print('test_missing_res_dir_names_the_producer passed')


def test_label_x_clears_the_bar_and_the_marker():
    import matplotlib
    matplotlib.use('Agg')
    import plot_ood_class_resemblance as pcr

    # no marker, no r_ID bar (its default): the old behaviour, bar end +
    # margin
    assert pcr.label_x(3.0, None, 0.5) == 3.5
    # marker beyond the bar: text clears the marker, not just the bar
    x = pcr.label_x(3.0, 7.0, 0.5)
    assert x == 7.5 and x > 3.0 and x > 7.0
    # bar beyond the marker: text clears the bar, not just the marker
    x = pcr.label_x(7.0, 3.0, 0.5)
    assert x == 7.5 and x > 3.0 and x > 7.0
    # C3: r_ID bar beyond the r_OOD bar and any marker: text clears the
    # r_ID bar too, not just the r_OOD bar and the marker
    x = pcr.label_x(3.0, None, 0.5, id_bar_end=7.0)
    assert x == 7.5 and x > 3.0 and x > 7.0
    x = pcr.label_x(3.0, 4.0, 0.5, id_bar_end=7.0)
    assert x == 7.5 and x > 3.0 and x > 4.0 and x > 7.0
    # r_ID bar within the other two: no effect, same as the default 0.0
    x = pcr.label_x(7.0, None, 0.5, id_bar_end=3.0)
    assert x == 7.5
    print('test_label_x_clears_the_bar_and_the_marker passed')


def test_feature_panels_skips_unmeasured_classes():
    import matplotlib
    matplotlib.use('Agg')
    import plot_ood_class_resemblance as pcr

    labels = OrderedDict([('cetran', 'Cetran')])
    profiles = OrderedDict([
        ('cetran', [
            {'class': 'car', 'feat_div_id': 1.0, 'feat_div_ood': 2.0,
             'improvement': 5.0, 'r_ood': 3.0},
            # no ID samples in the bank: left out, not a measured 0 %
            {'class': 'bicycle', 'feat_div_id': 4.0, 'feat_div_ood': 6.0,
             'improvement': -2.0, 'r_ood': float('nan')},
        ]),
    ])
    panels, skipped = pcr.feature_panels(labels, profiles)
    classes = [p['text'] for p in panels['Cetran']]
    assert classes == ['car'], classes
    assert 'bicycle' not in classes
    assert skipped['Cetran'] == 1, skipped
    print('test_feature_panels_skips_unmeasured_classes passed')


def test_profile_figure_labels_have_a_halo_and_x_is_unclipped():
    """Ruling 23: every value label / 'not in the bank' caption in
    profile_figure carries a white withStroke halo (so the dashed
    no-preference line cannot turn a minus into a plus), and the
    absent-class x markers are drawn with clip_on=False (so the one at
    x = 0 is not half swallowed by the axis)."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.patheffects import withStroke
    import plot_divided_mass as pdm
    import plot_ood_class_resemblance as pcr

    labels = OrderedDict([('cetran', 'Cetran'), ('test', 'Test'),
                          ('test_cetran', 'Test + Cetran')])
    rows = [
        {'class': 'car', 'r_ood': 20.0, 'r_id': 5.0, 'improvement': -28.6},
        # absent from the bank: NaN r_ood / r_id, drawn as an x + caption.
        {'class': 'bicycle', 'r_ood': float('nan'), 'r_id': float('nan'),
         'improvement': 6.6},
    ]
    profiles = OrderedDict((key, rows) for key in labels)

    captured = {}
    real_save = pdm.save
    pdm.save = lambda fig, stem: captured.setdefault('fig', fig)
    try:
        pcr.profile_figure(labels, profiles, OrderedDict(), 'unused-stem')
    finally:
        pdm.save = real_save

    fig = captured['fig']
    assert len(fig.axes) == 3, 'expected one panel per set'
    for ax in fig.axes:
        assert ax.texts, 'no value/caption text drawn'
        for t in ax.texts:
            effects = t.get_path_effects() or []
            assert any(isinstance(e, withStroke) for e in effects), (
                f'{t.get_text()!r} has no white halo')
        # the one PathCollection per panel is the absent-class x markers
        # (the appearance-space marker scatter is skipped: no appearance
        # data was passed in).
        assert ax.collections, 'no x marker drawn for the absent class'
        assert all(c.get_clip_on() is False for c in ax.collections), (
            'the absent-class x marker is clipped to the axes')
    print('test_profile_figure_labels_have_a_halo_and_x_is_unclipped passed')


def test_profile_figure_keeps_text_inside_its_panel():
    """S1/S2 regression: text_max (the shared right x-limit's budget) used
    to track only each label's *start* x -- via label_x -- never its own
    rendered width, so nothing stopped a text from running past the axis
    it is drawn in. A profile shaped like a real one (shares near xmax,
    summing to ~100 %) turns out to leave enough headroom regardless
    (checked directly against the unfixed code: see the task report), and
    so does this test's small/uniform r_ood/r_id with several absent
    classes alone -- an absent row's caption starts near x = 0, which
    gets nearly the whole panel's width as headroom either way. The one
    path that margin does not cover is an appearance-space marker
    (label_x's other input): xmax is computed from r_ood/r_id alone, never
    from the appearance profile, so a marker placed well past xmax can
    push a value label's *start* close to the unfixed x_upper with almost
    no room left for the label's own text. This fixture keeps every
    ingredient of the brief's recipe (small uniform r_ood/r_id, three
    absent classes, one long improvement value driving the longest
    string, the 'not in the bank' caption) and adds that marker so the
    combination actually clips on the unfixed code (confirmed: about
    18 px of overflow before the fix, 0 after, at profile_figure's real
    figsize and panel count -- see probe_s1.py in the task report)."""
    import matplotlib
    matplotlib.use('Agg')
    import plot_divided_mass as pdm
    import plot_ood_class_resemblance as pcr

    labels = OrderedDict([('cetran', 'Cetran'), ('test', 'Test'),
                          ('test_cetran', 'Test + Cetran')])
    classes = ['car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
              'rider', 'traffic-sign']
    rows = []
    for i, c in enumerate(classes):
        absent = i in (1, 4, 6)
        rows.append({
            'class': c,
            'r_ood': float('nan') if absent else 4.0 + 0.1 * i,
            'r_id': float('nan') if absent else 3.0 + 0.1 * i,
            'improvement': -123.4 if i == 4 else float(2.0 * i - 5)})
    profiles = OrderedDict((key, rows) for key in labels)
    # xmax only looks at r_ood/r_id, so an appearance-space marker far
    # past it (here, 'car' at 45 against a full-space xmax of ~4.7) is
    # invisible to x_upper's own 30 % margin.
    appearance = OrderedDict(
        (key, [{'class': c, 'r_ood': 45.0 if c == 'car' else float('nan')}
              for c in classes]) for key in labels)

    captured = {}
    real_save = pdm.save
    pdm.save = lambda fig, stem: captured.setdefault('fig', fig)
    try:
        pcr.profile_figure(labels, profiles, appearance, 'unused-stem')
    finally:
        pdm.save = real_save

    fig = captured['fig']
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(fig.axes) == 3, 'expected one panel per set'
    checked = 0
    for ax in fig.axes:
        ax_box = ax.get_window_extent(renderer)
        for t in ax.texts:
            box = t.get_window_extent(renderer)
            assert box.x0 >= ax_box.x0 - 1 and box.x1 <= ax_box.x1 + 1, (
                f'{t.get_text()!r} sits outside its panel: text x '
                f'[{box.x0:.1f}, {box.x1:.1f}], axis x '
                f'[{ax_box.x0:.1f}, {ax_box.x1:.1f}]')
            checked += 1
    assert checked, 'fixture drew no text to check'
    print('test_profile_figure_keeps_text_inside_its_panel passed')


if __name__ == '__main__':
    test_figures_are_written()
    test_missing_res_dir_names_the_producer()
    test_label_x_clears_the_bar_and_the_marker()
    test_feature_panels_skips_unmeasured_classes()
    test_profile_figure_labels_have_a_halo_and_x_is_unclipped()
    test_profile_figure_keeps_text_inside_its_panel()
    print('ALL TESTS PASSED')
