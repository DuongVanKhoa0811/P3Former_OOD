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


def test_dense_singletons_fit_and_refine_does_not_worsen_overlap():
    """Regression test for the review finding on bubble_singletons_0.05.png:
    labels colliding in dense clusters, a label sitting on the panel title,
    and a label clipped at the figure edge. Reproduces the densest real
    panel (the 24 single-class splits of the 'test' set, the one where the
    review found the title collision and the edge clipping) at the actual
    bubble_grid figure size, and checks every label ends up fully inside the
    figure and clear of the title, and that refinement (the default) never
    leaves more total label-to-label overlap than the greedy pass alone."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    import plot_divided_mass as pdm

    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        rows = dm.read_tsv(os.path.join(tmp, 'test.tsv'))
        flat = dm.read_tsv(os.path.join(tmp, 'flat.tsv'))
    key = dm.delta_key(dm.HEADLINE)
    points = [dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                    value=r['improvement'], text=r['group_A'])
              for r in rows if r['size_A'] == 1]
    star = next((row[f'id_unc@{key}'], row[f'ood_unc@{key}']) for row in flat
                if row['set'] == dm.SETS['test']['label'])
    xlim = pdm.log_limits([p['x'] for p in points] + [star[0]])
    ylim = pdm.log_limits([p['y'] for p in points] + [star[1]])

    def render(refine):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6))
        ax = axes[0, 0]
        items = pdm.bubble_panel(ax, points, xlim, ylim, star=star)
        ax.set_title('(a) Test', loc='left')
        kwargs = dict(star=star) if refine is None else dict(star=star,
                                                              refine=refine)
        pdm.place_labels(ax, items, **kwargs)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [pdm._box(Text.get_window_extent(t, renderer))
                 for t in ax.texts]
        fig_box = pdm._box(fig.bbox)
        title_box = pdm._box(Text.get_window_extent(ax.title, renderer))
        plt.close(fig)
        return boxes, fig_box, title_box

    def pairwise_overlap(boxes):
        return sum(pdm._overlap(a, b)
                   for i, a in enumerate(boxes) for b in boxes[i + 1:])

    boxes, fig_box, title_box = render(refine=None)  # the real default
    assert len(boxes) == len(points)
    for box in boxes:
        assert pdm._outside(box, fig_box) <= 1e-6, ('label outside the '
                                                     f'figure: {box}')
        assert pdm._overlap(box, title_box) == 0.0, ('label on the title: '
                                                      f'{box}')
    refined_total = pairwise_overlap(boxes)

    greedy_boxes, _, _ = render(refine=False)
    greedy_total = pairwise_overlap(greedy_boxes)
    assert refined_total <= greedy_total, (refined_total, greedy_total)
    print(f'label-overlap area: greedy-only {greedy_total:.1f} px^2, '
          f'refined {refined_total:.1f} px^2')
    print('test_dense_singletons_fit_and_refine_does_not_worsen_overlap '
          'passed')


def test_labels_are_drawn_above_every_bubble():
    """Regression test for F1: bubble_panel draws each bubble at zorder
    3 + its rank among the panel's points (largest first), which used to
    reach and then pass the labels' own fixed zorder (10) from the 8th
    bubble on, covering them (bubble_singletons_0.05.png's 'building',
    'vegetation' and 'perimeter-barrier'). Every label -- and, separately,
    its leader line where place_labels pushes it away from a crowded
    bubble -- must now sit above every bubble, however many a panel has.

    Reuses the dense 'test' singleton panel (real panel size, real title;
    see test_dense_singletons_fit_and_refine_does_not_worsen_overlap
    below) because it reliably produces leader lines (confirmed: 10 of
    its 24 labels get one), unlike a sparser layout where every label
    fits at zero gap and there is no arrow_patch to check at all. This
    matters because matplotlib 3.5's ax.annotate(..., zorder=...) reaches
    the Text but not the separate FancyArrowPatch arrowprops creates for
    the leader line, which used to keep the Patch default zorder (1),
    under every bubble -- confirmed to fail before the fix (arrow zorder
    1, text zorder 1000, both under the code's own eyes) by temporarily
    reverting _annotate's arrowprops to drop its ``zorder`` key."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import plot_divided_mass as pdm

    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        rows = dm.read_tsv(os.path.join(tmp, 'test.tsv'))
        flat = dm.read_tsv(os.path.join(tmp, 'flat.tsv'))
    key = dm.delta_key(dm.HEADLINE)
    points = [dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                    value=r['improvement'], text=r['group_A'])
              for r in rows if r['size_A'] == 1]
    star = next((row[f'id_unc@{key}'], row[f'ood_unc@{key}']) for row in flat
                if row['set'] == dm.SETS['test']['label'])
    xlim = pdm.log_limits([p['x'] for p in points] + [star[0]])
    ylim = pdm.log_limits([p['y'] for p in points] + [star[1]])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6))
    ax = axes[0, 0]
    items = pdm.bubble_panel(ax, points, xlim, ylim, star=star)
    ax.set_title('(a) Test', loc='left')
    pdm.place_labels(ax, items, star=star)

    bubble_zorders = [c.get_zorder() for c in ax.collections]
    text_zorders = [t.get_zorder() for t in ax.texts]
    leader_zorders = [t.arrow_patch.get_zorder() for t in ax.texts
                      if t.arrow_patch is not None]
    assert len(text_zorders) == len(points)
    assert leader_zorders, ('fixture produced no leader lines to check -- '
                            'not a meaningful test of F1')
    assert min(text_zorders) > max(bubble_zorders), (text_zorders,
                                                      bubble_zorders)
    assert min(leader_zorders) > max(bubble_zorders), (leader_zorders,
                                                        bubble_zorders)
    plt.close(fig)
    print(f'{len(leader_zorders)} leader lines checked, all above every '
          'bubble')
    print('test_labels_are_drawn_above_every_bubble passed')


def test_robust_panels_labels_by_rank():
    """F2 regression: the robust chart's foreground points are labelled by
    rank number (1..top, by worst-set improvement), not by the raw split
    name -- those pile up over the bubbles of a dense cluster."""
    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        tables = OrderedDict(
            (key, dm.read_tsv(os.path.join(tmp, f'{key}.tsv')))
            for key in dm.SETS)
    import plot_divided_mass as pdm
    labels = OrderedDict((k, s['label']) for k, s in dm.SETS.items())
    panels, _, key = pdm.robust_panels(tables, labels, dm.HEADLINE, 2)
    # _fake_divided_dir's two robust splits, best worst-case first
    expected_names = {sb.partition_name((16,)), sb.partition_name((2, 5, 16))}
    assert [row[0] for row in key] == [1, 2], key
    assert {row[1] for row in key} == expected_names, key
    for pts in panels.values():
        texts = sorted(p['text'] for p in pts if p['text'])
        assert texts == ['1', '2'], texts
    print('test_robust_panels_labels_by_rank passed')


def test_robust_legend_lists_every_top_name():
    """F2 regression: the legend cell keeps every top-N split's name and
    classes (the spec calls them 'labelled by name'), even though the
    panel itself now only shows their rank."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    import plot_divided_mass as pdm

    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        tables = OrderedDict(
            (key, dm.read_tsv(os.path.join(tmp, f'{key}.tsv')))
            for key in dm.SETS)
        flat = dm.read_tsv(os.path.join(tmp, 'flat.tsv'))
    labels = OrderedDict((k, s['label']) for k, s in dm.SETS.items())
    stars = pdm.flat_stars(flat, dm.HEADLINE)
    _, _, key_rows = pdm.robust_panels(tables, labels, dm.HEADLINE, 2)

    captured = {}
    orig_save = pdm.save
    pdm.save = lambda fig, stem: captured.setdefault('fig', fig)
    try:
        pdm.bubble_robust(tables, labels, dm.HEADLINE, 2, 'x', 'y', 'stem',
                          stars)
    finally:
        pdm.save = orig_save
    legend_ax = captured['fig'].axes[3]  # axes[1, 1], the legend cell
    texts = [t.get_text() for t in legend_ax.texts]
    for child in legend_ax.get_children():
        if isinstance(child, Legend):
            texts += [t.get_text() for t in child.get_texts()]
    blob = '\n'.join(texts)
    assert key_rows, key_rows
    for _, name, _ in key_rows:
        assert name in blob, (name, blob)
    plt.close(captured['fig'])
    print('test_robust_legend_lists_every_top_name passed')


def test_size_handle_span_contains_the_widest_marker():
    """Important 2 regression: the robust chart's shrunk size key used a
    fixed handlelength/handleheight (2.0, at fontsize 5.5: an 11 pt row)
    that did not scale with SIZE_KEY's markers, whose own diameter
    (sqrt(area(value)), independent of the legend's fontsize) reaches
    21.9 pt for the largest -- so with ncol=1 (one marker per row) the
    '40' circle overlapped the '20' row above it and clipped its numeral
    (bubble_robust_0.05.png's legend cell; confirmed by direct render --
    see the task report for the crop). _size_handle_span's whole point is
    to keep the handle box at least as large as that marker, in points, at
    whatever fontsize legend_cell shrinks the size key to; this checks the
    invariant directly rather than through matplotlib's own legend layout
    (Line2D.get_window_extent's *width* for a marker-only handle reflects
    its handle-box slot, not the rendered marker, so a geometric pairwise-
    overlap check on it is not a meaningful test here -- confirmed by
    rendering this exact scenario and inspecting the PNG: no overlap)."""
    import math

    import plot_divided_mass as pdm

    fontsize = 5.5
    span_pt = pdm._size_handle_span(fontsize) * fontsize
    widest_marker_pt = math.sqrt(pdm.area(max(pdm.SIZE_KEY)))
    assert span_pt > widest_marker_pt, (span_pt, widest_marker_pt)
    # the previous, broken parameters (ncol=1, handlelength=handleheight=
    # 2.0) would have failed this same check, confirming it is meaningful
    broken_span_pt = 2.0 * fontsize
    assert broken_span_pt < widest_marker_pt, (broken_span_pt,
                                               widest_marker_pt)
    print('test_size_handle_span_contains_the_widest_marker passed')


def test_top_above_max_is_rejected():
    """Important 3 / Ruling 20 regression: --top above MAX_TOP used to run
    the robust key over the colour and size keys (33 robust splits at
    --top 500 on the real tables). main() now refuses it at the argparse
    level, naming the cap and robust.tsv; exactly MAX_TOP is still
    accepted."""
    import plot_divided_mass as pdm
    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        proc = subprocess.run(
            [sys.executable, SCRIPT, tmp, '--top', str(pdm.MAX_TOP + 1)],
            capture_output=True, text=True, cwd=_REPO_ROOT)
        assert proc.returncode == 2, proc.stderr
        assert '--top' in proc.stderr and str(pdm.MAX_TOP) in proc.stderr, (
            proc.stderr)
        assert 'robust.tsv' in proc.stderr, proc.stderr
        proc = subprocess.run(
            [sys.executable, SCRIPT, tmp, '--top', str(pdm.MAX_TOP)],
            capture_output=True, text=True, cwd=_REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
    print('test_top_above_max_is_rejected passed')


def test_robust_panels_handles_top_beyond_count_and_no_robust_splits():
    """Important 3 (Ruling 20): robust_panels must not error when --top
    exceeds the number of robust splits that exist (labels all of them)
    or when there are none at all (labels nothing, an empty key) -- both
    routine on the real tables at a large --top, or on a set with no
    robust split."""
    import plot_divided_mass as pdm
    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        tables = OrderedDict(
            (key, dm.read_tsv(os.path.join(tmp, f'{key}.tsv')))
            for key in dm.SETS)
    labels = OrderedDict((k, s['label']) for k, s in dm.SETS.items())

    # top (1000) far exceeds the fixture's 2 robust splits: both named.
    panels, _, key = pdm.robust_panels(tables, labels, dm.HEADLINE, 1000)
    assert [row[0] for row in key] == [1, 2], key
    for pts in panels.values():
        texts = sorted(p['text'] for p in pts if p['text'])
        assert texts == ['1', '2'], texts

    # no robust splits anywhere: no error, nothing named, every split is
    # a background dot.
    no_robust = OrderedDict((k, [dict(r, robust=0) for r in rows])
                            for k, rows in tables.items())
    panels, backgrounds, key = pdm.robust_panels(no_robust, labels,
                                                 dm.HEADLINE, 8)
    assert key == [], key
    assert all(pts == [] for pts in panels.values()), panels
    assert all(len(pts) == len(rows) for pts, rows in
              zip(backgrounds.values(), no_robust.values()))
    print('test_robust_panels_handles_top_beyond_count_and_no_robust_splits '
          'passed')


def test_foreground_limits_ignores_far_background_points():
    """F3 regression: bubble_robust's axis limits must come from the
    robust bubbles and the flat-MSP star alone, not the background splits
    -- one of which can sit many decades below them, as in the review
    finding (bubble_robust_0.05.png's Cetran panel, squeezed into the top
    by a single background point at 6e-5 %)."""
    import plot_divided_mass as pdm

    panels = OrderedDict([('Cetran', [
        dict(x=1.0, y=10.0, value=15.0, text='1'),
        dict(x=3.0, y=40.0, value=20.0, text='2'),
    ])])
    stars = {'Cetran': (15.0, 80.0)}
    background_y = 6e-5  # the review finding's stray background point
    xlim, ylim = pdm.foreground_limits(panels, stars)
    assert ylim[0] > 100 * background_y, ylim
    # contrast: folding the same background point into the limits (what
    # bubble_grid's own default computation does, and what bubble_robust
    # used to rely on) drags the lower limit far below it
    naive = pdm.log_limits([p['y'] for pts in panels.values() for p in pts]
                           + [v for _, v in stars.values()] + [background_y])
    assert naive[0] < ylim[0], (naive, ylim)
    print('test_foreground_limits_ignores_far_background_points passed')


def test_bubble_grid_requires_both_limits_or_neither():
    """Minor regression: bubble_grid's ``if xlim is None or ylim is
    None:`` used to recompute *both* from the defaults whenever only one
    was missing, silently discarding the other override; passing exactly
    one is now a clear error rather than a mixed, meaningless limit."""
    import plot_divided_mass as pdm

    panels = OrderedDict([('Cetran', [dict(x=1.0, y=2.0, value=5.0,
                                           text='a')])])
    for kwargs in (dict(xlim=(1, 10)), dict(ylim=(1, 10))):
        try:
            pdm.bubble_grid(panels, 'x', 'y', 'stem', **kwargs)
        except ValueError:
            continue
        assert False, f'expected ValueError for {kwargs}'
    print('test_bubble_grid_requires_both_limits_or_neither passed')


def test_bubble_grid_draws_notes_only_in_their_own_panel():
    """F5 regression: plot_ood_class_resemblance.py's feature_bubbles used
    to keep its own copy of bubble_grid's 2 x 2 layout only to draw a
    per-panel note; bubble_grid now takes the note itself, so there is one
    layout, not two that can drift apart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import plot_divided_mass as pdm

    panels = OrderedDict([
        ('Cetran', [dict(x=1.0, y=2.0, value=5.0, text='car')]),
        ('Test', [dict(x=1.0, y=2.0, value=-5.0, text='truck')]),
        ('Test + Cetran', [dict(x=1.0, y=2.0, value=5.0, text='bus')]),
    ])
    note = 'not measured (no ID samples): 3 classes'
    captured = {}
    orig_save = pdm.save
    pdm.save = lambda fig, stem: captured.setdefault('fig', fig)
    try:
        pdm.bubble_grid(panels, 'x', 'y', 'stem', notes={'Cetran': note})
    finally:
        pdm.save = orig_save
    axes = captured['fig'].axes
    matches = [[t for t in ax.texts if t.get_text() == note]
              for ax in axes[:3]]
    assert len(matches[0]) == 1, matches  # Cetran is panel (a), axes[0, 0]
    assert matches[1] == [] and matches[2] == [], matches
    plt.close(captured['fig'])
    print('test_bubble_grid_draws_notes_only_in_their_own_panel passed')


if __name__ == '__main__':
    test_figures_are_written()
    test_labels_do_not_overlap_when_there_is_room()
    test_dense_singletons_fit_and_refine_does_not_worsen_overlap()
    test_labels_are_drawn_above_every_bubble()
    test_robust_panels_labels_by_rank()
    test_robust_legend_lists_every_top_name()
    test_size_handle_span_contains_the_widest_marker()
    test_top_above_max_is_rejected()
    test_robust_panels_handles_top_beyond_count_and_no_robust_splits()
    test_foreground_limits_ignores_far_background_points()
    test_bubble_grid_requires_both_limits_or_neither()
    test_bubble_grid_draws_notes_only_in_their_own_panel()
    print('ALL TESTS PASSED')
