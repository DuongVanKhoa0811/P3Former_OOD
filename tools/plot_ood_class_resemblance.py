#!/usr/bin/env python
"""Figures of tools/ood_class_resemblance.py.

Reads the tables of tools/ood_class_resemblance.py (default
work_dirs/p3former_2xb1_3x_dso_ood_dump/resemblance) and writes, next to
them, as PDF and PNG in the style of tools/plot_divided_mass.py:

    profile_<space>                    per set, the 24 classes (one order,
                                       by r_OOD on Test + Cetran): bars of
                                       r_OOD, coloured by the sign of
                                       {c} | rest's improvement, and of
                                       r_ID (grey); the appearance-space
                                       r_OOD as hollow circles (full space
                                       only); no preference (100/24 %)
                                       dashed; a class with no ID samples
                                       in the set's kNN bank (NaN r_OOD /
                                       r_ID) as a grey x at 0 with a 'not
                                       in the bank' label, never as a
                                       0 % bar
    bubble_feature_singletons_<space>  ID (x) against OOD (y)
                                       feature-divided % of the single-class
                                       splits, area = |improvement|: the
                                       feature-space twin of
                                       bubble_singletons; a class with no
                                       ID samples in the set's kNN bank is
                                       left out (its feature-divided
                                       shares are 0 by construction, not
                                       measurement), noted in-panel when
                                       any are
    hypothesis_<space>                 improvement against R_A ("together"),
                                       the OOD feature-divided % ("cut
                                       through") and the log OOD/ID
                                       feature-divided ratio ("cut through
                                       (ratio)") over every split, robust
                                       splits highlighted, Spearman rho in
                                       the titles

Run from the repo root:
    python tools/plot_ood_class_resemblance.py [RESEMBLANCE_DIR] [--space full]
"""
import argparse
import os.path as osp
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import divided_mass as dm  # noqa: E402
import ood_class_resemblance as res  # noqa: E402
import plot_divided_mass as pdm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402


def load(res_dir, space):
    read = lambda name: dm.read_tsv(osp.join(res_dir, name))  # noqa: E731
    labels = OrderedDict((k, s['label']) for k, s in res.SETS.items())
    profiles = OrderedDict((k, read(f'profile_{k}_{space}.tsv'))
                           for k in res.SETS)
    appearance = OrderedDict()
    if space == 'full':
        appearance = OrderedDict(
            (k, read(f'profile_{k}_appearance.tsv')) for k in res.SETS
            if osp.exists(osp.join(res_dir, f'profile_{k}_appearance.tsv')))
    splits = OrderedDict((k, read(f'splits_{k}_{space}.tsv'))
                         for k in res.SETS)
    return labels, profiles, appearance, splits, read('rho.tsv')


def label_x(bar_end, marker_x, margin):
    """x position for a row's text label (an improvement value, or the
    'not in the bank' caption): to the right of both ``bar_end`` (the
    row's full-space bar end, 0 for a class with no bar) and ``marker_x``
    (the appearance-space marker, or None when the panel has none for this
    class), plus ``margin`` -- so the text never sits on the marker."""
    base = bar_end if marker_x is None else max(bar_end, marker_x)
    return base + margin


def profile_figure(labels, profiles, appearance, stem):
    """One panel per set, classes in one order (by r_OOD on the last set).
    A class with no ID samples in the set's kNN bank has NaN r_ood / r_id
    (tools/ood_class_resemblance.py's 'excluded' classes): its row gets no
    bar and no appearance marker -- only a grey x at 0 and a grey 'not in
    the bank' caption, so it is never mistaken for an actual 0 % share."""
    last = list(profiles)[-1]
    order = [r['class'] for r in sorted(
        profiles[last], key=lambda r: np.nan_to_num(r['r_ood'], nan=-1.0))]
    y = np.arange(len(order))
    row_y = dict(zip(order, y))
    xmax = max(max(np.nan_to_num(r['r_ood']), np.nan_to_num(r['r_id']))
               for rows in profiles.values() for r in rows)
    margin = 0.015 * xmax

    # One pass to gather every row's bar / marker / text data and the
    # widest text position, before any drawing: the shared x limit (below)
    # depends on it, and every panel must use the same one.
    panel_rows, text_max, any_absent = OrderedDict(), 0.0, False
    for key, rows in profiles.items():
        by = {r['class']: r for r in rows}
        app = ({r['class']: float(r['r_ood']) for r in appearance[key]
               if not np.isnan(r['r_ood'])} if key in appearance else {})
        info = []
        for c in order:
            nan = bool(np.isnan(by[c]['r_ood']))
            any_absent = any_absent or nan
            marker_x = None if nan else app.get(c)
            bar_end = 0.0 if nan else by[c]['r_ood']
            tx = label_x(bar_end, marker_x, margin)
            text_max = max(text_max, tx)
            info.append(dict(c=c, nan=nan, r_ood=by[c]['r_ood'],
                             r_id=by[c]['r_id'], marker_x=marker_x,
                             improvement=by[c]['improvement'], text_x=tx))
        panel_rows[key] = info
    x_upper = max(xmax * 1.3, text_max + margin)

    fig, axes = plt.subplots(1, len(profiles), figsize=(7.0, 4.8),
                             sharey=True, squeeze=False)
    for ax, key in zip(axes[0], panel_rows):
        present = [r for r in panel_rows[key] if not r['nan']]
        absent = [r for r in panel_rows[key] if r['nan']]
        edge = [pdm.GAIN if r['improvement'] > 0 else pdm.DROP
               for r in present]
        ax.barh([row_y[r['c']] + 0.2 for r in present],
                [r['r_ood'] for r in present], height=0.4,
                color=[pdm.tint(e) for e in edge], edgecolor=edge,
                linewidth=0.6)
        ax.barh([row_y[r['c']] - 0.2 for r in present],
                [r['r_id'] for r in present], height=0.4, color='0.85',
                edgecolor='0.55', linewidth=0.6)
        marked = [r for r in present if r['marker_x'] is not None]
        if marked:
            ax.scatter([r['marker_x'] for r in marked],
                       [row_y[r['c']] + 0.2 for r in marked], s=10,
                       facecolor='none', edgecolor='black', linewidths=0.6,
                       zorder=4)
        for r in present:
            ax.text(r['text_x'], row_y[r['c']] + 0.2,
                    pdm.signed(r['improvement']), va='center', fontsize=5.5,
                    color=pdm.GAIN_TXT if r['improvement'] > 0
                    else pdm.DROP_TXT)
        if absent:
            ax.scatter([0.0] * len(absent),
                       [row_y[r['c']] for r in absent], marker='x', s=18,
                       color=pdm.NA, linewidths=0.8, zorder=4)
            for r in absent:
                ax.text(r['text_x'], row_y[r['c']],
                        f"not in the bank ({pdm.signed(r['improvement'])})",
                        va='center', fontsize=5.5, color=pdm.NA)
        ax.axvline(100.0 / sb.NUM_CLASSES, color='0.4', ls=(0, (3, 2)),
                   lw=0.6)
        ax.set_xlim(0, x_upper)
        ax.set_title(labels[key], loc='left')
        ax.set_xlabel('share of the neighbours (%)')
        ax.grid(axis='x', color='0.9', lw=0.4)
        ax.set_axisbelow(True)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(order, fontsize=6.5)
    handles = [
        Line2D([], [], ls='none', marker='s', markersize=6,
               markerfacecolor=pdm.tint(pdm.GAIN), markeredgecolor=pdm.GAIN),
        Line2D([], [], ls='none', marker='s', markersize=6,
               markerfacecolor=pdm.tint(pdm.DROP), markeredgecolor=pdm.DROP),
        Line2D([], [], ls='none', marker='s', markersize=6,
               markerfacecolor='0.85', markeredgecolor='0.55'),
        Line2D([], [], ls=(0, (3, 2)), lw=0.6, color='0.4')]
    names = [r'$r_\mathrm{OOD}$, {c} | rest improves (value: improvement)',
             r'$r_\mathrm{OOD}$, {c} | rest drops',
             r'$r_\mathrm{ID}$: ID points of the other classes',
             'no preference (100/24 %)']
    if appearance:
        handles.append(Line2D([], [], ls='none', marker='o', markersize=3.5,
                              markerfacecolor='none', markeredgecolor='black'))
        names.append(r'$r_\mathrm{OOD}$, appearance space')
    if any_absent:
        handles.append(Line2D([], [], ls='none', marker='x', markersize=5,
                              markeredgecolor=pdm.NA, markeredgewidth=0.8))
        names.append("class absent from the set's ID points")
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.2,
                        wspace=0.08)
    pdm.save(fig, stem)


def feature_panels(labels, profiles):
    """Bubble-chart points of every set: ID (x) vs OOD (y) feature-divided
    % of the single-class splits, skipping a class with no ID samples in
    that set's kNN bank (NaN r_ood in the profile). Its feature-divided
    shares are 0 by construction -- an empty bank has no neighbour of
    that class for any query to divide against -- not by measurement, so
    plotting it would misread as an actual, measured 0 %. Returns
    (panels: label -> list of point dicts, skipped: label -> number of
    classes left out)."""
    panels, skipped = OrderedDict(), OrderedDict()
    for key, rows in profiles.items():
        label = labels[key]
        measured = [r for r in rows if not np.isnan(r['r_ood'])]
        panels[label] = [dict(x=r['feat_div_id'], y=r['feat_div_ood'],
                              value=r['improvement'], text=r['class'])
                         for r in measured]
        skipped[label] = len(rows) - len(measured)
    return panels, skipped


def feature_bubbles(labels, profiles, stem):
    """The feature-space twin of plot_divided_mass.py's bubble_singletons
    (see feature_panels for the skip rule). This reimplements
    pdm.bubble_grid's 2 x 2 layout from its own public pieces instead of
    calling it directly: a panel that skipped any class also needs a
    small note drawn inside its own axes, and bubble_grid saves and
    closes its figure internally, with no hook and no returned axes to
    add one afterwards."""
    panels, skipped = feature_panels(labels, profiles)
    xs = [p['x'] for pts in panels.values() for p in pts]
    ys = [p['y'] for pts in panels.values() for p in pts]
    xlim, ylim = pdm.log_limits(xs), pdm.log_limits(ys)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.07,
                        wspace=0.24, hspace=0.32)
    cells = (axes[0, 0], axes[0, 1], axes[1, 0])
    labelled, zeros = [], False
    for i, (ax, (label, points)) in enumerate(zip(cells, panels.items())):
        labelled.append((ax, pdm.bubble_panel(ax, points, xlim, ylim)))
        zeros = zeros or any(p['x'] <= 0 or p['y'] <= 0 for p in points)
        ax.set_title(f'({"abc"[i]}) {label}', loc='left')
        ax.set_xlabel('ID feature-divided (%)')
        if i != 1:
            ax.set_ylabel('OOD feature-divided (%)')
        if skipped[label]:
            ax.text(0.97, 0.03, 'not measured (no ID samples): '
                    + f'{skipped[label]} classes', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=6, color='0.5')
    for ax, items in labelled:
        pdm.place_labels(ax, items)
    pdm.legend_cell(axes[1, 1], zeros=zeros)
    pdm.save(fig, stem)


def hypothesis_figure(labels, splits, rho, space, stem):
    # The 3-column titles include the longest set label ('Test + Cetran')
    # and the longest column name ('cut through (ratio)'): at the shared
    # STYLE title size (8.5 pt) that string alone renders about 2.05 in
    # wide (measured), wider than a 7.0-in-wide, 3-column panel can fit
    # without starving the gaps every other title needs; title_fontsize
    # (scoped to this figure only, not pdm.STYLE) is the smallest change
    # that keeps every title on one line and clear of its neighbours.
    title_fontsize = 7.0
    fig, axes = plt.subplots(len(splits), 3, figsize=(7.0, 8.4),
                             squeeze=False)
    for i, (key, rows) in enumerate(splits.items()):
        sel = {r['x']: r['rho'] for r in rho if r['set'] == key
               and r['space'] == space and r['population'] == 'splits'
               and r['y'] == 'improvement'}
        other = [r for r in rows if not r['robust']]
        robust = [r for r in rows if r['robust']]
        for j, (x, name, xlabel) in enumerate((
                ('R_A', 'together', r'$R_A$: OOD resemblance on the smaller '
                 'side (%)'),
                ('feat_div_ood', 'cut through', 'OOD feature-divided (%)'),
                ('feat_log_ratio', 'cut through (ratio)',
                 'log10 OOD/ID feature-divided ratio'))):
            ax = axes[i, j]
            ax.scatter([r[x] for r in other], [r['improvement'] for r in other],
                       s=5, color='0.72', linewidths=0)
            ax.scatter([r[x] for r in robust],
                       [r['improvement'] for r in robust], s=12,
                       color=pdm.GAIN, linewidths=0)
            ax.axhline(0.0, color='0.4', lw=0.6)
            ax.set_title(f'{labels[key]}, {name}: ' + r'$\rho$ = '
                         + dm.fmt(sel.get(x, float('nan'))), loc='left',
                        fontsize=title_fontsize)
            ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel('improvement')
            ax.grid(color='0.92', lw=0.4)
            ax.set_axisbelow(True)
    handles = [Line2D([], [], ls='none', marker='o', markersize=3,
                      color='0.72'),
               Line2D([], [], ls='none', marker='o', markersize=4,
                      color=pdm.GAIN)]
    fig.legend(handles, ['splits', 'robust splits (improvement > 0 on every '
                         'set)'], loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.965, bottom=0.075,
                        hspace=0.6, wspace=0.28)
    pdm.save(fig, stem)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('res_dir', nargs='?',
                    default=osp.join(dm.ROOT, 'resemblance'),
                    help='output directory of tools/ood_class_resemblance.py')
    ap.add_argument('--space', choices=res.SPACES, default='full')
    args = ap.parse_args()
    plt.rcParams.update(pdm.STYLE)
    labels, profiles, appearance, splits, rho = load(args.res_dir, args.space)
    stem = lambda name: osp.join(args.res_dir, f'{name}_{args.space}')  # noqa: E731
    profile_figure(labels, profiles, appearance, stem('profile'))
    feature_bubbles(labels, profiles, stem('bubble_feature_singletons'))
    hypothesis_figure(labels, splits, rho, args.space, stem('hypothesis'))


if __name__ == '__main__':
    main()
