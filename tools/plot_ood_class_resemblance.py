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
                                       only); no preference (100/24 %) dashed
    bubble_feature_singletons_<space>  ID (x) against OOD (y)
                                       feature-divided % of the single-class
                                       splits, area = |improvement|: the
                                       feature-space twin of
                                       bubble_singletons
    hypothesis_<space>                 improvement against R_A ("together")
                                       and against the OOD feature-divided
                                       % ("cut through") over every split,
                                       robust splits highlighted, Spearman
                                       rho in the titles

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


def profile_figure(labels, profiles, appearance, stem):
    """One panel per set, classes in one order (by r_OOD on the last set)."""
    last = list(profiles)[-1]
    order = [r['class'] for r in sorted(
        profiles[last], key=lambda r: np.nan_to_num(r['r_ood'], nan=-1.0))]
    y = np.arange(len(order))
    xmax = max(max(np.nan_to_num(r['r_ood']), np.nan_to_num(r['r_id']))
               for rows in profiles.values() for r in rows)
    fig, axes = plt.subplots(1, len(profiles), figsize=(7.0, 4.8),
                             sharey=True, squeeze=False)
    for ax, (key, rows) in zip(axes[0], profiles.items()):
        by = {r['class']: r for r in rows}
        r_ood = [float(np.nan_to_num(by[c]['r_ood'])) for c in order]
        r_id = [float(np.nan_to_num(by[c]['r_id'])) for c in order]
        imp = [by[c]['improvement'] for c in order]
        edge = [pdm.GAIN if v > 0 else pdm.DROP for v in imp]
        ax.barh(y + 0.2, r_ood, height=0.4, color=[pdm.tint(c) for c in edge],
                edgecolor=edge, linewidth=0.6)
        ax.barh(y - 0.2, r_id, height=0.4, color='0.85', edgecolor='0.55',
                linewidth=0.6)
        if key in appearance:
            app = {r['class']: float(np.nan_to_num(r['r_ood']))
                   for r in appearance[key]}
            ax.scatter([app[c] for c in order], y + 0.2, s=10,
                       facecolor='none', edgecolor='black', linewidths=0.6,
                       zorder=4)
        for yi, x, v in zip(y, r_ood, imp):
            ax.text(x + 0.015 * xmax, yi + 0.2, pdm.signed(v), va='center',
                    fontsize=5.5,
                    color=pdm.GAIN_TXT if v > 0 else pdm.DROP_TXT)
        ax.axvline(100.0 / sb.NUM_CLASSES, color='0.4', ls=(0, (3, 2)),
                   lw=0.6)
        ax.set_xlim(0, xmax * 1.3)
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
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.2,
                        wspace=0.08)
    pdm.save(fig, stem)


def feature_bubbles(labels, profiles, stem):
    panels = OrderedDict(
        (labels[key], [dict(x=r['feat_div_id'], y=r['feat_div_ood'],
                            value=r['improvement'], text=r['class'])
                       for r in rows])
        for key, rows in profiles.items())
    pdm.bubble_grid(panels, 'ID feature-divided (%)',
                    'OOD feature-divided (%)', stem)


def hypothesis_figure(labels, splits, rho, space, stem):
    fig, axes = plt.subplots(len(splits), 2, figsize=(7.0, 7.6),
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
                ('feat_div_ood', 'cut through', 'OOD feature-divided (%)'))):
            ax = axes[i, j]
            ax.scatter([r[x] for r in other], [r['improvement'] for r in other],
                       s=5, color='0.72', linewidths=0)
            ax.scatter([r[x] for r in robust],
                       [r['improvement'] for r in robust], s=12,
                       color=pdm.GAIN, linewidths=0)
            ax.axhline(0.0, color='0.4', lw=0.6)
            ax.set_title(f'{labels[key]}, {name}: ' + r'$\rho$ = '
                         + dm.fmt(sel.get(x, float('nan'))), loc='left')
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
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09,
                        hspace=0.55, wspace=0.18)
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
