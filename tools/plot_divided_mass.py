#!/usr/bin/env python
"""Figures of tools/divided_mass.py: bubble charts and correlation curves.

Reads the tables of tools/divided_mass.py (default
work_dirs/p3former_2xb1_3x_dso_ood_dump/divided_mass) and writes, next to
them, each figure as a PDF (vector) and a PNG (300 dpi) in the style of the
paper figures: serif text, Okabe-Ito blue for a gain and orange for a drop,
bubble area proportional to |improvement|, log-log axes, one panel per set
in a 2 x 2 grid whose fourth cell holds the legends.

    bubble_singletons_<delta>   ID divided % (x) against OOD divided % (y) at
                                --threshold for the 24 single-class splits,
                                with flat MSP at the same threshold (star) and
                                the iso-ratio diagonal through it: a split
                                above it has a higher OOD:ID ratio than flat
    bubble_robust_<delta>       the same axes for the robust splits
                                (improvement > 0 on every set) over every
                                other split in grey; the --top robust splits
                                by worst-set improvement carry their names
                                (their classes are in robust.tsv)
    rho_vs_threshold            Spearman rho of each divided statistic
                                against each metric delta, over the
                                thresholds

Run from the repo root:
    python tools/plot_divided_mass.py [DIVIDED_DIR] [--threshold 0.05] [--top 8]
"""
import argparse
import math
import os.path as osp
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import divided_mass as dm  # noqa: E402

STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'pdf.fonttype': 42,  # embedded TrueType fonts (camera-ready requirement)
    'ps.fonttype': 42,
}
GAIN, DROP, NA = '#0072B2', '#D55E00', '#8C8C8C'  # Okabe-Ito
GAIN_TXT, DROP_TXT = '#00507D', '#A34700'  # darker shades for label text
AREA_PER_UNIT = 12  # marker area (pt^2) per unit of |improvement|
SIZE_KEY = (5, 20, 40)
STAT_STYLE = OrderedDict([  # rho_vs_threshold: one colour per statistic
    ('ood_div', ('#0072B2', 'OOD divided %')),
    ('id_div', ('#D55E00', 'ID divided %')),
    ('log_ratio', ('#009E73', 'log OOD/ID divided ratio')),
    ('precision', ('#CC79A7', 'divided precision')),
])
TARGET_LABELS = OrderedDict([
    ('d_auroc', r'$\Delta$AUROC'), ('d_ap', r'$\Delta$AP'),
    ('d_fpr95', r'$\Delta$FPR@95'), ('improvement', 'improvement')])
# Label candidates: 8 directions, then the same further out with a leader.
LABEL_ANGLES = (0, 180, 90, 270, 45, 135, 315, 225)
LABEL_GAPS = (0.0, 7.0, 14.0)  # extra points beyond the bubble edge


def tint(colour, k=0.35):
    """Opaque light version of ``colour`` (k = share of the colour)."""
    return tuple(k * np.array(to_rgb(colour)) + (1 - k))


def area(value):
    return AREA_PER_UNIT * abs(value)


def signed(value):
    return f'{value:+.1f}'.replace('-', '−')  # typographic minus


def save(fig, stem):
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    print(f'saved {stem}.pdf / .png')


def log_limits(values, pad=1.6):
    """(low, high) of a log axis around the positive finite ``values``."""
    pos = [v for v in values if v > 0 and math.isfinite(v)]
    if not pos:
        return 0.01, 100.0
    return min(pos) / pad, max(pos) * pad


def format_log_axes(ax, xlim, ylim):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(which='major', color='0.88', lw=0.5)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- labels
def _box(extent):
    return (extent.x0, extent.y0, extent.x1, extent.y1)


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _outside(box, frame):
    return (box[2] - box[0]) * (box[3] - box[1]) - _overlap(box, frame)


def _annotate(ax, item, angle, gap, fontsize, leader):
    th = math.radians(angle)
    dist = item['radius'] + 3.0 + gap
    dx, dy = dist * math.cos(th), dist * math.sin(th)
    ha = 'left' if dx > 0.3 * dist else 'right' if dx < -0.3 * dist else 'center'
    va = 'bottom' if dy > 0.3 * dist else 'top' if dy < -0.3 * dist else 'center'
    arrow = (dict(arrowstyle='-', lw=0.4, color=item['colour'], shrinkA=0,
                  shrinkB=item['radius']) if leader else None)
    return ax.annotate(item['text'], (item['x'], item['y']), xytext=(dx, dy),
                       textcoords='offset points', ha=ha, va=va,
                       multialignment=ha, color=item['colour'],
                       fontsize=fontsize, linespacing=1.05, zorder=10,
                       arrowprops=arrow, annotation_clip=False)


def place_labels(ax, items, fontsize=6.5):
    """Label every item (dict: x, y in data units, text, colour, radius in
    points), largest bubble first, at the candidate position -- 8 directions
    at 3 distances -- whose text box overlaps the placed labels, the other
    bubbles and the outside of the axes least. A label pushed away from its
    bubble gets a thin leader line."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    bubbles = []
    for item in items:
        cx, cy = ax.transData.transform((item['x'], item['y']))
        r = item['radius'] * px
        bubbles.append((cx - r, cy - r, cx + r, cy + r))
    frame = _box(ax.get_window_extent(renderer))
    placed = []
    for i in sorted(range(len(items)), key=lambda j: -items[j]['radius']):
        best = None
        for gap in LABEL_GAPS:
            for angle in LABEL_ANGLES:
                ann = _annotate(ax, items[i], angle, gap, fontsize, False)
                box = _box(Text.get_window_extent(ann, renderer))
                ann.remove()
                cost = (sum(_overlap(box, b) for b in placed)
                        + sum(_overlap(box, b) for j, b in enumerate(bubbles)
                              if j != i)
                        + 10.0 * _outside(box, frame))
                if best is None or cost < best[0]:
                    best = (cost, angle, gap)
            if best[0] == 0.0:
                break
        _, angle, gap = best
        ann = _annotate(ax, items[i], angle, gap, fontsize, leader=gap > 0)
        placed.append(_box(Text.get_window_extent(ann, renderer)))


# ---------------------------------------------------------------- bubbles
def bubble_panel(ax, points, xlim, ylim, star=None, background=None):
    """Draw one panel. points: dicts with x, y (percent; 0 is drawn on the
    axis floor, hollow), value (improvement) and text (None: no label);
    star: flat MSP (x, y); background: (x, y) pairs drawn as grey dots.
    Returns the label items for :func:`place_labels`."""
    format_log_axes(ax, xlim, ylim)
    if background:
        ax.scatter([max(x, xlim[0]) for x, _ in background],
                   [max(y, ylim[0]) for _, y in background],
                   s=4, color='0.78', linewidths=0, zorder=1)
    if star is not None:
        x0, y0 = star
        xs = np.array(xlim)
        ax.plot(xs, xs * (y0 / x0), ls=(0, (4, 3)), lw=0.7, color='0.35',
                zorder=2)
        ax.scatter([x0], [y0], marker='*', s=70, color='black', zorder=20)
    items = []
    for z, pt in enumerate(sorted(points, key=lambda p: -abs(p['value']))):
        colour, text_colour = ((GAIN, GAIN_TXT) if pt['value'] > 0
                               else (DROP, DROP_TXT))
        zero = pt['x'] <= 0 or pt['y'] <= 0
        x, y = max(pt['x'], xlim[0]), max(pt['y'], ylim[0])
        ax.scatter([x], [y], s=area(pt['value']),
                   facecolor='none' if zero else tint(colour),
                   edgecolor=colour, linewidths=1.0, zorder=3 + z,
                   clip_on=False)
        if pt.get('text'):
            items.append(dict(x=x, y=y, colour=text_colour,
                              text=f'{pt["text"]}\n{signed(pt["value"])}',
                              radius=math.sqrt(area(pt['value'])) / 2))
    return items


def _bubble_handle(colour, value, face=True):
    return Line2D([], [], ls='none', marker='o',
                  markersize=math.sqrt(area(value)),
                  markerfacecolor=tint(colour) if face else 'none',
                  markeredgecolor=colour, markeredgewidth=1.0)


def legend_cell(ax, star=False, background=False, zeros=False):
    """The fourth cell of the grid: colour, marker and size keys."""
    ax.axis('off')
    handles = [_bubble_handle(GAIN, 10), _bubble_handle(DROP, 10)]
    labels = ['improvement > 0', 'improvement < 0']
    if zeros:
        handles.append(_bubble_handle(NA, 10, face=False))
        labels.append('0 %, drawn on the axis')
    if star:
        handles += [Line2D([], [], ls='none', marker='*', markersize=9,
                           color='black'),
                    Line2D([], [], ls=(0, (4, 3)), lw=0.7, color='0.35')]
        labels += ['flat MSP at the same threshold',
                   'same OOD:ID ratio as flat MSP']
    if background:
        handles.append(Line2D([], [], ls='none', marker='o', markersize=2.5,
                              color='0.78'))
        labels.append('other splits')
    first = ax.legend(handles, labels, loc='upper left', frameon=False,
                      handletextpad=0.5, borderaxespad=0.2)
    ax.add_artist(first)
    ax.legend([_bubble_handle('0.4', v, face=False) for v in SIZE_KEY],
              [str(v) for v in SIZE_KEY], title='|improvement| (bubble area)',
              loc='lower left', ncol=len(SIZE_KEY), frameon=False,
              handlelength=3.2, handleheight=3.2, columnspacing=1.0,
              borderaxespad=0.2)


def bubble_grid(panels, xlabel, ylabel, stem, stars=None, backgrounds=None):
    """2 x 2 figure: one bubble panel per set (panels: label -> points) and
    a legend cell; the log axes are the same in every panel."""
    stars, backgrounds = stars or {}, backgrounds or {}
    xs = [p['x'] for pts in panels.values() for p in pts]
    ys = [p['y'] for pts in panels.values() for p in pts]
    for x, y in stars.values():
        xs.append(x)
        ys.append(y)
    for points in backgrounds.values():
        xs += [x for x, _ in points]
        ys += [y for _, y in points]
    xlim, ylim = log_limits(xs), log_limits(ys)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08,
                        wspace=0.22, hspace=0.3)
    cells = (axes[0, 0], axes[0, 1], axes[1, 0])
    labelled, zeros = [], False
    for i, (ax, (label, points)) in enumerate(zip(cells, panels.items())):
        labelled.append((ax, bubble_panel(ax, points, xlim, ylim,
                                          stars.get(label),
                                          backgrounds.get(label))))
        zeros = zeros or any(p['x'] <= 0 or p['y'] <= 0 for p in points)
        ax.set_title(f'({"abc"[i]}) {label}', loc='left')
        ax.set_xlabel(xlabel)
        if i != 1:
            ax.set_ylabel(ylabel)
    for ax, items in labelled:
        place_labels(ax, items)
    legend_cell(axes[1, 1], star=bool(stars), background=bool(backgrounds),
                zeros=zeros)
    save(fig, stem)


# ---------------------------------------------------------------- figures
def load(divided_dir):
    tables = OrderedDict((key, dm.read_tsv(osp.join(divided_dir, f'{key}.tsv')))
                         for key in dm.SETS)
    labels = OrderedDict((key, spec['label']) for key, spec in dm.SETS.items())
    flat = dm.read_tsv(osp.join(divided_dir, 'flat.tsv'))
    rho = dm.read_tsv(osp.join(divided_dir, 'rho.tsv'))
    return tables, labels, flat, rho


def flat_stars(flat, threshold):
    key = dm.delta_key(threshold)
    return {row['set']: (row[f'id_unc@{key}'], row[f'ood_unc@{key}'])
            for row in flat}


def singleton_panels(tables, labels, threshold):
    key = dm.delta_key(threshold)
    return OrderedDict(
        (labels[k], [dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                          value=r['improvement'], text=r['group_A'])
                     for r in rows if r['size_A'] == 1])
        for k, rows in tables.items())


def robust_panels(tables, labels, threshold, top):
    """Robust splits as bubbles (the ``top`` by worst-set improvement named)
    and every other split as a grey background dot."""
    key = dm.delta_key(threshold)
    worst = {}
    for rows in tables.values():
        for r in rows:
            if r['robust']:
                worst[r['name']] = min(worst.get(r['name'], float('inf')),
                                       r['improvement'])
    named = set(sorted(worst, key=lambda n: -worst[n])[:top])
    panels, backgrounds = OrderedDict(), OrderedDict()
    for k, rows in tables.items():
        panels[labels[k]] = [
            dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                 value=r['improvement'],
                 text=r['name'] if r['name'] in named else None)
            for r in rows if r['robust']]
        backgrounds[labels[k]] = [(r[f'id_div@{key}'], r[f'ood_div@{key}'])
                                  for r in rows if not r['robust']]
    return panels, backgrounds


def rho_figure(rho, labels, stem):
    """Spearman rho against the threshold: one row per target, one column
    per set; solid = single-class splits, dashed = all splits."""
    fig, axes = plt.subplots(len(TARGET_LABELS), len(labels),
                             figsize=(7.0, 8.0), sharex=True, sharey=True,
                             squeeze=False)
    for i, (target, target_label) in enumerate(TARGET_LABELS.items()):
        for j, (key, set_label) in enumerate(labels.items()):
            ax = axes[i, j]
            for stat, (colour, _) in STAT_STYLE.items():
                for population, ls in (('single', '-'), ('all', '--')):
                    pts = sorted(
                        (float(r['delta']), r['rho']) for r in rho
                        if r['set'] == key and r['target'] == target
                        and r['statistic'] == stat
                        and r['population'] == population)
                    ax.plot([d for d, _ in pts], [v for _, v in pts], ls=ls,
                            color=colour, lw=1.0, marker='o', markersize=2.2)
            ax.axhline(0.0, color='0.5', lw=0.5)
            ax.set_xscale('log')
            ax.set_ylim(-1.05, 1.05)
            ax.grid(which='major', color='0.9', lw=0.4)
            if i == 0:
                ax.set_title(set_label)
            if j == 0:
                ax.set_ylabel(r'$\rho$ vs ' + target_label)
            if i == len(TARGET_LABELS) - 1:
                ax.set_xlabel(r'threshold $\delta$')
    handles = [Line2D([], [], color=c, lw=1.2) for c, _ in STAT_STYLE.values()]
    handles += [Line2D([], [], color='0.3', ls='-', lw=1.0),
                Line2D([], [], color='0.3', ls='--', lw=1.0)]
    names = [n for _, n in STAT_STYLE.values()] + ['single-class splits',
                                                   'all splits']
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.12,
                        hspace=0.25, wspace=0.12)
    save(fig, stem)


def main():
    choices = ', '.join(dm.delta_key(d) for d in dm.DELTAS)
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('divided_dir', nargs='?',
                    default=osp.join(dm.ROOT, 'divided_mass'),
                    help='output directory of tools/divided_mass.py; the '
                    'figures are written there')
    ap.add_argument('--threshold', type=float, default=dm.HEADLINE,
                    help=f'divided threshold of the bubble charts: {choices}')
    ap.add_argument('--top', type=int, default=8,
                    help='robust splits named in bubble_robust')
    args = ap.parse_args()
    if args.threshold not in dm.DELTAS:
        ap.error(f'--threshold must be one of {choices}')
    plt.rcParams.update(STYLE)
    tables, labels, flat, rho = load(args.divided_dir)
    key = dm.delta_key(args.threshold)
    xlabel = f'ID divided at $\\delta$ = {key} (%)'
    ylabel = f'OOD divided at $\\delta$ = {key} (%)'
    stars = flat_stars(flat, args.threshold)
    bubble_grid(singleton_panels(tables, labels, args.threshold), xlabel,
                ylabel, osp.join(args.divided_dir, f'bubble_singletons_{key}'),
                stars=stars)
    panels, backgrounds = robust_panels(tables, labels, args.threshold,
                                        args.top)
    bubble_grid(panels, xlabel, ylabel,
                osp.join(args.divided_dir, f'bubble_robust_{key}'),
                stars=stars, backgrounds=backgrounds)
    rho_figure(rho, labels, osp.join(args.divided_dir, 'rho_vs_threshold'))


if __name__ == '__main__':
    main()
