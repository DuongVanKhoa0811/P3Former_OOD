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
    bubble_robust_<delta>       the same axes, limited to the robust
                                bubbles and the flat-MSP star (over every
                                other split in grey, which can range far
                                outside that box); the --top robust splits
                                by worst-set improvement are labelled by
                                rank, keyed to their names and classes in
                                the legend cell (also in robust.tsv)
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
import textwrap
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
STAT_STYLE = OrderedDict([  # rho_vs_threshold: one colour per statistic,
                            # drawn in this order; log_ratio last (F4: it
                            # ranks like precision, so it must sit on top,
                            # not the reverse) and thin, see LOG_RATIO_*
    ('ood_div', ('#0072B2', 'OOD divided %')),
    ('id_div', ('#D55E00', 'ID divided %')),
    ('precision', ('#CC79A7', 'divided precision')),
    ('log_ratio', ('#009E73', 'log OOD/ID divided ratio (ranks like '
                   'divided precision)')),
])
LOG_RATIO_LS = {'single': (0, (1, 1.2)), 'all': (0, (5, 1.5, 1, 1.5))}
LOG_RATIO_LW = 0.6  # thinner than the other statistics' 1.0 (F4)
TARGET_LABELS = OrderedDict([
    ('d_auroc', r'$\Delta$AUROC'), ('d_ap', r'$\Delta$AP'),
    ('d_fpr95', r'$\Delta$FPR@95'), ('improvement', 'improvement')])
# Label candidates: 16 directions (right, left, up, down first, then the
# diagonals, then the remaining eighth-turns), each at 5 extra gaps beyond
# the bubble edge; a gap > 0 draws a leader line.
LABEL_ANGLES = (0, 180, 90, 270, 45, 135, 225, 315,
                22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5)
LABEL_GAPS = (0.0, 6.0, 12.0, 20.0, 30.0)
_FIG_FIT_TOL = 1e-6  # a candidate box spilling less than this still "fits"
_STAR_SIZE = 70  # bubble_panel's scatter ``s`` for the flat-MSP star
# F1: a bubble's zorder is 3 + its rank among the panel's points (largest
# first), so a panel of more than a handful of points reached the labels'
# old zorder of 10. LABEL_ZORDER sits above every bubble a panel can ever
# have (at most NUM_CLASSES of them, the singleton panels' size).
LABEL_ZORDER = 1000
# F2: legend_cell's rank -> name: classes key for bubble_robust. bubble_
# robust's colour key always carries the star and background entries
# ('flat MSP at the same threshold', 'same OOD:ID ratio as flat MSP',
# 'other splits'), which are already too wide to fit beside the key at a
# readable size regardless of how many splits the key itself lists -- so
# the key always takes the cell's upper band, wrapped at
# ROBUST_KEY_WRAP_CHARS, and the colour/size keys shrink into the corners
# below it.
ROBUST_KEY_FONTSIZE = 6.0
ROBUST_KEY_WRAP_CHARS = 60


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


def _annotate(ax, item, angle, gap, fontsize, leader, renderer):
    th = math.radians(angle)
    dist = item['radius'] + 3.0 + gap
    dx, dy = dist * math.cos(th), dist * math.sin(th)
    ha = 'left' if dx > 0.3 * dist else 'right' if dx < -0.3 * dist else 'center'
    va = 'bottom' if dy > 0.3 * dist else 'top' if dy < -0.3 * dist else 'center'
    arrow = (dict(arrowstyle='-', lw=0.4, color=item['colour'], shrinkA=0,
                  shrinkB=item['radius']) if leader else None)
    ann = ax.annotate(item['text'], (item['x'], item['y']), xytext=(dx, dy),
                      textcoords='offset points', ha=ha, va=va,
                      multialignment=ha, color=item['colour'],
                      fontsize=fontsize, linespacing=1.05,
                      zorder=LABEL_ZORDER, arrowprops=arrow,
                      annotation_clip=False)
    # A freshly created annotation's own position is not yet resolved against
    # the current transform, so get_window_extent() on it would read back a
    # stale/placeholder box (e.g. anchored near the origin) until something
    # draws it; ann.draw(renderer) resolves it cheaply, without a full and
    # much more expensive fig.canvas.draw() of every artist on the figure.
    ann.draw(renderer)
    return ann


def place_labels(ax, items, fontsize=6.0, star=None, refine=True):
    """Label every item (dict: x, y in data units, text, colour, radius in
    points), largest bubble first, at the candidate position -- 16
    directions at 5 distances, right/left/up/down tried first -- whose text
    box overlaps the placed labels, the other bubbles, the flat-MSP
    ``star`` (data coords, when the panel has one) and the panel's title
    least, restricted to the candidates that fit fully inside the figure
    (the one that spills least, when none does). A label pushed away from
    its bubble gets a thin leader line.

    With ``refine`` (the default), up to 3 more passes then revisit every
    label, largest first, and search again -- against every other label's
    *current* position -- for a candidate that fits. The move is committed
    only when it does not increase that label's own overlap with the other
    labels *and* it strictly lowers the label's full cost (which also counts
    the bubbles, the star and the title, and 20x the area outside the axes),
    so a label can still move to clear the title or the axes frame when it
    can do so without crowding another label more. A pass that moves
    nothing stops the refinement early. Since every other label stays put
    while one is reconsidered, and its own overlap with them never goes up,
    the total label-to-label overlap of the whole panel never increases.
    ``refine=False`` returns the greedy pass alone."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    bubbles = []
    for item in items:
        cx, cy = ax.transData.transform((item['x'], item['y']))
        r = item['radius'] * px
        bubbles.append((cx - r, cy - r, cx + r, cy + r))
    star_box = None
    if star is not None:
        cx, cy = ax.transData.transform(star)
        r = math.sqrt(_STAR_SIZE / math.pi) * px
        star_box = (cx - r, cy - r, cx + r, cy + r)
    title_box = _box(Text.get_window_extent(ax.title, renderer))
    ax_frame = _box(ax.get_window_extent(renderer))
    fig_frame = _box(fig.bbox)

    def cost(i, box, others):
        """Overlap of ``box`` (label ``i``) with every obstacle: the other
        current labels (``others``), every bubble but its own, the star and
        the title (all at weight 1), plus 20x the area outside the axes."""
        c = (sum(_overlap(box, b) for b in others)
             + sum(_overlap(box, b) for j, b in enumerate(bubbles) if j != i)
             + _overlap(box, title_box) + 20.0 * _outside(box, ax_frame))
        if star_box is not None:
            c += _overlap(box, star_box)
        return c

    def label_overlap(box, others):
        """``box``'s overlap with the other current labels alone -- the
        quantity that must never increase across a refinement move, since
        it is exactly this label's contribution to the panel's total
        label-to-label overlap."""
        return sum(_overlap(box, b) for b in others)

    def candidate(i, item, others):
        """Best (angle, gap) for ``item``: least ``cost`` among the 80
        candidates that fit fully inside the figure, else the one among all
        80 that spills least from it."""
        best, fallback = None, None
        for gap in LABEL_GAPS:
            for angle in LABEL_ANGLES:
                ann = _annotate(ax, item, angle, gap, fontsize, False, renderer)
                box = _box(Text.get_window_extent(ann, renderer))
                ann.remove()
                spill = _outside(box, fig_frame)
                if fallback is None or spill < fallback[0]:
                    fallback = (spill, angle, gap)
                if spill <= _FIG_FIT_TOL:
                    c = cost(i, box, others)
                    if best is None or c < best[0]:
                        best = (c, angle, gap)
            if best is not None and best[0] == 0.0:
                break
        return (best[1], best[2]) if best is not None else fallback[1:]

    def place(i, item, others):
        angle, gap = candidate(i, item, others)
        ann = _annotate(ax, item, angle, gap, fontsize, gap > 0, renderer)
        return ann, _box(Text.get_window_extent(ann, renderer))

    order = sorted(range(len(items)), key=lambda j: -items[j]['radius'])
    annotations, placed = [None] * len(items), [None] * len(items)
    for i in order:
        others = [b for b in placed if b is not None]
        annotations[i], placed[i] = place(i, items[i], others)

    for _ in range(3 if refine else 0):
        moved = False
        for i in order:
            others = [placed[j] for j in range(len(items)) if j != i]
            old_cost = cost(i, placed[i], others)
            if old_cost <= 0.0:
                continue
            old_overlap = label_overlap(placed[i], others)
            ann, box = place(i, items[i], others)
            better = (label_overlap(box, others) <= old_overlap
                      and cost(i, box, others) < old_cost)
            if better:
                annotations[i].remove()
                annotations[i], placed[i] = ann, box
                moved = True
            else:
                ann.remove()
        if not moved:
            break


# ---------------------------------------------------------------- bubbles
def bubble_panel(ax, points, xlim, ylim, star=None, background=None):
    """Draw one panel. points: dicts with x, y (percent; 0 is drawn on the
    axis floor, hollow), value (improvement) and text (None: no label);
    star: flat MSP (x, y); background: (x, y) pairs drawn as grey dots.
    Returns the label items for :func:`place_labels`."""
    format_log_axes(ax, xlim, ylim)
    if background:
        # F3: unlike the foreground bubbles below, a background dot is
        # never pinned to the axis floor -- outside xlim/ylim it is simply
        # clipped (the scatter's default clip_on=True), so a split far
        # below the chosen limits (bubble_robust's limits cover only the
        # robust bubbles and the star, not the background) does not read
        # as though it sat on the axis.
        ax.scatter([x for x, _ in background], [y for _, y in background],
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


def _wrap_robust_key(robust, width):
    """``robust`` ([(rank, name, classes)]) as 'k  name: classes' lines,
    each entry wrapped to ``width``. ``break_long_words`` and
    ``break_on_hyphens`` are off: a name or a class has no whitespace of
    its own, so breaking inside one would make it unsearchable as a whole
    string in the rendered key."""
    lines = []
    for rank, name, classes in robust:
        lines += textwrap.wrap(f'{rank}  {name}: {classes}', width,
                               subsequent_indent='    ',
                               break_long_words=False,
                               break_on_hyphens=False) or ['']
    return lines


def legend_cell(ax, star=False, background=False, zeros=False, robust=None):
    """The fourth cell of the grid: colour, marker and size keys, plus,
    when ``bubble_robust`` names its top-N splits by rank (F2), the
    rank -> name: classes key (``robust``: [(rank, name, classes)], best
    worst-case first) in the cell's upper band, with the colour and size
    keys shrunk into the corners below it (see ROBUST_KEY_WRAP_CHARS for
    why there is no alternative, more compact layout) -- the figure size
    never changes."""
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
    size_handles = [_bubble_handle('0.4', v, face=False) for v in SIZE_KEY]
    size_labels = [str(v) for v in SIZE_KEY]
    size_title = '|improvement| (bubble area)'

    if not robust:
        first = ax.legend(handles, labels, loc='upper left', frameon=False,
                          handletextpad=0.5, borderaxespad=0.2)
        ax.add_artist(first)
        ax.legend(size_handles, size_labels, title=size_title,
                  loc='lower left', ncol=len(SIZE_KEY), frameon=False,
                  handlelength=3.2, handleheight=3.2, columnspacing=1.0,
                  borderaxespad=0.2)
        return

    # The key takes the cell's upper band, and the colour / size keys
    # shrink into the lower corners (F2's fallback layout -- see
    # ROBUST_KEY_WRAP_CHARS).
    lines = _wrap_robust_key(robust, ROBUST_KEY_WRAP_CHARS)
    ax.text(0.0, 0.98, '\n'.join(lines), transform=ax.transAxes, ha='left',
            va='top', fontsize=ROBUST_KEY_FONTSIZE, linespacing=1.25)
    first = ax.legend(handles, labels, loc='lower left', frameon=False,
                      handletextpad=0.4, borderaxespad=0.15, fontsize=5.5)
    ax.add_artist(first)
    ax.legend(size_handles, size_labels, title=size_title, loc='lower right',
              ncol=1, frameon=False, handlelength=2.0, handleheight=2.0,
              columnspacing=0.6, borderaxespad=0.15, fontsize=5.5,
              title_fontsize=5.5)


def bubble_grid(panels, xlabel, ylabel, stem, stars=None, backgrounds=None,
                notes=None, xlim=None, ylim=None, robust_key=None):
    """2 x 2 figure: one bubble panel per set (panels: label -> points) and
    a legend cell; the log axes are the same in every panel.

    ``xlim`` / ``ylim`` override the default axis limits (every panel's
    points, plus ``stars`` and ``backgrounds``) when given -- e.g.
    :func:`bubble_robust` passes the robust-only limits from
    :func:`foreground_limits` (F3), so the background dots cannot stretch
    them. ``notes``: label -> a short string drawn inside that panel's own
    axes (F5, e.g. plot_ood_class_resemblance.py's skipped-class count).
    ``robust_key``: forwarded to :func:`legend_cell` (F2)."""
    stars, backgrounds, notes = stars or {}, backgrounds or {}, notes or {}
    if xlim is None or ylim is None:
        xs = [p['x'] for pts in panels.values() for p in pts]
        ys = [p['y'] for pts in panels.values() for p in pts]
        for x, y in stars.values():
            xs.append(x)
            ys.append(y)
        for points in backgrounds.values():
            xs += [x for x, _ in points]
            ys += [y for _, y in points]
        xlim = log_limits(xs) if xlim is None else xlim
        ylim = log_limits(ys) if ylim is None else ylim
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.07,
                        wspace=0.24, hspace=0.32)
    cells = (axes[0, 0], axes[0, 1], axes[1, 0])
    labelled, zeros = [], False
    for i, (ax, (label, points)) in enumerate(zip(cells, panels.items())):
        labelled.append((ax, bubble_panel(ax, points, xlim, ylim,
                                          stars.get(label),
                                          backgrounds.get(label)),
                         stars.get(label)))
        zeros = zeros or any(p['x'] <= 0 or p['y'] <= 0 for p in points)
        ax.set_title(f'({"abc"[i]}) {label}', loc='left')
        ax.set_xlabel(xlabel)
        if i != 1:
            ax.set_ylabel(ylabel)
        if notes.get(label):
            ax.text(0.97, 0.03, notes[label], transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=6, color='0.5')
    for ax, items, star in labelled:
        place_labels(ax, items, star=star)
    legend_cell(axes[1, 1], star=bool(stars), background=bool(backgrounds),
                zeros=zeros, robust=robust_key)
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
    """Robust splits as bubbles and every other split as a grey background
    dot. The ``top`` splits by worst-set improvement are labelled by rank
    (F2: their names are too long and too dense to label the panel with
    directly), best worst-case first. Returns (panels, backgrounds, key:
    [(rank, name, classes)] for legend_cell's robust key)."""
    key = dm.delta_key(threshold)
    worst, group_a = {}, {}
    for rows in tables.values():
        for r in rows:
            if r['robust']:
                worst[r['name']] = min(worst.get(r['name'], float('inf')),
                                       r['improvement'])
                group_a[r['name']] = r['group_A']
    ranked = sorted(worst, key=lambda n: -worst[n])[:top]
    rank_of = {name: i + 1 for i, name in enumerate(ranked)}
    panels, backgrounds = OrderedDict(), OrderedDict()
    for k, rows in tables.items():
        panels[labels[k]] = [
            dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                 value=r['improvement'],
                 text=(str(rank_of[r['name']]) if r['name'] in rank_of
                       else None))
            for r in rows if r['robust']]
        backgrounds[labels[k]] = [(r[f'id_div@{key}'], r[f'ood_div@{key}'])
                                  for r in rows if not r['robust']]
    return (panels, backgrounds,
            [(rank_of[n], n, group_a[n]) for n in ranked])


def foreground_limits(panels, stars=None):
    """(xlim, ylim) from the bubbles in ``panels`` and ``stars`` alone --
    what :func:`bubble_robust` hands ``bubble_grid`` (F3). The background
    splits it also draws span down to a fraction of a percent; including
    them in the shared log limits squeezed the robust bubbles into the top
    of the panel."""
    stars = stars or {}
    xs = [p['x'] for pts in panels.values() for p in pts]
    ys = [p['y'] for pts in panels.values() for p in pts]
    for x, y in stars.values():
        xs.append(x)
        ys.append(y)
    return log_limits(xs), log_limits(ys)


def bubble_robust(tables, labels, threshold, top, xlabel, ylabel, stem,
                  stars):
    """The robust-splits figure: :func:`robust_panels`' bubbles (labelled
    by rank) over every other split in grey, axis limits from the robust
    bubbles and the flat-MSP star alone (F3), and the rank -> name: classes
    key in the legend cell (F2)."""
    panels, backgrounds, robust_key = robust_panels(tables, labels,
                                                     threshold, top)
    xlim, ylim = foreground_limits(panels, stars)
    bubble_grid(panels, xlabel, ylabel, stem, stars=stars,
                backgrounds=backgrounds, xlim=xlim, ylim=ylim,
                robust_key=robust_key)


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
                    d, v = [d for d, _ in pts], [v for _, v in pts]
                    if stat == 'log_ratio':
                        # F4: within one set, log_ratio ranks the splits
                        # almost exactly like precision (up to the +0.5
                        # continuity correction), so its rho coincides with
                        # precision's; drawn thinner, on top (STAT_STYLE
                        # puts it after precision) and dashed its own way,
                        # it still shows where the two differ.
                        ax.plot(d, v, ls=LOG_RATIO_LS[population],
                                color=colour, lw=LOG_RATIO_LW, marker='o',
                                markersize=1.6)
                    else:
                        ax.plot(d, v, ls=ls, color=colour, lw=1.0,
                                marker='o', markersize=2.2)
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
    bubble_robust(tables, labels, args.threshold, args.top, xlabel, ylabel,
                  osp.join(args.divided_dir, f'bubble_robust_{key}'), stars)
    rho_figure(rho, labels, osp.join(args.divided_dir, 'rho_vs_threshold'))


if __name__ == '__main__':
    main()
