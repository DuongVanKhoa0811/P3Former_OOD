#!/usr/bin/env python
"""Summarise the hierarchy-ablation logs as deltas versus the flat scores.

Reads the ``method | AUROC | AP | FPR@95`` rows printed by _OODPointMetric
from one or more test.py logs and groups the hierarchy-aware scores by
hierarchy (the key prefix before ``_group_`` / ``_gn_``; the unprefixed keys
are the current hierarchy, labelled ``current``). For every hierarchy and
family (Group, GN) it prints, per baseline, the fluctuation

    delta = hierarchy score - flat score        (AUROC, AP, FPR@95)

i.e. ``group_msp - msp``, ``gn_energy - energy``, ..., and the mean of these
deltas over the baselines. AUROC / AP improve when the delta is positive,
FPR@95 improves when it is negative. Rows that appear in several logs (flat
scores, current hierarchy) are de-duplicated.

Run from the repo root (``--family`` is required: one table per family,
ordered by ``improvement`` = mean dAUROC + mean dAP - mean dFPR@95, shown
as the last column):
    python tools/summarize_hierarchy_ablation.py --family group work_dirs/p3former_2xb1_3x_dso_ood_hier/b*/*/*.log
    python tools/summarize_hierarchy_ablation.py --family gn --exclude odin,msp LOG...
    python tools/summarize_hierarchy_ablation.py --family group --exclude odin --plot top4_group.png LOG...
      (--plot draws the --top hierarchies, one panel each: dAUROC / dAP /
      -dFPR@95 bars per baseline plus their mean, caption = improvement)
``--family gn`` is refused for tools/sweep_bipartitions.py logs: their GN
MSP / GN Entropy rows are approximate (2026-09-24 note in DOCs.md).
"""
import argparse
import glob
import re
from collections import OrderedDict, defaultdict

ROW = re.compile(r'^\s*(\S+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*$')
FAMILIES = ('group', 'gn')
BASELINES = ('msp', 'maxlogit', 'odin', 'energy', 'entropy')
METRICS = ('AUROC', 'AP', 'FPR@95')
# Always dropped: the group / GN MaxLogit formulation (max within groups,
# then max over groups) does not follow the paper's intent, so it is left out
# of the columns and the means regardless of --exclude.
ALWAYS_EXCLUDE = ('maxlogit', )
# Marker line of a tools/sweep_bipartitions.py log. Its GN MSP / GN Entropy
# rows are approximate (the sweep's histogram bins do not resolve the
# interior value those scores pile up at; FPR@95 off by up to ~20 points),
# so the GN family is refused for such logs.
SWEEP_MARKER = '# offline bipartition sweep'


def parse_logs(paths):
    """Return OrderedDict method -> (auroc, ap, fpr95)."""
    rows = OrderedDict()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                # strip the mmengine "date - mmengine - INFO - " prefix
                line = line.split(' - INFO - ')[-1]
                m = ROW.match(line)
                if m and m.group(1) != 'method':
                    rows[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    return rows


def config_names(paths):
    """Config names behind the logs, taken from the ``work_dir = 'work_dirs/
    <name>/...'`` line of each config dump (batch sub-dirs are dropped)."""
    names = set()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                m = re.match(r"work_dir = '(?:.*/)?work_dirs/([^/']+)", line)
                if m:
                    names.add(m.group(1))
                    break
    return sorted(names)


def offline_sweep_logs(paths):
    """The logs written by tools/sweep_bipartitions.py (marker line)."""
    found = []
    for path in paths:
        with open(path) as fh:
            head = [fh.readline() for _ in range(5)]
        if any(line.startswith(SWEEP_MARKER) for line in head):
            found.append(path)
    return found


def check_family(family, paths):
    """Refuse ``--family gn`` on offline sweep logs (see SWEEP_MARKER)."""
    sweep = offline_sweep_logs(paths)
    if family == 'gn' and sweep:
        raise ValueError(
            '--family gn is not available for offline bipartition sweep logs '
            f'({", ".join(sweep)}): their GN MSP / GN Entropy rows are '
            'approximate. Rank the GN family from test.py logs only.')


def split_key(method):
    """'m2_go_group_msp' -> ('m2_go', 'group', 'msp'); 'gn_msp' ->
    ('current', 'gn', 'msp'); flat keys -> (None, None, key)."""
    for fam in FAMILIES:
        tag = f'_{fam}_'
        if tag in method:
            # rsplit: a hierarchy name may itself end in the family tag
            # (m2_gn = ground+nature -> 'm2_gn_gn_msp').
            prefix, base = method.rsplit(tag, 1)
            return prefix, fam, base
        if method.startswith(f'{fam}_'):
            return 'current', fam, method[len(fam) + 1:]
    return None, None, method


def summarise(rows, exclude=()):
    """Return (flat, summary) where flat is {baseline: (auroc, ap, fpr95)}
    and summary is OrderedDict hierarchy -> family -> {'deltas': {baseline:
    (dAUROC, dAP, dFPR95)}, 'mean': (dAUROC, dAP, dFPR95)}."""
    flat = {k: v for k, v in rows.items() if split_key(k)[0] is None}
    per_hier = defaultdict(lambda: {fam: OrderedDict() for fam in FAMILIES})
    for method, vals in rows.items():
        hier, fam, base = split_key(method)
        if hier is None or base in exclude or base not in flat:
            continue
        per_hier[hier][fam][base] = tuple(
            v - f for v, f in zip(vals, flat[base]))
    summary = OrderedDict()
    for hier, fams in per_hier.items():
        entry = {}
        for fam in FAMILIES:
            deltas = fams[fam]
            if deltas:
                mean = tuple(
                    sum(d[i] for d in deltas.values()) / len(deltas)
                    for i in range(len(METRICS)))
                entry[fam] = {'deltas': deltas, 'mean': mean}
        summary[hier] = entry
    return flat, summary


def _fmt(delta):
    return '/'.join(f'{x:+.2f}' for x in delta)


def _improvement(mean_delta):
    """Single ranking number: mean dAUROC + mean dAP - mean dFPR@95 (every
    term positive when the hierarchy beats the flat scores)."""
    d_auroc, d_ap, d_fpr95 = mean_delta
    return d_auroc + d_ap - d_fpr95


# --plot: one panel per hierarchy in a 2-column grid, three metric rows each
# (AUROC, AP, -FPR@95 so that up = better), one bar per baseline plus a Mean
# bar, caption = improvement (the sum of the three Mean bars).
PLOT_LABELS = {'msp': 'MSP', 'maxlogit': 'MaxLogit', 'odin': 'ODIN',
               'energy': 'Energy', 'entropy': 'Entropy'}
PLOT_ROWS = (('AUROC', 0, 1.0), ('AP', 1, 1.0), ('- FPR@95', 2, -1.0))


def plot_top(summary, fam, order, baselines, out, title=None):
    """Save the delta bar panels of the hierarchies in ``order`` to ``out``."""
    import math

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    ink, bar_fill, mean_fill = 'black', 'white', '#555555'
    # Shared y-range per metric row (signed as plotted), with label headroom.
    ylims = []
    for _, mi, sign in PLOT_ROWS:
        vals = [sign * v for h in order for v in (
            [summary[h][fam]['deltas'][b][mi] for b in baselines]
            + [summary[h][fam]['mean'][mi]])]
        lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
        span = (hi - lo) or 1.0
        ylims.append((lo - 0.25 * span, hi + 0.25 * span))

    ncols = min(2, len(order))
    nrows = math.ceil(len(order) / ncols)
    # Panel rows plus one shorter full-width row for the improvement
    # distribution over every hierarchy.
    fig = plt.figure(figsize=(6.5 * ncols, 6.0 * nrows + 2.8))
    if title:
        fig.suptitle(title, fontsize=13, y=0.985)
    outer = GridSpec(nrows + 1, ncols, figure=fig, left=0.12, right=0.98,
                     top=0.95 if title else 0.97, bottom=0.05,
                     hspace=0.45, wspace=0.35,
                     height_ratios=[6.0] * nrows + [2.4])
    x = list(range(len(baselines))) + [len(baselines) + 1]  # gap before Mean
    names = [PLOT_LABELS.get(b, b) for b in baselines] + ['Mean']
    for k, hier in enumerate(order):
        entry = summary[hier][fam]
        spec = outer[k // ncols, k % ncols]
        inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, hspace=0.9)
        for row, (label, mi, sign) in enumerate(PLOT_ROWS):
            ax = fig.add_subplot(inner[row])
            heights = [sign * entry['deltas'][b][mi] for b in baselines]
            heights.append(sign * entry['mean'][mi])
            ax.bar(x, heights, width=0.8,
                   color=[bar_fill] * len(baselines) + [mean_fill],
                   edgecolor=ink, linewidth=1.2, zorder=2)
            ax.axhline(0, color=ink, linewidth=1.2, zorder=3)
            lo, hi = ylims[mi]
            pad = 0.02 * (hi - lo)
            for xi, h in zip(x, heights):
                ax.text(xi, h + (pad if h >= 0 else -pad), f'{h:+.1f}',
                        ha='center', va='bottom' if h >= 0 else 'top',
                        fontsize=8, color=ink)
            ax.set_ylim(lo, hi)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9)
            ax.set_yticks([])
            for side in ('top', 'right', 'left', 'bottom'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='x', length=0)
            ax.set_ylabel(label, rotation=0, ha='right', va='center',
                          fontsize=12, labelpad=14)
        pos = spec.get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, pos.y0 - 0.035,
                 f'{hier}_{fam} - flat\n'
                 f'(improvement = {_improvement(entry["mean"]):+.2f})',
                 ha='center', va='top', fontsize=12)

    # Improvement distribution over all hierarchies of this family.
    values = [_improvement(summary[h][fam]['mean'])
              for h in summary if fam in summary[h]]
    ax = fig.add_subplot(outer[nrows, :])
    ax.hist(values, bins=20, color='#dddddd', edgecolor=ink, linewidth=1.0,
            zorder=2)
    ax.axvline(0, color='red', linewidth=1.5, zorder=3)
    # Reference hierarchies: the current six-group one and the split one.
    for name, color, style in (('current', 'tab:blue', '--'),
                               ('sp', 'tab:green', ':')):
        if name in summary and fam in summary[name]:
            value = _improvement(summary[name][fam]['mean'])
            ax.axvline(value, color=color, linestyle=style, linewidth=1.8,
                       zorder=3, label=f'{name} ({value:+.2f})')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left', fontsize=9, frameon=False)
    ax.set_xlabel('improvement = mean dAUROC + mean dAP - mean dFPR@95',
                  fontsize=10)
    ax.set_ylabel('# hierarchies', fontsize=10)
    n_pos = sum(v > 0 for v in values)
    ax.set_title(f'improvement distribution: {len(values)} hierarchies, '
                 f'{n_pos} above 0', fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('logs', nargs='+', help='test.py log files (globs ok)')
    ap.add_argument('--exclude', default=[],
                    type=lambda s: [b for b in s.split(',') if b],
                    help='comma-separated baselines to drop, e.g. '
                    '--exclude odin or --exclude odin,msp')
    ap.add_argument('--family', required=True, choices=list(FAMILIES),
                    help='score family to report: group or gn; the table is '
                    'ordered by improvement (mean dAUROC + mean dAP - '
                    'mean dFPR@95)')
    ap.add_argument('--plot', metavar='PNG', default=None,
                    help='also save a figure of the --top hierarchies (one '
                    'panel each: dAUROC / dAP / -dFPR@95 bars per baseline '
                    'plus their mean) to this path')
    ap.add_argument('--top', type=int, default=4,
                    help='number of hierarchies in the --plot figure')
    args = ap.parse_args()

    paths = sorted(p for pattern in args.logs for p in glob.glob(pattern))
    try:
        check_family(args.family, paths)
    except ValueError as err:
        ap.error(str(err))
    rows = parse_logs(paths)
    exclude = [b for b in BASELINES
               if b in set(args.exclude) | set(ALWAYS_EXCLUDE)]
    flat, summary = summarise(rows, exclude=set(exclude))
    print(f'{len(paths)} logs, {len(rows)} score rows, '
          f'{len(summary)} hierarchies (excluding {", ".join(exclude)})')
    print('flat reference (AUROC/AP/FPR@95): ' + '  '.join(
        f'{k} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}' for k, v in flat.items()))
    print('delta = hierarchy - flat, as dAUROC/dAP/dFPR@95 '
          '(AUROC, AP better when +; FPR@95 better when -); '
          'improvement = mean dAUROC + mean dAP - mean dFPR@95')

    fam = args.family
    baselines = [b for b in BASELINES if b not in set(exclude)]
    # Hierarchies with this family, largest improvement first.
    order = sorted((h for h in summary if fam in summary[h]),
                   key=lambda h: -_improvement(summary[h][fam]['mean']))
    print()
    print(f'| hierarchy ({fam}) | ' + ' | '.join(baselines)
          + ' | mean | improvement |')
    print('| --- | ' + ' | '.join('---' for _ in baselines) + ' | --- | --- |')
    for hier in order:
        entry = summary[hier][fam]
        cells = [_fmt(entry['deltas'][b]) if b in entry['deltas'] else '-'
                 for b in baselines]
        print(f'| {hier} | ' + ' | '.join(cells)
              + f' | {_fmt(entry["mean"])} '
              f'| {_improvement(entry["mean"]):+.2f} |')

    if args.plot:
        title = (f'{", ".join(config_names(paths)) or "?"}  |  family: {fam}'
                 f'  |  excluded: {", ".join(exclude) or "none"}')
        plot_top(summary, fam, order[:args.top], baselines, args.plot, title)


if __name__ == '__main__':
    main()
