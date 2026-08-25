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
    args = ap.parse_args()

    paths = sorted(p for pattern in args.logs for p in glob.glob(pattern))
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


if __name__ == '__main__':
    main()
