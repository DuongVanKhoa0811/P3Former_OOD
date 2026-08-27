#!/usr/bin/env python
"""Average the OOD metrics of each score family (Normal / Group / GN).

The markdown table below (copied from DOCs.md) is the fixed input. Rows are
grouped by their prefix -- ``Group <name>`` -> Group, ``GN <name>`` -> Group
Normalisation, anything else -> Normal -- and AUROC / AP / FPR@95 are
averaged over the five baselines (MSP, MaxLogit, ODIN, Energy, Entropy) of
each family.

Usage:
    python average_family_scores.py                 # embedded table
    python average_family_scores.py other_table.md  # same format, from a file
    python average_family_scores.py --exclude ODIN  # drop a baseline everywhere
"""
import argparse
import sys
from collections import OrderedDict

TABLE = """
| method   | AUROC | AP    | FPR@95 |
| -------- | ----- | ----- | ------ |
| MSP      | 87.76 | 18.06 | 45.76  |
| MaxLogit | 92.67 | 35.90 | 38.12  |
| ODIN     | 89.36 | 27.19 | 55.64  |
| Energy   | 92.88 | 35.55 | 37.94  |
| Entropy  | 89.38 | 26.12 | 44.90  |
| Group MSP      | 89.97 | 16.30 | 40.66  |
| Group MaxLogit | 92.67 | 35.90 | 38.12  |
| Group ODIN     | 24.43 |  1.11 | 96.08  |
| Group Energy   | 92.88 | 36.03 | 37.95  |
| Group Entropy  | 90.45 | 21.85 | 40.78  |
| GN MSP         | 92.07 | 19.81 | 39.89  |
| GN MaxLogit    | 93.75 | 36.28 | 30.36  |
| GN ODIN        | 70.95 |  8.60 | 94.82  |
| GN Energy      | 93.94 | 36.31 | 29.96  |
| GN Entropy     | 91.75 | 14.49 | 39.89  |
"""

METRICS = ('AUROC', 'AP', 'FPR@95')
FAMILIES = OrderedDict([('Normal', ''), ('Group', 'Group '), ('GN', 'GN ')])


def parse_table(text):
    """Return [(family, baseline, (auroc, ap, fpr95)), ...] from markdown rows."""
    rows = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != 4 or cells[0] in ('method', '') or set(cells[0]) <= {'-'}:
            continue
        name = cells[0]
        family, baseline = 'Normal', name
        for fam, prefix in FAMILIES.items():
            if prefix and name.startswith(prefix):
                family, baseline = fam, name[len(prefix):]
                break
        rows.append((family, baseline, tuple(float(c) for c in cells[1:])))
    return rows


def family_averages(rows, exclude=()):
    """Return OrderedDict family -> (n, [mean AUROC, mean AP, mean FPR@95])."""
    out = OrderedDict()
    for fam in FAMILIES:
        vals = [v for f, b, v in rows if f == fam and b not in exclude]
        if vals:
            out[fam] = (len(vals),
                        [sum(v[i] for v in vals) / len(vals)
                         for i in range(len(METRICS))])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('table', nargs='?', help='markdown table file (default: embedded)')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='baseline names to drop, e.g. --exclude ODIN')
    args = ap.parse_args()

    text = open(args.table).read() if args.table else TABLE
    rows = parse_table(text)
    if not rows:
        sys.exit('no table rows found')
    averages = family_averages(rows, exclude=set(args.exclude))

    base = averages.get('Normal')
    print(f'{"family":8} {"n":>2}  ' + '  '.join(f'{m:>7}' for m in METRICS)
          + '    (delta vs Normal)')
    for fam, (n, means) in averages.items():
        line = f'{fam:8} {n:>2}  ' + '  '.join(f'{x:7.2f}' for x in means)
        if base is not None and fam != 'Normal':
            line += '    ' + '  '.join(f'{x - b:+7.2f}' for x, b in zip(means, base[1]))
        print(line)
    if args.exclude:
        print(f'(excluded: {", ".join(args.exclude)})')


if __name__ == '__main__':
    main()
