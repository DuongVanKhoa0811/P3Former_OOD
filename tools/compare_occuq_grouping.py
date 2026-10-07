#!/usr/bin/env python
"""Compare OCCUQ density scores with the robust grouping splits.

Reads logs only:
- grouping's 33 robust splits (divided_mass/robust.tsv), with their
  per-score rows from the bipartition-sweep logs;
- the test.py logs of the OCCUQ runs.

Every score x is reduced to a single-score improvement against the mean R
of the flat MSP, Energy and Entropy rows of its estimator:
    I(x) = (AUROC_x - R_AUROC) + (AP_x - R_AP) - (FPR@95_x - R_FPR@95)
R comes from the sweep logs for grouping, and from the base model's online
flat rows (the pe runs) for every OCCUQ variant.

A split counts with its best single Group score (like-for-like).
robust.tsv's averaged improvement is shown alongside and must be reproduced
by the rows. Verdict per variant and set, against the best split:
- better if every run is above it;
- worse if every run is below it;
- too close to call otherwise.

Spec: docs/superpowers/specs/2026-10-07-occuq-density-design.md.

Run from the repo root:
    python tools/compare_occuq_grouping.py \\
        --pe work_dirs/p3former_2xb1_3x_dso_occuq_pe \\
        --a work_dirs/p3former_2xb1_3x_dso_occuq_a/seed{0,1,2} \\
        --c work_dirs/p3former_2xb1_3x_dso_occuq_c/seed{0,1,2} \\
        --out work_dirs/p3former_2xb1_3x_dso_occuq_compare/occuq_vs_grouping.md
"""
import argparse
import csv
import glob
import os
import os.path as osp
import re
import statistics
from typing import Dict, List, Sequence, Tuple

Metrics = Tuple[float, float, float]  # AUROC, AP, FPR@95 in %

SETS = ('cetran', 'test', 'test_cetran')
SET_TITLES = {'cetran': 'Cetran', 'test': 'test', 'test_cetran': 'test + Cetran'}
SWEEP_DIRS = {
    'cetran': ('bipartitions', 'singletons'),
    'test': ('bipartitions_test', 'singletons_test'),
    'test_cetran': ('bipartitions_test_cetran', 'singletons_test_cetran'),
}
FLAT = ('msp', 'energy', 'entropy')
GROUP_SCORES = ('group_msp', 'group_energy', 'group_entropy')
NUMBER = r'(-?\d+(?:\.\d+)?)'
# a name starting with a letter, so the numeric rows the panoptic metric
# also logs with '|' separators (a confusion matrix) never match
ROW = re.compile(
    rf'([A-Za-z_][\w.]*)\s*\|\s*{NUMBER}\s*\|\s*{NUMBER}\s*\|\s*{NUMBER}\s*$')
EPOCH_TEST = re.compile(r'Epoch\(test\).*?\bpq: ([\d.]+).*?\bmiou: ([\d.]+)')
CONSISTENCY_TOL = 0.02
DUPLICATE_TOL = 0.011  # the rows are printed with 2 decimals


def parse_rows(path: str) -> Dict[str, Metrics]:
    """The ``method | AUROC | AP | FPR@95`` rows of a test.py or sweep log."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            m = ROW.search(line)
            if m:
                rows[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                    float(m.group(4)))
    return rows


def merge_rows(a: Dict[str, Metrics], b: Dict[str, Metrics],
               where: str) -> Dict[str, Metrics]:
    """Union of two row dicts; a name present in both must agree."""
    out = dict(a)
    for name, metrics in b.items():
        if name in out and max(
                abs(x - y) for x, y in zip(out[name], metrics)) > DUPLICATE_TOL:
            raise ValueError(f'{name} differs between the logs of {where}: '
                             f'{out[name]} vs {metrics}')
        out[name] = metrics
    return out


def reference(rows: Dict[str, Metrics], where: str) -> Metrics:
    """R: the mean of the flat MSP, Energy and Entropy rows."""
    missing = [k for k in FLAT if k not in rows]
    if missing:
        raise KeyError(f'flat rows {missing} are missing in {where}')
    return tuple(
        sum(rows[k][i] for k in FLAT) / len(FLAT) for i in range(3))


def improvement(m: Metrics, ref: Metrics) -> float:
    """I = (AUROC - R_AUROC) + (AP - R_AP) - (FPR@95 - R_FPR@95)."""
    return (m[0] - ref[0]) + (m[1] - ref[1]) - (m[2] - ref[2])


def read_robust(path: str) -> List[Dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def grouping_table(robust, rows, set_name) -> List[Dict]:
    """Per robust split on ``set_name``: its best single Group score and
    like-for-like value, and robust.tsv's averaged improvement, which the
    rows must reproduce."""
    ref = reference(rows, f'the sweep logs of {set_name}')
    table = []
    for split in robust:
        name = split['name']
        scores = {}
        for key in GROUP_SCORES:
            row = f'{name}_{key}'
            if row not in rows:
                raise KeyError(f'{row} is missing from the sweep logs of '
                               f'{set_name}')
            scores[key] = rows[row]
        values = {k: improvement(m, ref) for k, m in scores.items()}
        averaged = sum(values.values()) / len(values)
        logged = float(split[f'improvement_{set_name}'])
        if abs(averaged - logged) > CONSISTENCY_TOL:
            raise ValueError(f'{name} on {set_name}: the rows give an '
                             f'averaged improvement of {averaged:.4f}, '
                             f'robust.tsv {logged:.4f}')
        best = max(values, key=values.get)
        table.append(dict(name=name, best_score=best, metrics=scores[best],
                          value=values[best], averaged=logged))
    return table


def _has_rows(path: str) -> bool:
    with open(path) as fh:
        return any(ROW.search(line) for line in fh)


def find_log(run_dir: str, set_name: str) -> str:
    """The newest test.py log with an OOD table under run_dir/set_name."""
    logs = [p for p in sorted(glob.glob(osp.join(run_dir, set_name, '*',
                                                 '*.log'))) if _has_rows(p)]
    if not logs:
        raise FileNotFoundError(f'no test.py log with an OOD table in '
                                f'{run_dir}/{set_name}/*/')
    return logs[-1]  # timestamp directories sort chronologically


def occuq_run(run_dir: str, set_name: str, key: str) -> Dict:
    """One OCCUQ evaluation: its density row, flat Energy row, flat
    reference, and PQ and mIoU in %."""
    path = find_log(run_dir, set_name)
    rows = parse_rows(path)
    if key not in rows:
        raise KeyError(f'the {key} row is missing in {path}')
    pq = miou = None
    with open(path) as fh:
        for line in fh:
            m = EPOCH_TEST.search(line)
            if m:
                pq, miou = 100 * float(m.group(1)), 100 * float(m.group(2))
    return dict(path=path, density=rows[key], energy=rows['energy'],
                reference=reference(rows, path), pq=pq, miou=miou)


def verdict(values: Sequence[float], best: float) -> str:
    """better if every value is above ``best``, worse if every value is
    below it, too close to call otherwise."""
    if all(v > best for v in values):
        return 'better'
    if all(v < best for v in values):
        return 'worse'
    return 'too close to call'


def rank(value: float, table: List[Dict]) -> int:
    """How many robust splits ``value`` beats (like-for-like)."""
    return sum(value > row['value'] for row in table)


def _fmt(m: Metrics) -> str:
    return f'{m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f}'


def _mean(runs, key) -> Metrics:
    return tuple(statistics.mean(r[key][i] for r in runs) for i in range(3))


def render_set(set_name, table, pe, variants, ref) -> Tuple[str, Dict]:
    """The markdown section of one set, and its verdict per variant."""
    best = max(table, key=lambda row: row['value'])
    n = len(table)
    lines = [
        f'## {SET_TITLES[set_name]}', '',
        "R (mean of the base model's flat MSP, Energy and Entropy): "
        f'{ref[0]:.2f} / {ref[1]:.2f} / {ref[2]:.2f}', '',
        '| method | AUROC | AP | FPR@95 | I (like-for-like) | averaged '
        'improvement | splits beaten | verdict |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |'
    ]
    energy_i = improvement(pe['energy'], ref)
    lines.append(f"| flat Energy (base model) | {_fmt(pe['energy'])} | "
                 f'{energy_i:.2f} | | {rank(energy_i, table)}/{n} | |')
    lines.append(f"| best robust split `{best['name']}` "
                 f"({best['best_score']}) | {_fmt(best['metrics'])} | "
                 f"{best['value']:.2f} | {best['averaged']:.2f} | | |")
    verdicts = {}
    for label, runs in variants.items():
        values = [improvement(r['density'], ref) for r in runs]
        mean = statistics.mean(values)
        spread = (f' [{min(values):.2f}, {max(values):.2f}]'
                  if len(values) > 1 else '')
        verdicts[label] = verdict(values, best['value'])
        lines.append(f'| OCCUQ {label} ({len(runs)} run'
                     f'{"s" if len(runs) > 1 else ""}) | '
                     f"{_fmt(_mean(runs, 'density'))} | {mean:.2f}{spread} | "
                     f'| {rank(mean, table)}/{n} | {verdicts[label]} |')
    c_runs = variants['C']
    if all(r['pq'] is not None for r in c_runs) and pe['pq'] is not None:
        lines += [
            '', f"C after fine-tuning (mean over runs): PQ "
            f"{statistics.mean(r['pq'] for r in c_runs):.2f}, mIoU "
            f"{statistics.mean(r['miou'] for r in c_runs):.2f}, own flat "
            f"Energy {_fmt(_mean(c_runs, 'energy'))}; base model PQ "
            f"{pe['pq']:.2f}, mIoU {pe['miou']:.2f}."
        ]
    return '\n'.join(lines), verdicts


def compare(grouping_dir: str, pe_dir: str, a_dirs: Sequence[str],
            c_dirs: Sequence[str]) -> str:
    """The markdown report."""
    robust = read_robust(osp.join(grouping_dir, 'divided_mass', 'robust.tsv'))
    sections, summary = [], {}
    for set_name in SETS:
        rows = {}
        for sub in SWEEP_DIRS[set_name]:
            rows = merge_rows(
                rows, parse_rows(osp.join(grouping_dir, sub,
                                          'bipartitions.log')), set_name)
        table = grouping_table(robust, rows, set_name)
        pe = occuq_run(pe_dir, set_name, 'density_pe')
        variants = {
            'pe': [pe],
            'A': [occuq_run(d, set_name, 'density') for d in a_dirs],
            'C': [occuq_run(d, set_name, 'density') for d in c_dirs],
        }
        text, summary[set_name] = render_set(set_name, table, pe, variants,
                                             pe['reference'])
        sections.append(text)
    head = [
        '# OCCUQ vs the robust grouping splits', '',
        'Like-for-like single-score improvement I = (AUROC - R_AUROC) + '
        '(AP - R_AP) - (FPR@95 - R_FPR@95); a split counts with its best '
        'Group score. Verdict against the best robust split: better if '
        'every run is above it, worse if every run is below it, too close '
        'to call otherwise.', '',
        '| variant | ' + ' | '.join(SET_TITLES[s] for s in SETS) + ' |',
        '| --- |' + ' --- |' * len(SETS)
    ]
    for label in ('pe', 'A', 'C'):
        head.append(f'| {label} | ' +
                    ' | '.join(summary[s][label] for s in SETS) + ' |')
    tail = [
        '## Caveats', '',
        '- The robust splits were selected on these same sets, and their '
        'numbers are offline (bipartition sweep); some of their FPR@95 values '
        'are limited by float32 ties (divided_mass/summary.md).',
        "- OCCUQ's settings were fixed in advance, and its numbers come from "
        'the online metric.', '- Both points favour grouping.'
    ]
    return '\n\n'.join(['\n'.join(head)] + sections + ['\n'.join(tail)]) + '\n'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grouping-dir',
                    default='work_dirs/p3former_2xb1_3x_dso_ood_dump',
                    help='holds divided_mass/robust.tsv and the sweep logs')
    ap.add_argument('--pe', required=True,
                    help='run dir of the pe evaluations (<set>/ subdirs)')
    ap.add_argument('--a', nargs='+', required=True,
                    help='run dirs of the variant A seeds')
    ap.add_argument('--c', nargs='+', required=True,
                    help='run dirs of the variant C seeds')
    ap.add_argument('--out', required=True, help='markdown report')
    args = ap.parse_args()
    report = compare(args.grouping_dir, args.pe, args.a, args.c)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(report)
    print(report)


if __name__ == '__main__':
    main()
