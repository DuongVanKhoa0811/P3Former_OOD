"""Tests for tools/compare_occuq_grouping.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_compare_occuq_grouping.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.compare_occuq_grouping import (SWEEP_DIRS, compare,  # noqa: E402
                                          improvement, merge_rows,
                                          parse_rows, reference, verdict)

FLAT_ROWS = {'msp': (88.0, 18.0, 45.0), 'energy': (93.0, 36.0, 38.0),
             'entropy': (89.0, 27.0, 44.0)}  # R = (90.0, 27.0, 42.33)
# the five flat rows of the base model's runs (pe, and A's frozen model)
RUN_FLAT = dict(FLAT_ROWS, maxlogit=(93.0, 38.0, 37.0), odin=(91.0, 31.0, 41.0))
C_FLAT = {'msp': (89.0, 19.0, 44.0), 'maxlogit': (94.0, 39.0, 36.0),
          'odin': (92.0, 32.0, 40.0), 'energy': (94.0, 37.0, 37.0),
          'entropy': (90.0, 28.0, 43.0)}  # C's fine-tuned model
SPLITS = {  # group_msp, group_energy, group_entropy; the best split is last
    # I = 6.33, 13.33, 6.33: best group_energy
    'sB': ((91.0, 30.0, 40.0), (92.0, 35.0, 39.0), (91.0, 30.0, 40.0)),
    # I = 27.33, 16.33, 28.33: best group_entropy, averaged 24.00
    'sA': ((93.0, 39.0, 30.0), (93.0, 36.0, 38.0), (93.0, 40.0, 30.0)),
}
GROUP_KEYS = ('group_msp', 'group_energy', 'group_entropy')
# a different best split on every set: sB on cetran, sA on test, sC on
# test_cetran; I = a + p - f - 74.67 against R = (90.0, 27.0, 42.33).
# sB ties group_msp with group_entropy on cetran, as real splits do
SPLITS_PER_SET = {
    'cetran': {  # I: sA 27.33 16.33 28.33, sB 40.33 13.33 40.33, sC 5.33 8.33 5.33
        'sA': ((93.0, 39.0, 30.0), (93.0, 36.0, 38.0), (93.0, 40.0, 30.0)),
        'sB': ((95.0, 45.0, 25.0), (92.0, 35.0, 39.0), (95.0, 45.0, 25.0)),
        'sC': ((90.0, 30.0, 40.0), (91.0, 31.0, 39.0), (90.0, 30.0, 40.0)),
    },
    'test': {  # I: sA 31.33 16.33 32.33, sB 6.33 13.33 6.33, sC 14.33 8.33 15.33
        'sA': ((94.0, 41.0, 29.0), (93.0, 36.0, 38.0), (94.0, 42.0, 29.0)),
        'sB': ((91.0, 30.0, 40.0), (92.0, 35.0, 39.0), (91.0, 30.0, 40.0)),
        'sC': ((92.0, 34.0, 37.0), (91.0, 31.0, 39.0), (92.0, 35.0, 37.0)),
    },
    'test_cetran': {  # I: sA 25.33 15.33 26.33, sB 11.33 13.33 11.33,
        # sC 43.33 18.33 42.33
        'sA': ((93.0, 38.0, 31.0), (92.0, 36.0, 38.0), (93.0, 39.0, 31.0)),
        'sB': ((92.0, 32.0, 38.0), (92.0, 35.0, 39.0), (92.0, 32.0, 38.0)),
        'sC': ((96.0, 46.0, 24.0), (93.0, 37.0, 37.0), (96.0, 45.0, 24.0)),
    },
}
PREFIX = '2026/10/07 12:00:00 - mmengine - INFO - '


def _write_log(path, rows, prefix='', extra=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f'{prefix}    method |    AUROC |       AP |   FPR@95\n')
        for name, (a, p, f) in rows.items():
            fh.write(f'{prefix}{name:>10} | {a:8.2f} | {p:8.2f} | {f:8.2f}\n')
        fh.write(extra)


def _run(run_dir, set_name, key, density, pq=0.4619, flat=RUN_FLAT):
    rows = dict(flat, **{key: density})
    extra = (f'{PREFIX}Epoch(test) [10/10]    pq: {pq:.4f}  '
             'pq_dagger: 0.5000  miou: 0.4794\n')
    _write_log(os.path.join(run_dir, set_name, '20261007_120000',
                            '20261007_120000.log'), rows, PREFIX, extra)


def _layout(tmp, tsv_improvement=None, pe_flat=RUN_FLAT, splits=None):
    """splits: set -> {split: its three Group rows}; SPLITS on every set by
    default."""
    if splits is None:
        splits = {s: SPLITS for s in SWEEP_DIRS}
    grouping = os.path.join(tmp, 'dump')
    ref = reference(FLAT_ROWS, 'synthetic')
    header = ('name\tsize_A\tgroup_A\timprovement_cetran\timprovement_test\t'
              'improvement_test_cetran\tworst')
    lines = [header]
    for name in splits['cetran']:
        values = [sum(improvement(m, ref) for m in splits[s][name]) / 3
                  if tsv_improvement is None else tsv_improvement
                  for s in SWEEP_DIRS]
        lines.append(f'{name}\t1\tx' + ''.join(f'\t{v:.4f}' for v in values)
                     + f'\t{min(values):.4f}')
    os.makedirs(os.path.join(grouping, 'divided_mass'))
    with open(os.path.join(grouping, 'divided_mass', 'robust.tsv'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    for set_name, (bip, sing) in SWEEP_DIRS.items():
        # the last split in the bipartition log, every split in the
        # singleton log: the last one is in both
        names = list(splits[set_name])
        for sub, subset in ((bip, names[-1:]), (sing, names)):
            rows = dict(FLAT_ROWS)
            for name in subset:
                rows.update({f'{name}_{k}': m for k, m in
                             zip(GROUP_KEYS, splits[set_name][name])})
            _write_log(os.path.join(grouping, sub, 'bipartitions.log'), rows)
    pe = os.path.join(tmp, 'pe')
    a_dirs = [os.path.join(tmp, f'a{s}') for s in range(3)]
    c_dirs = [os.path.join(tmp, f'c{s}') for s in range(3)]
    for set_name in SWEEP_DIRS:
        _run(pe, set_name, 'density_pe', (92.0, 30.0, 40.0),  # I = 7.33
             flat=pe_flat)
        for d, auroc in zip(a_dirs, (94.0, 93.5, 94.5)):  # I = 31.0 +- 0.5
            _run(d, set_name, 'density', (auroc, 40.0, 28.33))
        for d, auroc in zip(c_dirs, (93.5, 92.5, 93.0)):  # around 28.33
            _run(d, set_name, 'density', (auroc, 40.0, 30.0), pq=0.4500,
                 flat=C_FLAT)
    return grouping, pe, a_dirs, c_dirs


def test_parse_rows_with_the_mmengine_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'x.log')
        rows = {'energy': (92.88, 35.55, 37.94),
                's0.1.3.5.10.16_group_msp': (94.04, 40.45, 28.52)}
        # a confusion-matrix row, as the panoptic metric logs it
        matrix = f'{PREFIX}2832\t|\t1260\t|\t3605\t|\t0\t|\t0\n'
        _write_log(path, rows, prefix=PREFIX, extra=matrix)
        assert parse_rows(path) == rows
    print('PASS test_parse_rows_with_the_mmengine_prefix')


def test_improvement_reproduces_the_spec_example():
    # s16 on test + Cetran, with the sweep's offline flat rows (spec)
    rows = {'msp': (87.76, 18.06, 45.76), 'energy': (92.88, 35.54, 37.95),
            'entropy': (89.38, 26.11, 44.91)}
    ref = reference(rows, 'test_cetran')
    values = [improvement(m, ref) for m in ((94.04, 40.45, 28.52),
                                            (92.90, 36.40, 37.97),
                                            (94.06, 40.45, 28.43))]
    for got, want in zip(values, (32.27, 17.63, 32.38)):
        assert abs(got - want) < 0.01, (got, want)
    assert abs(sum(values) / 3 - 27.42) < 0.01
    print('PASS test_improvement_reproduces_the_spec_example')


def test_verdict_rule():
    assert verdict([30.0, 31.0, 32.0], best=29.0) == 'better'
    assert verdict([30.0, 31.0, 32.0], best=33.0) == 'worse'
    assert verdict([30.0, 31.0, 32.0], best=31.0) == 'too close to call'
    assert verdict([40.0], best=31.0) == 'better'
    print('PASS test_verdict_rule')


def test_merge_rows_rejects_conflicts():
    a = {'s16_group_msp': (94.04, 40.45, 28.52)}
    assert merge_rows(a, dict(a), 'x') == a
    try:
        merge_rows(a, {'s16_group_msp': (94.04, 40.95, 28.52)}, 'x')
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting duplicate rows were accepted')
    print('PASS test_merge_rows_rejects_conflicts')


def test_compare_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        report = compare(*_layout(tmp))
    assert '| pe | worse | worse | worse |' in report
    assert '| A | better | better | better |' in report
    assert '| C | too close to call | too close to call | too close to call |' in report
    assert 'PQ 45.00' in report and 'base model PQ 46.19' in report
    assert 'own flat Energy 94.00 | 37.00 | 37.00' in report
    print('PASS test_compare_end_to_end')


def test_compare_lists_every_flat_score_of_the_base_model():
    # R = (90.0, 27.0, 42.33); the splits' values are sA 28.33 and sB 13.33
    with tempfile.TemporaryDirectory() as tmp:
        report = compare(*_layout(tmp))
    for row in (
            '| flat MSP (base model) | 88.00 | 18.00 | 45.00 | -13.67 | | 0/2 | |',
            '| flat MaxLogit (base model) | 93.00 | 38.00 | 37.00 | 19.33 | | 1/2 | |',
            '| flat ODIN (base model) | 91.00 | 31.00 | 41.00 | 6.33 | | 0/2 | |',
            '| flat Energy (base model) | 93.00 | 36.00 | 38.00 | 16.33 | | 1/2 | |',
            '| flat Entropy (base model) | 89.00 | 27.00 | 44.00 | -2.67 | | 0/2 | |'):
        assert report.count(row) == 3, row  # once per set
    print('PASS test_compare_lists_every_flat_score_of_the_base_model')


def test_compare_lists_the_three_group_scores_of_the_best_split():
    # sA is the best split (28.33 > 13.33); its averaged improvement is the
    # mean of its three I, (27.33 + 16.33 + 28.33) / 3 = 24.00
    with tempfile.TemporaryDirectory() as tmp:
        report = compare(*_layout(tmp))
    for row in (
            '| best robust split `sA`: group_msp | 93.00 | 39.00 | 30.00 | 27.33 | 24.00 | | |',
            '| best robust split `sA`: group_energy | 93.00 | 36.00 | 38.00 | 16.33 | 24.00 | | |',
            '| best robust split `sA`: group_entropy (best) | 93.00 | 40.00 | 30.00 | 28.33 | 24.00 | | |'):
        assert report.count(row) == 3, row  # once per set
    assert '`sB`' not in report
    print('PASS test_compare_lists_the_three_group_scores_of_the_best_split')


def test_compare_reports_each_sets_best_split_on_every_set():
    # bests: sB on Cetran (40.33, group_msp before its tied group_entropy),
    # sA on test (32.33, group_entropy), sC on test + Cetran (43.33,
    # group_msp). Each table lists the reported set's own best split first,
    # with that set's rows, averaged improvement and splits beaten; (best) is
    # the best score there, (chosen) the score another set chose its split by
    header = ('| method | AUROC | AP | FPR@95 | I (like-for-like) | '
              'averaged improvement | splits beaten |\n'
              '| --- | --- | --- | --- | --- | --- | --- |\n')
    tables = [
        '### Reported on Cetran\n\n' + header +
        '| best robust split of Cetran `sB`: group_msp (best) | 95.00 | 45.00 | 25.00 | 40.33 | 31.33 | 2/3 |\n'
        '| best robust split of Cetran `sB`: group_energy | 92.00 | 35.00 | 39.00 | 13.33 | 31.33 | 1/3 |\n'
        '| best robust split of Cetran `sB`: group_entropy | 95.00 | 45.00 | 25.00 | 40.33 | 31.33 | 2/3 |\n'
        '| best robust split of test `sA`: group_msp | 93.00 | 39.00 | 30.00 | 27.33 | 24.00 | 1/3 |\n'
        '| best robust split of test `sA`: group_energy | 93.00 | 36.00 | 38.00 | 16.33 | 24.00 | 1/3 |\n'
        '| best robust split of test `sA`: group_entropy (best, chosen) | 93.00 | 40.00 | 30.00 | 28.33 | 24.00 | 1/3 |\n'
        '| best robust split of test + Cetran `sC`: group_msp (chosen) | 90.00 | 30.00 | 40.00 | 5.33 | 6.33 | 0/3 |\n'
        '| best robust split of test + Cetran `sC`: group_energy (best) | 91.00 | 31.00 | 39.00 | 8.33 | 6.33 | 0/3 |\n'
        '| best robust split of test + Cetran `sC`: group_entropy | 90.00 | 30.00 | 40.00 | 5.33 | 6.33 | 0/3 |\n',
        '### Reported on test\n\n' + header +
        '| best robust split of test `sA`: group_msp | 94.00 | 41.00 | 29.00 | 31.33 | 26.67 | 2/3 |\n'
        '| best robust split of test `sA`: group_energy | 93.00 | 36.00 | 38.00 | 16.33 | 26.67 | 2/3 |\n'
        '| best robust split of test `sA`: group_entropy (best) | 94.00 | 42.00 | 29.00 | 32.33 | 26.67 | 2/3 |\n'
        '| best robust split of Cetran `sB`: group_msp (chosen) | 91.00 | 30.00 | 40.00 | 6.33 | 8.67 | 0/3 |\n'
        '| best robust split of Cetran `sB`: group_energy (best) | 92.00 | 35.00 | 39.00 | 13.33 | 8.67 | 0/3 |\n'
        '| best robust split of Cetran `sB`: group_entropy | 91.00 | 30.00 | 40.00 | 6.33 | 8.67 | 0/3 |\n'
        '| best robust split of test + Cetran `sC`: group_msp (chosen) | 92.00 | 34.00 | 37.00 | 14.33 | 12.67 | 1/3 |\n'
        '| best robust split of test + Cetran `sC`: group_energy | 91.00 | 31.00 | 39.00 | 8.33 | 12.67 | 0/3 |\n'
        '| best robust split of test + Cetran `sC`: group_entropy (best) | 92.00 | 35.00 | 37.00 | 15.33 | 12.67 | 1/3 |\n',
        '### Reported on test + Cetran\n\n' + header +
        '| best robust split of test + Cetran `sC`: group_msp (best) | 96.00 | 46.00 | 24.00 | 43.33 | 34.67 | 2/3 |\n'
        '| best robust split of test + Cetran `sC`: group_energy | 93.00 | 37.00 | 37.00 | 18.33 | 34.67 | 1/3 |\n'
        '| best robust split of test + Cetran `sC`: group_entropy | 96.00 | 45.00 | 24.00 | 42.33 | 34.67 | 2/3 |\n'
        '| best robust split of Cetran `sB`: group_msp (chosen) | 92.00 | 32.00 | 38.00 | 11.33 | 12.00 | 0/3 |\n'
        '| best robust split of Cetran `sB`: group_energy (best) | 92.00 | 35.00 | 39.00 | 13.33 | 12.00 | 0/3 |\n'
        '| best robust split of Cetran `sB`: group_entropy | 92.00 | 32.00 | 38.00 | 11.33 | 12.00 | 0/3 |\n'
        '| best robust split of test `sA`: group_msp | 93.00 | 38.00 | 31.00 | 25.33 | 22.33 | 1/3 |\n'
        '| best robust split of test `sA`: group_energy | 92.00 | 36.00 | 38.00 | 15.33 | 22.33 | 1/3 |\n'
        '| best robust split of test `sA`: group_entropy (best, chosen) | 93.00 | 39.00 | 31.00 | 26.33 | 22.33 | 1/3 |\n',
    ]
    with tempfile.TemporaryDirectory() as tmp:
        report = compare(*_layout(tmp, splits=SPLITS_PER_SET))
    heading = '\n## Best robust splits across sets\n'
    assert heading in report, 'no section for the best splits across sets'
    section = report.index(heading)
    assert (report.index('\n## test + Cetran\n') < section
            < report.index('\n## Caveats\n'))
    # the three tables, in order and with nothing else, end the section
    start = report.index('\n### Reported on Cetran\n', section) + 1
    assert report[start:report.index('\n## Caveats\n')] == '\n'.join(tables)
    print('PASS test_compare_reports_each_sets_best_split_on_every_set')


def test_compare_marks_the_chosen_score_when_sets_share_their_best_split():
    # sA is every set's best split, by group_entropy, as s16 is on test and
    # test + Cetran: (chosen) still marks it on the two other sets
    with tempfile.TemporaryDirectory() as tmp:
        report = compare(*_layout(tmp))
    for source in ('Cetran', 'test', 'test + Cetran'):
        label = f'| best robust split of {source} `sA`: group_entropy'
        values = '| 93.00 | 40.00 | 30.00 | 28.33 | 24.00 | 1/2 |'
        assert report.count(f'{label} (best, chosen) {values}') == 2, source
        assert report.count(f'{label} (best) {values}') == 1, source
    print('PASS test_compare_marks_the_chosen_score_when_sets_share_their_'
          'best_split')


def test_compare_rejects_a_run_without_a_flat_row():
    pe_flat = {k: m for k, m in RUN_FLAT.items() if k != 'odin'}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            compare(*_layout(tmp, pe_flat=pe_flat))
        except KeyError as err:
            assert 'odin' in str(err)
        else:
            raise AssertionError('a run log without the odin row was accepted')
    print('PASS test_compare_rejects_a_run_without_a_flat_row')


def test_compare_rejects_a_robust_tsv_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            compare(*_layout(tmp, tsv_improvement=99.0))
        except ValueError as err:
            assert 'robust.tsv' in str(err)
        else:
            raise AssertionError('a robust.tsv mismatch was accepted')
    print('PASS test_compare_rejects_a_robust_tsv_mismatch')


if __name__ == '__main__':
    test_parse_rows_with_the_mmengine_prefix()
    test_improvement_reproduces_the_spec_example()
    test_verdict_rule()
    test_merge_rows_rejects_conflicts()
    test_compare_end_to_end()
    test_compare_lists_every_flat_score_of_the_base_model()
    test_compare_lists_the_three_group_scores_of_the_best_split()
    test_compare_reports_each_sets_best_split_on_every_set()
    test_compare_marks_the_chosen_score_when_sets_share_their_best_split()
    test_compare_rejects_a_run_without_a_flat_row()
    test_compare_rejects_a_robust_tsv_mismatch()
    print('ALL TESTS PASSED')
