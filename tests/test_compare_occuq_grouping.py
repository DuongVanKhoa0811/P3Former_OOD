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
SPLITS = {  # group_msp, group_energy, group_entropy
    'sA': ((93.0, 40.0, 30.0), (93.0, 36.0, 38.0), (93.0, 40.0, 30.0)),
    'sB': ((91.0, 30.0, 40.0), (93.0, 36.0, 38.0), (91.0, 30.0, 40.0)),
}
GROUP_KEYS = ('group_msp', 'group_energy', 'group_entropy')
PREFIX = '2026/10/07 12:00:00 - mmengine - INFO - '


def _write_log(path, rows, prefix='', extra=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f'{prefix}    method |    AUROC |       AP |   FPR@95\n')
        for name, (a, p, f) in rows.items():
            fh.write(f'{prefix}{name:>10} | {a:8.2f} | {p:8.2f} | {f:8.2f}\n')
        fh.write(extra)


def _run(run_dir, set_name, key, density, pq=0.4619):
    rows = dict(FLAT_ROWS, **{key: density})
    extra = (f'{PREFIX}Epoch(test) [10/10]    pq: {pq:.4f}  '
             'pq_dagger: 0.5000  miou: 0.4794\n')
    _write_log(os.path.join(run_dir, set_name, '20261007_120000',
                            '20261007_120000.log'), rows, PREFIX, extra)


def _layout(tmp, tsv_improvement=None):
    grouping = os.path.join(tmp, 'dump')
    ref = reference(FLAT_ROWS, 'synthetic')
    header = ('name\tsize_A\tgroup_A\timprovement_cetran\timprovement_test\t'
              'improvement_test_cetran\tworst')
    lines = [header]
    for name, scores in SPLITS.items():
        value = (sum(improvement(m, ref) for m in scores) / 3
                 if tsv_improvement is None else tsv_improvement)
        lines.append(f'{name}\t1\tx' + f'\t{value:.4f}' * 4)
    os.makedirs(os.path.join(grouping, 'divided_mass'))
    with open(os.path.join(grouping, 'divided_mass', 'robust.tsv'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    for bip, sing in SWEEP_DIRS.values():
        # sA in the bipartition log, sB in the singleton log, sA in both
        for sub, names in ((bip, ('sA', )), (sing, ('sA', 'sB'))):
            rows = dict(FLAT_ROWS)
            for name in names:
                rows.update({f'{name}_{k}': m
                             for k, m in zip(GROUP_KEYS, SPLITS[name])})
            _write_log(os.path.join(grouping, sub, 'bipartitions.log'), rows)
    pe = os.path.join(tmp, 'pe')
    a_dirs = [os.path.join(tmp, f'a{s}') for s in range(3)]
    c_dirs = [os.path.join(tmp, f'c{s}') for s in range(3)]
    for set_name in SWEEP_DIRS:
        _run(pe, set_name, 'density_pe', (92.0, 30.0, 40.0))  # I = 7.33
        for d, auroc in zip(a_dirs, (94.0, 93.5, 94.5)):  # I = 31.0 +- 0.5
            _run(d, set_name, 'density', (auroc, 40.0, 28.33))
        for d, auroc in zip(c_dirs, (93.5, 92.5, 93.0)):  # around 28.33
            _run(d, set_name, 'density', (auroc, 40.0, 30.0), pq=0.4500)
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
    assert '`sA` (group_msp)' in report
    assert 'PQ 45.00' in report and 'base model PQ 46.19' in report
    print('PASS test_compare_end_to_end')


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
    test_compare_rejects_a_robust_tsv_mismatch()
    print('ALL TESTS PASSED')
