"""Smoke test for tools/plot_ood_class_resemblance.py (synthetic tables).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_ood_class_resemblance.py
"""
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import ood_class_resemblance as res  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402


def test_figures_are_written():
    rng = np.random.RandomState(0)
    subsets = sb.singleton_subsets() + [(12, 16), (0, 1, 2, 3, 4)]
    with tempfile.TemporaryDirectory() as tmp:
        rho = []
        for key in res.SETS:
            for space in res.SPACES:
                shares = 100.0 * rng.dirichlet(np.ones(sb.NUM_CLASSES))
                dm.write_tsv(os.path.join(tmp, f'profile_{key}_{space}.tsv'), [
                    OrderedDict([
                        ('class', name), ('bank', 40), ('queries', 20),
                        ('r_ood', float(shares[c])),
                        ('r_id', float(rng.uniform(0, 8))),
                        ('contrast', float(rng.uniform(0, 5))),
                        ('feat_div_ood', float(rng.uniform(0.1, 60))),
                        ('feat_div_id', 0.0 if c == 3 else float(rng.uniform(0.1, 10))),
                        ('ood_div', float(rng.uniform(0, 50))),
                        ('id_div', float(rng.uniform(0, 5))),
                        ('improvement', float(rng.uniform(-40, 35)))])
                    for c, name in enumerate(sb.CLASSES)])
                dm.write_tsv(os.path.join(tmp, f'splits_{key}_{space}.tsv'), [
                    OrderedDict([
                        ('name', sb.partition_name(s)), ('size_A', len(s)),
                        ('group_A', 'x'), ('R_A', float(rng.uniform(0, 60))),
                        ('E_A', float(rng.uniform(0, 5))),
                        ('feat_div_ood', float(rng.uniform(0, 60))),
                        ('feat_div_id', float(rng.uniform(0, 10))),
                        ('feat_log_ratio', float(rng.uniform(-1, 2))),
                        ('ood_div', 1.0), ('id_div', 1.0),
                        ('improvement', float(rng.uniform(-40, 35))),
                        ('robust', int(s == (16, )))]) for s in subsets])
                for population, pairs in (('classes', res.CLASS_PAIRS),
                                          ('splits', res.SPLIT_PAIRS)):
                    rho += [OrderedDict([
                        ('set', key), ('space', space), ('k', 10),
                        ('population', population), ('x', x), ('y', y),
                        ('rho', float(rng.uniform(-1, 1))), ('n', 24)])
                        for x, y in pairs]
        dm.write_tsv(os.path.join(tmp, 'rho.tsv'), rho)
        script = os.path.join(_REPO_ROOT, 'tools',
                              'plot_ood_class_resemblance.py')
        for space in res.SPACES:
            proc = subprocess.run([sys.executable, script, tmp, '--space',
                                   space], capture_output=True, text=True,
                                  cwd=_REPO_ROOT)
            assert proc.returncode == 0, proc.stderr
            for name in ('profile', 'bubble_feature_singletons', 'hypothesis'):
                for ext in ('.pdf', '.png'):
                    path = os.path.join(tmp, f'{name}_{space}{ext}')
                    assert os.path.getsize(path) > 1000, path
    print('test_figures_are_written passed')


if __name__ == '__main__':
    test_figures_are_written()
    print('ALL TESTS PASSED')
