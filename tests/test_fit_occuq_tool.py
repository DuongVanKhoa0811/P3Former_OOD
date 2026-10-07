"""Tests for tools/fit_occuq_gmm.py without a model: the --check
comparison, the float32 guard, the report, the running-sums file, and
fit() itself on a fake model.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_fit_occuq_tool.py
"""
import contextlib
import io
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.fit_occuq_gmm as tool  # noqa: E402
from p3former.utils.gmm_fit import GaussianStats, finalize  # noqa: E402
from tools.fit_occuq_gmm import (CHECK_TOL, check_float32_finite,  # noqa: E402
                                 compare_log_densities, load_sums, report,
                                 save_sums, sums_path)


def _stats(dim=4, dead=(), n=600, seed=0):
    """Running sums of 3 classes plus ignore (label 3); the ``dead``
    dimensions are 0 everywhere."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 4, (n, ), generator=g)
    feats = torch.randn(n, dim, generator=g) * 2 + labels[:, None].float()
    feats[:, list(dead)] = 0.0
    stats = GaussianStats(3, dim)
    stats.update(feats, labels, ignore_index=3)
    return stats


def test_check_rejects_non_finite_log_densities():
    good = torch.linspace(-50.0, -10.0, 8, dtype=torch.float64)
    for bad in (float('nan'), float('inf'), float('-inf')):
        broken = good.clone()
        broken[3] = bad
        for fast, exact, which, other in ((broken, good, 'float32', 'float64'),
                                          (good, broken, 'float64', 'float32')):
            try:
                compare_log_densities(fast, exact, frame='frame 7')
            except FloatingPointError as err:
                msg = str(err)
                assert 'frame 7' in msg and which in msg, msg
                assert other not in msg, msg
            else:
                raise AssertionError(f'{bad} in the {which} log-density '
                                     'passed')
    print('PASS test_check_rejects_non_finite_log_densities')


def test_check_tolerance():
    exact = torch.linspace(-50.0, -10.0, 8, dtype=torch.float64)
    assert abs(compare_log_densities(exact + 0.04, exact) - 0.04) < 1e-9
    assert compare_log_densities(exact[:0], exact[:0]) == 0.0  # no voxels
    try:
        compare_log_densities(exact + CHECK_TOL + 0.01, exact, frame='frame 2')
    except RuntimeError as err:
        assert 'frame 2' in str(err), str(err)
    else:
        raise AssertionError(f'a difference above {CHECK_TOL} passed')
    print('PASS test_check_tolerance')


def test_non_finite_float32_gaussian_names_the_class():
    gmm = finalize(_stats(), min_count=10)
    gmm['class_names'] = ['car', 'road', 'pole']
    check_float32_finite(gmm)  # a clean fit passes
    gmm['prec_chol'][1, 0, 0] = 1e200  # finite in float64, inf in float32
    try:
        check_float32_finite(gmm)
    except FloatingPointError as err:
        msg = str(err)
        assert '1 (road)' in msg and 'prec_chol' in msg, msg
        assert 'car' not in msg and 'pole' not in msg, msg
        assert f'ridge {float(gmm["ridge"]):.3e}' in msg, msg
        assert 'shared jitter 0' in msg, msg
    else:
        raise AssertionError('a float32 inf was accepted')
    print('PASS test_non_finite_float32_gaussian_names_the_class')


def test_report_prints_the_regularisation():
    gmm = finalize(_stats(dim=6, dead=(1, 4)), min_count=10)
    gmm['class_names'] = ['car', 'road', 'pole']
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report(gmm)
    text = out.getvalue()
    mean = float(gmm['pooled_var'].mean())
    assert (f'ridge {float(gmm["ridge"]):.3e} = 1e-06 x the mean pooled '
            f'variance {mean:.4g}; shared jitter 0') in text, text
    assert '2 dimensions exactly 0: [1, 4]' in text, text
    assert '2 below 1e-12 x the mean' in text, text
    print('PASS test_report_prints_the_regularisation')


def test_sums_round_trip_reproduces_finalize():
    stats = _stats()
    correct = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64)
    total = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
    meta = dict(features='head', fingerprint='abc123', checkpoint='e9.pth',
                config='cfg.py', ann_file='dso_infos_train.pkl', frames=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = sums_path(os.path.join(tmp, 'gmm_head.pth'))
        assert path == os.path.join(tmp, 'gmm_head.sums.pth')
        save_sums(path, stats, correct, total, **meta)
        raw = torch.load(path)  # plain tensors and metadata, no objects
        for key in ('count', 'sum', 'outer', 'correct', 'total'):
            assert raw[key].dtype == torch.float64, key
            assert raw[key].device.type == 'cpu', key
        assert all(isinstance(v, (torch.Tensor, str, int))
                   for v in raw.values())
        loaded, data = load_sums(path, 'head', 'abc123')
        expected, got = finalize(stats, 10), finalize(loaded, 10)
        assert set(expected) == set(got)
        for key in expected:
            assert torch.equal(expected[key], got[key]), key
        assert torch.equal(data['correct'], correct)
        assert torch.equal(data['total'], total)
        for key, value in meta.items():
            assert data[key] == value, key
        for features, fp in (('pe', 'abc123'), ('head', 'other')):
            try:
                load_sums(path, features, fp)
            except ValueError:
                pass
            else:
                raise AssertionError(f'accepted features={features}, '
                                     f'fingerprint={fp}')
        try:
            save_sums(path, stats, correct, total, **meta)
        except FileExistsError:
            pass
        else:
            raise AssertionError('overwrote a sums file')
    print('PASS test_sums_round_trip_reproduces_finalize')


def _fake_loader(frames=3, dim=6, per_class=900, seed=0):
    """Batch-size-1 frames whose voxels cover the 24 classes, MIN_COUNT
    times in total, and ignore."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(tool.NUM_CLASSES + 1).repeat_interleave(per_class)
    loader = []
    for i in range(frames):
        loader.append(dict(
            feats=torch.randn(len(labels), dim, generator=g) +
            0.1 * labels[:, None].float(),
            labels=labels,
            logits=torch.randn(len(labels), tool.NUM_CLASSES + 1, generator=g),
            data_samples=[SimpleNamespace(
                metainfo=dict(lidar_path=f'seq/{i:06d}.ply'))]))
    return loader


def test_fit_saves_the_sums_first_and_finalises_them_with_from_sums():
    assert 3 * 900 >= tool.MIN_COUNT
    loader, frames_seen = _fake_loader(), []

    class Head:
        occuq_head = torch.nn.Identity()

        @staticmethod
        def density_fingerprint(features):
            return f'weights-{features}'

    class Model:
        decode_head = Head()

        def to(self, device):
            return self

        def eval(self):
            return self

    def frame_voxels(model, data, features):
        frames_seen.append(data['data_samples'][0].metainfo['lidar_path'])
        return data['feats'], data['labels'], data['logits']

    names = [f'class{c}' for c in range(tool.NUM_CLASSES)]
    real = (tool.build, tool.load_weights, tool.frame_voxels)
    tool.build = lambda config, ann, workers: (dict(class_names=names),
                                               Model(), loader)
    tool.load_weights = lambda model, checkpoint, features: None
    tool.frame_voxels = frame_voxels
    try:
        with tempfile.TemporaryDirectory() as tmp, \
                contextlib.redirect_stdout(io.StringIO()):
            out = os.path.join(tmp, 'gmm_head.pth')
            sums = sums_path(out)
            kw = dict(features='head', device='cpu')
            first = tool.fit('cfg.py', 'e9.pth', out, check=2, **kw)
            assert len(frames_seen) == 3 + 2  # the pass, then --check 2
            assert os.path.isfile(out) and os.path.isfile(sums)
            assert first['frames'] == 3 and first['checkpoint'] == 'e9.pth'
            assert first['train_accuracy'].shape == (tool.NUM_CLASSES, )
            # neither file is overwritten, and no pass starts over saved sums
            try:
                tool.fit('cfg.py', 'e9.pth', out, **kw)
            except FileExistsError as err:
                assert out in str(err), str(err)
            else:
                raise AssertionError('overwrote the Gaussian file')
            os.remove(out)
            try:
                tool.fit('cfg.py', 'e9.pth', out, **kw)
            except FileExistsError as err:
                assert sums in str(err) and '--from-sums' in str(err), str(err)
            else:
                raise AssertionError('redid the pass over saved sums')
            assert len(frames_seen) == 5  # both refusals came before a pass
            second = tool.fit('cfg.py', 'e9.pth', out, from_sums=sums, **kw)
            assert len(frames_seen) == 5  # no pass
            assert set(second) == set(first)
            for key, value in first.items():
                if torch.is_tensor(value):
                    assert torch.equal(value, second[key]), key
                else:
                    assert value == second[key], key
            assert torch.equal(torch.load(out)['prec_chol'],
                               first['prec_chol'])
            os.remove(out)
            try:
                tool.fit('cfg.py', 'e9.pth', out, from_sums=sums,
                         features='pe', device='cpu')
            except ValueError as err:
                assert 'head' in str(err), str(err)
            else:
                raise AssertionError('head sums were used for --features pe')
    finally:
        tool.build, tool.load_weights, tool.frame_voxels = real
    print('PASS test_fit_saves_the_sums_first_and_finalises_them_with_'
          'from_sums')


if __name__ == '__main__':
    test_check_rejects_non_finite_log_densities()
    test_check_tolerance()
    test_non_finite_float32_gaussian_names_the_class()
    test_report_prints_the_regularisation()
    test_sums_round_trip_reproduces_finalize()
    test_fit_saves_the_sums_first_and_finalises_them_with_from_sums()
    print('ALL TESTS PASSED')
