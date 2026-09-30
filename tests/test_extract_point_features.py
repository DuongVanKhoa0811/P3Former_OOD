"""Tests for tools/extract_point_features.py (stand-in head, no model).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_extract_point_features.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import extract_point_features as epf  # noqa: E402


def test_sample_frame():
    label = np.full(2000, epf.IGNORE_INDEX)
    label[:100] = 0
    label[100:110] = 5
    ood = np.zeros(2000, bool)
    ood[500:1500] = True  # OOD points carry the ignore label
    index, weight = epf.sample_frame(label, ood, 16, 64,
                                     np.random.default_rng([0, 3]))
    assert len(index) == len(set(index.tolist())) == 16 + 10 + 64
    assert (label[index] == 0).sum() == 16 and (label[index] == 5).sum() == 10
    assert ood[index].sum() == 64
    # the weights restore the stratum sizes
    assert np.isclose(weight[label[index] == 0].sum(), 100)
    assert np.isclose(weight[label[index] == 5].sum(), 10)
    assert np.isclose(weight[ood[index]].sum(), 1000)
    # ignored points that are not OOD are never drawn
    assert not np.any((label[index] == epf.IGNORE_INDEX) & ~ood[index])
    again, _ = epf.sample_frame(label, ood, 16, 64,
                                np.random.default_rng([0, 3]))
    assert np.array_equal(index, again)
    # a frame without OOD points, and one without any valid point
    index, _ = epf.sample_frame(label, np.zeros(2000, bool), 16, 64,
                                np.random.default_rng(1))
    assert len(index) == 26
    index, weight = epf.sample_frame(np.full(5, epf.IGNORE_INDEX),
                                     np.zeros(5, bool), 16, 64,
                                     np.random.default_rng(1))
    assert index.shape == (0, ) and weight.shape == (0, )
    print('test_sample_frame passed')


class _Head(torch.nn.Module):
    """Stand-in for _P3FormerHead: init_inputs returns (queries,
    pe_features, mpe, sem_preds) with sem_preds = pe_features @ W^T."""

    def __init__(self, feat, pos, weight):
        super().__init__()
        self.sem_queries = torch.nn.Conv3d(feat.shape[1], weight.shape[0], 1,
                                           bias=False)
        with torch.no_grad():
            self.sem_queries.weight.copy_(
                weight.reshape(self.sem_queries.weight.shape))
        self._feat, self._pos = feat, pos

    def init_inputs(self, features, voxel_coors, batch_size):
        w = self.sem_queries.weight.reshape(self.sem_queries.weight.shape[0], -1)
        return [None], [self._feat], [self._pos], [self._feat @ w.t()]

    def forward(self):
        return self.init_inputs(None, None, 1)


def _data_sample(p2v, label, raw):
    return SimpleNamespace(
        gt_pts_seg=SimpleNamespace(point2voxel_map=torch.from_numpy(p2v)),
        eval_ann_info=dict(
            pts_instance_mask=(np.arange(len(raw)) << 16) | raw,
            pts_semantic_mask=label),
        metainfo=dict(lidar_path='seq/000001.ply'))


def test_capture_and_frame_record():
    torch.manual_seed(0)
    voxels, dim = 50, 16
    feat, pos = torch.randn(voxels, dim), torch.randn(voxels, dim)
    weight = torch.randn(25, dim)  # 24 classes + the ignore channel
    head = _Head(feat, pos, weight)
    rng = np.random.RandomState(0)
    n = 400
    p2v = rng.randint(0, voxels, n)
    raw = rng.choice([1, 3, 17, 28, 40], n)
    label = np.where(np.isin(raw, [17, 28, 40]), 24, rng.randint(0, 24, n))
    sample = _data_sample(p2v, label, raw)
    with epf.FeatureCapture(head) as capture:
        head.forward()
        assert capture.feat[0] is feat and capture.pos[0] is pos
        record, diff = epf.frame_record(capture, sample,
                                        epf.classifier_weight(head), 8, 32,
                                        np.random.default_rng(0), frame=7)
    assert 'init_inputs' not in head.__dict__  # the wrapper is gone
    idx = record['index']
    assert np.allclose(record['feat'], feat[p2v[idx]].numpy(), atol=1e-2)
    assert np.allclose(record['pos'], pos[p2v[idx]].numpy(), atol=1e-2)
    want = (feat @ weight[:24].t())[p2v[idx]].numpy()
    assert np.allclose(record['logits'].astype(np.float32), want, atol=2e-2)
    assert np.array_equal(record['raw'], raw[idx])
    assert np.array_equal(record['ood'], np.isin(raw[idx], [17, 28]))
    assert np.array_equal(record['label'], label[idx])
    assert record['frame'] == 7 and record['lidar_path'] == 'seq/000001.ply'
    assert diff < 1e-4
    # the wrong tensor captured: the logits check raises
    capture.logits = [torch.randn(voxels, 25)]
    try:
        epf.frame_record(capture, sample, epf.classifier_weight(head), 8, 32,
                         np.random.default_rng(0), frame=7)
    except RuntimeError as err:
        assert 'reproduce the logits' in str(err)
    else:
        raise AssertionError('mismatched logits accepted')
    print('test_capture_and_frame_record passed')


def test_dump_comparison():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.RandomState(1)
        n = 300
        logits = rng.randn(n, 24).astype(np.float16)
        mapped = rng.randint(0, 25, n).astype(np.int16)
        ood = mapped == 24
        path = os.path.join(tmp, 'r0_000000.npz')
        np.savez(path, logits=logits, ood=ood, valid=np.ones(n, bool),
                 mapped=mapped, lidar_path='seq/000001.ply')
        index = np.array([3, 10, 200], np.int32)
        record = dict(index=index, logits=logits[index], label=mapped[index],
                      ood=ood[index], frame=0, lidar_path='seq/000001.ply')
        assert epf.index_dump(tmp) == {'seq/000001.ply': path}
        assert epf.compare_with_dump(record, path) == 0.0
        for key, bad in (('logits', logits[index] + np.float16(0.5)),
                         ('label', (mapped[index] + 1) % 24),
                         ('ood', ~ood[index])):
            try:
                epf.compare_with_dump(dict(record, **{key: bad}), path)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'mismatched {key} accepted')
    print('test_dump_comparison passed')


def test_out_dir_must_be_empty():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, 'f000000.npz'), 'w').close()
        try:
            epf.extract('config.py', 'checkpoint.pth', tmp)
        except FileExistsError as err:
            assert tmp in str(err)
        else:
            raise AssertionError('non-empty out dir accepted')
    print('test_out_dir_must_be_empty passed')


if __name__ == '__main__':
    test_sample_frame()
    test_capture_and_frame_record()
    test_dump_comparison()
    test_out_dir_must_be_empty()
    print('ALL TESTS PASSED')
