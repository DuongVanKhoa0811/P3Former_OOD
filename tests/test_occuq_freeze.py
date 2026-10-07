"""Tests for OCCUQ variant A's freezing (p3former/utils/freeze.py and
_P3Former). Builds the real DSO model on the CPU (about 3 s).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_occuq_freeze.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from p3former.decode_heads.occuq_head import _OCCUQHead  # noqa: E402
from p3former.utils.freeze import eval_all_but, freeze_all_but  # noqa: E402


def test_helpers_on_a_toy_model():
    keep = _OCCUQHead(in_channels=8, num_classes=3)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8),
                                torch.nn.BatchNorm1d(8), keep)
    freeze_all_but(model, keep)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith('2.') for n in trainable)
    model.train()
    eval_all_but(model, keep)
    assert not model[0].training and not model[1].training
    assert all(m.training for m in keep.modules())
    before = model[1].running_mean.clone()
    model(torch.randn(16, 8))
    assert torch.equal(model[1].running_mean, before)  # BN stats fixed
    print('PASS test_helpers_on_a_toy_model')


def test_variant_a_freezes_the_real_model():
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    cfg = Config.fromfile(
        os.path.join(ROOT, 'configs/p3former/p3former_2xb1_3x_dso_ood.py'))
    cfg.model.decode_head.occuq_cfg = dict(freeze_base=True)
    init_default_scope('mmdet3d')
    from mmdet3d.registry import MODELS
    model = MODELS.build(cfg.model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith('decode_head.occuq_head.') for n in trainable)
    model.train()
    # the preprocessor builds voxel_semantic_mask only in training mode
    assert model.data_preprocessor.training
    occuq = set(model.decode_head.occuq_head.modules())
    for name, module in model.named_modules():
        if module in occuq:
            assert module.training, name
        elif name.startswith(('voxel_encoder', 'backbone', 'decode_head')):
            assert not module.training, name
    model.eval()
    assert not any(m.training for m in model.modules())
    print('PASS test_variant_a_freezes_the_real_model')


if __name__ == '__main__':
    test_helpers_on_a_toy_model()
    test_variant_a_freezes_the_real_model()
    print('ALL TESTS PASSED')
