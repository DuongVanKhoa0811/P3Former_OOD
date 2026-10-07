"""Tests for the OCCUQ head (p3former/decode_heads/occuq_head.py).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_occuq_head.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p3former.decode_heads.occuq_head import _OCCUQHead  # noqa: E402


def test_shapes():
    head = _OCCUQHead(in_channels=32, num_classes=5)
    logits, feature = head(torch.randn(7, 32))
    assert logits.shape == (7, 5)
    assert feature.shape == (7, 32)
    print('PASS test_shapes')


def test_layout_mirrors_occuq():
    head = _OCCUQHead(in_channels=32, num_classes=5)
    keys = set(head.state_dict())
    # OCCUQ's widened 1x1 conv: no bias, no spectral normalisation
    assert {k for k in keys if k.startswith('input_proj.')} == {
        'input_proj.weight'}
    assert len(head.blocks) == 4
    for prefix in [f'blocks.{i}.' for i in range(4)] + ['classifier.']:
        for name in ('weight_orig', 'weight_u', 'weight_v', 'bias'):
            assert prefix + name in keys, prefix + name
    print('PASS test_layout_mirrors_occuq')


def test_spectral_norm_is_one_after_power_iterations():
    torch.manual_seed(0)
    head = _OCCUQHead(in_channels=32, num_classes=5).train()
    x = torch.randn(64, 32)
    for _ in range(100):  # one power iteration per training-mode forward
        head(x)
    for layer in list(head.blocks) + [head.classifier]:
        sigma = torch.linalg.svdvals(layer.weight.detach())[0].item()
        assert abs(sigma - 1.0) < 1e-2, sigma
    print('PASS test_spectral_norm_is_one_after_power_iterations')


def test_feature_is_last_block_output():
    torch.manual_seed(0)
    head = _OCCUQHead(in_channels=16, num_classes=3).eval()
    x = torch.randn(5, 16)
    with torch.no_grad():
        logits, feature = head(x)
        h = head.input_proj(x)
        for block in head.blocks:
            h = h + torch.relu(block(h))
        assert torch.allclose(feature, h, atol=1e-6)
        assert torch.allclose(logits, head.classifier(h), atol=1e-6)
    print('PASS test_feature_is_last_block_output')


if __name__ == '__main__':
    test_shapes()
    test_layout_mirrors_occuq()
    test_spectral_norm_is_one_after_power_iterations()
    test_feature_is_last_block_output()
    print('ALL TESTS PASSED')
