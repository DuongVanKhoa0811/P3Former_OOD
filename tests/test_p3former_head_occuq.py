"""Tests for the OCCUQ integration in _P3FormerHead (occuq_cfg).

Builds a tiny head on the CPU (about 1 s).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_p3former_head_occuq.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmengine.registry import init_default_scope  # noqa: E402

init_default_scope('mmdet3d')
import mmdet.models  # noqa: E402,F401  registers the mmdet.* losses
import mmdet3d.models  # noqa: E402,F401  registers LovaszLoss
from mmdet3d.registry import MODELS  # noqa: E402
from mmdet3d.structures import Det3DDataSample, PointData  # noqa: E402

import p3former.decode_heads.p3former_head  # noqa: E402,F401
from p3former.utils.gmm_fit import GaussianStats, finalize  # noqa: E402

NUM_VOXELS = 30


def build_head(occuq_cfg=None, seed=0):
    torch.manual_seed(seed)
    return MODELS.build(
        dict(
            type='_P3FormerHead',
            num_classes=5,
            num_queries=4,
            embed_dims=16,
            thing_class=[0, 1],
            stuff_class=[2, 3],
            ignore_index=4,
            num_decoder_layers=1,
            cls_channels=(16, 16, 5),
            mask_channels=(16, 16, 16, 16, 16),
            point_cloud_range=[0, -3.14159265359, -3, 50, 3.14159265359, 13],
            loss_mask=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                           gamma=2.0, alpha=0.25, reduction='mean',
                           loss_weight=1.0),
            loss_dice=dict(type='mmdet.DiceLoss', loss_weight=2.0),
            loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True,
                          gamma=4.0, alpha=0.25, loss_weight=1.0),
            ood_cfg=dict(num_ood_logits=4),
            occuq_cfg=occuq_cfg))


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    coors = torch.cat([
        torch.zeros(NUM_VOXELS, 1, dtype=torch.int32),
        torch.randint(0, 30, (NUM_VOXELS, 3), generator=g, dtype=torch.int32)
    ], 1)
    return torch.randn(NUM_VOXELS, 16, generator=g), coors


def test_existing_outputs_unchanged_by_the_occuq_head():
    base = build_head(None)
    occuq = build_head(dict(head=True))
    missing, unexpected = occuq.load_state_dict(base.state_dict(),
                                                strict=False)
    assert missing and all(k.startswith('occuq_head.') for k in missing)
    assert not unexpected
    base.eval()
    occuq.eval()
    feats, coors = inputs()
    with torch.no_grad():
        a = base.forward(feats.clone(), coors)
        b = occuq.forward(feats.clone(), coors)
        pe = occuq.extract_pe_features(feats.clone(), coors)
    assert len(a) == len(b) == 5 and a[4] is None
    for x, y in zip(a[3], b[3]):  # sem_preds
        assert torch.equal(x, y)
    for x, y in zip(a[1][-1], b[1][-1]):  # last layer's mask predictions
        assert torch.equal(x, y)
    assert b[4]['logits'][0].shape == (NUM_VOXELS, 5)
    assert b[4]['feature'][0].shape == (NUM_VOXELS, 16)
    assert torch.equal(b[4]['pe'][0], pe[0])
    print('PASS test_existing_outputs_unchanged_by_the_occuq_head')


def test_freeze_base_loss_skips_the_decoder():
    head = build_head(dict(freeze_base=True))
    feats, coors = inputs()
    sample = Det3DDataSample()
    sample.gt_pts_seg = PointData(
        voxel_semantic_mask=torch.randint(0, 5, (NUM_VOXELS, )))
    losses = head.loss(
        dict(features=feats, voxels=dict(voxel_coors=coors)), [sample],
        train_cfg=None)
    assert set(losses) == {'loss_occuq_ce', 'loss_occuq_lovasz'}
    sum(v.mean() for v in losses.values()).backward()
    with_grad = {n for n, p in head.named_parameters() if p.grad is not None}
    assert any(n.startswith('occuq_head.') for n in with_grad)
    decoder = ('transformer_decoder.', 'fc_mask.', 'fc_cls.', 'sem_queries.',
               'queries.')
    assert not any(n.startswith(decoder) for n in with_grad), with_grad
    print('PASS test_freeze_base_loss_skips_the_decoder')


def test_predict_emits_density_scores_and_checks_the_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        files = dict(head=os.path.join(tmp, 'gmm_head.pth'),
                     pe=os.path.join(tmp, 'gmm_pe.pth'))
        cfg = dict(head=True, gmm_file=files['head'],
                   pe_gmm_file=files['pe'])
        head = build_head(cfg).eval()
        feats, coors = inputs()
        with torch.no_grad():
            out = head.forward(feats.clone(), coors)[4]
        for source, key in (('head', 'feature'), ('pe', 'pe')):
            stats = GaussianStats(2, 16)
            stats.update(out[key][0], torch.arange(NUM_VOXELS) % 2,
                         ignore_index=2)
            torch.save(
                dict(finalize(stats, min_count=2), features=source,
                     fingerprint=head.density_fingerprint(source)),
                files[source])
        sample = Det3DDataSample()
        sample.gt_pts_seg = PointData(
            point2voxel_map=torch.randint(0, NUM_VOXELS, (50, )))
        with torch.no_grad():
            _, _, scores = head.predict(
                dict(features=feats.clone(), voxels=dict(voxel_coors=coors)),
                [sample])
        for key in ('density', 'density_pe'):
            assert scores[0][key].shape == (50, ), key
            assert np.isfinite(scores[0][key]).all(), key
        # a head with other weights must refuse these Gaussians
        other = build_head(cfg, seed=1).eval()
        try:
            with torch.no_grad():
                other.predict(
                    dict(features=feats.clone(),
                         voxels=dict(voxel_coors=coors)), [sample])
        except ValueError as err:
            assert 'fitted for other weights' in str(err)
        else:
            raise AssertionError('Gaussians of other weights were accepted')
    print('PASS test_predict_emits_density_scores_and_checks_the_fingerprint')


def test_invalid_occuq_cfg_is_rejected():
    for cfg in (dict(freeze_base=True, head=False),
                dict(head=False, gmm_file='x.pth'),
                dict(unknown_key=1)):
        try:
            build_head(cfg)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted occuq_cfg={cfg}')
    print('PASS test_invalid_occuq_cfg_is_rejected')


if __name__ == '__main__':
    test_existing_outputs_unchanged_by_the_occuq_head()
    test_freeze_base_loss_skips_the_decoder()
    test_predict_emits_density_scores_and_checks_the_fingerprint()
    test_invalid_occuq_cfg_is_rejected()
    print('ALL TESTS PASSED')
