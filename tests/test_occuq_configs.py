"""Tests for the OCCUQ configs (configs/p3former/p3former_2xb1_3x_dso_occuq_*.py).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_occuq_configs.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmengine.config import Config  # noqa: E402

FLAT = ('msp', 'maxlogit', 'odin', 'energy', 'entropy')


def load(variant):
    return Config.fromfile(
        os.path.join(ROOT, 'configs/p3former/'
                     f'p3former_2xb1_3x_dso_occuq_{variant}.py'))


def test_pe_config():
    cfg = load('pe')
    assert dict(cfg.model.decode_head.occuq_cfg) == dict(head=False)
    assert tuple(cfg.test_evaluator[1].score_keys) == FLAT + ('density_pe', )
    assert cfg.test_evaluator[0].learning_map_inv[24] == 0
    print('PASS test_pe_config')


def test_training_configs():
    for variant, freeze in (('a', True), ('c', False)):
        cfg = load(variant)
        assert cfg.model.decode_head.occuq_cfg.freeze_base is freeze
        assert tuple(cfg.test_evaluator[1].score_keys) == FLAT + ('density', )
        assert cfg.test_evaluator[0].learning_map_inv[24] == 0
        # end-of-training validation runs before any Gaussians exist
        assert [m.type for m in cfg.val_evaluator] == ['_PanopticSegMetric']
        assert cfg.load_from == 'work_dirs/p3former_2xb1_3x_dso/epoch_36.pth'
        opt = cfg.optim_wrapper.optimizer
        assert (opt.type, opt.lr, opt.weight_decay) == ('AdamW', 2e-4, 0.01)
        assert cfg.train_cfg.max_epochs == 9
        assert cfg.train_cfg.val_interval == 9
        warmup, cosine = cfg.param_scheduler
        assert (warmup.type, warmup.end, warmup.by_epoch) == ('LinearLR', 500,
                                                              False)
        assert abs(warmup.start_factor - 1 / 3) < 1e-12
        assert (cosine.type, cosine.T_max, cosine.eta_min) == (
            'CosineAnnealingLR', 9, 2e-7)
        assert cosine.convert_to_iter_based is True
        assert cfg.default_hooks.checkpoint.interval == 9
        assert cfg.default_hooks.checkpoint.max_keep_ckpts == 1
        assert cfg.randomness.seed == 0
        assert cfg.train_dataloader.batch_size == 1
    print('PASS test_training_configs')


if __name__ == '__main__':
    test_pe_config()
    test_training_configs()
    print('ALL TESTS PASSED')
