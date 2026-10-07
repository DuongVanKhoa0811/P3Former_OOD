"""Freeze all of a model except one submodule (OCCUQ variant A)."""
import torch.nn as nn


def freeze_all_but(model: nn.Module, keep: nn.Module) -> None:
    """Set requires_grad=False on every parameter of ``model`` outside
    ``keep``, and True on the parameters inside it."""
    keep_ids = {id(p) for p in keep.parameters()}
    for param in model.parameters():
        param.requires_grad_(id(param) in keep_ids)


def eval_all_but(model: nn.Module, keep: nn.Module) -> None:
    """Put every module of ``model`` outside ``keep`` in eval mode, and
    leave ``keep``'s modules as they are.

    Call it after ``model.train()``: BatchNorm statistics and dropout of the
    frozen part then stay fixed. The flag is set module by module, because
    ``Module.eval()`` would recurse into ``keep``."""
    keep_ids = {id(m) for m in keep.modules()}
    for module in model.modules():
        if id(module) not in keep_ids:
            module.training = False
