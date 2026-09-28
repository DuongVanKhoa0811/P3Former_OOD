# Environment

- Pinned stack: Python 3.8, torch 1.10.1+cu111, mmengine 0.7.4, mmcv 2.0.0rc4, mmdet 3.0.0, mmdet3d 1.1.0, torch_scatter, spconv, and `yapf==0.40.1`, because newer yapf breaks mmengine 0.7.4. The conda env is `p3former`.
- **In Claude Code's shell, `python` resolves to the `vos_detr` env, not `p3former`.** Call `/home/khoadv/miniconda3/envs/p3former/bin/python` explicitly. The `dist_*.sh` scripts invoke bare `python`, so put `/home/khoadv/miniconda3/envs/p3former/bin` first on `PATH` for them.
- Every shell call prints a `libtinfo.so.6: no version information available` warning, which is harmless.
- There is no setup.py; the repo is never installed. **Run everything from the repo root**, because custom modules are imported by path via each config's `custom_imports`.
