# Registry pattern and `custom_imports`

Custom code lives in three top-level namespace packages with no `__init__.py` anywhere: `p3former/` (model), `datasets/` (datasets and pipeline transforms) and `evaluation/` (metrics).

- Every class name is prefixed with `_` (e.g. `_P3Former`, `_LaserMix`) and registered into the mmdet3d registries, often with `force=True` to shadow the upstream class of the same un-prefixed name. Config `type=` strings use the underscore names.
- Nothing is imported except through the `custom_imports` list at the bottom of each top-level config. **A new module only takes effect if it is registered and also added to `custom_imports`**; for example, only the `_submit` config imports `datasets.transforms.formating`.
