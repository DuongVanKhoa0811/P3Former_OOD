# Tests

```bash
# 120 tests, ~70 s. Every test file also runs as a plain script.
python -m pytest -q tests
python -m pytest -q tests/test_ood_scores.py::test_class_groups_variants
```

`python` is the `p3former` interpreter (see `environment.md`). There is no lint setup.
