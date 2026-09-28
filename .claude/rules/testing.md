# Tests

```bash
# 39 tests, ~5 s. Every test file also runs as a plain script.
python -m pytest -q tests
python -m pytest -q tests/test_ood_scores.py::test_energy_temperature
```

`python` is the `p3former` interpreter (see `environment.md`). There is no lint setup.
