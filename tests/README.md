# Test scope

The release includes lightweight automated tests for package integrity and the
fixed learning dataset. They verify:

- required release files and directory structure;
- absence of Python bytecode and cache directories;
- package-level SHA-256 checksums;
- dataset shapes, wavelength grid, and four-region geometry constraints;
- legal material-sequence rules and the expected 400-sequence registry;
- disjoint and exhaustive train/validation/test indices;
- numerical energy closure, `Q_ext = Q_sca + Q_abs`, within the recorded
  floating-point tolerance;
- the frozen controlled-ablation component ladder, exact registered-parameter
  matching, and isolation of the Fourier and physics information switches.

Run from the repository root after installing the test extra:

```bash
python -m pip install -e ".[test]" --no-deps
python -B -m pytest
```

The tests are intentionally lightweight. Full dataset regeneration, model
training, inverse design, and full-factorial robustness evaluation remain
separate reproduction workflows because they require substantial compute.
