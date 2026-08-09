# Reproducibility

## Local checks

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
python examples/j2_precession.py --orbits 40
python examples/harmonic_validation.py
```

The J2 example writes:

- `results/j2_validation.json`
- `assets/results/j2_precession.svg`

The spherical-harmonic validation writes:

- `results/harmonic_validation.json`

## External GRGM1200A data

Download the official NASA PDS file with:

```bash
python scripts/download_grgm1200a.py
```

The script downloads both the SHADR coefficient table and PDS3 label, verifies the archived file size, then parses the header and low-degree coefficients to check GM and reference radius.

The downloaded `.tab` and `.lbl` products are ignored by Git because they are external scientific data, not repository source.

Evaluate the full field at a body-fixed point with:

```bash
python examples/grgm1200a_gravity.py \
  --model data/gggrx_1200a_sha.tab \
  --degree 1200 \
  --position-km 1900 200 300
```

Lower truncation degrees can be selected with `--degree` for faster sensitivity studies.

## What CI verifies

CI does not download the 88 MB NASA data product. It verifies the implementation independently using:

- two-body invariants
- first-order J2 secular rates
- J2 versus normalized `C20` equivalence
- analytical harmonic acceleration versus finite-difference potential gradients
- longitude dependence of tesseral terms
- body-fixed/inertial rotation consistency
- SHADR format parsing
- degree-1200 normalized Legendre stability
- degree-1200 acceleration finiteness
- surface-event and integration-convergence regressions

This separates source-code verification from availability of an external archive.
