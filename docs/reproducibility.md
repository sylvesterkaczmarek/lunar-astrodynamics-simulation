# Reproducibility

## Environment

Python 3.10 or newer is supported. Dependencies and version bounds are declared in `pyproject.toml` and `requirements.txt`.

## Clean run

```bash
git clone https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation.git
cd lunar-astrodynamics-simulation
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
python examples/j2_precession.py --orbits 40
```

The demonstration contains no random sampling. Given the same numerical-library versions and platform, results should agree to floating-point integration precision.

## Validation strategy

The test suite checks six independent properties:

1. Point-mass energy and angular-momentum conservation.
2. Conservation of axial angular momentum under the axisymmetric J2 field.
3. Numerical RAAN and periapsis precession against first-order analytical J2 secular rates.
4. Terminal detection of an inward trajectory crossing the mean-radius lunar surface.
5. Rejection of initial states at or below that surface.
6. Agreement of the default propagator with a tighter DOP853 reference integration.

The example writes machine-readable metrics to `results/j2_validation.json`.
