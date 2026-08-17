# Gravity-field uncertainty

This library supports two deliberately distinct approaches to lunar gravity-field uncertainty:

1. covariance-derived GRGM1200A clone perturbations archived by NASA PDS and added to the nominal GRGM1200A coefficients;
2. an explicitly opt-in diagonal approximation that perturbs nominal coefficients independently using the uncertainty fields stored in a SHADR product.

The clone approach is preferred when coefficient correlations matter. The diagonal mode is useful for controlled sensitivity studies and testing, but it is not a substitute for a covariance model.

## SHADR uncertainty fields

The NASA/PDS SHADR Software Interface Specification defines each coefficient record with six fields:

```text
DEGREE, ORDER, C, S, C UNCERTAINTY, S UNCERTAINTY
```

The final two fields are defined as the uncertainty in `Cnm` and `Snm`. The SHADR header separately contains `UNCERTAINTY IN CONSTANT` for the stored gravitational constant `GM`.

`read_shadr(...)` preserves these values as:

```python
model.sigma_c
model.sigma_s
model.mu_sigma_m3_s2
```

Individual coefficient uncertainties can be inspected with:

```python
sigma_c, sigma_s = model.coefficient_uncertainty(n, m)
```

For GRGM1200A specifically, NASA Goddard describes the distributed coefficient values as having calibrated uncertainties.

Primary sources:

- NASA/PDS SHADR SIS: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/document/shadr.htm
- NASA Goddard GRGM1200A product page: https://pgda.gsfc.nasa.gov/products/50

The `sigma_c` and `sigma_s` arrays are coefficient-level uncertainty metadata. They do not encode off-diagonal correlations.

## Independent sigma sampling

A set of coefficient uncertainties does not by itself define a joint distribution. An independence assumption must be stated explicitly.

For this reason the library rejects:

```python
sample_independent_coefficient_uncertainty(model, seed=1234, count=100)
```

A caller must acknowledge the diagonal approximation:

```python
realizations = sample_independent_coefficient_uncertainty(
    model,
    seed=1234,
    count=100,
    assume_independent=True,
)
```

Each coefficient is then sampled independently from a Gaussian centered on the nominal value with the archived uncertainty used as its scale. `sigma_scale` can be changed for controlled sensitivity experiments. GM sampling is separate and disabled unless `include_mu=True` is supplied.

The NumPy generator is created from the explicit integer seed, so identical inputs and seeds reproduce identical draws.

The function name and mandatory `assume_independent=True` argument are intentional safeguards against presenting this approximation as full-covariance Monte Carlo propagation.

## GRGM1200A covariance products

NASA Goddard states that the full degree/order-1200 covariance matrix would be about 8 TB and was not archived as one complete matrix. Truncated covariance products are available at selected degrees, while clone fields provide a more practical way to propagate the correlated gravity error.

NASA states that the GRGM1200A clones are produced from the full covariance information and account for cross-correlations among coefficients. Five hundred clone files are archived at PDS.

Official archive:

https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/

The published GRGM1200A literature describes these clone coefficients as deviations from the base GRGM1200A model. Therefore the files must not be propagated as standalone lunar gravity models. They are coefficient perturbations that are added to the compatible nominal solution.

The library enforces this distinction with the type:

```python
GravityCoefficientPerturbation
```

and this workflow:

```python
from lunar_astrodynamics import (
    GRGM1200A,
    apply_coefficient_perturbation,
    read_grgm1200a_clone,
    read_shadr,
)

nominal = read_shadr(
    "data/gggrx_1200a_sha.tab",
    max_degree=120,
    frame=GRGM1200A.body_fixed_frame,
)

delta = read_grgm1200a_clone(
    "data/grgm1200a_clones/gggrx_1200a_clone0001_sha.tab",
    max_degree=120,
)

realization = apply_coefficient_perturbation(nominal, delta)
```

`GravityCoefficientPerturbation` has `C00 = S00 = 0`. It deliberately cannot be passed directly to the propagation API, which accepts complete `SphericalHarmonicModel` objects.

## Clone download tooling

The repository does not bundle the large PDS files. Clone indices 1 through 500 are mapped to the official archive structure with `grgm1200a_clone_url(index)`.

Download only the selected files needed for a study:

```bash
python scripts/download_grgm1200a_clones.py 1 2 3 4 5
```

The downloader checks the archived byte size and parses a low-degree prefix as a format smoke check. It does not default to downloading all 500 files.

The nominal GRGM1200A SHADR file is downloaded separately:

```bash
python scripts/download_grgm1200a.py
```

## Loading a correlated ensemble

A compatible nominal model is required explicitly:

```python
from pathlib import Path
from lunar_astrodynamics import (
    GRGM1200A,
    load_grgm1200a_clone_ensemble,
    read_shadr,
)

nominal = read_shadr(
    "data/gggrx_1200a_sha.tab",
    max_degree=120,
    frame=GRGM1200A.body_fixed_frame,
)

paths = sorted(Path("data/grgm1200a_clones").glob("*_sha.tab"))
models = load_grgm1200a_clone_ensemble(
    nominal,
    paths,
    max_degree=120,
)
```

The loader checks the nominal GRGM1200A GM, reference radius, normalization, and frame metadata before applying clone perturbations. Each output member is a complete gravity realization with nominal coefficients plus one covariance-derived clone delta.

The output realization does not carry `sigma_c` or `sigma_s` because it is already one sampled field rather than a new uncertainty estimate.

## Ensemble orbit propagation

Any sequence of complete gravity realizations can be propagated from the same initial Cartesian state:

```python
result = propagate_gravity_ensemble(
    initial_state,
    duration_s,
    models,
    body_fixed_from_inertial,
    max_degree=120,
)
```

Every member receives the same propagation settings, initial state, frame transformation, collision boundary, and output sampling times.

For each trajectory the library reports:

- minimum instantaneous radial altitude;
- maximum instantaneous radial altitude;
- minimum osculating periselene altitude encountered in the sampled history;
- maximum osculating aposelene altitude encountered in the sampled history;
- maximum eccentricity;
- final eccentricity;
- lifetime to the configured collision event, or the requested duration if no impact occurs;
- impact status.

The default summary reports the 5th, 50th, and 95th percentiles for numeric metrics and the fraction of realizations that impact. Percentile levels are configurable.

Eccentricity and apsis calculations are obtained directly from Cartesian position and velocity vectors. They do not depend on RAAN or argument of periapsis and therefore avoid the classical-element singularities of circular or equatorial orbits.

## Altitude convention

Until terrain-aware topography is added, altitude means radial distance minus a caller-selected reference radius. The default is the lunar mean radius used elsewhere in the package.

Consequently:

- minimum altitude is not minimum terrain clearance;
- impact uses the configured spherical collision boundary;
- low-altitude uncertainty results are not LOLA-aware clearance predictions.

## What the ensemble does not represent

A gravity ensemble isolates sensitivity to the supplied gravity realization. It does not automatically include uncertainty in:

- spacecraft initial state or orbit determination;
- lunar orientation or SPICE frame realization;
- Earth/Sun ephemerides;
- unmodeled third-body forces;
- solar radiation pressure or spacecraft optical properties;
- maneuvers or navigation errors;
- lunar topography;
- time-variable gravity or tides;
- systematic model errors not represented by the covariance used to generate the clone ensemble.

The clone ensemble is therefore one component of a mission uncertainty budget, not a complete mission-risk distribution.

## Finite ensemble interpretation

Percentiles calculated from clone runs are empirical statistics. A small selected set cannot resolve distribution tails reliably. Report the number of realizations together with percentile results and choose an ensemble size appropriate to the decision being made.

The PDS archive contains 500 GRGM1200A clone perturbations. The API does not require all 500 to be used.

## Reproducible example

The self-contained example runs without external data:

```bash
python examples/gravity_uncertainty.py \
  --samples 16 \
  --seed 20260817 \
  --duration-days 1
```

This mode uses a small synthetic low-degree field and labels the output as an independent-sigma demonstration.

For a correlated GRGM1200A study:

```bash
python scripts/download_grgm1200a.py
python scripts/download_grgm1200a_clones.py 1 2 3 4 5

python examples/gravity_uncertainty.py \
  --nominal data/gggrx_1200a_sha.tab \
  --degree 120 \
  --duration-days 1 \
  --clones data/grgm1200a_clones/gggrx_1200a_clone0001_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0002_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0003_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0004_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0005_sha.tab
```

The JSON output records the method, seed when applicable, ensemble size, individual trajectory metrics, percentile summaries, and impact fraction.
