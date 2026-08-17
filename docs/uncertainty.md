# Gravity-field uncertainty

This library supports two deliberately distinct ways to study lunar gravity-field uncertainty:

1. covariance-derived GRGM1200A clone realizations archived by NASA PDS;
2. an explicitly opt-in diagonal approximation that perturbs coefficients independently using the uncertainty fields stored in a SHADR product.

The first approach is preferred when the objective is a scientifically meaningful GRGM1200A gravity-error ensemble because it preserves coefficient correlations represented by the underlying least-squares covariance system. The second approach is useful for controlled sensitivity studies and testing, but it is not a substitute for a covariance model.

## What the SHADR uncertainty fields mean

The NASA/PDS SHADR Software Interface Specification defines each coefficient record with six fields:

```text
DEGREE, ORDER, C, S, C UNCERTAINTY, S UNCERTAINTY
```

The specification describes the final two fields as the uncertainty in `Cnm` and the uncertainty in `Snm`. The SHADR header separately contains `UNCERTAINTY IN CONSTANT` for the gravitational constant `GM` stored in the product.

The parser therefore preserves these values as:

```python
model.sigma_c
model.sigma_s
model.mu_sigma_m3_s2
```

and exposes individual coefficient values with:

```python
sigma_c, sigma_s = model.coefficient_uncertainty(n, m)
```

For GRGM1200A specifically, NASA Goddard describes the coefficient values distributed with the model as calibrated uncertainties.

These arrays are uncertainty metadata. They do not contain the off-diagonal coefficient correlations.

Primary sources:

- NASA/PDS SHADR SIS: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/document/shadr.htm
- NASA Goddard GRGM1200A product page: https://pgda.gsfc.nasa.gov/products/50

## Why independent sigma sampling is opt-in

A set of coefficient standard uncertainties does not define a joint probability distribution unless the coefficient correlations are also known or an independence assumption is made.

For that reason the library refuses this call:

```python
sample_independent_coefficient_uncertainty(model, seed=1234, count=100)
```

A caller must acknowledge the approximation explicitly:

```python
realizations = sample_independent_coefficient_uncertainty(
    model,
    seed=1234,
    count=100,
    assume_independent=True,
)
```

Each sampled coefficient is then treated as an independent Gaussian draw centered on the nominal coefficient with the archived uncertainty as its scale. An optional `sigma_scale` can be used for controlled sensitivity experiments. Sampling of the archived `GM` uncertainty is separate and disabled by default.

The random generator is initialized from the supplied integer seed, so repeated runs with the same inputs and seed reproduce the same coefficient realizations.

This mode is intentionally named `sample_independent_coefficient_uncertainty` so results cannot easily be mistaken for a full-covariance Monte Carlo analysis.

## GRGM1200A covariance and clone products

NASA Goddard documents that a complete degree/order-1200 covariance matrix would be extremely large. The PDS archive therefore includes selected truncated covariance products and 500 full-field clone gravity realizations generated from the full covariance information. NASA states that these clones account for cross-correlations among coefficients.

The clone archive is external to this repository:

https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/

The library maps clone indices 1 through 500 to their official PDS locations with:

```python
from lunar_astrodynamics import grgm1200a_clone_url

print(grgm1200a_clone_url(1))
print(grgm1200a_clone_url(500))
```

Selected clones can be downloaded reproducibly without committing the large files to Git:

```bash
python scripts/download_grgm1200a_clones.py 1 2 3 4 5
```

The downloader requires explicit indices. It does not default to downloading the complete archive.

Load the downloaded realizations with:

```python
from pathlib import Path
from lunar_astrodynamics import load_grgm1200a_clone_ensemble

paths = sorted(Path("data/grgm1200a_clones").glob("*_sha.tab"))
models = load_grgm1200a_clone_ensemble(paths, max_degree=120)
```

`read_grgm1200a_clone(...)` treats each clone as one correlated coefficient realization and applies the archived nominal GRGM1200A `GM`, reference radius, 4pi normalization, and principal-axes frame metadata. It does not attach `sigma_c` or `sigma_s` arrays to a clone because a clone is a sampled field, not a new uncertainty estimate.

## Ensemble orbit propagation

Any sequence of `SphericalHarmonicModel` realizations can be propagated from the same initial state:

```python
result = propagate_gravity_ensemble(
    initial_state,
    duration_s,
    models,
    body_fixed_from_inertial,
    max_degree=120,
)
```

The same propagation settings, initial Cartesian state, frame transformation, collision boundary, and sampling times are applied to every realization.

For each trajectory the library reports:

- minimum instantaneous radial altitude;
- maximum instantaneous radial altitude;
- minimum osculating periselene altitude encountered in the sampled history;
- maximum osculating aposelene altitude encountered in the sampled history;
- maximum eccentricity;
- final eccentricity;
- lifetime to the configured surface event, or the requested propagation duration when no impact occurs;
- impact status.

The default ensemble summary reports the 5th, 50th, and 95th percentiles for the numeric metrics plus the fraction of realizations that impacted. Percentile levels are configurable.

The eccentricity and osculating apsis calculations use Cartesian position/velocity vectors directly rather than RAAN or argument of periapsis, so the uncertainty summary does not inherit the classical-element singularities of circular or equatorial orbits.

## Altitude convention

Until terrain-aware topography is implemented, altitude metrics are radial distance minus a caller-selected reference radius. The default is the mean lunar radius used elsewhere in the package.

Consequently:

- minimum altitude is not minimum terrain clearance;
- impact uses the configured spherical collision boundary;
- low-altitude conclusions should not be interpreted as LOLA-aware surface-clearance predictions.

## What the uncertainty does not represent

A gravity ensemble isolates sensitivity to the gravity realization supplied to the propagator. It does not automatically include uncertainty in:

- the spacecraft initial state or orbit determination;
- lunar orientation or SPICE frame realization;
- Earth/Sun ephemerides;
- unmodeled third-body forces;
- solar radiation pressure or spacecraft optical properties;
- maneuvers or navigation errors;
- lunar topography;
- time-variable gravity or tides;
- systematic modeling errors that are not represented by the gravity covariance used to produce the ensemble.

The covariance-derived clone fields should therefore be interpreted as one component of a broader mission uncertainty budget.

## Finite ensemble interpretation

Percentiles from a finite clone ensemble are empirical statistics. With only a few selected clones, tail percentiles are poorly resolved. Use enough realizations for the decision being made and report the ensemble size alongside percentile results.

The PDS archive provides 500 GRGM1200A clone realizations, but using all 500 is a computational choice rather than a requirement of the API.

## Reproducible example

Run the self-contained example:

```bash
python examples/gravity_uncertainty.py --samples 16 --seed 20260817 --duration-days 1
```

This default mode uses a small synthetic low-degree field and explicitly labels the result as an independent-sigma demonstration. It is intended to demonstrate the workflow without downloading external datasets.

For correlated GRGM1200A analysis, first download selected clones and pass them to the same example:

```bash
python scripts/download_grgm1200a_clones.py 1 2 3 4 5
python examples/gravity_uncertainty.py \
  --degree 120 \
  --duration-days 1 \
  --clones data/grgm1200a_clones/gggrx_1200a_clone0001_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0002_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0003_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0004_sha.tab \
           data/grgm1200a_clones/gggrx_1200a_clone0005_sha.tab
```

The resulting JSON identifies the uncertainty method, number of realizations, seed when applicable, individual trajectory metrics, percentile summaries, and impact fraction.
