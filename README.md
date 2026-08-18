# Lunar Astrodynamics Simulation

![Lunar Astrodynamics Simulation](assets/social/github-social-card-lunar-astrodynamics.png)

[![CI](https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A scientific Python package for **low lunar orbit propagation and preliminary mission analysis**.

It is designed for work where a two-body orbit is not enough: high-degree GRAIL gravity, explicit lunar frames, Earth/Sun perturbations, solar radiation pressure and eclipses, LOLA terrain clearance, nonsingular orbital analysis, frozen-orbit search, sensitivity/targeting, station-keeping estimates, gravity-fidelity selection, uncertainty ensembles, ground tracks, surface access and coverage.

The package keeps the mathematics and model choices visible. Gravity degree/order, frame names, kernels, force components, terrain, integration tolerances, uncertainty assumptions and surface-site coordinates remain explicit and are recorded in result provenance.

## Start with one mission file

The main user workflow is a TOML mission configuration plus the Python API.

```bash
git clone https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation.git
cd lunar-astrodynamics-simulation
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]

lunar-mission analyse examples/missions/polar_quick.toml \
  --output-dir results/polar_quick
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The self-contained example uses low-degree J2 screening dynamics and requires no external data. It demonstrates the workflow without presenting the screening model as the preferred model for low-altitude science work.

The result directory contains:

```text
mission.json
trajectory.csv
summary.txt
plots/
  altitude.svg
  apsides.svg
  eccentricity.svg
```

A typical human-readable summary reports requested and achieved duration, minimum/maximum reference-radius altitude, minimum terrain clearance when terrain is enabled, periselene and aposelene evolution, eccentricity and eccentricity-vector drift, impact status, enabled perturbations, and harmonic degree/order/fidelity.

The JSON keeps the same information together with initial state, frame/kernel provenance, force parameters, integration settings, site definitions and software versions.

## Python API first

The CLI is a convenience layer over the same scientific functions used by the Python API.

```python
from lunar_astrodynamics.mission import (
    build_mission_context,
    load_mission_config,
    run_access_workflow,
    run_mission,
)

config = load_mission_config("examples/missions/polar_quick.toml")
context = build_mission_context(config)

try:
    mission = run_mission(context)
    access = run_access_workflow(context, mission)

    print(mission.human_summary())
    mission.write_json("results/mission.json")
    mission.write_csv("results/trajectory.csv")
finally:
    context.close()
```

Users who need lower-level control can still call the individual force, propagation, analysis, search, targeting, uncertainty, fidelity and access APIs directly.

See [`docs/mission_workflow.md`](docs/mission_workflow.md) for the complete TOML schema and composition rules.

## CLI workflows

```bash
lunar-mission propagate CONFIG.toml
lunar-mission analyse CONFIG.toml
lunar-mission search CONFIG.toml
lunar-mission uncertainty CONFIG.toml
lunar-mission fidelity CONFIG.toml
lunar-mission access CONFIG.toml
lunar-mission all CONFIG.toml
```

`all` runs the nominal mission plus every optional downstream workflow configured in the TOML file.

The commands produce JSON and CSV rather than relying on plots. SVG plots are additional scientific outputs for quick inspection.

## Complete public-data example

A second example uses official public lunar and ephemeris products.

```bash
python -m pip install -e .[spice]
python scripts/download_example_mission_data.py

lunar-mission all examples/missions/polar_high_fidelity.toml \
  --output-dir results/polar_high_fidelity
```

The downloader retrieves:

- PDS **GRGM900C** GRAIL gravity;
- NAIF **DE421** planetary ephemeris;
- NAIF DE421 lunar principal-axes and frame kernels;
- NAIF leap-seconds kernel;
- PDS **LOLA LDEM_4** terrain.

It writes a manifest containing each source URL, file size and SHA-256 digest.

The example additionally uses NASA-published coordinates for Apollo 11 Tranquility Base (`0.67409° N, 23.47298° E`) and Apollo 17 Taurus-Littrow (`20.1911° N, 30.7769° E`).

Large external products are never committed to the repository.

## What makes this different from a basic lunar orbit demo

### GRAIL gravity

- geodesy 4π-normalised spherical harmonics;
- NASA PDS SHADR parsing;
- arbitrary degree/order truncation;
- pole-safe evaluation through degree/order 1200;
- retained coefficient and GM uncertainty fields;
- covariance-derived GRGM1200A clone support;
- fidelity-versus-runtime analysis for choosing a defensible truncation.

### Frames and ephemerides

- explicit body-fixed/inertial transformations;
- SPICE UTC/ET handling;
- kernel, epoch, observer and frame provenance;
- no assumption that different lunar body-fixed frames are interchangeable.

### Perturbation forces

- Moon-centred differential Earth and Sun point-mass gravity;
- cannonball solar radiation pressure;
- finite apparent Sun/Moon eclipse geometry including partial shadow.

### Lunar surface geometry

- NASA Goddard MOON_PA and PDS LOLA terrain ingestion;
- terrain-aware clearance and impact;
- body-fixed ground tracks;
- surface-site access, elevation and slant range;
- optional sampled terrain line of sight;
- regional/global coverage dwell and revisit statistics;
- Earth visibility with lunar occultation.

### Orbit design and operations analysis

- eccentricity vector and angular-momentum vector;
- modified equinoctial elements;
- singularity-aware orbital history;
- secular drift and bounded-oscillation statistics;
- automated coarse and coarse-to-fine frozen/stability search;
- finite-difference sensitivity with step-size diagnostics;
- differential correction with explicit convergence/failure reporting;
- preliminary impulsive station-keeping estimates;
- transparent per-candidate metrics and ranking terms.

## Frozen-orbit search

The search layer does not define a frozen orbit as merely avoiding impact. It evaluates periselene/aposelene spread, eccentricity magnitude variation, eccentricity-vector secular drift and bounded residual motion, apsidal evolution where physically defined, orbital-plane evolution, reference-radius altitude, optional terrain clearance and impact-free lifetime.

Configuration example:

```toml
[search]
semi_major_axis_altitudes_km = [90.0, 110.0]
eccentricities = [0.005, 0.01]
inclinations_deg = [85.0, 90.0, 95.0]
raan_deg = [27.0]
periapsis_deg = [90.0, 270.0]
duration_s = 10800.0
sample_count = 65
minimum_reference_altitude_km = 20.0
```

The full Python API supports deterministic coarse-to-fine refinement, configurable hard constraints, parallel execution for explicitly safe models, uncertainty realizations and stability maps.

## Gravity uncertainty

SHADR coefficient uncertainty fields are retained, but the package does not pretend they are a full covariance matrix.

Two paths are supported:

1. explicit seeded diagonal sampling with `assume_independent = true`;
2. archived covariance-derived GRGM1200A clone perturbations.

The standalone uncertainty CLI intentionally reports the gravity-field ensemble produced by `propagate_gravity_ensemble`. It does not silently mix additional deterministic force assumptions into that product.

## Gravity degree/order fidelity

Users do not need to default every propagation to degree 1200.

The fidelity API compares tested truncations with an explicit reference at the actual trajectory/location samples and reports absolute/relative acceleration error, radial/along-track/cross-track error where defined, final and maximum trajectory separation, velocity difference, periselene/eccentricity outcome differences, terrain-clearance and impact/lifetime differences, and measured runtime.

A tolerance policy can select the lowest **tested** degree/order meeting the configured accuracy requirement. The selection is explicitly tied to the supplied trajectory, epoch, location, duration, force environment and tolerance.

## Independent scientific validation

The repository includes a separate external validation campaign using public NASA/PDS/NAIF data and independent numerical software.

Key results from the recorded campaign:

| Validation | Result |
|---|---:|
| GRGM900C vs independent SHTOOLS gravity, worst tested through 600×600 | `1.13e-14` relative acceleration difference |
| GRGM1200B vs SHTOOLS at 1200×1200 | `1.18e-14` relative acceleration difference |
| Lunar frame rotation vs direct CSPICE | `0.0` matrix difference at tested epochs |
| Independent PDS LDEM_4 pixel decode | `0.0 m` elevation difference at tested pixels |
| Independent same-force propagation, 2 h | `0.000130 m` maximum position difference |
| Open-loop 600×600 propagation vs reconstructed LRO SPK, 6 h | `12.56 m` maximum position difference |

The reconstructed LRO comparison is deliberately reported as a physical-model residual rather than forced through an artificial pass threshold. The open-loop model does not reproduce the complete precision orbit-determination force, manoeuvre and estimation setup.

See [`docs/independent_validation.md`](docs/independent_validation.md) and [`results/independent_validation.json`](results/independent_validation.json).

## Tests

The repository contains **173 automated tests** across gravity, uncertainty, terrain, force models, frames, orbital mathematics, stability search, targeting, fidelity, access/coverage, configuration, orchestration and the CLI.

GitHub Actions runs the test matrix on Python 3.10, 3.12 and 3.13. The Python 3.12 job also runs deterministic scientific smoke examples, including the configured mission workflow.

## External data policy

Normal installation and CI do not require large NASA datasets or network access.

Download scripts are provided for the public products used by specific analyses:

- `scripts/download_example_mission_data.py`
- `scripts/download_grgm1200a.py`
- `scripts/download_grgm1200a_clones.py`
- `scripts/download_lola_pa_shape.py`
- `scripts/download_force_model_kernels.py`
- `scripts/download_groundtrack_kernels.py`
- `scripts/download_independent_validation_data.py`

Always preserve gravity-product, frame, kernel and terrain provenance with the results.

## Scientific documentation

- [`docs/mission_workflow.md`](docs/mission_workflow.md)
- [`docs/model.md`](docs/model.md)
- [`docs/forces.md`](docs/forces.md)
- [`docs/terrain.md`](docs/terrain.md)
- [`docs/orbital_analysis.md`](docs/orbital_analysis.md)
- [`docs/frozen_orbit_search.md`](docs/frozen_orbit_search.md)
- [`docs/targeting.md`](docs/targeting.md)
- [`docs/fidelity.md`](docs/fidelity.md)
- [`docs/access.md`](docs/access.md)
- [`docs/uncertainty.md`](docs/uncertainty.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/independent_validation.md`](docs/independent_validation.md)

## Intended use

The package is intended for research, numerical experiments and preliminary lunar mission design.

It can support low lunar orbit trade studies, frozen/stability candidate screening, force-model and gravity-degree selection, terrain-clearance studies, site access and coverage analysis, sensitivity and local targeting, preliminary station-keeping comparisons, gravity-field uncertainty studies, and reproducible propagation experiments.

Operational flight dynamics, navigation, orbit determination, manoeuvre execution planning and mission-specific assurance require additional models, estimation, uncertainty treatment, verification and operational constraints beyond this package.

## License

MIT.
