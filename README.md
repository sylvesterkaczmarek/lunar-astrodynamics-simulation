# Lunar Astrodynamics Simulation

![Lunar Astrodynamics Simulation](assets/social/github-social-card-lunar-astrodynamics.png)

[![CI](https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Validated Python simulation of lunar orbital dynamics from central gravity and J2 through high-degree GRAIL spherical harmonics. The code reads NASA PDS SHADR gravity products, evaluates fully normalized `Cnm/Snm` fields in the lunar body-fixed frame with a pole-safe analytical Cartesian gradient, and supports explicit rotation back to an inertial propagation frame.

## At a glance

```mermaid
flowchart LR
    A[GRAIL SHADR coefficients] --> B[4pi-normalized parser]
    B --> C[Body-fixed harmonic potential]
    C --> D[Pole-safe analytical gravity gradient]
    D --> E[Frame rotation]
    E --> F[Inertial propagation]
    G[J2 closed form] --> H[Secular-rate validation]
    B --> I[C20 versus J2 validation]
```

The repository deliberately keeps the low-degree J2 benchmark because it provides an independent analytical check on the numerical machinery. The spherical-harmonic layer then adds longitude-dependent tesseral and sectoral gravity terms, including the high-degree gravity signatures associated with lunar mascons.

## Implemented models

- central lunar gravity
- closed-form J2 perturbation
- geodesy 4pi-normalized spherical harmonics
- arbitrary `Cnm/Snm` degree and order truncation
- NASA PDS SHADR parsing using the published fixed-column record layout
- GRGM1200A metadata and download tooling
- pole-safe body-fixed harmonic acceleration, including exact rotation-axis evaluation
- body-fixed to inertial force transformation
- optional SPICE frame transformations
- terminal mean-radius surface event

The evaluator is stress-tested through degree 1200, including equatorial, mid-latitude, near-pole, and exact north/south-axis evaluations. The longitudinal gravity term is evaluated with a direct recurrence for `Pbar_nm / cos(phi)` rather than dividing by `cos(phi)` or displacing a pole evaluation to an artificial nearby point. The SHADR reader is regression-tested with byte-faithful 244-byte header and 122-byte coefficient records matching the PDS specification, including arbitrary coefficient ordering, malformed-row rejection, and duplicate detection. The 88 MB GRGM1200A coefficient table remains an external NASA data product and is not copied into this repository.

## Validation

The current automated suite contains **65 tests**. The gravity-specific validation includes:

- normalized `C20` acceleration versus an independent closed-form J2 implementation;
- analytical Cartesian acceleration versus an independent Cartesian finite-difference gradient of the potential;
- zonal, tesseral, and sectoral synthetic fields;
- equatorial, mid-latitude, low-altitude, and higher-altitude locations;
- very near both poles and exactly on both rotation-axis poles;
- convergence to the same Cartesian pole field from multiple approach azimuths;
- continuity while crossing a pole despite the longitude coordinate jump;
- degree/order truncation equivalence;
- finite normalized-function and acceleration evaluation through degree/order 1200;
- body-fixed/inertial rotation consistency, including an exact-pole case.

Two retained numerical checks are:

| Check | Result |
|---|---:|
| Normalized `C20` acceleration versus closed-form J2 | `1.78e-16` relative difference |
| Tesseral analytical acceleration versus finite-difference potential gradient | `9.86e-10` relative difference |

The original 40-orbit J2 regression remains unchanged:

| Quantity | Analytical | Numerical | Relative difference |
|---|---:|---:|---:|
| RAAN rate | -0.773273 deg/day | -0.773527 deg/day | 0.033% |
| Periapsis-argument rate | 0.820180 deg/day | 0.817987 deg/day | 0.267% |

See [`results/j2_validation.json`](results/j2_validation.json), [`results/harmonic_validation.json`](results/harmonic_validation.json), and [`docs/model.md`](docs/model.md) for the equations, pole treatment, conventions, references, and limitations.

## Quick start

```bash
git clone https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation.git
cd lunar-astrodynamics-simulation
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
python -m pytest
python examples/j2_precession.py --orbits 40
python examples/harmonic_validation.py
```

Windows PowerShell users can activate the environment with `.venv\Scripts\Activate.ps1`.

## Use GRGM1200A

Download the official NASA PDS product:

```bash
python scripts/download_grgm1200a.py
```

Then evaluate the full degree/order 1200 field at a body-fixed point:

```bash
python examples/grgm1200a_gravity.py \
  --model data/gggrx_1200a_sha.tab \
  --degree 1200 \
  --position-km 1900 200 300
```

The archived GRGM1200A metadata used by the code are GM `4902.80011526323 km^3/s^2`, reference radius `1738.0 km`, geodesy 4pi normalization, and a DE430-defined lunar principal-axes body-fixed frame.

## High-degree timing

A timing harness is provided for measuring acceleration synthesis without turning variable shared-runner timings into a CI pass/fail criterion:

```bash
python scripts/benchmark_harmonics.py --degree 1200 --repetitions 5
```

It can also benchmark an externally downloaded SHADR model with `--model`.

## Inertial propagation

High-degree lunar gravity must not be frozen in an inertial frame. Supply an explicit body-fixed-from-inertial rotation:

```python
from lunar_astrodynamics import (
    gravity_acceleration_inertial,
    propagate_with_acceleration,
    spice_rotation_provider,
)

rotation = spice_rotation_provider(
    "J2000",
    "YOUR_LOADED_LUNAR_PA_FRAME",
    et_offset_s=et0,
)

acceleration = lambda t, r: gravity_acceleration_inertial(
    t,
    r,
    model,
    rotation,
    max_degree=1200,
)

solution = propagate_with_acceleration(
    initial_state,
    duration_s,
    acceleration,
    collision_radius_m=1_737_400.0,
)
```

The frame name is intentionally not hard-coded. It must correspond to the SPICE kernel set loaded by the user.

## Mascons

The implementation does not approximate mascons as separate point masses. Their gravity signatures are represented by the high-degree GRAIL spherical-harmonic coefficients. This is why `mascons` and `spherical-harmonics` are accurate repository topics once an actual GRAIL coefficient set is used.

## Repository layout

```text
lunar-astrodynamics-simulation/
├── .github/workflows/ci.yml
├── assets/
│   ├── results/j2_precession.svg
│   └── social/github-social-card-lunar-astrodynamics.png
├── docs/
│   ├── model.md
│   └── reproducibility.md
├── examples/
│   ├── grgm1200a_gravity.py
│   ├── harmonic_validation.py
│   └── j2_precession.py
├── scripts/
│   ├── benchmark_harmonics.py
│   └── download_grgm1200a.py
├── src/lunar_astrodynamics/
│   ├── analysis.py
│   ├── constants.py
│   ├── dynamics.py
│   ├── elements.py
│   ├── frames.py
│   ├── harmonics.py
│   └── propagation.py
├── tests/
├── CITATION.cff
├── LICENSE
├── Makefile
├── pyproject.toml
└── README.md
```

## What this repository does not claim

This remains a research and validation implementation, not certified flight-dynamics software. Degree/order 1200 is the current tested high-degree target; the direct normalized recursion is not presented as an arbitrary ultra-high-degree extended-range/Clenshaw implementation. The repository does not bundle NASA's coefficient file, lunar topography, automatic SPICE kernels, third-body gravity, solar radiation pressure, covariance propagation, or orbit-determination estimation.

See [docs/model.md](docs/model.md) for the force model and [docs/reproducibility.md](docs/reproducibility.md) for exact checks.

## Cite this repository

If you use or adapt this repository, please cite:

> Kaczmarek, S. (2026). *Lunar Astrodynamics Simulation*. GitHub. https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation

```bibtex
@software{Kaczmarek_2026_Lunar_Astrodynamics_Simulation,
  author = {Sylvester Kaczmarek},
  title  = {Lunar Astrodynamics Simulation},
  year   = {2026},
  url    = {https://github.com/sylvesterkaczmarek/lunar-astrodynamics-simulation}
}
```

## License

MIT. See [LICENSE](LICENSE).

© **Sylvester Kaczmarek** · [https://www.sylvesterkaczmarek.com](https://www.sylvesterkaczmarek.com)
