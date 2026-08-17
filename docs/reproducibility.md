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
python examples/gravity_uncertainty.py --samples 16 --seed 20260817 --duration-days 1
python examples/terrain_clearance.py
python examples/nonsingular_analysis.py --orbits 10
python examples/frozen_orbit_search.py --quick \
  --output /tmp/frozen_search.json \
  --csv /tmp/frozen_search.csv \
  --map-csv /tmp/frozen_map.csv
```

The terrain example is self-contained and compares a mean-radius collision model with a synthetic gridded surface containing a 6 km mountain. The nonsingular example includes an exact circular-equatorial state, for which classical RAAN is intentionally rejected, and a nearly circular polar J2 propagation analysed with eccentricity vectors and modified equinoctial elements.

The frozen-orbit quick example is also self-contained. It performs a deterministic low-degree J2 coarse-to-fine search and is intended to verify the search workflow and metric plumbing, not to serve as a high-degree lunar frozen-orbit truth result.

## Stability-search reproducibility

A reproducible search should preserve more than the final ranked table. Store or record:

- the complete `StabilitySearchSpace`, including whether altitude means semimajor-axis altitude or absolute semimajor axis;
- periapsis parameterization: argument of periapsis or longitude of periapsis;
- initial anomaly range;
- the coarse grid and `RefinementSettings`;
- propagation duration and output sample count;
- integration method, tolerances and maximum step;
- `apsis_eccentricity_threshold`;
- all `StabilityConstraints`;
- every scale and weight in `StabilityRankingPolicy`;
- nominal `SearchDynamics.provenance()`;
- selected harmonic degree/order;
- all uncertainty-dynamics provenance and the adverse percentile;
- terrain product, terrain frame, prepared resolution and clearance-search resolution when terrain is used;
- worker count and whether each supplied force model was explicitly marked parallel-safe.

`StabilitySearchResult.write_json(...)` stores the search definition, settings, dynamics provenance and every candidate metric/ranking breakdown. `CoarseToFineSearchResult.write_json(...)` additionally stores the coarse and refined stages and the coarse seed candidate ids.

The grid and refinement algorithms do not use random sampling. Candidate ids are assigned in deterministic physical-state order, and physically duplicate circular/equatorial states are removed using their modified-equinoctial representation before propagation. Parallel execution uses ordered executor mapping so returned candidate ordering remains deterministic for force providers that themselves behave deterministically.

A ranking penalty is reproducible only together with its scales and weights. It is not a universal definition of a frozen orbit. Preserve the individual periselene/aposelene, eccentricity-vector, eccentricity, apsidal, plane, clearance and lifetime metrics when comparing studies.

See `docs/frozen_orbit_search.md` for the metric definitions, ranking convention and interpretation limits.

## Orbital-analysis conventions

For reproducible stability studies, record the Cartesian reference frame, gravitational parameter, reference radius and the element convention used in analysis.

The library's nonsingular scalar convention is the prograde tangent modified-equinoctial set `(p,f,g,h,k,L)`. It is singular at the exact 180-degree retrograde-equatorial limit. Classical RAAN and argument of periapsis are never filled with arbitrary values when undefined.

`orbit_history(...)` reports both raw evolution and a least-squares linear trend plus detrended residual oscillation. For each stability metric, preserve the propagation time span and sampling cadence because fitted drift and sampled oscillation bounds are properties of that analysis interval.

If terrain clearance is requested, also record the terrain product, its frame and the body-fixed transform. `orbit_history(...)` evaluates terrain clearance at the supplied trajectory samples; refined clearance extrema and impact roots remain the responsibility of `propagate_with_terrain(...)` / `analyze_terrain_clearance(...)`.

See `docs/orbital_analysis.md` for the exact MEE definitions and singularity policy.

## External GRGM1200A data

Download the official NASA PDS file with:

```bash
python scripts/download_grgm1200a.py
```

The nominal SHADR table remains external to Git. Covariance-derived clone perturbations can be downloaded selectively with `scripts/download_grgm1200a_clones.py`.

For a high-degree stability search, the SHADR model must be paired with an appropriate caller-loaded lunar body-fixed transformation. The search API records the supplied frame and harmonic truncation but deliberately does not infer that an arbitrary lunar SPICE frame is compatible with the gravity solution.

## External LOLA terrain data

The recommended global gravity/topography source is NASA Goddard's 2024 LOLA MOON_PA global shape grid. Download the 64 pixels/degree gridline product with:

```bash
python scripts/download_lola_pa_shape.py --registration gridline
```

For GMT/netCDF support install:

```bash
python -m pip install -e .[terrain]
```

A smaller explicit-resolution working grid can be prepared with:

```bash
python scripts/prepare_lola_pa_grid.py \
  data/LDEM64_PA_gridline_202405.grd \
  --stride 8 \
  --output data/lola_moon_pa_8ppd.npz
```

The prepared NPZ preserves terrain frame, reference radius, registration and source metadata.

## Independent LOLA PDS reference check

The repository contains an external validation utility for the standard NASA/PDS `LDEM_4` global product:

```bash
python scripts/validate_lola_pds_reference.py
```

It downloads the official `LDEM_4.IMG` and `LDEM_4.LBL`, decodes them using `load_lola_pds_global_gdr(...)`, and prints selected reference grid values.

On 17 August 2026 the observed product metadata and values were:

```text
frame=MEAN EARTH/POLAR AXIS OF DE421
reference_radius_m=1737400.0
shape=(720, 1440)
lat=0.125,lon=0.125,elevation_m=-796.0
lat=0.125,lon=90.125,elevation_m=-3814.5
lat=0.125,lon=180.125,elevation_m=2432.0
lat=45.125,lon=45.125,elevation_m=-699.0
lat=-45.125,lon=315.125,elevation_m=-301.0
lat=89.875,lon=0.125,elevation_m=-119.5
lat=-89.875,lon=180.125,elevation_m=172.0
```

These values were copied into `tests/test_lola_reference.py` as an offline indexing/scaling regression. Normal CI therefore does not require NASA network access. The live validation script can be run when archive-level revalidation is desired.

The PDS LDEM_4 grid is in the DE421 Mean Earth/Polar Axis frame. It is used to validate the file reader, not as an implicit substitute for the recommended DE421 Principal Axes terrain grid.

## SPICE Earth/Sun perturbation workflow

Install the optional SPICE dependency and download the small generic kernel set used by the force-model example:

```bash
python -m pip install -e .[spice]
python scripts/download_force_model_kernels.py
```

The script downloads:

```text
data/spice/naif0012.tls
data/spice/de440s.bsp
data/spice/pck00011.tpc
```

Run the four-level force comparison with an explicit UTC epoch:

```bash
python examples/force_model_comparison.py \
  --kernel-dir data/spice \
  --epoch-utc 2026-08-17T00:00:00 \
  --duration-days 7 \
  --samples 1001 \
  --output results/force_model_example.json
```

`SpiceEphemeris` converts that epoch to ET using the loaded leap-seconds kernel and records the numerical ET, inertial frame, Moon observer, aberration mode and loaded kernel list. Integration time is always interpreted as elapsed seconds from that fixed ET epoch.

The force context deliberately uses geometric SPICE positions (`abcorr=NONE`). Earth and Sun are returned relative to the Moon in the chosen inertial frame. The low-degree comparison example evaluates J2 in `IAU_MOON` using `pck00011.tpc` and rotates its acceleration back to J2000.

The archive-level two-day SPICE validation performed on 17 August 2026 is stored in `results/force_model_spice_validation.json`. Routine CI does not download NAIF kernels; it tests third-body, SRP, eclipse and SPICE-context behaviour using analytical and synthetic fixtures.

## Frame provenance

A reproducible combined GRAIL/LOLA/perturbation run should record each frame transformation independently. The recommended Goddard terrain grid is `MOON_PA_DE421`, GRGM1200A is a DE430 gravity solution, and the low-degree force-isolation example uses `IAU_MOON`. These names are not interchangeable.

For science propagation and search, record:

- SPICE kernel filenames and versions;
- inertial frame;
- gravity body-fixed frame;
- terrain body-fixed frame;
- UTC and numerical ET epoch;
- SPICE aberration correction;
- gravity product and truncation;
- terrain product and prepared resolution;
- enabled third bodies and their mass parameters;
- SRP mass, area, reflectivity coefficient and eclipse model when enabled;
- orbital-analysis representation and reference radius;
- propagation sampling cadence, integration tolerances and maximum step;
- stability search grid, refinement, constraints and ranking policy;
- uncertainty realizations and adverse percentile;
- parallel worker count and force-provider thread-safety declaration.

## What CI verifies

Routine CI avoids large external GRGM1200A and LOLA products and avoids network-dependent NAIF kernel downloads. It verifies the implementations with analytical, synthetic and frozen-reference fixtures, including:

- two-body and J2 regressions;
- high-degree spherical-harmonic validation through degree 1200;
- pole-safe gravity evaluation;
- SHADR uncertainty retention and reproducible diagonal sampling;
- GRGM1200A clone perturbation semantics and ensemble metrics;
- terrain bilinear interpolation and periodic longitude;
- exact-pole and pixel-polar-cap behaviour;
- explicit terrain-frame mismatch rejection;
- terrain-aware impact event root finding and impact geometry;
- GMT/netCDF terrain loading;
- PDS3 IMG/label scaling, byte order and coordinate reconstruction;
- selected reference elevations independently observed from official LOLA products;
- prepared NPZ metadata round trips;
- composable force summation and per-component diagnostics;
- differential third-body acceleration against analytic geometries;
- distant-third-body inverse-cube limiting behaviour;
- SPICE epoch, target and kernel-provenance behaviour with a deterministic mock;
- SRP one-AU magnitude and inverse-square distance scaling;
- full sunlight, full lunar umbra, annular and partial-disk eclipse cases;
- continuous penumbra transitions;
- classical-element round trips where classical angles are defined;
- modified-equinoctial round trips from circular/equatorial through polar and 179-degree inclined cases;
- explicit rejection of singular classical angles and the prograde-MEE retrograde-equatorial state conversion limit;
- eccentricity-vector and orbital-plane histories;
- periselene/aposelene and reference-altitude analysis;
- separate sampled terrain clearance;
- secular drift and detrended bounded-oscillation statistics;
- stability metrics beyond survival, including periselene/aposelene spread and eccentricity-vector drift;
- physical deduplication of singular classical search parameter combinations;
- deterministic serial/parallel candidate ordering for parallel-safe dynamics;
- uncertainty realization retention and adverse summaries;
- terrain-clearance search constraints;
- coarse-to-fine refinement and stability-map generation;
- harmonic-degree provenance and JSON/CSV search output;
- surface-safe public low-lunar default ranges;
- end-to-end terrain-clearance, nonsingular-analysis and frozen-orbit-search smoke workflows.

This keeps source-code verification deterministic while retaining separate scripts and recorded results for live archive validation.
