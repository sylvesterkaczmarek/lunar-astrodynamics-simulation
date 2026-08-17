# Gravity model

The repository contains two lunar gravity implementations:

1. a low-degree central-plus-J2 model used for independent analytical validation;
2. a fully normalized spherical-harmonic evaluator intended for GRAIL SHADR products such as GRGM1200A.

The spherical-harmonic evaluator is designed for body-fixed lunar gravity synthesis through degree/order 1200. Its Cartesian acceleration is explicitly pole-safe and its SHADR model object preserves archived coefficient-uncertainty fields for separate uncertainty analysis.

## Authoritative conventions

The implementation follows the NASA/PDS SHADR convention used by archived GRAIL products: positive `GM/r` potential, east-positive longitude, geodesy 4pi normalized coefficients, and no Condon-Shortley phase.

The normalization factor is

```text
Pi_nm^2 = (2 - delta_0m) (2n + 1) (n-m)! / (n+m)!
```

and a pure J2 field satisfies

```text
Cbar_20 = -J2 / sqrt(5)
```

Primary references:

- NASA/PDS SHADR SIS: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/document/shadr.htm
- GRAIL Data Product SIS: https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/document/dpsis.htm
- NASA CR-147478, *Pines nonsingular gravitational potential derivation, description and implementation*: https://ntrs.nasa.gov/citations/19760011100
- Holmes and Featherstone (2002), DOI 10.1007/s00190-002-0216-2.

## GRGM1200A metadata

The archived GRGM1200A product uses reference radius `1738.0 km`, GM `4902.80011526323 km^3/s^2`, degree/order `1200/1200`, geodesy 4pi normalization, and a DE430-defined lunar principal-axes frame. The nominal coefficient file is external and can be retrieved with `scripts/download_grgm1200a.py`.

Official product directory:

`https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/`

## Potential

For body-fixed radius `r`, latitude `phi`, and east-positive longitude `lambda`,

```text
U = GM/r SUM[n=0..N] (R/r)^n SUM[m=0..min(n,M)]
    Pbar_nm(sin(phi)) [Cbar_nm cos(m lambda) + Sbar_nm sin(m lambda)]
```

`Cbar_00 = 1` is inserted by the parser because the archived table begins at degree 1. The reader follows the PDS fixed-column offsets and indexes coefficients by explicit `(n,m)` values.

The same SHADR records contain `C UNCERTAINTY` and `S UNCERTAINTY`. These are retained as `sigma_c` and `sigma_s`; the header's `UNCERTAINTY IN CONSTANT` is retained as `mu_sigma_m3_s2`. Their statistical use is documented in [`uncertainty.md`](uncertainty.md).

## Normalized Legendre recursion

Let `x = sin(phi)` and `t = cos(phi)`. The normalized associated Legendre functions are evaluated directly:

```text
Pbar_nn = sqrt((2n+1)/(2n)) t Pbar_(n-1,n-1)
Pbar_(n,n-1) = sqrt(2n+1) x Pbar_(n-1,n-1)
Pbar_nm = a_nm x Pbar_(n-1,m) - b_nm Pbar_(n-2,m)
```

with

```text
a_nm = sqrt((4n^2 - 1)/(n^2 - m^2))
b_nm = sqrt((2n+1)((n-1)^2-m^2) / ((2n-3)(n^2-m^2)))
```

Latitude derivatives are propagated by differentiating the same recurrences. Direct normalized evaluation avoids factorial overflow at degree 1200.

## Pole-safe longitudinal gradient

The conventional term

```text
a_lambda = (1 / (r cos(phi))) dU/dlambda
```

has a spherical-coordinate singularity at the poles. The physical Cartesian field remains finite.

The implementation synthesizes

```text
Qbar_nm = Pbar_nm / cos(phi)
```

without numerical division, using

```text
Qbar_11 = sqrt(3)
Qbar_nn = sqrt((2n+1)/(2n)) Pbar_(n-1,n-1)
Qbar_(n,n-1) = sqrt(2n+1) x Qbar_(n-1,n-1)
Qbar_nm = a_nm x Qbar_(n-1,m) - b_nm Qbar_(n-2,m)
```

for the appropriate orders. The longitude acceleration is then

```text
a_lambda = GM/r^2 SUM_n (R/r)^n SUM_m
    Qbar_nm m [-Cbar_nm sin(m lambda) + Sbar_nm cos(m lambda)]
```

with no `1/cos(phi)` operation. At the exact rotation axis, the local longitude basis is fixed to the body x/y axes and the unique Cartesian limiting field is returned. This replaced the earlier artificial proxy-point treatment.

## Cartesian and inertial frames

The spherical components are transformed with

```text
e_r      = [cos(phi) cos(lambda), cos(phi) sin(lambda), sin(phi)]
e_phi    = [-sin(phi) cos(lambda), -sin(phi) sin(lambda), cos(phi)]
e_lambda = [-sin(lambda), cos(lambda), 0]
```

Direction cosines near the axis are derived directly from Cartesian position components.

Non-zonal gravity must be evaluated in the model's body-fixed frame. The inertial path is

```text
r_inertial -> body-fixed rotation -> harmonic gravity -> inverse rotation -> a_inertial
```

The rotation provider is explicit. A GRGM1200A study should use and record a SPICE lunar principal-axes frame compatible with the gravity solution.

## Gravity-field uncertainty

Uncertainty handling is separate from deterministic force evaluation. `SphericalHarmonicModel` can carry SHADR uncertainty metadata, while `uncertainty.py` creates or loads alternative gravity realizations.

Two modes are supported:

- covariance-derived GRGM1200A clone coefficient perturbations supplied from the official PDS archive and added to a compatible nominal GRGM1200A model;
- explicitly acknowledged independent Gaussian coefficient perturbations based on archived sigma fields.

The clone files are deviations from the nominal solution rather than standalone lunar gravity models. `read_grgm1200a_clone(...)` therefore returns a `GravityCoefficientPerturbation`, and a separate application step constructs each complete gravity realization. The independent mode requires `assume_independent=True` because coefficient uncertainties do not contain off-diagonal covariance. For correlated GRGM1200A studies, nominal-plus-clone realizations are preferred. See [`uncertainty.md`](uncertainty.md).

## Validation strategy

The test suite covers central gravity, independent C20/J2 agreement, Cartesian finite-difference gradients, zonal/tesseral/sectoral terms, equatorial and polar geometry, exact-axis limits, pole crossings, degree/order truncation, degree-1200 finiteness, and inertial/body-fixed rotation consistency.

Uncertainty tests additionally cover SHADR sigma retention, validation and truncation, seeded diagonal sampling, mandatory opt-in to independence, clone archive URL mapping, coefficient-only clone-perturbation loading, incomplete-clone rejection, application of clone deltas to a nominal field, multi-clone realization loading, percentile summaries, impact fractions, and end-to-end propagation through multiple gravity realizations.

Terrain tests cover bilinear interpolation, longitude wrapping, antimeridian continuity, exact-pole behavior, pixel-registration polar caps, explicit frame mismatch rejection, terrain-impact root finding and geometry reporting, GMT/netCDF loading, PDS3 IMG/label decoding, prepared-grid metadata round trips, and selected reference elevations independently observed from the official NASA/PDS LDEM_4 product. See [`terrain.md`](terrain.md).

## Numerical complexity

Direct harmonic synthesis is `O(N^2)` and maintains triangular arrays for `Pbar_nm`, its latitude derivative, and `Qbar_nm`. `scripts/benchmark_harmonics.py` reports timing without imposing a variable CI speed threshold.

Ensemble propagation scales with the number of supplied gravity realizations and currently executes members serially for deterministic, transparent behavior.

Terrain clearance evaluation is dominated by frame transformation and local bilinear grid interpolation. `propagate_with_terrain(...)` also retains dense ODE output and performs a post-propagation clearance scan with local scalar refinement.

## Mascons and surface handling

Mascon signatures are represented through high-degree GRAIL coefficients rather than separate point masses.

The gravity reference radius is not treated as a physical lunar surface. Two collision models are available:

- `make_mean_radius_surface_event(...)` is the explicitly named spherical fallback;
- `make_terrain_impact_event(...)` and `propagate_with_terrain(...)` compare spacecraft radius with a local body-fixed terrain shape model.

The terrain model can use the NASA Goddard 2024 LOLA MOON_PA global shape grid or compatible external LOLA-derived grids. Because the recommended topography is `MOON_PA_DE421` and GRGM1200A is associated with DE430, the gravity and terrain rotations remain separate model inputs. The code rejects a terrain rotation whose declared frame does not equal `terrain.frame`.

See [`terrain.md`](terrain.md) for product provenance, interpolation rules and resolution limits.

## Current scientific limitations

- Degree/order 1200 is the current tested high-degree target; this is not an arbitrary ultra-high-degree scaled-Clenshaw implementation.
- No claim is made for spherical-harmonic convergence below a model's applicable exterior region or Brillouin sphere.
- The package does not construct the complete GRGM1200A covariance matrix internally.
- Correlated gravity uncertainty is represented by covariance-derived PDS clone perturbations applied to a compatible nominal GRGM1200A model.
- Diagonal coefficient sampling omits correlations and requires explicit acknowledgement.
- The nominal GRGM1200A, clone and LOLA terrain datasets are not bundled.
- SPICE kernels and compatible lunar principal-axes frames remain caller supplied.
- Terrain is represented as a radial gridded shape with bilinear interpolation; sub-grid relief and terrain uncertainty are not modeled.
- The recommended global PA terrain product is DE421 while GRGM1200A is DE430, so separate explicit frame transformations are required.
- Earth/Sun third-body gravity, solar radiation pressure, and time-variable tides are not yet included.
- Spacecraft state covariance propagation and orbit determination are not included.
- High-degree synthesis is CPU/NumPy based and is not certified flight-dynamics software.

These exclusions keep the simulation from being mistaken for a complete flight-clearance or mission uncertainty system.
