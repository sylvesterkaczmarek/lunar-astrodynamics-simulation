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

- covariance-derived GRGM1200A clone fields supplied from the official PDS archive;
- explicitly acknowledged independent Gaussian coefficient perturbations based on archived sigma fields.

The independent mode requires `assume_independent=True` because coefficient uncertainties do not contain off-diagonal covariance. For correlated GRGM1200A studies, covariance-derived clone fields are preferred. See [`uncertainty.md`](uncertainty.md).

## Validation strategy

The test suite covers central gravity, independent C20/J2 agreement, Cartesian finite-difference gradients, zonal/tesseral/sectoral terms, equatorial and polar geometry, exact-axis limits, pole crossings, degree/order truncation, degree-1200 finiteness, and inertial/body-fixed rotation consistency.

Uncertainty tests additionally cover SHADR sigma retention, validation and truncation, seeded diagonal sampling, mandatory opt-in to independence, clone archive URL mapping, coefficient-only clone loading, incomplete-clone rejection, percentile summaries, impact fractions, and end-to-end propagation through multiple gravity realizations.

## Numerical complexity

Direct harmonic synthesis is `O(N^2)` and maintains triangular arrays for `Pbar_nm`, its latitude derivative, and `Qbar_nm`. `scripts/benchmark_harmonics.py` reports timing without imposing a variable CI speed threshold.

Ensemble propagation scales with the number of supplied gravity realizations and currently executes members serially for deterministic, transparent behavior.

## Mascons and surface handling

Mascon signatures are represented through high-degree GRAIL coefficients rather than separate point masses.

The gravity reference radius is not a physical surface. Orbit termination still uses a spherical mean-radius collision boundary. Reported altitude uncertainty is therefore radial altitude uncertainty, not terrain-clearance uncertainty.

## Current scientific limitations

- Degree/order 1200 is the current tested high-degree target; this is not an arbitrary ultra-high-degree scaled-Clenshaw implementation.
- No claim is made for spherical-harmonic convergence below a model's applicable exterior region or Brillouin sphere.
- The package does not construct the complete GRGM1200A covariance matrix internally.
- Correlated gravity uncertainty is represented by externally supplied covariance-derived clone realizations.
- Diagonal coefficient sampling omits correlations and requires explicit acknowledgement.
- The nominal GRGM1200A and clone datasets are not bundled.
- SPICE kernels and compatible lunar principal-axes frames remain caller supplied.
- Lunar topography, Earth/Sun third-body gravity, solar radiation pressure, and time-variable tides are not yet included.
- Spacecraft state covariance propagation and orbit determination are not included.
- High-degree synthesis is CPU/NumPy based and is not certified flight-dynamics software.

These exclusions keep gravity-field uncertainty results from being mistaken for a complete mission uncertainty budget.
