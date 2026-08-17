# Gravity model

The repository contains two lunar gravity implementations:

1. a low-degree central-plus-J2 model used for independent analytical secular-rate validation;
2. a fully normalized spherical-harmonic evaluator intended for GRAIL SHADR products such as GRGM1200A.

The spherical-harmonic evaluator is designed for body-fixed lunar gravity synthesis through the degree/order 1200 scale of GRGM1200A. Its Cartesian acceleration evaluation is explicitly pole-safe: it does not move an evaluation point away from the rotation axis and it does not suppress tesseral or sectoral terms near a pole.

## Authoritative conventions

The implementation follows the NASA/PDS SHADR spherical-harmonic convention documented for GRAIL data products:

- positive gravitational potential `GM/r` plus spherical-harmonic corrections;
- east-positive longitude;
- normalized gravity coefficients when the SHADR normalization state is `1`;
- normalization factor

```text
Pi_nm^2 = (2 - delta_0m) (2n + 1) (n-m)! / (n+m)!
```

so that the normalized coefficients and functions use the geodesy 4pi convention employed by the archived GRAIL products. In particular,

```text
Cbar_20 = -J2 / sqrt(5)
```

for a pure J2 field.

Primary references used to audit the implementation are:

- NASA/PDS SHADR Software Interface Specification: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/document/shadr.htm
- GRAIL Data Product SIS: https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/document/dpsis.htm
- NASA CR-147478, *Pines nonsingular gravitational potential derivation, description and implementation*: https://ntrs.nasa.gov/citations/19760011100
- Holmes and Featherstone (2002), *A unified approach to the Clenshaw summation and the recursive computation of very high degree and order normalised associated Legendre functions*, DOI 10.1007/s00190-002-0216-2.

The Pines reference is used as an independent statement of the underlying numerical requirement: the physical Cartesian gravity field must not inherit the spherical-coordinate singularity at the poles. This implementation removes that singular factor analytically while retaining the GRAIL coefficient representation.

## GRGM1200A metadata

The NASA PDS GRGM1200A product label identifies the field as a degree-and-order 1200 model derived from the full GRAIL data set. The archived coefficients are fully normalized using the geodesy 4pi convention. The label specifies:

- reference radius: 1738.0 km
- GM: 4902.80011526323 km^3/s^2
- maximum degree/order: 1200/1200
- coordinate system: lunar body-fixed principal-axes frame defined using DE430
- product ID: `GGGRX_1200A_SHA.TAB`

Official product directory:

`https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/`

Product label:

`https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.lbl`

The coefficient file is intentionally not committed to this repository. It is approximately 88 MB and can be downloaded with `scripts/download_grgm1200a.py`.

## Potential

For body-fixed spherical coordinates radius `r`, latitude `phi`, and east-positive longitude `lambda`, the implemented positive gravitational potential is

```text
U = GM/r SUM[n=0..N] (R/r)^n SUM[m=0..min(n,M)]
    Pbar_nm(sin(phi)) [Cbar_nm cos(m lambda) + Sbar_nm sin(m lambda)]
```

`Pbar_nm` uses geodesy 4pi normalization and excludes the Condon-Shortley phase. `Cbar_00 = 1` is inserted by the parser because the GRAIL SHADR coefficient table begins at degree 1.

The SHADR reader follows the PDS fixed-column offsets directly: a 137-byte header data row within a 244-byte header block and 107-byte coefficient data rows within 122-byte coefficient blocks. Coefficient rows are indexed by their explicit degree and order because the SHADR specification does not require ordering or completeness.

## Normalized Legendre recursion

Let

```text
x = sin(phi)
t = cos(phi)
```

The normalized associated Legendre functions are evaluated directly in normalized form. The diagonal and subdiagonal recurrences are

```text
Pbar_nn = sqrt((2n+1)/(2n)) t Pbar_(n-1,n-1)

Pbar_(n,n-1) = sqrt(2n+1) x Pbar_(n-1,n-1)
```

and for `0 <= m <= n-2`,

```text
Pbar_nm = a_nm x Pbar_(n-1,m) - b_nm Pbar_(n-2,m)

a_nm = sqrt((4n^2 - 1)/(n^2 - m^2))

b_nm = sqrt(
    (2n+1) ((n-1)^2 - m^2)
    / ((2n-3) (n^2 - m^2))
)
```

Latitude derivatives are propagated alongside the functions by differentiating the same recurrences. This avoids forming unnormalized high-degree functions and the associated factorial overflow.

## Pole-safe longitudinal gradient

The conventional spherical gradient contains

```text
a_lambda = (1 / (r cos(phi))) dU/dlambda
```

which is a coordinate singularity at `cos(phi) = 0`. The potential and Cartesian gravitational field themselves remain finite.

The implementation therefore never numerically divides the longitudinal harmonic sum by `cos(phi)`. For every `m >= 1`, it synthesizes the auxiliary function

```text
Qbar_nm = Pbar_nm / cos(phi)
```

directly by recurrence. The finite pole value is obtained from the recurrence itself rather than from a nearby proxy coordinate.

The base and recurrence relations are

```text
Qbar_11 = sqrt(3)

Qbar_nn = sqrt((2n+1)/(2n)) Pbar_(n-1,n-1)

Qbar_(n,n-1) = sqrt(2n+1) x Qbar_(n-1,n-1)

Qbar_nm = a_nm x Qbar_(n-1,m) - b_nm Qbar_(n-2,m)
              for 1 <= m <= n-2
```

The longitudinal acceleration is then accumulated as

```text
a_lambda = GM/r^2 SUM_n (R/r)^n SUM_m
    Qbar_nm m [-Cbar_nm sin(m lambda) + Sbar_nm cos(m lambda)]
```

with no division by `cos(phi)`.

At an exact pole, longitude is undefined as a coordinate. The implementation fixes the local longitude basis to the body-fixed x/y axes by taking `lambda = 0` at the axis. The resulting Cartesian acceleration remains unique. The `m=1` terms supply the finite horizontal pole gravity; higher-order non-zonal terms approach their proper limits through the `Qbar_nm` recurrence.

This replaces the previous method that displaced exact/near-pole evaluations to an artificial nearby point.

## Cartesian acceleration

The radial, latitude, and pole-safe longitudinal components are transformed to Cartesian body-fixed acceleration with

```text
e_r      = [cos(phi) cos(lambda), cos(phi) sin(lambda), sin(phi)]
e_phi    = [-sin(phi) cos(lambda), -sin(phi) sin(lambda), cos(phi)]
e_lambda = [-sin(lambda), cos(lambda), 0]

a_bf = a_r e_r + a_phi e_phi + a_lambda e_lambda
```

For numerical robustness near the axis, `sin(phi)`, `cos(phi)`, `sin(lambda)`, and `cos(lambda)` are computed directly from the Cartesian position components rather than reconstructed from rounded angular values.

## Body-fixed and inertial frames

Terms with `m > 0` depend on longitude and therefore must be evaluated in the gravity model's body-fixed frame. The repository provides:

- `gravity_acceleration_body_fixed(...)`
- `gravity_acceleration_inertial(...)`
- `spice_rotation_provider(...)`

The inertial wrapper performs this sequence:

```text
r_inertial -> body-fixed rotation -> harmonic gravity -> inverse rotation -> a_inertial
```

The rotation provider is explicit. The library does not silently assume a lunar spin rate or invent a principal-axes orientation model.

For GRGM1200A, users should choose a lunar principal-axes frame compatible with the DE430-defined gravity solution and their loaded SPICE kernel set. NAIF lunar frame products should be treated as part of the numerical model provenance, not as interchangeable labels.

## Validation strategy

The automated tests intentionally combine independent checks rather than testing one implementation path against itself.

They cover:

- normalized low-degree Legendre values;
- pure central gravity;
- normalized `C20` versus an independently coded closed-form J2 acceleration;
- analytical Cartesian acceleration versus a central finite-difference gradient of the independently evaluated potential;
- zonal, tesseral, and sectoral synthetic fields;
- equatorial, mid-latitude, low-altitude, and higher-altitude positions;
- points very close to the north and south poles;
- exact north and south rotation-axis positions;
- convergence to the same Cartesian pole field from multiple approach azimuths;
- continuity while crossing a pole despite the longitude coordinate jump;
- degree/order truncation equivalence;
- finite normalized-function and acceleration evaluation through degree/order 1200;
- inertial/body-fixed rotation consistency, including a pole case.

Finite-difference tests are deliberately based on Cartesian perturbations of the potential, so they do not reuse the spherical acceleration derivative formula being tested.

## Numerical complexity and performance

Direct synthesis is `O(N^2)` in maximum degree. Acceleration evaluation maintains three triangular work arrays at high degree:

- `Pbar_nm`
- `dPbar_nm/dphi`
- `Qbar_nm`

The extra `Qbar` array is the cost of removing the longitudinal pole singularity without altering the coefficient representation.

`scripts/benchmark_harmonics.py` provides a reproducible timing harness for high-degree acceleration evaluation. Timing is reported rather than enforced as a CI pass/fail threshold because shared CI hardware is not a stable performance reference.

## Mascons

There is no separate collection of point-mass mascon objects. GRAIL high-degree spherical harmonics encode the spatial gravity anomalies associated with lunar mass concentrations. Consequently, using the actual GRGM1200A `Cbar_nm` and `Sbar_nm` coefficients includes mascon signatures through the gravity field itself.

## Surface handling

The gravity model's 1738.0 km reference radius is not treated as a physical surface. Orbit termination continues to use a separate 1737.4 km mean-radius collision boundary. This is a deliberately simple collision model, not terrain-aware lunar topography.

## Current scientific limitations

The gravity engine is a research implementation and has explicit limits:

- Degree/order 1200 is the current tested target. The direct normalized recurrence is not an extended-range or arbitrary-ultra-high-degree implementation such as a full scaled Clenshaw synthesis.
- No claim is made for reliable spherical-harmonic convergence below a gravity model's applicable exterior region or Brillouin sphere.
- The repository does not yet propagate GRAIL coefficient covariance or clone-field uncertainty.
- The repository does not currently bundle GRGM1200A coefficients.
- SPICE kernels and compatible lunar principal-axes frames remain caller supplied.
- Lunar topography is not yet used for collision geometry.
- Earth/Sun third-body gravity is not yet included.
- Solar radiation pressure is not yet included.
- Tidal time-variable gravity is not included.
- Covariance propagation and orbit determination are not included.
- High-degree synthesis is CPU/NumPy based and is not optimized as flight dynamics production software.

These exclusions are documented so the library is not mistaken for certified or flight-qualified dynamics software.
