# Gravity model

The repository contains two lunar gravity implementations:

1. a low-degree central-plus-J2 model used for analytical secular-rate validation;
2. a fully normalized spherical-harmonic evaluator intended for GRAIL SHADR products such as GRGM1200A.

## GRGM1200A metadata

The NASA PDS GRGM1200A product label identifies the field as a degree-and-order 1200 model derived from the full GRAIL data set. The archived coefficients are fully normalized using the geodesy 4π convention. The label specifies:

- reference radius: 1738.0 km
- GM: 4902.80011526323 km³/s²
- maximum degree/order: 1200/1200
- coordinate system: lunar body-fixed principal-axes frame defined using DE430
- product ID: `GGGRX_1200A_SHA.TAB`

Official product directory:

`https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/`

Product label:

`https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.lbl`

The coefficient file is intentionally not committed to this repository. It is approximately 88 MB and can be downloaded with `scripts/download_grgm1200a.py`.

## Potential

For body-fixed spherical coordinates radius `r`, latitude `φ`, and east-positive longitude `λ`, the implemented positive gravitational potential is

```text
U = GM/r Σ[n=0..N] (R/r)^n Σ[m=0..min(n,M)]
    P̄_nm(sin φ) [C̄_nm cos(mλ) + S̄_nm sin(mλ)]
```

`P̄_nm` uses geodesy 4π normalization and excludes the Condon-Shortley phase. `C̄_00 = 1` is inserted by the parser because the GRAIL SHADR coefficient table begins at degree 1.

The Cartesian acceleration is obtained from the analytical radial, latitude, and longitude derivatives of this potential. The normalized associated Legendre functions and their latitude derivatives are computed by a forward recursion that operates directly on normalized values. This avoids the factorial overflow that would occur if degree-1200 unnormalized functions were formed first.

## J2 relationship

For the same 4π convention, a pure J2 field is represented by

```text
C̄_20 = -J2 / sqrt(5)
```

The test suite verifies that the spherical-harmonic evaluator using this `C̄_20` reproduces the independent closed-form J2 acceleration to floating-point precision.

## Tesseral and sectoral terms

Terms with `m > 0` depend on longitude and therefore must be evaluated in the gravity model's body-fixed frame. The repository provides:

- `gravity_acceleration_body_fixed(...)`
- `gravity_acceleration_inertial(...)`
- `spice_rotation_provider(...)`

The inertial wrapper performs this sequence:

```text
r_inertial -> body-fixed rotation -> harmonic gravity -> inverse rotation -> a_inertial
```

The rotation provider is explicit. The library does not silently assume a lunar spin rate or invent a principal-axes orientation model.

For flight propagation, load an appropriate SPICE kernel set, construct a rotation provider for the chosen inertial and lunar body-fixed frames, and pass the resulting acceleration to `propagate_with_acceleration(...)`.

## Mascons

There is no separate collection of point-mass mascon objects. GRAIL high-degree spherical harmonics encode the spatial gravity anomalies associated with lunar mass concentrations. Consequently, using the actual GRGM1200A `C̄_nm` and `S̄_nm` coefficients includes mascon signatures through the gravity field itself.

## Surface handling

The gravity model's 1738.0 km reference radius is not treated as a physical surface. Orbit termination continues to use a separate 1737.4 km mean-radius collision boundary. This is a deliberately simple collision model, not terrain-aware lunar topography.

## Exclusions

The repository does not currently include:

- a bundled GRGM1200A coefficient file
- automatic SPICE kernel acquisition
- lunar topography collision geometry
- Earth/Sun third-body gravity
- solar radiation pressure
- tidal time-variable gravity
- covariance propagation
- force-model estimation

Those omissions are documented so the repository is not mistaken for mission-grade flight dynamics software.
