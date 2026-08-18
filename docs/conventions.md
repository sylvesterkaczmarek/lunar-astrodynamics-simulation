# Scientific conventions

This page collects the conventions most likely to affect interpretation or reproducibility. Specialist derivations and model details remain in the linked documentation.

## Units and state vectors

The scientific Python APIs use SI units unless an interface explicitly states otherwise:

- distance: metres;
- velocity: metres per second;
- time intervals: seconds;
- gravitational parameter: cubic metres per second squared;
- acceleration: metres per second squared;
- angles in the numerical APIs: radians.

TOML mission configuration uses degrees for fields explicitly named with the `_deg` suffix and kilometres only for fields explicitly named with the `_km` suffix.

Cartesian states are ordered

```text
[x, y, z, vx, vy, vz]
```

with position first and velocity second.

## Inertial frame and time

Mission propagation is Moon-centred. SPICE-enabled workflows normally use `J2000` as the inertial frame, with spacecraft position and velocity expressed relative to the Moon.

A configured UTC epoch is converted by SPICE to ephemeris time (ET). ET is TDB-based seconds past J2000. The integration variable is elapsed seconds from that epoch:

```text
ET(t) = epoch_ET + elapsed_time_s
```

Kernel coverage must span the complete requested analysis interval. See [Force models](forces.md) and [Mission workflow](mission_workflow.md).

## Lunar body-fixed frames

Non-zonal lunar gravity, terrain coordinates, ground tracks and surface sites require an explicit body-fixed frame. A lunar frame name is part of the scientific model, not presentation metadata.

Do not silently treat DE421, DE430, DE440 or other lunar frame realisations as interchangeable. A gravity product and a terrain product may legitimately require different body-fixed transformations during the same run.

For example, a GRAIL field associated with one principal-axes realisation must not be evaluated in a different principal-axes frame merely because both are Moon-fixed. The mission workflow supports separate gravity and surface frames for this reason.

## Latitude and longitude

Surface coordinates use planetocentric latitude and east-positive longitude.

Longitudes are normally wrapped to

```text
0 <= longitude < 360 degrees
```

for reported ground-track and site coordinates. Continuous ground-track products may additionally retain an unwrapped longitude history. At an exact pole longitude is geometrically undefined; code that reports ground tracks does not invent a physically meaningful meridian there.

## Reference radius, altitude and terrain clearance

These quantities are deliberately distinct.

The common lunar analysis/collision reference sphere is the 1737.4 km mean-radius sphere. Reference-radius altitude is

```text
|r_spacecraft| - R_analysis
```

A spherical-harmonic gravity product can use a different reference radius for its coefficient expansion. GRAIL products commonly use a 1738.0 km gravity reference radius. The gravity reference radius is not a physical lunar surface.

Terrain clearance is

```text
|r_spacecraft| - r_terrain(latitude, longitude)
```

in the terrain model's own body-fixed frame. A positive reference-radius altitude does not guarantee positive terrain clearance.

See [Gravity model](model.md) and [Terrain](terrain.md).

## GRAIL SHADR harmonics

The SHADR reader follows the archived geodesy convention used by the supported GRAIL products:

- real spherical harmonics;
- geodesy 4π normalisation;
- east-positive longitude;
- no Condon-Shortley phase.

For a pure J2 field under this normalisation,

```text
Cbar_20 = -J2 / sqrt(5)
```

Degree `n` describes the spherical-harmonic degree and order `m` the longitudinal order. A truncation described as `N x M` evaluates terms with

```text
n <= N
m <= min(n, M)
```

Higher degree is not automatically required for every orbit or accuracy target. Use the [Fidelity](fidelity.md) workflow to evaluate the actual trajectory/location dependence.

## LOLA terrain frames

LOLA products carry their own cartographic/frame definition and reference radius. The recommended Goddard MOON_PA products and standard PDS cylindrical products are not interchangeable merely because both represent lunar topography.

The terrain loader records the declared frame, and terrain propagation/access APIs require a matching transformation. If gravity and terrain use different frame realisations, maintain separate transformations and record both in provenance.

## Osculating and secular quantities

Classical or nonsingular elements computed from an instantaneous Cartesian state are osculating quantities. Their short-period variation should not be confused with secular drift.

The analysis layer therefore reports both histories and fitted evolution statistics. Where a secular drift is reported, it is derived from the documented least-squares trend over the analysed interval; it is not an analytical guarantee outside that interval.

## Singular classical angles

RAAN, argument of periapsis and related classical angles become undefined or numerically fragile for circular and/or equatorial orbits. The library does not manufacture zero-valued angles to make singular states appear well-defined.

Use eccentricity vectors, angular-momentum/plane vectors and modified equinoctial elements for near-singular analysis. Some classical-element functions intentionally raise when a requested angle is undefined. See [Orbital analysis](orbital_analysis.md).

## Impact and lifetime

When a terrain model is supplied, terrain-aware propagation uses the configured radial terrain surface for impact detection and refined minimum clearance. Without terrain, the mean-radius spherical event is the explicit fallback.

Impact-free lifetime is the time survived under the stated model and horizon. Surviving the requested interval is not, by itself, evidence of a frozen or stable orbit. Stability search additionally evaluates apsis, eccentricity-vector and plane evolution. See [Frozen-orbit search](frozen_orbit_search.md).

## Gravity uncertainty

SHADR coefficient sigma fields are individual coefficient uncertainties, not a complete covariance matrix.

Two uncertainty approaches are supported:

- **diagonal sigma sampling:** independent Gaussian coefficient perturbations, requiring explicit acknowledgement that correlations are being ignored;
- **covariance-derived clone perturbations:** archived gravity-field perturbation realisations applied to a compatible nominal solution, preferred when those products are available for correlated gravity uncertainty studies.

Neither approach represents complete mission uncertainty. Initial-state, navigation, manoeuvre, spacecraft-property, ephemeris, terrain and other uncertainties remain separate unless explicitly modelled. See [Uncertainty](uncertainty.md).

## Reproducibility

A defensible run should preserve at least:

- UTC/ET epoch;
- inertial and body-fixed frame names;
- loaded SPICE kernels;
- gravity product and degree/order;
- terrain product/frame/resolution;
- enabled force components and spacecraft SRP parameters;
- initial state;
- numerical integration settings;
- output cadence and analysis horizon;
- uncertainty assumptions and seeds/realisation provenance where relevant.

The configuration-driven workflow writes this information into machine-readable provenance. See [Reproducibility](reproducibility.md) for the full checklist.
