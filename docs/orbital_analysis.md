# Nonsingular orbital mathematics and stability analysis

Lunar frozen-orbit and long-duration stability work should not depend only on classical Keplerian angles. Classical elements remain useful for ordinary non-circular, non-equatorial trajectories, but RAAN is undefined for an equatorial orbit and argument of periapsis is undefined for a circular orbit. Near those limits, the angles can also become numerically sensitive even when a software routine can still produce a number.

The library therefore keeps classical elements and adds two coordinate-safe layers:

1. physical orbital vectors derived directly from Cartesian state;
2. prograde modified equinoctial elements (MEE) for nonsingular scalar parameterization.

## Physical vector quantities

For Moon-centred inertial position `r`, velocity `v`, and lunar gravitational parameter `mu`, the library evaluates

```text
h_vec = r x v
h     = |h_vec|
e_vec = (v x h_vec)/mu - r/|r|
e     = |e_vec|
n_hat = h_vec/h
energy = |v|^2/2 - mu/|r|
p      = h^2/mu
a      = -mu/(2 energy)
```

`orbital_vectors_from_state(...)` exposes:

- eccentricity vector;
- specific angular-momentum vector;
- unit orbital-plane normal;
- semimajor axis;
- semilatus rectum;
- eccentricity magnitude;
- inclination relative to the input Cartesian reference plane;
- specific orbital energy.

These quantities do not require RAAN or argument of periapsis. The eccentricity vector is especially useful in nearly circular frozen-orbit studies because its small Cartesian components retain the direction and magnitude of the osculating eccentricity without dividing the geometry into a poorly conditioned periapsis angle.

For a bound ellipse,

```text
r_periselene = p/(1 + e)
r_aposelene  = p/(1 - e)
```

and reference-radius altitudes are obtained by subtracting the selected lunar reference radius.

## Modified equinoctial elements

The selected nonsingular scalar representation is the Walker modified equinoctial set

```text
(p, f, g, h, k, L)
```

with the prograde tangent convention

```text
p = a (1 - e^2)
f = e cos(Omega + omega)
g = e sin(Omega + omega)
h = tan(i/2) cos(Omega)
k = tan(i/2) sin(Omega)
L = Omega + omega + nu
```

where `Omega` is RAAN, `omega` is argument of periapsis, and `nu` is true anomaly when those classical angles are individually defined.

The implementation does **not** compute MEE by first recovering the classical angles. Cartesian-to-MEE conversion is formed directly from the angular-momentum direction, eccentricity vector, and an equinoctial in-plane basis, so circular and prograde-equatorial states remain valid.

Primary references:

- M. J. H. Walker, B. Ireland, and J. Owens, *A Set of Modified Equinoctial Orbit Elements*, Celestial Mechanics 36 (1985), 409-419, DOI `10.1007/BF01227493`.
- M. J. H. Walker, erratum, Celestial Mechanics 38 (1986), 391-392, DOI `10.1007/BF01238929`.
- R. A. Broucke and P. J. Cefola, *On the Equinoctial Orbit Elements*, Celestial Mechanics 5 (1972); NASA NTRS record `19720051438`.
- NAIF `eqncpv_c` documentation for the tangent half-angle equinoctial convention and its distinction from the sine formulation.

NASA's earlier equinoctial work explicitly motivated these variables by removing the zero-eccentricity and zero-inclination singularities, and notes that a retrograde formulation is required near 180 degrees inclination.

## Singularities and conditioning

The implemented MEE convention is the standard **prograde** tangent formulation.

It is regular at:

- `e = 0` circular orbits;
- `i = 0` prograde-equatorial orbits;
- `i = 90 deg` polar orbits;
- ordinary eccentric and inclined lunar orbits.

It is singular at exactly

```text
i = 180 deg
```

because `tan(i/2)` diverges. The `h` and `k` components also become increasingly large and numerically ill-conditioned as the orbit approaches that retrograde-equatorial limit. `modified_equinoctial_from_state(...)` detects the singular limit and raises instead of returning misleading enormous values.

A future retrograde-MEE formulation could cover that complementary region. It is not silently substituted because the convention itself is part of the element definition and must remain explicit.

## Direct state reconstruction

For

```text
s2 = 1 + h^2 + k^2
```

define the fixed equinoctial basis vectors

```text
F = [1 + h^2 - k^2,  2 h k,          -2 k] / s2
G = [2 h k,            1 - h^2 + k^2, 2 h] / s2
```

and

```text
w = 1 + f cos(L) + g sin(L)
r = p/w
```

Then

```text
r_vec = r [cos(L) F + sin(L) G]
v_vec = sqrt(mu/p) [-(g + sin(L)) F + (f + cos(L)) G]
```

`state_from_modified_equinoctial(...)` implements this direct reconstruction. It does not need classical RAAN, argument of periapsis, or true anomaly as separate quantities.

## Classical-angle policy

Classical elements remain supported through `ClassicalElements`, `state_from_elements(...)`, and `elements_from_state(...)`.

The library deliberately does not assign arbitrary zero angles in singular cases:

- equatorial state -> `elements_from_state(...)` raises that RAAN is undefined;
- circular state -> it raises that argument of periapsis is undefined;
- `classical_from_modified_equinoctial(...)` follows the same policy;
- `ModifiedEquinoctialElements.raan_rad` and `.longitude_of_periapsis_rad` return `None` when the corresponding angle is not physically defined.

The legacy `element_history(...)` remains available for trajectories on which all classical angles exist. If any sample is singular, it raises and directs the caller to `orbit_history(...)`.

## Nonsingular trajectory analysis

`orbit_history(...)` is the analysis path intended for downstream frozen-orbit searches. For every supplied Cartesian sample it returns:

- semimajor axis and semilatus rectum;
- eccentricity magnitude and eccentricity vector;
- specific angular-momentum vector;
- orbital-plane normal;
- inclination;
- osculating periselene and aposelene radius;
- periselene and aposelene altitude above the selected reference radius;
- instantaneous reference-radius altitude;
- MEE `p, f, g, h, k, L` history;
- orbital-plane angular change from the initial plane;
- apsidal direction and angular change where eccentricity is large enough for a direction to be meaningful;
- optional terrain clearance.

A configurable `apsis_eccentricity_threshold` controls when the **direction** of the eccentricity vector is considered meaningful. The eccentricity vector itself is always retained. Below the threshold, apsidal direction fields are explicitly marked undefined rather than filled with an arbitrary angle.

The current analysis product assumes a bound elliptic trajectory because it reports a finite aposelene. Hyperbolic/parabolic state conversion remains possible at the vector/MEE layer, but it is outside the lunar frozen-orbit history use case.

## Drift and bounded oscillation

Long propagations frequently contain both secular drift and periodic oscillation. Reporting only `max - min` can confuse those effects.

For each scalar stability quantity, `orbit_history(...).statistics` reports:

- initial/final value;
- minimum/maximum/mean;
- total peak-to-peak range;
- least-squares linear rate;
- fitted linear drift over the full time span;
- RMS of the detrended residual;
- peak-to-peak range of the detrended residual.

The same component-wise decomposition is available for the eccentricity vector and orbital-plane normal. MEE `f/g` provide a nonsingular apsidal stability target; MEE `h/k` provide a nonsingular plane-orientation target for all inclinations except the retrograde-equatorial singular limit.

This separation is intended to support later frozen-orbit objective functions such as minimizing secular periselene drift, eccentricity-vector drift, or detrended oscillation amplitude.

## Reference altitude versus terrain clearance

`orbit_history(...)` always reports radial altitude above its explicit `reference_radius_m`.

If a terrain model, body-fixed rotation, and matching terrain frame are supplied, it additionally reports `terrain_clearance_m`. The two are not interchangeable:

```text
reference altitude = |r| - reference_radius
terrain clearance  = |r| - local terrain surface radius
```

The existing terrain frame checks remain in force. A LOLA terrain grid is evaluated only after transforming the spacecraft position into that grid's declared lunar body-fixed frame.

The terrain clearance values in `orbit_history(...)` are evaluated at the supplied trajectory sample times. For refined minimum-clearance and impact root finding, continue to use `propagate_with_terrain(...)` / `analyze_terrain_clearance(...)`.

## Example

Run

```bash
python examples/nonsingular_analysis.py --orbits 10
```

The example includes an exact circular/equatorial state for which classical angles are intentionally rejected while MEE remains valid, followed by a nearly circular polar J2 propagation. It reports eccentricity-vector, MEE, apsidal, orbital-plane, apsis-altitude, secular-drift, and bounded-oscillation metrics.
