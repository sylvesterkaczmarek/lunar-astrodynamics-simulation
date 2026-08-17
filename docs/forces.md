# Composable lunar perturbation force model

This module extends the lunar gravity propagator with independently testable force components for Earth and Sun third-body gravity and direct solar radiation pressure. The design is deliberately compositional: lunar gravity, third bodies, SRP, future perturbations and user-defined accelerations remain separate callables rather than branches inside one monolithic equations-of-motion function.

## Force composition

`CompositeForceModel` evaluates each named component at the same integration time and Moon-centred inertial spacecraft position and sums the returned accelerations:

```text
a_total(t, r) = sum_k a_k(t, r)
```

`component_accelerations(...)` returns the individual vectors for diagnostics. `provenance()` records the metadata exposed by each component.

Existing acceleration functions remain usable directly. `CallableForce` provides a named wrapper, so an existing central/J2 or high-degree GRAIL acceleration can be inserted without changing its implementation.

For example, a high-degree gravity science run can use a component whose callable is `gravity_acceleration_inertial(...)`, while Earth, Sun and SRP remain independent components.

## Moon-centred third-body gravity

Let

- `r` be the spacecraft position relative to the Moon;
- `r_b` be the disturbing body's position relative to the Moon;
- `mu_b` be the disturbing body's gravitational parameter.

The correct relative acceleration in a Moon-centred frame is

```text
a_b = mu_b [ (r_b - r) / |r_b - r|^3 - r_b / |r_b|^3 ]
```

The second term is essential. It subtracts the acceleration of the Moon caused by the same third body. Adding only the direct attraction of Earth or Sun on the spacecraft would be incorrect for Moon-centred relative dynamics.

`third_body_acceleration(...)` implements this equation directly. `ThirdBodyGravity` couples it to a time-dependent body-position provider.

The default Earth and Sun mass parameters are JPL DE440 values:

```text
GM Earth = 3.98600435507e14 m^3/s^2
GM Sun   = 1.32712440041279419e20 m^3/s^2
```

They are constants for convenience rather than hidden SPICE lookups. A user reproducing another ephemeris solution can supply different values when constructing the components.

JPL reference: https://ssd.jpl.nasa.gov/astro_par.html

## SPICE ephemeris context

`SpiceEphemeris` defines the time and frame convention used by ephemeris-driven forces.

The integration variable `time_s` is elapsed SI seconds from a fixed SPICE ephemeris-time epoch:

```text
ET(t) = epoch_et_s + time_s
```

`epoch_et_s` is seconds past J2000 TDB, following SPICE convention. UTC construction is available through `spice_ephemeris_from_utc(...)`; a leap-seconds kernel must already be loaded so SPICE can convert the UTC string to ET.

Positions are obtained with `spiceypy.spkpos` using:

```text
target   = EARTH or SUN
observer = MOON
frame    = caller-selected inertial frame, normally J2000
abcorr   = NONE
```

`NONE` is enforced for force-model contexts because Newtonian gravity and the simple SRP model require geometric same-epoch positions, not apparent light-time or stellar-aberration corrected observations.

NAIF references:

- https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkpos_c.html
- https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html

### Kernel provenance

When a `SpiceEphemeris` is created with the normal constructors, it snapshots the currently loaded SPICE kernel pool using `ktotal`/`kdata`. Its provenance contains:

- UTC epoch when supplied;
- numerical ET epoch;
- inertial frame;
- Moon observer name;
- aberration correction (`NONE`);
- loaded kernel paths, types and source meta-kernels.

This metadata is also propagated into Earth, Sun and SRP component provenance when their position providers come from the same context.

## Minimal generic kernel workflow

The repository does not bundle planetary SPK files. The example uses the generic NAIF leap-seconds and DE440 short planetary ephemeris kernels:

```bash
python -m pip install -e .[spice]
python scripts/download_force_model_kernels.py
```

This downloads:

```text
data/spice/naif0012.tls
data/spice/de440s.bsp
```

The example explicitly loads them with `spiceypy.furnsh` before creating the ephemeris context.

NAIF generic kernels are available at https://naif.jpl.nasa.gov/pub/naif/generic_kernels/ .

For a controlled project, pin and archive the exact kernel set used by the analysis rather than relying on mutable generic-kernel directory listings.

## Solar radiation pressure

`SolarRadiationPressure` uses a spherical/cannonball effective-area model. Let

- `P_0` be the selected mean solar radiation pressure at 1 AU;
- `C_R` be the effective reflectivity coefficient;
- `A` be effective illuminated area;
- `m` be spacecraft mass;
- `d` be Sun-spacecraft distance;
- `AU` be the astronomical unit;
- `nu` be the visible solar fraction after lunar occultation;
- `u_sun_to_sc` be the unit vector from the Sun toward the spacecraft.

The acceleration is

```text
a_SRP = nu P_0 C_R (A/m) (AU/d)^2 u_sun_to_sc
```

The default `P_0 = 4.56e-6 N/m^2` is a representative mean direct solar momentum flux at 1 AU and is configurable. NASA references commonly use approximately 4.56 to 4.57 micro-newtons per square metre at 1 AU.

NASA reference: https://ntrs.nasa.gov/citations/20130014732

SRP is optional. It should be omitted when spacecraft mass, effective illuminated area or optical coefficient are unknown rather than populated with arbitrary values.

The cannonball model does not represent attitude-dependent projected area, separate surfaces, self-shadowing, specular/diffuse optical properties, thermal reradiation or spacecraft articulation.

## Lunar eclipse and penumbra

The SRP model uses finite apparent disks for the Sun and spherical Moon. From the spacecraft, the apparent angular radii are

```text
alpha_sun  = asin(R_sun  / d_sun)
alpha_moon = asin(R_moon / d_moon)
```

and the angular separation is obtained from the angle between the spacecraft-to-Sun and spacecraft-to-Moon vectors.

The visible fraction is determined from the overlap area of the two apparent circular disks:

- no overlap: `nu = 1`;
- Moon fully covers the solar disk: `nu = 0`;
- annular eclipse: subtract the Moon disk area from the solar disk area;
- partial eclipse: subtract the standard two-circle intersection area.

This gives continuous penumbra transitions and a true full-umbra state instead of switching SRP at a cylindrical shadow boundary.

NASA describes the same apparent-radius and disk-overlap eclipse geometry in its trajectory and navigation literature, including partial-eclipse illumination from the ratio of solar/body overlap area to solar disk area.

References:

- https://ntrs.nasa.gov/citations/20170003689
- https://ntrs.nasa.gov/api/citations/20250006484/downloads/GEONSMSR3_1.pdf

### Eclipse limitations

The occulting Moon is a sphere with a configurable radius, defaulting to the 1737.4 km mean radius. LOLA terrain, lunar limb topography and non-spherical limb geometry are not included in SRP occultation. Solar limb darkening is not modeled. Those effects are much more specialized than the present preliminary orbit force model.

## Example

After downloading the kernels:

```bash
python examples/force_model_comparison.py \
  --kernel-dir data/spice \
  --epoch-utc 2026-08-17T00:00:00 \
  --duration-days 7 \
  --samples 1001
```

The example propagates the same low polar lunar orbit through four models:

1. lunar central + J2 gravity;
2. lunar gravity + Earth third body;
3. lunar gravity + Earth + Sun third bodies;
4. lunar gravity + Earth + Sun + SRP with lunar shadowing.

It reports minimum/maximum reference-sphere altitude, osculating eccentricity range, impact state if any, final/max/RMS position separation from the gravity-only solution, velocity separation, eclipse fractions and complete force/SPICE provenance.

The example intentionally uses central + J2 lunar gravity so the incremental Earth/Sun/SRP effects are easy to reproduce without downloading GRGM1200A or lunar orientation kernels. For higher-fidelity lunar studies, wrap the existing body-fixed GRAIL acceleration in a `CallableForce` and combine it with the same perturbation components.

## Excluded perturbations

Adding Earth, Sun and SRP does not make the propagator a mission-grade truth model. Important effects that may matter depending on mission duration, orbit and accuracy target include:

- high-degree lunar gravity if a run uses only central/J2 gravity;
- gravity-field uncertainty and clone-field dispersion;
- lunar body tides and time-variable gravity;
- Earth oblateness acting as an extended third body;
- planetary third bodies beyond Earth and Sun;
- relativistic corrections;
- Earth radiation pressure, lunar albedo and thermal radiation;
- detailed spacecraft attitude/area and thermal reradiation;
- maneuvers, thrust, mass depletion and navigation-estimation errors;
- finite-spacecraft terrain contact geometry.

Force fidelity should be selected and validated against the required prediction horizon and error budget rather than inferred from the number of enabled perturbations.
