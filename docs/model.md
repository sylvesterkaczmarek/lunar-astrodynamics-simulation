# Model definition

This repository implements a deliberately limited lunar gravity model for quantitative demonstration and regression testing.

## Implemented dynamics

The propagated state is Moon-centered Cartesian position and velocity. The acceleration is

\[
\mathbf a = -\frac{\mu}{r^3}\mathbf r + \mathbf a_{J_2}.
\]

The J2 contribution is the standard axisymmetric degree-2 zonal term

\[
a_x = \frac{3 J_2 \mu R^2 x}{2r^5}\left(5\frac{z^2}{r^2}-1\right),
\]

\[
a_y = \frac{3 J_2 \mu R^2 y}{2r^5}\left(5\frac{z^2}{r^2}-1\right),
\]

\[
a_z = \frac{3 J_2 \mu R^2 z}{2r^5}\left(5\frac{z^2}{r^2}-3\right).
\]

The demonstration uses the GRGM1200A gravity-model reference radius of 1738.0 km and its archived GM of 4902.80011526323 km^3/s^2. J2 is represented by the rounded GRGM1200A low-degree value 203.224e-6. The simple impact boundary uses the JPL mean lunar radius of 1737.4 km.

## Frame scope

For the J2-only model, the z-axis is treated as the symmetry axis. Rotation about that axis does not alter an axisymmetric zonal term.

This is not sufficient for a full GRAIL gravity field. GRGM1200A coefficients are fully normalized and defined in the lunar body-fixed principal-axes frame. Tesseral and sectoral terms therefore require an epoch-aware inertial-to-body-fixed attitude transformation, gravity evaluation in the body-fixed frame, and rotation of the resulting acceleration back to the inertial frame.

## Sources

- NASA PDS GRAIL GRGM1200A label: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.lbl
- NASA JPL planetary satellite physical parameters: https://ssd.jpl.nasa.gov/sats/phys_par/
- NASA GSFC GRGM1200A archive page: https://pgda.gsfc.nasa.gov/products/50
- Lemoine et al. (2014), GRGM900C: https://doi.org/10.1002/2014GL060027

## Deliberate exclusions

The current package does not implement:

- higher-degree/order GRAIL spherical harmonics or explicit mascon point-mass models
- lunar libration or a time-dependent body-fixed frame
- Earth or Sun third-body gravity
- solar radiation pressure
- relativistic corrections
- lunar topography for terrain-aware impact detection

Those effects matter for mission-grade long-duration low lunar orbit propagation. The package is intended to make the low-degree J2 demonstration correct, testable, and explicit about its limits.
