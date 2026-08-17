"""Mission-agnostic default parameter ranges for lunar stability screening."""

from __future__ import annotations

import numpy as np

from .stability import StabilitySearchSpace


def default_low_lunar_search_space() -> StabilitySearchSpace:
    """Return a broad, surface-safe low and near-polar lunar screening grid.

    The grid is deliberately generic rather than tuned to one mission.  Its
    largest default eccentricity is limited so the lowest 60 km semimajor-axis
    altitude remains above the 1737.4 km mean-radius sphere at nominal
    periselene. Terrain clearance can be more restrictive and must be evaluated
    separately when a LOLA shape model is supplied.
    """
    return StabilitySearchSpace(
        semi_major_axis_altitudes_m=(60_000.0, 90_000.0, 120_000.0, 150_000.0),
        semi_major_axes_m=None,
        eccentricities=(0.005, 0.015, 0.025),
        inclinations_rad=tuple(np.deg2rad([80.0, 85.0, 90.0, 95.0, 100.0])),
        raan_rad=tuple(np.deg2rad([0.0, 90.0, 180.0, 270.0])),
        periapsis_rad=tuple(np.deg2rad([0.0, 90.0, 180.0, 270.0])),
        initial_anomaly_rad=(0.0,),
        periapsis_parameterization="longitude_of_periapsis",
    )
