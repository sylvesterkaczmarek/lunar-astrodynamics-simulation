"""Small, validated lunar J2 astrodynamics demonstration package."""

from .analysis import ElementHistory, element_history, linear_rate
from .constants import GRGM1200A_J2, MOON_GM_DE440_M3_S2, MOON_MEAN_RADIUS_M, LunarJ2Model
from .dynamics import central_acceleration, equations_of_motion, j2_acceleration, total_acceleration
from .elements import (
    ClassicalElements,
    analytical_j2_secular_rates,
    elements_from_state,
    mean_motion_rad_s,
    orbital_period_s,
    state_from_elements,
)
from .propagation import PropagationSettings, make_surface_event, propagate

__all__ = [
    "ClassicalElements",
    "ElementHistory",
    "GRGM1200A_J2",
    "LunarJ2Model",
    "MOON_GM_DE440_M3_S2",
    "MOON_MEAN_RADIUS_M",
    "PropagationSettings",
    "analytical_j2_secular_rates",
    "central_acceleration",
    "element_history",
    "elements_from_state",
    "equations_of_motion",
    "j2_acceleration",
    "linear_rate",
    "make_surface_event",
    "mean_motion_rad_s",
    "orbital_period_s",
    "propagate",
    "state_from_elements",
    "total_acceleration",
]
