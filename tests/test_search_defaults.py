from lunar_astrodynamics import GRGM1200A_J2, default_low_lunar_search_space


def test_default_low_lunar_search_space_has_positive_nominal_periselene() -> None:
    space = default_low_lunar_search_space()
    points = space.points(GRGM1200A_J2.collision_radius_m)
    assert points
    for point in points:
        nominal_periselene_radius_m = point.semi_major_axis_m * (1.0 - point.eccentricity)
        assert nominal_periselene_radius_m > GRGM1200A_J2.collision_radius_m
