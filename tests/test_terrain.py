from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from lunar_astrodynamics import (
    LOLA_MOON_PA_DE421_FRAME,
    LOLA_REFERENCE_RADIUS_M,
    PropagationSettings,
    RegularLatLonTerrain,
    load_lola_moon_pa_grd,
    load_lola_pds_global_gdr,
    load_terrain_npz,
    make_mean_radius_surface_event,
    make_surface_event,
    propagate_with_terrain,
    save_terrain_npz,
    terrain_clearance_m,
)


def _gridline_terrain(*, elevation_m: float = 0.0, frame: str = "TEST_FRAME") -> RegularLatLonTerrain:
    lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    elevation = np.full((lat.size, lon.size), elevation_m, dtype=float)
    return RegularLatLonTerrain(
        lat,
        lon,
        elevation,
        reference_radius_m=LOLA_REFERENCE_RADIUS_M,
        name="synthetic gridline terrain",
        frame=frame,
        registration="gridline",
    )


def test_gridline_bilinear_interpolation_matches_known_surface() -> None:
    lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    elevation = np.zeros((lat.size, lon.size))
    for i, latitude in enumerate(lat):
        for j, longitude in enumerate(lon):
            elevation[i, j] = 2.0 * latitude + 0.5 * longitude
    elevation[:, -1] = elevation[:, 0]
    elevation[0, :] = -100.0
    elevation[-1, :] = 100.0
    terrain = RegularLatLonTerrain(lat, lon, elevation, frame="TEST_FRAME")

    expected = 2.0 * 22.5 + 0.5 * 135.0
    actual = terrain.elevation_m(np.deg2rad(22.5), np.deg2rad(135.0))
    assert actual == pytest.approx(expected)


def test_gridline_requires_matching_periodic_boundary_columns() -> None:
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([0.0, 180.0, 360.0])
    elevation = np.zeros((3, 3))
    elevation[1, -1] = 1.0
    with pytest.raises(ValueError, match="boundary columns must match"):
        RegularLatLonTerrain(lat, lon, elevation, frame="TEST_FRAME")


def test_longitude_wrap_and_antimeridian_are_continuous() -> None:
    terrain = _gridline_terrain(elevation_m=250.0)
    latitude = np.deg2rad(10.0)
    assert terrain.elevation_m(latitude, np.deg2rad(-10.0)) == pytest.approx(
        terrain.elevation_m(latitude, np.deg2rad(350.0))
    )
    assert terrain.elevation_m(latitude, np.deg2rad(180.0)) == pytest.approx(
        terrain.elevation_m(latitude, np.deg2rad(-180.0))
    )
    assert terrain.elevation_m(latitude, 0.0) == pytest.approx(
        terrain.elevation_m(latitude, 2.0 * np.pi)
    )


def test_exact_pole_value_is_independent_of_longitude_without_double_counting_seam() -> None:
    lat = np.array([-90.0, 0.0, 90.0])
    lon = np.array([0.0, 120.0, 240.0, 360.0])
    elevation = np.array(
        [
            [-2.0, -2.0, -2.0, -2.0],
            [10.0, 20.0, 30.0, 10.0],
            [101.0, 99.0, 100.0, 101.0],
        ]
    )
    terrain = RegularLatLonTerrain(lat, lon, elevation, frame="TEST_FRAME")
    values = [terrain.elevation_m(np.pi / 2.0, np.deg2rad(lon_deg)) for lon_deg in (0, 45, 180, 359)]
    assert values == pytest.approx([100.0] * 4)


def test_pixel_registered_polar_cap_extrapolates_to_unique_pole() -> None:
    lat = np.array([-67.5, -22.5, 22.5, 67.5])
    lon = np.array([45.0, 135.0, 225.0, 315.0])
    elevation = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [100.0, 200.0, 300.0, 400.0],
        ]
    )
    terrain = RegularLatLonTerrain(lat, lon, elevation, frame="TEST_FRAME", registration="pixel")
    assert terrain.elevation_m(np.pi / 2.0, 0.0) == pytest.approx(250.0)
    value = terrain.elevation_m(np.deg2rad(78.75), np.deg2rad(45.0))
    assert value == pytest.approx(175.0)


def test_terrain_clearance_requires_explicit_matching_frame() -> None:
    terrain = _gridline_terrain(elevation_m=500.0, frame="MOON_PA_DE421")
    position = np.array([LOLA_REFERENCE_RADIUS_M + 1500.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="terrain frame mismatch"):
        terrain_clearance_m(0.0, position, terrain, lambda _t: np.eye(3), terrain_frame="MOON_PA_DE430")
    assert terrain_clearance_m(
        0.0,
        position,
        terrain,
        lambda _t: np.eye(3),
        terrain_frame="MOON_PA_DE421",
    ) == pytest.approx(1000.0)


def test_terrain_impact_event_finds_linear_descent_and_reports_geometry() -> None:
    terrain = _gridline_terrain(elevation_m=1000.0, frame="TEST_FRAME")
    surface_radius = LOLA_REFERENCE_RADIUS_M + 1000.0
    initial = np.array([surface_radius + 1000.0, 0.0, 0.0, -100.0, 0.0, 0.0])
    result = propagate_with_terrain(
        initial,
        20.0,
        lambda _time, _position: np.zeros(3),
        terrain,
        lambda _time: np.eye(3),
        terrain_frame="TEST_FRAME",
        settings=PropagationSettings(
            rtol=1e-11,
            position_atol_m=1e-6,
            velocity_atol_m_s=1e-9,
            max_step_s=1.0,
        ),
        clearance_search_samples=129,
    )
    report = result.clearance
    assert report.impacted
    assert report.impact_time_s == pytest.approx(10.0, abs=1e-8)
    assert report.minimum_time_s == pytest.approx(10.0, abs=1e-8)
    assert report.minimum_clearance_m == pytest.approx(0.0, abs=1e-6)
    assert report.impact_location is not None
    assert report.impact_location.latitude_deg == pytest.approx(0.0, abs=1e-10)
    assert report.impact_location.longitude_deg_east == pytest.approx(0.0, abs=1e-10)
    assert report.impact_location.terrain_elevation_m == pytest.approx(1000.0)


def test_clearance_search_uses_full_dense_interval_when_t_eval_omits_endpoints() -> None:
    terrain = _gridline_terrain(elevation_m=0.0, frame="TEST_FRAME")
    initial = np.array([LOLA_REFERENCE_RADIUS_M + 1000.0, 0.0, 0.0, -10.0, 0.0, 0.0])
    result = propagate_with_terrain(
        initial,
        50.0,
        lambda _time, _position: np.zeros(3),
        terrain,
        lambda _time: np.eye(3),
        terrain_frame="TEST_FRAME",
        sample_times_s=np.array([10.0, 20.0]),
        settings=PropagationSettings(
            rtol=1e-11,
            position_atol_m=1e-6,
            velocity_atol_m_s=1e-9,
            max_step_s=5.0,
        ),
        clearance_search_samples=101,
    )
    assert not result.clearance.impacted
    assert result.clearance.minimum_time_s == pytest.approx(50.0, abs=1e-8)
    assert result.clearance.minimum_clearance_m == pytest.approx(500.0, abs=1e-6)


def test_terrain_propagation_rejects_initial_state_below_local_surface() -> None:
    terrain = _gridline_terrain(elevation_m=5000.0, frame="TEST_FRAME")
    initial = np.array([LOLA_REFERENCE_RADIUS_M + 1000.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="at or below the terrain surface"):
        propagate_with_terrain(
            initial,
            10.0,
            lambda _time, _position: np.zeros(3),
            terrain,
            lambda _time: np.eye(3),
            terrain_frame="TEST_FRAME",
        )


def test_mean_radius_event_has_explicit_name_and_legacy_alias() -> None:
    radius = LOLA_REFERENCE_RADIUS_M
    state = np.array([radius + 12.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    named = make_mean_radius_surface_event(radius)
    legacy = make_surface_event(radius)
    assert named(0.0, state) == pytest.approx(12.0)
    assert legacy(0.0, state) == pytest.approx(12.0)
    assert named.terminal and legacy.terminal


def test_prepared_npz_round_trip_preserves_metadata(tmp_path: Path) -> None:
    terrain = _gridline_terrain(elevation_m=123.5, frame=LOLA_MOON_PA_DE421_FRAME)
    path = tmp_path / "terrain.npz"
    save_terrain_npz(terrain, path)
    loaded = load_terrain_npz(path)
    assert loaded.frame == terrain.frame
    assert loaded.reference_radius_m == pytest.approx(terrain.reference_radius_m)
    assert loaded.registration == "gridline"
    assert loaded.elevation_grid_m == pytest.approx(terrain.elevation_grid_m)


def test_gmt_netcdf_loader_reads_64ppd_pixel_coordinates_and_elevation(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.grd"
    spacing = 1.0 / 64.0
    lon = np.array([0.5, 1.5, 2.5, 3.5, 4.5]) * spacing
    lat = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * spacing
    z = np.add.outer(np.arange(lat.size), np.arange(lon.size)).astype(np.float32)
    with netcdf_file(path, "w") as dataset:
        dataset.createDimension("y", lat.size)
        dataset.createDimension("x", lon.size)
        xvar = dataset.createVariable("x", "f8", ("x",))
        yvar = dataset.createVariable("y", "f8", ("y",))
        zvar = dataset.createVariable("z", "f4", ("y", "x"))
        xvar[:] = lon
        yvar[:] = lat
        zvar[:] = z
    terrain = load_lola_moon_pa_grd(path, registration="pixel")
    assert terrain.frame == LOLA_MOON_PA_DE421_FRAME
    assert terrain.reference_radius_m == pytest.approx(LOLA_REFERENCE_RADIUS_M)
    assert terrain.elevation_m(0.0, np.deg2rad(lon[2])) == pytest.approx(z[2, 2])


def _synthetic_pds_label(*, resolution: int = 1, frame: str = "MEAN EARTH/POLAR AXIS OF DE421") -> str:
    return f'''PDS_VERSION_ID = PDS3
LINES = {180 * resolution}
LINE_SAMPLES = {360 * resolution}
SAMPLE_TYPE = MSB_INTEGER
SAMPLE_BITS = 16
SCALING_FACTOR = 0.5
OFFSET = 1737400 <m>
MAP_RESOLUTION = {resolution} <pix/deg>
POSITIVE_LONGITUDE_DIRECTION = EAST
COORDINATE_SYSTEM_NAME = "{frame}"
'''


def test_pds_global_gdr_loader_uses_label_scaling_and_pixel_centers(tmp_path: Path) -> None:
    label_path = tmp_path / "LDEM_TEST.LBL"
    image_path = tmp_path / "LDEM_TEST.IMG"
    label_path.write_text(_synthetic_pds_label(), encoding="ascii")
    raw = np.zeros((180, 360), dtype=">i2")
    raw[0, 0] = 2468
    raw[-1, -1] = -200
    raw.tofile(image_path)
    terrain = load_lola_pds_global_gdr(image_path, label_path)
    assert terrain.registration == "pixel"
    assert terrain.frame == "MEAN EARTH/POLAR AXIS OF DE421"
    assert terrain.latitude_deg[-1] == pytest.approx(89.5)
    assert terrain.longitude_deg_east[0] == pytest.approx(0.5)
    assert terrain.elevation_grid_m[-1, 0] == pytest.approx(1234.0)
    assert terrain.elevation_grid_m[0, -1] == pytest.approx(-100.0)
