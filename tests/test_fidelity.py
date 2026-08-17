import numpy as np
import pytest

from lunar_astrodynamics.constants import GRGM1200A_J2, MOON_MEAN_RADIUS_M
from lunar_astrodynamics.dynamics import j2_acceleration
from lunar_astrodynamics.elements import ClassicalElements, state_from_elements
from lunar_astrodynamics.fidelity import (
    FidelityTolerance,
    ForceModelCase,
    HarmonicTruncation,
    build_force_model_ladder,
    compare_force_model_ladder,
    compare_harmonic_accelerations,
    compare_harmonic_trajectories,
    default_harmonic_truncations,
    select_lowest_harmonic_truncation,
)
from lunar_astrodynamics.harmonics import SphericalHarmonicModel
from lunar_astrodynamics.propagation import PropagationSettings
from lunar_astrodynamics.stability import SearchDynamics
from lunar_astrodynamics.terrain import RegularLatLonTerrain

MU = GRGM1200A_J2.mu_m3_s2
R = MOON_MEAN_RADIUS_M
J2 = GRGM1200A_J2.j2


def _identity(_time_s: float) -> np.ndarray:
    return np.eye(3)


def _j2_model() -> SphericalHarmonicModel:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -J2 / np.sqrt(5.0)
    return SphericalHarmonicModel(
        MU,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="synthetic pure J2 field",
        frame="TEST_FIXED",
    )


def _tesseral_model() -> SphericalHarmonicModel:
    c = np.zeros((3, 3))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    c[2, 0] = -J2 / np.sqrt(5.0)
    c[2, 2] = 3.0e-5
    s[2, 2] = -2.0e-5
    return SphericalHarmonicModel(
        MU,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="synthetic J2 plus C22/S22 field",
        frame="TEST_FIXED",
    )


def _central_degree4_model() -> SphericalHarmonicModel:
    c = np.zeros((5, 5))
    s = np.zeros_like(c)
    c[0, 0] = 1.0
    return SphericalHarmonicModel(
        MU,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="degree-4 central-only fixture",
        frame="TEST_FIXED",
    )


def _state() -> np.ndarray:
    return state_from_elements(
        ClassicalElements(
            semi_major_axis_m=R + 120_000.0,
            eccentricity=0.01,
            inclination_rad=np.deg2rad(70.0),
            raan_rad=np.deg2rad(20.0),
            argument_of_periapsis_rad=np.deg2rad(40.0),
            true_anomaly_rad=np.deg2rad(15.0),
        ),
        MU,
    )


def _propagation() -> PropagationSettings:
    return PropagationSettings(
        rtol=1e-10,
        position_atol_m=1e-4,
        velocity_atol_m_s=1e-7,
        max_step_s=60.0,
    )


def test_default_truncation_ladder_is_clipped_and_includes_model_maximum() -> None:
    model = _central_degree4_model()
    assert default_harmonic_truncations(model) == (
        HarmonicTruncation(2, 2),
        HarmonicTruncation(4, 0),
    )


def test_pure_j2_omission_error_matches_independent_closed_form_acceleration() -> None:
    model = _j2_model()
    position = np.array([R + 100_000.0, 180_000.0, 250_000.0])
    velocity = np.array([-100.0, 1550.0, 350.0])
    report = compare_harmonic_accelerations(
        model,
        _identity,
        position,
        velocities_m_s=velocity,
        truncations=(HarmonicTruncation(0, 0), HarmonicTruncation(2, 0)),
        reference=HarmonicTruncation(2, 0),
        benchmark_repetitions=1,
    )
    central = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(0, 0))
    expected_j2 = j2_acceleration(position, MU, model.reference_radius_m, J2)
    assert central.maximum_absolute_error_m_s2 == pytest.approx(
        np.linalg.norm(expected_j2), rel=2e-13
    )
    sample = central.samples[0]
    np.testing.assert_allclose(
        np.array(sample.acceleration_error_m_s2),
        -expected_j2,
        rtol=2e-13,
        atol=1e-15,
    )
    assert sample.along_track_error_m_s2 is not None
    assert sample.cross_track_error_m_s2 is not None
    reference = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(2, 0))
    assert reference.maximum_absolute_error_m_s2 == pytest.approx(0.0)
    assert reference.runtime_seconds_per_evaluation > 0.0


def test_order_truncation_detects_tesseral_content() -> None:
    model = _tesseral_model()
    positions = np.array(
        [
            [R + 90_000.0, 100_000.0, 50_000.0],
            [1_100_000.0, 1_400_000.0, 700_000.0],
        ]
    )
    report = compare_harmonic_accelerations(
        model,
        _identity,
        positions,
        truncations=((2, 0), (2, 2)),
        reference=(2, 2),
        benchmark_repetitions=1,
    )
    zonal = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(2, 0))
    full = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(2, 2))
    assert zonal.maximum_absolute_error_m_s2 > 0.0
    assert zonal.maximum_relative_error > 0.0
    assert full.maximum_absolute_error_m_s2 == pytest.approx(0.0)
    assert full.maximum_relative_error == pytest.approx(0.0)
    assert all(sample.along_track_error_m_s2 is None for sample in zonal.samples)


def test_acceleration_selector_returns_lowest_tested_truncation_meeting_tolerance() -> None:
    model = _j2_model()
    position = np.array([R + 100_000.0, 50_000.0, 80_000.0])
    report = compare_harmonic_accelerations(
        model,
        _identity,
        position,
        truncations=((0, 0), (2, 0)),
        reference=(2, 0),
        benchmark_repetitions=1,
    )
    central = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(0, 0))
    strict = select_lowest_harmonic_truncation(
        report,
        FidelityTolerance(
            maximum_absolute_acceleration_error_m_s2=0.5 * central.maximum_absolute_error_m_s2
        ),
    )
    assert strict.selected_truncation == HarmonicTruncation(2, 0)
    relaxed = select_lowest_harmonic_truncation(
        report,
        FidelityTolerance(
            maximum_absolute_acceleration_error_m_s2=2.0 * central.maximum_absolute_error_m_s2
        ),
    )
    assert relaxed.selected_truncation == HarmonicTruncation(0, 0)


def test_central_only_model_trajectory_converges_at_zero_degree() -> None:
    model = _central_degree4_model()
    report = compare_harmonic_trajectories(
        model,
        _identity,
        _state(),
        1800.0,
        truncations=((0, 0), (2, 2), (4, 0)),
        reference=(4, 0),
        sample_count=33,
        propagation=_propagation(),
    )
    for entry in report.entries:
        assert entry.final_position_difference_m == pytest.approx(0.0, abs=5e-7)
        assert entry.final_velocity_difference_m_s == pytest.approx(0.0, abs=5e-10)
        assert entry.periselene_variation_difference_m == pytest.approx(0.0, abs=1e-6)
        assert entry.eccentricity_variation_difference == pytest.approx(0.0, abs=1e-12)
        assert entry.impact_matches_reference
        assert entry.runtime_seconds > 0.0
    selection = select_lowest_harmonic_truncation(
        report,
        FidelityTolerance(
            maximum_final_position_difference_m=1e-3,
            maximum_final_velocity_difference_m_s=1e-6,
        ),
    )
    assert selection.selected_truncation == HarmonicTruncation(0, 0)


def test_trajectory_fidelity_reports_terrain_clearance_difference() -> None:
    model = _j2_model()
    terrain = RegularLatLonTerrain(
        latitude_deg=np.array([-90.0, 0.0, 90.0]),
        longitude_deg_east=np.array([0.0, 180.0, 360.0]),
        elevation_grid_m=np.zeros((3, 3)),
        reference_radius_m=R,
        name="flat test terrain",
        frame="TEST_FIXED",
        registration="gridline",
    )
    report = compare_harmonic_trajectories(
        model,
        _identity,
        _state(),
        900.0,
        truncations=((0, 0), (2, 0)),
        reference=(2, 0),
        sample_count=17,
        propagation=_propagation(),
        terrain=terrain,
        terrain_body_fixed_from_inertial=_identity,
        terrain_frame="TEST_FIXED",
        terrain_clearance_search_samples=33,
    )
    central = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(0, 0))
    assert central.outcome.minimum_terrain_clearance_m is not None
    assert central.minimum_terrain_clearance_difference_m is not None
    reference = next(entry for entry in report.entries if entry.truncation == HarmonicTruncation(2, 0))
    assert reference.minimum_terrain_clearance_difference_m == pytest.approx(0.0)


def test_force_model_ladder_builds_requested_levels_and_comparison() -> None:
    model = _tesseral_model()
    extra = type(
        "ConstantForce",
        (),
        {
            "name": "constant test perturbation",
            "__call__": lambda self, _time_s, _position_m: np.array([1e-8, 0.0, 0.0]),
            "provenance": lambda self: {"name": "constant test perturbation"},
        },
    )()
    srp = type(
        "ConstantSrp",
        (),
        {
            "name": "constant test SRP",
            "__call__": lambda self, _time_s, _position_m: np.array([0.0, 1e-9, 0.0]),
            "provenance": lambda self: {"name": "constant test SRP"},
        },
    )()
    ladder = build_force_model_ladder(
        model,
        _identity,
        truncated=HarmonicTruncation(2, 0),
        high_degree=HarmonicTruncation(2, 2),
        third_body_forces=(extra,),
        srp=srp,
    )
    assert [case.name for case in ladder] == [
        "central gravity",
        "J2 zonal gravity",
        "truncated GRAIL 2x0",
        "high-degree GRAIL 2x2",
        "high-degree GRAIL 2x2 + third bodies",
        "high-degree GRAIL 2x2 + third bodies + SRP",
    ]
    report = compare_force_model_ladder(
        _state(),
        600.0,
        ladder,
        sample_count=13,
        propagation=_propagation(),
    )
    assert report.reference_case == ladder[-1].name
    reference = report.entries[-1]
    assert reference.final_position_difference_m == pytest.approx(0.0)
    assert reference.final_velocity_difference_m_s == pytest.approx(0.0)
    assert all(entry.runtime_seconds > 0.0 for entry in report.entries)


def test_force_model_comparison_accepts_arbitrary_search_dynamics_cases() -> None:
    zero = SearchDynamics(
        name="zero",
        mu_m3_s2=MU,
        analysis_reference_radius_m=1.0,
        collision_radius_m=1.0,
        acceleration=lambda _time_s, _position_m: np.zeros(3),
        fidelity="synthetic",
        parallel_safe=True,
    )
    weak = SearchDynamics(
        name="weak",
        mu_m3_s2=MU,
        analysis_reference_radius_m=1.0,
        collision_radius_m=1.0,
        acceleration=lambda _time_s, _position_m: np.array([1e-6, 0.0, 0.0]),
        fidelity="synthetic",
        parallel_safe=True,
    )
    initial = np.array([10_000_000.0, 0.0, 0.0, 0.0, 100.0, 0.0])
    report = compare_force_model_ladder(
        initial,
        100.0,
        (ForceModelCase("zero", zero), ForceModelCase("weak", weak)),
        reference_case="weak",
        sample_count=5,
        propagation=PropagationSettings(rtol=1e-12, max_step_s=10.0),
    )
    assert report.entries[0].final_position_difference_m is not None
    assert report.entries[0].final_position_difference_m > 0.0
