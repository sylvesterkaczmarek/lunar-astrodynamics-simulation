import json

import numpy as np
import pytest

from lunar_astrodynamics import (
    GRGM1200A_J2,
    MOON_MEAN_RADIUS_M,
    PropagationSettings,
    RefinementSettings,
    RegularLatLonTerrain,
    SphericalHarmonicModel,
    StabilityConstraints,
    StabilitySearchSettings,
    StabilitySearchSpace,
    harmonic_search_dynamics,
    j2_search_dynamics,
    make_stability_map,
    orbital_period_s,
    run_coarse_to_fine_search,
    run_stability_search,
)

MU = GRGM1200A_J2.mu_m3_s2


def _settings(*, workers: int = 1, constraints: StabilityConstraints | None = None) -> StabilitySearchSettings:
    period = orbital_period_s(MOON_MEAN_RADIUS_M + 100_000.0, MU)
    return StabilitySearchSettings(
        duration_s=0.2 * period,
        sample_count=33,
        propagation=PropagationSettings(rtol=1e-10, position_atol_m=1e-4, velocity_atol_m_s=1e-7, max_step_s=60.0),
        workers=workers,
        terrain_clearance_search_samples=65,
        constraints=constraints or StabilityConstraints(),
    )


def _space(
    *,
    altitudes=(100_000.0,),
    eccentricities=(0.01,),
    inclinations_deg=(90.0,),
    raan_deg=(0.0,),
    periapsis_deg=(0.0,),
    anomaly_deg=(0.0,),
) -> StabilitySearchSpace:
    return StabilitySearchSpace(
        semi_major_axis_altitudes_m=tuple(float(value) for value in altitudes),
        semi_major_axes_m=None,
        eccentricities=tuple(float(value) for value in eccentricities),
        inclinations_rad=tuple(np.deg2rad(inclinations_deg)),
        raan_rad=tuple(np.deg2rad(raan_deg)),
        periapsis_rad=tuple(np.deg2rad(periapsis_deg)),
        initial_anomaly_rad=tuple(np.deg2rad(anomaly_deg)),
        periapsis_parameterization="longitude_of_periapsis",
    )


def test_two_body_search_reports_bounded_metrics_not_only_survival() -> None:
    dynamics = j2_search_dynamics(include_j2=False)
    result = run_stability_search(
        _space(altitudes=(90_000.0, 110_000.0), eccentricities=(0.0, 0.01)),
        dynamics,
        settings=_settings(),
    )
    assert result.raw_grid_size == 4
    assert result.unique_candidate_count == 4
    assert len(result.candidates) == 4
    assert all(candidate.metrics.survived_duration_fraction == pytest.approx(1.0) for candidate in result.candidates)
    assert all(candidate.metrics.periselene_altitude_peak_to_peak_m < 2e-4 for candidate in result.candidates)
    assert all(candidate.metrics.aposelene_altitude_peak_to_peak_m < 2e-4 for candidate in result.candidates)
    assert all(candidate.metrics.eccentricity_vector_linear_drift_norm < 1e-10 for candidate in result.candidates)
    assert "periselene_spread" in result.candidates[0].ranking.normalised_terms
    assert "eccentricity_vector_drift" in result.candidates[0].ranking.normalised_terms


def test_circular_equatorial_redundant_classical_angles_are_deduplicated() -> None:
    dynamics = j2_search_dynamics(include_j2=False)
    space = _space(
        eccentricities=(0.0,),
        inclinations_deg=(0.0,),
        raan_deg=(0.0, 90.0),
        periapsis_deg=(0.0, 90.0),
    )
    result = run_stability_search(space, dynamics, settings=_settings())
    assert result.raw_grid_size == 4
    assert result.unique_candidate_count == 2
    assert len(result.candidates) == 2


def test_survival_is_a_constraint_not_definition_of_frozen_orbit() -> None:
    dynamics = j2_search_dynamics(include_j2=False)
    period = orbital_period_s(MOON_MEAN_RADIUS_M + 5_000.0, MU)
    settings = StabilitySearchSettings(
        duration_s=period,
        sample_count=65,
        propagation=PropagationSettings(rtol=1e-10, max_step_s=30.0),
        constraints=StabilityConstraints(require_full_duration=True),
    )
    result = run_stability_search(
        _space(
            altitudes=(5_000.0,),
            eccentricities=(0.1,),
            inclinations_deg=(45.0,),
            anomaly_deg=(180.0,),
        ),
        dynamics,
        settings=settings,
    )
    candidate = result.candidates[0]
    assert candidate.metrics.impacted
    assert candidate.metrics.survived_duration_fraction < 1.0
    assert not candidate.passed_constraints
    assert any("survive" in violation for violation in candidate.constraint_violations)
    assert candidate.ranking.normalised_terms["lifetime_shortfall"] > 0.0
    assert "periselene_spread" in candidate.ranking.normalised_terms


def test_parallel_execution_is_deterministic_for_parallel_safe_dynamics() -> None:
    dynamics = j2_search_dynamics(include_j2=False)
    space = _space(altitudes=(90_000.0, 110_000.0), inclinations_deg=(85.0, 95.0))
    serial = run_stability_search(space, dynamics, settings=_settings(workers=1))
    parallel = run_stability_search(space, dynamics, settings=_settings(workers=2))
    assert [candidate.parameters.as_dict() for candidate in serial.candidates] == [
        candidate.parameters.as_dict() for candidate in parallel.candidates
    ]
    np.testing.assert_allclose(
        [candidate.ranking.penalty for candidate in serial.candidates],
        [candidate.ranking.penalty for candidate in parallel.candidates],
        rtol=0.0,
        atol=1e-13,
    )


def test_uncertainty_realizations_are_retained_and_summarized() -> None:
    dynamics = j2_search_dynamics(include_j2=False)
    result = run_stability_search(
        _space(),
        dynamics,
        settings=_settings(),
        uncertainty_dynamics=(dynamics, dynamics),
    )
    candidate = result.candidates[0]
    assert len(candidate.uncertainty_metrics) == 2
    assert candidate.uncertainty_summary is not None
    assert candidate.uncertainty_summary.realization_count == 2
    assert candidate.uncertainty_summary.impact_fraction == pytest.approx(0.0)
    assert candidate.uncertainty_summary.periselene_altitude_peak_to_peak_m == pytest.approx(
        candidate.metrics.periselene_altitude_peak_to_peak_m,
        rel=1e-10,
        abs=1e-8,
    )


def test_terrain_clearance_constraint_uses_terrain_not_reference_altitude() -> None:
    terrain = RegularLatLonTerrain(
        latitude_deg=np.array([-90.0, 0.0, 90.0]),
        longitude_deg_east=np.array([0.0, 180.0, 360.0]),
        elevation_grid_m=np.full((3, 3), 1_000.0),
        reference_radius_m=MOON_MEAN_RADIUS_M,
        frame="TEST_BODY",
    )
    dynamics = j2_search_dynamics(include_j2=False)
    constraints = StabilityConstraints(minimum_terrain_clearance_m=8_500.0)
    result = run_stability_search(
        _space(altitudes=(10_000.0,), eccentricities=(0.0,), inclinations_deg=(0.0,)),
        dynamics,
        settings=_settings(constraints=constraints),
        terrain=terrain,
        terrain_body_fixed_from_inertial=lambda _time_s: np.eye(3),
        terrain_frame="TEST_BODY",
    )
    candidate = result.candidates[0]
    assert candidate.metrics.minimum_reference_altitude_m == pytest.approx(10_000.0, abs=0.05)
    assert candidate.metrics.minimum_terrain_clearance_m == pytest.approx(9_000.0, abs=0.05)
    assert candidate.passed_constraints


def test_coarse_to_fine_refinement_stays_structured_and_reproducible() -> None:
    dynamics = j2_search_dynamics(include_j2=True)
    space = _space(
        altitudes=(80_000.0, 120_000.0),
        eccentricities=(0.01,),
        inclinations_deg=(85.0, 95.0),
    )
    result = run_coarse_to_fine_search(
        space,
        dynamics,
        settings=_settings(),
        refinement=RefinementSettings(
            top_candidates=1,
            points_per_axis=3,
            spacing_fraction=0.5,
            refine_axes=("semi_major_axis_m", "inclination_rad"),
        ),
    )
    assert result.seed_candidate_ids == (result.coarse.candidates[0].candidate_id,)
    assert result.refined.unique_candidate_count >= 4
    assert all(candidate.candidate_id.startswith("R") for candidate in result.refined.candidates)
    assert all(80_000.0 <= candidate.parameters.semi_major_axis_altitude_m <= 120_000.0 for candidate in result.refined.candidates)


def test_stability_map_selects_best_ranked_candidate_in_each_cell() -> None:
    dynamics = j2_search_dynamics(include_j2=True)
    result = run_stability_search(
        _space(
            altitudes=(90_000.0, 110_000.0),
            eccentricities=(0.005, 0.01),
            inclinations_deg=(85.0, 95.0),
        ),
        dynamics,
        settings=_settings(),
    )
    stability_map = make_stability_map(
        result,
        "semi_major_axis_altitude_m",
        "inclination_deg",
        metric="periselene_altitude_peak_to_peak_m",
    )
    assert stability_map.x_values == (90_000.0, 110_000.0)
    assert stability_map.y_values == (85.0, 95.0)
    assert len(stability_map.values) == 2
    assert all(len(row) == 2 for row in stability_map.values)
    assert all(candidate_id is not None for row in stability_map.candidate_ids for candidate_id in row)


def test_harmonic_search_records_selected_degree_and_exports_json_csv(tmp_path) -> None:
    c = np.zeros((3, 3))
    s = np.zeros((3, 3))
    c[0, 0] = 1.0
    c[2, 0] = -GRGM1200A_J2.j2 / np.sqrt(5.0)
    model = SphericalHarmonicModel(
        MU,
        GRGM1200A_J2.reference_radius_m,
        c,
        s,
        name="synthetic degree-2 test field",
        frame="TEST_BODY",
    )
    dynamics = harmonic_search_dynamics(
        model,
        lambda _time_s: np.eye(3),
        max_degree=2,
        max_order=2,
    )
    assert dynamics.harmonic_degree == 2
    assert dynamics.harmonic_order == 2
    assert dynamics.provenance()["gravity_frame"] == "TEST_BODY"

    result = run_stability_search(_space(), dynamics, settings=_settings())
    json_path = tmp_path / "search.json"
    csv_path = tmp_path / "search.csv"
    result.write_json(json_path)
    result.write_csv(csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dynamics"]["harmonic_degree"] == 2
    assert payload["candidates"][0]["metrics"]["periselene_altitude_peak_to_peak_m"] is not None
    assert "candidate_id" in csv_path.read_text(encoding="utf-8").splitlines()[0]
