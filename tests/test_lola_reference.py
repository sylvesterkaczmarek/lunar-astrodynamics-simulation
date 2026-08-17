from pathlib import Path

import numpy as np
import pytest

from lunar_astrodynamics import load_lola_pds_global_gdr

# These values were read on 2026-08-17 from the official NASA/PDS
# LDEM_4.IMG + LDEM_4.LBL product by scripts/validate_lola_pds_reference.py.
# The URLs and data provenance are documented in docs/terrain.md.
OFFICIAL_LDEM4_REFERENCE_POINTS = (
    (0.125, 0.125, -796.0),
    (0.125, 90.125, -3814.5),
    (0.125, 180.125, 2432.0),
    (45.125, 45.125, -699.0),
    (-45.125, 315.125, -301.0),
    (89.875, 0.125, -119.5),
    (-89.875, 180.125, 172.0),
)


def _official_metadata_label() -> str:
    return '''PDS_VERSION_ID = PDS3
LINES = 720
LINE_SAMPLES = 1440
SAMPLE_TYPE = MSB_INTEGER
SAMPLE_BITS = 16
SCALING_FACTOR = 0.5
OFFSET = 1737400 <m>
MAP_RESOLUTION = 4 <pix/deg>
POSITIVE_LONGITUDE_DIRECTION = EAST
COORDINATE_SYSTEM_NAME = "MEAN EARTH/POLAR AXIS OF DE421"
'''


def test_selected_official_ldem4_grid_values_preserve_pds_indexing_and_scaling(tmp_path: Path) -> None:
    """Regression against selected values observed directly in the official PDS grid."""
    lines, samples = 720, 1440
    raw = np.zeros((lines, samples), dtype=">i2")
    for latitude_deg, longitude_deg, elevation_m in OFFICIAL_LDEM4_REFERENCE_POINTS:
        loaded_i = int(round((latitude_deg - (-89.875)) / 0.25))
        file_i = lines - 1 - loaded_i
        j = int(round((longitude_deg - 0.125) / 0.25))
        raw[file_i, j] = int(round(elevation_m / 0.5))

    image = tmp_path / "LDEM_4_reference_subset.IMG"
    label = tmp_path / "LDEM_4_reference_subset.LBL"
    raw.tofile(image)
    label.write_text(_official_metadata_label(), encoding="ascii")
    terrain = load_lola_pds_global_gdr(image, label)

    assert terrain.reference_radius_m == pytest.approx(1_737_400.0)
    assert terrain.frame == "MEAN EARTH/POLAR AXIS OF DE421"
    for latitude_deg, longitude_deg, expected_elevation_m in OFFICIAL_LDEM4_REFERENCE_POINTS:
        i = int(np.argmin(np.abs(terrain.latitude_deg - latitude_deg)))
        j = int(np.argmin(np.abs(terrain.longitude_deg_east - longitude_deg)))
        assert terrain.latitude_deg[i] == pytest.approx(latitude_deg)
        assert terrain.longitude_deg_east[j] == pytest.approx(longitude_deg)
        assert terrain.elevation_grid_m[i, j] == pytest.approx(expected_elevation_m)
