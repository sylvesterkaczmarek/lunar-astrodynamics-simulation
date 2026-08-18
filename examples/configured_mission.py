"""Run the configuration-driven low-polar lunar mission workflow through Python."""

from __future__ import annotations

import argparse
from pathlib import Path

from lunar_astrodynamics.mission import (
    build_mission_context,
    load_mission_config,
    run_access_workflow,
    run_mission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path(__file__).parent / "missions" / "polar_quick.toml")
    parser.add_argument("--output-dir", type=Path, default=Path("results/configured_mission"))
    args = parser.parse_args()

    config = load_mission_config(args.config)
    context = build_mission_context(config)
    try:
        mission = run_mission(context)
        access = run_access_workflow(context, mission)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        mission.write_json(args.output_dir / "mission.json")
        mission.write_csv(args.output_dir / "trajectory.csv")
        mission.write_standard_plots(args.output_dir / "plots")
        access.write_json(args.output_dir / "access.json")
        access.ground_track.write_csv(args.output_dir / "ground_track.csv")
        if access.sites is not None:
            access.sites.write_windows_csv(args.output_dir / "site_access_windows.csv")
        if access.coverage is not None:
            access.coverage.write_csv(args.output_dir / "coverage.csv")
        print(mission.human_summary())
    finally:
        context.close()


if __name__ == "__main__":
    main()
