"""Command-line access to the configuration-driven lunar mission workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from .mission import (
    build_mission_context,
    ensemble_to_dict,
    load_mission_config,
    run_access_workflow,
    run_fidelity_workflow,
    run_frozen_orbit_workflow,
    run_mission,
    run_uncertainty_workflow,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunar-mission",
        description="Configuration-driven lunar mission analysis using the package scientific APIs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("propagate", "propagate the configured orbit and write machine-readable trajectory outputs"),
        ("analyse", "propagate, analyse orbital evolution, and write the standard scientific plots"),
        ("search", "run the configured low-lunar stability/frozen-orbit search"),
        ("uncertainty", "run the configured gravity-field uncertainty ensemble"),
        ("fidelity", "evaluate configured gravity degree/order convergence and runtime"),
        ("access", "calculate ground track, site access, coverage and Earth visibility"),
        ("all", "run propagation/analysis and every optional configured downstream workflow"),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("config", type=Path, help="mission TOML configuration")
        item.add_argument("--output-dir", type=Path, default=Path("results/mission"), help="directory for JSON, CSV, summary and plot outputs")
    return parser


def _write_mission(context, output: Path, *, plots: bool):
    run = run_mission(context)
    output.mkdir(parents=True, exist_ok=True)
    run.write_json(output / "mission.json")
    run.write_csv(output / "trajectory.csv")
    (output / "summary.txt").write_text(run.human_summary() + "\n", encoding="utf-8")
    if plots:
        run.write_standard_plots(output / "plots")
    print(run.human_summary())
    return run


def _write_access(context, output: Path, mission=None) -> None:
    result = run_access_workflow(context, mission)
    output.mkdir(parents=True, exist_ok=True)
    result.write_json(output / "access.json")
    result.ground_track.write_csv(output / "ground_track.csv")
    if result.sites is not None:
        result.sites.write_windows_csv(output / "site_access_windows.csv")
    if result.coverage is not None:
        result.coverage.write_csv(output / "coverage.csv")
    if result.earth_visibility is not None:
        result.earth_visibility.write_csv(output / "earth_visibility.csv")  # type: ignore[attr-defined]


def _write_search(context, output: Path) -> None:
    result = run_frozen_orbit_workflow(context)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": context.provenance(), "search": result.to_dict()}
    (output / "frozen_orbit_search.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if hasattr(result, "refined"):
        result.refined.write_csv(output / "frozen_orbit_candidates.csv")
    else:
        result.write_csv(output / "frozen_orbit_candidates.csv")


def _write_uncertainty(context, output: Path) -> None:
    result = run_uncertainty_workflow(context)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gravity_uncertainty.json").write_text(json.dumps(ensemble_to_dict(result, context.provenance()), indent=2) + "\n", encoding="utf-8")


def _write_fidelity(context, output: Path, mission=None) -> None:
    result = run_fidelity_workflow(context, mission)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fidelity.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    result.acceleration.write_csv(output / "fidelity_acceleration.csv")
    result.trajectory.write_csv(output / "fidelity_trajectory.csv")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_mission_config(args.config)
        context = build_mission_context(config)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        output = args.output_dir
        if args.command == "propagate":
            _write_mission(context, output, plots=False)
        elif args.command == "analyse":
            _write_mission(context, output, plots=True)
        elif args.command == "search":
            _write_search(context, output)
        elif args.command == "uncertainty":
            _write_uncertainty(context, output)
        elif args.command == "fidelity":
            mission = _write_mission(context, output, plots=False)
            _write_fidelity(context, output, mission)
        elif args.command == "access":
            mission = _write_mission(context, output, plots=False)
            _write_access(context, output, mission)
        else:
            mission = _write_mission(context, output, plots=True)
            _write_access(context, output, mission)
            raw = config.raw
            if isinstance(raw.get("search"), dict):
                _write_search(context, output)
            if isinstance(raw.get("uncertainty"), dict):
                _write_uncertainty(context, output)
            if isinstance(raw.get("fidelity"), dict):
                _write_fidelity(context, output, mission)
        return 0
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 2
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
