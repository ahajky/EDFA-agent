"""Command line interface for the EDFA simulation MVP."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .config import load_config, validate_config
from .model import simulate_suite


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _case_row(case: dict[str, Any]) -> dict[str, Any]:
    summary = case["summary"]
    comparison = case["comparison_to_baseline"]
    stage_gains = ";".join(
        f"{node_id}={value:.3f}"
        for node_id, value in summary["stage_gains_db"].items()
        if value is not None
    )
    tap_observed = ";".join(
        f"{tap_id}={'missing' if value is None else f'{value:.3f}'}"
        for tap_id, value in summary["tap_observed_dbm"].items()
    )
    root_causes = ";".join(
        item["root_cause"] for item in case.get("diagnostic_evidence", [])
    )
    evidence = " | ".join(
        observation
        for item in case.get("diagnostic_evidence", [])
        for observation in item["observations"]
    )
    return {
        "case_id": case["case"]["id"],
        "root_causes": root_causes,
        "output_signal_total_dbm": f"{summary['output_signal_total_dbm']:.4f}",
        "output_osa_total_dbm": (
            "missing"
            if not any(
                value is not None
                for value in case["measurements"]["output_osa"][
                    "observed_signal_dbm"
                ]
            )
            else f"{_sum_db_values(case['measurements']['output_osa']['observed_signal_dbm']):.4f}"
        ),
        "total_gain_db": f"{summary['total_signal_gain_db']:.4f}",
        "flatness_db": f"{summary['output_channel_flatness_db']:.4f}",
        "effective_nf_db": f"{summary['effective_nf_db_mean']:.4f}",
        "stage_gains_db": stage_gains,
        "tap_observed_dbm": tap_observed,
        "output_delta_vs_baseline_db": (
            f"{comparison['output_signal_total_delta_db']:.4f}"
        ),
        "flatness_delta_vs_baseline_db": (
            f"{comparison['flatness_delta_db']:.4f}"
        ),
        "controller_iterations": str(case["truth"]["controller_iterations"]),
        "diagnostic_evidence": evidence,
    }


def _sum_db_values(values: Sequence[float | None]) -> float:
    total_mw = sum(
        10.0 ** (float(value) / 10.0) for value in values if value is not None
    )
    return 10.0 * math.log10(max(total_mw, 1e-18))


def _write_suite(output_dir: Path, config: dict[str, Any], suite: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "config_snapshot.json", config)
    for case in suite["cases"]:
        _json_dump(output_dir / f"{case['case']['id']}.json", case)

    rows = [_case_row(case) for case in suite["cases"]]
    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _json_dump(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "config_name": config["name"],
            "baseline_case": suite["baseline_case"],
            "seed": suite["seed"],
            "case_ids": [case["case"]["id"] for case in suite["cases"]],
            "parameter_source": config["metadata"]["parameter_source"],
            "vendor_confirmed": config["metadata"]["vendor_confirmed"],
        },
    )


def _print_summary(suite: dict[str, Any]) -> None:
    print(
        f"{'case':32} {'out dBm':>9} {'gain dB':>8} "
        f"{'flat dB':>8} {'Δout dB':>9} evidence"
    )
    for case in suite["cases"]:
        summary = case["summary"]
        comparison = case["comparison_to_baseline"]
        root = ", ".join(
            item["root_cause"] for item in case.get("diagnostic_evidence", [])
        )
        print(
            f"{case['case']['id']:32} "
            f"{summary['output_signal_total_dbm']:9.3f} "
            f"{summary['total_signal_gain_db']:8.2f} "
            f"{summary['output_channel_flatness_db']:8.3f} "
            f"{comparison['output_signal_total_delta_db']:9.3f} {root}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate explainable EDFA fault cases"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run configured scenarios")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/three_stage_demo.json"),
        help="scenario configuration JSON",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/three_stage_demo"),
        help="output directory",
    )
    run_parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="run only this scenario id; may be repeated",
    )
    run_parser.add_argument("--seed", type=int, help="override configuration seed")

    validate_parser = subparsers.add_parser(
        "validate", help="validate a scenario configuration"
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/three_stage_demo.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        print(
            f"valid: {config['name']} "
            f"({len(config['topology'])} nodes, "
            f"{len(config['scenarios'])} scenarios)"
        )
        return 0

    suite = simulate_suite(config, seed=args.seed, scenario_ids=args.cases)
    _write_suite(args.output, config, suite)
    _print_summary(suite)
    print(f"\nWrote {len(suite['cases'])} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
