"""Configuration loading and validation for the EDFA simulation MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a simulation configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON configuration."""

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc

    validate_config(config)
    return config


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{context} is missing required field '{key}'")
    return mapping[key]


def validate_config(config: dict[str, Any]) -> None:
    """Validate the fields needed by the MVP engine."""

    if _require(config, "schema_version", "config") != 1:
        raise ConfigError("unsupported schema_version; expected 1")

    _require(config, "name", "config")
    _require(config, "seed", "config")
    metadata = _require(config, "metadata", "config")
    if "parameter_source" not in metadata or "vendor_confirmed" not in metadata:
        raise ConfigError(
            "config.metadata must state parameter_source and vendor_confirmed"
        )

    channels = _require(config, "channels", "config")
    for key in ("start_nm", "stop_nm", "count", "bandwidth_hz", "power_dbm"):
        _require(channels, key, "config.channels")
    if channels["count"] < 1:
        raise ConfigError("config.channels.count must be positive")
    if channels["start_nm"] >= channels["stop_nm"]:
        raise ConfigError("config.channels.start_nm must be below stop_nm")

    topology = _require(config, "topology", "config")
    if not topology:
        raise ConfigError("config.topology must contain at least one node")

    node_ids: set[str] = set()
    stage_count = 0
    for index, node in enumerate(topology):
        context = f"config.topology[{index}]"
        node_id = _require(node, "id", context)
        node_type = _require(node, "type", context)
        if node_id in node_ids:
            raise ConfigError(f"duplicate topology node id: {node_id}")
        node_ids.add(node_id)

        if node_type == "edfa_stage":
            stage_count += 1
            for key in (
                "target_gain_db",
                "reference_gain_db",
                "gain_max_db",
                "p_max_dbm",
                "nf_min_db",
                "pump",
            ):
                _require(node, key, context)
            pump = node["pump"]
            for key in ("scale_power_mw", "max_power_mw", "health"):
                _require(pump, key, f"{context}.pump")
            if node["gain_max_db"] <= node["target_gain_db"]:
                raise ConfigError(
                    f"{context}.gain_max_db must exceed target_gain_db"
                )
            if pump["scale_power_mw"] <= 0 or pump["max_power_mw"] <= 0:
                raise ConfigError(f"{context}.pump power values must be positive")
        elif node_type == "passive":
            _require(node, "loss_db", context)
        elif node_type == "tap":
            _require(node, "through_loss_db", context)
            monitor = _require(node, "monitor", context)
            for key in ("split_db", "bias_db", "noise_std_db", "dropout"):
                _require(monitor, key, f"{context}.monitor")
        else:
            raise ConfigError(f"{context} has unsupported type '{node_type}'")

    if stage_count == 0:
        raise ConfigError("config.topology must contain at least one EDFA stage")

    scenarios = _require(config, "scenarios", "config")
    if not scenarios:
        raise ConfigError("config.scenarios must contain at least one scenario")
    scenario_ids: set[str] = set()
    for index, scenario in enumerate(scenarios):
        context = f"config.scenarios[{index}]"
        scenario_id = _require(scenario, "id", context)
        if scenario_id in scenario_ids:
            raise ConfigError(f"duplicate scenario id: {scenario_id}")
        scenario_ids.add(scenario_id)
        faults = _require(scenario, "faults", context)
        for fault_index, fault in enumerate(faults):
            fault_context = f"{context}.faults[{fault_index}]"
            for key in ("id", "target", "kind", "value"):
                _require(fault, key, fault_context)
            if fault["target"] not in node_ids:
                raise ConfigError(
                    f"{fault_context}.target references unknown node "
                    f"'{fault['target']}'"
                )
