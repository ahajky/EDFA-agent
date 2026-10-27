"""Explainable grey-box simulation of a multi-stage EDFA chain."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import validate_config

SPEED_OF_LIGHT_M_S = 299_792_458.0
PLANCK_J_S = 6.626_070_15e-34
MIN_POWER_MW = 1e-18
INACTIVE_POWER_DBM = -120.0


@dataclass
class ChannelGrid:
    """Channel plan and active-channel mask."""

    wavelength_nm: np.ndarray
    frequency_hz: np.ndarray
    bandwidth_hz: np.ndarray
    normalized_position: np.ndarray
    active: np.ndarray
    input_signal_mw: np.ndarray
    input_power_dbm: np.ndarray


def dbm_to_mw(value: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value) / 10.0)


def mw_to_dbm(value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(value), MIN_POWER_MW))


def db_to_linear(value: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value) / 10.0)


def linear_to_db(value: float | np.ndarray) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(value), 1e-30))


def _sum_dbm(power_mw: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = power_mw if mask is None else power_mw[mask]
    return float(mw_to_dbm(np.sum(selected, dtype=float)))


def _channel_grid(
    config: dict[str, Any],
    input_power_delta_db: float = 0.0,
) -> ChannelGrid:
    channels = config["channels"]
    count = int(channels["count"])
    wavelength_nm = np.linspace(
        float(channels["start_nm"]),
        float(channels["stop_nm"]),
        count,
        dtype=float,
    )
    frequency_hz = SPEED_OF_LIGHT_M_S / (wavelength_nm * 1e-9)
    bandwidth_hz = np.full(count, float(channels["bandwidth_hz"]), dtype=float)
    if count == 1:
        normalized_position = np.zeros(1, dtype=float)
    else:
        normalized_position = np.linspace(-1.0, 1.0, count, dtype=float)

    configured_power = channels["power_dbm"]
    if isinstance(configured_power, list):
        if len(configured_power) != count:
            raise ValueError(
                "channels.power_dbm list length must match channels.count"
            )
        input_power_dbm = np.asarray(configured_power, dtype=float)
    else:
        input_power_dbm = np.full(count, float(configured_power), dtype=float)

    if channels.get("active_channels") is not None:
        active = np.zeros(count, dtype=bool)
        active_indices = np.asarray(channels["active_channels"], dtype=int)
        if np.any(active_indices < 0) or np.any(active_indices >= count):
            raise ValueError("channels.active_channels contains an invalid index")
        active[active_indices] = True
    else:
        active = input_power_dbm > -100.0

    input_power_dbm = input_power_dbm + float(input_power_delta_db)
    input_power_dbm = np.where(active, input_power_dbm, INACTIVE_POWER_DBM)
    input_signal_mw = np.where(active, dbm_to_mw(input_power_dbm), 0.0)
    return ChannelGrid(
        wavelength_nm=wavelength_nm,
        frequency_hz=frequency_hz,
        bandwidth_hz=bandwidth_hz,
        normalized_position=normalized_position,
        active=active,
        input_signal_mw=input_signal_mw,
        input_power_dbm=input_power_dbm,
    )


def _node_by_id(config: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(node for node in config["topology"] if node["id"] == node_id)


def _get_path(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        current = current[key]
    return current


def _set_path(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


FAULT_PATHS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "pump_health": (
        ("pump", "health"),
        "absolute",
        "pump conversion health changes; AGC may raise the command",
    ),
    "pump_max_scale": (
        ("pump", "max_power_mw"),
        "multiply",
        "pump current/power headroom is reduced",
    ),
    "add_loss_db": (
        ("loss_db",),
        "add",
        "additional passive insertion loss is inserted at this node",
    ),
    "gain_max_db_delta": (
        ("gain_max_db",),
        "add",
        "the stage's maximum intrinsic gain changes",
    ),
    "gain_ripple_shift_nm": (
        ("gain_ripple_shift_nm",),
        "add",
        "the wavelength-dependent gain or filter shape shifts",
    ),
    "pd_bias_db": (
        ("monitor", "bias_db"),
        "add",
        "the tap/PD measurement chain adds a calibration bias",
    ),
    "pd_dropout": (
        ("monitor", "dropout"),
        "absolute",
        "the tap/PD output is unavailable",
    ),
    "temperature_delta_c": (
        ("temperature_c",),
        "add",
        "the device temperature changes",
    ),
}


def _apply_scenario(
    config: dict[str, Any], scenario: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    effective = copy.deepcopy(config)
    events: list[dict[str, Any]] = []
    input_power_delta_db = float(scenario.get("input_power_delta_db", 0.0))
    if input_power_delta_db:
        events.append(
            {
                "id": "input_workpoint",
                "target": "input",
                "kind": "input_power_delta_db",
                "parameter": "channels.power_dbm",
                "before": 0.0,
                "after": input_power_delta_db,
                "delta": input_power_delta_db,
                "mechanism": "all configured channel powers shift by the same amount",
            }
        )

    for fault in scenario["faults"]:
        node = _node_by_id(effective, fault["target"])
        kind = fault["kind"]
        if kind not in FAULT_PATHS:
            raise ValueError(f"unsupported fault kind: {kind}")
        path, operation, mechanism = FAULT_PATHS[kind]
        before = _get_path(node, path)
        if operation == "absolute":
            after = fault["value"]
        elif operation == "add":
            after = before + fault["value"]
        elif operation == "multiply":
            after = before * fault["value"]
        else:
            raise AssertionError(operation)
        _set_path(node, path, after)
        try:
            delta: float | None = float(after) - float(before)
        except (TypeError, ValueError):
            delta = None
        events.append(
            {
                "id": fault["id"],
                "target": fault["target"],
                "kind": kind,
                "parameter": ".".join((fault["target"], *path)),
                "before": before,
                "after": after,
                "delta": delta,
                "mechanism": mechanism,
                "description": fault.get("description", ""),
            }
        )
    return effective, events, input_power_delta_db


def _shifted_profile(
    profile_db: Iterable[float],
    normalized_position: np.ndarray,
    active: np.ndarray,
    shift_nm: float,
    span_nm: float,
) -> np.ndarray:
    profile = np.asarray(list(profile_db), dtype=float)
    if profile.size == 0:
        return np.zeros_like(normalized_position)
    control_position = np.linspace(-1.0, 1.0, profile.size)
    shift_position = 2.0 * shift_nm / span_nm if span_nm > 0 else 0.0
    query = np.clip(normalized_position - shift_position, -1.0, 1.0)
    result = np.interp(query, control_position, profile)
    if np.any(active):
        result = result - float(np.mean(result[active]))
    return result


def _pump_required_mw(stage: dict[str, Any], requested_gain_db: float) -> float:
    gain_max = float(stage["gain_max_db"])
    requested = min(float(requested_gain_db), gain_max * (1.0 - 1e-9))
    if requested <= 0:
        return 1e-9
    scale = float(stage["pump"]["scale_power_mw"])
    return -scale * math.log1p(-requested / gain_max)


def _pump_command_mw(stage: dict[str, Any], requested_gain_db: float) -> float:
    pump = stage["pump"]
    health = max(float(pump.get("health", 1.0)), 1e-6)
    required = _pump_required_mw(stage, requested_gain_db)
    return min(required / health, float(pump["max_power_mw"]))


def _pump_actual_mw(
    stage: dict[str, Any], command_mw: float
) -> tuple[float, bool]:
    pump = stage["pump"]
    health = max(float(pump.get("health", 1.0)), 1e-6)
    max_power = float(pump["max_power_mw"])
    limited = command_mw >= max_power * (1.0 - 1e-9)
    return min(command_mw, max_power) * health, limited


def _small_signal_gain(stage: dict[str, Any], actual_pump_mw: float) -> float:
    gain_max = float(stage["gain_max_db"])
    scale = float(stage["pump"]["scale_power_mw"])
    return gain_max * (1.0 - math.exp(-actual_pump_mw / scale))


def _gain_profile(
    stage: dict[str, Any],
    grid: ChannelGrid,
    small_signal_gain_db: float,
) -> np.ndarray:
    span_nm = float(grid.wavelength_nm[-1] - grid.wavelength_nm[0])
    ripple = _shifted_profile(
        stage.get("gain_ripple_db", []),
        grid.normalized_position,
        grid.active,
        float(stage.get("gain_ripple_shift_nm", 0.0)),
        span_nm,
    )
    reference_gain = float(stage["reference_gain_db"])
    dynamic_tilt = float(stage.get("dgt_db_per_db", 0.0))
    temperature_coefficient = float(
        stage.get("temperature_coefficient_db_per_c", 0.0)
    )
    temperature_delta = float(stage.get("temperature_c", 25.0)) - float(
        stage.get("reference_temperature_c", 25.0)
    )
    shape = (
        ripple
        + dynamic_tilt
        * (small_signal_gain_db - reference_gain)
        * grid.normalized_position
        + temperature_coefficient * temperature_delta * grid.normalized_position
    )
    if np.any(grid.active):
        shape = shape - float(np.mean(shape[grid.active]))
    return small_signal_gain_db + shape


def _noise_figure(
    stage: dict[str, Any],
    grid: ChannelGrid,
    small_signal_gain_db: float,
) -> np.ndarray:
    deficit = max(float(stage["gain_max_db"]) - small_signal_gain_db, 0.0)
    nf_db = float(stage["nf_min_db"]) + float(
        stage.get("nf_gain_penalty_db_per_db", 0.0)
    ) * deficit
    span_nm = float(grid.wavelength_nm[-1] - grid.wavelength_nm[0])
    nf_ripple = _shifted_profile(
        stage.get("nf_ripple_db", []),
        grid.normalized_position,
        grid.active,
        float(stage.get("gain_ripple_shift_nm", 0.0)),
        span_nm,
    )
    temperature_delta = float(stage.get("temperature_c", 25.0)) - float(
        stage.get("reference_temperature_c", 25.0)
    )
    nf_db = (
        nf_db
        + nf_ripple
        + float(stage.get("nf_temperature_db_per_c", 0.0)) * temperature_delta
    )
    return nf_db


def _apply_stage(
    node: dict[str, Any],
    signal_mw: np.ndarray,
    ase_mw: np.ndarray,
    grid: ChannelGrid,
    command_mw: float,
    requested_gain_db: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    input_signal_total_dbm = _sum_dbm(signal_mw)
    input_ase_total_dbm = _sum_dbm(ase_mw)
    actual_pump_mw, pump_limited = _pump_actual_mw(node, command_mw)
    small_signal_gain_db = _small_signal_gain(node, actual_pump_mw)
    gain_profile_db = _gain_profile(node, grid, small_signal_gain_db)
    nf_profile_db = _noise_figure(node, grid, small_signal_gain_db)
    gain_linear = db_to_linear(gain_profile_db)

    nf_linear = db_to_linear(nf_profile_db)
    added_ase_before_gain_mw = (
        PLANCK_J_S
        * grid.frequency_hz
        * grid.bandwidth_hz
        * nf_linear
        * 1_000.0
    )
    ase_before_gain_mw = ase_mw + added_ase_before_gain_mw
    raw_signal_mw = signal_mw * gain_linear
    raw_ase_mw = ase_before_gain_mw * gain_linear
    raw_total_mw = float(np.sum(raw_signal_mw + raw_ase_mw))
    output_cap_mw = float(dbm_to_mw(node["p_max_dbm"]))
    compression_db = 0.0
    if raw_total_mw > output_cap_mw:
        scale = output_cap_mw / raw_total_mw
        signal_mw = raw_signal_mw * scale
        ase_mw = raw_ase_mw * scale
        compression_db = float(-linear_to_db(scale))
    else:
        signal_mw = raw_signal_mw
        ase_mw = raw_ase_mw

    trace = {
        "node_id": node["id"],
        "type": "edfa_stage",
        "input_signal_total_dbm": input_signal_total_dbm,
        "input_ase_total_dbm": input_ase_total_dbm,
        "output_signal_total_dbm": _sum_dbm(signal_mw),
        "output_ase_total_dbm": _sum_dbm(ase_mw),
        "requested_small_signal_gain_db": requested_gain_db,
        "small_signal_gain_db": small_signal_gain_db,
        "mean_signal_gain_db": None,
        "gain_flatness_out_db": None,
        "pump_command_mw": command_mw,
        "pump_actual_mw": actual_pump_mw,
        "pump_health": float(node["pump"].get("health", 1.0)),
        "pump_limited": pump_limited,
        "saturation_compression_db": compression_db,
        "mean_nf_db": float(np.mean(nf_profile_db[grid.active]))
        if np.any(grid.active)
        else None,
    }
    return signal_mw, ase_mw, trace


def _apply_passive(
    node: dict[str, Any],
    signal_mw: np.ndarray,
    ase_mw: np.ndarray,
    grid: ChannelGrid,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    span_nm = float(grid.wavelength_nm[-1] - grid.wavelength_nm[0])
    ripple = _shifted_profile(
        node.get("gain_ripple_db", []),
        grid.normalized_position,
        grid.active,
        float(node.get("gain_ripple_shift_nm", 0.0)),
        span_nm,
    )
    temperature_delta = float(node.get("temperature_c", 25.0)) - float(
        node.get("reference_temperature_c", 25.0)
    )
    loss_db = (
        float(node["loss_db"])
        + ripple
        + float(node.get("temperature_coefficient_db_per_c", 0.0))
        * temperature_delta
    )
    loss_db = np.maximum(loss_db, 0.0)
    transmission = db_to_linear(-loss_db)
    output_signal_mw = signal_mw * transmission
    output_ase_mw = ase_mw * transmission
    trace = {
        "node_id": node["id"],
        "type": "passive",
        "input_signal_total_dbm": _sum_dbm(signal_mw),
        "output_signal_total_dbm": _sum_dbm(output_signal_mw),
        "loss_db": float(node["loss_db"]),
        "mean_total_loss_db": float(np.mean(loss_db[grid.active]))
        if np.any(grid.active)
        else float(np.mean(loss_db)),
    }
    return output_signal_mw, output_ase_mw, trace


def _apply_tap(
    node: dict[str, Any],
    signal_mw: np.ndarray,
    ase_mw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    total_power_mw = float(np.sum(signal_mw + ase_mw))
    monitor_true_dbm = float(mw_to_dbm(total_power_mw)) + float(
        node["monitor"]["split_db"]
    )
    transmission = float(db_to_linear(-float(node["through_loss_db"])))
    output_signal_mw = signal_mw * transmission
    output_ase_mw = ase_mw * transmission
    trace = {
        "node_id": node["id"],
        "type": "tap",
        "input_signal_total_dbm": _sum_dbm(signal_mw),
        "input_total_dbm": float(mw_to_dbm(total_power_mw)),
        "monitor_true_dbm": monitor_true_dbm,
        "through_loss_db": float(node["through_loss_db"]),
        "output_signal_total_dbm": _sum_dbm(output_signal_mw),
    }
    return output_signal_mw, output_ase_mw, trace


def _propagate(
    config: dict[str, Any],
    grid: ChannelGrid,
    commands_mw: dict[str, float],
    requested_gains_db: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    signal_mw = grid.input_signal_mw.copy()
    ase_mw = np.zeros_like(signal_mw)
    traces: list[dict[str, Any]] = []

    for node in config["topology"]:
        node_type = node["type"]
        if node_type == "edfa_stage":
            stage_input_signal = signal_mw.copy()
            signal_mw, ase_mw, trace = _apply_stage(
                node,
                signal_mw,
                ase_mw,
                grid,
                commands_mw[node["id"]],
                requested_gains_db[node["id"]],
            )
            gains = linear_to_db(
                signal_mw[grid.active]
                / np.maximum(stage_input_signal[grid.active], MIN_POWER_MW)
            )
            output_dbm = mw_to_dbm(signal_mw[grid.active])
            trace["mean_signal_gain_db"] = (
                float(np.mean(gains)) if gains.size else None
            )
            trace["gain_flatness_out_db"] = (
                float(np.max(output_dbm) - np.min(output_dbm))
                if output_dbm.size
                else None
            )
        elif node_type == "passive":
            signal_mw, ase_mw, trace = _apply_passive(
                node, signal_mw, ase_mw, grid
            )
        elif node_type == "tap":
            signal_mw, ase_mw, trace = _apply_tap(
                node, signal_mw, ase_mw
            )
        else:
            raise AssertionError(f"unsupported node type: {node_type}")
        traces.append(trace)
    return signal_mw, ase_mw, traces


def _measure_taps(
    config: dict[str, Any],
    traces: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    taps = {node["id"]: node for node in config["topology"] if node["type"] == "tap"}
    measurements: dict[str, dict[str, Any]] = {}
    for trace in traces:
        if trace["type"] != "tap":
            continue
        node_id = trace["node_id"]
        monitor = taps[node_id]["monitor"]
        true_dbm = float(trace["monitor_true_dbm"])
        if bool(monitor.get("dropout", False)):
            observed_dbm = None
            residual_db = None
        else:
            observed_dbm = (
                true_dbm
                + float(monitor.get("bias_db", 0.0))
                + float(rng.normal(0.0, float(monitor.get("noise_std_db", 0.0))))
            )
            residual_db = observed_dbm - true_dbm
        measurements[node_id] = {
            "true_dbm": true_dbm,
            "observed_dbm": observed_dbm,
            "residual_db": residual_db,
            "quality": "missing" if observed_dbm is None else "ok",
        }
    return measurements


def _measure_osa(
    config: dict[str, Any],
    signal_mw: np.ndarray,
    ase_mw: np.ndarray,
    grid: ChannelGrid,
    rng: np.random.Generator,
) -> dict[str, Any]:
    osa = config.get("output_osa", {})
    true_signal_dbm = mw_to_dbm(signal_mw)
    true_spectral_dbm = mw_to_dbm(signal_mw + ase_mw)
    center_nm = float(np.mean(grid.wavelength_nm))
    bias_db = float(osa.get("bias_db", 0.0))
    slope_db_per_nm = float(osa.get("slope_db_per_nm", 0.0))
    noise_std_db = float(osa.get("noise_std_db", 0.0))
    dropout_channels = set(int(channel) for channel in osa.get("dropout_channels", []))
    observed_signal_dbm: list[float | None] = []
    observed_spectral_dbm: list[float | None] = []
    for index in range(grid.wavelength_nm.size):
        if not grid.active[index] or index in dropout_channels:
            observed_signal_dbm.append(None)
            observed_spectral_dbm.append(None)
            continue
        correction = (
            bias_db
            + slope_db_per_nm * (grid.wavelength_nm[index] - center_nm)
            + float(rng.normal(0.0, noise_std_db))
        )
        observed_signal_dbm.append(float(true_signal_dbm[index]) + correction)
        observed_spectral_dbm.append(float(true_spectral_dbm[index]) + correction)
    observed_values = np.asarray(
        [value for value in observed_signal_dbm if value is not None], dtype=float
    )
    true_values = true_signal_dbm[grid.active]
    return {
        "wavelength_nm": grid.wavelength_nm.tolist(),
        "true_signal_dbm": [
            float(value) if bool(active) else None
            for value, active in zip(true_signal_dbm, grid.active)
        ],
        "true_spectral_dbm": [
            float(value) if bool(active) else None
            for value, active in zip(true_spectral_dbm, grid.active)
        ],
        "observed_signal_dbm": observed_signal_dbm,
        "observed_spectral_dbm": observed_spectral_dbm,
        "true_flatness_db": float(np.max(true_values) - np.min(true_values))
        if true_values.size
        else None,
        "observed_flatness_db": float(np.max(observed_values) - np.min(observed_values))
        if observed_values.size
        else None,
    }


def _summary(
    grid: ChannelGrid,
    signal_mw: np.ndarray,
    ase_mw: np.ndarray,
    traces: list[dict[str, Any]],
    measurements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_total_dbm = _sum_dbm(grid.input_signal_mw, grid.active)
    output_total_dbm = _sum_dbm(signal_mw, grid.active)
    output_channels_dbm = mw_to_dbm(signal_mw[grid.active])
    output_ase_dbm = _sum_dbm(ase_mw, grid.active)
    total_gain_db = output_total_dbm - input_total_dbm
    stage_gains = {
        trace["node_id"]: trace["mean_signal_gain_db"]
        for trace in traces
        if trace["type"] == "edfa_stage"
    }
    effective_nf_db: dict[str, float] = {}
    for index in np.flatnonzero(grid.active):
        channel_gain = float(signal_mw[index] / grid.input_signal_mw[index])
        if channel_gain <= 0:
            continue
        equivalent_nf_linear = float(ase_mw[index]) / (
            PLANCK_J_S
            * float(grid.frequency_hz[index])
            * float(grid.bandwidth_hz[index])
            * channel_gain
            * 1_000.0
        )
        effective_nf_db[str(int(index))] = float(linear_to_db(equivalent_nf_linear))
    tap_true = {
        node_id: values["true_dbm"]
        for node_id, values in measurements.items()
    }
    tap_observed = {
        node_id: values["observed_dbm"]
        for node_id, values in measurements.items()
    }
    if output_channels_dbm.size > 1:
        slope = np.polyfit(
            grid.normalized_position[grid.active],
            output_channels_dbm,
            1,
        )[0]
        tilt_db = float(2.0 * slope)
    else:
        tilt_db = 0.0
    return {
        "input_signal_total_dbm": input_total_dbm,
        "output_signal_total_dbm": output_total_dbm,
        "output_ase_total_dbm": output_ase_dbm,
        "total_signal_gain_db": total_gain_db,
        "output_channel_flatness_db": float(
            np.max(output_channels_dbm) - np.min(output_channels_dbm)
        )
        if output_channels_dbm.size
        else None,
        "output_channel_tilt_db": tilt_db,
        "stage_gains_db": stage_gains,
        "effective_nf_db_mean": float(np.mean(list(effective_nf_db.values())))
        if effective_nf_db
        else None,
        "tap_true_dbm": tap_true,
        "tap_observed_dbm": tap_observed,
    }


def _comparison(
    result: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    current = result["summary"]
    reference = baseline["summary"]

    def delta(key: str) -> float | None:
        left = current.get(key)
        right = reference.get(key)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    stage_deltas: dict[str, float | None] = {}
    all_stages = set(current["stage_gains_db"]) | set(reference["stage_gains_db"])
    for stage_id in sorted(all_stages):
        left = current["stage_gains_db"].get(stage_id)
        right = reference["stage_gains_db"].get(stage_id)
        stage_deltas[stage_id] = (
            float(left) - float(right)
            if left is not None and right is not None
            else None
        )
    tap_deltas: dict[str, float | None] = {}
    for tap_id in current["tap_true_dbm"]:
        left = current["tap_true_dbm"].get(tap_id)
        right = reference["tap_true_dbm"].get(tap_id)
        tap_deltas[tap_id] = (
            float(left) - float(right)
            if left is not None and right is not None
            else None
        )
    return {
        "output_signal_total_delta_db": delta("output_signal_total_dbm"),
        "total_gain_delta_db": delta("total_signal_gain_db"),
        "flatness_delta_db": delta("output_channel_flatness_db"),
        "tilt_delta_db": delta("output_channel_tilt_db"),
        "stage_gain_deltas_db": stage_deltas,
        "tap_true_deltas_db": tap_deltas,
    }


def _downstream_tap_deltas(
    result: dict[str, Any],
    baseline: dict[str, Any],
    target: str,
) -> list[dict[str, Any]]:
    traces = result["truth"]["propagation"]
    baseline_traces = {
        trace["node_id"]: trace for trace in baseline["truth"]["propagation"]
    }
    target_index = next(
        (index for index, trace in enumerate(traces) if trace["node_id"] == target),
        None,
    )
    if target_index is None:
        return []
    evidence: list[dict[str, Any]] = []
    for trace in traces[target_index + 1 :]:
        if trace["type"] != "tap" or trace["node_id"] not in baseline_traces:
            continue
        reference = baseline_traces[trace["node_id"]]
        evidence.append(
            {
                "tap": trace["node_id"],
                "true_delta_db": float(trace["input_total_dbm"])
                - float(reference["input_total_dbm"]),
            }
        )
    return evidence


def _diagnostic_evidence(
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    comparison = result["comparison_to_baseline"]
    stage_traces = {
        trace["node_id"]: trace
        for trace in result["truth"]["propagation"]
        if trace["type"] == "edfa_stage"
    }
    baseline_stages = {
        trace["node_id"]: trace
        for trace in baseline["truth"]["propagation"]
        if trace["type"] == "edfa_stage"
    }
    tap_deltas = comparison["tap_true_deltas_db"]

    for event in result["truth"]["events"]:
        kind = event["kind"]
        target = event["target"]
        item: dict[str, Any] = {
            "event_id": event["id"],
            "root_cause": f"{target}.{kind}",
            "mechanism": event["mechanism"],
            "observations": [],
        }
        if kind == "pump_health" and target in stage_traces:
            current = stage_traces[target]
            reference = baseline_stages[target]
            command_delta = float(current["pump_command_mw"]) - float(
                reference["pump_command_mw"]
            )
            gain_delta = float(current["mean_signal_gain_db"]) - float(
                reference["mean_signal_gain_db"]
            )
            item["observations"] = [
                f"AGC pump command changed by {command_delta:+.2f} mW",
                f"stage mean gain changed by {gain_delta:+.2f} dB",
                f"pump command limited: {current['pump_limited']}",
            ]
        elif kind == "add_loss_db":
            downstream = _downstream_tap_deltas(result, baseline, target)
            item["observations"] = [
                f"downstream tap {entry['tap']} changed by "
                f"{entry['true_delta_db']:+.2f} dB"
                for entry in downstream
            ]
            item["observations"].append(
                "link output changed by "
                f"{comparison['output_signal_total_delta_db']:+.2f} dB"
            )
        elif kind == "gain_ripple_shift_nm":
            flatness_delta = comparison["flatness_delta_db"]
            item["observations"] = [
                f"output flatness changed by {flatness_delta:+.3f} dB"
                if flatness_delta is not None
                else "output flatness unavailable",
                f"mean stage gain changed by "
                f"{comparison['stage_gain_deltas_db'].get(target, 0.0):+.3f} dB",
            ]
        elif kind in {"pd_bias_db", "pd_dropout"}:
            current_tap = result["measurements"]["taps"].get(target, {})
            reference_tap = baseline["measurements"]["taps"].get(target, {})
            observed_delta = None
            if (
                current_tap.get("observed_dbm") is not None
                and reference_tap.get("observed_dbm") is not None
            ):
                observed_delta = float(current_tap["observed_dbm"]) - float(
                    reference_tap["observed_dbm"]
                )
            item["observations"] = [
                f"tap true power changed by {tap_deltas.get(target, 0.0):+.3f} dB",
                "tap observed power is missing"
                if current_tap.get("observed_dbm") is None
                else f"tap observed power changed by {observed_delta:+.3f} dB",
                "optical output changed by "
                f"{comparison['output_signal_total_delta_db']:+.3f} dB",
            ]
        elif kind == "input_power_delta_db":
            item["observations"] = [
                f"all input channels shifted by {event['after']:+.2f} dB",
                "link output changed by "
                f"{comparison['output_signal_total_delta_db']:+.2f} dB",
            ]
        else:
            item["observations"] = [
                f"parameter changed from {event['before']!r} to {event['after']!r}"
            ]
        evidence.append(item)
    return evidence


def simulate_case(
    config: dict[str, Any],
    scenario: dict[str, Any],
    seed: int | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate one scenario and return truth, measurements and causal ledger."""

    validate_config(config)
    actual_seed = int(config["seed"] if seed is None else seed)
    effective, events, input_delta = _apply_scenario(config, scenario)
    grid = _channel_grid(effective, input_power_delta_db=input_delta)
    stages = [node for node in effective["topology"] if node["type"] == "edfa_stage"]
    requested = {
        stage["id"]: float(stage["target_gain_db"]) for stage in stages
    }
    control = effective.get("control", {})
    max_iterations = int(control.get("max_iterations", 24))
    tolerance_db = float(control.get("tolerance_db", 0.005))
    damping = float(control.get("damping", 0.75))

    signal_mw = grid.input_signal_mw
    ase_mw = np.zeros_like(signal_mw)
    traces: list[dict[str, Any]] = []
    commands_mw: dict[str, float] = {}
    controller_iterations = 0
    for controller_iterations in range(1, max_iterations + 1):
        commands_mw = {
            stage["id"]: _pump_command_mw(stage, requested[stage["id"]])
            for stage in stages
        }
        signal_mw, ase_mw, traces = _propagate(
            effective, grid, commands_mw, requested
        )
        stage_traces = {
            trace["node_id"]: trace
            for trace in traces
            if trace["type"] == "edfa_stage"
        }
        errors = {
            stage["id"]: float(stage["target_gain_db"])
            - float(stage_traces[stage["id"]]["mean_signal_gain_db"])
            for stage in stages
        }
        if max(abs(error) for error in errors.values()) <= tolerance_db:
            break
        for stage in stages:
            stage_id = stage["id"]
            requested[stage_id] = min(
                requested[stage_id] + damping * errors[stage_id],
                float(stage["gain_max_db"]) * (1.0 - 1e-7),
            )

    rng = np.random.default_rng(actual_seed)
    tap_measurements = _measure_taps(effective, traces, rng)
    osa_measurement = _measure_osa(effective, signal_mw, ase_mw, grid, rng)
    summary = _summary(grid, signal_mw, ase_mw, traces, tap_measurements)
    result: dict[str, Any] = {
        "schema_version": 1,
        "case": {
            "id": scenario["id"],
            "description": scenario.get("description", ""),
            "seed": actual_seed,
        },
        "config_metadata": copy.deepcopy(config["metadata"]),
        "truth": {
            "events": events,
            "controller_iterations": controller_iterations,
            "pump_commands_mw": commands_mw,
            "requested_small_signal_gains_db": requested,
            "input_power_dbm": [
                float(value) if bool(active) else None
                for value, active in zip(grid.input_power_dbm, grid.active)
            ],
            "output_signal_dbm": [
                float(value) if bool(active) else None
                for value, active in zip(mw_to_dbm(signal_mw), grid.active)
            ],
            "output_ase_dbm": [
                float(value) if bool(active) else None
                for value, active in zip(mw_to_dbm(ase_mw), grid.active)
            ],
            "propagation": traces,
        },
        "measurements": {
            "taps": tap_measurements,
            "output_osa": osa_measurement,
        },
        "summary": summary,
    }
    if baseline is None:
        result["comparison_to_baseline"] = {
            "output_signal_total_delta_db": 0.0,
            "total_gain_delta_db": 0.0,
            "flatness_delta_db": 0.0,
            "tilt_delta_db": 0.0,
            "stage_gain_deltas_db": {
                stage_id: 0.0 for stage_id in summary["stage_gains_db"]
            },
            "tap_true_deltas_db": {
                tap_id: 0.0 for tap_id in summary["tap_true_dbm"]
            },
        }
        result["diagnostic_evidence"] = []
    else:
        result["comparison_to_baseline"] = _comparison(result, baseline)
        result["diagnostic_evidence"] = _diagnostic_evidence(result, baseline)
    return result


def simulate_suite(
    config: dict[str, Any],
    seed: int | None = None,
    scenario_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Simulate a baseline and a selected set of scenarios."""

    validate_config(config)
    actual_seed = int(config["seed"] if seed is None else seed)
    selected = set(scenario_ids) if scenario_ids is not None else None
    scenarios = [
        scenario
        for scenario in config["scenarios"]
        if selected is None or scenario["id"] in selected
    ]
    if not scenarios:
        raise ValueError("no scenarios selected")

    baseline_id = config.get("baseline_case", scenarios[0]["id"])
    baseline_scenario = next(
        (
            scenario
            for scenario in config["scenarios"]
            if scenario["id"] == baseline_id
        ),
        None,
    )
    if baseline_scenario is None:
        raise ValueError(f"baseline scenario '{baseline_id}' not found")

    baseline = simulate_case(config, baseline_scenario, seed=actual_seed)
    cases: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario["id"] == baseline_id:
            cases.append(baseline)
        else:
            cases.append(
                simulate_case(
                    config,
                    scenario,
                    seed=actual_seed,
                    baseline=baseline,
                )
            )
    return {
        "schema_version": 1,
        "baseline_case": baseline_id,
        "seed": actual_seed,
        "config": {
            "name": config["name"],
            "metadata": copy.deepcopy(config["metadata"]),
        },
        "cases": cases,
    }
