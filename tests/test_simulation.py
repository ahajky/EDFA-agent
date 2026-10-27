from __future__ import annotations

import copy

import pytest

from edfa_sim import load_config, simulate_suite


@pytest.fixture()
def config() -> dict:
    return load_config("configs/three_stage_demo.json")


def _cases(config: dict, scenario_ids: list[str]) -> dict[str, dict]:
    suite = simulate_suite(config, scenario_ids=scenario_ids)
    return {case["case"]["id"]: case for case in suite["cases"]}


def _stage(case: dict, node_id: str) -> dict:
    return next(
        trace
        for trace in case["truth"]["propagation"]
        if trace["node_id"] == node_id
    )


def test_simulation_is_reproducible(config: dict) -> None:
    selected = ["normal", "interstage_loss_3db", "out_tap_bias_2db"]
    first = _cases(config, selected)
    second = _cases(copy.deepcopy(config), selected)
    assert first == second


def test_normal_case_is_finite_and_marked_as_assumption(config: dict) -> None:
    case = _cases(config, ["normal"])["normal"]
    summary = case["summary"]
    assert case["config_metadata"]["parameter_source"] == "engineering_assumption"
    assert case["config_metadata"]["vendor_confirmed"] is False
    assert 15.0 < summary["output_signal_total_dbm"] < 28.0
    assert 20.0 < summary["total_signal_gain_db"] < 35.0
    assert summary["effective_nf_db_mean"] >= 3.0
    assert summary["output_channel_flatness_db"] >= 0.0


def test_interstage_loss_reduces_all_downstream_taps(config: dict) -> None:
    case = _cases(config, ["interstage_loss_3db"])["interstage_loss_3db"]
    comparison = case["comparison_to_baseline"]
    assert comparison["output_signal_total_delta_db"] == pytest.approx(
        -3.0, abs=0.001
    )
    assert comparison["tap_true_deltas_db"]["mid_tap1"] == pytest.approx(
        -3.0, abs=0.001
    )
    assert any(
        item["root_cause"] == "isolator1.add_loss_db"
        for item in case["diagnostic_evidence"]
    )


def test_agc_can_hide_pump_health_fault_in_command(config: dict) -> None:
    cases = _cases(
        config, ["normal", "stage2_pump_health_60"]
    )
    normal = cases["normal"]
    degraded = cases["stage2_pump_health_60"]
    normal_command = _stage(normal, "stage2")["pump_command_mw"]
    degraded_command = _stage(degraded, "stage2")["pump_command_mw"]
    assert degraded_command > normal_command * 1.4
    assert degraded["comparison_to_baseline"]["output_signal_total_delta_db"] == (
        pytest.approx(0.0, abs=0.001)
    )
    assert not _stage(degraded, "stage2")["pump_limited"]


def test_severe_pump_fault_hits_command_limit(config: dict) -> None:
    cases = _cases(
        config, ["normal", "stage1_pump_health_35"]
    )
    degraded = cases["stage1_pump_health_35"]
    stage = _stage(degraded, "stage1")
    assert stage["pump_limited"] is True
    assert stage["mean_signal_gain_db"] < 11.5
    assert (
        degraded["comparison_to_baseline"]["output_signal_total_delta_db"]
        < -1.0
    )


def test_sensor_bias_does_not_change_optical_truth(config: dict) -> None:
    cases = _cases(config, ["normal", "out_tap_bias_2db"])
    normal = cases["normal"]
    biased = cases["out_tap_bias_2db"]
    assert biased["summary"]["output_signal_total_dbm"] == pytest.approx(
        normal["summary"]["output_signal_total_dbm"], abs=1e-6
    )
    tap = biased["measurements"]["taps"]["out_tap"]
    assert tap["residual_db"] == pytest.approx(2.0, abs=0.1)
    assert tap["true_dbm"] == pytest.approx(
        normal["measurements"]["taps"]["out_tap"]["true_dbm"], abs=1e-6
    )


def test_gff_shift_changes_flatness_but_not_mean_power_much(config: dict) -> None:
    case = _cases(config, ["stage2_ripple_shift"])["stage2_ripple_shift"]
    comparison = case["comparison_to_baseline"]
    assert abs(comparison["flatness_delta_db"]) > 0.005
    assert comparison["output_signal_total_delta_db"] == pytest.approx(
        0.0, abs=0.01
    )
    assert case["diagnostic_evidence"][0]["root_cause"] == (
        "gff2.gain_ripple_shift_nm"
    )


def test_high_input_power_is_limited_by_saturation(config: dict) -> None:
    case = _cases(config, ["high_power_saturation"])["high_power_saturation"]
    stage3 = _stage(case, "stage3")
    assert stage3["saturation_compression_db"] > 0.0
    assert stage3["mean_signal_gain_db"] < 8.0
    assert case["comparison_to_baseline"]["output_signal_total_delta_db"] < 12.0
    assert case["truth"]["events"][0]["kind"] == "input_power_delta_db"
