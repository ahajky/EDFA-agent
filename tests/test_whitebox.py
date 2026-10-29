from __future__ import annotations

import copy

import numpy as np
import pytest

from edfa_whitebox import load_chain_config, simulate_chain
from edfa_whitebox.chain import _spectroscopy
from edfa_whitebox.solver import FiberSpec, PumpSpec, solve_stage

LN10_OVER_10 = np.log(10.0) / 10.0


class PaperSpectroscopy:
    """Giles & Desurvire 1991, Fig. 8: alpha and g* given in dB/m at 1480/1550 nm."""

    def __init__(self, table):
        self.table = table

    def _lookup(self, wavelength_nm, index):
        wl = np.asarray(wavelength_nm, dtype=float)
        key = np.where(np.abs(wl - 1480.0) < 1.0, 1480.0, 1550.0)
        return np.where(key == 1480.0, self.table[1480][index], self.table[1550][index]) * LN10_OVER_10

    def sigma_a(self, wavelength_nm):
        return self._lookup(wavelength_nm, 0)

    def sigma_e(self, wavelength_nm):
        return self._lookup(wavelength_nm, 1)


FIG8 = PaperSpectroscopy({1480: (3.0, 1.1), 1550: (3.3, 4.8)})
LIFETIME = 0.01


def fig8_fiber(length_m: float, zeta: float) -> FiberSpec:
    return FiberSpec(
        length_m=length_m,
        doping_m3=1.0,
        erbium_area_m2=zeta * LIFETIME,
        lifetime_s=LIFETIME,
        overlap_signal=1.0,
        overlap_pump=1.0,
    )


@pytest.fixture()
def config() -> dict:
    return load_chain_config("configs/whitebox_three_stage_demo.json")


def _stage_inputs(config: dict):
    element = next(item for item in config["elements"] if item["type"] == "edf")
    wavelengths = np.linspace(1530.0, 1560.0, 8)
    return wavelengths, FiberSpec(**element["fiber"]), _spectroscopy(config["spectroscopy"])


def test_unpumped_fiber_only_absorbs(config: dict) -> None:
    wavelengths, fiber, spectroscopy = _stage_inputs(config)
    result = solve_stage(wavelengths, np.full(8, 1e-4), [], fiber, spectroscopy, steps=60)
    expected_db = -10 * np.log10(np.e) * fiber.length_m * (
        fiber.overlap_signal * spectroscopy.sigma_a(wavelengths) * fiber.doping_m3
    ) - fiber.background_loss_db_per_m * fiber.length_m
    assert result.gain_db == pytest.approx(expected_db, rel=1e-3)


def test_full_inversion_limit_bounds_gain(config: dict) -> None:
    wavelengths, fiber, spectroscopy = _stage_inputs(config)
    result = solve_stage(
        wavelengths,
        np.full(8, 1e-6),
        [PumpSpec(power_mw=1e4)],
        fiber,
        spectroscopy,
        steps=60,
    )
    upper = 10 * np.log10(np.e) * fiber.length_m * (
        fiber.overlap_signal * spectroscopy.sigma_e(wavelengths) * fiber.doping_m3
    )
    assert np.all(result.gain_db < upper + 1e-6)
    assert result.gain_db[0] > 0.5 * upper[0]


def test_high_pump_limit_matches_paper_eq31() -> None:
    sigma = FIG8.table
    alpha_p, gstar_p = sigma[1480]
    alpha_s, gstar_s = sigma[1550]
    nsp_paper = 1.0 / (1.0 - gstar_p * alpha_s / (alpha_p * gstar_s))
    inversion_paper = alpha_p / (alpha_p + gstar_p)

    result = solve_stage(
        [1550.0],
        [1e-3],
        [PumpSpec(power_mw=3000.0, wavelength_nm=1480.0)],
        fig8_fiber(5.0, 1.5e15),
        FIG8,
        ase_grid_nm=[1550.0],
        ase_bandwidth_hz=[125e9],
        steps=400,
    )
    gain = 10 ** (result.gain_db[0] / 10)
    noise_factor = 10 ** (result.nf_db[0] / 10)
    nsp_solver = (noise_factor * gain - 1.0) / (2.0 * (gain - 1.0))
    assert nsp_solver == pytest.approx(nsp_paper, rel=5e-3)
    assert result.n2.mean() == pytest.approx(inversion_paper, rel=5e-3)


def test_fig8_small_signal_gain_trend() -> None:
    def gain(length, pump_mw, zeta=1.5e15):
        return solve_stage(
            [1550.0],
            [1e-4],
            [PumpSpec(power_mw=pump_mw, wavelength_nm=1480.0)],
            fig8_fiber(length, zeta),
            FIG8,
            steps=300,
        ).gain_db[0]

    assert gain(20.0, 10.0) > gain(10.0, 10.0) > gain(5.0, 10.0)
    assert gain(20.0, 2.0) < gain(20.0, 6.0) < gain(20.0, 10.0)
    assert gain(20.0, 2.0, zeta=3e15) < 0.0 < gain(5.0, 2.0, zeta=3e15)
    assert 30.0 < gain(20.0, 10.0) < 50.0


def test_gain_compresses_with_input_power() -> None:
    def gain(input_mw):
        return solve_stage(
            [1550.0],
            [input_mw],
            [PumpSpec(power_mw=6.0, wavelength_nm=1480.0)],
            fig8_fiber(15.0, 3e15),
            FIG8,
            steps=300,
        ).gain_db[0]

    gains = [gain(value) for value in (1e-5, 1e-3, 1e-2, 1e-1)]
    assert gains == sorted(gains, reverse=True)
    assert gains[0] - gains[-1] > 3.0


def test_ase_grid_is_resolved_and_physical(config: dict) -> None:
    result = simulate_chain(config)
    assert result["converged"]
    grid = result["spectra"]["ase_grid_nm"]
    assert grid[0] == pytest.approx(1460.0)
    assert grid[-1] == pytest.approx(1570.0)
    stage1 = result["spectra"]["stages"]["stage1"]
    nf = np.array(stage1["nf_db"])
    gain = np.array(stage1["gain_db"])
    assert np.all(nf[gain > 15.0] > 2.95)
    assert 3.0 < result["summary"]["nf_db_mean"] < 10.0
    assert len(stage1["ase_forward_out_dbm"]) == len(grid)


def test_more_pump_gives_more_gain_and_less_pump_leftover(config: dict) -> None:
    low = copy.deepcopy(config)
    high = copy.deepcopy(config)
    for case, power in ((low, 50.0), (high, 100.0)):
        next(e for e in case["elements"] if e["id"] == "stage1")["pumps"][0]["power_mw"] = power
    out_low = simulate_chain(low)["nodes"][1]
    out_high = simulate_chain(high)["nodes"][1]
    assert out_high["gain_db_mean"] > out_low["gain_db_mean"]
    assert out_high["pump_out_mw"][0] > out_low["pump_out_mw"][0]


@pytest.mark.parametrize("length_m,pump_mw,zeta", [(5.0, 6.0, 1.5e15), (15.0, 6.0, 1.5e15), (20.0, 10.0, 3e15)])
def test_numerical_solver_matches_implicit_eq25_without_ase(length_m, pump_mw, zeta) -> None:
    from edfa_whitebox.analytic import solve_stage_implicit

    pumps = [PumpSpec(power_mw=pump_mw, wavelength_nm=1480.0)]
    fiber = fig8_fiber(length_m, zeta)
    numerical = solve_stage([1550.0], [1e-2], pumps, fiber, FIG8, steps=400, tolerance=1e-10)
    implicit = solve_stage_implicit([1550.0], [1e-2], pumps, fiber, FIG8)
    assert numerical.gain_db[0] == pytest.approx(implicit["gain_db"][0], abs=1e-4)
    assert numerical.pump_out_mw[0] == pytest.approx(implicit["pump_out_mw"][0], rel=1e-5)


def test_numerical_solver_matches_implicit_eq25_multichannel(config: dict) -> None:
    from edfa_whitebox.analytic import solve_stage_implicit

    wavelengths, fiber, spectroscopy = _stage_inputs(config)
    fiber = FiberSpec(**{**fiber.__dict__, "background_loss_db_per_m": 0.0})
    element = next(item for item in config["elements"] if item["id"] == "stage1")
    pumps = [PumpSpec(**item) for item in element["pumps"]]
    signal = np.full(8, 4e-3)
    numerical = solve_stage(wavelengths, signal, pumps, fiber, spectroscopy, steps=400, tolerance=1e-10)
    implicit = solve_stage_implicit(wavelengths, signal, pumps, fiber, spectroscopy)
    assert numerical.gain_db == pytest.approx(implicit["gain_db"], abs=1e-3)


def test_channel_plan_forms_and_validation() -> None:
    from edfa_whitebox.chain import channel_plan

    explicit, power = channel_plan(
        {"wavelengths_nm": [1550.12, 1530.5, 1560.0], "power_dbm": [-20, -22, -24]}
    )
    assert explicit.tolist() == [1530.5, 1550.12, 1560.0]
    assert power.tolist() == [-22.0, -20.0, -24.0]

    itu, _ = channel_plan(
        {"start_thz": 193.1, "spacing_ghz": 100.0, "count": 8, "power_dbm": -24.0}
    )
    assert itu.size == 8
    assert itu[0] == pytest.approx(1546.917, abs=1e-3)
    assert itu[-1] == pytest.approx(1552.524, abs=1e-3)
    assert np.all(np.diff(itu) > 0)

    even, _ = channel_plan(
        {"start_nm": 1530.0, "stop_nm": 1560.0, "count": 4, "power_dbm": -20.0}
    )
    assert even.tolist() == [1530.0, 1540.0, 1550.0, 1560.0]

    with pytest.raises(ValueError):
        channel_plan({"wavelengths_nm": [1550, 1550], "power_dbm": -20})
    with pytest.raises(ValueError):
        channel_plan({"wavelengths_nm": [1550, 1551], "power_dbm": [-20]})


def test_default_demo_uses_eight_channels(config: dict) -> None:
    result = simulate_chain(config)
    assert len(result["spectra"]["wavelengths_nm"]) == 8
    assert result["converged"]


def test_passive_loss_spectrum_acts_on_signal_and_ase() -> None:
    base = load_chain_config("configs/whitebox_three_stage_demo.json")
    without = copy.deepcopy(base)
    for element in without["elements"]:
        element.pop("loss_spectrum_db", None)
    with_gff = simulate_chain(base)
    no_gff = simulate_chain(without)
    ase_with = np.array(with_gff["spectra"]["stages"]["stage2"]["ase_forward_out_dbm"])
    ase_without = np.array(no_gff["spectra"]["stages"]["stage2"]["ase_forward_out_dbm"])
    assert ase_with.max() < ase_without.max() - 2.0
    assert with_gff["summary"]["output_total_dbm"] > no_gff["summary"]["output_total_dbm"]


def test_passive_loss_spectrum_is_validated() -> None:
    base = load_chain_config("configs/whitebox_three_stage_demo.json")
    for element in base["elements"]:
        if "loss_spectrum_db" in element:
            element["loss_spectrum_db"] = {"wavelengths_nm": [1550, 1540], "loss_db": [0, 1]}
            break
    with pytest.raises(ValueError):
        simulate_chain(base)


def test_stage_nf_never_below_quantum_floor_per_channel(config: dict) -> None:
    result = simulate_chain(config)
    for stage in result["spectra"]["stages"].values():
        gain = 10 ** (np.array(stage["gain_db"]) / 10)
        floor = 10 * np.log10((2 * gain - 1) / gain)
        assert np.all(np.array(stage["nf_db"]) >= floor - 1e-6)
