from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .solver import (
    C_LIGHT,
    H_PLANCK,
    FiberSpec,
    PumpSpec,
    Spectroscopy,
    dbm_to_mw,
    mw_to_dbm,
    ase_bin_bandwidth_hz,
    interp_psd,
    solve_stage,
)


def load_chain_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _spectroscopy(raw: dict) -> Spectroscopy:
    return Spectroscopy(
        absorption_gaussians=tuple(tuple(item) for item in raw["absorption_gaussians"]),
        mccumber_epsilon_nm=raw["mccumber_epsilon_nm"],
        temperature_k=raw.get("temperature_k", 300.0),
    )


def channel_plan(channels: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelengths_nm, input power in dBm per channel).

    Three forms, highest priority first:
    - `wavelengths_nm`: explicit list;
    - `start_thz` + `spacing_ghz` + `count`: frequency grid (ITU style);
    - `start_nm` + `stop_nm` + `count`: evenly spaced in wavelength.
    `power_dbm` is a scalar or a per-channel list in the order given.
    The result is sorted by increasing wavelength.
    """
    if "wavelengths_nm" in channels:
        wavelengths = np.asarray(channels["wavelengths_nm"], dtype=float)
    elif "start_thz" in channels:
        frequencies_hz = (
            channels["start_thz"] * 1e12
            + channels["spacing_ghz"] * 1e9 * np.arange(int(channels["count"]))
        )
        wavelengths = C_LIGHT / frequencies_hz * 1e9
    else:
        wavelengths = np.linspace(
            channels["start_nm"], channels["stop_nm"], int(channels["count"])
        )
    if wavelengths.size == 0:
        raise ValueError("channel plan is empty")
    power = np.asarray(channels["power_dbm"], dtype=float)
    if power.ndim == 0:
        power = np.full(wavelengths.size, float(power))
    if power.shape != wavelengths.shape:
        raise ValueError("power_dbm must be a scalar or match the channel count")
    order = np.argsort(wavelengths)
    wavelengths, power = wavelengths[order], power[order]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("channel wavelengths must be distinct")
    return wavelengths, power


def _ase_grid(channels: dict) -> np.ndarray:
    grid = channels.get("ase_grid_nm")
    if grid is None:
        return np.zeros(0)
    count = int(round((grid["stop_nm"] - grid["start_nm"]) / grid["step_nm"])) + 1
    return grid["start_nm"] + grid["step_nm"] * np.arange(count)


def simulate_chain(config: dict) -> dict:
    channels = config["channels"]
    wavelengths, power_dbm = channel_plan(channels)
    grid = _ase_grid(channels)
    bin_hz = ase_bin_bandwidth_hz(grid) if grid.size else np.zeros(0)
    signal = dbm_to_mw(power_dbm)
    input_mw = signal.copy()
    ase = np.zeros(grid.size)
    spectroscopy = _spectroscopy(config["spectroscopy"])
    photon_energy = H_PLANCK * C_LIGHT / (wavelengths * 1e-9)
    solver_opts = config.get("solver", {})

    nodes = []
    spectra = {
        "wavelengths_nm": wavelengths.tolist(),
        "ase_grid_nm": grid.tolist(),
        "stages": {},
    }
    all_converged = True
    for element in config["elements"]:
        kind = element["type"]
        if kind == "passive":
            base_db = float(element["loss_db"])
            loss_db = np.full(wavelengths.size, base_db)
            ase_loss_db = np.full(grid.size, base_db)
            spectrum = element.get("loss_spectrum_db")
            if spectrum:
                points_nm = np.asarray(spectrum["wavelengths_nm"], dtype=float)
                points_db = np.asarray(spectrum["loss_db"], dtype=float)
                if points_nm.shape != points_db.shape or np.any(np.diff(points_nm) <= 0):
                    raise ValueError(f"{element['id']}: invalid loss_spectrum_db")
                loss_db = loss_db + np.interp(wavelengths, points_nm, points_db)
                ase_loss_db = ase_loss_db + np.interp(grid, points_nm, points_db)
            if element.get("ideal_flatten"):
                level_db = mw_to_dbm(signal)
                extra_db = level_db - level_db.min()
                loss_db = loss_db + extra_db
                if grid.size:
                    ase_loss_db = ase_loss_db + np.interp(
                        grid, wavelengths, extra_db, left=0.0, right=0.0
                    )
            signal = signal * 10.0 ** (-loss_db / 10.0)
            if grid.size:
                ase = ase * 10.0 ** (-ase_loss_db / 10.0)
            nodes.append({"id": element["id"], "type": kind, "loss_db": element["loss_db"]})
            continue
        fiber = FiberSpec(**element["fiber"])
        pumps = [PumpSpec(**item) for item in element["pumps"]]
        result = solve_stage(
            wavelengths,
            signal,
            pumps,
            fiber,
            spectroscopy,
            ase_grid_nm=grid if grid.size else None,
            ase_in_mw=ase if grid.size else None,
            ase_bandwidth_hz=bin_hz if grid.size else None,
            **solver_opts,
        )
        all_converged &= result.converged
        signal = result.signal_out_mw
        ase = result.ase_forward_out_mw
        spectra["stages"][element["id"]] = {
            "gain_db": result.gain_db.tolist(),
            "nf_db": result.nf_db.tolist(),
            "signal_out_dbm": mw_to_dbm(result.signal_out_mw).tolist(),
            "ase_forward_out_dbm": mw_to_dbm(result.ase_forward_out_mw).tolist(),
            "ase_backward_in_dbm": mw_to_dbm(result.ase_backward_in_mw).tolist(),
        }
        nodes.append(
            {
                "id": element["id"],
                "type": kind,
                "gain_db_mean": float(result.gain_db.mean()),
                "gain_db_ripple": float(np.ptp(result.gain_db)),
                "nf_db_mean": float(result.nf_db.mean()),
                "output_signal_total_dbm": float(mw_to_dbm(result.signal_out_mw.sum())),
                "pump_out_mw": result.pump_out_mw,
                "mean_inversion": float(result.n2.mean()),
                "iterations": result.iterations,
                "residual": result.residual,
                "converged": result.converged,
            }
        )

    total_gain = signal / input_mw
    if grid.size:
        psd = interp_psd(ase, bin_hz, grid, wavelengths)
        nf_lin = psd / (total_gain * photon_energy) + 1.0 / total_gain
        nf_mean = float(np.mean(10 * np.log10(nf_lin)))
        ase_total = float(mw_to_dbm(ase.sum()))
    else:
        nf_mean = None
        ase_total = None
    return {
        "name": config.get("name", "whitebox"),
        "parameter_source": config.get("metadata", {}).get("parameter_source"),
        "vendor_confirmed": config.get("metadata", {}).get("vendor_confirmed", False),
        "converged": bool(all_converged),
        "summary": {
            "input_total_dbm": float(mw_to_dbm(input_mw.sum())),
            "output_total_dbm": float(mw_to_dbm(signal.sum())),
            "gain_db_mean": float(np.mean(10 * np.log10(total_gain))),
            "gain_flatness_db": float(np.ptp(10 * np.log10(total_gain))),
            "nf_db_mean": nf_mean,
            "output_ase_total_dbm": ase_total,
        },
        "nodes": nodes,
        "spectra": spectra,
    }
