from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
K_BOLTZMANN = 1.380649e-23


def dbm_to_mw(value_dbm):
    return 10.0 ** (np.asarray(value_dbm, dtype=float) / 10.0)


def mw_to_dbm(value_mw):
    return 10.0 * np.log10(np.maximum(np.asarray(value_mw, dtype=float), 1e-30))


@dataclass(frozen=True)
class Spectroscopy:
    """Er3+ cross sections. Emission follows McCumber unless given explicitly."""

    absorption_gaussians: tuple[tuple[float, float, float], ...]
    mccumber_epsilon_nm: float
    temperature_k: float = 300.0

    def sigma_a(self, wavelength_nm):
        wl = np.asarray(wavelength_nm, dtype=float)
        total = np.zeros_like(wl)
        for peak_m2, center_nm, fwhm_nm in self.absorption_gaussians:
            sigma = fwhm_nm / 2.3548200450309493
            total += peak_m2 * np.exp(-0.5 * ((wl - center_nm) / sigma) ** 2)
        return total

    def sigma_e(self, wavelength_nm):
        wl = np.asarray(wavelength_nm, dtype=float)
        delta = H_PLANCK * C_LIGHT * (1e9 / self.mccumber_epsilon_nm - 1e9 / wl)
        exponent = np.clip(-delta / (K_BOLTZMANN * self.temperature_k), -50.0, 50.0)
        emission = self.sigma_a(wl) * np.exp(exponent)
        return np.where(wl < 1450.0, 0.0, emission)


@dataclass(frozen=True)
class FiberSpec:
    length_m: float
    doping_m3: float
    erbium_area_m2: float
    lifetime_s: float
    overlap_signal: float
    overlap_pump: float
    background_loss_db_per_m: float = 0.0


@dataclass(frozen=True)
class PumpSpec:
    power_mw: float
    wavelength_nm: float = 980.0
    direction: str = "forward"


@dataclass
class StageResult:
    wavelengths_nm: np.ndarray
    signal_in_mw: np.ndarray
    signal_out_mw: np.ndarray
    gain_db: np.ndarray
    nf_db: np.ndarray
    ase_grid_nm: np.ndarray
    ase_bandwidth_hz: np.ndarray
    ase_forward_out_mw: np.ndarray
    ase_backward_in_mw: np.ndarray
    pump_out_mw: list[float]
    z_m: np.ndarray
    n2: np.ndarray
    iterations: int
    residual: float
    converged: bool


def _coefficients(wavelengths_nm, fiber, spectroscopy):
    alpha = fiber.overlap_signal * spectroscopy.sigma_a(wavelengths_nm) * fiber.doping_m3
    gain_star = fiber.overlap_signal * spectroscopy.sigma_e(wavelengths_nm) * fiber.doping_m3
    return alpha, gain_star


def ase_bin_bandwidth_hz(ase_grid_nm):
    grid = np.asarray(ase_grid_nm, dtype=float)
    step_nm = np.gradient(grid) if grid.size > 1 else np.array([1.0])
    return C_LIGHT * np.abs(step_nm) * 1e-9 / (grid * 1e-9) ** 2


def interp_psd(ase_mw, bandwidth_hz, ase_grid_nm, wavelengths_nm):
    psd = np.asarray(ase_mw, dtype=float) * 1e-3 / bandwidth_hz
    if psd.size == 1:
        return np.full(np.shape(wavelengths_nm), psd[0])
    return np.interp(wavelengths_nm, ase_grid_nm, psd)


def solve_stage(
    wavelengths_nm,
    signal_in_mw,
    pumps,
    fiber,
    spectroscopy,
    ase_grid_nm=None,
    ase_in_mw=None,
    ase_bandwidth_hz=None,
    steps=200,
    max_iterations=60,
    tolerance=1e-7,
):
    """Solve one EDF section (Giles & Desurvire 1991, Eqs. 19-20).

    Signals sit on `wavelengths_nm`. ASE is resolved on its own grid
    `ase_grid_nm` (forward and backward), each bin carrying
    `ase_bandwidth_hz`. Without a grid, ASE is ignored. Powers are in mW;
    `ase_in_mw` is the forward ASE entering the section.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    signal_in = np.asarray(signal_in_mw, dtype=float) * 1e-3
    if ase_grid_nm is None:
        grid = np.zeros(0)
        bandwidth = np.zeros(0)
    else:
        grid = np.atleast_1d(np.asarray(ase_grid_nm, dtype=float))
        if ase_bandwidth_hz is None:
            bandwidth = ase_bin_bandwidth_hz(grid)
        else:
            bandwidth = np.broadcast_to(
                np.asarray(ase_bandwidth_hz, dtype=float), grid.shape
            ).copy()
    ase_in = (
        np.zeros(grid.size)
        if ase_in_mw is None
        else np.asarray(ase_in_mw, dtype=float) * 1e-3
    )

    energy_s = H_PLANCK * C_LIGHT / (wavelengths * 1e-9)
    energy_a = H_PLANCK * C_LIGHT / (grid * 1e-9)
    alpha_s, gstar_s = _coefficients(wavelengths, fiber, spectroscopy)
    alpha_a, gstar_a = _coefficients(grid, fiber, spectroscopy)
    background = fiber.background_loss_db_per_m * np.log(10.0) / 10.0
    source_a = 2.0 * bandwidth * energy_a

    pump_wl = np.array([pump.wavelength_nm for pump in pumps], dtype=float)
    pump_energy = H_PLANCK * C_LIGHT / (pump_wl * 1e-9)
    pump_alpha = fiber.overlap_pump * spectroscopy.sigma_a(pump_wl) * fiber.doping_m3
    pump_gain = fiber.overlap_pump * spectroscopy.sigma_e(pump_wl) * fiber.doping_m3
    pump_is_forward = np.array([pump.direction == "forward" for pump in pumps], dtype=bool)
    pump_in = np.array([pump.power_mw for pump in pumps], dtype=float) * 1e-3

    saturation = fiber.erbium_area_m2 * fiber.doping_m3 / fiber.lifetime_s

    nodes = steps + 1
    dz = fiber.length_m / steps
    sig = np.zeros((nodes, wavelengths.size))
    asef = np.zeros((nodes, grid.size))
    aseb = np.zeros((nodes, grid.size))
    pf = np.zeros((nodes, len(pumps)))
    pb = np.zeros((nodes, len(pumps)))
    pf[:, pump_is_forward] = pump_in[pump_is_forward]
    pb[:, ~pump_is_forward] = pump_in[~pump_is_forward]

    def n2_of(signal, ase_total, pump_total):
        absorbed = np.sum(alpha_s * signal / energy_s, axis=-1)
        total = np.sum((alpha_s + gstar_s) * signal / energy_s, axis=-1)
        absorbed = absorbed + np.sum(alpha_a * ase_total / energy_a, axis=-1)
        total = total + np.sum((alpha_a + gstar_a) * ase_total / energy_a, axis=-1)
        absorbed = absorbed + np.sum(pump_alpha * pump_total / pump_energy, axis=-1)
        total = total + np.sum(
            (pump_alpha + pump_gain) * pump_total / pump_energy, axis=-1
        )
        return absorbed / (saturation + total)

    def derivatives(signal, ase_own, pump_own, ase_other, pump_other):
        n2_value = n2_of(signal, ase_own + ase_other, pump_own + pump_other)
        d_signal = ((alpha_s + gstar_s) * n2_value - alpha_s - background) * signal
        d_ase = ((alpha_a + gstar_a) * n2_value - alpha_a - background) * ase_own
        d_ase = d_ase + gstar_a * n2_value * source_a
        d_pump = ((pump_alpha + pump_gain) * n2_value - pump_alpha - background) * pump_own
        return d_signal, d_ase, d_pump

    def rk4(state, step, counter_start, counter_mid, counter_end, signal_fn):
        k1 = signal_fn(state, counter_start)
        k2 = signal_fn(tuple(x + 0.5 * step * k for x, k in zip(state, k1)), counter_mid)
        k3 = signal_fn(tuple(x + 0.5 * step * k for x, k in zip(state, k2)), counter_mid)
        k4 = signal_fn(tuple(x + step * k for x, k in zip(state, k3)), counter_end)
        return tuple(
            x + step / 6.0 * (a + 2 * b + 2 * c + d)
            for x, a, b, c, d in zip(state, k1, k2, k3, k4)
        )

    def sweep_forward():
        state = (signal_in.copy(), ase_in.copy(), pf[0].copy())
        sig[0], asef[0], pf[0] = state
        for node in range(steps):
            counter_mid = (0.5 * (aseb[node] + aseb[node + 1]), 0.5 * (pb[node] + pb[node + 1]))
            counter_start = (aseb[node], pb[node])
            counter_end = (aseb[node + 1], pb[node + 1])
            state = rk4(
                state,
                dz,
                counter_start,
                counter_mid,
                counter_end,
                lambda x, c: derivatives(x[0], x[1], x[2], c[0], c[1]),
            )
            sig[node + 1], asef[node + 1] = state[0], state[1]
            pf[node + 1, :] = np.where(pump_is_forward, state[2], 0.0)

    def sweep_backward():
        state = (np.zeros(grid.size), pb[-1].copy())
        aseb[-1], pb[-1] = state
        for node in range(steps, 0, -1):
            counter_start = (sig[node], asef[node], pf[node])
            counter_mid = tuple(0.5 * (a + b) for a, b in zip(counter_start, (sig[node - 1], asef[node - 1], pf[node - 1])))
            counter_end = (sig[node - 1], asef[node - 1], pf[node - 1])

            def fn(x, c):
                _, d_ase, d_pump = derivatives(c[0], x[0], x[1], c[1], c[2])
                return d_ase, d_pump

            state = rk4(state, dz, counter_start, counter_mid, counter_end, fn)
            aseb[node - 1] = state[0]
            pb[node - 1, :] = np.where(pump_is_forward, 0.0, state[1])

    residual = np.inf
    iterations = 0
    previous = None
    for iterations in range(1, max_iterations + 1):
        sweep_forward()
        sweep_backward()
        current = np.concatenate([sig[-1], asef[-1], aseb[0], pb[0]])
        if previous is not None:
            scale = np.maximum(np.abs(current), 1e-15)
            residual = float(np.max(np.abs(current - previous) / scale))
            if residual < tolerance:
                break
        previous = current
    converged = residual < tolerance

    n2 = n2_of(sig, asef + aseb, pf + pb)
    gain_lin = np.exp(
        np.trapezoid(
            (alpha_s + gstar_s) * n2[:, None] - alpha_s - background, dx=dz, axis=0
        )
    )
    if grid.size:
        gain_ase_lin = np.exp(
            np.trapezoid(
                (alpha_a + gstar_a) * n2[:, None] - alpha_a - background, dx=dz, axis=0
            )
        )
        added_ase_mw = np.maximum(asef[-1] - ase_in * gain_ase_lin, 0.0) * 1e3
        added_psd = interp_psd(added_ase_mw, bandwidth, grid, wavelengths)
        nf_lin = added_psd / (gain_lin * energy_s) + 1.0 / gain_lin
    else:
        nf_lin = np.full(wavelengths.size, np.nan)
    pump_out = np.where(pump_is_forward, pf[-1], pb[0]) * 1e3
    return StageResult(
        wavelengths_nm=wavelengths,
        signal_in_mw=signal_in * 1e3,
        signal_out_mw=sig[-1] * 1e3,
        gain_db=10.0 * np.log10(gain_lin),
        nf_db=10.0 * np.log10(np.maximum(nf_lin, 1e-30)),
        ase_grid_nm=grid,
        ase_bandwidth_hz=bandwidth,
        ase_forward_out_mw=asef[-1] * 1e3,
        ase_backward_in_mw=aseb[0] * 1e3,
        pump_out_mw=[float(value) for value in pump_out],
        z_m=np.linspace(0.0, fiber.length_m, nodes),
        n2=n2,
        iterations=iterations,
        residual=float(residual),
        converged=bool(converged),
    )
