"""Implicit analytic EDF model of Giles & Desurvire (1991), Eqs. (21)-(25).

Valid without ASE and excess loss, constant overlap, co-propagating beams.
"""

from __future__ import annotations

import numpy as np

from .solver import C_LIGHT, H_PLANCK, FiberSpec, PumpSpec


def solve_implicit_flux(flux_in, alpha, gain_star, zeta, length_m, tolerance=1e-13, max_iterations=200):
    """Solve Eq. (25) for the total output photon flux with safeguarded Newton-Raphson."""
    flux_in = np.asarray(flux_in, dtype=float)
    coeff = (alpha + gain_star) / zeta
    total_in = float(flux_in.sum())
    attenuation = alpha * length_m

    def residual(total_out):
        exponent = np.clip(coeff * (total_in - total_out) - attenuation, -700.0, 700.0)
        terms = flux_in * np.exp(exponent)
        return terms.sum() - total_out, -(coeff * terms).sum() - 1.0

    low, high = 0.0, total_in
    current = high
    for iteration in range(1, max_iterations + 1):
        value, slope = residual(current)
        if value > 0:
            low = max(low, current)
        else:
            high = min(high, current)
        candidate = current - value / slope
        if not (low < candidate < high):
            candidate = 0.5 * (low + high)
        if abs(candidate - current) <= tolerance * max(total_in, 1e-300):
            current = candidate
            break
        current = candidate
    return current, iteration


def solve_stage_implicit(wavelengths_nm, signal_in_mw, pumps, fiber: FiberSpec, spectroscopy):
    if any(pump.direction != "forward" for pump in pumps):
        raise ValueError("Eq. (25) as derived here covers co-propagating beams only")
    wavelengths = np.concatenate(
        [np.asarray(wavelengths_nm, dtype=float), [pump.wavelength_nm for pump in pumps]]
    )
    overlap = np.concatenate(
        [np.full(len(np.atleast_1d(wavelengths_nm)), fiber.overlap_signal), np.full(len(pumps), fiber.overlap_pump)]
    )
    power_in_w = np.concatenate(
        [np.asarray(signal_in_mw, dtype=float), [pump.power_mw for pump in pumps]]
    ) * 1e-3
    alpha = overlap * spectroscopy.sigma_a(wavelengths) * fiber.doping_m3
    gain_star = overlap * spectroscopy.sigma_e(wavelengths) * fiber.doping_m3
    photon_energy = H_PLANCK * C_LIGHT / (wavelengths * 1e-9)
    flux_in = power_in_w / photon_energy
    zeta = fiber.erbium_area_m2 * fiber.doping_m3 / fiber.lifetime_s

    total_out, iterations = solve_implicit_flux(flux_in, alpha, gain_star, zeta, fiber.length_m)
    exponent = (alpha + gain_star) / zeta * (flux_in.sum() - total_out) - alpha * fiber.length_m
    flux_out = flux_in * np.exp(exponent)
    power_out_mw = flux_out * photon_energy * 1e3
    count = len(np.atleast_1d(wavelengths_nm))
    return {
        "signal_out_mw": power_out_mw[:count],
        "pump_out_mw": power_out_mw[count:],
        "gain_db": 10.0 * np.log10(flux_out[:count] / flux_in[:count]),
        "iterations": iterations,
    }
