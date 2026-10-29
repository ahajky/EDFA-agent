"""White-box (Giles-Desurvire rate-propagation) EDFA model."""

from .chain import load_chain_config, simulate_chain
from .solver import FiberSpec, PumpSpec, Spectroscopy, solve_stage

__all__ = [
    "FiberSpec",
    "PumpSpec",
    "Spectroscopy",
    "solve_stage",
    "load_chain_config",
    "simulate_chain",
]
