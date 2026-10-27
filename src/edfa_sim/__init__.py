"""Explainable EDFA chain simulation."""

from .config import ConfigError, load_config, validate_config
from .model import simulate_case, simulate_suite

__all__ = [
    "ConfigError",
    "load_config",
    "validate_config",
    "simulate_case",
    "simulate_suite",
]
