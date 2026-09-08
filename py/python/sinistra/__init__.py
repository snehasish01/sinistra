"""Python bindings for sinistra-core, a drift-diffusion model simulator.

All parameter/result values use one representation: a plain ``dict`` with keys
``drift_rate``, ``boundary_separation``, ``starting_point``,
``non_decision_time`` and ``noise_sd``. ``fit_sim`` results additionally carry
``iterations`` and ``final_cost``.
"""

from ._sinistra import (
    SinistraError,
    __version__,
    fit_ez,
    fit_sim,
    simulate,
)

__all__ = [
    "simulate",
    "fit_ez",
    "fit_sim",
    "SinistraError",
    "__version__",
]
