"""Smoke test for the sinistra Python bindings.

This checks the bindings are wired up and round-trip
(``simulate`` -> ``fit_ez`` / ``fit_sim``). It is NOT a statistical
validation of parameter recovery: the 15% tolerance below is deliberately
loose. Rigorous recovery tests with first-principles tolerances live in the
Rust suite (``core/src/lib.rs``: ``ez_diffusion_recovers_known_parameters``,
``fit_simulation_recovers_known_parameters``).
"""

import numpy as np
import pytest

import sinistra

TRUE_DRIFT = 1.2
TRUE_BOUNDARY = 1.0
TRUE_START = 0.5
TRUE_T0 = 0.25

PARAM_KEYS = {
    "drift_rate",
    "boundary_separation",
    "starting_point",
    "non_decision_time",
    "noise_sd",
}


def make_data():
    choices, rts, n_excluded = sinistra.simulate(
        drift=TRUE_DRIFT,
        boundary=TRUE_BOUNDARY,
        start=TRUE_START,
        t0=TRUE_T0,
        n=50_000,
        seed=42,
    )
    assert n_excluded == 0  # these parameters never hit the sim cap
    assert isinstance(choices, np.ndarray) and choices.dtype == np.bool_
    assert isinstance(rts, np.ndarray) and rts.dtype == np.float64
    assert choices.shape == rts.shape == (50_000,)
    return choices, rts


def test_simulate_reports_and_excludes_timeouts():
    # Low drift + wide boundary: a real fraction of trials hit the 10 s cap.
    choices, rts, n_excluded = sinistra.simulate(
        drift=0.1, boundary=6.0, start=0.5, t0=0.1, n=3_000, seed=1
    )
    assert n_excluded > 0
    assert choices.shape[0] == rts.shape[0] == 3_000 - n_excluded


def test_fit_ez_round_trips():
    choices, rts = make_data()
    est = sinistra.fit_ez(choices, rts)
    assert PARAM_KEYS <= set(est)
    assert abs(est["drift_rate"] - TRUE_DRIFT) / TRUE_DRIFT < 0.15


def test_fit_sim_round_trips():
    choices, rts = make_data()
    est = sinistra.fit_sim(choices, rts, max_iters=200)
    assert PARAM_KEYS <= set(est)
    assert est["iterations"] > 0
    assert est["final_cost"] >= 0.0
    assert abs(est["drift_rate"] - TRUE_DRIFT) / TRUE_DRIFT < 0.15


def test_fit_sim_accepts_ez_dict_as_warm_start():
    choices, rts = make_data()
    guess = sinistra.fit_ez(choices, rts)
    est = sinistra.fit_sim(choices, rts, initial_guess=guess, max_iters=150)
    assert abs(est["drift_rate"] - TRUE_DRIFT) / TRUE_DRIFT < 0.15


def test_mismatched_array_lengths_raise():
    with pytest.raises(ValueError):
        sinistra.fit_ez(np.array([True, False]), np.array([0.5]))


def test_fit_ez_raises_clear_error_on_degenerate_data():
    # Identical choice and RT for every trial -> zero RT variance.
    choices = np.ones(200, dtype=np.bool_)
    rts = np.full(200, 0.5)
    with pytest.raises(sinistra.SinistraError):
        sinistra.fit_ez(choices, rts)


def test_fit_sim_raises_on_empty_data():
    empty_c = np.array([], dtype=np.bool_)
    empty_r = np.array([], dtype=np.float64)
    with pytest.raises(sinistra.SinistraError):
        sinistra.fit_sim(empty_c, empty_r)
