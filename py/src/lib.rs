//! Python bindings for `sinistra-core`.
//!
//! Three functions are exposed — [`simulate`], [`fit_ez`], [`fit_sim`] — plus a
//! [`SinistraError`] exception.
//!
//! Trial data crosses the boundary as a **pair of NumPy arrays**, `choices`
//! (`bool`) and `rts` (`float64`), rather than a Python list of tuples: at
//! large `n` the per-trial object marshalling of a list dominates, whereas two
//! contiguous arrays are a single buffer copy each.
//!
//! Parameters and fit results use one consistent representation: a plain
//! `dict` with the keys `drift_rate`, `boundary_separation`, `starting_point`,
//! `non_decision_time`, `noise_sd` (fit results from [`fit_sim`] add
//! `iterations` and `final_cost`).

use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sinistra_core::{
    default_fit_guess, ez_diffusion, fit_simulation, simulate_n, EzError, FitError, Params, Trial,
};

create_exception!(
    sinistra,
    SinistraError,
    PyValueError,
    "Raised when parameter recovery fails on bad or degenerate input data."
);

// --- conversions ----------------------------------------------------------

/// Borrow a 1-D array as a slice, turning a non-contiguous array into a clear
/// Python error rather than a panic.
fn slice_of<'a, T: numpy::Element>(
    name: &str,
    arr: &'a PyReadonlyArray1<'_, T>,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|e| PyValueError::new_err(format!("`{name}` must be a contiguous 1-D array: {e}")))
}

/// Rebuild `Vec<Trial>` from the `(choices, rts)` array pair. Trials handed in
/// from Python are completed observations by construction — `simulate` already
/// drops the timed-out ones — so `timed_out` is always `false` here.
fn trials_from_arrays(choices: &[bool], rts: &[f64]) -> PyResult<Vec<Trial>> {
    if choices.len() != rts.len() {
        return Err(PyValueError::new_err(format!(
            "choices and rts must have equal length ({} vs {})",
            choices.len(),
            rts.len()
        )));
    }
    Ok(choices
        .iter()
        .zip(rts)
        .map(|(&upper_boundary, &rt)| Trial {
            upper_boundary,
            rt,
            timed_out: false,
        })
        .collect())
}

fn params_to_dict<'py>(py: Python<'py>, p: &Params) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("drift_rate", p.drift_rate)?;
    d.set_item("boundary_separation", p.boundary_separation)?;
    d.set_item("starting_point", p.starting_point)?;
    d.set_item("non_decision_time", p.non_decision_time)?;
    d.set_item("noise_sd", p.noise_sd)?;
    Ok(d)
}

/// Parse an `initial_guess` dict. Only `drift_rate`, `boundary_separation` and
/// `non_decision_time` are required (the three parameters the simulation fit
/// actually varies); `starting_point` and `noise_sd` default to `0.5` / `1.0`
/// so that a dict returned by [`fit_ez`] round-trips straight in.
fn params_from_map(m: &HashMap<String, f64>) -> PyResult<Params> {
    let required = |key: &str| -> PyResult<f64> {
        m.get(key)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("initial_guess is missing '{key}'")))
    };
    Ok(Params {
        drift_rate: required("drift_rate")?,
        boundary_separation: required("boundary_separation")?,
        starting_point: m.get("starting_point").copied().unwrap_or(0.5),
        non_decision_time: required("non_decision_time")?,
        noise_sd: m.get("noise_sd").copied().unwrap_or(1.0),
    })
}

fn ez_error(e: EzError) -> PyErr {
    let msg = match e {
        EzError::NotEnoughTrials(n) => {
            format!("not enough data: {n} non-timed-out trial(s), need at least 2")
        }
        EzError::NotEnoughCorrect(n) => format!(
            "not enough correct responses: {n} upper-boundary trial(s), \
             need at least 2 to form RT statistics"
        ),
        EzError::ChancePerformance => {
            "accuracy is exactly 0.5; drift rate is not identifiable from chance performance".into()
        }
        EzError::NonPositiveVariance => {
            "the RT variance of correct responses is not positive; cannot recover parameters".into()
        }
        EzError::NoRealSolution => {
            "the EZ-diffusion equations have no real solution for these summary statistics".into()
        }
    };
    SinistraError::new_err(msg)
}

fn fit_error(e: FitError) -> PyErr {
    let msg = match e {
        FitError::NoData => {
            "target data has no usable trials, or fewer than two upper-boundary trials".into()
        }
        FitError::NoSolution => "the optimizer did not produce a solution".into(),
        FitError::Solver(m) => format!("optimizer error: {m}"),
    };
    SinistraError::new_err(msg)
}

// --- exposed functions ---------------------------------------------------

/// `(choices: bool[], rts: float64[], n_excluded)` — the return of [`simulate`].
type SimulateOutput<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<f64>>, usize);

/// simulate(drift, boundary, start, t0, n, seed, noise_sd=1.0)
///
/// Simulate `n` drift-diffusion trials. Returns `(choices, rts, n_excluded)`:
///
/// * `choices` — `np.ndarray` of `bool`, `True` if the trial hit the upper
///   boundary;
/// * `rts` — `np.ndarray` of `float64`, response times in seconds;
/// * `n_excluded` — how many trials hit the 10 s simulation cap and were
///   dropped from both arrays (their choice/RT are unreliable).
///
/// The two arrays are built in one pass over the filtered trials — no
/// intermediate Python objects.
#[pyfunction]
#[pyo3(signature = (drift, boundary, start, t0, n, seed, noise_sd=1.0))]
#[allow(clippy::too_many_arguments)]
fn simulate<'py>(
    py: Python<'py>,
    drift: f64,
    boundary: f64,
    start: f64,
    t0: f64,
    n: usize,
    seed: u64,
    noise_sd: f64,
) -> PyResult<SimulateOutput<'py>> {
    if boundary <= 0.0 || boundary.is_nan() {
        return Err(PyValueError::new_err("boundary must be > 0"));
    }
    if !(0.0..=1.0).contains(&start) {
        return Err(PyValueError::new_err("start must be in [0, 1]"));
    }
    if t0 < 0.0 || t0.is_nan() {
        return Err(PyValueError::new_err("t0 must be >= 0"));
    }
    if noise_sd <= 0.0 || noise_sd.is_nan() {
        return Err(PyValueError::new_err("noise_sd must be > 0"));
    }

    let params = Params {
        drift_rate: drift,
        boundary_separation: boundary,
        starting_point: start,
        non_decision_time: t0,
        noise_sd,
    };

    // Simulate with the GIL released, then fold the filtered trials straight
    // into two typed buffers.
    let (choices, rts, excluded) = py.detach(|| {
        let all = simulate_n(&params, n, seed);
        let mut choices = Vec::with_capacity(all.len());
        let mut rts = Vec::with_capacity(all.len());
        for t in &all {
            if t.timed_out {
                continue;
            }
            choices.push(t.upper_boundary);
            rts.push(t.rt);
        }
        let excluded = all.len() - choices.len();
        (choices, rts, excluded)
    });

    Ok((choices.into_pyarray(py), rts.into_pyarray(py), excluded))
}

/// fit_ez(choices, rts) -> dict
///
/// Recover parameters from `(choices, rts)` arrays with closed-form
/// EZ-diffusion. Raises `SinistraError` (a `ValueError` subclass) with a
/// specific message when the data cannot be fitted.
#[pyfunction]
fn fit_ez<'py>(
    py: Python<'py>,
    choices: PyReadonlyArray1<'py, bool>,
    rts: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let trials = trials_from_arrays(slice_of("choices", &choices)?, slice_of("rts", &rts)?)?;
    match ez_diffusion(&trials) {
        Ok(params) => params_to_dict(py, &params),
        Err(e) => Err(ez_error(e)),
    }
}

/// fit_sim(choices, rts, initial_guess=None, max_iters=200) -> dict
///
/// Recover parameters by simulation-based Nelder-Mead fitting. `initial_guess`
/// is a params dict (e.g. the return value of `fit_ez`); when omitted the fit
/// warm-starts from the EZ estimate. The result dict adds `iterations` and
/// `final_cost` to the usual params keys. Raises `SinistraError` on failure.
#[pyfunction]
#[pyo3(signature = (choices, rts, initial_guess=None, max_iters=200))]
fn fit_sim<'py>(
    py: Python<'py>,
    choices: PyReadonlyArray1<'py, bool>,
    rts: PyReadonlyArray1<'py, f64>,
    initial_guess: Option<HashMap<String, f64>>,
    max_iters: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let trials = trials_from_arrays(slice_of("choices", &choices)?, slice_of("rts", &rts)?)?;

    let guess = match initial_guess {
        Some(map) => params_from_map(&map)?,
        None => default_fit_guess(&trials),
    };

    match py.detach(|| fit_simulation(&trials, guess, max_iters)) {
        Ok(fit) => {
            let d = params_to_dict(py, &fit.params)?;
            d.set_item("iterations", fit.iterations)?;
            d.set_item("final_cost", fit.final_cost)?;
            Ok(d)
        }
        Err(e) => Err(fit_error(e)),
    }
}

#[pymodule]
fn _sinistra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("SinistraError", m.py().get_type::<SinistraError>())?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ez, m)?)?;
    m.add_function(wrap_pyfunction!(fit_sim, m)?)?;
    Ok(())
}
