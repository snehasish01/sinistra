//! # sinistra-core
//!
//! Core simulation logic for a **drift-diffusion model** (DDM), the standard
//! sequential-sampling model of two-alternative forced-choice decisions in
//! cognitive science.
//!
//! A single decision is modelled as a one-dimensional diffusion process
//! `x(t)` that starts between two absorbing boundaries and drifts until it
//! hits one of them:
//!
//! ```text
//!   x = boundary_separation   ── "upper" response
//!   x = start * boundary_sep  ── initial value
//!   x = 0                     ── "lower" response
//! ```
//!
//! The process is integrated with the **Euler–Maruyama** scheme at a fixed
//! step `dt` = [`DT`]:
//!
//! ```text
//!   dx = drift_rate * dt + noise_sd * sqrt(dt) * N(0, 1)
//! ```
//!
//! The reported response time is the elapsed *simulated* time plus a fixed
//! `non_decision_time` offset (encoding + motor latency).
//!
//! ## The simulation-time cap
//!
//! Pathological parameters (e.g. zero drift with tiny noise, or a boundary
//! far from the start) can make an individual trial take arbitrarily long to
//! terminate. To keep every call bounded, a trial is force-terminated once
//! its *simulated* time reaches [`MAX_SIM_TIME`] (10 s), **before** the
//! non-decision offset is added. On a forced termination the response is
//! attributed to whichever boundary the process is nearer to (see
//! [`simulate_trial`]). With well-behaved parameters this cap is never
//! reached; if you see response times clustered exactly at
//! `MAX_SIM_TIME + non_decision_time`, your parameters are degenerate.

use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
use argmin::solver::neldermead::NelderMead;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Fixed Euler–Maruyama integration step, in seconds.
pub const DT: f64 = 0.001;

/// Hard cap on *simulated* decision time, in seconds.
///
/// A trial that has not hit a boundary after this much simulated time is
/// force-terminated. The cap guards against non-terminating loops on
/// degenerate parameters. See the crate-level docs for details.
pub const MAX_SIM_TIME: f64 = 10.0;

/// Parameters of a drift-diffusion model.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Mean rate of evidence accumulation (units of `x` per second).
    /// Positive values pull the process toward the upper boundary.
    pub drift_rate: f64,
    /// Distance between the two absorbing boundaries (`> 0`).
    pub boundary_separation: f64,
    /// Starting point as a fraction of `boundary_separation`, in `[0, 1]`.
    /// `0.5` is unbiased; larger values start nearer the upper boundary.
    pub starting_point: f64,
    /// Non-decision time (encoding + motor latency), in seconds, added to
    /// every response time.
    pub non_decision_time: f64,
    /// Standard deviation of the within-trial noise. Conventionally fixed at
    /// `1.0`, which is the value produced by [`Params::new`] and
    /// [`Params::default`].
    pub noise_sd: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            drift_rate: 0.0,
            boundary_separation: 1.0,
            starting_point: 0.5,
            non_decision_time: 0.0,
            noise_sd: 1.0,
        }
    }
}

impl Params {
    /// Construct parameters with the conventional `noise_sd = 1.0`.
    pub fn new(
        drift_rate: f64,
        boundary_separation: f64,
        starting_point: f64,
        non_decision_time: f64,
    ) -> Self {
        Self {
            drift_rate,
            boundary_separation,
            starting_point,
            non_decision_time,
            noise_sd: 1.0,
        }
    }

    /// Builder-style override of [`Params::noise_sd`].
    #[must_use]
    pub fn with_noise_sd(mut self, noise_sd: f64) -> Self {
        self.noise_sd = noise_sd;
        self
    }
}

/// The outcome of a single simulated decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trial {
    /// `true` if the process terminated at the upper boundary, `false` for
    /// the lower boundary. On a [`MAX_SIM_TIME`] timeout this holds the
    /// nearer boundary (see [`simulate_trial`]).
    pub upper_boundary: bool,
    /// Response time in seconds: simulated decision time plus
    /// `non_decision_time`.
    pub rt: f64,
    /// `true` if the trial was force-terminated at the [`MAX_SIM_TIME`]
    /// cutoff rather than by a genuine boundary crossing. Such trials have
    /// an unreliable `rt` and `upper_boundary` and should be excluded from
    /// downstream estimation (e.g. [`ez_diffusion`] drops them).
    pub timed_out: bool,
}

/// Simulate one trial of the diffusion process with Euler–Maruyama steps.
///
/// The process starts at `starting_point * boundary_separation` and is
/// stepped with `dt = `[`DT`] until it crosses `0` or `boundary_separation`,
/// or until simulated time reaches [`MAX_SIM_TIME`].
///
/// Boundary attribution uses the midpoint test `x >= boundary_separation / 2`:
///
/// * On normal termination `x` has already crossed a boundary, so the test
///   selects the boundary that was actually crossed.
/// * On a [`MAX_SIM_TIME`] timeout `x` is still strictly inside the interval,
///   and the test attributes the response to the nearer boundary. Such a
///   trial is flagged with [`Trial::timed_out`]` == true`.
///
/// `non_decision_time` is added to the elapsed simulated time to form
/// [`Trial::rt`].
pub fn simulate_trial(params: &Params, rng: &mut impl Rng) -> Trial {
    let sqrt_dt = DT.sqrt();
    let boundary = params.boundary_separation;
    let mut x = params.starting_point * boundary;
    let mut t = 0.0_f64;

    while x > 0.0 && x < boundary && t < MAX_SIM_TIME {
        let z: f64 = rng.sample(StandardNormal);
        x += params.drift_rate * DT + params.noise_sd * sqrt_dt * z;
        t += DT;
    }

    // If the process is still strictly inside the interval, the loop can only
    // have exited on the time cap.
    let timed_out = x > 0.0 && x < boundary;

    Trial {
        upper_boundary: x >= 0.5 * boundary,
        rt: t + params.non_decision_time,
        timed_out,
    }
}

/// Simulate `n` independent trials in parallel with [`rayon`].
///
/// # Determinism
///
/// The result is a deterministic function of `(params, n, seed)` and does
/// **not** depend on the number of worker threads or on scheduling order.
///
/// Each trial `i` gets its own [`ChaCha8Rng`] seeded from `seed` and then
/// switched to cipher **stream** `i` via `set_stream`. ChaCha's streams are
/// disjoint, non-overlapping keystreams of length `2^64` blocks each, so
/// every trial draws from an independent, reproducible random sequence that
/// is fixed by `(seed, i)` alone. Results are collected through rayon's
/// order-preserving indexed collect, so `trials[i]` always corresponds to
/// trial index `i`.
pub fn simulate_n(params: &Params, n: usize, seed: u64) -> Vec<Trial> {
    (0..n as u64)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            rng.set_stream(i);
            simulate_trial(params, &mut rng)
        })
        .collect()
}

/// The three summary statistics both estimators ([`ez_diffusion`],
/// [`fit_simulation`]) reduce a data set to. Computed by [`summary_stats`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SummaryStats {
    /// `Pc`: proportion of usable trials that hit the **upper** boundary
    /// (treated as the "correct" response).
    pub accuracy: f64,
    /// `MRT`: mean RT, in seconds, of the upper-boundary ("correct") trials.
    pub mean_rt: f64,
    /// `VRT`: sample variance (`/(m-1)`) of the upper-boundary trials' RT.
    pub var_rt: f64,
    /// Number of trials that were not timed out (the denominator of `accuracy`).
    pub n_usable: usize,
    /// Number of upper-boundary trials (the sample behind `mean_rt`/`var_rt`).
    pub n_correct: usize,
}

/// Reduce choice-RT data to the `(Pc, MRT, VRT)` triple used by every
/// estimator in this crate. This is the single source of truth for those
/// statistics.
///
/// * Timed-out trials ([`Trial::timed_out`]) are excluded entirely.
/// * `accuracy` is over all remaining ("usable") trials.
/// * `mean_rt` and `var_rt` are over the **upper-boundary** trials only, per
///   the EZ-diffusion convention (Wagenmakers et al., 2007). For the unbiased
///   model the correct- and error-RT distributions coincide, so this is not a
///   modelling choice so much as a convention; it does make the RT moments
///   noisier when the upper boundary is the minority response.
///
/// Returns `None` when the statistics are undefined: no usable trials, or
/// fewer than two upper-boundary trials (so the sample variance has no
/// denominator).
pub fn summary_stats(trials: &[Trial]) -> Option<SummaryStats> {
    let n_usable = trials.iter().filter(|t| !t.timed_out).count();
    if n_usable == 0 {
        return None;
    }

    let correct_rts: Vec<f64> = trials
        .iter()
        .filter(|t| !t.timed_out && t.upper_boundary)
        .map(|t| t.rt)
        .collect();
    let n_correct = correct_rts.len();
    if n_correct < 2 {
        return None;
    }

    let m = n_correct as f64;
    let mean_rt = correct_rts.iter().sum::<f64>() / m;
    let var_rt = correct_rts
        .iter()
        .map(|rt| (rt - mean_rt).powi(2))
        .sum::<f64>()
        / (m - 1.0);

    Some(SummaryStats {
        accuracy: n_correct as f64 / n_usable as f64,
        mean_rt,
        var_rt,
        n_usable,
        n_correct,
    })
}

/// Scaling parameter (within-trial noise SD) assumed by [`ez_diffusion`] and
/// [`fit_simulation`].
///
/// The EZ equations are invariant to the choice of `s` as long as it matches
/// the data-generating process. This crate's simulator uses `noise_sd = 1.0`
/// by default ([`Params::new`], [`Params::default`]), so the estimators assume
/// the same. Wagenmakers et al. (2007) instead fix `s = 0.1`; recovered
/// `drift_rate` and `boundary_separation` scale linearly with `s`.
pub const EZ_SCALING_S: f64 = 1.0;

/// Reasons [`ez_diffusion`] cannot return an estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EzError {
    /// Fewer than two non-timed-out trials were supplied.
    NotEnoughTrials(usize),
    /// Fewer than two upper-boundary ("correct") trials, so mean and variance
    /// of correct RT are undefined.
    NotEnoughCorrect(usize),
    /// Accuracy is exactly 0.5: `logit(Pc) = 0`, drift is not identifiable.
    ChancePerformance,
    /// The variance of correct RT is not positive.
    NonPositiveVariance,
    /// The EZ drift equation has no real solution for these statistics.
    NoRealSolution,
}

impl std::fmt::Display for EzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EzError::NotEnoughTrials(n) => {
                write!(f, "need >= 2 non-timed-out trials, got {n}")
            }
            EzError::NotEnoughCorrect(n) => {
                write!(
                    f,
                    "need >= 2 upper-boundary trials for RT statistics, got {n}"
                )
            }
            EzError::ChancePerformance => {
                write!(f, "accuracy is exactly 0.5; drift rate is not identifiable")
            }
            EzError::NonPositiveVariance => {
                write!(f, "variance of correct RT is not positive")
            }
            EzError::NoRealSolution => {
                write!(
                    f,
                    "EZ drift equation has no real solution for these statistics"
                )
            }
        }
    }
}

impl std::error::Error for EzError {}

/// Recover DDM parameters from choice-RT data with the **EZ-diffusion** method
/// of Wagenmakers, van der Maas & Grasman (2007), *"An EZ-diffusion model for
/// response time and accuracy"*, Psychonomic Bulletin & Review, 14(1), 3–22.
///
/// EZ inverts the closed-form Wiener first-passage equations for the
/// *unbiased* diffusion model (start point fixed at `a/2`, no across-trial
/// variability in drift, start, or non-decision time). It maps three summary
/// statistics of the data onto three parameters:
///
/// | statistic | symbol | source |
/// |-----------|--------|--------|
/// | accuracy  | `Pc`   | proportion of trials hitting the **upper** boundary |
/// | mean RT   | `MRT`  | mean RT of upper-boundary trials |
/// | RT variance | `VRT` | sample variance (`/(m-1)`) of upper-boundary RT |
///
/// The upper boundary is treated as the "correct" response. For the unbiased
/// model the correct and error RT distributions are identical, so restricting
/// the RT moments to upper-boundary trials (as the paper specifies) matches
/// the theory; it does, however, make the estimate noisier when the upper
/// boundary is the minority response (drift rate far below zero).
///
/// # Formulas
///
/// With `s` = [`EZ_SCALING_S`], `s2 = s²`, and `L = logit(Pc) = ln(Pc/(1-Pc))`:
///
/// ```text
///   v   = sign(Pc - 1/2) · s · ( L · (L·Pc² - L·Pc + Pc - 1/2) / VRT )^(1/4)   (Eq. 5)
///   a   = s2 · L / v                                                           (Eq. 4)
///   y   = -v·a / s2
///   MDT = (a / 2v) · (1 - e^y) / (1 + e^y)                                      (Eq. 9)
///   Ter = MRT - MDT                                                            (Eq. 8)
/// ```
///
/// # Edge correction
///
/// `Pc` of exactly 0 or 1 has no EZ solution. Following the paper (p. 9,
/// "Edge corrections"), it is nudged to `1/(2N)` or `1 - 1/(2N)` respectively,
/// where `N` is the number of non-timed-out trials. `Pc == 0.5` exactly is
/// returned as [`EzError::ChancePerformance`] rather than corrected, since the
/// drift sign is then undefined.
///
/// # Returns
///
/// A [`Params`] with the recovered `drift_rate`, `boundary_separation` and
/// `non_decision_time`; `starting_point` is set to `0.5` (the EZ assumption)
/// and `noise_sd` to [`EZ_SCALING_S`]. Timed-out trials are excluded from all
/// statistics.
pub fn ez_diffusion(trials: &[Trial]) -> Result<Params, EzError> {
    let n_usable = trials.iter().filter(|t| !t.timed_out).count();
    if n_usable < 2 {
        return Err(EzError::NotEnoughTrials(n_usable));
    }

    let stats = summary_stats(trials).ok_or_else(|| {
        let n_correct = trials
            .iter()
            .filter(|t| !t.timed_out && t.upper_boundary)
            .count();
        EzError::NotEnoughCorrect(n_correct)
    })?;

    let mut pc = stats.accuracy;

    // Edge correction (Wagenmakers et al., 2007, p. 9). `Pc == 0` cannot occur
    // here: it would mean zero correct RTs, which `summary_stats` already
    // rejects as `NotEnoughCorrect`.
    if pc == 1.0 {
        pc = 1.0 - 1.0 / (2.0 * n_usable as f64);
    } else if pc == 0.5 {
        return Err(EzError::ChancePerformance);
    }

    let mrt = stats.mean_rt;
    let vrt = stats.var_rt;
    if vrt.is_nan() || vrt <= 0.0 {
        return Err(EzError::NonPositiveVariance);
    }

    let s = EZ_SCALING_S;
    let s2 = s * s;
    let l = (pc / (1.0 - pc)).ln();

    // Eq. 5.
    let bracket = l * (l * pc * pc - l * pc + pc - 0.5) / vrt;
    if bracket.is_nan() || bracket < 0.0 {
        return Err(EzError::NoRealSolution);
    }
    let v = (pc - 0.5).signum() * s * bracket.powf(0.25);

    // Eq. 4.
    let a = s2 * l / v;

    // Eqs. 8-9.
    let y = -v * a / s2;
    let mdt = (a / (2.0 * v)) * ((1.0 - y.exp()) / (1.0 + y.exp()));
    let ter = mrt - mdt;

    Ok(Params {
        drift_rate: v,
        boundary_separation: a,
        starting_point: 0.5,
        non_decision_time: ter,
        noise_sd: s,
    })
}

// ---------------------------------------------------------------------------
// Simulation-based fitter (Nelder–Mead over summary statistics)
// ---------------------------------------------------------------------------

/// Number of trials simulated per objective-function evaluation in
/// [`fit_simulation`].
///
/// This is the main speed/accuracy knob. Each Nelder–Mead step evaluates the
/// objective once or twice, and a fit takes ~150–300 evaluations, so the cost
/// of a fit is roughly `250 * FIT_N_PER_EVAL` simulated trials.
///
/// * **Noise.** The objective compares simulated `(Pc, MRT, VRT)` against the
///   data's. The per-evaluation Monte-Carlo error on those falls as
///   `1/sqrt(n)`: at `n = 5000`, `SE(Pc) ≈ 0.006`, `SE(MRT)/MRT ≈ 0.5%`,
///   `SE(VRT)/VRT ≈ 2–3%`. Because every evaluation reuses the same RNG seed
///   (common random numbers, see [`fit_simulation`]), the objective is a
///   *deterministic* surface and the noise that matters is its ruggedness, not
///   the raw SE — 5000 is enough to keep the surface smooth enough for
///   Nelder–Mead near the optimum.
/// * **Speed.** 5000 trials is ~1 ms (release, parallelised), keeping a whole
///   fit well under a second. `n = 1000` makes the surface too rugged and the
///   simplex stalls; `n = 50_000` gives ~3x less noise for 10x the time and
///   barely moves the estimate once the search is warm-started near the truth.
///
/// 5000 is the knee of that trade-off and the recommended default.
pub const FIT_N_PER_EVAL: usize = 5_000;

/// The RNG seed used for *every* objective evaluation within one
/// [`fit_simulation`] call (common random numbers): holding it fixed makes the
/// objective a deterministic surface, which Nelder–Mead requires — it has no
/// defense against a stochastic objective.
///
/// The seed is *derived from the target statistics* rather than being a global
/// constant. Each evaluation's 5000-trial simulation deviates from its
/// expectation by one particular noise draw, and the fitted parameters absorb
/// whatever offset compensates for it. A global constant would make that
/// offset point the *same way* for every data set (a spurious directional
/// bias); keying the seed to the data instead scatters it, so across data sets
/// the error is mean-zero. The fit stays fully deterministic given
/// `(trials, initial_guess, max_iters)`.
fn fit_eval_seed(stats: &SummaryStats) -> u64 {
    // splitmix64-style bit mixer over the data summary.
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
    for x in [
        stats.accuracy.to_bits(),
        stats.mean_rt.to_bits(),
        stats.var_rt.to_bits(),
        stats.n_usable as u64,
        stats.n_correct as u64,
    ] {
        h = (h ^ x).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
    }
    h.wrapping_mul(0xC4CE_B9FE_1A85_EC53)
}

/// Relative weights on the squared relative errors of `(Pc, MRT, VRT)` in the
/// [`fit_simulation`] objective. Using *relative* errors already removes the
/// raw-scale differences between the three; `VRT` is additionally halved
/// because the sample variance carries roughly twice the relative sampling
/// error of the sample mean, so equal weighting would let `VRT` noise steer
/// the search.
const FIT_WEIGHTS: [f64; 3] = [1.0, 1.0, 0.5];

/// Reasons [`fit_simulation`] cannot return an estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// [`summary_stats`] of the target data is undefined (no usable trials, or
    /// fewer than two upper-boundary trials).
    NoData,
    /// The optimizer ran but produced no best parameter vector.
    NoSolution,
    /// The optimizer returned an error.
    Solver(String),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::NoData => write!(
                f,
                "target data has no usable trials or fewer than two upper-boundary trials"
            ),
            FitError::NoSolution => write!(f, "optimizer produced no solution"),
            FitError::Solver(msg) => write!(f, "optimizer error: {msg}"),
        }
    }
}

impl std::error::Error for FitError {}

/// Sum-of-squared-relative-differences objective between a candidate parameter
/// vector `[drift_rate, boundary_separation, non_decision_time]` and the target
/// summary statistics. `starting_point` is fixed at `0.5` and `noise_sd` at
/// [`EZ_SCALING_S`], matching the model both estimators assume.
#[derive(Clone)]
struct SimFitCost {
    target: SummaryStats,
    n_per_eval: usize,
    seed: u64,
}

impl CostFunction for SimFitCost {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        let (v, a, t0) = (p[0], p[1], p[2]);

        // Nelder–Mead searches unconstrained R^3; fence off the invalid region
        // with a large finite penalty rather than an error.
        if !v.is_finite() || !a.is_finite() || !t0.is_finite() || a <= 1e-6 || t0 < 0.0 {
            return Ok(1e9);
        }

        let params = Params {
            drift_rate: v,
            boundary_separation: a,
            starting_point: 0.5,
            non_decision_time: t0,
            noise_sd: EZ_SCALING_S,
        };
        let trials = simulate_n(&params, self.n_per_eval, self.seed);

        let Some(cand) = summary_stats(&trials) else {
            return Ok(1e9);
        };

        let sq_rel = |c: f64, t: f64| {
            let d = (c - t) / t;
            d * d
        };
        let cost = FIT_WEIGHTS[0] * sq_rel(cand.accuracy, self.target.accuracy)
            + FIT_WEIGHTS[1] * sq_rel(cand.mean_rt, self.target.mean_rt)
            + FIT_WEIGHTS[2] * sq_rel(cand.var_rt, self.target.var_rt);

        Ok(cost)
    }
}

/// The result of a [`fit_simulation`] run: the recovered parameters plus a
/// little diagnostic detail about the search.
#[derive(Debug, Clone, PartialEq)]
pub struct SimFit {
    /// Recovered parameters (`starting_point = 0.5`, `noise_sd` =
    /// [`EZ_SCALING_S`]).
    pub params: Params,
    /// Number of Nelder–Mead iterations performed.
    pub iterations: u64,
    /// Objective value at `params` (weighted sum of squared relative errors
    /// in `Pc`, `MRT`, `VRT`).
    pub final_cost: f64,
}

/// A reasonable starting point for [`fit_simulation`]: the closed-form
/// [`ez_diffusion`] estimate when it succeeds, otherwise a neutral default.
///
/// This is a deliberate design choice — EZ is essentially free (a handful of
/// `ln`/`exp` calls) and lands close enough to the truth that the expensive
/// simulation search only has to polish it, cutting the iteration count
/// several-fold versus a cold start.
pub fn default_fit_guess(trials: &[Trial]) -> Params {
    ez_diffusion(trials).unwrap_or_else(|_| Params::new(1.0, 1.0, 0.5, 0.2))
}

/// Recover DDM parameters by **simulation-based fitting**: minimise, with
/// Nelder–Mead, the discrepancy between the data's `(Pc, MRT, VRT)` and those
/// same statistics computed from a fresh [`simulate_n`] run at the candidate
/// parameters.
///
/// Unlike [`ez_diffusion`] this makes no closed-form approximation — the
/// forward model *is* the simulator — so it has no analogue of EZ's
/// discretisation-driven bias (the `dt` overshoot affects the candidate and
/// the data identically and cancels). Its error is instead optimizer
/// convergence plus the Monte-Carlo noise of each evaluation (see
/// [`FIT_N_PER_EVAL`]).
///
/// Three parameters are fitted: `drift_rate`, `boundary_separation`,
/// `non_decision_time`. `starting_point` is held at `0.5` and `noise_sd` at
/// [`EZ_SCALING_S`].
///
/// # Arguments
///
/// * `initial_guess` — where the simplex is centred. Pass
///   [`default_fit_guess`] (the EZ estimate) unless you have something better;
///   warm-starting from the cheap closed-form fit is the intended use.
/// * `max_iters` — total Nelder–Mead iteration budget. ~100–200 is plenty for
///   this 3-parameter problem. The budget is split across two rounds: an
///   initial search, then one simplex restart around the best point found (a
///   standard guard against the simplex collapsing prematurely onto a ridge of
///   the slightly rugged Monte-Carlo objective).
///
/// # Determinism
///
/// Every objective evaluation uses one RNG seed derived from the data (see
/// [`fit_eval_seed`]), so the whole fit is deterministic given
/// `(trials, initial_guess, max_iters)`. The estimate is conditioned on that
/// single noise realisation of the evaluation simulator; averaging over seeds
/// would cut its variance at proportional cost.
pub fn fit_simulation(
    trials: &[Trial],
    initial_guess: Params,
    max_iters: usize,
) -> Result<SimFit, FitError> {
    let target = summary_stats(trials).ok_or(FitError::NoData)?;

    let cost = SimFitCost {
        target,
        n_per_eval: FIT_N_PER_EVAL,
        seed: fit_eval_seed(&target),
    };

    let g = [
        initial_guess.drift_rate,
        initial_guess.boundary_separation,
        initial_guess.non_decision_time,
    ];

    // Round 1: ~55% of the budget from a simplex around the guess.
    let round1_iters = (max_iters * 55 / 100).max(1) as u64;
    let (mut best, mut best_cost, mut iters) =
        run_neldermead(cost.clone(), simplex_around(&g, 0.15, 0.05), round1_iters)?;

    // Round 2: the remainder, restarting from round 1's best with a tighter
    // simplex.
    let round2_iters = (max_iters as u64).saturating_sub(iters).max(1);
    let (b2, c2, i2) = run_neldermead(cost, simplex_around(&best, 0.05, 0.02), round2_iters)?;
    iters += i2;
    if c2 < best_cost {
        best = b2;
        best_cost = c2;
    }

    Ok(SimFit {
        params: Params {
            drift_rate: best[0],
            boundary_separation: best[1],
            starting_point: 0.5,
            non_decision_time: best[2],
            noise_sd: EZ_SCALING_S,
        },
        iterations: iters,
        final_cost: best_cost,
    })
}

/// A 4-vertex simplex around `center`: the point itself, plus one vertex per
/// axis stepped by `frac` of that coordinate's magnitude (floored at `floor`
/// so a near-zero coordinate still spreads).
fn simplex_around(center: &[f64; 3], frac: f64, floor: f64) -> Vec<Vec<f64>> {
    let step = |v: f64| (v.abs() * frac).max(floor);
    vec![
        center.to_vec(),
        vec![center[0] + step(center[0]), center[1], center[2]],
        vec![center[0], center[1] + step(center[1]), center[2]],
        vec![center[0], center[1], center[2] + floor.max(0.05)],
    ]
}

/// Run Nelder–Mead once and return `(best_param, best_cost, iterations)`.
fn run_neldermead(
    cost: SimFitCost,
    simplex: Vec<Vec<f64>>,
    max_iters: u64,
) -> Result<([f64; 3], f64, u64), FitError> {
    let solver = NelderMead::new(simplex)
        .with_sd_tolerance(1e-10)
        .map_err(|e| FitError::Solver(e.to_string()))?;

    let result = Executor::new(cost, solver)
        .configure(|state| state.max_iters(max_iters))
        .run()
        .map_err(|e| FitError::Solver(e.to_string()))?;

    let state = result.state();
    let best = state.get_best_param().ok_or(FitError::NoSolution)?;
    Ok((
        [best[0], best[1], best[2]],
        state.get_best_cost(),
        state.get_iter(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prop_upper(trials: &[Trial]) -> f64 {
        trials.iter().filter(|t| t.upper_boundary).count() as f64 / trials.len() as f64
    }

    fn mean_rt(trials: &[Trial]) -> f64 {
        trials.iter().map(|t| t.rt).sum::<f64>() / trials.len() as f64
    }

    /// Closed-form probability that a Wiener process with drift `v` and
    /// infinitesimal SD `s`, started at `z_frac * a`, is absorbed at the
    /// upper boundary `a` rather than at `0`.
    ///
    /// `P = (e^{-2 v z / s^2} - 1) / (e^{-2 v a / s^2} - 1)`, with the
    /// `v -> 0` limit `z / a`. This is the standard DDM choice probability
    /// and gives the tests something real to check the sampler against.
    fn analytic_p_upper(v: f64, a: f64, z_frac: f64, s: f64) -> f64 {
        let z = z_frac * a;
        if v.abs() < 1e-12 {
            return z / a;
        }
        let k = -2.0 * v / (s * s);
        (k * z).exp_m1() / (k * a).exp_m1()
    }

    /// Zero drift + centered start is a symmetric random walk: the sampler
    /// should match the analytic 50/50 split over a large sample.
    #[test]
    fn zero_drift_centered_start_is_balanced() {
        let params = Params::new(0.0, 1.0, 0.5, 0.0);
        let trials = simulate_n(&params, 40_000, 0xC0FFEE);
        let p = prop_upper(&trials);
        assert!((p - 0.5).abs() < 0.02, "expected ~0.5 split, got {p}");
    }

    /// The empirical choice probability should track the closed-form DDM
    /// absorption probability across a range of drifts and start points.
    #[test]
    fn choice_probability_matches_closed_form() {
        for &(v, a, z) in &[
            (1.0, 1.0, 0.5),
            (2.5, 1.0, 0.5),
            (-2.0, 1.0, 0.5),
            (0.0, 1.0, 0.8),
            (0.8, 1.4, 0.3),
        ] {
            let params = Params::new(v, a, z, 0.0);
            let p = prop_upper(&simulate_n(&params, 60_000, 17));
            let expected = analytic_p_upper(v, a, z, params.noise_sd);
            assert!(
                (p - expected).abs() < 0.03,
                "v={v} a={a} z={z}: empirical {p:.3} vs analytic {expected:.3}"
            );
        }
    }

    /// Stronger drift toward the upper boundary should (a) send more trials
    /// there and (b) shorten the mean response time.
    #[test]
    fn stronger_drift_favors_boundary_and_speeds_decisions() {
        let weak = Params::new(0.5, 1.0, 0.5, 0.0);
        let strong = Params::new(2.5, 1.0, 0.5, 0.0);

        let weak_trials = simulate_n(&weak, 40_000, 7);
        let strong_trials = simulate_n(&strong, 40_000, 7);

        let (p_weak, p_strong) = (prop_upper(&weak_trials), prop_upper(&strong_trials));
        let (rt_weak, rt_strong) = (mean_rt(&weak_trials), mean_rt(&strong_trials));

        assert!(
            p_strong > p_weak,
            "stronger drift should favor the boundary more: {p_weak} -> {p_strong}"
        );
        assert!(
            rt_strong < rt_weak,
            "stronger drift should be faster: {rt_weak} -> {rt_strong}"
        );
    }

    /// Drift sign controls *which* boundary is favored.
    #[test]
    fn negative_drift_favors_lower_boundary() {
        let params = Params::new(-2.0, 1.0, 0.5, 0.0);
        let trials = simulate_n(&params, 40_000, 42);
        assert!(
            prop_upper(&trials) < 0.2,
            "negative drift should favor the lower boundary, got {}",
            prop_upper(&trials)
        );
    }

    /// Same seed -> bit-identical output; different seed -> different output.
    #[test]
    fn seeding_is_reproducible_and_seed_sensitive() {
        let params = Params::new(0.8, 1.2, 0.4, 0.2);
        let a = simulate_n(&params, 5_000, 2024);
        let b = simulate_n(&params, 5_000, 2024);
        let c = simulate_n(&params, 5_000, 2025);

        assert_eq!(a, b, "identical seed must reproduce identical trials");
        assert_ne!(a, c, "different seed should produce different trials");
    }

    /// Determinism must not depend on the rayon pool size.
    #[test]
    fn determinism_is_independent_of_thread_count() {
        let params = Params::new(1.0, 1.0, 0.5, 0.0);
        let reference = simulate_n(&params, 8_000, 99);

        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| simulate_n(&params, 8_000, 99));

        assert_eq!(reference, one_thread);
    }

    /// `non_decision_time` is a pure additive shift on `rt` and leaves the
    /// diffusion path (hence the choice) untouched.
    #[test]
    fn non_decision_time_is_an_additive_shift() {
        let base = Params::new(1.0, 1.0, 0.5, 0.0);
        let shifted = Params::new(1.0, 1.0, 0.5, 0.3);

        let a = simulate_n(&base, 3_000, 1);
        let b = simulate_n(&shifted, 3_000, 1);

        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.upper_boundary, y.upper_boundary);
            assert!((y.rt - x.rt - 0.3).abs() < 1e-9);
        }
    }

    /// A start biased toward the upper boundary favors upper responses even
    /// with zero drift.
    #[test]
    fn biased_start_favors_near_boundary() {
        let params = Params::new(0.0, 1.0, 0.8, 0.0);
        let trials = simulate_n(&params, 40_000, 5);
        assert!(
            prop_upper(&trials) > 0.7,
            "start near the upper boundary should favor upper responses"
        );
    }

    /// End-to-end parameter recovery: simulate with known params, then check
    /// that `ez_diffusion` inverts back to them.
    ///
    /// ## Tolerance
    ///
    /// EZ and this simulator share the same generative assumptions (unbiased
    /// start, no across-trial variability in drift/start/Ter), so EZ's
    /// well-documented bias at extreme parameters -- driven by across-trial
    /// variability it cannot see -- does *not* apply here. The residual error
    /// has two sources we can bound:
    ///
    /// 1. **Euler-Maruyama discretization.** With `dt = 1e-3` and `s = 1`, the
    ///    process overshoots a boundary by `O(s*sqrt(dt)) ~= 0.03` per
    ///    crossing, i.e. ~3% of `a = 1`. This inflates observed RT and, through
    ///    Eqs. 4-5, the recovered `a` and `v` by a few percent, and shifts
    ///    `Ter` by up to ~1-2 discretization steps plus the RT inflation.
    /// 2. **Monte-Carlo error.** With `n = 200_000` the SE of `Pc` is
    ///    `< 0.0012` and of `MRT` is well under 1 ms, contributing < 1%.
    ///
    /// So we allow **8% relative** error on `drift_rate` and
    /// `boundary_separation` and **20 ms absolute** on `non_decision_time`.
    /// These are set from the discretization-bias budget above, not tuned to
    /// pass: a bug that (say) doubled `a` or dropped a factor of `s` would blow
    /// straight through them. Drift is kept positive so the upper boundary is
    /// the majority ("correct") response, matching the EZ coding convention.
    #[test]
    fn ez_diffusion_recovers_known_parameters() {
        // (drift, boundary, start, t0)
        let truths = [
            Params::new(1.0, 1.0, 0.5, 0.20),
            Params::new(2.0, 1.2, 0.5, 0.30),
            Params::new(0.5, 1.5, 0.5, 0.15),
            Params::new(3.0, 0.8, 0.5, 0.25),
        ];

        for (k, truth) in truths.iter().enumerate() {
            let trials = simulate_n(truth, 200_000, 1000 + k as u64);
            let est = ez_diffusion(&trials).expect("recovery should succeed");

            let rel = |got: f64, want: f64| (got - want).abs() / want;

            assert!(
                rel(est.drift_rate, truth.drift_rate) < 0.08,
                "case {k}: drift {:.4} vs true {:.4} ({:.1}%)",
                est.drift_rate,
                truth.drift_rate,
                100.0 * rel(est.drift_rate, truth.drift_rate),
            );
            assert!(
                rel(est.boundary_separation, truth.boundary_separation) < 0.08,
                "case {k}: boundary {:.4} vs true {:.4} ({:.1}%)",
                est.boundary_separation,
                truth.boundary_separation,
                100.0 * rel(est.boundary_separation, truth.boundary_separation),
            );
            assert!(
                (est.non_decision_time - truth.non_decision_time).abs() < 0.020,
                "case {k}: Ter {:.4} vs true {:.4}",
                est.non_decision_time,
                truth.non_decision_time,
            );
        }
    }

    /// Perfect accuracy triggers the `1 - 1/(2N)` edge correction rather than
    /// a division-by-zero or NaN. Built by hand so `Pc` is exactly 1.
    #[test]
    fn ez_diffusion_edge_corrects_perfect_accuracy() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let trials: Vec<Trial> = (0..5_000)
            .map(|_| Trial {
                upper_boundary: true,
                // spread RT so VRT > 0
                rt: 0.4 + 0.05 * rng.sample::<f64, _>(StandardNormal),
                timed_out: false,
            })
            .collect();
        assert!(trials.iter().all(|t| t.upper_boundary));

        let est = ez_diffusion(&trials).expect("edge correction should apply");
        assert!(est.drift_rate.is_finite() && est.drift_rate > 0.0);
        assert!(est.boundary_separation.is_finite() && est.boundary_separation > 0.0);
        assert!(est.non_decision_time.is_finite());
    }

    /// Timed-out trials must not enter the summary statistics.
    #[test]
    fn ez_diffusion_excludes_timed_out_trials() {
        let mut trials = simulate_n(&Params::new(1.0, 1.0, 0.5, 0.2), 5_000, 9);
        let clean = ez_diffusion(&trials).unwrap();

        // Injecting garbage timed-out trials should not change the estimate.
        for i in 0..2_000 {
            trials.push(Trial {
                upper_boundary: i % 2 == 0,
                rt: 10.0 + i as f64,
                timed_out: true,
            });
        }
        let with_junk = ez_diffusion(&trials).unwrap();

        assert!((clean.drift_rate - with_junk.drift_rate).abs() < 1e-12);
        assert!((clean.boundary_separation - with_junk.boundary_separation).abs() < 1e-12);
        assert!((clean.non_decision_time - with_junk.non_decision_time).abs() < 1e-12);
    }

    #[test]
    fn summary_stats_are_correct_and_exclude_timeouts() {
        let trials = vec![
            Trial {
                upper_boundary: true,
                rt: 0.5,
                timed_out: false,
            },
            Trial {
                upper_boundary: true,
                rt: 0.7,
                timed_out: false,
            },
            Trial {
                upper_boundary: true,
                rt: 0.9,
                timed_out: false,
            },
            Trial {
                upper_boundary: false,
                rt: 0.6,
                timed_out: false,
            },
            // excluded entirely:
            Trial {
                upper_boundary: true,
                rt: 10.1,
                timed_out: true,
            },
            Trial {
                upper_boundary: false,
                rt: 10.1,
                timed_out: true,
            },
        ];
        let s = summary_stats(&trials).unwrap();
        assert_eq!(s.n_usable, 4);
        assert_eq!(s.n_correct, 3);
        assert!((s.accuracy - 0.75).abs() < 1e-12);
        assert!((s.mean_rt - 0.7).abs() < 1e-12);
        // sample variance of {0.5, 0.7, 0.9} = 0.04
        assert!((s.var_rt - 0.04).abs() < 1e-12);

        assert!(summary_stats(&[]).is_none());
    }

    #[test]
    fn ez_diffusion_and_summary_stats_agree() {
        // ez_diffusion must be driven by the same statistics summary_stats reports.
        let trials = simulate_n(&Params::new(1.3, 1.1, 0.5, 0.2), 20_000, 4);
        let s = summary_stats(&trials).unwrap();
        let est = ez_diffusion(&trials).unwrap();

        // Re-derive Ter from the reported (v, a) and MRT; it must match.
        let (v, a) = (est.drift_rate, est.boundary_separation);
        let y = -v * a / (EZ_SCALING_S * EZ_SCALING_S);
        let mdt = (a / (2.0 * v)) * ((1.0 - y.exp()) / (1.0 + y.exp()));
        assert!((est.non_decision_time - (s.mean_rt - mdt)).abs() < 1e-9);
    }

    #[test]
    fn fit_simulation_is_deterministic() {
        let trials = simulate_n(&Params::new(1.2, 1.0, 0.5, 0.25), 20_000, 11);
        let guess = default_fit_guess(&trials);
        let a = fit_simulation(&trials, guess.clone(), 80).unwrap();
        let b = fit_simulation(&trials, guess, 80).unwrap();
        assert_eq!(a, b);
    }

    /// End-to-end recovery for the simulation fitter, the analogue of
    /// `ez_diffusion_recovers_known_parameters`.
    ///
    /// Tolerance, reasoned from this method's error sources rather than EZ's.
    /// The forward model here *is* the simulator, so the `dt` discretization
    /// overshoot hits the candidate and the target identically and cancels --
    /// there is no one-directional bias like the +3-4% EZ inherited on `a`.
    /// The error is instead per-evaluation Monte-Carlo noise
    /// ([`FIT_N_PER_EVAL`] = 5000 trials, one data-derived seed).
    ///
    /// `SE(Pc) ~ sqrt(p(1-p)/5000)`, and the fit drives candidate `Pc` to the
    /// target within that. Since `dPc ~= p(1-p) * d(v*a/s^2)`, the relative
    /// error this induces on `v*a` is `SE(Pc) / (p(1-p) * logit(Pc))`, which is
    /// strongly drift-dependent: ~1.5% at `v=3, a=0.8` (`Pc ~= 0.92`), ~3% at
    /// `v=1, a=1` (`Pc ~= 0.73`), and ~4% at `v=0.5, a=1.5` (`Pc ~= 0.68`,
    /// nearest chance, where the `Pc -> v` map is steepest). `VRT` (shift-
    /// invariant, no `Pc` term) anchors `a`, so most of this noise lands on
    /// `v`. Its sign is mean-zero across data sets because the eval seed is
    /// keyed to the data. Target sampling at `n = 150_000` adds `SE(Pc) <
    /// 0.0013` (~4x smaller) and the optimizer resolves the minimum to <1%
    /// (final costs ~1e-5 or below) -- both subdominant.
    ///
    /// So expected `|error|` on `v` is ~1.5% (high drift) to ~4% 1-sigma (the
    /// low-drift corner here). Tolerances: `drift_rate` **8%** (~2-sigma at the
    /// worst case -- 5% would do for `v >= 1`, the extra room is specifically
    /// for `Pc -> v` amplification near chance, not slack); `boundary_
    /// separation` **5%** (anchored by `VRT`, observed errors ~1-3%);
    /// `non_decision_time` **18 ms** (`Ter = MRT - MDT(v,a)`: `SE(MRT) ~= 2-5
    /// ms` plus `v,a` error propagating through an `MDT` of up to ~0.5 s).
    /// This budget differs from EZ's in kind: EZ's was a one-directional
    /// discretization allowance uniform across parameters; this is
    /// two-directional MC noise, worst for `v` and worst still at low drift.
    #[test]
    fn fit_simulation_recovers_known_parameters() {
        let truths = [
            Params::new(1.0, 1.0, 0.5, 0.20),
            Params::new(2.0, 1.2, 0.5, 0.30),
            Params::new(0.5, 1.5, 0.5, 0.15),
            Params::new(3.0, 0.8, 0.5, 0.25),
        ];

        for (k, truth) in truths.iter().enumerate() {
            let trials = simulate_n(truth, 150_000, 2000 + k as u64);
            let guess = default_fit_guess(&trials);
            let fit = fit_simulation(&trials, guess, 150).expect("fit should succeed");
            let est = fit.params;

            let rel = |got: f64, want: f64| (got - want).abs() / want;

            assert!(
                rel(est.drift_rate, truth.drift_rate) < 0.08,
                "case {k}: drift {:.4} vs true {:.4} ({:.1}%), cost {:.2e}",
                est.drift_rate,
                truth.drift_rate,
                100.0 * rel(est.drift_rate, truth.drift_rate),
                fit.final_cost,
            );
            assert!(
                rel(est.boundary_separation, truth.boundary_separation) < 0.05,
                "case {k}: boundary {:.4} vs true {:.4} ({:.1}%), cost {:.2e}",
                est.boundary_separation,
                truth.boundary_separation,
                100.0 * rel(est.boundary_separation, truth.boundary_separation),
                fit.final_cost,
            );
            assert!(
                (est.non_decision_time - truth.non_decision_time).abs() < 0.018,
                "case {k}: Ter {:.4} vs true {:.4}, cost {:.2e}",
                est.non_decision_time,
                truth.non_decision_time,
                fit.final_cost,
            );
        }
    }
}
