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
    /// the lower boundary.
    pub upper_boundary: bool,
    /// Response time in seconds: simulated decision time plus
    /// `non_decision_time`.
    pub rt: f64,
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
///   and the test attributes the response to the nearer boundary.
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

    Trial {
        upper_boundary: x >= 0.5 * boundary,
        rt: t + params.non_decision_time,
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
}
