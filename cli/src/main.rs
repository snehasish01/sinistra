//! Thin CLI wrapper around [`sinistra_core`].

use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use sinistra_core::{
    default_fit_guess, ez_diffusion, fit_simulation, simulate_n, Params, Trial, FIT_N_PER_EVAL,
};

#[derive(Parser)]
#[command(name = "sinistra", version, about = "Drift-diffusion model simulator")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Simulate N trials and write them to a CSV file (columns: choice, rt).
    Simulate(SimulateArgs),
    /// Recover DDM parameters from a (choice, rt) CSV via EZ-diffusion.
    FitEz(FitEzArgs),
    /// Recover DDM parameters by simulation-based Nelder-Mead fitting.
    FitSim(FitSimArgs),
}

#[derive(Args)]
struct SimulateArgs {
    /// Drift rate (evidence units per second).
    #[arg(long)]
    drift: f64,

    /// Boundary separation (must be > 0).
    #[arg(long)]
    boundary: f64,

    /// Starting point as a fraction of boundary separation, in [0, 1].
    #[arg(long)]
    start: f64,

    /// Non-decision time in seconds (must be >= 0).
    #[arg(long)]
    t0: f64,

    /// Number of trials to simulate.
    #[arg(long)]
    n: usize,

    /// RNG seed (same seed => identical output).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Output CSV path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct FitEzArgs {
    /// Input CSV with a header and columns: choice (upper|lower), rt (seconds).
    #[arg(long)]
    input: PathBuf,
}

#[derive(Args)]
struct FitSimArgs {
    /// Input CSV with a header and columns: choice (upper|lower), rt (seconds).
    #[arg(long)]
    input: PathBuf,

    /// Nelder-Mead iteration budget (~100-200 is plenty for 3 parameters).
    #[arg(long, default_value_t = 150)]
    max_iters: usize,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Simulate(args) => run_simulate(args),
        Command::FitEz(args) => run_fit_ez(args),
        Command::FitSim(args) => run_fit_sim(args),
    }
}

fn run_simulate(args: SimulateArgs) -> Result<()> {
    ensure!(args.boundary > 0.0, "--boundary must be > 0");
    ensure!(
        (0.0..=1.0).contains(&args.start),
        "--start must be in [0, 1]"
    );
    ensure!(args.t0 >= 0.0, "--t0 must be >= 0");

    let params = Params::new(args.drift, args.boundary, args.start, args.t0);
    let total = args.n;
    let (completed, excluded) = filter_completed(simulate_n(&params, total, args.seed));

    // Timed-out trials have an unreliable rt/choice, so they never reach the
    // CSV. Report the count on stderr to keep stdout and the file clean.
    eprintln!("excluded {excluded} timed-out trials (of {total})");

    let mut writer = csv::Writer::from_path(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    writer.write_record(["choice", "rt"])?;
    for trial in &completed {
        let choice = if trial.upper_boundary {
            "upper"
        } else {
            "lower"
        };
        writer.write_record([choice, &format!("{:.6}", trial.rt)])?;
    }
    writer.flush().context("flushing CSV output")?;

    eprintln!("wrote {} trials to {}", completed.len(), args.out.display());
    Ok(())
}

/// Split simulated trials into completed ones (kept, in order) and a count of
/// those force-terminated at the [`sinistra_core::MAX_SIM_TIME`] cutoff.
fn filter_completed(trials: Vec<Trial>) -> (Vec<Trial>, usize) {
    let total = trials.len();
    let completed: Vec<Trial> = trials.into_iter().filter(|t| !t.timed_out).collect();
    let excluded = total - completed.len();
    (completed, excluded)
}

/// Read a `(choice, rt)` CSV (with header) into trials. Rows loaded from disk
/// are completed trials by construction (`simulate` never writes timed-out
/// ones), so `timed_out` is always `false`.
fn read_trials_csv(path: &PathBuf) -> Result<Vec<Trial>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;

    let mut trials = Vec::new();
    for (row, result) in reader.records().enumerate() {
        let record = result.with_context(|| format!("reading row {}", row + 1))?;
        let choice = record
            .get(0)
            .with_context(|| format!("row {}: missing choice column", row + 1))?;
        let rt: f64 = record
            .get(1)
            .with_context(|| format!("row {}: missing rt column", row + 1))?
            .trim()
            .parse()
            .with_context(|| format!("row {}: parsing rt", row + 1))?;
        let upper_boundary = match choice.trim() {
            "upper" => true,
            "lower" => false,
            other => bail!("row {}: unrecognized choice value {other:?}", row + 1),
        };
        trials.push(Trial {
            upper_boundary,
            rt,
            timed_out: false,
        });
    }
    ensure!(!trials.is_empty(), "no trials in {}", path.display());
    Ok(trials)
}

fn print_params(params: &Params) {
    println!("  drift_rate          {:.6}", params.drift_rate);
    println!("  boundary_separation {:.6}", params.boundary_separation);
    println!("  starting_point      {:.6}", params.starting_point);
    println!("  non_decision_time   {:.6}", params.non_decision_time);
    println!("  noise_sd            {:.6}", params.noise_sd);
}

fn run_fit_ez(args: FitEzArgs) -> Result<()> {
    let trials = read_trials_csv(&args.input)?;
    let params = ez_diffusion(&trials).map_err(|e| anyhow::anyhow!("EZ-diffusion failed: {e}"))?;

    println!("recovered parameters ({} trials):", trials.len());
    print_params(&params);
    Ok(())
}

fn run_fit_sim(args: FitSimArgs) -> Result<()> {
    let trials = read_trials_csv(&args.input)?;

    let guess = default_fit_guess(&trials);
    let started = std::time::Instant::now();
    let fit = fit_simulation(&trials, guess, args.max_iters)
        .map_err(|e| anyhow::anyhow!("simulation fit failed: {e}"))?;
    let elapsed = started.elapsed();

    println!("recovered parameters ({} trials):", trials.len());
    print_params(&fit.params);
    println!(
        "  [{} Nelder-Mead iterations, {} trials/eval, final cost {:.3e}, {:.2?}]",
        fit.iterations, FIT_N_PER_EVAL, fit.final_cost, elapsed
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(upper: bool, rt: f64, timed_out: bool) -> Trial {
        Trial {
            upper_boundary: upper,
            rt,
            timed_out,
        }
    }

    #[test]
    fn filter_completed_drops_timed_out_and_counts_them() {
        let trials = vec![
            trial(true, 0.5, false),
            trial(false, 9.9, true),
            trial(true, 0.7, false),
            trial(true, 10.1, true),
            trial(false, 0.6, false),
        ];

        let (completed, excluded) = filter_completed(trials);

        assert_eq!(excluded, 2);
        assert_eq!(completed.len(), 3);
        assert!(completed.iter().all(|t| !t.timed_out));
        // order is preserved
        assert_eq!(completed[0].rt, 0.5);
        assert_eq!(completed[1].rt, 0.7);
        assert_eq!(completed[2].rt, 0.6);
    }

    #[test]
    fn filter_completed_all_kept_when_none_time_out() {
        let trials = vec![trial(true, 0.4, false), trial(false, 0.5, false)];
        let (completed, excluded) = filter_completed(trials);
        assert_eq!(excluded, 0);
        assert_eq!(completed.len(), 2);
    }
}
