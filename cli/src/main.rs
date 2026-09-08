//! Thin CLI wrapper around [`sinistra_core`].

use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use sinistra_core::{simulate_n, Params};

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

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Simulate(args) => run_simulate(args),
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
    let trials = simulate_n(&params, args.n, args.seed);

    let mut writer = csv::Writer::from_path(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    writer.write_record(["choice", "rt"])?;
    for trial in &trials {
        let choice = if trial.upper_boundary {
            "upper"
        } else {
            "lower"
        };
        writer.write_record([choice, &format!("{:.6}", trial.rt)])?;
    }
    writer.flush().context("flushing CSV output")?;

    eprintln!("wrote {} trials to {}", trials.len(), args.out.display());
    Ok(())
}
