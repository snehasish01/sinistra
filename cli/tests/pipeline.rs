//! End-to-end checks of the `simulate` -> CSV -> `fit-ez` pipeline, driving
//! the built binary.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_sinistra");

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "sinistra-it-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Number of data rows (excluding the header) in a `choice,rt` CSV.
fn csv_data_rows(path: &PathBuf) -> usize {
    let text = fs::read_to_string(path).unwrap();
    text.lines().filter(|l| !l.is_empty()).count() - 1
}

#[test]
fn simulate_excludes_timed_out_trials_from_csv() {
    let dir = scratch_dir();
    let csv = dir.join("trials.csv");
    let n: usize = 1_500;

    // Low drift + a wide boundary means a substantial fraction of trials never
    // reach a bound within MAX_SIM_TIME and are force-terminated.
    let output = Command::new(BIN)
        .args([
            "simulate",
            "--drift",
            "0.3",
            "--boundary",
            "6.0",
            "--start",
            "0.5",
            "--t0",
            "0.1",
            "--n",
            &n.to_string(),
            "--seed",
            "7",
            "--out",
        ])
        .arg(&csv)
        .output()
        .expect("failed to run simulate");

    assert!(output.status.success(), "simulate exited non-zero");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        String::from_utf8(output.stdout).unwrap().is_empty(),
        "simulate must not write to stdout"
    );

    // Parse: "excluded N timed-out trials (of TOTAL)"
    // words: ["excluded", "N", "timed-out", "trials", "(of", "TOTAL)"]
    let line = stderr
        .lines()
        .find(|l| l.starts_with("excluded "))
        .expect("missing exclusion line on stderr");
    let words: Vec<&str> = line.split_whitespace().collect();
    let excluded: usize = words[1].parse().unwrap();
    let total: usize = words[5].trim_end_matches(')').parse().unwrap();

    assert_eq!(total, n, "reported total should equal --n");
    assert!(
        excluded > 0 && excluded < n,
        "expected a non-trivial, partial timeout rate, got {excluded}/{n}"
    );

    // The CSV must contain exactly the completed trials.
    let rows = csv_data_rows(&csv);
    assert_eq!(
        rows,
        n - excluded,
        "CSV rows ({rows}) should equal total - excluded ({} - {excluded})",
        n
    );

    // Every retained RT is a real crossing time, below the cap + t0 = 10.1 s.
    // The 10.1 is MAX_SIM_TIME (10 s, sinistra_core) + this test's --t0 (0.1 s);
    // update it if either of those changes.
    let text = fs::read_to_string(&csv).unwrap();
    for row in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let rt: f64 = row.split(',').nth(1).unwrap().parse().unwrap();
        assert!(rt < 10.1, "retained trial has cap-length RT: {rt}");
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn simulate_then_fit_ez_roundtrip() {
    let dir = scratch_dir();
    let csv = dir.join("clean.csv");

    let sim = Command::new(BIN)
        .args([
            "simulate",
            "--drift",
            "1.2",
            "--boundary",
            "1.0",
            "--start",
            "0.5",
            "--t0",
            "0.25",
            "--n",
            "50000",
            "--seed",
            "42",
            "--out",
        ])
        .arg(&csv)
        .output()
        .expect("failed to run simulate");
    assert!(sim.status.success());
    let sim_stderr = String::from_utf8(sim.stderr).unwrap();
    // These parameters never time out.
    assert!(sim_stderr.contains("excluded 0 timed-out trials (of 50000)"));

    let fit = Command::new(BIN)
        .args(["fit-ez", "--input"])
        .arg(&csv)
        .output()
        .expect("failed to run fit-ez");
    assert!(
        fit.status.success(),
        "fit-ez exited non-zero: {}",
        String::from_utf8_lossy(&fit.stderr)
    );

    let stdout = String::from_utf8(fit.stdout).unwrap();
    let drift_line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("drift_rate"))
        .expect("no drift_rate in output");
    let drift: f64 = drift_line
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (drift - 1.2).abs() / 1.2 < 0.08,
        "recovered drift {drift} not within 8% of 1.2"
    );

    let _ = fs::remove_dir_all(&dir);
}
