"""Wall-clock throughput of the NumPy-array `simulate()` binding.

Not a pytest test — run it directly:

    cd py
    maturin develop --release      # MUST be --release for meaningful numbers
    python benches/marshalling.py

It times `sinistra.simulate()` end to end (Rust `simulate_n` + timeout filter
+ building the two NumPy arrays) at several `n`, so we can see how close the
Python-callable API gets to the core `simulate_n` criterion result from
phase 1 (~1.52M trials/sec, same parameters).
"""

import statistics
import time

import sinistra

# Same parameters as the phase-1 criterion benchmark (core/benches/simulate.rs).
PARAMS = dict(drift=0.5, boundary=1.0, start=0.5, t0=0.2)
CORE_BENCH_TRIALS_PER_SEC = 1_520_000

SIZES = [10_000, 100_000, 1_000_000]
REPEATS = 7


def time_once(n: int, seed: int) -> float:
    start = time.perf_counter()
    choices, rts, n_excluded = sinistra.simulate(**PARAMS, n=n, seed=seed)
    elapsed = time.perf_counter() - start
    assert len(choices) == len(rts) == n - n_excluded
    return elapsed


def main() -> None:
    # Warm up (thread pool spin-up, allocator, page faults).
    time_once(50_000, seed=0)

    print(f"sinistra {sinistra.__version__} — simulate() throughput")
    print(f"params: {PARAMS}\n")
    print(f"{'n':>12}  {'best (ms)':>10}  {'median (ms)':>12}  {'trials/sec':>14}  {'vs core':>8}")
    print("-" * 64)

    for n in SIZES:
        samples = [time_once(n, seed=s) for s in range(REPEATS)]
        best = min(samples)
        median = statistics.median(samples)
        tps = n / best
        frac = tps / CORE_BENCH_TRIALS_PER_SEC
        print(
            f"{n:>12,}  {best * 1e3:>10.2f}  {median * 1e3:>12.2f}  "
            f"{tps:>14,.0f}  {frac:>7.0%}"
        )

    print(
        f"\ncore simulate_n criterion baseline (phase 1): "
        f"~{CORE_BENCH_TRIALS_PER_SEC:,} trials/sec"
    )


if __name__ == "__main__":
    main()
