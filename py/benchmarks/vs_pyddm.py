"""sinistra vs PyDDM — a dev-only comparison. NOT a pytest test, NOT a dep.

Run:
    cd py
    . .venv/bin/activate            # or your env
    pip install pyddm matplotlib    # dev-only, not in pyproject
    maturin develop --release
    python benchmarks/vs_pyddm.py

Produces two figures in this directory and prints two results tables.

------------------------------------------------------------------------------
What PyDDM actually offers for *simulation* (checked against pyddm 0.9.0):

PyDDM's engine is a Fokker-Planck PDE solver (`Model.solve()`, C backend
`solve_numerical_c`) that computes the whole RT distribution directly - it
never simulates individual sample paths.

It DOES also ship a genuine trial-by-trial path simulator:
  * `Model.simulate_trial()`     - one Euler (or RK4) SDE trajectory
  * `Model.simulated_solution(size=n)` - loops simulate_trial n times -> Sample
This is the same kind of computation sinistra.simulate() does. But PyDDM's own
docstrings call it "an outdated method ... should be used for comparison
purposes only" and "you should never need to use this function" - it is a
pure-Python per-timestep loop, unoptimised by design, and runs at ~65
trials/sec. sinistra runs at ~1.5M/sec. So PART 1 is a real like-for-like
comparison of the trial-simulation methods each library exposes, but note
that it pits sinistra's optimised parallel Rust core against PyDDM's
deliberately-unoptimised debugging helper.

PART 2 (time to a fit) uses each library's *intended* tool and is explicitly
NOT a like-for-like engine benchmark - see the caption there.
------------------------------------------------------------------------------
"""

import statistics
import time
import warnings

_T0 = time.perf_counter()

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sinistra

try:
    import pyddm
except ImportError:
    raise SystemExit(
        "This dev-only benchmark needs PyDDM, which is deliberately not a "
        "dependency of the sinistra package.\n    pip install pyddm matplotlib"
    )

pyddm.set_log_level(40)  # ERROR only; simulated_solution is chatty

HERE = Path(__file__).parent

# Shared "truth" - an unbiased DDM, noise s = 1, matching sinistra's convention.
TRUE_DRIFT = 1.2
TRUE_BOUNDARY = 1.0       # sinistra: distance between the two bounds
TRUE_B = TRUE_BOUNDARY / 2  # PyDDM: symmetric bound at +/- B
TRUE_T0 = 0.25

SINISTRA_C = "#1f6feb"
PYDDM_C = "#d1651a"


def sinistra_simulate(n, seed):
    choices, rts, _ = sinistra.simulate(
        drift=TRUE_DRIFT,
        boundary=TRUE_BOUNDARY,
        start=0.5,
        t0=TRUE_T0,
        n=n,
        seed=seed,
        noise_sd=1.0,
    )
    return choices, rts


def pyddm_sim_model():
    # Euler integration (rk4=False in the call), dt matched to sinistra's 1e-3,
    # 10 s cap matched to sinistra's MAX_SIM_TIME. No overlay: simulate_trial
    # ignores overlays anyway (non-decision time is a solve()-time convolution).
    return pyddm.Model(
        drift=pyddm.DriftConstant(drift=TRUE_DRIFT),
        noise=pyddm.NoiseConstant(noise=1.0),
        bound=pyddm.BoundConstant(B=TRUE_B),
        IC=pyddm.ICPointSourceCenter(),
        dt=0.001,
        dx=0.001,
        T_dur=10.0,
    )


# ---------------------------------------------------------------------------
# PART 1 - raw trial-simulation throughput (apples to apples)
# ---------------------------------------------------------------------------

SINISTRA_SIZES = [10_000, 100_000, 1_000_000]
# PyDDM's loop is ~65 trials/sec, so 1e5 would take ~25 min and 1e6 ~4 h.
# Measure it at sizes that finish in seconds and project the rest - the rate
# is flat in n (it is a per-trial Python loop).
PYDDM_MEASURE_SIZES = [1_000, 3_000]


def part1():
    print("=" * 74)
    print("PART 1  raw trial-simulation throughput  (apples to apples)")
    print("=" * 74)

    # sinistra: best of 3 at each size
    sin = {}
    sinistra_simulate(20_000, seed=0)  # warm up thread pool / allocator
    for n in SINISTRA_SIZES:
        samples = [
            _time(lambda: sinistra_simulate(n, seed=s))[0] for s in range(3)
        ]
        sin[n] = min(samples)

    # PyDDM: real measurements at small n
    model = pyddm_sim_model()
    pyddm_meas = {}
    for n in PYDDM_MEASURE_SIZES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            elapsed, _ = _time(lambda: model.simulated_solution(size=n, rk4=False, seed=1))
        pyddm_meas[n] = elapsed
    pyddm_rate = statistics.mean(n / t for n, t in pyddm_meas.items())

    print(f"\nPyDDM simulated_solution measured: "
          + ", ".join(f"n={n} -> {t:.1f}s" for n, t in pyddm_meas.items())
          + f"   (=> {pyddm_rate:,.0f} trials/sec, flat in n)")
    print(f"sinistra.simulate() is parallel (rayon); PyDDM's loop is single-threaded pure Python.\n")

    rows = []
    for n in SINISTRA_SIZES:
        s_t = sin[n]
        p_t = n / pyddm_rate  # projected
        rows.append((n, s_t, n / s_t, p_t, pyddm_rate, p_t / s_t))

    print(f"{'n trials':>12} | {'sinistra':>12} {'trials/sec':>14} | "
          f"{'PyDDM (proj.)':>14} {'trials/sec':>12} | {'PyDDM / sinistra':>16}")
    print("-" * 92)
    for n, s_t, s_r, p_t, p_r, ratio in rows:
        print(f"{n:>12,} | {s_t*1e3:>10.1f}ms {s_r:>14,.0f} | "
              f"{_fmt_time(p_t):>14} {p_r:>12,.0f} | {ratio:>15,.0f}x")

    _part1_figure(rows, pyddm_meas, pyddm_rate)
    return rows


def _part1_figure(rows, pyddm_meas, pyddm_rate):
    ns = [r[0] for r in rows]
    sin_t = [r[1] for r in rows]
    pyddm_t = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(len(ns))
    w = 0.38

    b1 = ax.bar(x - w / 2, sin_t, w, color=SINISTRA_C,
                label="sinistra.simulate()  (measured, parallel)")
    ax.bar(x + w / 2, pyddm_t, w, color=PYDDM_C, hatch="///", edgecolor="white",
           label="PyDDM simulated_solution  (projected from its flat small-n rate)")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_xlabel("number of trials simulated")
    ax.set_ylabel("wall-clock time (s, log scale)")
    ax.set_ylim(top=max(pyddm_t) * 60)  # headroom for the ratio labels
    ax.set_title("Raw trial-simulation throughput: sinistra vs PyDDM   "
                 "(same computation: Euler SDE integration)")

    for rect, t in zip(b1, sin_t):
        ax.annotate(_fmt_time(t), (rect.get_x() + rect.get_width() / 2, t),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    for xi, t in zip(x + w / 2, pyddm_t):
        ax.annotate(_fmt_time(t), (xi, t),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    for i, ratio in enumerate(r[5] for r in rows):
        ax.annotate(f"{ratio:,.0f}× slower", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=9, fontweight="bold")

    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.92))
    meas = " and ".join(f"n={n} ({t:.0f}s)" for n, t in pyddm_meas.items())
    fig.text(
        0.5, 0.02,
        f"PyDDM's simulated_solution is a pure-Python per-timestep loop (~{pyddm_rate:,.0f} "
        "trials/sec) that its own docs call a debugging helper.\n"
        "PyDDM's real engine is a Fokker–Planck PDE solver that never simulates sample paths.\n"
        f"PyDDM bars are projected from its flat rate (measured at {meas}); "
        "n = 1,000,000 would take ~4 h to actually run.",
        ha="center", va="bottom", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    out = HERE / "vs_pyddm_simulation.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


# ---------------------------------------------------------------------------
# PART 2 - time to a fitted parameter estimate (NOT apples to apples)
# ---------------------------------------------------------------------------

FIT_N = 10_000
FIT_REPS = 3


def part2():
    print("\n" + "=" * 74)
    print("PART 2  time to a fit result  (NOT apples to apples)")
    print("=" * 74)

    choices, rts = sinistra_simulate(FIT_N, seed=7)
    ez = sinistra.fit_ez(choices, rts)

    # sinistra: Nelder-Mead over Monte-Carlo summary stats, EZ warm start
    sin_times, sin_est = [], None
    for _ in range(FIT_REPS):
        t, res = _time(lambda: sinistra.fit_sim(choices, rts, initial_guess=ez, max_iters=200))
        sin_times.append(t)
        sin_est = res

    # PyDDM: its default fitter — differential evolution over the PDE
    # likelihood, on its default solver grid.
    arr = np.column_stack([rts.astype(float), choices.astype(int)])
    sample = pyddm.Sample.from_numpy_array(arr)  # (rt, choice) with 1==correct
    pyddm_times, pyddm_params = [], None
    for _ in range(FIT_REPS):
        model = _pyddm_fit_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t, _ = _time(lambda: model.fit(sample=sample, verbose=False))
        pyddm_times.append(t)
        pyddm_params = [float(p) for p in model.get_model_parameters()]

    sin_t = min(sin_times)
    pyddm_t = min(pyddm_times)

    print(f"\ntrue params:            v={TRUE_DRIFT}  a={TRUE_BOUNDARY}  t0={TRUE_T0}")
    print(f"sinistra fit_sim:        v={sin_est['drift_rate']:.3f}  "
          f"a={sin_est['boundary_separation']:.3f}  t0={sin_est['non_decision_time']:.3f}   "
          f"[{sin_est['iterations']} iters]")
    print(f"PyDDM fit (default):     v={pyddm_params[0]:.3f}  "
          f"a={2*pyddm_params[1]:.3f}  t0={pyddm_params[2]:.3f}")
    print()
    print(f"{'method':<28}{'best of %d (s)' % FIT_REPS:>16}{'all runs (s)':>26}")
    print("-" * 70)
    print(f"{'sinistra.fit_sim':<28}{sin_t:>16.2f}"
          f"{'  '.join(f'{x:.2f}' for x in sin_times):>26}")
    print(f"{'PyDDM Model.fit':<28}{pyddm_t:>16.2f}"
          f"{'  '.join(f'{x:.2f}' for x in pyddm_times):>26}")

    caption = ("Different underlying algorithms (Monte Carlo search vs numerical PDE + "
               "likelihood) — this compares practical\ntime-to-answer using each library's own "
               "tools, not simulation speed. Not a like-for-like engine benchmark.")
    print("\n" + caption)

    _part2_figure(sin_t, pyddm_t, caption)
    return sin_t, pyddm_t


def _pyddm_fit_model():
    return pyddm.Model(
        drift=pyddm.DriftConstant(drift=pyddm.Fittable(minval=0.1, maxval=5.0)),
        noise=pyddm.NoiseConstant(noise=1.0),
        bound=pyddm.BoundConstant(B=pyddm.Fittable(minval=0.3, maxval=2.0)),
        IC=pyddm.ICPointSourceCenter(),
        overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0.0, maxval=0.6)),
        T_dur=6.0,
    )


def _part2_figure(sin_t, pyddm_t, caption):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    bars = ax.bar(
        ["sinistra\nfit_sim\n(MC + Nelder–Mead)", "PyDDM\nModel.fit\n(PDE + diff. evolution)"],
        [sin_t, pyddm_t],
        color=[SINISTRA_C, PYDDM_C],
        width=0.55,
    )
    for rect, t in zip(bars, [sin_t, pyddm_t]):
        ax.annotate(f"{t:.2f} s", (rect.get_x() + rect.get_width() / 2, t),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
    ax.set_ylabel("wall-clock time to a fitted estimate (s)")
    ax.set_ylim(0, max(sin_t, pyddm_t) * 1.35)
    ax.set_title(f"Time to a parameter fit on the same {FIT_N:,}-trial data set")
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=7.8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out = HERE / "vs_pyddm_fit.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


# ---------------------------------------------------------------------------

def _time(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def _fmt_time(s):
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def main():
    part1()
    part2()
    print(f"\ntotal wall-clock: {_fmt_time(time.perf_counter() - _T0)}")


if __name__ == "__main__":
    main()
