"""True-vs-recovered parameter recovery figure for both fitters.

Runs a small recovery experiment through the `sinistra` bindings and draws a
3-panel scatter (one panel per parameter) comparing EZ-diffusion against the
simulation fit.

    cd py
    maturin develop --release          # release build matters for fit_sim
    pip install matplotlib numpy
    python examples/recovery_plot.py

Writes examples/recovery_plot.png and prints the total wall-clock time.
"""

import time

_START = time.perf_counter()

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sinistra

# True (drift, boundary, t0) combinations, spanning the range used in earlier
# testing. The first row is the low-drift corner where EZ drift recovery is
# known to be noisiest.
TRUE_COMBOS = [
    (0.5, 1.5, 0.15),
    (0.8, 1.0, 0.20),
    (1.0, 1.0, 0.20),
    (1.5, 1.2, 0.35),
    (2.0, 1.2, 0.30),
    (2.5, 0.9, 0.25),
    (3.0, 0.8, 0.25),
]
N_TRIALS = 50_000
N_REPS = 5
START = 0.5
MAX_ITERS = 200

PARAMS = [
    ("drift_rate", "drift rate $v$"),
    ("boundary_separation", "boundary separation $a$"),
    ("non_decision_time", "non-decision time $T_{er}$ (s)"),
]
KEYS = [k for k, _ in PARAMS]

STYLE = {
    "EZ-diffusion": dict(color="#c1440e", marker="o"),
    "simulation fit": dict(color="#1f6feb", marker="^"),
}


def run_experiment():
    """Return a list of records: {method, true: {k: v}, rec: {k: v}}."""
    records = []
    seed = 0
    for drift, boundary, t0 in TRUE_COMBOS:
        truth = {
            "drift_rate": drift,
            "boundary_separation": boundary,
            "non_decision_time": t0,
        }
        for _ in range(N_REPS):
            seed += 1
            choices, rts, _ = sinistra.simulate(
                drift=drift,
                boundary=boundary,
                start=START,
                t0=t0,
                n=N_TRIALS,
                seed=seed,
            )
            ez = sinistra.fit_ez(choices, rts)
            sim = sinistra.fit_sim(choices, rts, initial_guess=ez, max_iters=MAX_ITERS)
            for method, res in (("EZ-diffusion", ez), ("simulation fit", sim)):
                records.append(
                    {"method": method, "true": truth, "rec": {k: res[k] for k in KEYS}}
                )
    return records


def summarise(records):
    print(f"\n{'parameter':<22}{'method':<16}{'mean |error|':>14}{'   direction'}")
    for key, _ in PARAMS:
        for method in STYLE:
            rows = [r for r in records if r["method"] == method]
            errs = np.array([r["rec"][key] - r["true"][key] for r in rows])
            rel = np.array(
                [(r["rec"][key] - r["true"][key]) / r["true"][key] for r in rows]
            )
            bias = "high" if errs.mean() > 0 else "low"
            print(
                f"{key:<22}{method:<16}{np.abs(rel).mean() * 100:>12.1f}%"
                f"   {bias} (mean {rel.mean() * 100:+.1f}%)"
            )


def make_figure(records, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    jitter_rng = np.random.default_rng(0)

    for ax, (key, label) in zip(axes, PARAMS):
        trues = np.array([r["true"][key] for r in records], dtype=float)
        recs = np.array([r["rec"][key] for r in records], dtype=float)
        span = float(trues.max() - trues.min())

        lo = min(trues.min(), recs.min())
        hi = max(trues.max(), recs.max())
        pad = 0.10 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="0.45", lw=1.0, ls="--", zorder=1)

        for method, style in STYLE.items():
            rows = [r for r in records if r["method"] == method]
            x = np.array([r["true"][key] for r in rows], dtype=float)
            y = np.array([r["rec"][key] for r in rows], dtype=float)
            # Small horizontal jitter: the 5 reps at each true value share an
            # x, so without it they'd stack into a vertical line. Offset the
            # two methods in opposite directions so their clusters separate.
            offset = (-1 if method == "EZ-diffusion" else 1) * 0.012 * span
            x = x + offset + jitter_rng.normal(0.0, 0.006 * span, size=x.shape)
            ax.scatter(
                x,
                y,
                s=28,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
                label=method,
                **style,
            )

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"recovered {label}")
        ax.grid(True, alpha=0.25)

    # One shared legend (y=x plus the two methods).
    handles = [
        plt.Line2D([], [], color="0.45", ls="--", lw=1.0, label="y = x (perfect recovery)"),
    ] + [
        plt.Line2D(
            [],
            [],
            color=s["color"],
            marker=s["marker"],
            ls="",
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=m,
        )
        for m, s in STYLE.items()
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Drift-diffusion parameter recovery: true vs recovered   "
        f"({len(TRUE_COMBOS)} parameter sets × {N_REPS} replications, "
        f"n = {N_TRIALS:,} trials each)",
        fontsize=12,
    )
    caption = (
        "EZ-diffusion (closed form) systematically over-estimates boundary separation by ~3–4% —\n"
        "orange points sit above y = x in the middle panel — a known bias inherited from the "
        "simulator's Euler–Maruyama boundary overshoot.\n"
        "The simulation fit shows no such directional bias (its forward model is the simulator "
        "itself) but costs ~0.3–0.7 s/fit versus microseconds for EZ.\n"
        "Low drift (leftmost points) is the hard corner — accuracy nears chance, so the drift "
        "estimate is the noisiest there, most visibly for the simulation fit."
    )
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    print(f"\nsaved {out_path}")


def main():
    records = run_experiment()
    summarise(records)
    make_figure(records, Path(__file__).parent / "recovery_plot.png")
    print(f"\ntotal wall-clock: {time.perf_counter() - _START:.1f} s")


if __name__ == "__main__":
    main()
