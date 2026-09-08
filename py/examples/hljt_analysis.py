"""Phase 7 — does the HLJT biomechanical-constraints effect decompose into a
drift, boundary, or non-decision-time difference?

The "biomechanical constraints" effect (Parsons, 1994): laterally-rotated hand
stimuli take longer to judge than medially-rotated ones, because the implied
movement is anatomically harder. Moreno-Verdú et al. (2024/2025) confirm the
RT effect in their HLJT. Here we fit a drift-diffusion model to each design
cell (via sinistra's EZ and simulation fitters) and ask which DDM parameter
carries the Medial-vs-Lateral difference.

Data: data/hljt/all_data_inperson.csv (see data/hljt/README.md for attribution;
CC BY 4.0). In-person version only.

    cd py
    . .venv/bin/activate
    pip install matplotlib numpy
    maturin develop --release
    python examples/hljt_analysis.py

IMPORTANT SIMPLIFICATION (flag this in any writeup):
  This pools *all trials from all participants* into one DDM per cell rather
  than fitting a hierarchical / per-subject model. Between-subject variation —
  notably individual differences in motor-imagery ability (MIQ-3 varies a lot
  in this sample) — is absorbed into the aggregate estimate, which inflates
  the effective noise and can bias the pooled parameters relative to the mean
  of per-subject fits. Adequate for an exploratory "which parameter moves"
  question; not a substitute for a hierarchical DDM.

Also:
  * Foot and Bimanual response groups are POOLED. Justified: the paper reports
    no Group x Angle interaction on RT for the in-person version.
  * Choice is accuracy-coded: upper boundary = correct response, lower =
    error. So "higher drift" == "more evidence for the correct answer".
"""

import csv
import time
from pathlib import Path

_START = time.perf_counter()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sinistra

DATA = Path(__file__).resolve().parents[2] / "data" / "hljt" / "all_data_inperson.csv"
OUT = Path(__file__).parent / "hljt_recovery.png"

VIEWS = ["Palmar", "Dorsal"]
DIRECTIONS = ["Medial", "Lateral"]
ANGLES = [45, 90, 135]
LOW_N = 300  # cells below this are fit but flagged low-confidence

PARAMS = [
    ("drift_rate", "drift rate  $v$"),
    ("boundary_separation", "boundary separation  $a$"),
    ("non_decision_time", "non-decision time  $T_{er}$  (s)"),
]

DIR_COLOR = {"Medial": "#2a9d8f", "Lateral": "#e76f51"}
METHOD_STYLE = {
    "EZ": dict(marker="o", linestyle="-", markersize=6),
    "sim-fit": dict(marker="^", linestyle="--", markersize=6, markerfacecolor="white"),
}


def load_cells():
    """Return {(view, direction, angle): {'rows': [...], 'choices', 'rts'}}."""
    rows = []
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            if r["Direction"] not in ("Medial", "Lateral"):
                continue  # drops Angle_unif 0 and 180 (no medial/lateral contrast)
            rows.append(
                {
                    "Direction": r["Direction"],
                    "View": r["View"],
                    "Angle_unif": int(r["Angle_unif"]),
                    "Accuracy": int(r["Accuracy"]),
                    "RT_s": float(r["RT"]) / 1000.0,  # ms -> s, sinistra's units
                }
            )

    cells = {}
    for view in VIEWS:
        for direction in DIRECTIONS:
            for angle in ANGLES:
                sub = [
                    r
                    for r in rows
                    if r["View"] == view
                    and r["Direction"] == direction
                    and r["Angle_unif"] == angle
                ]
                choices = np.array([r["Accuracy"] == 1 for r in sub], dtype=bool)
                rts = np.array([r["RT_s"] for r in sub], dtype=np.float64)
                cells[(view, direction, angle)] = dict(
                    n=len(sub),
                    accuracy=float(choices.mean()) if len(sub) else float("nan"),
                    mean_rt=float(rts.mean()) if len(sub) else float("nan"),
                    choices=choices,
                    rts=rts,
                )
    return cells


def fit_cells(cells):
    for key, c in cells.items():
        c["EZ"] = None
        c["sim-fit"] = None
        if c["n"] < 4:
            continue
        try:
            ez = sinistra.fit_ez(c["choices"], c["rts"])
            c["EZ"] = ez
        except sinistra.SinistraError as e:
            print(f"  fit_ez failed for {key}: {e}")
            ez = None
        try:
            guess = ez if ez is not None else None
            c["sim-fit"] = sinistra.fit_sim(
                c["choices"], c["rts"], initial_guess=guess, max_iters=200
            )
        except sinistra.SinistraError as e:
            print(f"  fit_sim failed for {key}: {e}")
    return cells


def print_table(cells):
    print("\n" + "=" * 108)
    print("Per-cell DDM fits  (in-person HLJT, Foot+Bimanual pooled, accuracy-coded)")
    print("=" * 108)
    hdr = (
        f"{'view':<7}{'dir':<8}{'ang':>4}{'N':>7}{'acc':>7}{'mRT':>7}  "
        f"{'v_EZ':>7}{'a_EZ':>7}{'t0_EZ':>7}   {'v_sim':>7}{'a_sim':>7}{'t0_sim':>8}"
    )
    print(hdr)
    print("-" * 108)
    def cell(d, k, width=7):
        return f"{d[k]:{width}.3f}" if d else f"{'--':>{width}}"

    for view in VIEWS:
        for direction in DIRECTIONS:
            for angle in ANGLES:
                c = cells[(view, direction, angle)]
                ez, sim = c["EZ"], c["sim-fit"]
                flag = "  << low N" if c["n"] < LOW_N else ""
                print(
                    f"{view:<7}{direction:<8}{angle:>4}{c['n']:>7}"
                    f"{c['accuracy']:>7.3f}{c['mean_rt']:>7.3f}  "
                    f"{cell(ez,'drift_rate')}{cell(ez,'boundary_separation')}"
                    f"{cell(ez,'non_decision_time')}   "
                    f"{cell(sim,'drift_rate')}{cell(sim,'boundary_separation')}"
                    f"{cell(sim,'non_decision_time', 8)}{flag}"
                )


def print_effect_summary(cells):
    """Lateral - Medial difference per parameter, averaged over view x angle."""
    print("\n" + "=" * 72)
    print("Biomechanical-constraints effect (Lateral - Medial), mean over the")
    print("6 view x angle combinations, per DDM parameter and method")
    print("=" * 72)
    # raw behavioural effect first
    for method_label, key in [("raw accuracy", "accuracy"), ("raw mean RT (s)", "mean_rt")]:
        diffs = []
        for view in VIEWS:
            for angle in ANGLES:
                m = cells[(view, "Medial", angle)][key]
                latv = cells[(view, "Lateral", angle)][key]
                diffs.append(latv - m)
        print(f"  {method_label:<22} Lateral - Medial = {np.mean(diffs):+.4f}")

    print()
    for method in ("EZ", "sim-fit"):
        print(f"  [{method}]")
        for pkey, plabel in PARAMS:
            diffs = []
            for view in VIEWS:
                for angle in ANGLES:
                    md = cells[(view, "Medial", angle)][method]
                    ld = cells[(view, "Lateral", angle)][method]
                    if md and ld:
                        diffs.append(ld[pkey] - md[pkey])
            if diffs:
                rel = np.mean(diffs) / np.mean(
                    [
                        cells[(v, "Medial", a)][method][pkey]
                        for v in VIEWS
                        for a in ANGLES
                        if cells[(v, "Medial", a)][method]
                    ]
                )
                print(
                    f"    {plabel:<28} Lateral - Medial = {np.mean(diffs):+.4f}"
                    f"   ({rel:+.1%} of the Medial level)"
                )


def make_figure(cells):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.5), sharex=True, sharey="row")

    for row, (pkey, plabel) in enumerate(PARAMS):
        for col, view in enumerate(VIEWS):
            ax = axes[row, col]
            for direction in DIRECTIONS:
                color = DIR_COLOR[direction]
                for method in ("EZ", "sim-fit"):
                    ys = [
                        cells[(view, direction, a)][method][pkey]
                        if cells[(view, direction, a)][method]
                        else np.nan
                        for a in ANGLES
                    ]
                    ax.plot(
                        ANGLES,
                        ys,
                        color=color,
                        label=f"{direction} · {method}",
                        **METHOD_STYLE[method],
                    )
            ax.set_xticks(ANGLES)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(f"{view} view", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"recovered {plabel}")
            if row == 2:
                ax.set_xlabel("angular disparity from upright (°)")

    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle(
        "HLJT biomechanical-constraints effect decomposed by DDM parameter\n"
        "in-person data, one pooled DDM per View × Direction × Angle cell "
        "(Foot + Bimanual pooled)",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.012,
        "Teal = medially-rotated stimuli, orange = laterally-rotated (the harder, slower condition). "
        "Solid/circle = EZ-diffusion, dashed/open-triangle = simulation fit.\n"
        "One DDM is fit to all trials pooled across participants per cell — individual differences "
        "(e.g. motor-imagery ability) are folded into each estimate.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUT, dpi=140)
    print(f"\nsaved {OUT}")


def main():
    print(f"loading {DATA.relative_to(Path(__file__).resolve().parents[2])}")
    cells = load_cells()

    print("\ncell trial counts (View × Direction × Angle_unif):")
    for view in VIEWS:
        for direction in DIRECTIONS:
            ns = [cells[(view, direction, a)]["n"] for a in ANGLES]
            parts = [
                f"{a}°: {n}{'  LOW' if n < LOW_N else ''}" for a, n in zip(ANGLES, ns)
            ]
            print(f"  {view:<7} {direction:<8} " + "   ".join(parts))
    total = sum(c["n"] for c in cells.values())
    print(f"  total trials in analysis: {total}")

    fit_cells(cells)
    print_table(cells)
    print_effect_summary(cells)
    make_figure(cells)
    print(f"\ntotal wall-clock: {time.perf_counter() - _START:.1f} s")


if __name__ == "__main__":
    main()
