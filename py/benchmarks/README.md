# sinistra vs PyDDM — dev-only comparison

**Not** run by pytest. **Not** a package dependency. `pyddm` is installed by
hand only when someone wants to regenerate these numbers.

```sh
cd py
. .venv/bin/activate                 # or your environment
pip install pyddm matplotlib         # dev-only, deliberately not in pyproject.toml
maturin develop --release            # --release matters
python benchmarks/vs_pyddm.py
```

Runs in ~70 s and writes `vs_pyddm_simulation.png` and `vs_pyddm_fit.png`
here, plus two tables on stdout.

## What PyDDM offers for *simulation* (pyddm 0.9.0)

PyDDM's engine is a **Fokker–Planck PDE solver** (`Model.solve()`, C backend
`solve_numerical_c`) that computes the whole RT distribution directly — it
never simulates individual sample paths.

It *also* ships a genuine trial-by-trial path simulator:

| method | what it does |
|---|---|
| `Model.simulate_trial()` | one Euler (or RK4) SDE trajectory |
| `Model.simulated_solution(size=n)` | loops `simulate_trial` n times → `Sample` |

This is the same *kind* of computation `sinistra.simulate()` does. But PyDDM's
own docstrings call it *"an outdated method … should be used for comparison
purposes only"* and *"you should never need to use this function"* — it is a
pure-Python per-timestep loop and runs at **~68 trials/sec**. So the two
libraries **do** both expose a real trial simulator, and Part 1 compares them
directly, but be aware it pits sinistra's optimised parallel Rust core against
a helper PyDDM never intends anyone to use in anger.

## Part 1 — raw trial-simulation throughput (apples to apples)

`sinistra.simulate()` vs `Model.simulated_solution()` on equivalent
parameters (drift 1.2, symmetric bound ±0.5, noise 1, Euler, dt = 1e-3).
sinistra is measured at n = 10k / 100k / 1M; PyDDM is measured at small n and
**projected** to those sizes (its rate is flat in n — it is a per-trial
Python loop; n = 1e6 would take ~4 h to actually run).

## Part 2 — time to a fitted estimate (explicitly NOT apples to apples)

`sinistra.fit_sim` (Monte-Carlo summary-statistic search + Nelder–Mead) vs
`pyddm.Model.fit` (numerical PDE + likelihood, differential evolution) on the
same synthetic data set. **Different algorithms** — this measures practical
time-to-answer using each library's own tools, not engine speed. Both land
in the sub-second range and recover the parameters well.

## Notes on fairness

- PyDDM's `paranoid` runtime type-checking is left at its default (on). It
  does not materially change either result.
- `sinistra.simulate()` is parallel (rayon); PyDDM's loop is single-threaded.
  The per-core gap (compiled Rust vs a Python loop) is ~1000×+ regardless, so
  parallelism is not what drives the headline number.
- Fit timings vary run to run (differential evolution is stochastic; machine
  load); the script reports best-of-3 and every run.
