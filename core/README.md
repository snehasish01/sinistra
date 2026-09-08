# sinistra-core

The simulation and parameter-recovery core of [sinistra][repo] — a
drift-diffusion model (DDM) toolkit for cognitive-science research.

```rust
use sinistra_core::{Params, simulate_n, ez_diffusion, fit_simulation, default_fit_guess};

let truth = Params::new(1.2, 1.0, 0.5, 0.25); // drift, boundary, start, non-decision time
let trials = simulate_n(&truth, 100_000, 42); // parallel, deterministic for a given seed

let ez = ez_diffusion(&trials).unwrap();      // closed-form (Wagenmakers et al., 2007)
let sim = fit_simulation(&trials, default_fit_guess(&trials), 200).unwrap(); // simulation-based
```

- `simulate_trial` / `simulate_n` — Euler–Maruyama diffusion, `dt = 1 ms`,
  10 s cap; `simulate_n` runs across all cores and is bit-reproducible for a
  seed (per-trial ChaCha8 cipher streams).
- `ez_diffusion` — closed-form inversion of `(accuracy, mean RT, RT variance)`.
- `fit_simulation` — Nelder–Mead search that re-simulates the model at each
  candidate; no closed-form approximation, so no discretization bias.
- `summary_stats` — the one `(Pc, MRT, VRT)` reduction both estimators share.

Full write-up, benchmarks, and a real-data analysis: [the repository][repo].

[repo]: https://github.com/snehasish01/sinistra

## License

MIT.
