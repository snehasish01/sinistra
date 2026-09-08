# sinistra

A drift-diffusion model (DDM) simulator for cognitive science research.

> Status: **phase 0/1** — simulator, plus two parameter estimators
> (closed-form EZ-diffusion and simulation-based Nelder–Mead). No Python
> bindings yet.

## Workspace layout

| Crate  | Kind    | Contents                                                                        |
| ------ | ------- | ------------------------------------------------------------------------------ |
| `core` | library | `Params`, `Trial`, `simulate_trial`, `simulate_n`, `summary_stats`, `ez_diffusion`, `fit_simulation` |
| `cli`  | binary  | `sinistra` — thin command-line wrapper around `core`                            |

## The model

Each trial integrates a 1-D diffusion process with the Euler–Maruyama scheme
(`dt = 0.001 s`):

```text
dx = drift_rate * dt + noise_sd * sqrt(dt) * N(0, 1)
```

starting at `starting_point * boundary_separation` and running until `x`
reaches `0` (lower response) or `boundary_separation` (upper response).
`non_decision_time` is added to the elapsed time to give the response time.
A trial is force-terminated after `MAX_SIM_TIME = 10 s` of simulated time to
guard against degenerate parameters.

`simulate_n` runs trials in parallel with rayon and is deterministic given a
seed: trial `i` uses a ChaCha8 RNG on cipher stream `i`, independent of
thread count.

## Usage

```sh
cargo run -p sinistra-cli -- simulate \
  --drift 1.2 --boundary 1.0 --start 0.5 --t0 0.3 \
  --n 10000 --seed 42 --out trials.csv
```

Output CSV has columns `choice` (`upper` / `lower`) and `rt` (seconds).
Trials that hit the `MAX_SIM_TIME` cap are dropped before writing; the count
is reported on stderr (`excluded N timed-out trials (of TOTAL)`), so the CSV
only ever holds completed trials.

Recover parameters from such a CSV with EZ-diffusion (Wagenmakers, van der
Maas & Grasman, 2007):

```sh
cargo run -p sinistra-cli -- fit-ez --input trials.csv
```

EZ assumes an unbiased start point and no across-trial parameter variability;
it recovers `drift_rate`, `boundary_separation`, and `non_decision_time`.

Or fit by simulation — Nelder–Mead (via `argmin`) minimising the gap between
the data's `(accuracy, mean RT, RT variance)` and those same statistics from a
fresh `simulate_n` run at each candidate. It warm-starts from the EZ estimate:

```sh
cargo run -p sinistra-cli -- fit-sim --input trials.csv --max-iters 150
```

Both estimators reduce data through the one shared `summary_stats` helper.
The simulation fit is slower (≈0.3–2 s vs microseconds) but makes no
closed-form approximation, so it carries no discretization bias.

## Development

`[profile.test]` is set to `opt-level = 3` — the suite runs millions of
Monte-Carlo trials and is unusably slow unoptimized.

```sh

```sh
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo bench -p sinistra-core
```

## License

MIT — see [LICENSE](LICENSE).
