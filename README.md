# sinistra

A drift-diffusion model (DDM) simulator for cognitive science research.

> Status: **phase 0/1** — simulation core + CLI only. No parameter fitting,
> no Python bindings yet.

## Workspace layout

| Crate  | Kind    | Contents                                             |
| ------ | ------- | --------------------------------------------------- |
| `core` | library | `Params`, `Trial`, `simulate_trial`, `simulate_n`   |
| `cli`  | binary  | `sinistra` — thin command-line wrapper around `core` |

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

## Development

```sh
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
cargo bench -p sinistra-core
```

## License

MIT — see [LICENSE](LICENSE).
