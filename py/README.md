# sinistra

Fast **drift-diffusion model** (DDM) simulation and parameter recovery,
powered by a Rust core.

- **`simulate(...)`** — generate choice + response-time data from a DDM.
  ~1.5M trials/sec, returned as NumPy arrays.
- **`fit_ez(...)`** — closed-form EZ-diffusion parameter recovery
  (Wagenmakers, van der Maas & Grasman, 2007). Microseconds.
- **`fit_sim(...)`** — simulation-based recovery: a Nelder–Mead search that
  re-simulates the model at each candidate and matches its summary statistics
  to the data. Sub-second, and makes no closed-form approximation.

## Install

```sh
pip install sinistra
```

Wheels are built against the CPython limited API (abi3), so one wheel covers
CPython 3.9+. `numpy` is the only runtime dependency.

## Usage

```python
import sinistra

# Simulate 1,000,000 trials
choices, rts, n_excluded = sinistra.simulate(
    drift=1.2, boundary=1.0, start=0.5, t0=0.25, n=1_000_000, seed=42
)
# choices    : np.ndarray[bool]     — True == upper boundary
# rts        : np.ndarray[float64]  — response time in seconds
# n_excluded : int                  — trials that hit the 10 s cap, dropped from both arrays

# Recover the parameters two independent ways
ez  = sinistra.fit_ez(choices, rts)
sim = sinistra.fit_sim(choices, rts, initial_guess=ez, max_iters=200)

print(ez)
# {'drift_rate': 1.20, 'boundary_separation': 1.04, 'starting_point': 0.5,
#  'non_decision_time': 0.25, 'noise_sd': 1.0}
```

### API

| function | returns |
| --- | --- |
| `simulate(drift, boundary, start, t0, n, seed, noise_sd=1.0)` | `(choices, rts, n_excluded)` |
| `fit_ez(choices, rts)` | params `dict` |
| `fit_sim(choices, rts, initial_guess=None, max_iters=200)` | params `dict` + `iterations`, `final_cost` |

Trial data crosses the boundary as **two NumPy arrays** — `choices` (`bool`,
`True` == upper boundary) and `rts` (`float64`, seconds) — not a list of
tuples, so large-`n` calls stay cheap.

Parameter dicts always carry the keys `drift_rate`, `boundary_separation`,
`starting_point`, `non_decision_time`, `noise_sd`. `initial_guess` accepts a
dict of the same shape (e.g. the output of `fit_ez`).

Data that cannot be fitted — too few trials, zero RT variance, chance-level
accuracy — raises `sinistra.SinistraError` (a subclass of `ValueError`) with a
specific message.

## The model

Each trial integrates `dx = drift_rate·dt + noise_sd·√dt·N(0,1)`
(Euler–Maruyama, `dt = 1 ms`) from `starting_point · boundary_separation`
until it reaches `0` or `boundary_separation`; `non_decision_time` is added to
the crossing time. Trials are capped at 10 s of simulated time. `noise_sd` is
conventionally fixed at `1.0`; `fit_ez` and `fit_sim` recover `drift_rate`,
`boundary_separation` and `non_decision_time`.

`simulate()` is parallel and deterministic given a seed (each trial draws from
its own ChaCha8 cipher stream, independent of thread count).

## Building from source

```sh
pip install maturin
maturin develop --release      # build + install into the active virtualenv
# or:  maturin build --release  # produce a wheel in target/wheels/
```

A Rust toolchain is required (https://rustup.rs).

## License

MIT. Source and full documentation: https://github.com/snehasish01/sinistra
