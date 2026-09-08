# sinistra (Python bindings)

PyO3 bindings for `sinistra-core`, built with [maturin](https://www.maturin.rs/).
The extension targets the Python **abi3** limited API (`abi3-py39`), so one
wheel runs on CPython 3.9+.

## Layout

```
py/
├── Cargo.toml              cdylib + rlib, pyo3 (abi3-py39)
├── pyproject.toml          maturin build backend
├── src/lib.rs              the #[pymodule]
├── python/sinistra/        pure-Python package (re-exports the compiled ext)
├── tests/test_smoke.py     pytest binding smoke test
└── benches/marshalling.py  simulate() throughput (not pytest)
```

## Build / develop

```sh
cd py
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pytest

# Compile the extension and install it into the active venv (editable-ish):
maturin develop                # debug
maturin develop --release      # optimized — use this before benchmarking or
                               # running fit_sim on real data

pytest                         # run the smoke test

# Produce a distributable wheel (lands in ../target/wheels/):
maturin build --release
```

## API

Trial data crosses as **two NumPy arrays** — `choices` (`bool`) and `rts`
(`float64`) — not a list of tuples: at large `n` the per-trial object
marshalling of a Python list dominates, whereas each array is one buffer copy.

Parameters and fit results are plain `dict`s with keys `drift_rate`,
`boundary_separation`, `starting_point`, `non_decision_time`, `noise_sd`
(`fit_sim` adds `iterations`, `final_cost`).

```python
import sinistra

choices, rts, n_excluded = sinistra.simulate(
    drift=1.2, boundary=1.0, start=0.5, t0=0.25, n=1_000_000, seed=42
)
# choices : np.ndarray[bool]     -> True == hit upper boundary
# rts     : np.ndarray[float64]  -> response time, seconds
# n_excluded : int               -> trials that hit the 10 s cap, dropped from both arrays

ez  = sinistra.fit_ez(choices, rts)                          # -> dict
sim = sinistra.fit_sim(choices, rts, initial_guess=ez)       # -> dict + iterations, final_cost
```

`fit_ez` / `fit_sim` raise `sinistra.SinistraError` (a `ValueError` subclass)
with a specific message when the data cannot be fitted.

### Throughput

`benches/marshalling.py` (needs a `--release` build) times `simulate()` end to
end. At `n = 1_000_000` it runs at ~1.52M trials/sec — matching the core
`simulate_n` criterion baseline from phase 1, i.e. the NumPy boundary adds no
measurable overhead at scale.
