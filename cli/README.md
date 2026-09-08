# sinistra-cli

Command-line front end for [`sinistra-core`][repo] — simulate a drift-diffusion
model and recover its parameters from choice/response-time CSVs.

```sh
cargo install sinistra-cli

# Simulate 10k trials -> CSV (columns: choice = upper|lower, rt = seconds)
sinistra simulate --drift 1.2 --boundary 1.0 --start 0.5 --t0 0.3 \
  --n 10000 --seed 42 --out trials.csv

# Recover parameters — closed form (near-instant)
sinistra fit-ez --input trials.csv

# Recover parameters — simulation search, warm-started from EZ
sinistra fit-sim --input trials.csv --max-iters 150
```

`simulate` writes only completed trials; any that hit the 10 s simulation cap
are dropped and the count is reported on stderr.

Full write-up and library docs: [the repository][repo].

[repo]: https://github.com/snehasish01/sinistra

## License

MIT.
