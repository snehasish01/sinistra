//! Throughput benchmark for [`sinistra_core::simulate_n`].
//!
//! Reports trials/sec (criterion "elements" throughput) at
//! n = 1_000 / 10_000 / 100_000 / 1_000_000.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use sinistra_core::{simulate_n, Params};

fn bench_simulate_n(c: &mut Criterion) {
    // A representative, well-behaved parameter set: moderate drift, unbiased
    // start, typical non-decision time.
    let params = Params::new(0.5, 1.0, 0.5, 0.2);

    let mut group = c.benchmark_group("simulate_n");
    group.sample_size(10);

    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| simulate_n(black_box(&params), black_box(n), 42));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_simulate_n);
criterion_main!(benches);
