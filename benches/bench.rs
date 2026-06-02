use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_sc_dendrogram::{CorMethod, GroupMeans, Method, compute};

fn make(g: usize, f: usize) -> GroupMeans {
    let categories = (0..g).map(|i| format!("g{i:03}")).collect();
    let means = (0..g)
        .map(|i| {
            (0..f)
                .map(|j| ((i * 31 + j * 17) % 97) as f64 * 0.1)
                .collect()
        })
        .collect();
    GroupMeans { categories, means }
}

fn bench(c: &mut Criterion) {
    let gm = make(40, 2000);
    c.bench_function("dendrogram_40x2000_pearson_complete", |b| {
        b.iter(|| compute(&gm, CorMethod::Pearson, Method::Complete))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
