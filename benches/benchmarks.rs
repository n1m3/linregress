use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use linregress::*;

fn bench(c: &mut Criterion) {
    let y = vec![1., 2., 3., 4., 5.];
    let x1 = vec![5., 4., 3., 2., 1.];
    let x2 = vec![729.53, 439.0367, 42.054, 1., 0.];
    let x3 = vec![258.589, 616.297, 215.061, 498.361, 0.];
    let data = vec![("Y", y), ("X1", x1), ("X2", x2), ("X3", x3)];
    let data = RegressionDataBuilder::new().build_from(data).unwrap();
    let formula = "Y ~ X1 + X2 + X3";
    let input = (data, formula);
    let mut group = c.benchmark_group("Linregress");
    group.bench_with_input(
        BenchmarkId::new("with_stats", "formula"),
        &input,
        |b, (data, formula)| {
            b.iter(|| {
                FormulaRegressionBuilder::new()
                    .data(data)
                    .formula(*formula)
                    .fit()
                    .unwrap();
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("without_stats", "formula"),
        &input,
        |b, (data, formula)| {
            b.iter(|| {
                FormulaRegressionBuilder::new()
                    .data(data)
                    .formula(*formula)
                    .fit_without_statistics()
                    .unwrap();
            });
        },
    );
    let columns = ("Y", ["X1", "X2", "X3"]);
    let (data, _formula) = input;
    let input = (data, columns);
    group.bench_with_input(
        BenchmarkId::new("with_stats", "data_columns"),
        &input,
        |b, (data, columns)| {
            b.iter(|| {
                FormulaRegressionBuilder::new()
                    .data(data)
                    .data_columns(columns.0, columns.1)
                    .fit()
                    .unwrap();
            });
        },
    );
    group.bench_with_input(
        BenchmarkId::new("without_stats", "data_columns"),
        &input,
        |b, (data, columns)| {
            b.iter(|| {
                FormulaRegressionBuilder::new()
                    .data(data)
                    .data_columns(columns.0, columns.1)
                    .fit_without_statistics()
                    .unwrap();
            });
        },
    );
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
