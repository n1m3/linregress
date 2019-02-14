#![warn(rust_2018_idioms)]
use failure::Error;
use nalgebra::{RowDVector, DMatrix};


fn main() {
    let inputs = RowDVector::from_vec(vec![1., 3., 4., 5., 2., 3., 4.]);
    let outputs = DMatrix::from_vec(7,2,
                                    vec![
                                    1., 1., 1., 1., 1., 1., 1.,
                                    1., 2., 3., 4., 5., 6., 7.]);
    let qr = outputs.to_owned().qr();
    let (q, r) = (qr.q(), qr.r());
    let normalized_cov_params = &r.tr_mul(&r).pseudo_inverse(0.);
    let singular_values = q.to_owned().svd(false, false);
    let effects = get_sum_of_products(&q.transpose(), &inputs);
    dbg!(effects);
}
fn get_sum_of_products(matrix: &DMatrix<f64>, vector: &RowDVector<f64>) -> Result<DMatrix<f64>, Error> {
    let mut v: Vec<f64> = Vec::new();
    for row_index in 0..matrix.nrows() {
        let row = matrix.row(row_index);
        let mut sum = 0.;
        for (x, y) in row.iter().zip(vector.iter()) {
            sum += x * y;
        }
        v.push(sum);
    }
    let result: DMatrix<f64> = DMatrix::from_vec(matrix.nrows(), 1, v);
    Ok(result)
}
