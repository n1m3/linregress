#![warn(rust_2018_idioms)]
use failure::{bail, err_msg, Error};
use nalgebra::{DMatrix, RowDVector};

pub fn ols_qr(inputs: &RowDVector<f64>, outputs: &DMatrix<f64>) -> Result<(f64, f64), Error> {
    let qr = outputs.to_owned().qr();
    let (q, r) = (qr.q(), qr.r());
    let _normalized_cov_params = &r.tr_mul(&r).pseudo_inverse(0.);
    let _singular_values = q.to_owned().svd(false, false);
    let effects = get_sum_of_products(&q.transpose(), &inputs);
    let result = r.qr().solve(&effects).ok_or(err_msg("Solving failed"))?;
    if result.len() < 2 {
        bail!("Invalid result matrix");
    }
    let slope = result[0];
    let intercept = result[1];
    Ok((slope, intercept))
}
fn get_sum_of_products(matrix: &DMatrix<f64>, vector: &RowDVector<f64>) -> DMatrix<f64> {
    let mut v: Vec<f64> = Vec::new();
    for row_index in 0..matrix.nrows() {
        let row = matrix.row(row_index);
        let mut sum = 0.;
        for (x, y) in row.iter().zip(vector.iter()) {
            sum += x * y;
        }
        v.push(sum);
    }
    DMatrix::from_vec(matrix.nrows(), 1, v)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ols_qr() {
        let inputs = RowDVector::from_vec(vec![1., 3., 4., 5., 2., 3., 4.]);
        #[rustfmt::skip]
        let outputs = DMatrix::from_vec(7,2,
                                        vec![
                                        1., 1., 1., 1., 1., 1., 1.,
                                        1., 2., 3., 4., 5., 6., 7.]);
        let (slope, intercept) = ols_qr(&inputs, &outputs).expect("Solving failed!");
        assert_eq!((slope, intercept), (2.1428571428571423, 0.25000000000000006));

    }
}
