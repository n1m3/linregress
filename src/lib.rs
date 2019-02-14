#![warn(rust_2018_idioms)]
use failure::{bail, err_msg, Error};
use nalgebra::{DMatrix, RowDVector};

/// Performs a linear regression.
///
/// Peforms a ordinary least squared linear regression using the pseudo inverse method to solve the linear system.
/// This method supports multiple linear regression.
/// If successful it returns a `Vec` of the form `vec![slope1, slope2, ..., intercept]`.
///
pub fn ols_pinv(inputs: &RowDVector<f64>, outputs: &DMatrix<f64>) -> Result<Vec<f64>, Error> {
    let singular_values = &outputs.to_owned().svd(false, false).singular_values;
    let diag = DMatrix::from_diagonal(&singular_values);
    let _rank = &diag.rank(0.0);
    let pinv = outputs
        .to_owned()
        .pseudo_inverse(0.)
        .map_err(|_| err_msg("Taking the pinv of the output matrix failed"))?;
    let _normalized_cov_params = &pinv * &pinv.transpose();
    let result = get_sum_of_products(&pinv, &inputs);
    if result.len() < 2 {
        bail!("Invalid result matrix");
    }
    let result: Vec<f64> = result.iter().cloned().collect();
    Ok(result)
}

/// Performs a linear regression.
///
/// Peforms a ordinary least squared linear regression using the QR method to solve the linear system.
/// This method does not support multile linear regression.
/// If successful it returns a tuple of the form `(slope, intercept)`.
///
pub fn ols_qr(inputs: &RowDVector<f64>, outputs: &DMatrix<f64>) -> Result<(f64, f64), Error> {
    let qr = outputs.to_owned().qr();
    let (q, r) = (qr.q(), qr.r());
    let _normalized_cov_params = &r.tr_mul(&r).pseudo_inverse(0.);
    let _singular_values = q.to_owned().svd(false, false);
    let effects = get_sum_of_products(&q.transpose(), &inputs);
    let result = r
        .qr()
        .solve(&effects)
        .ok_or_else(|| err_msg("Solving failed"))?;
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
    use math::round;
    #[test]
    fn test_ols_qr() {
        let inputs = RowDVector::from_vec(vec![1., 3., 4., 5., 2., 3., 4.]);
        #[rustfmt::skip]
        let outputs = DMatrix::from_vec(7,2,
                                        vec![
                                        1., 1., 1., 1., 1., 1., 1.,
                                        1., 2., 3., 4., 5., 6., 7.]);
        let (slope, intercept) = ols_qr(&inputs, &outputs).expect("Solving failed!");
        let slope = round::half_up(slope, 8);
        let intercept = round::half_up(intercept, 2);
        assert_eq!((slope, intercept), (2.14285714, 0.25));
    }
    #[test]
    fn test_ols_pinv_single_regession() {
        let inputs = RowDVector::from_vec(vec![1., 3., 4., 5., 2., 3., 4.]);
        #[rustfmt::skip]
        let outputs = DMatrix::from_vec(7,2,
                                        vec![
                                        1., 1., 1., 1., 1., 1., 1.,
                                        1., 2., 3., 4., 5., 6., 7.]);
        let params = ols_pinv(&inputs, &outputs).expect("Solving failed!");
        let slope = round::half_up(params[0], 8);
        let intercept = round::half_up(params[1], 2);
        assert_eq!((slope, intercept), (2.14285714, 0.25));
    }
    #[test]
    fn test_ols_pinv_multiple_regression() {
        let inputs = RowDVector::from_vec(vec![1., 3., 4., 5., 2., 3., 4.]);
        #[rustfmt::skip]
        let outputs = DMatrix::from_vec(7,3,
                                        vec![
                                        1., 1., 1., 1., 1., 1., 1.,
                                        1., 2., 3., 4., 5., 6., 7.,
                                        7., 6., 5., 4., 3., 2., 1.]);
        let params = ols_pinv(&inputs, &outputs).expect("Solving failed!");
        let slope = round::half_up(params[0], 8);
        let slope2 = round::half_up(params[1], 8);
        let intercept = round::half_up(params[2], 8);
        assert_eq!(
            (slope, slope2, intercept),
            (0.0952381, 0.50595238, 0.25595238)
        );
    }
}
