#![warn(rust_2018_idioms)]
use failure::{bail, err_msg, Error};
use nalgebra::{DMatrix, DVector, RowDVector};
use statrs::distribution::{StudentsT, Univariate};
use statrs::function;
use std::collections::HashMap;
use std::iter;

/// A builder to create and fit linear regression model.
///
/// Given a dataset and a regression formula this builder
/// will produce an ordinary least squared linear regression model.
///
/// The pseudo inverse method is used to fit the model.
///
/// # Usage
///
/// ```
/// use std::collections::HashMap;
/// use linregress::FormulaRegressionBuilder;
///
/// # use failure::Error;
/// # fn main() -> Result<(), Error> {
/// let mut data = HashMap::new();
/// data.insert("inputs".to_string(), vec![1.,2. ,3. , 4.]);
/// data.insert("outputs".to_string(), vec![4., 3., 2., 1.]);
/// let model = FormulaRegressionBuilder::new().data(&data).formula("inputs ~ outputs").fit()?;
/// assert_eq!(model.parameters.intercept, 5.0);
/// assert_eq!(model.parameters.slopes[0], -0.9999999999999993);
/// assert_eq!(model.parameters.slope_names[0], "outputs");
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct FormulaRegressionBuilder {
    data: Option<HashMap<String, Vec<f64>>>,
    formula: Option<String>,
}
impl Default for FormulaRegressionBuilder {
    fn default() -> Self {
        FormulaRegressionBuilder::new()
    }
}
impl FormulaRegressionBuilder {
    /// Create as new FormulaRegressionBuilder with no data or formula set.
    pub fn new() -> Self {
        FormulaRegressionBuilder {
            data: None,
            formula: None,
        }
    }
    pub fn data(mut self, data: &HashMap<String, Vec<f64>>) -> Self {
        self.data = Some(data.to_owned());
        self
    }
    /// Set the formula to use for the regression.
    ///
    /// The expected format is "<regressand> ~ <regressor 1> + <regressor 2>".
    ///
    /// E.g. for a regressand named Y and three regressors named A, B and C
    /// the correct format would be "Y ~ A + B + C".
    ///
    /// Note that there is currently no special support for categorical variables.
    /// So if you have a categorical variable with more than two distinct values
    /// you will need to perform "dummy coding" yourself.
    pub fn formula<T: Into<String>>(mut self, formula: T) -> Self {
        self.formula = Some(formula.into());
        self
    }
    /// Fits the model and returns a [`RegressionModel`] if successful.
    /// You need to set the data with [`data`] and a formula with [`formula`]
    /// before you can use it.
    ///
    /// [`RegressionModel`]: struct.RegressionModel.html
    /// [`data`]: struct.FormulaRegressionBuilder.html#method.data
    /// [`formula`]: struct.FormulaRegressionBuilder.html#method.formula
    pub fn fit(self) -> Result<RegressionModel, Error> {
        let data: Result<_, Error> = self
            .data
            .ok_or_else(|| err_msg("Cannot fit model without data"));
        let formula: Result<_, Error> = self
            .formula
            .ok_or_else(|| err_msg("Cannot fit model without formula"));
        let data = data?;
        let formula = formula?;
        let split_formula: Vec<_> = formula.split('~').collect();
        if split_formula.len() != 2 {
            bail!("Invalid formula. Expected formula of the form 'y ~ x1 + x2'");
        }
        let input = split_formula[0].trim();
        let outputs: Vec<_> = split_formula[1]
            .split('+')
            .map(|x| x.trim())
            .filter(|x| *x != "")
            .collect();
        if outputs.is_empty() {
            bail!("Invalid formula. Expected formula of the form 'y ~ x1 + x2'");
        }
        let input_vector = data
            .get(input)
            .ok_or_else(|| err_msg(format!("{} not found in data", input)))?;
        let input_vector = RowDVector::from_vec(input_vector.to_vec());
        let mut output_matrix = Vec::new();
        // Add column of all ones as the first column of the matrix
        let all_ones_column = iter::repeat(1.).take(input_vector.len());
        output_matrix.extend(all_ones_column);
        // Add each input as a new column of the matrix
        for output in outputs.to_owned() {
            let output_vec = data
                .get(output)
                .ok_or_else(|| err_msg(format!("{} not found in data", output)))?;
            if output_vec.len() != input_vector.len() {
                bail!(format!(
                    "Regressor dimensions for {} do not match regressand dimensions",
                    output
                ));
            }
            output_matrix.extend(output_vec.iter());
        }
        //
        let output_matrix = DMatrix::from_vec(input_vector.len(), outputs.len() + 1, output_matrix);
        let (model_parameter, singular_values, normalized_cov_params) =
            fit_ols_pinv(&input_vector, &output_matrix)?;
        let outputs: Vec<_> = outputs.iter().map(|x| x.to_string()).collect();
        RegressionModel::try_from_regression_parameters(
            &input_vector,
            &output_matrix,
            &outputs,
            &model_parameter,
            &singular_values,
            &normalized_cov_params,
        )
    }
}

/// A fitted regression model
///
#[derive(Debug)]
pub struct RegressionModel {
    pub parameters: RegressionParameters,
    pub se: Vec<f64>,
    pub ssr: f64,
    pub rsquared: f64,
    pub rsquared_adj: f64,
    pub pvalues: Vec<f64>,
    pub residuals: Vec<f64>,
}
impl RegressionModel {
    fn try_from_regression_parameters(
        inputs: &RowDVector<f64>,
        outputs: &DMatrix<f64>,
        output_names: &Vec<String>,
        parameters: &DMatrix<f64>,
        singular_values: &DVector<f64>,
        normalized_cov_params: &DMatrix<f64>,
    ) -> Result<Self, Error> {
        let diag = DMatrix::from_diagonal(&singular_values);
        let rank = &diag.rank(0.0);
        let input_vec: Vec<_> = inputs.iter().cloned().collect();
        let input_matrix = DMatrix::from_vec(inputs.len(), 1, input_vec);
        let residuals = &input_matrix - (outputs * parameters.to_owned());
        let ssr = residuals.dot(&residuals);
        let p = outputs.ncols() - 1;
        let n = inputs.ncols();
        let df_resid = n - rank;
        // TODO XXX add test that catches the differences between this correct version and
        // the old version:
        //  let scale = ssr / ((n - p) as f64);
        let scale = residuals.dot(&residuals) / df_resid as f64;
        let cov_params = normalized_cov_params.to_owned() * scale;
        let se = get_se_from_cov_params(&cov_params)?;
        let centered_input_matrix = subtract_value_from_matrix(&input_matrix, input_matrix.mean());
        let centered_tss = &centered_input_matrix.dot(&centered_input_matrix);
        let rsquared = 1. - (ssr / centered_tss);
        let rsquared_adj = 1. - ((n - 1) as f64 / df_resid as f64 * (1. - rsquared));
        let tvalues: Vec<_> = matrix_as_vec(parameters)
            .iter()
            .zip(matrix_as_vec(&se))
            .map(|(x, y)| x / y)
            .collect();
        let students_t = StudentsT::new(0.0, 1.0, df_resid as f64)?;
        let pvalues: Vec<_> = tvalues
            .iter()
            .cloned()
            .map(|x| (1. - students_t.cdf(x)) * 2.)
            .collect();
        // Convert these from interal Matrix types to user facing types
        let intercept = parameters[0];
        let slopes: Vec<_> = parameters.iter().cloned().skip(1).collect();
        if output_names.len() != slopes.len() {
            bail!("Number of slopes and output names is inconsistent");
        }
        let parameters = RegressionParameters {
            intercept,
            slopes,
            slope_names: output_names.to_vec(),
        };
        let se: Vec<_> = se.iter().cloned().collect();
        let residuals: Vec<_> = residuals.iter().cloned().collect();
        Ok(Self {
            parameters,
            se,
            ssr,
            rsquared,
            rsquared_adj,
            pvalues,
            residuals,
        })
    }
}
/// The intercept and slope(s) of a fitted regression model
///
/// The slopes and names of the regressors are given in the same order.
///
/// You can use this to obtain pairs like this:
///
/// ```
/// use std::collections::HashMap;
/// use linregress::FormulaRegressionBuilder;
///
/// # use failure::Error;
/// # fn main() -> Result<(), Error> {
/// let mut data = HashMap::new();
/// data.insert("Y".to_string(), vec![1.,2. ,3. , 4.]);
/// data.insert("X1".to_string(), vec![4., 3., 2., 1.]);
/// data.insert("X2".to_string(), vec![1., 2., 3., 4.]);
/// let model = FormulaRegressionBuilder::new().data(&data).formula("Y ~ X1 + X2").fit()?;
/// let params = model.parameters;
/// let pairs: Vec<_> = params.slope_names.iter().zip(params.slopes).collect();
/// assert_eq!(pairs[0], (&"X1".to_string(), -0.0370370370370372));
/// assert_eq!(pairs[1], (&"X2".to_string(), 0.9629629629629629));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RegressionParameters {
    pub intercept: f64,
    pub slope_names: Vec<String>,
    pub slopes: Vec<f64>,
}

/// Performs ordinary least squared linear regression using the pseudo inverse method to solve
/// the linear system.
///
/// Returns a tuple of the form `(model_parameter, singular_values, normalized_cov_params)` as `Result`.
fn fit_ols_pinv(
    inputs: &RowDVector<f64>,
    outputs: &DMatrix<f64>,
) -> Result<(DMatrix<f64>, DVector<f64>, DMatrix<f64>), Error> {
    let singular_values = &outputs.to_owned().svd(false, false).singular_values;
    let pinv = outputs
        .to_owned()
        .pseudo_inverse(0.)
        .map_err(|_| err_msg("Taking the pinv of the output matrix failed"))?;
    let normalized_cov_params = &pinv * &pinv.transpose();
    let params = get_sum_of_products(&pinv, &inputs);
    if params.len() < 2 {
        bail!("Invalid parameter matrix");
    }
    Ok((params, singular_values.to_owned(), normalized_cov_params))
}
/// Transforms a matrix into a flat Vec.
fn matrix_as_vec(matrix: &DMatrix<f64>) -> Vec<f64> {
    let mut vector = Vec::new();
    for row_index in 0..matrix.nrows() {
        let row = matrix.row(row_index);
        for i in row.iter() {
            vector.push(*i);
        }
    }
    vector
}
/// Subtracts `value` from all fields in `matrix` and returns the resulting new matrix.
fn subtract_value_from_matrix(matrix: &DMatrix<f64>, value: f64) -> DMatrix<f64> {
    let mut v = Vec::new();
    for row_index in 0..matrix.nrows() {
        let row = matrix.row(row_index);
        for i in row.iter().map(|i| i - value) {
            v.push(i);
        }
    }
    DMatrix::from_vec(matrix.nrows(), matrix.ncols(), v)
}
/// Calculates the standard errors given a models covariate parameters
fn get_se_from_cov_params(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, Error> {
    let mut v = Vec::new();
    for row_index in 0..matrix.ncols() {
        let row = matrix.row(row_index);
        if row_index > row.len() {
            bail!("Matrix is not square");
        }
        v.push(row[row_index].sqrt());
    }
    Ok(DMatrix::from_vec(matrix.ncols(), 1, v))
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
const MAX_GAMMA: f64 = 171.624376956302725;
const MINLOG: f64 = -7.08396418532264106224E2;
const MAXLOG: f64 = 7.09782712893383996843E2;
const MACHEP: f64 = 0.11102230246251565E-15;
const BIG: f64 = 4.503599627370496E15;
const BIG_INVERSE: f64 = 2.22044604925031308085E-16;
/// Power series for incomplete beta integral.
fn pseries(a: f64, b: f64, x: f64) -> f64 {
    assert!(a > 0. && b > 0. && x > 0. && x < 1.);
    let a_inverse = 1. / a;
    let mut u = (1. - b) * x;
    let mut v = u / (a + 1.0);
    let t1 = v;
    let mut t = u;
    let mut n = 2.0;
    let mut s = 0.0;
    let z = MACHEP * a_inverse;
    while v.abs() > z {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += a_inverse;
    u = a * x.ln();
    if (a + b) < MAX_GAMMA && u.abs() < MAXLOG {
        t = 1.0 / beta(a, b);
        s = s * t * x.powf(a);
    } else {
        t = -ln_beta(a, b) + u + s.ln();
        if t < MINLOG {
            s = 0.0;
        } else {
            s = t.exp();
        }
    }
    s
}
/// Returns incomplete beta integral of the arguments, evaluated
/// from zero to x.
///
/// Based on the C implementation in the cephes library (http://www.netlib.org/cephes/)
/// by Stephen L. Moshier
fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    assert!(a > 0. && b > 0.);
    assert!(x >= 0. || x <= 1.0);
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    if b * x <= 1.0 && x <= 0.95 {
        return pseries(a, b, x);
    }
    let mut x = x;
    let mut a = a;
    let mut b = b;
    let w = 1. - x;
    let mut xc = x;
    let mut was_swapped = false;
    // Swap a and b if x is greater than mean
    if x > a / (a + b) {
        was_swapped = true;
        let temp = b;
        b = a;
        a = temp;
        x = w;
        if b * x <= 1.0 && x <= 0.95 {
            return pseries(a, b, x);
        }
    } else {
        xc = w;
    }
    unimplemented!();
}
/// Calculates the value of the beta function for `a` and `b`.
///
/// Returns infinity for a <= 0.0 or b <= 0.0 rather than panic.
fn beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        std::f64::INFINITY
    } else {
        function::beta::beta(a, b)
    }
}
/// Calculates the natural logarithm of the beta function for `a` and `b`.
///
/// Returns infinity for a <= 0.0 or b <= 0.0 rather than panic.
fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        std::f64::INFINITY
    } else {
        function::beta::ln_beta(a, b)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn assert_almost_equal(a: f64, b: f64) {
        if (a - b).abs() > 1.0E-6 {
            panic!("{:?} vs {:?}", a, b);
        }
    }
    #[test]
    fn test_pinv_with_formula_builder() {
        use std::collections::HashMap;
        let inputs = vec![1., 3., 4., 5., 2., 3., 4.];
        let outputs1 = vec![1., 2., 3., 4., 5., 6., 7.];
        let outputs2 = vec![7., 6., 5., 4., 3., 2., 1.];
        let mut data = HashMap::new();
        data.insert("Y".to_string(), inputs);
        data.insert("X1".to_string(), outputs1);
        data.insert("X2".to_string(), outputs2);
        let regression = FormulaRegressionBuilder::new()
            .data(&data)
            .formula("Y ~ X1 + X2")
            .fit()
            .expect("Fitting model failed");

        let model_parameters = vec![0.09523809523809511, 0.5059523809523809, 0.25595238095238104];
        let se = vec![0.015457637291218271, 0.1417242813072997, 0.1417242813072997];
        let ssr = 9.107142857142858;
        let rsquared = 0.16118421052631582;
        let rsquared_adj = -0.006578947368421018;
        let pvalues = vec![
            0.0016390312044176625,
            0.01604408370984789,
            0.13074580446389206,
        ];
        let residuals = vec![
            -1.3928571428571432,
            0.35714285714285676,
            1.1071428571428568,
            1.8571428571428572,
            -1.3928571428571428,
            -0.6428571428571423,
            0.10714285714285765,
        ];
        assert_eq!(regression.parameters.intercept, model_parameters[0]);
        assert_eq!(regression.parameters.slopes[0], model_parameters[1]);
        assert_eq!(regression.parameters.slopes[1], model_parameters[2]);
        assert_eq!(regression.se, se);
        assert_eq!(regression.ssr, ssr);
        assert_eq!(regression.rsquared, rsquared);
        assert_eq!(regression.rsquared_adj, rsquared_adj);
        assert_eq!(regression.pvalues, pvalues);
        assert_eq!(regression.residuals, residuals);
    }
    #[test]
    fn test_beta() {
        assert_eq!(beta(0., 5.), std::f64::INFINITY);
        assert_almost_equal(beta(1., 5.), 0.2);
        assert_eq!(beta(7., 5.), 0.00043290043290043333);
        assert_eq!(beta(7., 2.3456), 0.010153149666293021);
        assert_eq!(beta(-5., 2.3456), std::f64::INFINITY);
        assert_eq!(beta(-0.1, 2.3456), std::f64::INFINITY);
    }
    #[test]
    fn test_ln_beta() {
        assert_eq!(ln_beta(0., 5.), std::f64::INFINITY);
        assert_eq!(ln_beta(1., 5.), -1.609437912434093);
        assert_eq!(ln_beta(7., 5.), -7.745002803515838);
        assert_eq!(ln_beta(7., 2.3456), -4.589971309681085);
        assert_eq!(ln_beta(-5., 2.3456), std::f64::INFINITY);
        assert_eq!(ln_beta(-0.1, 2.3456), std::f64::INFINITY);
    }
    #[test]
    fn test_pseries() {
        assert_almost_equal(pseries(1.0, 2.0, 0.2), 0.36);
    }
    #[test]
    fn test_inc_beta() {
        assert_eq!(inc_beta(1.0, 2.0, 0.0), 0.0);
        assert_eq!(inc_beta(1.0, 2.0, 1.0), 1.0);
        assert_almost_equal(inc_beta(1.0, 2.0, 0.2), 0.36);
        assert_almost_equal(inc_beta(5.0, 2.0, 0.5), 0.109375);
        // b * x > 1
        // x > a / (a + b)
        // a * x <= 1.0 && x <= 0.95
        assert_almost_equal(inc_beta(1.0, 3.0, 0.6), 0.063999999);
        // a * x > 1.0 && x <= 0.95
        dbg!(inc_beta(4.0, 3.0, 0.6));
    }
}
