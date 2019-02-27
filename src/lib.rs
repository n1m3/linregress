/*!
  Crate `linregress` provides an easy to use implementation of ordinary
  least squared linear regression.

  The builder [`FormulaRegressionBuilder`] is used to construct a model from a
  table of data and a R-style formula. Currently only very simple formulae are supported,
  see [`FormulaRegressionBuilder.formula`] for details.

  # Example

  ```
  use linregress::FormulaRegressionBuilder;

  # use failure::Error;
  # fn main() -> Result<(), Error> {
  let y = vec![1.,2. ,3. , 4., 5.];
  let x1 = vec![5., 4., 3., 2., 1.];
  let x2 = vec![729.53, 439.0367, 42.054, 1., 0.];
  let x3 = vec![258.589, 616.297, 215.061, 498.361, 0.];
  let data = vec![("Y", y), ("X1", x1), ("X2", x2), ("X3", x3)];
  let formula = "Y ~ X1 + X2 + X3";
  let model = FormulaRegressionBuilder::new()
      .data(data)
      .formula(formula)
      .fit()?;
  let parameters = model.parameters;
  let standard_errors = model.se;
  let pvalues = model.pvalues;
  assert_eq!(
      parameters.pairs(),
      vec![
          ("X1".to_string(), -0.9999999999999745),
          ("X2".to_string(), 0.00000000000000005637851296924623),
          ("X3".to_string(), 0.00000000000000008283304597789254),
      ]
  );
  assert_eq!(
      standard_errors.pairs(),
      vec![
          ("X1".to_string(), 0.00000000000019226371555402852),
          ("X2".to_string(), 0.0000000000000008718958950659518),
          ("X3".to_string(), 0.0000000000000005323837152041135),
      ]
  );
  assert_eq!(
      pvalues.pairs(),
      vec![
          ("X1".to_string(), 0.00000000000012239888283055414),
          ("X2".to_string(), 0.9588921357097694),
          ("X3".to_string(), 0.9017368322742073),
      ]
  );
  # Ok(())
  # }
  ```

  [`FormulaRegressionBuilder`]: struct.FormulaRegressionBuilder.html
  [`FormulaRegressionBuilder.formula`]: struct.FormulaRegressionBuilder.html#method.formula
*/

#![warn(rust_2018_idioms)]
#![cfg_attr(feature = "unstable", feature(test))]
use failure::{bail, err_msg, Error};
use nalgebra::{DMatrix, DVector, RowDVector};
use std::borrow::Cow;
use std::collections::HashMap;
use std::iter;

mod special_functions;
use special_functions::stdtr;

/// A builder to create and fit a linear regression model.
///
/// Given a dataset and a regression formula this builder
/// will produce an ordinary least squared linear regression model.
///
/// The pseudo inverse method is used to fit the model.
///
/// # Usage
///
/// ```
/// use linregress::FormulaRegressionBuilder;
///
/// # use failure::Error;
/// # fn main() -> Result<(), Error> {
/// let y = vec![1.,2. ,3. , 4.];
/// let x = vec![4., 3., 2., 1.];
/// let data = vec![("Y", y), ("X", x)];
/// let model = FormulaRegressionBuilder::new().data(data).formula("Y ~ X").fit()?;
/// assert_eq!(model.parameters.intercept_value, 5.0);
/// assert_eq!(model.parameters.regressor_values[0], -0.9999999999999993);
/// assert_eq!(model.parameters.regressor_names[0], "X");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
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
    /// Set the data to be used for the regression.
    ///
    /// Any type that implements the [`IntoIterator`] trait can be used for the data.
    /// This could for example be a [`Hashmap`] or a [`Vec`].
    ///
    /// The iterator must consist of tupels of the form `(S, Vec<f64>)` where
    /// `S` is a type that implements `Into<String>`, such as [`String`] or [`str`].
    ///
    /// You can think of this format as the representation of a table of data where
    /// each tuple `(S, Vec<f64>)` represents a column. The `S` is the header or label of the
    /// column and the `Vec<f64>` contains the data of the column.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use linregress::FormulaRegressionBuilder;
    ///
    /// # use failure::Error;
    /// # fn main() -> Result<(), Error> {
    /// let regression_builder = FormulaRegressionBuilder::new().formula("Y ~ X");
    ///
    /// let mut data1 = HashMap::new();
    /// data1.insert("Y", vec![1., 2., 3., 4.]);
    /// data1.insert("X", vec![4., 3., 2., 1.]);
    /// let model1 = regression_builder.to_owned().data(data1).fit()?;
    ///
    /// let y = vec![1., 2., 3., 4.];
    /// let x = vec![4., 3., 2., 1.];
    /// let data2 = vec![("X", x), ("Y", y)];
    /// let model2 = regression_builder.data(data2).fit()?;
    ///
    /// assert_eq!(model1.parameters.regressor_values, model2.parameters.regressor_values);
    /// assert_eq!(model1.parameters.regressor_names, model2.parameters.regressor_names);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`IntoIterator`]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
    /// [`Hashmap`]: https://doc.rust-lang.org/std/collections/struct.HashMap.html
    /// [`Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    /// [`String`]: https://doc.rust-lang.org/std/string/struct.String.html
    /// [`str`]: https://doc.rust-lang.org/std/primitive.str.html
    pub fn data<I, S>(mut self, data: I) -> Self
    where
        I: IntoIterator<Item = (S, Vec<f64>)>,
        S: Into<String>,
    {
        let mut temp = HashMap::new();
        for (key, value) in data {
            temp.insert(key.into(), value);
        }
        self.data = Some(temp);
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
        Self::check_if_data_valid(&data)?;
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
        let output_matrix = DMatrix::from_vec(input_vector.len(), outputs.len() + 1, output_matrix);
        let outputs: Vec<_> = outputs.iter().map(|x| x.to_string()).collect();
        RegressionModel::try_from_matrices_and_regressor_names(input_vector, output_matrix, outputs)
    }
    fn check_if_data_valid(data: &HashMap<String, Vec<f64>>) -> Result<(), Error> {
        for column in data.values() {
            if column.iter().any(|x| !x.is_finite()) {
                bail!("The data contains a non real value (NaN or infinity or negative infinity)");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// A container struct for the regression data. We use this so that `FormulaRegressionBuilder`
/// can implement the `Copy` trait by only keeping a reference to this struct.
/// This struct is obtained using a `RegressionDataBuilder`.
pub struct RegressionData<'a> {
    data: HashMap<Cow<'a, str>, Vec<f64>>,
}

impl<'a> RegressionData<'a> {
    /// Constructs a new `RegressionData` struct from any collection that
    /// implements the `IntoIterator` trait.
    ///
    /// The iterator must consist of tupels of the form `(S, Vec<f64>)` where
    /// `S` is a type that can be converted to a `Cow<'a, str>`.
    ///
    /// `invalid_value_handling` specifies what to do if invalid data is encountered.
    fn new<I, S>(
        data: I,
        invalid_value_handling: InvalidValueHandling,
    ) -> Result<RegressionData<'a>, Error>
    where
        I: IntoIterator<Item = (S, Vec<f64>)>,
        S: Into<Cow<'a, str>>,
    {
        let mut temp = HashMap::new();
        for (key, value) in data {
            temp.insert(key.into(), value);
        }
        if Self::check_if_data_is_valid(&temp) {
            return Ok(Self { data: temp });
        }
        match invalid_value_handling {
            InvalidValueHandling::ReturnError => bail!(
                "The data contains a non real value (NaN or infinity or negative infinity). \
                 If you would like to silently drop these values configure the builder with \
                 InvalidValueHandling::DropInvalid."
            ),
            InvalidValueHandling::DropInvalid => bail!("Not implemented"),
            _ => bail!("Unkown InvalidValueHandling option"),
        }
    }
    fn check_if_data_is_valid(data: &HashMap<Cow<'a, str>, Vec<f64>>) -> bool {
        for column in data.values() {
            if column.iter().any(|x| !x.is_finite()) {
                return false;
            }
        }
        true
    }
}

/// A builder to create a RegressionData struct for use with a `FormulaRegressionBuilder`.
#[derive(Debug, Clone)]
pub struct RegressionDataBuilder {
    handle_invalid_values: InvalidValueHandling,
}

impl Default for RegressionDataBuilder {
    fn default() -> RegressionDataBuilder {
        RegressionDataBuilder {
            handle_invalid_values: InvalidValueHandling::default(),
        }
    }
}

impl RegressionDataBuilder {
    /// Crate a new `RegressionDataBuilder`
    pub fn new() -> Self {
        Self::default()
    }
    /// Configure how to handle non real `f64` values (NaN or infinity or negative infinity).
    ///
    /// The default is `ReturnError`.
    pub fn invalid_value_handling(mut self, setting: InvalidValueHandling) -> Self {
        self.handle_invalid_values = setting;
        self
    }
    /// Build a `RegressionData` struct from the given data.
    ///
    /// Any type that implements the [`IntoIterator`] trait can be used for the data.
    /// This could for example be a [`Hashmap`] or a [`Vec`].
    ///
    /// The iterator must consist of tupels of the form `(S, Vec<f64>)` where
    /// `S` is a type that implements `Into<Cow<str>>`, such as [`String`] or [`str`].
    ///
    /// You can think of this format as the representation of a table of data where
    /// each tuple `(S, Vec<f64>)` represents a column. The `S` is the header or label of the
    /// column and the `Vec<f64>` contains the data of the column.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use linregress::RegressionDataBuilder;
    ///
    /// # use failure::Error;
    /// # fn main() -> Result<(), Error> {
    /// let builder = RegressionDataBuilder::new();
    ///
    /// let mut data1 = HashMap::new();
    /// data1.insert("Y", vec![1., 2., 3., 4.]);
    /// data1.insert("X", vec![4., 3., 2., 1.]);
    /// let regression_data1 = RegressionDataBuilder::new().build_from(data1)?;
    ///
    /// let y = vec![1., 2., 3., 4.];
    /// let x = vec![4., 3., 2., 1.];
    /// let data2 = vec![("X", x), ("Y", y)];
    /// let regression_data2 = RegressionDataBuilder::new().build_from(data2)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`IntoIterator`]: https://doc.rust-lang.org/std/iter/trait.IntoIterator.html
    /// [`Hashmap`]: https://doc.rust-lang.org/std/collections/struct.HashMap.html
    /// [`Vec`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
    /// [`String`]: https://doc.rust-lang.org/std/string/struct.String.html
    /// [`str`]: https://doc.rust-lang.org/std/primitive.str.html
    pub fn build_from<'a, I, S>(self, data: I) -> Result<RegressionData<'a>, Error>
    where
        I: IntoIterator<Item = (S, Vec<f64>)>,
        S: Into<Cow<'a, str>>,
    {
        Ok(RegressionData::new(data, self.handle_invalid_values)?)
    }
}

/// How to proceed if given non real `f64` values (NaN or infinity or negative infinity).
///
/// The default is `ReturnError`.
#[derive(Debug, Clone, Copy)]
pub enum InvalidValueHandling {
    /// Return an error to the caller.
    ReturnError,
    /// Drop the columns containing the invalid values.
    DropInvalid,
    /// Destructuring should not be exhaustive
    #[doc(hidden)]
    __Nonexhaustive,
}

impl Default for InvalidValueHandling {
    fn default() -> InvalidValueHandling {
        InvalidValueHandling::ReturnError
    }
}

/// A fitted regression model.
///
/// Is the result of [`FormulaRegressionBuilder.fit()`].
///
/// If a field has only one value for the model it is given as `f64`.
///
/// Otherwise it is given as a [`RegressionParameters`] struct.
///
///[`RegressionParameters`]: struct.RegressionParameters.html
///[`FormulaRegressionBuilder.fit()`]: struct.FormulaRegressionBuilder.html#method.fit
#[derive(Debug, Clone)]
pub struct RegressionModel {
    /// The models intercept and slopes (also known as betas).
    pub parameters: RegressionParameters,
    /// The standard errors of the parameter estimates.
    pub se: RegressionParameters,
    /// Sum of squared residuals.
    pub ssr: f64,
    /// R-squared of the model.
    pub rsquared: f64,
    /// Adjusted R-squared of the model.
    pub rsquared_adj: f64,
    /// The two-tailed p values for the t-stats of the params.
    pub pvalues: RegressionParameters,
    /// The residuals of the model.
    pub residuals: RegressionParameters,
    ///  A scale factor for the covariance matrix.
    ///
    ///  Note that the square root of `scale` is often
    ///  called the standard error of the regression.
    pub scale: f64,
}
impl RegressionModel {
    fn try_from_matrices_and_regressor_names<I: IntoIterator<Item = String>>(
        inputs: RowDVector<f64>,
        outputs: DMatrix<f64>,
        output_names: I,
    ) -> Result<Self, Error> {
        let low_level_result = fit_ols_pinv(inputs.to_owned(), outputs.to_owned())?;
        let parameters = low_level_result.params;
        let singular_values = low_level_result.singular_values;
        let normalized_cov_params = low_level_result.normalized_cov_params;
        let diag = DMatrix::from_diagonal(&singular_values);
        let rank = &diag.rank(0.0);
        let input_vec: Vec<_> = inputs.iter().cloned().collect();
        let input_matrix = DMatrix::from_vec(inputs.len(), 1, input_vec);
        let residuals = &input_matrix - (outputs * parameters.to_owned());
        let ssr = residuals.dot(&residuals);
        let n = inputs.ncols();
        let df_resid = n - rank;
        if df_resid < 1 {
            bail!("There are not enough residual degrees of freedom to perform statistics on this model");
        }
        let scale = residuals.dot(&residuals) / df_resid as f64;
        let cov_params = normalized_cov_params.to_owned() * scale;
        let se = get_se_from_cov_params(&cov_params)?;
        let centered_input_matrix = subtract_value_from_matrix(&input_matrix, input_matrix.mean());
        let centered_tss = &centered_input_matrix.dot(&centered_input_matrix);
        let rsquared = 1. - (ssr / centered_tss);
        let rsquared_adj = 1. - ((n - 1) as f64 / df_resid as f64 * (1. - rsquared));
        let tvalues: Vec<_> = matrix_as_vec(&parameters)
            .iter()
            .zip(matrix_as_vec(&se))
            .map(|(x, y)| x / y)
            .collect();
        let pvalues: Vec<_> = tvalues
            .iter()
            .cloned()
            .map(|x| stdtr(df_resid as i64, -(x.abs())) * 2.)
            .collect();
        // Convert these from interal Matrix types to user facing types
        let intercept = parameters[0];
        let slopes: Vec<_> = parameters.iter().cloned().skip(1).collect();
        let output_names: Vec<_> = output_names.into_iter().collect();
        if output_names.len() != slopes.len() {
            bail!("Number of slopes and output names is inconsistent");
        }
        let parameters = RegressionParameters {
            intercept_value: intercept,
            regressor_values: slopes,
            regressor_names: output_names.to_vec(),
        };
        let se: Vec<_> = se.iter().cloned().collect();
        let se = RegressionParameters {
            intercept_value: se[0],
            regressor_values: se.iter().cloned().skip(1).collect(),
            regressor_names: output_names.to_vec(),
        };
        let residuals: Vec<_> = residuals.iter().cloned().collect();
        let residuals = RegressionParameters {
            intercept_value: residuals[0],
            regressor_values: residuals.iter().cloned().skip(1).collect(),
            regressor_names: output_names.to_vec(),
        };
        let pvalues = RegressionParameters {
            intercept_value: pvalues[0],
            regressor_values: pvalues.iter().cloned().skip(1).collect(),
            regressor_names: output_names.to_vec(),
        };
        Ok(Self {
            parameters,
            se,
            ssr,
            rsquared,
            rsquared_adj,
            pvalues,
            residuals,
            scale,
        })
    }
}
/// A parameter of a fitted [`RegressionModel`] given for the intercept and each regressor.
///
/// The values and names of the regressors are given in the same order.
///
/// You can obtain name value pairs using [`pairs`].
///
/// [`RegressionModel`]: struct.RegressionModel.html
/// [`pairs`]: struct.RegressionParameters.html#method.pairs
#[derive(Debug, Clone)]
pub struct RegressionParameters {
    pub intercept_value: f64,
    pub regressor_names: Vec<String>,
    pub regressor_values: Vec<f64>,
}
impl RegressionParameters {
    /// Returns the parameters as a Vec of tuples of the form `(name: String, value: f64)`.
    ///
    /// # Usage
    ///
    /// ```
    /// use linregress::FormulaRegressionBuilder;
    ///
    /// # use failure::Error;
    /// # fn main() -> Result<(), Error> {
    /// let y = vec![1.,2. ,3. , 4.];
    /// let x1 = vec![4., 3., 2., 1.];
    /// let x2 = vec![1., 2., 3., 4.];
    /// let data = vec![("Y", y), ("X1", x1), ("X2", x2)];
    /// let model = FormulaRegressionBuilder::new().data(data).formula("Y ~ X1 + X2").fit()?;
    /// let pairs = model.parameters.pairs();
    /// assert_eq!(pairs[0], ("X1".to_string(), -0.0370370370370372));
    /// assert_eq!(pairs[1], ("X2".to_string(), 0.9629629629629629));
    /// # Ok(())
    /// # }
    /// ```
    pub fn pairs(self) -> Vec<(String, f64)> {
        self.regressor_names
            .iter()
            .zip(self.regressor_values)
            .map(|(x, y)| (x.to_owned(), y))
            .collect()
    }
}

/// Result of fitting a low level matrix based model
#[derive(Debug, Clone)]
struct LowLevelRegressionResult {
    params: DMatrix<f64>,
    singular_values: DVector<f64>,
    normalized_cov_params: DMatrix<f64>,
}

/// Performs ordinary least squared linear regression using the pseudo inverse method.
///
/// Returns a tuple `LowLevelRegressionResult`
fn fit_ols_pinv(
    inputs: RowDVector<f64>,
    outputs: DMatrix<f64>,
) -> Result<LowLevelRegressionResult, Error> {
    if inputs.is_empty() {
        bail!("Fitting the model failed because the input vector is empty");
    }
    if outputs.nrows() < 1 || outputs.ncols() < 1 {
        bail!("Fitting the model failed because the output matrix is empty");
    }
    let singular_values = outputs
        .to_owned()
        .try_svd(false, false, std::f64::EPSILON, 0)
        .ok_or_else(|| {
            err_msg("computing the singular-value decomposition of the output matrix failed")
        })?
        .singular_values;
    let pinv = outputs
        .pseudo_inverse(0.)
        .map_err(|_| err_msg("Taking the pinv of the output matrix failed"))?;
    let normalized_cov_params = &pinv * &pinv.transpose();
    let params = get_sum_of_products(&pinv, &inputs);
    if params.len() < 2 {
        bail!("Invalid parameter matrix");
    }
    Ok(LowLevelRegressionResult {
        params,
        singular_values,
        normalized_cov_params,
    })
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
#[cfg(test)]
mod tests {
    use super::*;
    fn assert_almost_equal(a: f64, b: f64) {
        if (a - b).abs() > 1.0E-14 {
            panic!("\n{:?} vs\n{:?}", a, b);
        }
    }
    fn assert_slices_almost_equal(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().cloned().zip(b.iter().cloned()).collect::<Vec<_>>() {
            assert_almost_equal(x, y);
        }
    }
    #[test]
    fn test_pinv_with_formula_builder() {
        use std::collections::HashMap;
        let inputs = vec![1., 3., 4., 5., 2., 3., 4.];
        let outputs1 = vec![1., 2., 3., 4., 5., 6., 7.];
        let outputs2 = vec![7., 6., 5., 4., 3., 2., 1.];
        let mut data = HashMap::new();
        data.insert("Y", inputs);
        data.insert("X1", outputs1);
        data.insert("X2", outputs2);
        let regression = FormulaRegressionBuilder::new()
            .data(data)
            .formula("Y ~ X1 + X2")
            .fit()
            .expect("Fitting model failed");

        let model_parameters = vec![0.09523809523809523, 0.5059523809523809, 0.2559523809523808];
        let se = vec![
            0.015457637291218289,
            0.1417242813072997,
            0.14172428130729975,
        ];
        let ssr = 9.107142857142858;
        let rsquared = 0.16118421052631582;
        let rsquared_adj = -0.006578947368421018;
        let scale = 1.8214285714285716;
        let pvalues = vec![
            0.001639031204417556,
            0.016044083709847945,
            0.13074580446389245,
        ];
        let residuals = vec![
            -1.392857142857142,
            0.3571428571428581,
            1.1071428571428577,
            1.8571428571428577,
            -1.3928571428571423,
            -0.6428571428571423,
            0.10714285714285765,
        ];
        assert_almost_equal(regression.parameters.intercept_value, model_parameters[0]);
        assert_almost_equal(
            regression.parameters.regressor_values[0],
            model_parameters[1],
        );
        assert_almost_equal(
            regression.parameters.regressor_values[1],
            model_parameters[2],
        );
        assert_almost_equal(regression.se.intercept_value, se[0]);
        assert_slices_almost_equal(&regression.se.regressor_values, &se[1..]);
        assert_almost_equal(regression.ssr, ssr);
        assert_almost_equal(regression.rsquared, rsquared);
        assert_almost_equal(regression.rsquared_adj, rsquared_adj);
        assert_almost_equal(regression.pvalues.intercept_value, pvalues[0]);
        assert_slices_almost_equal(&regression.pvalues.regressor_values, &pvalues[1..]);
        assert_almost_equal(regression.residuals.intercept_value, residuals[0]);
        assert_slices_almost_equal(&regression.residuals.regressor_values, &residuals[1..]);
        assert_eq!(regression.scale, scale);
    }
    #[test]
    fn test_invalid_input_empty_matrix() {
        let y = vec![];
        let x1 = vec![];
        let x2 = vec![];
        let data = vec![("Y", y), ("X1", x1), ("X2", x2)];
        let model = FormulaRegressionBuilder::new()
            .data(data)
            .formula("Y ~ X1 + X2")
            .fit();
        assert_eq!(model.is_ok(), false);
    }
    #[test]
    fn test_invalid_input_wrong_shape() {
        let y = vec![1., 2., 3.];
        let x1 = vec![1., 2., 3.];
        let x2 = vec![1., 2.];
        let data = vec![("Y", y), ("X1", x1), ("X2", x2)];
        let model = FormulaRegressionBuilder::new()
            .data(data)
            .formula("Y ~ X1 + X2")
            .fit();
        assert_eq!(model.is_ok(), false);
    }
    #[test]
    fn test_invalid_input_nan() {
        let y1 = vec![1., 2., 3., 4.];
        let x1 = vec![1., 2., 3., std::f64::NAN];
        let data1 = vec![("Y", y1), ("X", x1)];
        let y2 = vec![1., 2., 3., std::f64::NAN];
        let x2 = vec![1., 2., 3., 4.];
        let data2 = vec![("Y", y2), ("X", x2)];
        let model1 = FormulaRegressionBuilder::new()
            .data(data1)
            .formula("Y ~ X")
            .fit();
        let model2 = FormulaRegressionBuilder::new()
            .data(data2)
            .formula("Y ~ X")
            .fit();
        assert!(model1.is_err());
        assert!(model2.is_err());
    }
    #[test]
    fn test_invalid_input_infinity() {
        let y1 = vec![1., 2., 3., 4.];
        let x1 = vec![1., 2., 3., std::f64::INFINITY];
        let data1 = vec![("Y", y1), ("X", x1)];
        let y2 = vec![1., 2., 3., std::f64::NEG_INFINITY];
        let x2 = vec![1., 2., 3., 4.];
        let data2 = vec![("Y", y2), ("X", x2)];
        let model1 = FormulaRegressionBuilder::new()
            .data(data1)
            .formula("Y ~ X")
            .fit();
        let model2 = FormulaRegressionBuilder::new()
            .data(data2)
            .formula("Y ~ X")
            .fit();
        assert!(model1.is_err());
        assert!(model2.is_err());
    }
}
#[cfg(all(feature = "unstable", test))]
mod bench {
    use super::*;
    extern crate test;
    use test::Bencher;
    #[bench]
    fn bench(b: &mut Bencher) {
        let y = vec![1., 2., 3., 4., 5.];
        let x1 = vec![5., 4., 3., 2., 1.];
        let x2 = vec![729.53, 439.0367, 42.054, 1., 0.];
        let x3 = vec![258.589, 616.297, 215.061, 498.361, 0.];
        let data = vec![("Y", y), ("X1", x1), ("X2", x2), ("X3", x3)];
        let formula = "Y ~ X1 + X2 + X3";
        b.iter(|| {
            FormulaRegressionBuilder::new()
                .data(data.to_owned())
                .formula(formula)
                .fit()
        });
    }
}
