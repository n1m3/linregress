use std::error;
use std::fmt;

/// An error that can occur in this crate.
///
/// Generally this error corresponds to problems with input data or fitting
/// a regression model.
#[derive(Debug, Clone)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind) -> Error {
        Error { kind }
    }

    /// Returns the kind of this error
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InconsistentSlopes {
    output_name_count: usize,
    slope_count: usize,
}

impl InconsistentSlopes {
    pub(crate) fn new(output_name_count: usize, slope_count: usize) -> Self {
        Self {
            output_name_count,
            slope_count,
        }
    }

    pub fn get_output_name_count(&self) -> usize {
        self.output_name_count
    }

    pub fn get_slope_count(&self) -> usize {
        self.slope_count
    }
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// Number of slopes and output names is inconsistent.
    InconsistentSlopes(InconsistentSlopes),
    /// Cannot fit model without data.
    NoData,
    /// Cannot fit model without formula.
    NoFormula,
    /// Given formula is invalid.
    InvalidFormula,
    /// Requested column is not in data. (Column given as String)
    ColumnNotInData(String),
    /// Regressor and regressand dimensions do not match. (Column given as String)
    RegressorRegressandDimensionMismatch(String),
    /// Error while processing the regression data. (Details given as String)
    RegressionDataError(String),
    /// Error while fitting the model. (Details given as String)
    ModelFittingError(String),

    /// Hint that users should not exhaustively mathch on this enum
    ///
    /// This enum may gain additional variants, so we prevent exhaustive matching.
    /// This way adding a new variant won't break existing code.
    #[doc(hidden)]
    __Nonexhaustive,
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match self.kind {
            ErrorKind::InconsistentSlopes(_) => "Number of slopes and output names is inconsistent",
            ErrorKind::NoData => "Cannot fit model without data",
            ErrorKind::NoFormula => "Cannot fit model without formula",
            ErrorKind::ColumnNotInData(_) => "Requested column not in data",
            ErrorKind::InvalidFormula => "Invalid formula",
            ErrorKind::RegressorRegressandDimensionMismatch(_) => {
                "Regressor and regressand dimensions do not match"
            }
            ErrorKind::RegressionDataError(_) => "Error while processing the regression data",
            ErrorKind::ModelFittingError(_) => "Error while fitting the model",
            ErrorKind::__Nonexhaustive => unreachable!(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::InconsistentSlopes(inconsistent_slopes) => write!(
                f,
                "Number of slopes and output names is inconsistent. {} outputs != {} sloped",
                inconsistent_slopes.get_output_name_count(),
                inconsistent_slopes.get_slope_count()
            ),
            ErrorKind::NoData => write!(f, "Cannot fit model without data"),
            ErrorKind::NoFormula => write!(f, "Cannot fit model without formula"),
            ErrorKind::InvalidFormula => write!(
                f,
                "Invalid formula. Expected formula of the form 'y ~ x1 + x2'"
            ),
            ErrorKind::ColumnNotInData(column) => {
                write!(f, "Requested column {} is not in the data", column)
            }
            ErrorKind::RegressorRegressandDimensionMismatch(column) => write!(
                f,
                "Regressor dimensions for {} do not match regressand dimensions",
                column
            ),
            ErrorKind::RegressionDataError(detail) => {
                write!(f, "Error while processing the regression data: {}", detail)
            }
            ErrorKind::ModelFittingError(detail) => {
                write!(f, "Error while fitting the model: {}", detail)
            }
            ErrorKind::__Nonexhaustive => unreachable!(),
        }
    }
}
