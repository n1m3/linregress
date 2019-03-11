# linregress
A Rust library providing an easy to use implementation of ordinary
least squared linear regression with some basic statistics.

## Documentation

[Full API documentation](https://docs.rs/linregress)

## License
This project is licensed under the MIT License.
See LICENSE-MIT for details.

### Third party software
The special functions module contains functions that are based on the
C implementation in the [Cephes library](http://www.netlib.org/cephes/).
They are considered a derivative of the Cephes library that is compatibly licensed.
See LICENSE-THIRD-PARTY for details.

### Example

```rust,no_run
use linregress::FormulaRegressionBuilder;

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
```
