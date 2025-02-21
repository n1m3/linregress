// This file contains code adapted from the statrs library (https://github.com/statrs-dev/statrs/)
// licensed under the MIT License:
//
// MIT License
//
// Copyright (c) 2016 Michael Ma
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std::f64::consts::{FRAC_1_SQRT_2, LN_2, PI};

use super::*;
use crate::assert_almost_eq;

fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_reg(a, b, x).unwrap()
}

#[test]
fn test_students_t_cdf() {
    assert_eq!(students_t_cdf(0., 1).unwrap(), 0.5);
    assert_eq!(students_t_cdf(0., 2).unwrap(), 0.5);
    assert_almost_eq!(students_t_cdf(1., 1).unwrap(), 0.75, 1e-15);
    assert_almost_eq!(students_t_cdf(-1., 1).unwrap(), 0.25, 1e-15);
    assert_almost_eq!(students_t_cdf(2., 1).unwrap(), 0.852416382349567, 1e-15);
    assert_almost_eq!(students_t_cdf(-2., 1).unwrap(), 0.147583617650433, 1e-15);
    assert_almost_eq!(students_t_cdf(1., 2).unwrap(), 0.788675134594813, 1e-15);
    assert_almost_eq!(students_t_cdf(-1., 2).unwrap(), 0.211324865405187, 1e-15);
    assert_almost_eq!(students_t_cdf(2., 2).unwrap(), 0.908248290463863, 1e-15);
    assert_almost_eq!(students_t_cdf(-2., 2).unwrap(), 0.091751709536137, 1e-15);
}

#[test]
fn test_beta_reg() {
    assert_almost_eq!(beta_reg(0.5, 0.5, 0.5), 0.5, 1e-15);
    assert_eq!(beta_reg(0.5, 0.5, 1.0), 1.0);
    assert_almost_eq!(beta_reg(1.0, 0.5, 0.5), 0.292_893_218_813_452_5, 1e-15);
    assert_eq!(beta_reg(1.0, 0.5, 1.0), 1.0);
    assert_almost_eq!(beta_reg(2.5, 0.5, 0.5), 0.075_586_818_421_612_44, 1e-16);
    assert_eq!(beta_reg(2.5, 0.5, 1.0), 1.0);
    assert_almost_eq!(beta_reg(0.5, 1.0, 0.5), FRAC_1_SQRT_2, 1e-15);
    assert_eq!(beta_reg(0.5, 1.0, 1.0), 1.0);
    assert_almost_eq!(beta_reg(1.0, 1.0, 0.5), 0.5, 1e-15);
    assert_eq!(beta_reg(1.0, 1.0, 1.0), 1.0);
    assert_almost_eq!(beta_reg(2.5, 1.0, 0.5), 0.176_776_695_296_636_9, 1e-15);
    assert_eq!(beta_reg(2.5, 1.0, 1.0), 1.0);
    assert_eq!(beta_reg(0.5, 2.5, 0.5), 0.924_413_181_578_387_6);
    assert_eq!(beta_reg(0.5, 2.5, 1.0), 1.0);
    assert_almost_eq!(beta_reg(1.0, 2.5, 0.5), 0.823_223_304_703_363_1, 1e-15);
    assert_eq!(beta_reg(1.0, 2.5, 1.0), 1.0);
    assert_almost_eq!(beta_reg(2.5, 2.5, 0.5), 0.5, 1e-15);
    assert_eq!(beta_reg(2.5, 2.5, 1.0), 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_a_lte_0() {
    beta_reg(0.0, 1.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_b_lte_0() {
    beta_reg(1.0, 0.0, 1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_x_lt_0() {
    beta_reg(1.0, 1.0, -1.0);
}

#[test]
#[should_panic]
fn test_beta_reg_x_gt_1() {
    beta_reg(1.0, 1.0, 2.0);
}

#[test]
fn test_checked_beta_reg_a_lte_0() {
    assert!(checked_beta_reg(0.0, 1.0, 1.0).is_none());
}

#[test]
fn test_checked_beta_reg_b_lte_0() {
    assert!(checked_beta_reg(1.0, 0.0, 1.0).is_none());
}

#[test]
fn test_checked_beta_reg_x_lt_0() {
    assert!(checked_beta_reg(1.0, 1.0, -1.0).is_none());
}

#[test]
fn test_checked_beta_reg_x_gt_1() {
    assert!(checked_beta_reg(1.0, 1.0, 2.0).is_none());
}
#[test]
fn test_ln_gamma() {
    assert!(ln_gamma(f64::NAN).is_nan());
    assert_eq!(ln_gamma(1.000001e-35), 80.590_477_254_792_1);
    assert_almost_eq!(ln_gamma(1.000001e-10), 23.025_849_929_883_236, 1e-14);
    assert_almost_eq!(ln_gamma(1.000001e-5), 11.512_918_692_890_553, 1e-14);
    assert_eq!(ln_gamma(1.000001e-2), 4.599_478_872_433_667);
    assert_almost_eq!(ln_gamma(0.1), 2.252_712_651_734_206, 1e-14);
    assert_almost_eq!(ln_gamma(1.0 - 1.0e-14), 5.772_156_649_015_411e-15, 1e-15);
    assert_almost_eq!(ln_gamma(1.0), 0.0, 1e-15);
    assert_almost_eq!(ln_gamma(1.0 + 1.0e-14), -5.772_156_649_015_246e-15, 1e-15);
    assert_almost_eq!(ln_gamma(1.5), -0.120_782_237_635_245_22, 1e-14);
    assert_almost_eq!(
        ln_gamma(f64::consts::PI / 2.0),
        -0.115_903_800_845_502_42,
        1e-14
    );
    assert_eq!(ln_gamma(2.0), 0.0);
    assert_almost_eq!(ln_gamma(2.5), 0.284_682_870_472_919_2, 1e-13);
    assert_almost_eq!(ln_gamma(3.0), LN_2, 1e-14);
    assert_almost_eq!(ln_gamma(PI), 0.827_694_592_323_437_1, 1e-13);
    assert_almost_eq!(ln_gamma(3.5), 1.200_973_602_347_074_3, 1e-14);
    assert_almost_eq!(ln_gamma(4.0), 1.791_759_469_228_055, 1e-14);
    assert_almost_eq!(ln_gamma(4.5), 2.453_736_570_842_442_3, 1e-13);
    assert_almost_eq!(ln_gamma(5.0 - 1.0e-14), 3.178_053_830_347_930_7, 1e-14);
    assert_almost_eq!(ln_gamma(5.0), 3.178_053_830_347_945_8, 1e-14);
    assert_almost_eq!(ln_gamma(5.0 + 1.0e-14), 3.178_053_830_347_961, 1e-13);
    assert_almost_eq!(ln_gamma(5.5), 3.957_813_967_618_716_5, 1e-14);
    assert_almost_eq!(ln_gamma(10.1), 13.027_526_738_633_238, 1e-14);
    assert_almost_eq!(ln_gamma(150.0 + 1.0e-12), 600.009_470_555_332_4, 1e-12);
    assert_almost_eq!(ln_gamma(1.001e+7), 1.513_421_353_238_179e8, 1e-13);
}
