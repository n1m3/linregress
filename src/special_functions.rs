use statrs::function::beta::{beta, ln_beta};

const MAX_GAMMA: f64 = 171.624376956302725;
const MIN_LOG: f64 = -7.08396418532264106224E2;
const MAX_LOG: f64 = 7.09782712893383996843E2;
const MACHEP: f64 = 0.11102230246251565E-15;
const BIG: f64 = 4.503599627370496E15;
const BIG_INVERSE: f64 = 2.22044604925031308085E-16;

/// Returns incomplete beta integral of the arguments, evaluated
/// from zero to x.
///
/// Based on the C implementation in the cephes library (http://www.netlib.org/cephes/)
/// by Stephen L. Moshier
pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
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
    let mut w = 1. - x;
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
    let y = x * (a + b - 2.0) - (a - 1.0);
    if y < 0. {
        w = inc_bcf(a, b, x);
    } else {
        w = inc_bd(a, b, x) / xc;
    }
    let mut y = a * x.ln();
    let mut t = b * xc.ln();
    if a + b < MAX_GAMMA && y.abs() < MAX_LOG && t.abs() < MAX_LOG {
        t = xc.powf(b);
        t *= x.powf(a);
        t /= a;
        t *= w;
        t *= 1.0 / beta(a, b);
    } else {
        y += t - ln_beta(a, b);
        y += (w / a).ln();
        if y < MIN_LOG {
            t = 0.;
        } else {
            t = y.exp();
        }
    }
    if was_swapped {
        if t <= MACHEP {
            t = 1. - MACHEP;
        } else {
            t = 1. - t;
        }
    }
    t
}
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
    if (a + b) < MAX_GAMMA && u.abs() < MAX_LOG {
        t = 1.0 / beta(a, b);
        s = s * t * x.powf(a);
    } else {
        t = -ln_beta(a, b) + u + s.ln();
        if t < MIN_LOG {
            s = 0.0;
        } else {
            s = t.exp();
        }
    }
    s
}
/// Helper function for inc_beta
fn inc_bcf(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = a + b;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = b - 1.0;
    let mut k7 = k4;
    let mut k8 = a + 2.0;
    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let mut r = 1.0;
    let mut t;
    let mut answer = 1.0;
    let threshold = 3.0 * MACHEP;
    for _n in 0..300 {
        let xk = -(x * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        let xk = (x * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if qk != 0. {
            r = pk / qk;
        }
        if r != 0. {
            t = ((answer - r) / r).abs();
            answer = r;
        } else {
            t = 1.0;
        }
        if t < threshold {
            return answer;
        }
        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;
        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIG_INVERSE;
            pkm1 *= BIG_INVERSE;
            qkm2 *= BIG_INVERSE;
            qkm1 *= BIG_INVERSE;
        }
        if qk.abs() < BIG_INVERSE || pk.abs() < BIG_INVERSE {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    answer
}
/// Helper function for inc_beta
fn inc_bd(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = b - 1.0;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = a + b;
    let mut k7 = a + 1.0;
    let mut k8 = a + 2.0;
    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let z = x / (1.0 - x);
    let mut t;
    let mut answer = 1.0;
    let mut r = 1.0;
    let threshold = 3.0 * MACHEP;
    for _n in 0..300 {
        let xk = -(z * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        let xk = (z * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;
        if qk != 0. {
            r = pk / qk;
        }
        if r != 0. {
            t = ((answer - r) / r).abs();
            answer = r;
        } else {
            t = 1.0;
        }
        if t < threshold {
            return answer;
        }
        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;
        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIG_INVERSE;
            pkm1 *= BIG_INVERSE;
            qkm2 *= BIG_INVERSE;
            qkm1 *= BIG_INVERSE;
        }
        if qk.abs() < BIG_INVERSE || pk.abs() < BIG_INVERSE {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    answer
}
