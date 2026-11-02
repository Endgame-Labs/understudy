//! Weighted L2-regularized logistic regression, as scikit-learn's `LogisticRegression(C=0.5)` fits it for hunch:
//! a sigmoid over one weight vector for two classes, a softmax over one per class for more, unpenalized
//! intercepts, the objective `(Σ wᵢ lossᵢ + ‖W‖² / 2C) / Σ wᵢ` minimized by full-batch L-BFGS to a max-gradient
//! tolerance of 1e-4. Single-threaded and deterministic; callers run independent fits in parallel.

use crate::simd;
use serde::{Deserialize, Serialize};

pub const C: f64 = 0.5;
const TOL: f64 = 1e-4;
const MAX_ITER: usize = 1000;
const HISTORY: usize = 10;

/// A fitted model: `coef` is [rows x d] (rows = 1 for two classes, else one per class seen).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Logistic {
    pub d: usize,
    pub coef: Vec<f32>,
    pub intercept: Vec<f32>,
    /// The class indices the weights cover, ascending.
    pub classes: Vec<usize>,
    pub iters: usize,
}

impl Logistic {
    fn rows(&self) -> usize {
        self.intercept.len()
    }

    /// Probabilities over all `k` classes (zero for a class never seen in training).
    pub fn proba(&self, z: &[f32], k: usize) -> Vec<f64> {
        let scores: Vec<f64> =
            (0..self.rows()).map(|r| simd::dot(z, &self.coef[r * self.d..(r + 1) * self.d]) as f64 + self.intercept[r] as f64).collect();
        let q = link(&scores);
        let mut p = vec![0.0; k];
        for (c, x) in self.classes.iter().zip(q) {
            p[*c] = x;
        }
        p
    }
}

/// Class probabilities from scores: a sigmoid for one score, a softmax for several.
pub(crate) fn link(s: &[f64]) -> Vec<f64> {
    if s.len() == 1 {
        let p1 = sigmoid(s[0]);
        return vec![1.0 - p1, p1];
    }
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

/// log(1 + e^x), stably.
pub(crate) fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// Fit on the rows `idx` of `z` ([n x d], row-major, standardized) with classes `y` and weights `w`. None when
/// those rows hold a single class.
pub fn fit(z: &[f32], d: usize, idx: &[usize], y: &[usize], w: &[f64], c: f64) -> Option<Logistic> {
    let mut classes: Vec<usize> = idx.iter().map(|&i| y[i]).collect();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() < 2 {
        return None;
    }
    let rows = if classes.len() == 2 { 1 } else { classes.len() };
    let pos: Vec<usize> = idx.iter().map(|&i| classes.binary_search(&y[i]).unwrap()).collect();
    let sw: f64 = idx.iter().map(|&i| w[i]).sum();
    let l2 = 1.0 / (c * sw);
    let np = rows * (d + 1);
    // Parameters: [rows x d] weights, then `rows` intercepts.
    let eval = |theta: &[f64], grad: &mut [f64]| -> f64 {
        let wf: Vec<f32> = theta[..rows * d].iter().map(|&x| x as f32).collect();
        let mut gw = vec![0f32; rows * d];
        let mut gb = vec![0f64; rows];
        let mut loss = 0.0;
        let mut s = vec![0f64; rows];
        for (j, &i) in idx.iter().enumerate() {
            let x = &z[i * d..(i + 1) * d];
            for r in 0..rows {
                s[r] = simd::dot(x, &wf[r * d..(r + 1) * d]) as f64 + theta[rows * d + r];
            }
            let wi = w[i];
            if rows == 1 {
                let yv = pos[j] as f64;
                loss += wi * if pos[j] == 1 { softplus(-s[0]) } else { softplus(s[0]) };
                let r = wi * (sigmoid(s[0]) - yv);
                gb[0] += r;
                simd::axpy(r as f32, x, &mut gw[..d]);
            } else {
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = m + s.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
                loss += wi * (lse - s[pos[j]]);
                for r in 0..rows {
                    let g = wi * ((s[r] - lse).exp() - if r == pos[j] { 1.0 } else { 0.0 });
                    gb[r] += g;
                    simd::axpy(g as f32, x, &mut gw[r * d..(r + 1) * d]);
                }
            }
        }
        let mut reg = 0.0;
        for q in 0..rows * d {
            reg += theta[q] * theta[q];
            grad[q] = gw[q] as f64 / sw + l2 * theta[q];
        }
        for r in 0..rows {
            grad[rows * d + r] = gb[r] / sw;
        }
        loss / sw + 0.5 * l2 * reg
    };
    let mut theta = vec![0f64; np];
    let iters = lbfgs(&mut theta, eval);
    Some(Logistic {
        d,
        coef: theta[..rows * d].iter().map(|&x| x as f32).collect(),
        intercept: theta[rows * d..].iter().map(|&x| x as f32).collect(),
        classes,
        iters,
    })
}

/// Minimize a smooth convex function by L-BFGS with a backtracking Armijo line search. Returns the iterations.
pub fn lbfgs(x: &mut [f64], f: impl FnMut(&[f64], &mut [f64]) -> f64) -> usize {
    lbfgs_with(x, f, TOL, MAX_ITER)
}

/// `lbfgs` with its own max-gradient tolerance and iteration cap.
pub fn lbfgs_with(x: &mut [f64], mut f: impl FnMut(&[f64], &mut [f64]) -> f64, tol: f64, max_iter: usize) -> usize {
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut fx = f(x, &mut g);
    let mut hist: std::collections::VecDeque<(Vec<f64>, Vec<f64>, f64)> = Default::default();
    let mut xn = vec![0.0; n];
    let mut gn = vec![0.0; n];
    for it in 0..max_iter {
        if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) <= tol {
            return it;
        }
        // two-loop recursion: d = -H g
        let mut d: Vec<f64> = g.iter().map(|v| -v).collect();
        let mut alphas = Vec::with_capacity(hist.len());
        for (s, y, rho) in hist.iter().rev() {
            let a = rho * dotf(s, &d);
            axpyf(-a, y, &mut d);
            alphas.push(a);
        }
        if let Some((s, y, _)) = hist.back() {
            let gamma = dotf(s, y) / dotf(y, y);
            d.iter_mut().for_each(|v| *v *= gamma);
        } else {
            let gn2 = dotf(&g, &g).sqrt();
            d.iter_mut().for_each(|v| *v /= gn2.max(1.0));
        }
        for ((s, y, rho), a) in hist.iter().zip(alphas.into_iter().rev()) {
            let b = rho * dotf(y, &d);
            axpyf(a - b, s, &mut d);
        }
        let mut slope = dotf(&g, &d);
        if slope >= 0.0 {
            // not a descent direction: restart from steepest descent
            hist.clear();
            d = g.iter().map(|v| -v).collect();
            slope = dotf(&g, &d);
        }
        let mut step = 1.0;
        let mut fnew;
        let mut tries = 0;
        loop {
            for i in 0..n {
                xn[i] = x[i] + step * d[i];
            }
            fnew = f(&xn, &mut gn);
            if fnew <= fx + 1e-4 * step * slope || tries >= 30 {
                break;
            }
            step *= 0.5;
            tries += 1;
        }
        if fnew > fx {
            return it; // no progress possible at this precision
        }
        let s: Vec<f64> = (0..n).map(|i| xn[i] - x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| gn[i] - g[i]).collect();
        let sy = dotf(&s, &y);
        if sy > 1e-12 {
            if hist.len() == HISTORY {
                hist.pop_front();
            }
            hist.push_back((s, y, 1.0 / sy));
        }
        x.copy_from_slice(&xn);
        g.copy_from_slice(&gn);
        let rel = (fx - fnew) / fx.abs().max(fnew.abs()).max(1.0);
        fx = fnew;
        if rel <= 64.0 * f64::EPSILON {
            return it + 1;
        }
    }
    max_iter
}

fn dotf(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpyf(a: f64, x: &[f64], y: &mut [f64]) {
    y.iter_mut().zip(x).for_each(|(y, x)| *y += a * x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_and_matches_sklearn_shape() {
        // 1-d data: x = ±1 with label = (x > 0); the fit must put class 1 above 0.5 for x = 1.
        let z: Vec<f32> = (0..40).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let y: Vec<usize> = (0..40).map(|i| usize::from(i % 2 == 0)).collect();
        let w = vec![1.0; 40];
        let idx: Vec<usize> = (0..40).collect();
        let m = fit(&z, 1, &idx, &y, &w, C).unwrap();
        // sklearn: LogisticRegression(C=0.5).fit(x, y) on this data gives coef ≈ 2.128: w(1 + e^w) = 20.
        let p = m.proba(&[1.0], 2);
        assert!((p[1] - 0.8936).abs() < 1e-3, "{p:?}");
        assert!((m.coef[0] - 2.128).abs() < 0.005, "{}", m.coef[0]);
        // three classes on three points
        let z3 = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, -1.0];
        let y3 = vec![0usize, 1, 2];
        let m3 = fit(&z3, 2, &[0, 1, 2], &y3, &[1.0; 3], 100.0).unwrap();
        for i in 0..3 {
            let p = m3.proba(&z3[i * 2..i * 2 + 2], 4);
            assert_eq!(crate::stats::argmax(&p), i);
            assert_eq!(p[3], 0.0);
        }
    }
}
