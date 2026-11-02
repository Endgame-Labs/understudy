//! Readouts over an `Encoding`:
//! - `pool`: per-layer token mean and max, standardized, into `linear::Logistic` (hunch's classifier).
//! - `attn`: K learned queries attend over one layer's tokens; the K pooled vectors and the token mean, each
//!   standardized, feed a linear softmax head. Trained jointly by Adam on weighted cross-entropy + L2.
//! Pairs use `[x, y, x − y, x ⊙ y]` of either readout's per-part vector.

use crate::encoder::{Encoding, Featurizer};
use crate::linear::{self, Logistic};
use crate::rng::Xoshiro;
use crate::simd;
use half::slice::HalfFloatSliceExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Readout {
    Pool,
    Attn,
    /// The `keywords` understudent's sparse logistic regression.
    Keywords,
    /// The `neighbors` understudent's neighbour vote.
    Neighbors,
    /// The `rules` understudent's regex rule set.
    Rules,
    /// The `shape` understudent's logistic regression over structural features.
    Shape,
    /// The `blend` understudent's single sparse+dense logistic regression.
    Blend,
    /// The `memo` understudent's retrieval over past lead answers.
    Memo,
    /// The `net` understudent's one-hidden-layer network.
    Net,
    /// The `checklist` understudent's decision list.
    Checklist,
}

impl Readout {
    pub fn name(self) -> &'static str {
        match self {
            Readout::Pool => "pool",
            Readout::Attn => "attn",
            Readout::Keywords => "keywords",
            Readout::Neighbors => "neighbors",
            Readout::Rules => "rules",
            Readout::Shape => "shape",
            Readout::Blend => "blend",
            Readout::Memo => "memo",
            Readout::Net => "net",
            Readout::Checklist => "checklist",
        }
    }
}

/// Which readouts a refit trains: one, or both with the choice made by OOF agreement (`auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ReadoutChoice {
    Auto,
    Pool,
    Attn,
}

/// Per-dimension mean and standard deviation (+1e-6, as hunch) of training features.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Standardizer {
    pub mu: Vec<f32>,
    pub sd: Vec<f32>,
}

impl Standardizer {
    /// From rows of `x` ([n x d]); population sd, as numpy's `std`.
    pub fn fit(x: &[f32], d: usize) -> Self {
        let n = x.len() / d;
        let mut mu = vec![0f64; d];
        for row in x.chunks_exact(d) {
            mu.iter_mut().zip(row).for_each(|(m, v)| *m += *v as f64);
        }
        mu.iter_mut().for_each(|m| *m /= n as f64);
        let mut var = vec![0f64; d];
        for row in x.chunks_exact(d) {
            var.iter_mut().zip(row).zip(&mu).for_each(|((s, v), m)| s.add_assign_sq(*v as f64 - m));
        }
        Standardizer {
            mu: mu.iter().map(|&m| m as f32).collect(),
            sd: var.iter().map(|&v| ((v / n as f64).sqrt() + 1e-6) as f32).collect(),
        }
    }

    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        for j in 0..x.len() {
            out[j] = (x[j] - self.mu[j]) / self.sd[j];
        }
    }
}

trait AddSq {
    fn add_assign_sq(&mut self, x: f64);
}
impl AddSq for f64 {
    fn add_assign_sq(&mut self, x: f64) {
        *self += x * x;
    }
}

/// The pool readout: standardization plus logistic regression.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PoolModel {
    pub std: Standardizer,
    pub lr: Logistic,
}

impl PoolModel {
    pub fn proba(&self, x: &[f32], k: usize) -> Vec<f64> {
        let mut z = vec![0f32; x.len()];
        self.std.apply(x, &mut z);
        self.lr.proba(&z, k)
    }
}

/// Attention readout training settings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AttnParams {
    pub queries: usize,
    pub lr: f64,
    pub batch: usize,
    /// Epochs: `clamp(example_passes / n, min_epochs, max_epochs)`, so small sets get more passes and large ones
    /// a bounded cost (about `example_passes` example gradients per training).
    pub example_passes: usize,
    pub min_epochs: usize,
    pub max_epochs: usize,
}

impl Default for AttnParams {
    fn default() -> Self {
        AttnParams { queries: 4, lr: 3e-3, batch: 16, example_passes: 16_000, min_epochs: 8, max_epochs: 40 }
    }
}

impl AttnParams {
    pub fn epochs(&self, n: usize) -> usize {
        (self.example_passes / n.max(1)).clamp(self.min_epochs, self.max_epochs)
    }
}

/// A trained attention readout. `q` is [K x hidden] in standardized token space; scores are
/// `zₜ·q / √hidden`. The head is [classes seen x features], features = K pooled vectors + the token mean per
/// part (×4 for a pair).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AttnModel {
    pub hidden: usize,
    pub queries: usize,
    pub pair: bool,
    pub std: Standardizer,
    pub q: Vec<f32>,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub classes: Vec<usize>,
    pub epochs: usize,
}

/// One part's forward state, kept for the backward pass.
struct PartFwd {
    x: Vec<f32>,   // tokens as f32 [t x h]
    a: Vec<f32>,   // attention [K x t]
    v: Vec<f32>,   // standardized vector [(K + 1) x h]
}

/// Token-level standardization over every kept token of the given encodings (all parts).
pub fn token_standardizer(fz: &Featurizer, encs: &[&Encoding], h: usize) -> Standardizer {
    use rayon::prelude::*;
    let sums: Vec<(Vec<f64>, Vec<f64>, usize)> = encs
        .par_iter()
        .map(|e| {
            let (mut s, mut s2, mut n) = (vec![0f64; h], vec![0f64; h], 0);
            let mut buf = vec![0f32; h];
            for p in fz.parts(e) {
                for row in p.states.chunks_exact(h) {
                    row.convert_to_f32_slice(&mut buf);
                    for j in 0..h {
                        let v = buf[j] as f64;
                        s[j] += v;
                        s2[j] += v * v;
                    }
                    n += 1;
                }
            }
            (s, s2, n)
        })
        .collect();
    let (mut s, mut s2, mut n) = (vec![0f64; h], vec![0f64; h], 0usize);
    for (a, b, c) in sums {
        s.iter_mut().zip(&a).for_each(|(x, y)| *x += y);
        s2.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
        n += c;
    }
    let n = n.max(1) as f64;
    Standardizer {
        mu: s.iter().map(|x| (x / n) as f32).collect(),
        sd: s.iter().zip(&s2).map(|(a, b)| (((b / n) - (a / n).powi(2)).max(0.0).sqrt() + 1e-6) as f32).collect(),
    }
}

impl AttnModel {
    fn part_dim(&self) -> usize {
        (self.queries + 1) * self.hidden
    }

    fn shape_dim(&self) -> usize {
        self.part_dim() * if self.pair { 4 } else { 1 }
    }

    /// Scaled queries in raw token space: q̃ₖ[d] = qₖ[d] / (sd[d] √h). (The mean shift is constant per query and
    /// cancels in the softmax.)
    fn raw_queries(&self) -> Vec<f32> {
        let h = self.hidden;
        let s = (h as f32).sqrt();
        let mut out = vec![0f32; self.queries * h];
        for k in 0..self.queries {
            for d in 0..h {
                out[k * h + d] = self.q[k * h + d] / (self.std.sd[d] * s);
            }
        }
        out
    }

    fn forward_part(&self, fz: &Featurizer, p: &crate::encoder::Part, qr: &[f32]) -> PartFwd {
        let (h, kq, t) = (self.hidden, self.queries, p.tokens);
        let mut x = vec![0f32; t * h];
        p.states.convert_to_f32_slice(&mut x);
        let mut a = vec![0f32; kq * t];
        for (ti, row) in x.chunks_exact(h).enumerate() {
            let s = dot_q(row, qr, kq, h);
            for k in 0..kq {
                a[k * t + ti] = s[k];
            }
        }
        for k in 0..kq {
            softmax(&mut a[k * t..(k + 1) * t]);
        }
        let mut pooled = vec![0f32; kq * h];
        for (ti, row) in x.chunks_exact(h).enumerate() {
            axpy_q(&a, t, ti, row, &mut pooled, kq, h);
        }
        let mut v = vec![0f32; (kq + 1) * h];
        for k in 0..kq {
            self.std.apply(&pooled[k * h..(k + 1) * h], &mut v[k * h..(k + 1) * h]);
        }
        self.std.apply(fz.mid_mean(p), &mut v[kq * h..]);
        PartFwd { x, a, v }
    }

    fn features(&self, parts: &[PartFwd]) -> Vec<f32> {
        match parts {
            [p] => p.v.clone(),
            [x, y] => {
                let mut f = Vec::with_capacity(4 * x.v.len());
                f.extend_from_slice(&x.v);
                f.extend_from_slice(&y.v);
                f.extend(x.v.iter().zip(&y.v).map(|(a, b)| a - b));
                f.extend(x.v.iter().zip(&y.v).map(|(a, b)| a * b));
                f
            }
            _ => unreachable!(),
        }
    }

    fn logits(&self, f: &[f32]) -> Vec<f64> {
        let fd = self.shape_dim();
        (0..self.classes.len()).map(|r| simd::dot(f, &self.w[r * fd..(r + 1) * fd]) as f64 + self.b[r] as f64).collect()
    }

    /// Probabilities over all `k` classes.
    pub fn proba(&self, fz: &Featurizer, e: &Encoding, k: usize) -> Vec<f64> {
        let qr = self.raw_queries();
        let parts: Vec<PartFwd> = fz.parts(e).iter().map(|p| self.forward_part(fz, p, &qr)).collect();
        let s = self.logits(&self.features(&parts));
        let q = softmax64(&s);
        let mut p = vec![0.0; k];
        for (c, x) in self.classes.iter().zip(q) {
            p[*c] = x;
        }
        p
    }

    /// Train on `idx` of `encs` with classes `y` and weights `w`; None for a single class. `std` is the token
    /// standardization (computed once over all examples, as hunch standardizes before its folds).
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        fz: &Featurizer, encs: &[&Encoding], idx: &[usize], y: &[usize], w: &[f64], std: &Standardizer,
        params: &AttnParams, seed: u64,
    ) -> Option<AttnModel> {
        let mut classes: Vec<usize> = idx.iter().map(|&i| y[i]).collect();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 2 {
            return None;
        }
        let h = fz.hidden;
        let kq = params.queries;
        assert_eq!(kq, 4, "the kernels are written for four queries");
        let pair = fz.pair.is_some();
        let mut rng = Xoshiro::new(seed ^ 0x6174_746e);
        let mut m = AttnModel {
            hidden: h,
            queries: kq,
            pair,
            std: std.clone(),
            q: (0..kq * h).map(|_| (0.1 * rng.normal()) as f32).collect(),
            w: vec![],
            b: vec![0.0; classes.len()],
            classes: classes.clone(),
            epochs: params.epochs(idx.len()),
        };
        let fd = m.shape_dim();
        m.w = vec![0.0; classes.len() * fd];
        let rows = classes.len();
        let pos: Vec<usize> = y.iter().map(|c| classes.binary_search(c).unwrap_or(0)).collect();
        let sw: f64 = idx.iter().map(|&i| w[i]).sum();
        let l2 = (1.0 / (linear::C * sw)) as f32; // the pool readout's penalty, per unit of mean loss
        let np = kq * h + rows * fd + rows;
        let mut adam = Adam::new(np, params.lr as f32);
        let mut order = idx.to_vec();
        let sqrt_h = (h as f32).sqrt();
        for _ in 0..m.epochs {
            rng.shuffle(&mut order);
            for batch in order.chunks(params.batch) {
                let qr = m.raw_queries();
                let mut g = vec![0f32; np];
                let bw: f64 = batch.iter().map(|&i| w[i]).sum();
                for &i in batch {
                    let e = encs[i];
                    let parts: Vec<PartFwd> = fz.parts(e).iter().map(|p| m.forward_part(fz, p, &qr)).collect();
                    let f = m.features(&parts);
                    let p = softmax64(&m.logits(&f));
                    let scale = w[i] / bw;
                    let gl: Vec<f32> = (0..rows).map(|r| (scale * (p[r] - if r == pos[i] { 1.0 } else { 0.0 })) as f32).collect();
                    let (gq, rest) = g.split_at_mut(kq * h);
                    let (gw, gb) = rest.split_at_mut(rows * fd);
                    let mut gf = vec![0f32; fd];
                    for r in 0..rows {
                        simd::axpy(gl[r], &f, &mut gw[r * fd..(r + 1) * fd]);
                        gb[r] += gl[r];
                        simd::axpy(gl[r], &m.w[r * fd..(r + 1) * fd], &mut gf);
                    }
                    let pd = m.part_dim();
                    let gv: Vec<Vec<f32>> = if pair {
                        let (va, vb) = (&parts[0].v, &parts[1].v);
                        let ga = (0..pd).map(|j| gf[j] + gf[2 * pd + j] + gf[3 * pd + j] * vb[j]).collect();
                        let gb2 = (0..pd).map(|j| gf[pd + j] - gf[2 * pd + j] + gf[3 * pd + j] * va[j]).collect();
                        vec![ga, gb2]
                    } else {
                        vec![gf]
                    };
                    for (pf, gv) in parts.iter().zip(&gv) {
                        let t = pf.x.len() / h;
                        // gradient wrt the raw pooled vectors: through the standardization
                        let mut gp = vec![0f32; kq * h];
                        for k in 0..kq {
                            for d in 0..h {
                                gp[k * h + d] = gv[k * h + d] / m.std.sd[d];
                            }
                        }
                        // Σₜ aₖₜ (hₜ·gpₖ) = pooledₖ·gpₖ, needed for the softmax backward
                        let mut ga = vec![0f32; kq * t];
                        for (ti, row) in pf.x.chunks_exact(h).enumerate() {
                            let s = dot_q(row, &gp, kq, h);
                            for k in 0..kq {
                                ga[k * t + ti] = s[k];
                            }
                        }
                        let mut gs = vec![0f32; kq * t];
                        for k in 0..kq {
                            let a = &pf.a[k * t..(k + 1) * t];
                            let gak = &ga[k * t..(k + 1) * t];
                            let mean: f32 = a.iter().zip(gak).map(|(x, y)| x * y).sum();
                            for ti in 0..t {
                                gs[k * t + ti] = a[ti] * (gak[ti] - mean);
                            }
                        }
                        let mut gqr = vec![0f32; kq * h];
                        for (ti, row) in pf.x.chunks_exact(h).enumerate() {
                            axpy_q(&gs, t, ti, row, &mut gqr, kq, h);
                        }
                        for k in 0..kq {
                            for d in 0..h {
                                gq[k * h + d] += gqr[k * h + d] / (m.std.sd[d] * sqrt_h);
                            }
                        }
                    }
                }
                // L2 on the queries and the head's weights (not the bias)
                for j in 0..kq * h {
                    g[j] += l2 * m.q[j];
                }
                for j in 0..rows * fd {
                    g[kq * h + j] += l2 * m.w[j];
                }
                adam.step(&mut [&mut m.q, &mut m.w, &mut m.b], &g);
            }
        }
        Some(m)
    }
}

#[inline]
fn dot_q(row: &[f32], q: &[f32], kq: usize, h: usize) -> [f32; 4] {
    debug_assert_eq!(kq, 4);
    simd::dot4(row, [&q[..h], &q[h..2 * h], &q[2 * h..3 * h], &q[3 * h..4 * h]])
}

/// out[k] += a[k, ti] * row for the four queries.
#[inline]
fn axpy_q(a: &[f32], t: usize, ti: usize, row: &[f32], out: &mut [f32], kq: usize, h: usize) {
    debug_assert_eq!(kq, 4);
    let (o0, r) = out.split_at_mut(h);
    let (o1, r) = r.split_at_mut(h);
    let (o2, o3) = r.split_at_mut(h);
    simd::axpy4([a[ti], a[t + ti], a[2 * t + ti], a[3 * t + ti]], row, [o0, o1, o2, o3]);
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    x.iter_mut().for_each(|v| *v /= s);
}

fn softmax64(s: &[f64]) -> Vec<f64> {
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

/// Adam over several parameter slices laid end to end.
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
    lr: f32,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n], t: 0, lr }
    }

    fn step(&mut self, params: &mut [&mut Vec<f32>], g: &[f32]) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        self.t += 1;
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        let mut j = 0;
        for p in params.iter_mut() {
            for x in p.iter_mut() {
                self.m[j] = b1 * self.m[j] + (1.0 - b1) * g[j];
                self.v[j] = b2 * self.v[j] + (1.0 - b2) * g[j] * g[j];
                *x -= self.lr * (self.m[j] / c1) / ((self.v[j] / c2).sqrt() + eps);
                j += 1;
            }
        }
    }
}
