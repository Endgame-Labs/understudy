//! The `net` understudent: a small neural classifier, trained from scratch, over [deep's pooled encoder features] ++
//! [`shape`'s structural features].
//!
//! Model: one hidden layer of `HIDDEN` ReLU units with dropout `DROPOUT`, a softmax over the classes seen in training,
//! weighted cross-entropy with an L2 penalty (`WEIGHT_DECAY`), Adam on mini-batches of `BATCH` rows. Each block of the
//! input is standardized with the TRAINING rows' mean and sd (the caller fits it per fold) and clipped at `shape::CLIP`
//! standard deviations. Early stopping: a share (`VAL_PERCENT`) of the training rows (whole groups, picked by hash, so
//! siblings never straddle it) is held out as an inner split; the epoch with the lowest weighted validation loss wins
//! (patience `PATIENCE`) and its weights are restored. Everything is deterministic: a fixed seed per fit (initial
//! weights, batch order, dropout masks), single-threaded (callers run independent fits in parallel), and the number of
//! epochs is bounded by a MAC budget (`MAC_BUDGET`) instead of the clock, so fit time is bounded and reproducible.

use crate::shape;
use crate::readout::Standardizer;
use crate::rng::Xoshiro;
use crate::simd;
use serde::{Deserialize, Serialize};

pub const HIDDEN: usize = 64;
pub const DROPOUT: f32 = 0.1;
pub const LR: f32 = 2e-3;
pub const WEIGHT_DECAY: f32 = 1e-4;
pub const BATCH: usize = 32;
pub const MAX_EPOCHS: usize = 40;
pub const MIN_EPOCHS: usize = 6;
/// Epochs without a validation improvement before stopping.
pub const PATIENCE: usize = 5;
/// Share of the training rows held out for early stopping (percent).
pub const VAL_PERCENT: u64 = 15;
/// Multiply-accumulates one fit may spend (forward plus backward through the first layer, over all epochs); it caps
/// the epochs, so a 2,000-row fit on ~3,500 inputs runs in a few seconds.
pub const MAC_BUDGET: f64 = 1.6e10;

/// The network and what it needs to score a raw row.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NetModel {
    pub pool_std: Standardizer,
    pub shape_std: Standardizer,
    pub h: usize,
    /// [h x (dp + df)], row j holds hidden unit j's input weights.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// [classes x h]
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    /// The class indices the outputs cover, ascending.
    pub classes: Vec<usize>,
    /// The epoch whose weights were kept.
    pub epochs: usize,
}

impl NetModel {
    pub fn d(&self) -> usize {
        self.pool_std.mu.len() + self.shape_std.mu.len()
    }

    /// The network input of a raw row: both blocks standardized with this model's statistics and clipped.
    pub fn input(&self, pool: &[f32], feats: &[f32]) -> Vec<f32> {
        input(&self.pool_std, &self.shape_std, pool, feats)
    }

    /// Probabilities over all `k` classes (zero for a class never seen in training).
    pub fn proba(&self, pool: &[f32], feats: &[f32], k: usize) -> Vec<f64> {
        let z = self.input(pool, feats);
        let net = Net { d: z.len(), h: self.h, w1: &self.w1, b1: &self.b1, w2: &self.w2, b2: &self.b2 };
        let q = net.probs(&z);
        let mut p = vec![0.0; k];
        for (c, x) in self.classes.iter().zip(q) {
            p[*c] = x as f64;
        }
        p
    }
}

/// A raw row's network input under the given standardizers: pool block, then structural block, each clipped.
pub fn input(pool_std: &Standardizer, shape_std: &Standardizer, pool: &[f32], feats: &[f32]) -> Vec<f32> {
    let mut z = shape::standardize(pool_std, pool);
    z.extend(shape::standardize(shape_std, feats));
    z
}

/// Borrowed weights for the forward pass.
struct Net<'a> {
    d: usize,
    h: usize,
    w1: &'a [f32],
    b1: &'a [f32],
    w2: &'a [f32],
    b2: &'a [f32],
}

impl Net<'_> {
    fn hidden(&self, x: &[f32], out: &mut [f32]) {
        for j in 0..self.h {
            out[j] = (simd::dot(&self.w1[j * self.d..(j + 1) * self.d], x) + self.b1[j]).max(0.0);
        }
    }

    fn logits(&self, a: &[f32]) -> Vec<f32> {
        (0..self.b2.len()).map(|c| simd::dot(&self.w2[c * self.h..(c + 1) * self.h], a) + self.b2[c]).collect()
    }

    fn probs(&self, x: &[f32]) -> Vec<f32> {
        let mut a = vec![0f32; self.h];
        self.hidden(x, &mut a);
        softmax(&self.logits(&a))
    }
}

fn softmax(s: &[f32]) -> Vec<f32> {
    let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.into_iter().map(|x| x / z).collect()
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The inner split of the training rows `idx`: (fit rows, early-stopping rows). A row's side is a hash of its group
/// (or, ungrouped, of its index) and the seed, so a group never straddles it. Without enough rows on either side, or
/// with a single class on one, everything trains (and the epoch count is fixed).
pub fn inner_split(idx: &[usize], groups: &[Option<u64>], y: &[usize], seed: u64) -> (Vec<usize>, Vec<usize>) {
    let (mut fit, mut val) = (vec![], vec![]);
    for &i in idx {
        let key = groups[i].unwrap_or(i as u64 ^ (1 << 63));
        if mix(key ^ seed.wrapping_mul(0x2545_f491_4f6c_dd1d)) % 100 < VAL_PERCENT { val.push(i) } else { fit.push(i) }
    }
    let classes = |rows: &[usize]| rows.iter().map(|&i| y[i]).collect::<std::collections::BTreeSet<_>>().len();
    if fit.len() < 20 || val.len() < 10 || classes(&fit) < 2 || classes(&val) < 2 {
        return (idx.to_vec(), vec![]);
    }
    (fit, val)
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    fn new(n: usize) -> Self {
        Adam { m: vec![0.0; n], v: vec![0.0; n] }
    }

    fn step(&mut self, p: &mut [f32], g: &[f32], decay: f32, t: i32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        let (c1, c2) = (1.0 - B1.powi(t), 1.0 - B2.powi(t));
        for i in 0..p.len() {
            let gi = g[i] + decay * p[i];
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * gi;
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * gi * gi;
            p[i] -= LR * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + EPS);
        }
    }
}

/// The trained weights (before packaging with the standardizers).
pub struct Trained {
    pub h: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub classes: Vec<usize>,
    pub epochs: usize,
}

/// Train on the rows `idx` of `z` ([n x d], already standardized and clipped) with classes `y`, weights `w` and the
/// inner split `inner_split` picks. None when those rows hold a single class.
pub fn train(z: &[f32], d: usize, idx: &[usize], groups: &[Option<u64>], y: &[usize], w: &[f64], seed: u64) -> Option<Trained> {
    let (fit_rows, val_rows) = inner_split(idx, groups, y, seed);
    train_split(z, d, idx, fit_rows, val_rows, y, w, seed)
}

/// `train` over an inner split the caller already made (`inner_split` of `idx`), so the caller can fit its standardizers
/// on `fit_rows` alone: the validation rows then influence neither the weights nor the input transformation.
#[allow(clippy::too_many_arguments)]
pub fn train_split(z: &[f32], d: usize, idx: &[usize], fit_rows: Vec<usize>, val_rows: Vec<usize>, y: &[usize], w: &[f64], seed: u64) -> Option<Trained> {
    let mut classes: Vec<usize> = idx.iter().map(|&i| y[i]).collect();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() < 2 {
        return None;
    }
    let nc = classes.len();
    let ci = |c: usize| classes.iter().position(|&x| x == c).unwrap();
    let h = HIDDEN;
    let mean_w = fit_rows.iter().map(|&i| w[i]).sum::<f64>() / fit_rows.len() as f64;
    let wt = |i: usize| (w[i] / mean_w) as f32;

    let mut rng = Xoshiro::new(seed ^ 0x6d6c_7000);
    let s1 = (2.0 / d as f64).sqrt();
    let s2 = (1.0 / h as f64).sqrt();
    let mut w1: Vec<f32> = (0..h * d).map(|_| (rng.normal() * s1) as f32).collect();
    let mut w2: Vec<f32> = (0..nc * h).map(|_| (rng.normal() * s2) as f32).collect();
    let (mut b1, mut b2) = (vec![0f32; h], vec![0f32; nc]);
    let (mut a_w1, mut a_b1, mut a_w2, mut a_b2) = (Adam::new(h * d), Adam::new(h), Adam::new(nc * h), Adam::new(nc));
    let (mut g_w1, mut g_b1, mut g_w2, mut g_b2) = (vec![0f32; h * d], vec![0f32; h], vec![0f32; nc * h], vec![0f32; nc]);

    let macs = fit_rows.len() as f64 * d as f64 * h as f64 * 2.0;
    let max_epochs = ((MAC_BUDGET / macs) as usize).clamp(MIN_EPOCHS, MAX_EPOCHS);
    let early = !val_rows.is_empty();
    let epochs_cap = if early { max_epochs } else { max_epochs.min(15) };

    let val_loss = |w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]| -> f64 {
        let net = Net { d, h, w1, b1, w2, b2 };
        let (mut loss, mut tw) = (0f64, 0f64);
        for &i in &val_rows {
            let p = net.probs(&z[i * d..(i + 1) * d]);
            loss += w[i] * -(p[ci(y[i])].max(1e-7) as f64).ln();
            tw += w[i];
        }
        loss / tw
    };

    let mut best: Option<(f64, usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = None;
    let (mut since_best, mut t) = (0usize, 0i32);
    let mut order = fit_rows.clone();
    let mut a = vec![0f32; h];
    let mut da = vec![0f32; h];
    let keep = 1.0 - DROPOUT;
    let mut epochs_run = 0;
    for epoch in 1..=epochs_cap {
        epochs_run = epoch;
        rng.shuffle(&mut order);
        for batch in order.chunks(BATCH) {
            g_w1.iter_mut().for_each(|g| *g = 0.0);
            g_b1.iter_mut().for_each(|g| *g = 0.0);
            g_w2.iter_mut().for_each(|g| *g = 0.0);
            g_b2.iter_mut().for_each(|g| *g = 0.0);
            let inv = 1.0 / batch.len() as f32;
            for &i in batch {
                let x = &z[i * d..(i + 1) * d];
                let net = Net { d, h, w1: &w1, b1: &b1, w2: &w2, b2: &b2 };
                net.hidden(x, &mut a);
                let mut mask = [0f32; HIDDEN];
                for j in 0..h {
                    mask[j] = if rng.uniform() < DROPOUT as f64 { 0.0 } else { 1.0 / keep };
                    a[j] *= mask[j];
                }
                let mut dz = softmax(&net.logits(&a));
                dz[ci(y[i])] -= 1.0;
                let sw = wt(i) * inv;
                da.iter_mut().for_each(|v| *v = 0.0);
                for c in 0..nc {
                    let g = dz[c] * sw;
                    g_b2[c] += g;
                    simd::axpy(g, &a, &mut g_w2[c * h..(c + 1) * h]);
                    simd::axpy(g, &w2[c * h..(c + 1) * h], &mut da);
                }
                for j in 0..h {
                    // a[j] > 0 only if the unit was active and kept; the mask carries the dropout scale
                    let dj = if a[j] > 0.0 { da[j] * mask[j] } else { 0.0 };
                    if dj != 0.0 {
                        g_b1[j] += dj;
                        simd::axpy(dj, x, &mut g_w1[j * d..(j + 1) * d]);
                    }
                }
            }
            t += 1;
            a_w1.step(&mut w1, &g_w1, WEIGHT_DECAY, t);
            a_b1.step(&mut b1, &g_b1, 0.0, t);
            a_w2.step(&mut w2, &g_w2, WEIGHT_DECAY, t);
            a_b2.step(&mut b2, &g_b2, 0.0, t);
        }
        if early {
            let l = val_loss(&w1, &b1, &w2, &b2);
            if best.as_ref().is_none_or(|b| l < b.0 - 1e-4) {
                best = Some((l, epoch, w1.clone(), b1.clone(), w2.clone(), b2.clone()));
                since_best = 0;
            } else {
                since_best += 1;
                if since_best >= PATIENCE {
                    break;
                }
            }
        }
    }
    Some(match best {
        Some((_, epoch, w1, b1, w2, b2)) => Trained { h, w1, b1, w2, b2, classes, epochs: epoch },
        None => Trained { h, w1, b1, w2, b2, classes, epochs: epochs_run },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two noisy clusters in d dims: class 1 shifted along the first axis.
    fn blobs(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
        let mut r = Xoshiro::new(seed);
        let (mut z, mut y) = (vec![], vec![]);
        for _ in 0..n {
            let c = usize::from(r.uniform() < 0.5);
            for j in 0..d {
                let shift = if j == 0 { if c == 1 { 1.5 } else { -1.5 } } else { 0.0 };
                z.push((r.normal() + shift) as f32);
            }
            y.push(c);
        }
        (z, y)
    }

    #[test]
    fn learns_and_is_deterministic() {
        let (n, d) = (200, 12);
        let (z, y) = blobs(n, d, 1);
        let w = vec![1.0; n];
        let g = vec![None; n];
        let idx: Vec<usize> = (0..150).collect();
        let a = train(&z, d, &idx, &g, &y, &w, 7).unwrap();
        let b = train(&z, d, &idx, &g, &y, &w, 7).unwrap();
        assert_eq!((&a.w1, &a.w2, &a.b1, &a.b2, a.epochs), (&b.w1, &b.w2, &b.b1, &b.b2, b.epochs), "same seed, same weights");
        let c = train(&z, d, &idx, &g, &y, &w, 8).unwrap();
        assert_ne!(a.w1, c.w1, "another seed trains another net");
        let net = Net { d, h: a.h, w1: &a.w1, b1: &a.b1, w2: &a.w2, b2: &a.b2 };
        let hit = (150..n).filter(|&i| {
            let p = net.probs(&z[i * d..(i + 1) * d]);
            (p[1] > p[0]) == (y[i] == 1)
        });
        assert!(hit.count() > 40, "should separate the clusters");
    }

    #[test]
    fn single_class_is_none_and_inner_split_keeps_groups() {
        let y = vec![1usize; 30];
        assert!(train(&[0.0; 30], 1, &(0..30).collect::<Vec<_>>(), &[None; 30], &y, &[1.0; 30], 0).is_none());
        let n = 400;
        let groups: Vec<Option<u64>> = (0..n).map(|i| Some((i / 4) as u64)).collect();
        let y: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let idx: Vec<usize> = (0..n).collect();
        let (fit, val) = inner_split(&idx, &groups, &y, 3);
        assert!(!val.is_empty() && val.len() < n / 3);
        for &v in &val {
            assert!(fit.iter().all(|&f| groups[f] != groups[v]), "a group straddles the inner split");
        }
    }
}
