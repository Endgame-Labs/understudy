//! The `keywords` understudent's features and model, and the `neighbors` understudent's model: no neural network.
//!
//! Features: hashed lowercase character 3-5-grams plus word unigrams and bigrams of the state text(s), signed
//! hashing into `BUCKETS` buckets, sublinear tf (`1 + ln count`), L2-normalized. Model: the same weighted L2
//! logistic regression as `linear` (objective `(Σ wᵢ lossᵢ + ‖W‖²/2C) / Σ wᵢ`, unpenalized intercepts) on the sparse
//! vectors, with the weights of buckets seen in fewer than `MIN_DF` training rows left at zero (they cannot
//! generalize), so a fit works on the few tens of thousands of columns that matter, not all 2^18.
//!
//! Novelty: a fixed sparse random projection of the hashed vector to `PROJ` dims (each bucket adds ± its value to
//! `SLOTS` hashed dims), L2-normalized, so cosine similarity of the projections approximates the sparse cosine at a
//! fixed 256-float cost per reference.

use crate::linear::{self, lbfgs_with};
use serde::{Deserialize, Serialize};

pub const BUCKETS: usize = 1 << 18;
/// Regularization of the sparse fit: L2-normalized inputs need a far larger C than standardized dense features.
pub const C: f64 = 30.0;
/// A bucket needs this many training rows to get a weight.
pub const MIN_DF: usize = 2;
pub const PROJ: usize = 256;
const SLOTS: u64 = 4;
const TOL: f64 = 1e-3;
const MAX_ITER: usize = 300;
/// `neighbors`'s neighbours.
pub const NEIGHBORS_K: usize = 7;
/// `neighbors`'s vote smoothing, added to every class.
pub const NEIGHBORS_ALPHA: f64 = 0.5;

/// A sparse vector: ascending bucket indices and their values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sparse {
    pub idx: Vec<u32>,
    pub val: Vec<f32>,
}

fn fnv(salt: u32, bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ salt as u64;
    h = h.wrapping_mul(0x100000001b3);
    for b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    mix(h)
}

/// splitmix64's finalizer: FNV's low bits are weak, so bucket and sign come from a mixed hash.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Hash the state's texts (one per pair field) into a normalized sparse vector. Each part's features are salted by
/// its position, so a pair's parts occupy different buckets and are summed in one vector.
pub fn featurize(texts: &[String]) -> Sparse {
    let mut hashes: Vec<u64> = vec![];
    for (pi, t) in texts.iter().enumerate() {
        let lower: Vec<char> = t.to_lowercase().chars().collect();
        // character n-grams (class ids 0..=2 by n)
        let mut gram = String::new();
        for n in 3..=5usize {
            for w in lower.windows(n) {
                gram.clear();
                gram.extend(w);
                hashes.push(fnv(((n as u32 - 3) << 4) | pi as u32, gram.bytes()));
            }
        }
        // words: runs of alphanumerics
        let words: Vec<String> =
            lower.split(|c| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.iter().collect::<String>()).collect();
        for w in &words {
            hashes.push(fnv((3 << 4) | pi as u32, w.bytes()));
        }
        for p in words.windows(2) {
            hashes.push(fnv((4 << 4) | pi as u32, p[0].bytes().chain(std::iter::once(b' ')).chain(p[1].bytes())));
        }
    }
    hashes.sort_unstable();
    // run-length counts per full hash, sublinear tf, then signed accumulation into buckets
    let mut feats: Vec<(u32, f32)> = vec![];
    let mut i = 0;
    while i < hashes.len() {
        let mut j = i;
        while j < hashes.len() && hashes[j] == hashes[i] {
            j += 1;
        }
        let h = hashes[i];
        let tf = 1.0 + ((j - i) as f32).ln();
        let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        feats.push(((h as usize & (BUCKETS - 1)) as u32, sign * tf));
        i = j;
    }
    feats.sort_unstable_by_key(|f| f.0);
    let mut out = Sparse::default();
    for (b, v) in feats {
        if out.idx.last() == Some(&b) {
            *out.val.last_mut().unwrap() += v;
        } else {
            out.idx.push(b);
            out.val.push(v);
        }
    }
    // drop buckets whose collisions cancelled exactly, then normalize
    let keep: Vec<usize> = (0..out.idx.len()).filter(|&i| out.val[i] != 0.0).collect();
    out.idx = keep.iter().map(|&i| out.idx[i]).collect();
    out.val = keep.iter().map(|&i| out.val[i]).collect();
    let norm = out.val.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt() as f32 + 1e-12;
    out.val.iter_mut().for_each(|v| *v /= norm);
    out
}

/// Visit every hashed feature of one state text (field `pi`) whose bucket is in `want`, with a readable description of
/// the n-gram (the same enumeration and hashing as `featurize`): character 3-5-grams as `"abc"`, words as `word:x`,
/// word bigrams as `words:x y`; a pair's second field gets a `field2:` prefix.
pub fn for_each_gram(t: &str, pi: usize, want: &std::collections::HashSet<u32>, mut f: impl FnMut(u32, String)) {
    let lower: Vec<char> = t.to_lowercase().chars().collect();
    let tag = if pi == 0 { String::new() } else { format!("field{}:", pi + 1) };
    let bucket = |h: u64| (h as usize & (BUCKETS - 1)) as u32;
    let mut gram = String::new();
    for n in 3..=5usize {
        for w in lower.windows(n) {
            gram.clear();
            gram.extend(w);
            let b = bucket(fnv(((n as u32 - 3) << 4) | pi as u32, gram.bytes()));
            if want.contains(&b) {
                f(b, format!("{tag}{gram:?}"));
            }
        }
    }
    let words: Vec<String> = lower.split(|c| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.iter().collect::<String>()).collect();
    for w in &words {
        let b = bucket(fnv((3 << 4) | pi as u32, w.bytes()));
        if want.contains(&b) {
            f(b, format!("{tag}word:{w}"));
        }
    }
    for p in words.windows(2) {
        let b = bucket(fnv((4 << 4) | pi as u32, p[0].bytes().chain(std::iter::once(b' ')).chain(p[1].bytes())));
        if want.contains(&b) {
            f(b, format!("{tag}words:{} {}", p[0], p[1]));
        }
    }
}

/// The projection of every bucket: `SLOTS` (dim, sign) pairs, built once (`dim << 1 | negative`).
fn table() -> &'static [u16] {
    static T: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        (0..BUCKETS as u64 * SLOTS)
            .map(|q| {
                let h = mix(q + 0x9e3779b97f4a7c15);
                (((h as usize) % PROJ) as u16) << 1 | ((h >> 40) & 1) as u16
            })
            .collect()
    })
}

/// The novelty vector: a sparse random projection to `PROJ` dims, L2-normalized.
pub fn project(s: &Sparse) -> Vec<f32> {
    let t = table();
    let mut v = vec![0f32; PROJ];
    for (&b, &x) in s.idx.iter().zip(&s.val) {
        for e in &t[b as usize * SLOTS as usize..(b as usize + 1) * SLOTS as usize] {
            v[(*e >> 1) as usize] += if e & 1 == 1 { -x } else { x };
        }
    }
    let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32 + 1e-12;
    v.iter_mut().for_each(|x| *x /= n);
    v
}

/// Sparse logistic regression: dense weights over all buckets (zero where never trained), `rows` = 1 for two
/// classes else one per class seen.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SparseLr {
    pub coef: Vec<f32>,
    pub intercept: Vec<f32>,
    pub classes: Vec<usize>,
    pub iters: usize,
}

impl SparseLr {
    /// Probabilities over all `k` classes (zero for a class never seen in training).
    pub fn proba(&self, x: &Sparse, k: usize) -> Vec<f64> {
        let dim = self.coef.len() / self.intercept.len();
        let scores: Vec<f64> = (0..self.intercept.len())
            .map(|r| {
                let w = &self.coef[r * dim..(r + 1) * dim];
                x.idx.iter().zip(&x.val).map(|(&i, &v)| w[i as usize] as f64 * v as f64).sum::<f64>() + self.intercept[r] as f64
            })
            .collect();
        let q = linear::link(&scores);
        let mut p = vec![0.0; k];
        for (c, x) in self.classes.iter().zip(q) {
            p[*c] = x;
        }
        p
    }
}

/// Fit on the rows `idx` of `xs` with classes `y` and weights `w`. None when those rows hold a single class.
pub fn fit(xs: &[&Sparse], idx: &[usize], y: &[usize], w: &[f64], c: f64) -> Option<SparseLr> {
    fit_dim(xs, idx, y, w, c, BUCKETS)
}

/// `fit` over `dim` columns (`BUCKETS` for the hashed n-grams; `blend` appends dense columns after them).
pub fn fit_dim(xs: &[&Sparse], idx: &[usize], y: &[usize], w: &[f64], c: f64, dim: usize) -> Option<SparseLr> {
    let mut classes: Vec<usize> = idx.iter().map(|&i| y[i]).collect();
    classes.sort_unstable();
    classes.dedup();
    if classes.len() < 2 {
        return None;
    }
    let rows = if classes.len() == 2 { 1 } else { classes.len() };
    // compact the columns: buckets with enough document frequency
    let mut df = vec![0u32; dim];
    for &i in idx {
        xs[i].idx.iter().for_each(|&b| df[b as usize] += 1);
    }
    let mut col = vec![u32::MAX; dim];
    let mut m = 0usize;
    let mut cols: Vec<u32> = vec![];
    for b in 0..dim {
        if df[b] as usize >= MIN_DF {
            col[b] = m as u32;
            cols.push(b as u32);
            m += 1;
        }
    }
    // CSR over the kept columns
    let mut ptr = vec![0usize];
    let (mut ci, mut cv): (Vec<u32>, Vec<f32>) = (vec![], vec![]);
    for &i in idx {
        for (b, v) in xs[i].idx.iter().zip(&xs[i].val) {
            if col[*b as usize] != u32::MAX {
                ci.push(col[*b as usize]);
                cv.push(*v);
            }
        }
        ptr.push(ci.len());
    }
    let pos: Vec<usize> = idx.iter().map(|&i| classes.binary_search(&y[i]).unwrap()).collect();
    let sw: f64 = idx.iter().map(|&i| w[i]).sum();
    let l2 = 1.0 / (c * sw);
    // theta: [rows x m] weights, then `rows` intercepts. f32 inner loops, as `linear::fit`.
    let eval = |theta: &[f64], grad: &mut [f64]| -> f64 {
        let wf: Vec<f32> = theta[..rows * m].iter().map(|&x| x as f32).collect();
        let mut gw = vec![0f32; rows * m];
        let mut gb = vec![0f64; rows];
        let mut loss = 0.0;
        let mut s = vec![0f64; rows];
        let mut g = vec![0f32; rows];
        for (j, &i) in idx.iter().enumerate() {
            let (cols_j, vals_j) = (&ci[ptr[j]..ptr[j + 1]], &cv[ptr[j]..ptr[j + 1]]);
            for r in 0..rows {
                let wr = &wf[r * m..(r + 1) * m];
                s[r] = cols_j.iter().zip(vals_j).map(|(&c, &v)| wr[c as usize] * v).sum::<f32>() as f64 + theta[rows * m + r];
            }
            let wi = w[i];
            if rows == 1 {
                loss += wi * if pos[j] == 1 { linear::softplus(-s[0]) } else { linear::softplus(s[0]) };
                g[0] = (wi * (linear::sigmoid(s[0]) - pos[j] as f64)) as f32;
            } else {
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = mx + s.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
                loss += wi * (lse - s[pos[j]]);
                for r in 0..rows {
                    g[r] = (wi * ((s[r] - lse).exp() - if r == pos[j] { 1.0 } else { 0.0 })) as f32;
                }
            }
            for r in 0..rows {
                gb[r] += g[r] as f64;
                let gr = &mut gw[r * m..(r + 1) * m];
                for (&c, &v) in cols_j.iter().zip(vals_j) {
                    gr[c as usize] += g[r] * v;
                }
            }
        }
        let mut reg = 0.0;
        for q in 0..rows * m {
            reg += theta[q] * theta[q];
            grad[q] = gw[q] as f64 / sw + l2 * theta[q];
        }
        for r in 0..rows {
            grad[rows * m + r] = gb[r] / sw;
        }
        loss / sw + 0.5 * l2 * reg
    };
    let mut theta = vec![0f64; rows * (m + 1)];
    let iters = lbfgs_with(&mut theta, eval, TOL, MAX_ITER);
    let mut coef = vec![0f32; rows * dim];
    for r in 0..rows {
        for (cl, &b) in cols.iter().enumerate() {
            coef[r * dim + b as usize] = theta[r * m + cl] as f32;
        }
    }
    Some(SparseLr { coef, intercept: theta[rows * m..].iter().map(|&x| x as f32).collect(), classes, iters })
}

/// Cosine nearest-neighbour vote over L2-normalized dense vectors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Neighbors {
    pub d: usize,
    /// [n x d], each row L2-normalized.
    pub vecs: Vec<f32>,
    pub y: Vec<u32>,
}

impl Neighbors {
    pub fn new(d: usize, vecs: Vec<f32>, y: Vec<u32>) -> Self {
        Neighbors { d, vecs, y }
    }

    pub fn len(&self) -> usize {
        self.y.len()
    }

    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// The vote of the `NEIGHBORS_K` most similar of `rows` (all rows when None) for `x`, over `k` classes: each
    /// neighbour counts `max(cos, 0)`, every class gets `NEIGHBORS_ALPHA` added, and the votes are normalized. Ties in
    /// similarity go to the earlier row.
    pub fn proba(&self, x: &[f32], k: usize, rows: Option<&[usize]>) -> Vec<f64> {
        let sim = |i: usize| crate::simd::dot(x, &self.vecs[i * self.d..(i + 1) * self.d]);
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(NEIGHBORS_K + 1);
        let mut visit = |i: usize| {
            let s = sim(i);
            if best.len() < NEIGHBORS_K || s > best[NEIGHBORS_K - 1].0 {
                let at = best.iter().position(|b| s > b.0).unwrap_or(best.len());
                best.insert(at, (s, i));
                best.truncate(NEIGHBORS_K);
            }
        };
        match rows {
            Some(r) => r.iter().for_each(|&i| visit(i)),
            None => (0..self.len()).for_each(&mut visit),
        }
        let mut v = vec![NEIGHBORS_ALPHA; k];
        for (s, i) in best {
            v[self.y[i] as usize] += s.max(0.0) as f64;
        }
        let z: f64 = v.iter().sum();
        v.into_iter().map(|x| x / z).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_normalized_and_stable() {
        let a = featurize(&["Error: file not found".to_string()]);
        let b = featurize(&["error: FILE not found".to_string()]);
        assert_eq!(a, b, "lowercased");
        let n: f32 = a.val.iter().map(|v| v * v).sum();
        assert!((n - 1.0).abs() < 1e-4);
        assert!(a.idx.windows(2).all(|w| w[0] < w[1]));
        let c = featurize(&["all good here".to_string()]);
        let cos = |x: &Sparse, y: &Sparse| {
            let mut s = 0.0;
            let (mut i, mut j) = (0, 0);
            while i < x.idx.len() && j < y.idx.len() {
                match x.idx[i].cmp(&y.idx[j]) {
                    std::cmp::Ordering::Less => i += 1,
                    std::cmp::Ordering::Greater => j += 1,
                    _ => {
                        s += x.val[i] * y.val[j];
                        i += 1;
                        j += 1;
                    }
                }
            }
            s
        };
        assert!(cos(&a, &c) < 0.3);
        // the projection approximates the cosine
        let (pa, pc) = (project(&a), project(&c));
        assert!((crate::simd::dot(&pa, &pa) - 1.0).abs() < 1e-4);
        assert!((crate::simd::dot(&pa, &pc) - cos(&a, &c)).abs() < 0.25);
    }

    #[test]
    fn neighbors_votes_by_similarity() {
        let m = Neighbors::new(2, vec![1.0, 0.0, 0.0, 1.0, 0.9, 0.436], vec![0, 1, 0]);
        let p = m.proba(&[1.0, 0.0], 2, None);
        assert!(p[0] > 0.7, "{p:?}");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
