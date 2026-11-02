//! Interval estimates, folds and agreement statistics, as hunch computes them.

use crate::rng::Mt19937;

/// 95% Wilson interval (low, high) for `k` successes in `m` trials (z = 1.96); (0, 1) for no trials.
pub fn wilson(k: usize, m: usize) -> (f64, f64) {
    wilson_z(k, m, 1.96)
}

pub fn wilson_z(k: usize, m: usize, z: f64) -> (f64, f64) {
    if m == 0 {
        return (0.0, 1.0);
    }
    let (k, m) = (k as f64, m as f64);
    let p = k / m;
    let c = (p + z * z / (2.0 * m)) / (1.0 + z * z / m);
    let h = z * (p * (1.0 - p) / m + z * z / (4.0 * m * m)).sqrt() / (1.0 + z * z / m);
    (c - h, c + h)
}

/// The standard normal quantile (Acklam's rational approximation, relative error about 1e-9).
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// The base two-sided error level of the 95% bar (z = 1.96).
pub const BASE_ALPHA: f64 = 0.05;

/// The Wilson z of attempt `attempt` (1-based) under alpha-spending across graduation attempts, with `m` candidates
/// tested in that attempt (Bonferroni): the two-sided level `alpha_k / m`, `alpha_k = 0.05 * 6 / (pi^2 k^2)`, so the
/// levels of all attempts sum to at most 0.05. Attempt 0 is read as attempt 1.
pub fn sequential_z(attempt: usize, m: usize) -> f64 {
    let k = attempt.max(1) as f64;
    let alpha_k = BASE_ALPHA * 6.0 / (std::f64::consts::PI * std::f64::consts::PI * k * k);
    -norm_ppf(alpha_k / 2.0 / m.max(1) as f64)
}

/// scikit-learn's `KFold(k, shuffle=True, random_state=seed).split(range(n))`: (train, test) index lists. The
/// first `n % k` folds hold one extra item. Test indices come sorted, as scikit-learn returns them.
pub fn kfold(n: usize, k: usize, seed: u32) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut idx: Vec<usize> = (0..n).collect();
    Mt19937::numpy(seed).shuffle(&mut idx);
    let mut out = Vec::with_capacity(k);
    let mut start = 0;
    for f in 0..k {
        let size = n / k + usize::from(f < n % k);
        let mut test = idx[start..start + size].to_vec();
        test.sort_unstable();
        let mut is_test = vec![false; n];
        test.iter().for_each(|&i| is_test[i] = true);
        let train = (0..n).filter(|&i| !is_test[i]).collect();
        out.push((train, test));
        start += size;
    }
    out
}

/// `kfold` that never splits a group: rows sharing a `Some(group)` stay in one fold (a `None` row is its own group).
/// With no grouped row it is exactly `kfold`. Otherwise the groups (in order of first appearance) are shuffled with
/// the seeded generator and dealt one by one to the fold holding the fewest rows so far (ties to the lowest fold), as
/// scikit-learn's `GroupKFold` balances; test indices come sorted. Needs at least `k` groups to fill every fold.
pub fn group_kfold(groups: &[Option<u64>], k: usize, seed: u32) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = groups.len();
    if groups.iter().all(Option::is_none) {
        return kfold(n, k, seed);
    }
    let mut units: Vec<Vec<usize>> = vec![];
    let mut at: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        match g {
            Some(g) => match at.get(g) {
                Some(&u) => units[u].push(i),
                None => {
                    at.insert(*g, units.len());
                    units.push(vec![i]);
                }
            },
            None => units.push(vec![i]),
        }
    }
    let mut order: Vec<usize> = (0..units.len()).collect();
    Mt19937::numpy(seed).shuffle(&mut order);
    let mut tests: Vec<Vec<usize>> = vec![vec![]; k];
    for u in order {
        let f = (0..k).min_by_key(|&f| (tests[f].len(), f)).unwrap();
        tests[f].extend(&units[u]);
    }
    tests
        .into_iter()
        .map(|mut test| {
            test.sort_unstable();
            let mut is_test = vec![false; n];
            test.iter().for_each(|&i| is_test[i] = true);
            ((0..n).filter(|&i| !is_test[i]).collect(), test)
        })
        .collect()
}

/// numpy's default (linear) quantile of `xs`, `q` in [0, 1].
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// Cohen's kappa between two labelings over `k` classes; 1.0 when chance agreement is already total.
pub fn kappa(a: &[usize], b: &[usize], k: usize) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if a.is_empty() {
        return f64::NAN;
    }
    let (mut ca, mut cb, mut agree) = (vec![0.0; k], vec![0.0; k], 0.0);
    for (&x, &y) in a.iter().zip(b) {
        ca[x] += 1.0;
        cb[y] += 1.0;
        if x == y {
            agree += 1.0;
        }
    }
    let po = agree / n;
    let pe: f64 = ca.iter().zip(&cb).map(|(x, y)| x * y).sum::<f64>() / (n * n);
    if pe >= 1.0 { 1.0 } else { (po - pe) / (1.0 - pe) }
}

/// The first index of the largest value (numpy's `argmax`).
pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > p[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_matches_hunch() {
        // hunch.proxy.wilson(27, 30) and wilson(0, 50)
        let (lo, hi) = wilson(27, 30);
        assert!((lo - 0.7438).abs() < 1e-4 && (hi - 0.9654).abs() < 1e-4, "{lo} {hi}");
        let (lo, hi) = wilson(0, 50);
        assert!(lo.abs() < 1e-12 && (hi - 0.07135).abs() < 1e-4, "{hi}");
        assert_eq!(wilson(0, 0), (0.0, 1.0));
        let (lo, hi) = wilson(50, 50);
        assert!((hi - 1.0).abs() < 1e-12 && (lo - 0.92865).abs() < 1e-4);
    }

    #[test]
    fn group_kfold_keeps_groups_whole_and_matches_kfold_when_ungrouped() {
        let none = vec![None; 23];
        assert_eq!(group_kfold(&none, 5, 0), kfold(23, 5, 0));
        // 30 groups of 3 siblings (ids interleaved), 5 folds
        let groups: Vec<Option<u64>> = (0..90).map(|i| Some((i % 30) as u64)).collect();
        let folds = group_kfold(&groups, 5, 0);
        let mut seen = vec![0; 90];
        for (tr, te) in &folds {
            assert_eq!(tr.len() + te.len(), 90);
            for &i in te {
                seen[i] += 1;
            }
            // no group has rows on both sides of a split
            for g in 0..30u64 {
                let in_test = te.iter().filter(|&&i| groups[i] == Some(g)).count();
                assert!(in_test == 0 || in_test == 3, "group {g} split");
            }
            assert_eq!(te.len(), 18);
        }
        assert!(seen.iter().all(|&c| c == 1));
        // deterministic, and a different seed deals differently
        assert_eq!(group_kfold(&groups, 5, 0), folds);
        assert_ne!(group_kfold(&groups, 5, 1), folds);
        // None rows mix with groups, each its own unit
        let mixed = vec![Some(1), Some(1), None, None, Some(2), Some(2), None, None, None, None];
        for (_, te) in group_kfold(&mixed, 2, 0) {
            for g in [1u64, 2] {
                let c = te.iter().filter(|&&i| mixed[i] == Some(g)).count();
                assert!(c == 0 || c == 2);
            }
        }
    }

    #[test]
    fn sequential_z_spends_alpha_over_attempts_and_candidates() {
        // attempt 1 spends 0.05 * 6 / pi^2 = 0.0304 two-sided
        assert!((sequential_z(1, 1) - 2.1642).abs() < 2e-3, "{}", sequential_z(1, 1));
        assert_eq!(sequential_z(0, 1), sequential_z(1, 1));
        let zs: Vec<f64> = (1..=30).map(|k| sequential_z(k, 1)).collect();
        assert!(zs.windows(2).all(|w| w[1] > w[0]));
        assert!(sequential_z(5, 4) > sequential_z(5, 1));
        // the spent levels sum to at most the base alpha
        let total: f64 = (1..=100000).map(|k: usize| BASE_ALPHA * 6.0 / (std::f64::consts::PI.powi(2) * (k * k) as f64)).sum();
        assert!(total <= BASE_ALPHA);
        // the normal quantile
        assert!((norm_ppf(0.975) - 1.96).abs() < 1e-3);
    }

    #[test]
    fn kfold_partitions() {
        let folds = kfold(23, 5, 0);
        let sizes: Vec<usize> = folds.iter().map(|(_, t)| t.len()).collect();
        assert_eq!(sizes, vec![5, 5, 5, 4, 4]);
        let mut all: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..23).collect::<Vec<_>>());
        for (tr, te) in &folds {
            assert_eq!(tr.len() + te.len(), 23);
        }
    }

    #[test]
    fn quantile_and_kappa() {
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.05) - 1.15).abs() < 1e-12);
        assert!((kappa(&[0, 1, 0, 1], &[0, 1, 0, 1], 2) - 1.0).abs() < 1e-12);
        assert!(kappa(&[0, 1, 0, 1], &[1, 0, 1, 0], 2) < -0.99);
    }
}
