//! Abstention policies: how a fit turns predictions into "answer locally" versus "forward to the lead".
//!
//! `single` (the default, hunch's rule) is `understudy::graduation`'s symmetric scan over out-of-fold predictions: ONE
//! confidence threshold, the lowest `t` in 0.50, 0.51, ..., 0.99 whose answered share meets the bars. (Known limitation,
//! unchanged: its out-of-fold confidences are tallied without the novelty gate that serving applies, so the answered
//! population it certifies is not exactly the one served; see SPEC.md.)
//!
//! `calibrated` is a **certified frozen policy**, built so that the certificate measures exactly what is served:
//!
//! 1. The labeled rows are split into three GROUP-DISJOINT parts, TRAIN / SELECT / CERTIFY (about 60/20/20, by group,
//!    seeded; `three_way`).
//! 2. The model is fitted on TRAIN alone: all preprocessing, early stopping and the novelty reference set and floor.
//! 3. The per-class thresholds are chosen on SELECT (`choose`), with the novelty gate already applied. A class with too
//!    little SELECT support (`MIN_CLASS_SUPPORT` answered-eligible rows predicted as it) gets an INFINITE threshold:
//!    it is never answered.
//! 4. The exact frozen serving predicate (model, thresholds, novelty floor) is evaluated ONCE on CERTIFY. The policy
//!    graduates only if the Wilson lower bounds on CERTIFY's answered rows meet the bars, with the usual minimum counts.
//! 5. Serving uses that frozen model and those thresholds; nothing is refitted after certification. A later refit is a
//!    new model and is certified afresh by the same procedure.
//!
//! When several candidates are certified against the same CERTIFY rows (a ladder's tiers, auto's candidates), each must
//! pass at a Bonferroni-corrected z (`bonferroni_z`), so the choice among them cannot be a selection on CERTIFY noise.

use crate::rng::Xoshiro;
use crate::stats::{norm_ppf, wilson_z};
use serde::{Deserialize, Serialize};

/// The thresholds tried per class, as the single policy's scan.
pub fn grid() -> impl Iterator<Item = f64> {
    (0..50).map(|i| 0.5 + 0.01 * i as f64)
}

/// The least number of SELECT rows a class must be predicted on (not novel, not an abstention) for it to get a
/// threshold at all; a class with less support has an infinite threshold and is never answered.
pub const MIN_CLASS_SUPPORT: usize = 10;

/// The Wilson z of the shared 95% bar.
pub const BAR_Z: f64 = 1.96;

/// Which abstention policy decides graduation (`--abstain`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum Abstain {
    /// One confidence threshold (hunch's rule).
    #[default]
    #[serde(rename = "single")]
    #[value(name = "default")]
    Single,
    /// A frozen per-class policy, selected on one part of the data and certified on another (module docs).
    #[serde(rename = "calibrated")]
    #[value(name = "calibrated")]
    Calibrated,
}

impl Abstain {
    pub fn is_default(&self) -> bool {
        *self == Abstain::Single
    }
}

/// serde for `Fit::class_thr`: an infinite threshold (a class that is never answered) is a JSON null.
pub mod thr_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<Vec<f64>>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_some(&v.as_ref().map(|v| v.iter().map(|&t| t.is_finite().then_some(t)).collect::<Vec<Option<f64>>>()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f64>>, D::Error> {
        let v: Option<Vec<Option<f64>>> = Option::deserialize(d)?;
        Ok(v.map(|v| v.into_iter().map(|t| t.unwrap_or(f64::INFINITY)).collect()))
    }
}

/// Counts over an answered set: answered, agreed, minority answered, minority agreed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub sel: usize,
    pub agree: usize,
    pub ms: usize,
    pub ms_agree: usize,
}

impl Tally {
    pub fn add(&mut self, agree: bool, minor: bool) {
        self.sel += 1;
        self.agree += usize::from(agree);
        if minor {
            self.ms += 1;
            self.ms_agree += usize::from(agree);
        }
    }

    /// The symmetric graduation bars of `understudy::graduation`, verbatim.
    pub fn passes(&self, target: f64, minority: f64) -> bool {
        self.passes_z(target, minority, BAR_Z)
    }

    /// The same bars with the Wilson bounds at another z (`passes` is z = 1.96).
    pub fn passes_z(&self, target: f64, minority: f64, z: f64) -> bool {
        self.sel >= 30 && wilson_z(self.agree, self.sel, z).0 >= target && self.ms >= 15 && wilson_z(self.ms_agree, self.ms, z).0 >= minority
    }
}

/// The z of a two-sided 95% bar spent over `m` candidates (Bonferroni: the two-sided level 0.05 / m); exactly the
/// plain bar's 1.96 for `m <= 1`.
pub fn bonferroni_z(m: usize) -> f64 {
    if m <= 1 {
        return BAR_Z;
    }
    -norm_ppf(0.025 / m as f64)
}

/// The three group-disjoint parts of the labeled rows (row indices, ascending).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub train: Vec<usize>,
    pub select: Vec<usize>,
    pub certify: Vec<usize>,
}

/// Parts' target shares: TRAIN, SELECT, CERTIFY.
const SHARES: [f64; 3] = [0.6, 0.2, 0.2];

/// A seeded group-disjoint 60/20/20 split: rows sharing a group (a `None` row is its own group) stay together; the
/// groups are shuffled and dealt one by one to the part furthest under its share. None with fewer than 3 groups
/// (each part needs one).
pub fn three_way(groups: &[Option<u64>], seed: u64) -> Option<Split> {
    let mut units: Vec<Vec<usize>> = vec![];
    let mut at: std::collections::HashMap<u64, usize> = Default::default();
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
    if units.len() < 3 {
        return None;
    }
    Xoshiro::new(seed ^ 0x7e57_5917).shuffle(&mut units);
    let mut parts: [Vec<usize>; 3] = Default::default();
    for u in units {
        let p = (0..3).min_by(|&a, &b| (parts[a].len() as f64 / SHARES[a]).total_cmp(&(parts[b].len() as f64 / SHARES[b])).then(a.cmp(&b))).unwrap();
        parts[p].extend(u);
    }
    let [mut train, mut select, mut certify] = parts;
    train.sort_unstable();
    select.sort_unstable();
    certify.sort_unstable();
    Some(Split { train, select, certify })
}

/// The predictions a policy reads: the predicted class, the confidence of the answer the policy would give (-inf for a
/// row it never answers: an abstention or a novel row), whether it matched the lead and whether the lead's class is not
/// the majority.
pub struct Preds<'a> {
    pub pred: &'a [usize],
    pub cf: &'a [f64],
    pub agree: &'a [bool],
    pub minor: &'a [bool],
}

impl Preds<'_> {
    pub fn tally(&self, rows: &[usize], thr: &[f64]) -> Tally {
        let mut t = Tally::default();
        for &j in rows {
            if self.cf[j] >= thr[self.pred[j]] {
                t.add(self.agree[j], self.minor[j]);
            }
        }
        t
    }
}

/// The thresholds (one per class of `k`) maximizing the answered share of `rows` (SELECT) subject to the bars; None when
/// no choice passes. A class with fewer than `MIN_CLASS_SUPPORT` answerable rows predicted as it (and a class never
/// predicted) gets an infinite threshold: it is never answered. Two supported classes are searched exhaustively
/// (50 x 50); one by a scan; more by coordinate ascent from the strictest thresholds. Ties go to the stricter thresholds.
pub fn choose(p: &Preds, rows: &[usize], k: usize, target: f64, minority: f64) -> Option<Vec<f64>> {
    let mut support = vec![0usize; k];
    for &j in rows {
        if p.cf[j].is_finite() {
            support[p.pred[j]] += 1;
        }
    }
    let active: Vec<usize> = (0..k).filter(|&c| support[c] >= MIN_CLASS_SUPPORT).collect();
    let strict = 0.99;
    let mut thr = vec![f64::INFINITY; k];
    for &c in &active {
        thr[c] = strict;
    }
    let score = |thr: &[f64]| {
        let t = p.tally(rows, thr);
        t.passes(target, minority).then_some(t.sel)
    };
    let mut best: Option<(usize, f64, Vec<f64>)> = None;
    let consider = |thr: &[f64], sel: usize, best: &mut Option<(usize, f64, Vec<f64>)>| {
        let sum: f64 = active.iter().map(|&c| thr[c]).sum();
        if best.as_ref().is_none_or(|b| sel > b.0 || (sel == b.0 && sum > b.1 + 1e-12)) {
            *best = Some((sel, sum, thr.to_vec()));
        }
    };
    match active.len() {
        0 => return None,
        1 | 2 => {
            let (a, b) = (active[0], active.get(1).copied());
            for ta in grid() {
                thr[a] = ta;
                match b {
                    None => {
                        if let Some(s) = score(&thr) {
                            consider(&thr, s, &mut best);
                        }
                    }
                    Some(b) => {
                        for tb in grid() {
                            thr[b] = tb;
                            if let Some(s) = score(&thr) {
                                consider(&thr, s, &mut best);
                            }
                        }
                    }
                }
            }
        }
        _ => {
            for _ in 0..6 {
                let before = thr.clone();
                for &c in &active {
                    let mut local: Option<(usize, f64)> = None;
                    for t in grid() {
                        let mut cand = thr.clone();
                        cand[c] = t;
                        if let Some(s) = score(&cand) {
                            if local.is_none_or(|l| s > l.0 || (s == l.0 && t > l.1)) {
                                local = Some((s, t));
                            }
                        }
                    }
                    if let Some((_, t)) = local {
                        thr[c] = t;
                    }
                }
                if thr == before {
                    break;
                }
            }
            if let Some(s) = score(&thr) {
                consider(&thr, s, &mut best);
            }
        }
    }
    best.map(|b| b.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Data = (Vec<usize>, Vec<f64>, Vec<bool>, Vec<bool>);

    /// Class 0 is easy (confident and right), class 1 is confidently wrong often.
    fn data(n: usize, seed: u64) -> Data {
        let mut r = Xoshiro::new(seed);
        let (mut pred, mut cf, mut agree, mut minor) = (vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            let class = usize::from(r.uniform() < 0.4);
            let c = 0.5 + 0.5 * r.uniform();
            let p_ok = if class == 0 { 0.5 + c / 2.0 } else { 0.6 };
            pred.push(class);
            cf.push(c);
            agree.push(r.uniform() < p_ok);
            minor.push(class == 1);
        }
        (pred, cf, agree, minor)
    }

    /// Confidence carries NO information: agreement is a coin with a fixed rate below any bar, so no policy is reliable.
    fn noise(n: usize, seed: u64, rate: f64) -> Data {
        let mut r = Xoshiro::new(seed);
        let (mut pred, mut cf, mut agree, mut minor) = (vec![], vec![], vec![], vec![]);
        for _ in 0..n {
            let class = usize::from(r.uniform() < 0.4);
            pred.push(class);
            cf.push(0.5 + 0.5 * r.uniform());
            agree.push(r.uniform() < rate);
            minor.push(class == 1);
        }
        (pred, cf, agree, minor)
    }

    fn preds(d: &Data) -> Preds<'_> {
        Preds { pred: &d.0, cf: &d.1, agree: &d.2, minor: &d.3 }
    }

    #[test]
    fn a_choice_always_meets_every_bar() {
        for seed in 0..6 {
            let d = data(600, seed);
            let p = preds(&d);
            let rows: Vec<usize> = (0..600).collect();
            if let Some(thr) = choose(&p, &rows, 2, 0.8, 0.5) {
                assert!(p.tally(&rows, &thr).passes(0.8, 0.5), "seed {seed}");
            }
        }
    }

    #[test]
    fn no_choice_without_evidence() {
        // 25 rows can never reach 30 answered
        let d = data(25, 1);
        let rows: Vec<usize> = (0..25).collect();
        assert!(choose(&preds(&d), &rows, 2, 0.5, 0.0).is_none());
    }

    #[test]
    fn classes_without_select_support_are_never_answered() {
        // class 2 is predicted on 5 rows (a perfect record), class 3 never; classes 0 and 1 are supported
        let (mut pred, mut cf, mut agree, mut minor) = (vec![], vec![], vec![], vec![]);
        for i in 0..200 {
            pred.push(i % 2);
            cf.push(0.9);
            agree.push(true);
            minor.push(i % 2 == 1);
        }
        for _ in 0..5 {
            pred.push(2);
            cf.push(1.0);
            agree.push(true);
            minor.push(true);
        }
        let d = (pred, cf, agree, minor);
        let rows: Vec<usize> = (0..d.0.len()).collect();
        let thr = choose(&preds(&d), &rows, 4, 0.8, 0.5).unwrap();
        assert!(thr[0].is_finite() && thr[1].is_finite());
        assert!(thr[2].is_infinite() && thr[3].is_infinite(), "{thr:?}");
        // and an infinite threshold answers nothing, whatever the confidence
        assert_eq!(preds(&d).tally(&rows, &thr).sel, 200);
        // abstentions and novel rows (cf = -inf) are no support either
        let mut d2 = d.clone();
        d2.1.iter_mut().zip(&d.0).for_each(|(c, &p)| {
            if p == 1 {
                *c = f64::NEG_INFINITY;
            }
        });
        let thr = choose(&preds(&d2), &rows, 4, 0.5, 0.0);
        assert!(thr.is_none_or(|t| t[1].is_infinite()), "a class that is never answerable has no threshold");
    }

    #[test]
    fn the_split_is_group_disjoint_seeded_and_about_60_20_20() {
        let groups: Vec<Option<u64>> = (0..600).map(|i| Some((i % 200) as u64)).collect();
        let s = three_way(&groups, 7).unwrap();
        assert_eq!(three_way(&groups, 7).unwrap(), s);
        assert_ne!(three_way(&groups, 8).unwrap(), s);
        let mut all: Vec<usize> = [&s.train, &s.select, &s.certify].iter().flat_map(|p| p.iter().copied()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..600).collect::<Vec<_>>(), "a partition");
        let owner = |i: usize| if s.train.contains(&i) { 0 } else if s.select.contains(&i) { 1 } else { 2 };
        for g in 0..200usize {
            let o: Vec<usize> = (0..3).map(|k| owner(g + 200 * k)).collect();
            assert!(o.iter().all(|&x| x == o[0]), "group {g} straddles parts");
        }
        assert_eq!((s.train.len(), s.select.len(), s.certify.len()), (360, 120, 120));
        // ungrouped rows are their own groups; fewer than 3 groups cannot fill 3 parts
        assert!(three_way(&vec![None; 50], 0).is_some());
        assert!(three_way(&[Some(1), Some(1), Some(2), Some(2)], 0).is_none());
    }

    #[test]
    fn bonferroni_z_spends_the_error_budget() {
        assert_eq!(bonferroni_z(0), 1.96);
        assert_eq!(bonferroni_z(1), 1.96);
        assert!((bonferroni_z(2) - 2.2414).abs() < 1e-3, "{}", bonferroni_z(2));
        assert!((bonferroni_z(5) - 2.5758).abs() < 1e-3, "{}", bonferroni_z(5));
        assert!((bonferroni_z(4) - 2.4977).abs() < 1e-3, "{}", bonferroni_z(4));
        assert!(bonferroni_z(8) > bonferroni_z(4));
    }

    /// The old calibrated code certified thresholds on the same rows (or rows that had already shaped them); here the
    /// agreement is a coin that never reaches the bar, whatever the confidence. Searching thresholds on the rows that
    /// then certify them "passes" often; selecting on one part and certifying on a disjoint part almost never does.
    #[test]
    fn a_search_that_over_certifies_in_sample_is_refused_on_disjoint_rows() {
        // true agreement 0.80 against a bar of 0.80 on the Wilson LOWER bound: no threshold is reliable
        let (rate, target, minority, n) = (0.80, 0.80, 0.5, 240usize);
        let (mut old_hits, mut new_hits) = (0, 0);
        for seed in 0..300u64 {
            let d = noise(n, seed, rate);
            let p = preds(&d);
            let all: Vec<usize> = (0..n).collect();
            // old: choose on every row, certify those same rows
            if let Some(thr) = choose(&p, &all, 2, target, minority) {
                old_hits += usize::from(p.tally(&all, &thr).passes(target, minority));
            }
            // new: SELECT = the first half, CERTIFY = the disjoint second half (rows are exchangeable here)
            let (select, certify): (Vec<usize>, Vec<usize>) = ((0..n / 2).collect(), (n / 2..n).collect());
            if let Some(thr) = choose(&p, &select, 2, target, minority) {
                new_hits += usize::from(p.tally(&certify, &thr).passes(target, minority));
            }
        }
        // measured: old 58 of 300 (19%), new 2 of 300 (under 1%, the 95% bar's own false-pass rate)
        assert!(old_hits >= 30, "the in-sample search must over-certify on this data: {old_hits}");
        assert!(new_hits <= 8 && new_hits * 5 <= old_hits, "disjoint certification must refuse nearly all of them: old {old_hits}, new {new_hits}");
    }
}
