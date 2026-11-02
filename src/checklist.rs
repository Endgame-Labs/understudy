//! The `checklist` understudent: an automatically learned decision list over sparse binary features. No encoder.
//!
//! Features of a state (binary, "present or not"): every hashed n-gram bucket `keywords` produces for it (ids
//! `0..BUCKETS`) plus a fixed catalogue of indicator features read off `shape`'s structural vector (ids
//! `BUCKETS..`): per field slot the presence bit, the JSON type, a true boolean and a positive or negative number, and
//! for every field's text (and the whole state's) a length bucket (<= 10, 11-50, 51-200, 201-1000, > 1000 characters),
//! multiple lines, the presence of refusal, apology and hedging markers, URLs, code fences, markdown headers and
//! bullets, question marks and numbers, an echoed number, and mostly-digit or mostly-uppercase text.
//!
//! Induction is greedy sequential covering with one-literal rules ("IF feature THEN class"), CN2/RIPPER-style: among
//! the training rows no earlier rule covered, pick the (feature, class) whose Wilson lower bound of precision is
//! highest (ties: more rows, then the lower feature id, then the lower class), keep it if that bound is at least
//! `MIN_RULE_LB` and it covers at least `MIN_SUPPORT` rows, drop the rows it covers, repeat up to `MAX_RULES`. The
//! rules therefore come ordered by precision on the rows left to them. The default class is the majority of the rows
//! no rule covers. A rule's confidence is its Laplace-smoothed precision `(hits + 1) / (n + 2)` on the rows it
//! took.
//!
//! The understudent ANSWERS only when a rule fires: its prediction is the first firing rule's class at that rule's
//! confidence. When none does it abstains, the request is forwarded (the default class is reported but never answers).
//! Graduation, thresholds and audits are the shared ones over the out-of-fold answers; an abstention is neither
//! agreement nor disagreement.

use crate::shape;
use crate::keywords::{self, Sparse};
use crate::stats::wilson;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// The most rules a list holds.
pub const MAX_RULES: usize = 40;
/// A rule must cover at least this many rows (of those left to it).
pub const MIN_SUPPORT: usize = 5;
/// A rule's Wilson lower bound of precision must reach this.
pub const MIN_RULE_LB: f64 = 0.6;
/// A feature must be present in at least this many training rows to be a candidate.
pub const MIN_DF: usize = 5;
const BUCKETS: u32 = keywords::BUCKETS as u32;

/// One indicator of the catalogue: a test on one column of the structural vector, gated by a presence column.
struct Ind {
    name: String,
    col: usize,
    gate: Option<usize>,
    lo: f32,
    hi: f32,
}

fn catalogue() -> &'static [Ind] {
    static C: OnceLock<Vec<Ind>> = OnceLock::new();
    C.get_or_init(|| {
        let mut v: Vec<Ind> = vec![];
        let ln1p = |n: f32| (n + 1.0).ln() + 1e-4;
        // a text block's indicators: o = the block's first column (its 16 text features), prefix = the names
        let text = |v: &mut Vec<Ind>, o: usize, prefix: &str, gate: Option<usize>| {
            let mut add = |name: &str, col: usize, lo: f32, hi: f32| v.push(Ind { name: format!("{prefix}.{name}"), col: o + col, gate, lo, hi });
            let edges = [10.0, 50.0, 200.0, 1000.0];
            let names = ["chars<=10", "chars 11-50", "chars 51-200", "chars 201-1000", "chars>1000"];
            for (i, name) in names.iter().enumerate() {
                let lo = if i == 0 { f32::NEG_INFINITY } else { ln1p(edges[i - 1]) };
                let hi = if i == 4 { f32::INFINITY } else { ln1p(edges[i]) };
                add(name, 0, lo, hi);
            }
            add("multi-line", 2, 0.9, f32::INFINITY);
            for (name, col) in [("headers", 6), ("bullets", 7), ("code-fence", 8), ("url", 9), ("question-mark", 10), ("refusal", 11), ("apology", 12), ("hedge", 13), ("numbers", 14)] {
                add(name, col, 0.0, f32::INFINITY);
            }
            add("echoes-a-number", 15, 0.5, f32::INFINITY);
            add("mostly-digits", 3, 0.3, f32::INFINITY);
            add("mostly-uppercase", 4, 0.3, f32::INFINITY);
        };
        const TYPES: [&str; 6] = ["null", "bool", "number", "string", "list", "object"];
        for s in 0..shape::NAME_SLOTS {
            let o = s * shape::SLOT_DIM;
            let p = format!("slot{s}");
            v.push(Ind { name: format!("{p}.present"), col: o, gate: None, lo: 0.5, hi: f32::INFINITY });
            for (t, name) in TYPES.iter().enumerate() {
                v.push(Ind { name: format!("{p}.type={name}"), col: o + 1 + t, gate: None, lo: 0.5, hi: f32::INFINITY });
            }
            v.push(Ind { name: format!("{p}.true"), col: o + 8, gate: Some(o), lo: 0.5, hi: f32::INFINITY });
            v.push(Ind { name: format!("{p}.num>0"), col: o + 7, gate: Some(o), lo: 0.0, hi: f32::INFINITY });
            v.push(Ind { name: format!("{p}.num<0"), col: o + 7, gate: Some(o), lo: f32::NEG_INFINITY, hi: -1e-6 });
            text(&mut v, o + 9, &p, Some(o));
        }
        let o = shape::NAME_SLOTS * shape::SLOT_DIM;
        text(&mut v, o, "all", None);
        for (t, name) in TYPES.iter().enumerate() {
            v.push(Ind { name: format!("all.type={name}"), col: o + shape::TEXT_DIM + 1 + t, gate: None, lo: 0.5, hi: f32::INFINITY });
        }
        v
    })
}

/// The number of indicator features.
pub fn n_indicators() -> usize {
    catalogue().len()
}

/// The ascending feature ids a state has: its hashed n-gram buckets, then the indicators that hold.
pub fn row_ids(sp: &Sparse, f: &[f32]) -> Vec<u32> {
    let mut ids = sp.idx.clone();
    for (j, ind) in catalogue().iter().enumerate() {
        let v = f[ind.col];
        // (a threshold test: lo < v <= hi, except a value exactly on the open side, e.g. `num>0` needs v > 0)
        let on = ind.gate.is_none_or(|g| f[g] > 0.5) && v > ind.lo && v <= ind.hi;
        if on {
            ids.push(BUCKETS + j as u32);
        }
    }
    ids
}

/// One rule: IF `shape` is present THEN `class` at confidence `conf` (it took `n` training rows, `hits` of them `class`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub feature: u32,
    pub class: usize,
    pub conf: f32,
    pub n: u32,
    pub hits: u32,
}

/// An ordered list of rules and the class of the rows none covers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Checklist {
    pub rules: Vec<Rule>,
    pub default: usize,
    /// The rules' feature names, in rule order (hashed n-grams as their text when it was known at the fit, else `gram#<bucket>`).
    #[serde(default)]
    pub names: Vec<String>,
}

impl Checklist {
    /// The first rule that fires on a state's feature ids (ascending).
    pub fn firing(&self, ids: &[u32]) -> Option<usize> {
        self.rules.iter().position(|r| ids.binary_search(&r.feature).is_ok())
    }

    /// The answer: the firing rule's class at its confidence, the rest of the mass spread over the other classes;
    /// None (abstain) when no rule fires.
    pub fn proba(&self, ids: &[u32], k: usize) -> Option<Vec<f64>> {
        let r = &self.rules[self.firing(ids)?];
        let rest = (1.0 - r.conf as f64) / (k - 1).max(1) as f64;
        let mut p = vec![rest; k];
        p[r.class] = r.conf as f64;
        Some(p)
    }

    /// A human-readable listing, one rule per line.
    pub fn describe(&self, class_names: &[String]) -> String {
        let name = |c: usize| class_names.get(c).cloned().unwrap_or_else(|| c.to_string());
        let mut s = String::new();
        for (i, r) in self.rules.iter().enumerate() {
            let f = self.names.get(i).cloned().unwrap_or_else(|| feature_name(r.feature, None));
            s.push_str(&format!("{}. IF {} THEN {}   [conf {:.3}, {}/{} rows]\n", i + 1, f, name(r.class), r.conf, r.hits, r.n));
        }
        s.push_str(&format!("default (abstain, never answers): {}\n", name(self.default)));
        s
    }
}

/// A feature's name: an indicator's, a known n-gram's text, else `gram#<bucket>`.
pub fn feature_name(id: u32, gram: Option<&str>) -> String {
    if id >= BUCKETS {
        return catalogue().get((id - BUCKETS) as usize).map_or_else(|| format!("indicator#{id}"), |i| i.name.clone());
    }
    gram.map_or_else(|| format!("gram#{id}"), str::to_string)
}

/// Learn a list from the rows `idx` of `ids` (each row's ascending feature ids) with classes `y` over `k` classes.
pub fn induce(ids: &[Vec<u32>], idx: &[usize], y: &[usize], k: usize) -> Checklist {
    let m = idx.len();
    // compact the candidate features: document frequency among the training rows
    let mut df: HashMap<u32, u32> = HashMap::new();
    for &i in idx {
        for &f in &ids[i] {
            *df.entry(f).or_default() += 1;
        }
    }
    let mut feature_ids: Vec<u32> = df.iter().filter(|(_, c)| **c as usize >= MIN_DF).map(|(&f, _)| f).collect();
    feature_ids.sort_unstable();
    let col: HashMap<u32, usize> = feature_ids.iter().enumerate().map(|(c, &f)| (f, c)).collect();
    let nf = feature_ids.len();
    // rows -> compact features, and the postings (feature -> training positions)
    let rows: Vec<Vec<usize>> = idx.iter().map(|&i| ids[i].iter().filter_map(|f| col.get(f).copied()).collect()).collect();
    let mut post: Vec<Vec<u32>> = vec![vec![]; nf];
    let mut counts = vec![0u32; nf * k];
    for (r, fs) in rows.iter().enumerate() {
        for &c in fs {
            post[c].push(r as u32);
            counts[c * k + y[idx[r]]] += 1;
        }
    }
    let mut alive = vec![true; m];
    let mut out = Checklist::default();
    for _ in 0..MAX_RULES {
        // (lower bound, rows, -feature, -class): the best rule among the rows left
        let mut best: Option<(f64, u32, usize, usize)> = None;
        for c in 0..nf {
            let cc = &counts[c * k..(c + 1) * k];
            let tot: u32 = cc.iter().sum();
            if (tot as usize) < MIN_SUPPORT {
                continue;
            }
            for class in 0..k {
                let hits = cc[class];
                if hits == 0 {
                    continue;
                }
                let lb = wilson(hits as usize, tot as usize).0;
                let better = match best {
                    None => true,
                    Some((bl, bt, bc, bk)) => lb > bl || (lb == bl && (tot > bt || (tot == bt && (c < bc || (c == bc && class < bk))))),
                };
                if better {
                    best = Some((lb, tot, c, class));
                }
            }
        }
        let Some((lb, tot, c, class)) = best else { break };
        if lb < MIN_RULE_LB {
            break;
        }
        let hits = counts[c * k + class];
        out.rules.push(Rule { feature: feature_ids[c], class, conf: (hits as f32 + 1.0) / (tot as f32 + 2.0), n: tot, hits });
        for &r in &post[c] {
            let r = r as usize;
            if !alive[r] {
                continue;
            }
            alive[r] = false;
            for &g in &rows[r] {
                counts[g * k + y[idx[r]]] -= 1;
            }
        }
    }
    let mut left = vec![0usize; k];
    for (r, &i) in idx.iter().enumerate() {
        if alive[r] {
            left[y[i]] += 1;
        }
    }
    if left.iter().all(|&c| c == 0) {
        idx.iter().for_each(|&i| left[y[i]] += 1);
    }
    out.default = crate::stats::argmax(&left.iter().map(|&c| c as f64).collect::<Vec<_>>());
    out
}

/// The text of hashed n-gram buckets, from the texts that produced them: for each wanted bucket the most frequent
/// n-gram (ties: the lexicographically smallest) among `texts`, annotated when several distinct n-grams collide there.
/// `texts` are the states' full texts as `keywords::featurize` saw them (one per pair field).
pub fn gram_names(states: &[&[String]], want: &HashSet<u32>) -> HashMap<u32, String> {
    let mut seen: HashMap<(u32, String), usize> = HashMap::new();
    for texts in states {
        for (pi, t) in texts.iter().enumerate() {
            keywords::for_each_gram(t, pi, want, |bucket, desc| *seen.entry((bucket, desc)).or_default() += 1);
        }
    }
    let mut by_bucket: HashMap<u32, Vec<(usize, String)>> = HashMap::new();
    for ((b, d), c) in seen {
        by_bucket.entry(b).or_default().push((c, d));
    }
    by_bucket
        .into_iter()
        .map(|(b, mut v)| {
            v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let more = v.len() - 1;
            (b, if more > 0 { format!("{} (+{more} colliding)", v[0].1) } else { v[0].1.clone() })
        })
        .collect()
}

/// Name a list's rules: indicators by their catalogue name, n-grams by their text when the training texts are known.
pub fn name_rules(list: &mut Checklist, states: &[&[String]]) {
    let want: HashSet<u32> = list.rules.iter().map(|r| r.feature).filter(|&f| f < BUCKETS).collect();
    let grams = if want.is_empty() || states.is_empty() { HashMap::new() } else { gram_names(states, &want) };
    list.names = list.rules.iter().map(|r| feature_name(r.feature, grams.get(&r.feature).map(String::as_str))).collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_columns_exist_and_names_are_unique() {
        let c = catalogue();
        assert!(c.iter().all(|i| i.col < shape::DIM && i.gate.is_none_or(|g| g < shape::DIM)));
        let names: HashSet<&str> = c.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names.len(), c.len());
        let z = vec![0f32; shape::DIM];
        let ids = row_ids(&Sparse::default(), &z);
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "ids must be ascending");
        // an all-zero vector has no present slot, so no slot indicator fires
        assert!(ids.iter().all(|&id| !feature_name(id, None).starts_with("slot")), "{ids:?}");
    }

    #[test]
    fn induces_the_marker_rule_and_abstains_elsewhere() {
        // class 1 rows carry feature 7, class 0 rows carry feature 9; some noise features everywhere
        let mut ids = vec![];
        let mut y = vec![];
        for i in 0..80usize {
            let c = i % 2;
            let mut v = vec![if c == 1 { 7 } else { 9 }];
            v.push(100 + (i % 5) as u32);
            v.sort_unstable();
            ids.push(v);
            y.push(c);
        }
        let idx: Vec<usize> = (0..80).collect();
        let l = induce(&ids, &idx, &y, 2);
        assert_eq!(l.rules.len(), 2, "{:?}", l.rules);
        assert!(l.rules.iter().any(|r| r.feature == 7 && r.class == 1) && l.rules.iter().any(|r| r.feature == 9 && r.class == 0));
        assert!(l.proba(&[7, 100], 2).unwrap()[1] > 0.9);
        assert!(l.proba(&[55], 2).is_none(), "no rule fires: abstain");
    }
}
