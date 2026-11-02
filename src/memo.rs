//! The `memo` understudent: retrieval over the lead's past answers. No model and no encoder.
//!
//! A state's key is the hash of its whole text (exact match) and a MinHash signature of its character 5-gram set
//! (`NSIG` hashes; the share of equal positions estimates the Jaccard similarity). The model is the training window's
//! keys with their lead answers. For a new state it finds the stored states at similarity >= `thr_sim` (an exact
//! match counts 1.0): if there are none, or they disagree, it abstains (the request is forwarded); otherwise it
//! returns the lead answer of the most similar one (the newest among equals). The probabilities are rebuilt from the
//! stored label and the lead's confidence (the example weight; 1.0 for a labelled example), which is exact for a
//! two-class question.
//!
//! `thr_sim` is chosen by the fit's own cross-validation over `GRID`: the value whose out-of-fold answers graduate with
//! the most coverage (ties: the stricter), else `DEFAULT_SIM`. Graduation then is the shared Wilson test on the
//! out-of-fold agreement over the rows memo answers.

use serde::{Deserialize, Serialize};

pub const NSIG: usize = 128;
pub const SHINGLE: usize = 5;
pub const DEFAULT_SIM: f64 = 0.95;
/// The similarity thresholds the fit's CV chooses from (1.0 = identical signature only).
pub const GRID: [f64; 5] = [0.80, 0.85, 0.90, 0.95, 1.0];
/// The weakest similarity worth listing as a neighbour.
pub const MIN_SIM: f64 = GRID[0];

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn salts() -> &'static [u64; NSIG] {
    static S: std::sync::OnceLock<[u64; NSIG]> = std::sync::OnceLock::new();
    S.get_or_init(|| std::array::from_fn(|i| mix(i as u64 + 0x9e3779b97f4a7c15)))
}

/// A state's retrieval key.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoKey {
    /// The hash of the whole state text.
    pub exact: u64,
    pub sig: Vec<u32>,
}

impl MemoKey {
    pub fn from_text(text: &str) -> Self {
        let exact = u64::from_str_radix(&crate::encoder::text_hash(text), 16).unwrap_or(0);
        let chars: Vec<char> = text.chars().collect();
        let mut hs: Vec<u64> = if chars.len() < SHINGLE {
            vec![shingle(&chars)]
        } else {
            chars.windows(SHINGLE).map(shingle).collect()
        };
        hs.sort_unstable();
        hs.dedup();
        let salts = salts();
        let sig = (0..NSIG).map(|i| hs.iter().map(|&h| (mix(h ^ salts[i]) >> 32) as u32).min().unwrap_or(u32::MAX)).collect();
        MemoKey { exact, sig }
    }

    /// The estimated Jaccard similarity: 1.0 for an exact match, else the share of equal signature positions.
    pub fn similarity(&self, other: &MemoKey) -> f64 {
        if self.exact == other.exact {
            return 1.0;
        }
        self.sig.iter().zip(&other.sig).filter(|(a, b)| a == b).count() as f64 / NSIG as f64
    }
}

fn shingle(w: &[char]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &c in w {
        h = (h ^ c as u64).wrapping_mul(0x100000001b3);
    }
    mix(h)
}

/// The stored keys with their lead answers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoModel {
    pub thr_sim: f64,
    pub exact: Vec<u64>,
    /// [n x NSIG]
    pub sig: Vec<u32>,
    pub y: Vec<u32>,
    /// The lead's confidence in `y`.
    pub conf: Vec<f32>,
}

impl MemoModel {
    pub fn new(keys: &[&MemoKey], y: &[usize], w: &[f64], thr_sim: f64) -> Self {
        MemoModel {
            thr_sim,
            exact: keys.iter().map(|k| k.exact).collect(),
            sig: keys.iter().flat_map(|k| k.sig.iter().copied()).collect(),
            y: y.iter().map(|&c| c as u32).collect(),
            conf: w.iter().map(|&x| if x >= 2.0 { 1.0 } else { x.min(1.0) as f32 }).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.y.len()
    }

    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// Stored rows (of `rows`, or all) at similarity >= `min_sim` to `q`, most similar first; equal similarities put
    /// the newer (larger index) first.
    pub fn neighbours(&self, q: &MemoKey, min_sim: f64, rows: Option<&[usize]>) -> Vec<(f64, usize)> {
        let allowed = ((1.0 - min_sim) * NSIG as f64 + 1e-9).floor() as usize;
        let mut out = vec![];
        let mut visit = |i: usize| {
            if self.exact[i] == q.exact {
                out.push((1.0, i));
                return;
            }
            let row = &self.sig[i * NSIG..(i + 1) * NSIG];
            let mut miss = 0;
            for (a, b) in row.iter().zip(&q.sig) {
                if a != b {
                    miss += 1;
                    if miss > allowed {
                        return;
                    }
                }
            }
            out.push(((NSIG - miss) as f64 / NSIG as f64, i));
        };
        match rows {
            Some(r) => r.iter().for_each(|&i| visit(i)),
            None => (0..self.len()).for_each(&mut visit),
        }
        out.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
        out
    }

    /// The answer from sorted `neighbours`: those at similarity >= `thr`, all agreeing; the best one's lead answer.
    /// None (abstain) when there are none or they disagree.
    pub fn answer(&self, neighbours: &[(f64, usize)], thr: f64, k: usize) -> Option<Vec<f64>> {
        let near: Vec<&(f64, usize)> = neighbours.iter().take_while(|n| n.0 >= thr).collect();
        let first = near.first()?;
        let label = self.y[first.1];
        if near.iter().any(|n| self.y[n.1] != label) {
            return None;
        }
        let conf = self.conf[first.1] as f64;
        let rest = if k > 1 { (1.0 - conf) / (k - 1) as f64 } else { 0.0 };
        let mut p = vec![rest; k];
        p[label as usize] = conf;
        Some(p)
    }

    pub fn proba(&self, q: &MemoKey, k: usize) -> Option<Vec<f64>> {
        self.answer(&self.neighbours(q, self.thr_sim, None), self.thr_sim, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(texts: &[&str], y: &[usize], thr: f64) -> MemoModel {
        let keys: Vec<MemoKey> = texts.iter().map(|t| MemoKey::from_text(t)).collect();
        MemoModel::new(&keys.iter().collect::<Vec<_>>(), y, &vec![0.9; y.len()], thr)
    }

    const A: &str = "The deployment to production failed with exit code 2 because the migration timed out after 300 seconds.";

    #[test]
    fn an_exact_match_returns_the_stored_lead_answer() {
        let m = model(&[A, "Something else entirely: the weather is nice today and the park is open."], &[1, 0], 0.95);
        let p = m.proba(&MemoKey::from_text(A), 2).expect("exact match answers");
        assert!((p[1] - 0.9).abs() < 1e-6 && (p[0] - 0.1).abs() < 1e-6, "{p:?}");
        // the same text, again, is the same key
        assert_eq!(MemoKey::from_text(A), MemoKey::from_text(A));
    }

    #[test]
    fn a_near_duplicate_answers_and_a_different_state_abstains() {
        let m = model(&[A], &[1], 0.8);
        let near = format!("{A} Please advise.");
        let key = MemoKey::from_text(&near);
        let sim = key.similarity(&MemoKey::from_text(A));
        assert!(sim > 0.7 && sim < 1.0, "{sim}");
        let m2 = model(&[A], &[1], 0.5);
        assert!(m2.proba(&key, 2).is_some());
        // a stricter threshold than the similarity abstains
        let strict = model(&[A], &[1], 1.0);
        assert!(strict.proba(&key, 2).is_none());
        assert!(m.proba(&MemoKey::from_text("An unrelated sentence about cooking pasta with garlic and olive oil."), 2).is_none());
    }

    #[test]
    fn disagreeing_neighbours_abstain_and_the_newest_wins_ties() {
        let near = format!("{A} Thanks.");
        let m = model(&[A, &near], &[1, 0], 0.5);
        // both stored states are within the threshold of A and disagree
        assert!(m.proba(&MemoKey::from_text(A), 2).is_none());
        let same = model(&[A, A], &[1, 1], 0.95);
        let n = same.neighbours(&MemoKey::from_text(A), 0.95, None);
        assert_eq!(n[0].1, 1, "the newer of equal matches first");
    }
}
