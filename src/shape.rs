//! The `shape` and `blend` understudents' dense features: cheap structural statistics of a state, no encoder.
//!
//! A state's features are a fixed-length vector (`DIM`), a pure function of the state JSON (and the question's
//! optional `pair`, which only adds one length-ratio column):
//!
//! - per field slot (`NAME_SLOTS` of them, keyed by field NAME: a stable FNV-1a hash of the name picks the slot, so
//!   adding or removing a field never shifts another; on a collision the field with the smaller name keeps the slot;
//!   a non-object state is one field named ""): a presence bit, the JSON type (null/bool/number/string/list/object
//!   one-hot), the numeric value (signed log, clipped) and the boolean value, and the text features below of the
//!   field's text;
//! - overall: the text features of the whole state text, the number of fields, the state's own type, the log ratio
//!   of the char lengths of each pair of the first `RATIO_FIELDS` slots, and the log length ratio of the pair's two
//!   fields (0 without a pair or when either is absent).
//!
//! Text features (`TEXT_DIM`): log chars, words and lines; the share of digits, uppercase letters and punctuation;
//! log counts of markdown headers, bullets and code fences, URLs, question marks; log counts of refusal, apology and
//! hedging markers (a small fixed lexicon, substring counts on the lowercased text); the log count of numbers; and
//! the share of the text's numbers that also appear in another field (an answer echoing its question; for the
//! overall block, the share of numbers present in at least two fields).
//!
//! The model standardizes them (population mean/sd of the training rows), clips at `CLIP` standard deviations (a
//! column that was constant in training cannot blow up) and fits `linear.rs`'s logistic regression.

use crate::encoder::state_text;
use crate::keywords::{self, Sparse};
use crate::pyjson;
use crate::readout::{PoolModel, Standardizer};
use serde_json::Value;
use std::collections::HashSet;

/// Name-keyed field slots.
pub const NAME_SLOTS: usize = 16;
pub const TEXT_DIM: usize = 16;
pub const SLOT_DIM: usize = 1 + 6 + 2 + TEXT_DIM;
const RATIO_FIELDS: usize = 4;
const RATIOS: usize = RATIO_FIELDS * (RATIO_FIELDS - 1) / 2;
const OVERALL_DIM: usize = TEXT_DIM + 1 + 6 + RATIOS + 1;
pub const DIM: usize = NAME_SLOTS * SLOT_DIM + OVERALL_DIM;
/// Standardized features are clipped to ±`CLIP`.
pub const CLIP: f32 = 6.0;

const REFUSAL: &[&str] = &[
    "i can't", "i can’t", "i cannot", "i won't", "i won’t", "unable to", "not able to", "can't help", "cannot help", "can't assist",
    "cannot assist", "i must decline", "i refuse", "not appropriate", "against my", "as an ai",
];
const APOLOGY: &[&str] = &["sorry", "apolog", "regret", "unfortunately", "my mistake", "forgive"];
const HEDGE: &[&str] = &[
    "maybe", "perhaps", "might", "possibly", "probably", "it depends", "not sure", "i think", "i believe", "seems", "appears", "could be",
    "likely", "generally", "typically", "approximately", "roughly", "may ",
];

fn ln1p(x: usize) -> f32 {
    (x as f32).ln_1p()
}

fn count_any(lower: &str, pats: &[&str]) -> usize {
    pats.iter().map(|p| lower.matches(p).count()).sum()
}

/// The numbers in a text: maximal digit runs with an optional fractional part, in order of appearance.
fn numbers(text: &str) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    let mut out = vec![];
    let mut i = 0;
    while i < c.len() {
        if c[i].is_ascii_digit() {
            let mut j = i;
            while j < c.len() && c[j].is_ascii_digit() {
                j += 1;
            }
            if j + 1 < c.len() && c[j] == '.' && c[j + 1].is_ascii_digit() {
                j += 1;
                while j < c.len() && c[j].is_ascii_digit() {
                    j += 1;
                }
            }
            out.push(c[i..j].iter().collect());
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn is_bullet(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("- ") || t.starts_with("* ") || t.starts_with('•') {
        return true;
    }
    let d = t.chars().take_while(char::is_ascii_digit).count();
    d > 0 && t[d..].starts_with(". ")
}

/// The `TEXT_DIM` text features; `echo` is the share of numbers shared with another field.
pub fn text_features(text: &str, echo: f32, n_numbers: usize) -> [f32; TEXT_DIM] {
    let chars = text.chars().count();
    let denom = chars.max(1) as f32;
    let (mut digits, mut upper, mut punct, mut qm) = (0usize, 0usize, 0usize, 0usize);
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            digits += 1;
        } else if ch.is_uppercase() {
            upper += 1;
        } else if ch.is_ascii_punctuation() || matches!(ch, '“' | '”' | '‘' | '’' | '…' | '—' | '–') {
            punct += 1;
        }
        if ch == '?' {
            qm += 1;
        }
    }
    let words = text.split_whitespace().count();
    let lines = if text.is_empty() { 0 } else { text.lines().count() };
    let headers = text.lines().filter(|l| l.trim_start().starts_with('#')).count();
    let bullets = text.lines().filter(|l| is_bullet(l)).count();
    let fences = text.matches("```").count();
    let lower = text.to_lowercase();
    let urls = lower.matches("http://").count() + lower.matches("https://").count() + lower.matches("www.").count();
    [
        ln1p(chars),
        ln1p(words),
        ln1p(lines),
        digits as f32 / denom,
        upper as f32 / denom,
        punct as f32 / denom,
        ln1p(headers),
        ln1p(bullets),
        ln1p(fences),
        ln1p(urls),
        ln1p(qm),
        ln1p(count_any(&lower, REFUSAL)),
        ln1p(count_any(&lower, APOLOGY)),
        ln1p(count_any(&lower, HEDGE)),
        ln1p(n_numbers),
        echo,
    ]
}

fn type_index(v: &Value) -> usize {
    match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Number(_) => 2,
        Value::String(_) => 3,
        Value::Array(_) => 4,
        Value::Object(_) => 5,
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => pyjson::dumps(other),
    }
}

/// FNV-1a (64-bit) of a field name: the stable key of its slot.
fn name_hash(name: &str) -> u64 {
    name.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// The slot a field name lives in.
pub fn slot_of(name: &str) -> usize {
    (name_hash(name) % NAME_SLOTS as u64) as usize
}

/// The state's fields as (name, value); a non-object state is one field named "".
fn fields(state: &Value) -> Vec<(&str, &Value)> {
    match state {
        Value::Object(m) => m.iter().map(|(k, v)| (k.as_str(), v)).collect(),
        other => vec![("", other)],
    }
}

/// The field name each slot holds for this state ("" for an empty slot), with `features`' collision rule.
pub fn slot_names(state: &Value) -> Vec<String> {
    let fs = fields(state);
    let mut at: Vec<Option<&str>> = vec![None; NAME_SLOTS];
    for (name, _) in &fs {
        let s = slot_of(name);
        if at[s].is_none_or(|j| *name < j) {
            at[s] = Some(name);
        }
    }
    at.into_iter().map(|n| n.unwrap_or("").to_string()).collect()
}

/// The feature vector of a state (length `DIM`).
pub fn features(state: &Value, pair: Option<&(String, String)>) -> Vec<f32> {
    let fs = fields(state);
    let texts: Vec<String> = fs.iter().map(|(_, v)| value_text(v)).collect();
    let nums: Vec<Vec<String>> = texts.iter().map(|t| numbers(t)).collect();
    // slot -> field index; a collision goes to the smaller name, whatever the order the fields came in
    let mut at: [Option<usize>; NAME_SLOTS] = [None; NAME_SLOTS];
    for (i, (name, _)) in fs.iter().enumerate() {
        let s = slot_of(name);
        if at[s].is_none_or(|j| *name < fs[j].0) {
            at[s] = Some(i);
        }
    }
    let mut out: Vec<f32> = Vec::with_capacity(DIM);
    for slot in at {
        let Some(s) = slot else {
            out.extend([0.0; SLOT_DIM]);
            continue;
        };
        let v = fs[s].1;
        out.push(1.0);
        let mut ty = [0.0f32; 6];
        ty[type_index(v)] = 1.0;
        out.extend(ty);
        let (num, flag) = match v {
            Value::Number(n) => {
                let x = n.as_f64().unwrap_or(0.0).clamp(-1e9, 1e9);
                (x.signum() as f32 * (x.abs() as f32).ln_1p(), 0.0)
            }
            Value::Bool(b) => (0.0, f32::from(u8::from(*b))),
            _ => (0.0, 0.0),
        };
        out.extend([num, flag]);
        // the share of this field's distinct numbers that another field also holds
        let mine: HashSet<&String> = nums[s].iter().collect();
        let echo = if mine.is_empty() {
            0.0
        } else {
            let other: HashSet<&String> = nums.iter().enumerate().filter(|(i, _)| *i != s).flat_map(|(_, n)| n.iter()).collect();
            mine.iter().filter(|n| other.contains(**n)).count() as f32 / mine.len() as f32
        };
        out.extend(text_features(&texts[s], echo, nums[s].len()));
    }
    // overall
    let whole = state_text(state);
    let all_nums = numbers(&whole);
    let mut holders: std::collections::HashMap<&String, usize> = Default::default();
    for n in &nums {
        for x in n.iter().collect::<HashSet<_>>() {
            *holders.entry(x).or_default() += 1;
        }
    }
    let echo = if holders.is_empty() { 0.0 } else { holders.values().filter(|&&c| c >= 2).count() as f32 / holders.len() as f32 };
    out.extend(text_features(&whole, echo, all_nums.len()));
    out.push(ln1p(fs.len()));
    let mut ty = [0.0f32; 6];
    ty[type_index(state)] = 1.0;
    out.extend(ty);
    let slot_len = |i: usize| at[i].map_or(0, |s| texts[s].chars().count());
    for i in 0..RATIO_FIELDS {
        for j in i + 1..RATIO_FIELDS {
            out.push(((1 + slot_len(i)) as f32 / (1 + slot_len(j)) as f32).ln());
        }
    }
    let by_name = |n: &str| fs.iter().position(|(k, _)| *k == n).map(|i| texts[i].chars().count());
    out.push(match pair.and_then(|(a, b)| Some((by_name(a)?, by_name(b)?))) {
        Some((la, lb)) => ((1 + la) as f32 / (1 + lb) as f32).ln(),
        None => 0.0,
    });
    debug_assert_eq!(out.len(), DIM);
    out
}

/// `x` standardized and clipped to ±`CLIP`.
pub fn standardize(std: &Standardizer, x: &[f32]) -> Vec<f32> {
    let mut z = vec![0f32; x.len()];
    std.apply(x, &mut z);
    z.iter_mut().for_each(|v| *v = v.clamp(-CLIP, CLIP));
    z
}

/// The `shape` model's probabilities for a raw feature vector.
pub fn proba(m: &PoolModel, x: &[f32], k: usize) -> Vec<f64> {
    m.lr.proba(&standardize(&m.std, x), k)
}

/// `v` scaled to unit L2 norm (a zero vector stays zero).
pub fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>().sqrt() as f32 + 1e-12;
    v.iter_mut().for_each(|a| *a /= n);
    v
}

/// The novelty vector: the standardized (clipped) features, L2-normalized.
pub fn novelty(std: &Standardizer, x: &[f32]) -> Vec<f32> {
    unit(standardize(std, x))
}

// ---------------------------------------------------------------- blend

/// The `blend` model's standardizers for its two dense blocks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlendStd {
    pub shape: Standardizer,
    pub emb: Standardizer,
}

/// One row of the stacked design: the n-gram buckets as they are (unit L2 norm), then the standardized, clipped feature
/// block and embedding block, each then normalized to unit L2 norm per row (so every block has norm exactly 1, or 0
/// for an all-zero block, and the one `C` suits all three), at columns `keywords::BUCKETS ..`.
pub fn blend_row(sp: &Sparse, shape: &[f32], emb: &[f32], std: &BlendStd) -> Sparse {
    let mut out = sp.clone();
    let mut col = keywords::BUCKETS;
    for (block, s) in [(shape, &std.shape), (emb, &std.emb)] {
        for (j, v) in unit(standardize(s, block)).into_iter().enumerate() {
            if v != 0.0 {
                out.idx.push((col + j) as u32);
                out.val.push(v);
            }
        }
        col += block.len();
    }
    out
}

/// The stacked design's column count.
pub fn blend_dim(shape_d: usize, emb_d: usize) -> usize {
    keywords::BUCKETS + shape_d + emb_d
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn features_are_deterministic_and_fixed_length() {
        let a = json!({"question": "What is 12 + 30?", "answer": "I think it is 42. Sorry if that is wrong, see https://x.io\n- one\n- two", "ok": true, "n": 7});
        let f1 = features(&a, None);
        assert_eq!(f1.len(), DIM);
        assert_eq!(f1, features(&a, None), "same state, same vector");
        assert!(f1.iter().all(|v| v.is_finite()));
        // another shape has the same length; a string state too
        assert_eq!(features(&json!("hello"), None).len(), DIM);
        assert_eq!(features(&json!({"a": [1, 2, 3]}), None).len(), DIM);
        // field order does not depend on the JSON key order
        let b = json!({"n": 7, "ok": true, "answer": a["answer"].clone(), "question": a["question"].clone()});
        assert_eq!(f1, features(&b, None));
        // a different state differs
        assert_ne!(f1, features(&json!({"question": "Hi", "answer": "Hello", "ok": false, "n": 1}), None));
    }

    #[test]
    fn features_see_structure_markers_and_echoed_numbers() {
        let idx = |name: &str, off: usize| slot_of(name) * SLOT_DIM + 1 + 6 + 2 + off;
        assert_ne!(slot_of("answer"), slot_of("question"), "the test names must not collide");
        let echo = features(&json!({"question": "What is 42?", "answer": "It is 42."}), None);
        let none = features(&json!({"question": "What is 42?", "answer": "It is 7."}), None);
        assert_eq!(echo[idx("answer", 15)], 1.0);
        assert_eq!(none[idx("answer", 15)], 0.0);
        let md = features(&json!({"answer": "# T\n- a\n- b\n```x```\nhttps://a.b ?"}), None);
        assert!(md[idx("answer", 6)] > 0.0 && md[idx("answer", 7)] > 0.0 && md[idx("answer", 8)] > 0.0 && md[idx("answer", 9)] > 0.0 && md[idx("answer", 10)] > 0.0);
        let refuse = features(&json!({"answer": "I'm sorry, I cannot help; perhaps ask elsewhere."}), None);
        assert!(refuse[idx("answer", 11)] > 0.0 && refuse[idx("answer", 12)] > 0.0 && refuse[idx("answer", 13)] > 0.0);
        // JSON types
        let t = features(&json!({"a": true, "b": 5.5, "c": "s", "d": [1], "e": null}), None);
        for (name, ty) in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 0)] {
            assert_eq!(t[slot_of(name) * SLOT_DIM + 1 + ty], 1.0, "field {name}");
        }
    }

    /// The columns of the field slots only (not the overall block).
    fn slots(f: &[f32]) -> Vec<&[f32]> {
        f[..NAME_SLOTS * SLOT_DIM].chunks(SLOT_DIM).collect()
    }

    #[test]
    fn slots_are_name_keyed_so_adding_or_removing_a_field_shifts_nothing() {
        let names = ["question", "answer", "flag"];
        let slots_used: HashSet<usize> = names.iter().map(|n| slot_of(n)).collect();
        assert_eq!(slots_used.len(), names.len(), "the test names must not collide");
        let base = json!({"question": "What is it?", "answer": "It is fine.", "flag": true});
        let f = features(&base, None);
        let s = slots(&f);
        // pick extra names that do not share a slot with any base field
        let used: Vec<usize> = ["question", "answer", "flag"].iter().map(|n| slot_of(n)).collect();
        let extras: Vec<String> = (0..200).map(|i| format!("extra{i}")).filter(|n| !used.contains(&slot_of(n))).take(2).collect();
        assert_eq!(extras.len(), 2);
        // an earlier-sorting field (names sorted before "answer") must not move the others
        let mut with = base.as_object().unwrap().clone();
        with.insert("0aaa".into(), json!("early"));
        let early = if used.contains(&slot_of("0aaa")) { None } else { Some(slots_of_state(&json!(with))) };
        if let Some(e) = early {
            for n in ["question", "answer", "flag"] {
                assert_eq!(e[slot_of(n)], s[slot_of(n)], "{n}");
            }
        }
        for extra in &extras {
            let mut m = base.as_object().unwrap().clone();
            m.insert(extra.clone(), json!("more"));
            let g = features(&json!(m), None);
            for n in ["question", "answer", "flag"] {
                assert_eq!(slots(&g)[slot_of(n)], s[slot_of(n)], "{n} moved when {extra} was added");
            }
            // the new field fills its own slot (present, a string) and removing it restores the vector
            assert_eq!(slots(&g)[slot_of(extra)][0], 1.0);
            assert_ne!(g, f);
        }
        // removing a field only empties its own slot: presence 0, and the rest unchanged
        let fewer = features(&json!({"question": "What is it?", "flag": true}), None);
        let fs = slots(&fewer);
        assert!(fs[slot_of("answer")].iter().all(|&v| v == 0.0));
        for n in ["question", "flag"] {
            assert_eq!(fs[slot_of(n)], s[slot_of(n)], "{n}");
        }
        // and which field vanished is visible: a different name's presence bit drops
        let other = features(&json!({"answer": "It is fine.", "flag": true}), None);
        assert_eq!(slots(&other)[slot_of("question")][0], 0.0);
        assert_eq!(slots(&other)[slot_of("answer")][0], 1.0);
    }

    fn slots_of_state(s: &Value) -> Vec<Vec<f32>> {
        slots(&features(s, None)).into_iter().map(|c| c.to_vec()).collect()
    }

    #[test]
    fn name_collisions_resolve_to_the_smaller_name_whatever_the_order() {
        let a = "k0";
        let b = (1..10_000).map(|i| format!("k{i}")).find(|n| slot_of(n) == slot_of(a) && n.as_str() > a).expect("a colliding name");
        let x = features(&json!({a: "alpha", b.as_str(): "a much longer second value"}), None);
        let y = features(&json!({b.as_str(): "a much longer second value", a: "alpha"}), None);
        assert_eq!(x, y);
        let only_a = features(&json!({a: "alpha"}), None);
        assert_eq!(slots(&x)[slot_of(a)], slots(&only_a)[slot_of(a)], "the smaller name keeps the slot");
    }

    #[test]
    fn the_pair_adds_one_length_ratio_and_nothing_moves() {
        let s = json!({"zzz": "last", "a": "first-by-name", "q": "pair-a", "r": "pair-b-longer"});
        let pair = ("q".to_string(), "r".to_string());
        let f = features(&s, Some(&pair));
        let g = features(&s, None);
        assert_ne!(f, g);
        assert_eq!(f[..DIM - 1], g[..DIM - 1], "only the pair column differs");
        assert!((f[DIM - 1] - (7f32 / 14f32).ln()).abs() < 1e-6);
        // an absent pair field gives 0
        assert_eq!(features(&json!({"q": "x"}), Some(&pair))[DIM - 1], 0.0);
    }

    #[test]
    fn clipping_bounds_a_column_that_was_constant() {
        let std = Standardizer { mu: vec![0.0, 1.0], sd: vec![1e-6, 2.0] };
        let z = standardize(&std, &[5.0, 3.0]);
        assert_eq!(z, vec![CLIP, 1.0]);
    }

    #[test]
    fn blend_rows_extend_the_sparse_vector() {
        let sp = keywords::featurize(&["hello world".to_string()]);
        let std = BlendStd {
            shape: Standardizer { mu: vec![0.0; 3], sd: vec![1.0; 3] },
            emb: Standardizer { mu: vec![0.0; 4], sd: vec![1.0; 4] },
        };
        let r = blend_row(&sp, &[1.0, 0.0, -1.0], &[1.0, 1.0, 1.0, 1.0], &std);
        assert_eq!(r.idx.len(), sp.idx.len() + 2 + 4);
        assert!(r.idx.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*r.idx.last().unwrap() as usize, blend_dim(3, 4) - 1);
    }

    #[test]
    fn blend_dense_blocks_have_unit_norm_even_with_outliers() {
        let sp = keywords::featurize(&["hello world".to_string()]);
        let std = BlendStd {
            shape: Standardizer { mu: vec![0.0; 3], sd: vec![1e-6, 1.0, 1.0] },
            emb: Standardizer { mu: vec![0.0; 4], sd: vec![1.0; 4] },
        };
        // an outlier row: every feature clips at +6, so a 1/sqrt(dim) scaling would give a block norm of 6
        let r = blend_row(&sp, &[1e9, 1e9, 1e9], &[1e9, -1e9, 5.0, 0.1], &std);
        let block = |lo: usize, hi: usize| -> f64 {
            r.idx.iter().zip(&r.val).filter(|(i, _)| (lo..hi).contains(&(**i as usize))).map(|(_, v)| (*v as f64).powi(2)).sum::<f64>().sqrt()
        };
        assert!((block(keywords::BUCKETS, keywords::BUCKETS + 3) - 1.0).abs() < 1e-5);
        assert!((block(keywords::BUCKETS + 3, keywords::BUCKETS + 7) - 1.0).abs() < 1e-5);
        // an all-zero block stays zero
        let z = blend_row(&sp, &[0.0; 3], &[0.0; 4], &std);
        assert_eq!(z.idx.len(), sp.idx.len());
    }
}
