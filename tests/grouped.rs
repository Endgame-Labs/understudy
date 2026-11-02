//! Grouped cross-validation (ids `<item>#<k>` share a group) and the on-disk encode cache.

use understudy::encoder::{Encoder, Fake, Featurizer};
use understudy::proxy::{Kind, Proxy, Settings};
use understudy::question::Question;
use understudy::readout::ReadoutChoice;
use understudy::rng::Xoshiro;
use understudy::understudent::UnderstudentKind;
use understudy::lead::{Replay, Lead};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

const H: usize = 16;

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Is this response the best of the candidate responses?"}), None).unwrap()
}

const WORDS: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"];

fn sentence(r: &mut Xoshiro, n: usize) -> String {
    (0..n).map(|_| WORDS[(r.next_u64() % WORDS.len() as u64) as usize]).collect::<Vec<_>>().join(" ")
}

/// Per-response items: `items` instructions, each with two responses of no signal, exactly one "yes".
/// Returns (id, state, p_yes) lines, siblings contiguous.
fn per_response(items: usize, seed: u64) -> Vec<(String, Value, f64)> {
    let mut r = Xoshiro::new(seed);
    let mut out = vec![];
    for i in 0..items {
        let instruction = format!("task{i} {}", sentence(&mut r, 14));
        let best = (r.next_u64() % 2) as usize;
        for k in 0..2 {
            // the instruction dominates the state; the response carries no signal
            let state = json!({"instruction": instruction, "response": sentence(&mut r, 2)});
            out.push((format!("it{i:04}#{}", k + 1), state, if k == best { 0.95 } else { 0.05 }));
        }
    }
    out
}

const HC: usize = 128;

fn cv_agreement(group_sep: Option<char>) -> f64 {
    let mut s = Settings::default();
    s.fit.readout = ReadoutChoice::Pool;
    s.fit.understudent = UnderstudentKind::Probe;
    s.group_sep = group_sep;
    let fz = Featurizer::for_understudent(UnderstudentKind::Probe, None, None, None, HC).unwrap();
    let mut p = Proxy::new(s, fz, Arc::new(Fake { hidden: HC }), None);
    let si = p.understudent("q", &noul());
    for (id, state, py) in per_response(100, 7) {
        let (key, enc) = p.encode_many(&[&state]).unwrap().pop().unwrap();
        p.add_example_with_id(si, Kind::Sampled, &state, usize::from(py >= 0.5), Some(vec![1.0 - py, py]), key, enc, Some(&id));
    }
    p.refit(si);
    p.understudents[si].serving().fit.report["cv_agreement"].as_f64().unwrap()
}

#[test]
fn sibling_groups_keep_cv_at_or_above_chance_where_plain_folds_fall_below() {
    // 200 rows, 50% yes, no signal in the texts; a sibling in the training folds carries the opposite label
    let grouped = cv_agreement(Some('#'));
    let plain = cv_agreement(None);
    eprintln!("grouped {grouped} plain {plain}");
    assert!(grouped >= 0.45, "grouped CV agreement {grouped}");
    assert!(plain < 0.40, "plain-fold CV agreement {plain} (siblings leak with the opposite label)");
    assert!(grouped > plain + 0.1, "grouped {grouped} vs plain {plain}");
}

fn replay_lead(lines: &[Value]) -> Replay {
    let probs: HashMap<String, Vec<f64>> = lines
        .iter()
        .map(|l| (l["id"].as_str().unwrap().to_string(), l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()))
        .collect();
    Replay { probs, source: "test".into() }
}

/// A separable per-response stream: the response containing "error" is the yes.
fn stream(items: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    let mut out = vec![];
    for i in 0..items {
        let instruction = format!("task{i} {}", sentence(&mut r, 5));
        let best = (r.next_u64() % 3) as usize;
        for k in 0..3 {
            let marker = if k == best { "error" } else { "ok" };
            let py = if k == best { 0.95 } else { 0.05 };
            out.push(json!({"id": format!("i{i:04}#{}", k + 1), "state": {"instruction": instruction, "response": format!("{} {marker} {marker}", sentence(&mut r, 3))},
                            "lead_probs": [1.0 - py, py]}));
        }
    }
    out
}

fn run(kind: UnderstudentKind, lines: &[Value], cache: Option<&std::path::Path>) -> Vec<Value> {
    let mut s = Settings::default();
    s.fit.understudent = kind;
    s.audit_rate = 0.2;
    if kind == UnderstudentKind::Auto {
        s.fit.tiers = UnderstudentKind::default_tiers();
        s.cost_us = vec![1.0, 10.0, 100.0, 1000.0]; // fixed costs: measured ones differ between runs
    }
    let fz = Featurizer::for_understudent(kind, None, None, Some(("instruction".into(), "response".into())), H).unwrap();
    let mut p = Proxy::new(s, fz, Arc::new(Fake { hidden: H }), Some(Box::new(replay_lead(lines)) as Box<dyn Lead>));
    if let Some(d) = cache {
        p.set_encode_cache(d).unwrap();
    }
    p.understudent("q", &noul());
    let input: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    let mut out = vec![];
    understudy::replay::run(&mut p, input.as_bytes(), &mut out).unwrap();
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| {
            let mut v: Value = serde_json::from_str(l).unwrap();
            v.as_object_mut().unwrap().remove("understudent_us"); // a timing
            v
        })
        .collect()
}

#[test]
fn decisions_are_identical_with_and_without_the_encode_cache() {
    let lines = stream(120, 3);
    let dir = std::env::temp_dir().join(format!("understudy-encode-cache-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for kind in [UnderstudentKind::Probe, UnderstudentKind::Deep, UnderstudentKind::Lite, UnderstudentKind::Auto] {
        let plain = run(kind, &lines, None);
        let cold = run(kind, &lines, Some(&dir));
        let warm = run(kind, &lines, Some(&dir)); // every text served from disk
        assert_eq!(plain, cold, "{kind:?}: cold cache");
        assert_eq!(plain, warm, "{kind:?}: warm cache");
        assert!(plain.iter().any(|d| d["answered_by"] == "understudent"), "{kind:?} never answered locally; the test would prove little");
    }
    assert!(std::fs::read_dir(&dir).unwrap().count() > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_different_encoder_does_not_share_cache_entries() {
    let dir = std::env::temp_dir().join(format!("understudy-encode-cache-test2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let fz = Featurizer::for_understudent(UnderstudentKind::Probe, None, None, None, H).unwrap();
    let a = understudy::diskcache::DiskCache::open(&dir, &Fake { hidden: 16 }.fingerprint()).unwrap();
    let b = understudy::diskcache::DiskCache::open(&dir, &Fake { hidden: 32 }.fingerprint()).unwrap();
    assert_ne!(a.key(&fz, "some text"), b.key(&fz, "some text"));
    let fz2 = Featurizer::for_understudent(UnderstudentKind::Deep, None, None, None, H).unwrap();
    assert_ne!(a.key(&fz, "some text"), a.key(&fz2, "some text"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The sampled-example cap (`train --max-examples`) decides how many lead answers a fit sees: the default keeps the
/// newest 2000 (the oldest are evicted), a raised cap keeps every one.
#[test]
fn sampled_cap_keeps_the_first_items_when_raised() {
    let kept = |cap: Option<usize>| {
        let mut s = Settings::default();
        s.fit.understudent = UnderstudentKind::Probe;
        if let Some(c) = cap {
            s.sampled_cap = c;
        }
        let fz = Featurizer::for_understudent(UnderstudentKind::Probe, None, None, None, HC).unwrap();
        let mut p = Proxy::new(s, fz, Arc::new(Fake { hidden: HC }), None);
        let si = p.understudent("q", &noul());
        for i in 0..2500 {
            let state = json!({"instruction": format!("item {i}"), "response": "r"});
            let (key, enc) = p.encode_many(&[&state]).unwrap().pop().unwrap();
            p.add_example_with_id(si, Kind::Sampled, &state, i % 2, Some(vec![0.5, 0.5]), key, enc, Some(&format!("it{i:05}#1")));
        }
        let ex = &p.understudents[si].examples;
        (ex.len(), ex.iter().any(|e| e.seq == 1))
    };
    assert_eq!(kept(None), (2000, false), "default: the oldest 500 are evicted");
    assert_eq!(kept(Some(2500)), (2500, true), "raised cap: the first item survives");
}
