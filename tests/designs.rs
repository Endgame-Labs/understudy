//! The `shape`, `blend` and `memo` understudents on a fake encoder: feature determinism through the proxy, the stacked
//! model's shape, memo's exact-match and near-duplicate behaviour, snapshot round trips, and their place in `auto`.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use understudy::encoder::{Encoder, Fake, Featurizer};
use understudy::shape;
use understudy::lead::{Lead, Replay};
use understudy::snapshot;
use understudy::proxy::{Kind, Proxy, Settings};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::understudent::{Mode, UnderstudentKind};

const H: usize = 16;

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the assistant refuse?"}), None).unwrap()
}

fn proxy(kind: UnderstudentKind, tiers: &[UnderstudentKind], lead: Option<Replay>) -> (Proxy, usize) {
    let mut s = Settings::default();
    s.fit.understudent = kind;
    s.fit.tiers = tiers.to_vec();
    s.audit_rate = 0.1;
    if kind == UnderstudentKind::Auto {
        s.cost_us = vec![10.0; tiers.len()]; // measured costs are timing noise; fixed ones make a race repeatable
    }
    let enc: Arc<dyn Encoder> = Arc::new(Fake { hidden: H });
    let fz = Featurizer::for_understudent(kind, None, None, None, H).unwrap();
    let mut p = Proxy::new(s, fz, enc, lead.map(|t| Box::new(t) as Box<dyn Lead>));
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];

fn words(r: &mut Xoshiro, n: usize) -> String {
    (0..n).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect::<Vec<_>>().join(" ")
}

/// A question/answer state: a refusal is short and apologetic, a normal answer long and plain. Structure alone tells them apart.
fn qa(r: &mut Xoshiro, refusal: bool) -> Value {
    let long = 40 + (r.next_u64() % 20) as usize;
    let answer = if refusal { format!("I'm sorry, I cannot help with that. {}", words(r, 3)) } else { words(r, long) };
    json!({"question": words(r, 6), "answer": answer})
}

fn qa_stream(n: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.4;
            let py = if bad { 0.95 } else { 0.05 };
            json!({"id": format!("i{i:04}"), "state": qa(&mut r, bad), "lead_probs": [1.0 - py, py]})
        })
        .collect()
}

fn replay_lead(lines: &[Value]) -> Replay {
    let probs: HashMap<String, Vec<f64>> = lines
        .iter()
        .map(|l| (l["id"].as_str().unwrap().to_string(), l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()))
        .collect();
    Replay { probs, source: "test".into() }
}

fn run(p: &mut Proxy, lines: &[Value]) -> (Value, Vec<Value>) {
    let input: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    let mut out = vec![];
    let sum = understudy::replay::run(p, input.as_bytes(), &mut out).unwrap();
    let rows = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    (sum, rows)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("understudy-designs-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn local(rows: &[Value]) -> usize {
    rows.iter().filter(|d| d["answered_by"] == "understudent").count()
}

fn agree(rows: &[Value], lines: &[Value]) -> f64 {
    let (mut n, mut ok) = (0, 0);
    for (d, l) in rows.iter().zip(lines) {
        if d["answered_by"] == "understudent" {
            n += 1;
            let lead = usize::from(l["lead_probs"][1].as_f64().unwrap() > 0.5);
            ok += usize::from(d["label"].as_u64().unwrap() as usize == lead);
        }
    }
    ok as f64 / n.max(1) as f64
}

/// Add every line as a sampled example and refit once.
fn learn(p: &mut Proxy, si: usize, lines: &[Value]) {
    for l in lines {
        let state = &l["state"];
        let probs: Vec<f64> = l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let (key, enc) = p.encode_many(&[state]).unwrap().pop().unwrap();
        p.add_example(si, Kind::Sampled, state, usize::from(probs[1] > 0.5), Some(probs), key, enc);
    }
    p.refit(si);
}

#[test]
fn shape_is_deterministic_through_the_proxy_and_learns_structure() {
    let lines = qa_stream(500, 1);
    let (mut a, _) = proxy(UnderstudentKind::Shape, &[], Some(replay_lead(&lines)));
    let (mut b, si) = proxy(UnderstudentKind::Shape, &[], Some(replay_lead(&lines)));
    let (sum, rows) = run(&mut a, &lines);
    let (_, rows2) = run(&mut b, &lines);
    let strip = |rows: &[Value]| -> Vec<Value> {
        rows.iter().cloned().map(|mut d| { d.as_object_mut().unwrap().remove("understudent_us"); d }).collect()
    };
    assert_eq!(strip(&rows), strip(&rows2), "same stream, same decisions");
    assert_eq!(sum["understudent"], "shape");
    let f = &b.understudents[si].serving().fit;
    assert_eq!(f.mode, Some(Mode::Symmetric), "{}", f.report);
    assert!(local(&rows) > 100, "local {}", local(&rows));
    assert!(agree(&rows, &lines) > 0.95);
    assert!(rows.iter().any(|d| d["readout"] == "shape"));
    // the encoding carries the vector, and the vector is the state's pure function
    let (_, enc) = b.encode_many(&[&lines[0]["state"]]).unwrap().pop().unwrap();
    assert_eq!(**enc.shape.as_ref().unwrap(), shape::features(&lines[0]["state"], None));
    // a snapshot round trip keeps the fit exactly
    let dir = tmp("shape");
    snapshot::save(&b, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[si].serving().fit, b.understudents[si].serving().fit);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn blend_has_one_model_over_the_three_blocks() {
    let lines = qa_stream(400, 2);
    let (mut p, si) = proxy(UnderstudentKind::Blend, &[], None);
    learn(&mut p, si, &lines);
    let (_, enc) = p.encode_many(&[&lines[0]["state"]]).unwrap().pop().unwrap();
    let f = &p.understudents[si].serving().fit;
    let emb = p.fz.pool_dim();
    let sparse = f.sparse.as_ref().expect("a sparse model");
    let std = f.blend.as_ref().expect("the block standardizers");
    assert_eq!((std.shape.mu.len(), std.emb.mu.len()), (shape::DIM, emb));
    assert_eq!(sparse.coef.len() / sparse.intercept.len(), shape::blend_dim(shape::DIM, emb));
    assert_eq!(f.report["blend_dim"], shape::blend_dim(shape::DIM, emb));
    assert_eq!(f.mode, Some(Mode::Symmetric), "{}", f.report);
    // an encoding holds all three inputs
    assert!(enc.sparse.is_some() && enc.shape.is_some() && !p.fz.parts(&enc).is_empty());
    let pr = f.proba(&p.fz, &enc, 2).unwrap();
    assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // the model reads the dense columns: it differs from a bare keywords of the same text only through them
    let nonzero = sparse.coef[sparse.coef.len() - (shape::DIM + emb)..].iter().filter(|&&c| c != 0.0).count();
    assert!(nonzero > 0, "dense columns get weights");
    let dir = tmp("blend");
    snapshot::save(&p, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[si].serving().fit, p.understudents[si].serving().fit);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Templates answered by the lead; each reappears with a small change (a counter), so it is a near-duplicate.
fn memo_stream(n: usize, templates: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    let bases: Vec<(String, bool)> = (0..templates).map(|t| (format!("template {t}: {}", words(&mut r, 40)), t % 2 == 0)).collect();
    (0..n)
        .map(|i| {
            let (text, yes) = &bases[(r.next_u64() % templates as u64) as usize];
            let py = if *yes { 0.95 } else { 0.05 };
            json!({"id": format!("m{i:04}"), "state": {"text": format!("{text} (seen {i})")}, "lead_probs": [1.0 - py, py]})
        })
        .collect()
}

#[test]
fn memo_answers_near_duplicates_of_what_the_lead_answered_and_nothing_else() {
    let mut lines = memo_stream(600, 12, 3);
    // unrelated states the lead never saw a neighbour of
    let mut r = Xoshiro::new(9);
    let novel: Vec<Value> = (0..40)
        .map(|i| json!({"id": format!("n{i:03}"), "state": {"text": format!("unrelated {i}: {}", words(&mut r, 45))}, "lead_probs": [0.1, 0.9]}))
        .collect();
    lines.extend(novel);
    let (mut p, si) = proxy(UnderstudentKind::Memo, &[], Some(replay_lead(&lines)));
    let (sum, rows) = run(&mut p, &lines);
    assert_eq!(sum["understudent"], "memo");
    let f = &p.understudents[si].serving().fit;
    assert_eq!(f.mode, Some(Mode::Symmetric), "{}", f.report);
    assert!(f.report["memo_sim"].as_f64().unwrap() >= 0.8);
    assert!(local(&rows[..600]) > 300, "local {}", local(&rows[..600]));
    assert!(agree(&rows, &lines) > 0.99);
    assert!(rows.iter().filter(|d| d["answered_by"] == "understudent").all(|d| d["readout"] == "memo"));
    // the states nothing resembles are forwarded
    assert!(rows[600..].iter().all(|d| d["answered_by"] == "lead"), "novel states must forward");
    let dir = tmp("memo");
    snapshot::save(&p, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[si].serving().fit, p.understudents[si].serving().fit);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memo_returns_the_lead_answer_for_an_exact_match_and_abstains_otherwise() {
    let lines = memo_stream(300, 12, 4);
    let (mut p, si) = proxy(UnderstudentKind::Memo, &[], None);
    learn(&mut p, si, &lines);
    let known = &lines[10]; // a state the lead answered
    let (_, e) = p.encode_many(&[&known["state"]]).unwrap().pop().unwrap();
    let (_, e2) = p.encode_many(&[&json!({"text": "nothing like the others, at all, whatsoever, in any way, today"})]).unwrap().pop().unwrap();
    let st = &p.understudents[si];
    let pr = st.serving().predict(&p.fz, &e, 2);
    let want: Vec<f64> = known["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let got = pr.p.expect("an exact match answers");
    assert!((got[1] - want[1]).abs() < 1e-6, "{got:?} vs {want:?}");
    assert!(!pr.novel);
    assert_eq!(pr.eligible, st.serving().mode().is_some(), "graduated memo answers its exact matches");
    // a state with no neighbour abstains
    let pr = st.serving().predict(&p.fz, &e2, 2);
    assert!(pr.p.is_none() && !pr.eligible && !pr.local);
}

#[test]
fn the_new_designs_race_in_auto_only_when_asked() {
    assert_eq!(UnderstudentKind::default_tiers(), vec![UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep]);
    let lines = qa_stream(500, 5);
    let tiers = [UnderstudentKind::Keywords, UnderstudentKind::Shape, UnderstudentKind::Blend, UnderstudentKind::Memo];
    let (mut p, si) = proxy(UnderstudentKind::Auto, &tiers, Some(replay_lead(&lines)));
    let (sum, rows) = run(&mut p, &lines);
    assert_eq!(sum["candidates"].as_array().map(Vec::len), Some(4), "{sum}");
    let race = p.understudents[si].race.as_array().unwrap().iter().map(|r| r["understudent"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    assert_eq!(race, ["keywords", "shape", "blend", "memo"]);
    assert!(p.understudents[si].serving_idx.is_some(), "something graduated: {}", p.understudents[si].race);
    assert!(local(&rows) > 100);
    assert!(agree(&rows, &lines) > 0.93);
    // determinism (timings are noise, so a fixed cost makes the race repeatable)
    let (mut q, _) = proxy(UnderstudentKind::Auto, &tiers, Some(replay_lead(&lines)));
    let (_, rows2) = run(&mut q, &lines);
    let labels = |r: &[Value]| r.iter().map(|d| (d["answered_by"].clone(), d["label"].clone())).collect::<Vec<_>>();
    assert_eq!(labels(&rows), labels(&rows2));
}

/// Add every line as a sampled example under the id `<group>#<i>` and refit once.
fn learn_grouped(p: &mut Proxy, si: usize, lines: &[Value], group_of: impl Fn(usize) -> String) {
    for (i, l) in lines.iter().enumerate() {
        let state = &l["state"];
        let probs: Vec<f64> = l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let (key, enc) = p.encode_many(&[state]).unwrap().pop().unwrap();
        let id = format!("{}#{i}", group_of(i));
        p.add_example_with_id(si, Kind::Sampled, state, usize::from(probs[1] > 0.5), Some(probs), key, enc, Some(&id));
    }
    p.refit(si);
}

#[test]
fn few_groups_never_fall_back_to_ungrouped_folds() {
    // Three parent items, each with 100 near-duplicate siblings and its own label. Ungrouped folds would let memo
    // answer every held-out sibling from its parent's other siblings and graduate; grouped folds hold a whole parent out.
    let mut r = Xoshiro::new(5);
    let parents: Vec<String> = (0..3).map(|g| format!("parent {g}: {}", words(&mut r, 60))).collect();
    let lines: Vec<Value> = (0..300)
        .map(|i| {
            let g = i % 3;
            let py = if g == 0 { 0.95 } else { 0.05 };
            json!({"id": format!("p{g}#{i}"), "state": {"text": format!("{} (sibling {i})", parents[g])}, "lead_probs": [1.0 - py, py]})
        })
        .collect();
    let (mut p, si) = proxy(UnderstudentKind::Memo, &[], None);
    learn_grouped(&mut p, si, &lines, |i| format!("p{}", i % 3));
    let f = &p.understudents[si].serving().fit;
    assert_eq!(f.mode, None, "siblings must not cross folds: {}", f.report);
    assert_eq!(f.report["answered"], 0, "{}", f.report);
    assert_eq!(f.report["memo_sim_nested"].as_array().unwrap().len(), 3, "k = number of groups");
    // the same rows with no group prefix are free to cross folds, as before: memo graduates
    let (mut q, sj) = proxy(UnderstudentKind::Memo, &[], None);
    learn(&mut q, sj, &lines);
    assert!(q.understudents[sj].serving().fit.mode.is_some(), "{}", q.understudents[sj].serving().fit.report);
}

#[test]
fn a_single_group_declines_to_graduate_with_a_reason() {
    let lines = qa_stream(200, 8);
    for kind in [UnderstudentKind::Shape, UnderstudentKind::Memo, UnderstudentKind::Keywords] {
        let (mut p, si) = proxy(kind, &[], None);
        learn_grouped(&mut p, si, &lines, |_| "only".into());
        let f = &p.understudents[si].serving().fit;
        assert_eq!(f.mode, None, "{kind:?}");
        let why = f.report["why_not"].as_str().unwrap_or_default();
        assert!(why.contains("fewer than 2 id groups"), "{kind:?}: {}", f.report);
    }
}

#[test]
fn shape_slots_do_not_shift_when_a_state_gains_a_field() {
    // through the proxy's own feature path: the vector of the shared fields' slots is unchanged by an extra field
    let a = json!({"question": "What is it?", "answer": "It is fine."});
    let b = json!({"question": "What is it?", "answer": "It is fine.", "note": "extra"});
    let (fa, fb) = (shape::features(&a, None), shape::features(&b, None));
    let slot = |f: &[f32], name: &str| f[shape::slot_of(name) * 25..(shape::slot_of(name) + 1) * 25].to_vec();
    assert_ne!(shape::slot_of("note"), shape::slot_of("answer"));
    assert_ne!(shape::slot_of("note"), shape::slot_of("question"));
    for n in ["question", "answer"] {
        assert_eq!(slot(&fa, n), slot(&fb, n), "{n}");
    }
}
