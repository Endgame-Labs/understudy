//! Behaviour on a fake encoder (one pseudo-random vector per word): graduation, one-sided mode, novelty,
//! demotion, pair antisymmetry, replay determinism, snapshot round trips and train → judge.

use understudy::encoder::{Encoder, Fake, Featurizer};
use understudy::snapshot;
use understudy::proxy::{Kind, Proxy, Settings};
use understudy::question::Question;
use understudy::readout::ReadoutChoice;
use understudy::rng::Xoshiro;
use understudy::understudent::Mode;
use understudy::lead::Replay;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

const H: usize = 16;

fn fake() -> Arc<dyn Encoder> {
    Arc::new(Fake { hidden: H })
}

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap()
}

fn settings(readout: ReadoutChoice) -> Settings {
    let mut s = Settings::default();
    s.fit.readout = readout;
    s
}

fn proxy(s: Settings, pair: Option<(&str, &str)>, lead: Option<Replay>) -> (Proxy, usize) {
    let fz = Featurizer::new(vec![1, 2, 3], None, pair.map(|(a, b)| (a.to_string(), b.to_string())), H).unwrap();
    let mut p = Proxy::new(s, fz, fake(), lead.map(|t| Box::new(t) as Box<dyn understudy::lead::Lead>));
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"];

/// A tool-output-like text: a few filler words plus `marker`.
fn text(r: &mut Xoshiro, marker: &str) -> String {
    let mut w: Vec<&str> = (0..4).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
    let at = (r.next_u64() % 5) as usize;
    w.insert(at, marker);
    w.join(" ")
}

/// Add `n` sampled examples from `make` (returns (state, p_yes)) and refit.
fn fit_on(p: &mut Proxy, si: usize, n: usize, mut make: impl FnMut(usize) -> (Value, f64)) {
    for i in 0..n {
        let (state, py) = make(i);
        let (key, enc) = p.encode_many(&[&state]).unwrap().pop().unwrap();
        p.add_example(si, Kind::Sampled, &state, usize::from(py >= 0.5), Some(vec![1.0 - py, py]), key, enc);
    }
    p.refit(si);
}

fn enc(p: &mut Proxy, state: Value) -> Arc<understudy::encoder::Encoding> {
    p.encode_many(&[&state]).unwrap().pop().unwrap().1
}

fn pred(p: &Proxy, si: usize, e: &understudy::encoder::Encoding) -> understudy::understudent::Pred {
    p.understudents[si].serving().predict(&p.fz, e, 2)
}

/// A stream where "error" means yes, as stream.jsonl lines (id, state, lead_probs).
fn stream(n: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.3;
            let s = text(&mut r, if bad { "error" } else { "ok" });
            let py = if bad { 0.95 } else { 0.05 };
            json!({"id": format!("i{i:04}"), "state": {"call": "run", "output": s}, "lead_probs": [1.0 - py, py]})
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

fn run_replay(p: &mut Proxy, lines: &[Value]) -> Vec<Value> {
    let input: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    let mut out = vec![];
    understudy::replay::run(p, input.as_bytes(), &mut out).unwrap();
    String::from_utf8(out).unwrap().lines().map(|l| {
        let mut v: Value = serde_json::from_str(l).unwrap();
        v.as_object_mut().unwrap().remove("understudent_us");
        v
    }).collect()
}

#[test]
fn graduates_on_separable_not_on_inseparable() {
    let (mut p, si) = proxy(settings(ReadoutChoice::Pool), None, None);
    let mut r = Xoshiro::new(1);
    fit_on(&mut p, si, 200, |_| {
        let bad = r.uniform() < 0.3;
        (json!(text(&mut r, if bad { "error" } else { "ok" })), if bad { 0.95 } else { 0.05 })
    });
    let f = &p.understudents[si].serving().fit;
    assert_eq!(f.mode, Some(Mode::Symmetric), "{}", f.report);
    assert!(f.report["cv_agreement"].as_f64().unwrap() > 0.97);

    let (mut p, si) = proxy(settings(ReadoutChoice::Pool), None, None);
    let mut r = Xoshiro::new(2);
    fit_on(&mut p, si, 200, |_| {
        let yes = r.uniform() < 0.4; // unrelated to the text
        (json!(text(&mut r, "ok")), if yes { 0.9 } else { 0.1 })
    });
    let f = &p.understudents[si].serving().fit;
    assert_eq!(f.mode, None, "{}", f.report);
    assert!(f.report["why_not"].is_string());
}

#[test]
fn too_few_examples_dont_graduate() {
    let (mut p, si) = proxy(settings(ReadoutChoice::Auto), None, None);
    let mut r = Xoshiro::new(3);
    fit_on(&mut p, si, 40, |i| (json!(text(&mut r, if i % 3 == 0 { "error" } else { "ok" })), if i % 3 == 0 { 0.95 } else { 0.05 }));
    let f = &p.understudents[si].serving().fit;
    assert_eq!(f.mode, None);
    assert_eq!(f.report["why_not"], "too few examples");
    assert!(f.has_model());
}

#[test]
fn one_sided_when_yes_is_unpredictable() {
    // "ok" texts are always no; "warn" texts are yes half the time, at random: no confident "yes" exists, but a
    // confident "no" does.
    let (mut p, si) = proxy(settings(ReadoutChoice::Pool), None, None);
    let mut r = Xoshiro::new(4);
    fit_on(&mut p, si, 400, |_| {
        if r.uniform() < 0.7 {
            (json!(text(&mut r, "ok")), 0.03)
        } else {
            let yes = r.uniform() < 0.5;
            (json!(text(&mut r, "warn")), if yes { 0.9 } else { 0.1 })
        }
    });
    let f = &p.understudents[si].serving().fit;
    assert_eq!(f.mode, Some(Mode::OneSided), "{}", f.report);
    let ok = enc(&mut p, json!("alpha ok beta gamma delta"));
    let warn = enc(&mut p, json!("alpha warn beta gamma delta"));
    assert!(pred(&p, si, &ok).eligible);
    assert!(!pred(&p, si, &warn).eligible);
}

#[test]
fn novel_states_are_forwarded() {
    let (mut p, si) = proxy(settings(ReadoutChoice::Pool), None, None);
    let mut r = Xoshiro::new(5);
    fit_on(&mut p, si, 200, |_| {
        let bad = r.uniform() < 0.3;
        (json!(text(&mut r, if bad { "error" } else { "ok" })), if bad { 0.95 } else { 0.05 })
    });
    assert!(p.understudents[si].serving().mode().is_some());
    let known = enc(&mut p, json!("alpha beta ok gamma delta"));
    let novel = enc(&mut p, json!("zebra quokka narwhal axolotl okapi"));
    let (a, b) = (pred(&p, si, &known), pred(&p, si, &novel));
    assert!(!a.novel && a.eligible);
    assert!(b.novel && !b.eligible);
}

#[test]
fn bad_audits_demote_and_good_ones_restore() {
    let lines = stream(300, 6);
    // the lead changes its mind after item 200: everything is "yes"
    let flipped: Vec<Value> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| if i >= 200 { json!({"id": l["id"], "state": l["state"], "lead_probs": [0.05, 0.95]}) } else { l.clone() })
        .collect();
    let mut s = settings(ReadoutChoice::Pool);
    s.audit_rate = 1.0;
    let (mut p, _) = proxy(s, None, Some(replay_lead(&flipped)));
    let dec = run_replay(&mut p, &flipped);
    assert_eq!(dec[150]["answered_by"], "understudent");
    let first_demoted = dec.iter().position(|d| d["demoted"] == true).expect("demoted");
    assert!(first_demoted > 200 && first_demoted < 260, "{first_demoted}");
    // demoted: everything goes to the lead, though the understudent would still answer some
    assert!(dec[first_demoted..].iter().all(|d| d["answered_by"] == "lead" || d["demoted"] == false));

    // audits that agree lift it again
    let st = p.understudents[0].serving_mut();
    if st.mode().is_some() {
        for _ in 0..100 {
            st.audit(1, 1, 0.9);
        }
        assert!(!st.demoted);
    }
}

#[test]
fn pair_swap_flips_a_comparison() {
    // "is a better than b?": better = more "good" words
    let (mut p, si) = proxy(settings(ReadoutChoice::Pool), Some(("a", "b")), None);
    let mut r = Xoshiro::new(7);
    let side = |r: &mut Xoshiro| -> (String, usize) {
        let g = (r.next_u64() % 4) as usize;
        let mut w: Vec<&str> = (0..4).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
        w.extend(std::iter::repeat_n("good", g));
        (w.join(" "), g)
    };
    let mut pairs = vec![];
    while pairs.len() < 400 {
        let ((a, ga), (b, gb)) = (side(&mut r), side(&mut r));
        if ga != gb {
            pairs.push((a, b, ga > gb));
        }
    }
    fit_on(&mut p, si, 300, |i| {
        let (a, b, yes) = &pairs[i];
        (json!({"a": a, "b": b}), if *yes { 0.95 } else { 0.05 })
    });
    let mut flipped = 0;
    for (a, b, _) in &pairs[300..] {
        let x = enc(&mut p, json!({"a": a, "b": b}));
        let y = enc(&mut p, json!({"a": b, "b": a}));
        let (px, py) = (pred(&p, si, &x).p.unwrap(), pred(&p, si, &y).p.unwrap());
        flipped += usize::from((px[1] >= 0.5) != (py[1] >= 0.5));
    }
    assert!(flipped >= 95, "{flipped} of 100 flipped");
}

#[test]
fn replay_is_deterministic() {
    let lines = stream(260, 8);
    let run = || {
        let (mut p, _) = proxy(settings(ReadoutChoice::Auto), None, Some(replay_lead(&lines)));
        run_replay(&mut p, &lines)
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b);
    assert!(a.iter().any(|d| d["answered_by"] == "understudent"));
    assert!(a.iter().any(|d| d["audited"] == true));
    assert_eq!(a[0]["understudent_probs"], Value::Null);
}

fn tempdir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("understudy-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn snapshot_round_trip_predicts_identically() {
    for readout in [ReadoutChoice::Pool, ReadoutChoice::Attn] {
        let lines = stream(150, 9);
        let (mut p, si) = proxy(settings(readout), None, Some(replay_lead(&lines)));
        run_replay(&mut p, &lines);
        assert!(p.understudents[si].serving().fit.has_model());
        let dir = tempdir(&format!("roundtrip-{readout:?}"));
        snapshot::save(&p, &dir).unwrap();
        let q = snapshot::load(&dir, fake(), None, |_| {}).unwrap();
        {
        let (a, b) = (&p.understudents[si], &q.understudents[si]);
        assert_eq!(*a.serving().fit, *b.serving().fit);
        assert_eq!(a.serving().audits, b.serving().audits);
        assert_eq!(a.examples.len(), b.examples.len());
        assert!(a.examples.iter().zip(&b.examples).all(|(x, y)| x.seq == y.seq && x.state == y.state && *x.enc == *y.enc));
        }
        assert_eq!(p.rng, q.rng);
        for l in stream(20, 10) {
            let e = enc(&mut p, l["state"].clone());
            assert_eq!(pred(&p, si, &e).p, pred(&q, si, &e).p);
        }
        assert!(snapshot::load(&dir, Arc::new(Fake { hidden: 8 }), None, |_| {}).is_err(), "another encoder must be refused");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn train_then_judge_matches_replay() {
    let lines = stream(320, 11);
    let s = settings(ReadoutChoice::Auto);
    let (mut whole, _) = proxy(s.clone(), None, Some(replay_lead(&lines)));
    let want = run_replay(&mut whole, &lines);
    assert_eq!(want[99]["mode"], Value::Null, "the comparison needs no graduation in the first 100 items");

    let judge = || -> Vec<Value> {
        // train on the first 100 (as `understudy train`), save, load, judge the rest with saves in between
        let dir = tempdir("split");
        let (mut p, si) = proxy(s.clone(), None, None);
        understudy::ops::seed(&mut p, si, &lines[..100]).unwrap();
        p.refit(si);
        snapshot::save(&p, &dir).unwrap();
        let mut out = vec![];
        for chunk in lines[100..].chunks(70) {
            let mut q = snapshot::load(&dir, fake(), Some(Box::new(replay_lead(&lines))), |_| {}).unwrap();
            for l in chunk {
                let (mut v, _) = understudy::ops::judge_line(&mut q, &json!({"id": l["id"], "state": l["state"]})).unwrap();
                v.as_object_mut().unwrap().remove("understudent_us");
                out.push(v);
            }
            snapshot::save(&q, &dir).unwrap();
        }
        std::fs::remove_dir_all(&dir).unwrap();
        out
    };
    let (a, b) = (judge(), judge());
    assert_eq!(a, b, "train → judge is deterministic");
    if let Some(i) = a.iter().zip(&want[100..]).position(|(x, y)| x != y) {
        panic!("first difference at item {}: judge {} vs replay {}", i + 100, a[i], want[100 + i]);
    }
    assert_eq!(a.len(), want.len() - 100);
}

#[test]
fn frozen_judge_learns_nothing_and_matches_the_unfrozen_first_answers() {
    let lines = stream(320, 11);
    let s = settings(ReadoutChoice::Auto);
    let build = |frozen: bool| -> (Proxy, usize) {
        let (mut p, si) = proxy(s.clone(), None, Some(replay_lead(&lines)));
        understudy::ops::seed(&mut p, si, &lines[..100]).unwrap();
        p.refit(si);
        p.frozen = frozen;
        (p, si)
    };
    let (mut frozen, si) = build(true);
    let (mut live, _) = build(false);
    let n_examples = frozen.understudents[si].examples.len();
    for (i, l) in lines[100..].iter().enumerate() {
        let (mut a, _) = understudy::ops::judge_line(&mut frozen, &json!({"id": l["id"], "state": l["state"]})).unwrap();
        let (mut b, _) = understudy::ops::judge_line(&mut live, &json!({"id": l["id"], "state": l["state"]})).unwrap();
        for v in [&mut a, &mut b] {
            v.as_object_mut().unwrap().remove("understudent_us");
        }
        if i < 20 {
            assert_eq!(a, b, "before the live understudent's first refit both are the same fit (item {i})");
        }
        assert_eq!(a["audited"], json!(false), "a frozen judge records no audits");
    }
    assert_eq!(frozen.understudents[si].examples.len(), n_examples, "nothing is added to a frozen understudent");
    assert_eq!(frozen.understudents[si].since_refit, 0);
    assert!(live.understudents[si].examples.len() > n_examples, "the unfrozen one learned");
}
