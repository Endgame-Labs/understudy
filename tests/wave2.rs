//! The `net` and `checklist` understudents and `--abstain calibrated` through the proxy on a fake encoder: determinism,
//! learning, snapshot round trips, status output and a place in `auto`.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use understudy::abstain::Abstain;
use understudy::encoder::{Encoder, Fake, Featurizer};
use understudy::lead::{Lead, Replay};
use understudy::snapshot;
use understudy::proxy::{Proxy, Settings};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::understudent::{Mode, UnderstudentKind};

const H: usize = 16;

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the assistant refuse?"}), None).unwrap()
}

fn proxy(kind: UnderstudentKind, tiers: &[UnderstudentKind], abstain: Abstain, lead: Replay) -> (Proxy, usize) {
    let mut s = Settings::default();
    s.fit.understudent = kind;
    s.fit.tiers = tiers.to_vec();
    s.fit.abstain = abstain;
    s.audit_rate = 0.1;
    if kind == UnderstudentKind::Auto {
        s.cost_us = vec![10.0; tiers.len()];
    }
    let enc: Arc<dyn Encoder> = Arc::new(Fake { hidden: H });
    let fz = Featurizer::for_understudent(kind, None, None, None, H).unwrap();
    let mut p = Proxy::new(s, fz, enc, Some(Box::new(lead) as Box<dyn Lead>));
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];

fn words(r: &mut Xoshiro, n: usize) -> String {
    (0..n).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect::<Vec<_>>().join(" ")
}

fn stream(n: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.4;
            let answer = if bad { format!("I'm sorry, I cannot help with that. {}", words(&mut r, 3)) } else { let len = 40 + (r.next_u64() % 20) as usize; words(&mut r, len) };
            let py = if bad { 0.95 } else { 0.05 };
            json!({"id": format!("i{i:04}"), "state": {"question": words(&mut r, 6), "answer": answer}, "lead_probs": [1.0 - py, py]})
        })
        .collect()
}

fn lead(lines: &[Value]) -> Replay {
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

fn strip(rows: &[Value]) -> Vec<Value> {
    rows.iter().cloned().map(|mut d| { d.as_object_mut().unwrap().remove("understudent_us"); d }).collect()
}

fn local(rows: &[Value]) -> usize {
    rows.iter().filter(|d| d["answered_by"] == "understudent").count()
}

fn agreement(rows: &[Value], lines: &[Value]) -> f64 {
    let (mut n, mut ok) = (0, 0);
    for (d, l) in rows.iter().zip(lines) {
        if d["answered_by"] == "understudent" {
            n += 1;
            ok += usize::from(d["label"].as_u64().unwrap() as usize == usize::from(l["lead_probs"][1].as_f64().unwrap() > 0.5));
        }
    }
    ok as f64 / n.max(1) as f64
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("understudy-wave2-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn net_and_checklist_replay_deterministically_learn_and_round_trip() {
    let lines = stream(400, 1);
    for kind in [UnderstudentKind::Net, UnderstudentKind::Checklist] {
        let name = kind.name();
        let (mut a, _) = proxy(kind, &[], Abstain::Single, lead(&lines));
        let (mut b, si) = proxy(kind, &[], Abstain::Single, lead(&lines));
        let (sum, rows) = run(&mut a, &lines);
        let (_, rows2) = run(&mut b, &lines);
        assert_eq!(strip(&rows), strip(&rows2), "{name}: same stream, same decisions");
        assert_eq!(sum["understudent"], name);
        let f = &b.understudents[si].serving().fit;
        assert_eq!(f.mode, Some(Mode::Symmetric), "{name}: {}", f.report);
        assert!(local(&rows) > 80, "{name}: local {}", local(&rows));
        assert!(agreement(&rows, &lines) > 0.9, "{name}");
        assert!(rows.iter().any(|d| d["readout"] == name));
        // a snapshot round trip keeps the fit and its predictions exactly
        let dir = tmp(name);
        snapshot::save(&b, &dir).unwrap();
        let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
        let (_, enc) = b.encode_many(&[&lines[3]["state"]]).unwrap().pop().unwrap();
        let (fa, fb) = (&b.understudents[si].serving().fit, &q.understudents[si].serving().fit);
        assert_eq!((&fa.net, &fa.checklist, fa.mode, fa.thr), (&fb.net, &fb.checklist, fb.mode, fb.thr), "{name}");
        assert_eq!(fa.proba(&b.fz, &enc, 2), fb.proba(&q.fz, &enc, 2), "{name}");
    }
}

#[test]
fn checklist_status_lists_its_top_rules_by_feature_name() {
    let lines = stream(300, 2);
    let (mut p, si) = proxy(UnderstudentKind::Checklist, &[], Abstain::Single, lead(&lines));
    run(&mut p, &lines);
    let st = p.understudents[si].status();
    let listing = st["report"]["checklist"]["listing"].as_str().unwrap();
    assert!(listing.starts_with("1. IF "), "{listing}");
    assert!(listing.contains("THEN yes") && listing.contains("conf "), "{listing}");
    assert!(!listing.contains("gram#"), "the states' texts are known during a replay, so n-grams print as text: {listing}");
    // after a snapshot round trip the listing is still in the saved report (the status command reads it from there)
    let dir = tmp("checklist-status");
    snapshot::save(&p, &dir).unwrap();
    let meta = snapshot::read_meta(&dir).unwrap();
    assert!(meta["understudents"][0]["fit"]["report"]["checklist"]["listing"].as_str().unwrap().contains("IF "));
}

#[test]
fn calibrated_abstention_through_the_proxy_is_deterministic_and_certified() {
    let lines = stream(500, 3);
    let mk = |abstain| proxy(UnderstudentKind::Shape, &[], abstain, lead(&lines));
    let (mut s, si) = mk(Abstain::Single);
    let (mut c1, _) = mk(Abstain::Calibrated);
    let (mut c2, _) = mk(Abstain::Calibrated);
    let (_, single) = run(&mut s, &lines);
    let (_, cal1) = run(&mut c1, &lines);
    let (_, cal2) = run(&mut c2, &lines);
    assert_eq!(strip(&cal1), strip(&cal2));
    let (fs, fc) = (&s.understudents[si].serving().fit, &c1.understudents[si].serving().fit);
    assert_eq!(fs.mode, Some(Mode::Symmetric));
    assert!(fs.class_thr.is_none());
    assert_eq!(fc.report["abstain"], "calibrated");
    // the certified policy fits on its train part and graduates only on a passing certificate over held-out rows: it
    // need not match the single threshold's coverage, but whatever it serves it certified
    assert_eq!(fc.report["split"]["certify"].as_u64().unwrap() + fc.report["split"]["select"].as_u64().unwrap() + fc.report["split"]["train"].as_u64().unwrap(), fc.report["n"].as_u64().unwrap());
    if let Some(t) = &fc.class_thr {
        assert_eq!(t.len(), 2);
        assert_eq!(fc.mode, Some(Mode::Symmetric));
        let c = &fc.report["certificate"];
        assert!(c["passed"] == true && c["answered"].as_u64().unwrap() >= 30 && c["wilson_low"].as_f64().unwrap() >= 0.9, "{c}");
        assert!(agreement(&cal1, &lines) > 0.9);
    } else {
        assert!(fc.mode.is_none(), "calibrated never graduates without a certificate");
    }
    // class thresholds survive a snapshot round trip
    let dir = tmp("cal");
    snapshot::save(&c1, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[si].serving().fit.class_thr, fc.class_thr);
    assert_eq!(q.settings.fit.abstain, Abstain::Calibrated);
}

#[test]
fn net_and_checklist_race_in_auto_without_changing_the_default_race() {
    let lines = stream(400, 4);
    let tiers = [UnderstudentKind::Keywords, UnderstudentKind::Checklist, UnderstudentKind::Net, UnderstudentKind::Deep];
    let (mut p, si) = proxy(UnderstudentKind::Auto, &tiers, Abstain::Single, lead(&lines));
    let (sum, rows) = run(&mut p, &lines);
    let race: Vec<&str> = sum["candidates"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert_eq!(race, ["keywords", "checklist", "net", "deep"]);
    assert!(local(&rows) > 50, "auto should graduate something");
    let r = p.understudents[si].race.as_array().unwrap().clone();
    assert!(r.iter().any(|c| c["understudent"] == "net" && c["cv_agreement"].is_number()), "{r:?}");
    assert!(r.iter().any(|c| c["understudent"] == "checklist" && c["cv_agreement"].is_number()), "{r:?}");
    // the default candidates are unchanged
    assert_eq!(UnderstudentKind::default_tiers(), [UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep]);
}
