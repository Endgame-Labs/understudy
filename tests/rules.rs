//! The `rules` understudent through the proxy, with the LLM mocked behind the `RulesLlm` trait (no network): proposal,
//! held-out graduation, abstention, rejection of too-broad rules, cache replays offline with identical decisions,
//! snapshot round trips, and a ladder answering from `rules` without computing any other tier's features.

use anyhow::{Result, bail};
use understudy::encoder::{Encoder, Fake, Featurizer, NoEncoder, Tokens};
use understudy::snapshot;
use understudy::proxy::{Kind, Proxy, Settings};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::rules::{RulesClient, RulesLlm};
use understudy::understudent::{Mode, UnderstudentKind};
use understudy::lead::{Replay, Lead};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const H: usize = 16;

struct Mock {
    reply: String,
    calls: AtomicUsize,
}

impl Mock {
    fn new(reply: &str) -> Arc<Mock> {
        Arc::new(Mock { reply: reply.into(), calls: AtomicUsize::new(0) })
    }
}

struct Shared(Arc<Mock>);

impl RulesLlm for Shared {
    fn complete(&self, _: &Value) -> Result<String> {
        self.0.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.0.reply.clone())
    }
}

struct Offline;
impl RulesLlm for Offline {
    fn complete(&self, _: &Value) -> Result<String> {
        bail!("offline: this request is not in the cache")
    }
}

/// An encoder that counts every call.
struct Spy {
    inner: Fake,
    calls: AtomicUsize,
}

impl Encoder for Spy {
    fn hidden(&self) -> usize {
        H
    }
    fn fingerprint(&self) -> Value {
        self.inner.fingerprint()
    }
    fn encode(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<Tokens>> {
        self.calls.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.encode(texts, layers)
    }
    fn encode_input(&self, texts: &[&str]) -> Result<Vec<Tokens>> {
        self.calls.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.encode_input(texts)
    }
}

const RULES_YES_NO: &str = r#"{"rules": [{"pattern": "(?i)\\berror\\b", "label": "true", "why": "the word error"}], "default": "false"}"#;
const RULES_YES_ONLY: &str = r#"{"rules": [{"pattern": "\\berror\\b", "label": "true", "why": "the word error"}], "default": "abstain"}"#;
const RULES_BROAD: &str = r#"{"rules": [{"pattern": ".", "label": "true", "why": "everything"}, {"pattern": "(", "label": "true"}], "default": "abstain"}"#;

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap()
}

fn proxy(kind: UnderstudentKind, tiers: &[UnderstudentKind], enc: Arc<dyn Encoder>, client: Option<RulesClient>, lead: Option<Replay>, audit: f64) -> (Proxy, usize) {
    let mut s = Settings::default();
    s.fit.understudent = kind;
    s.fit.tiers = tiers.to_vec();
    s.audit_rate = audit;
    let fz = Featurizer::for_understudent(kind, None, None, None, enc.hidden()).unwrap();
    let mut p = Proxy::new(s, fz, enc, lead.map(|t| Box::new(t) as Box<dyn Lead>));
    if let Some(c) = client {
        p.rules = Some(Arc::new(c));
    }
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];

fn text(r: &mut Xoshiro, marker: &str) -> String {
    let mut w: Vec<&str> = (0..6).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
    w.insert((r.next_u64() % (w.len() as u64 + 1)) as usize, marker);
    w.join(" ")
}

fn fit_on(p: &mut Proxy, si: usize, n: usize, seed: u64) {
    let mut r = Xoshiro::new(seed);
    for _ in 0..n {
        let yes = r.uniform() < 0.35;
        let state = json!(text(&mut r, if yes { "error" } else { "fine" }));
        let (key, enc) = p.encode_many(&[&state]).unwrap().pop().unwrap();
        let py = if yes { 0.95 } else { 0.05 };
        p.add_example(si, Kind::Sampled, &state, usize::from(yes), Some(vec![1.0 - py, py]), key, enc);
    }
    p.refit(si);
}

fn stream(n: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.3;
            let t = text(&mut r, if bad { "error" } else { "fine" });
            let py = if bad { 0.95 } else { 0.05 };
            json!({"id": format!("i{i:04}"), "state": {"call": "run", "output": t}, "lead_probs": [1.0 - py, py]})
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
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| {
            let mut v: Value = serde_json::from_str(l).unwrap();
            v.as_object_mut().unwrap().remove("understudent_us");
            v
        })
        .collect()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("understudy-rules-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn rules_propose_validate_on_heldout_and_graduate() {
    let mock = Mock::new(RULES_YES_NO);
    let (mut p, si) = proxy(UnderstudentKind::Rules, &[], Arc::new(NoEncoder), Some(RulesClient::new(Box::new(Shared(mock.clone())))), None, 0.0);
    fit_on(&mut p, si, 300, 1);
    let serving = p.understudents[si].serving();
    assert_eq!(serving.mode(), Some(Mode::Symmetric), "{}", serving.fit.report);
    assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    let set = serving.fit.rules.as_ref().unwrap();
    assert_eq!(set.shown.len(), 40);
    assert_eq!(set.heldout.n, 300 - 40);
    assert_eq!(set.rules.len(), 1);
    assert!(set.rules[0].confidence > 0.9, "{}", set.describe());
    // held-out rows only: the shown examples are never counted for the rule
    assert_eq!(set.rules[0].heldout_matches + set.default_matches, 300 - 40);
    // another refit with nothing new revalidates and does not call the LLM again
    p.refit(si);
    assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    assert_eq!(p.understudents[si].serving().mode(), Some(Mode::Symmetric));
    // status shows the rules for a human
    let st = p.understudents[si].status();
    assert!(st["rules"]["listing"].as_str().unwrap().contains("error"));
}

#[test]
fn an_abstaining_default_is_never_local() {
    let mock = Mock::new(RULES_YES_ONLY);
    let (mut p, si) = proxy(UnderstudentKind::Rules, &[], Arc::new(NoEncoder), Some(RulesClient::new(Box::new(Shared(mock)))), None, 0.0);
    fit_on(&mut p, si, 300, 2);
    assert!(p.understudents[si].serving().mode().is_some(), "{}", p.understudents[si].serving().fit.report);
    let fz = p.fz.clone();
    let enc_of = |p: &mut Proxy, t: &str| p.encode_many(&[&json!(t)]).unwrap().pop().unwrap().1;
    let yes = enc_of(&mut p, "an error here");
    let no = enc_of(&mut p, "all fine here");
    let s = p.understudents[si].serving();
    let (py, pn) = (s.predict(&fz, &yes, 2), s.predict(&fz, &no, 2));
    assert!(py.local && py.p.unwrap()[1] > 0.5);
    assert!(pn.p.is_none() && !pn.eligible && !pn.local, "an abstention is not eligible");
    let cov = s.fit.report["cv_coverage"].as_f64().unwrap();
    assert!(cov > 0.2 && cov < 0.5, "coverage is the answered share only: {cov}");
}

#[test]
fn too_broad_and_uncompilable_rules_are_rejected() {
    let mock = Mock::new(RULES_BROAD);
    let (mut p, si) = proxy(UnderstudentKind::Rules, &[], Arc::new(NoEncoder), Some(RulesClient::new(Box::new(Shared(mock.clone())))), None, 0.0);
    fit_on(&mut p, si, 300, 3);
    let s = p.understudents[si].serving();
    assert!(s.fit.rules.is_none() && s.mode().is_none());
    assert_eq!(p.understudents[si].rules.proposals, 1);
    assert!(s.fit.report["llm_error"].as_str().unwrap().contains("no usable rules"));
    // a failing question waits for 100 new examples before the next proposal, and asks at most 5 times in all
    p.refit(si);
    assert_eq!(mock.calls.load(Ordering::SeqCst), 1);
    for round in 0..8 {
        fit_on(&mut p, si, 100, 10 + round);
    }
    assert_eq!(mock.calls.load(Ordering::SeqCst), 5);
}

#[test]
fn warm_cache_replays_offline_with_identical_decisions() {
    let lines = stream(300, 7);
    let dir = tmp("cache");
    let file = dir.join("cache.jsonl");
    let live = Mock::new(RULES_YES_NO);
    let client = RulesClient::new(Box::new(Shared(live.clone())));
    client.attach(&file).unwrap();
    let (mut p, _) = proxy(UnderstudentKind::Rules, &[], Arc::new(NoEncoder), Some(client), Some(replay_lead(&lines)), 0.1);
    let a = run_replay(&mut p, &lines);
    assert!(live.calls.load(Ordering::SeqCst) >= 1);
    assert!(a.iter().any(|d| d["answered_by"] == "understudent" && d["readout"] == "rules"));
    assert!(a.iter().any(|d| d["audited"] == true));
    let client = RulesClient::new(Box::new(Offline));
    assert!(client.attach(&file).unwrap() >= 1);
    let (mut p2, _) = proxy(UnderstudentKind::Rules, &[], Arc::new(NoEncoder), Some(client), Some(replay_lead(&lines)), 0.1);
    let b = run_replay(&mut p2, &lines);
    assert_eq!(a, b);
    assert_eq!(p2.rules.as_ref().unwrap().live.load(Ordering::SeqCst), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_ladder_answers_from_rules_without_computing_other_tiers() {
    let spy = Arc::new(Spy { inner: Fake { hidden: H }, calls: AtomicUsize::new(0) });
    let mock = Mock::new(RULES_YES_NO);
    let tiers = [UnderstudentKind::Rules, UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep];
    let (mut p, si) = proxy(UnderstudentKind::Ladder, &tiers, spy.clone(), Some(RulesClient::new(Box::new(Shared(mock)))), None, 0.0);
    fit_on(&mut p, si, 300, 4);
    assert_eq!(p.understudents[si].tiers[0].serving.mode(), Some(Mode::Symmetric));
    assert!(spy.calls.load(Ordering::SeqCst) > 0);
    spy.calls.store(0, Ordering::SeqCst);
    for t in ["only fine words", "so much error here"] {
        let o = p.answer(None, &json!(t), None).unwrap();
        let d = &o.decisions[0].1;
        assert_eq!((o.answered_by, d.readout.as_deref()), ("understudent", Some("rules")));
        assert_eq!(d.label, usize::from(t.contains("error")));
    }
    assert_eq!(spy.calls.load(Ordering::SeqCst), 0, "no other tier's features were computed");
    // rules must be the first tier
    assert!(understudy::proxy::check_tiers(&[UnderstudentKind::Keywords, UnderstudentKind::Rules]).is_err());
    assert!(understudy::proxy::check_tiers(&[UnderstudentKind::Rules, UnderstudentKind::Keywords]).is_ok());
}

#[test]
fn snapshots_round_trip_the_rule_set_and_its_bookkeeping() {
    let lines = stream(300, 9);
    let dir = tmp("snapshot");
    let mock = Mock::new(RULES_YES_NO);
    let (mut p, si) = proxy(UnderstudentKind::Rules, &[], Arc::new(NoEncoder), Some(RulesClient::new(Box::new(Shared(mock)))), Some(replay_lead(&lines)), 0.0);
    run_replay(&mut p, &lines);
    snapshot::save(&p, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(NoEncoder), None, |_| {}).unwrap();
    let (a, b) = (p.understudents[si].serving(), q.understudents[si].serving());
    assert_eq!(a.fit.rules, b.fit.rules);
    assert_eq!(a.mode(), b.mode());
    assert_eq!(p.understudents[si].rules, q.understudents[si].rules);
    assert!(q.understudents[si].examples.iter().all(|e| e.enc.rules.is_some()));
    // the loaded set answers like the original
    let inp = understudy::rules::RuleInput::from_state(&json!("an error occurred"));
    assert_eq!(a.fit.rules.as_ref().unwrap().apply(&inp), b.fit.rules.as_ref().unwrap().apply(&inp));
    assert!(b.fit.rules.as_ref().unwrap().apply(&inp).is_some());
    let meta = snapshot::read_meta(&dir).unwrap();
    assert!(meta["understudents"][0]["fit"]["rules"]["rules"][0]["pattern"].as_str().unwrap().contains("error"));
    let _ = std::fs::remove_dir_all(&dir);
}
