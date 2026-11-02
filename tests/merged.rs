//! The `rules` and `auto` understudents together (a ladder with a rules tier, auto racing `rules` among its candidates,
//! the rules LLM mocked), and the `one_sided_yes` graduation mode.

use anyhow::Result;
use understudy::encoder::{Encoder, Fake, Featurizer};
use understudy::snapshot;
use understudy::proxy::{Proxy, Settings};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::rules::{RulesClient, RulesLlm};
use understudy::understudent::{Fit, Mode, Serving, UnderstudentKind, graduation};
use understudy::lead::{Replay, Lead};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

const H: usize = 16;

struct Canned(&'static str);

impl RulesLlm for Canned {
    fn complete(&self, _: &Value) -> Result<String> {
        Ok(self.0.to_string())
    }
}

const RULES_YES_NO: &str = r#"{"rules": [{"pattern": "(?i)\\berror\\b", "label": "true", "why": "the word error"}], "default": "false"}"#;
const RULES_YES_ONLY: &str = r#"{"rules": [{"pattern": "\\berror\\b", "label": "true", "why": "the word error"}], "default": "abstain"}"#;

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap()
}

fn tiered(kind: UnderstudentKind, tiers: &[UnderstudentKind], reply: &'static str, lead: Option<Replay>, tweak: impl FnOnce(&mut Settings)) -> (Proxy, usize) {
    let mut s = Settings::default();
    s.fit.understudent = kind;
    s.fit.tiers = tiers.to_vec();
    s.audit_rate = 0.1;
    tweak(&mut s);
    let enc: Arc<dyn Encoder> = Arc::new(Fake { hidden: H });
    let fz = Featurizer::for_understudent(kind, None, None, None, H).unwrap();
    let mut p = Proxy::new(s, fz, enc, lead.map(|t| Box::new(t) as Box<dyn Lead>));
    p.rules = Some(Arc::new(RulesClient::new(Box::new(Canned(reply)))));
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];

fn text(r: &mut Xoshiro, marker: &str) -> String {
    let mut w: Vec<&str> = (0..6).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
    w.insert((r.next_u64() % (w.len() as u64 + 1)) as usize, marker);
    w.join(" ")
}

/// `n` lines, a share `p_yes` of them carrying the marker "error" (lead says yes).
fn stream(n: usize, seed: u64, p_yes: f64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < p_yes;
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

fn run(p: &mut Proxy, lines: &[Value]) -> (Value, Vec<Value>) {
    let input: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    let mut out = vec![];
    let sum = understudy::replay::run(p, input.as_bytes(), &mut out).unwrap();
    let rows = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    (sum, rows)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("understudy-merged-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_ladder_with_a_rules_tier_keeps_both_features_in_its_summary() {
    let lines = stream(600, 1, 0.4);
    let tiers = [UnderstudentKind::Rules, UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep];
    let (mut p, si) = tiered(UnderstudentKind::Ladder, &tiers, RULES_YES_NO, Some(replay_lead(&lines)), |_| {});
    let (sum, rows) = run(&mut p, &lines);
    assert_eq!(sum["understudent"], "ladder");
    assert_eq!(sum["tiers"].as_array().unwrap().len(), 5);
    assert!(sum["rules_llm"]["live_calls"].as_u64().unwrap() >= 1);
    assert!(sum["rules"]["listing"].as_str().unwrap().contains("error"));
    assert!(rows.iter().any(|d| d["answered_by"] == "understudent" && d["readout"] == "rules"));
    assert_eq!(p.understudents[si].tiers[0].serving.mode(), Some(Mode::Symmetric));
    let st = p.understudents[si].status();
    assert!(st["tiers"][0]["rules"]["listing"].as_str().is_some());
}

#[test]
fn auto_races_rules_as_a_candidate_on_heldout_coverage() {
    let lines = stream(600, 2, 0.4);
    let cands = [UnderstudentKind::Rules, UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep];
    let (mut p, si) = tiered(UnderstudentKind::Auto, &cands, RULES_YES_NO, Some(replay_lead(&lines)), |s| s.cost_us = vec![5.0, 20.0, 300.0, 40_000.0, 400_000.0]);
    let (sum, rows) = run(&mut p, &lines);
    let race = p.understudents[si].race.as_array().unwrap();
    assert_eq!(race.len(), 5);
    assert_eq!(race[0]["understudent"], "rules");
    assert!(!race[0]["mode"].is_null(), "rules graduates on its held-out rows: {}", race[0]);
    let rep = &p.understudents[si].tiers[0].serving.fit.report;
    assert!(rep["heldout_n"].as_u64().unwrap() > 0 && rep["heldout_n"].as_u64().unwrap() < 600);
    // rules is the cheapest and, on a separable stream, within one SE of the best coverage: it serves
    assert_eq!(p.understudents[si].serving_name(), Some("rules"), "{race:?}");
    assert_eq!(sum["serving"], "rules");
    assert_eq!(sum["candidates"].as_array().unwrap().len(), 5);
    assert!(sum["serving_history"].as_array().is_some() && sum["lead_only"].is_null());
    assert!(sum["rules_llm"]["live_calls"].as_u64().unwrap() >= 1);
    assert!(sum["rules"]["listing"].as_str().unwrap().contains("error"));
    assert!(rows.iter().any(|d| d["readout"] == "rules"));
    let status = p.understudents[si].status();
    assert_eq!(status["serving"], "rules");
    assert!(status["rules"]["listing"].as_str().is_some());

    // the snapshot keeps the race, the serving candidate and the rule bookkeeping
    let dir = tmp("auto");
    snapshot::save(&p, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[si].serving_idx, p.understudents[si].serving_idx);
    assert_eq!(q.understudents[si].race, p.understudents[si].race);
    assert_eq!(q.understudents[si].rules, p.understudents[si].rules);
    assert_eq!(q.understudents[si].tiers[0].serving.fit.rules, p.understudents[si].tiers[0].serving.fit.rules);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_default_auto_candidates_do_not_include_rules() {
    assert!(!UnderstudentKind::default_tiers().contains(&UnderstudentKind::Rules));
}

// ------------------------------------------------------------------------------------------- one_sided_yes

type Rows = (Vec<f64>, Vec<f64>, Vec<bool>, Vec<bool>, Vec<usize>);

/// 300 confident yes (`false_yes` of them, the most confident, lead-no), 20 unsure yes, 200 no with a low P(yes): only the
/// mirrored scan can graduate (the "no" rows are only just on the "no" side, so no symmetric threshold keeps them).
fn yes_only_rows(false_yes: usize) -> Rows {
    let mut p_yes = vec![];
    let mut y = vec![];
    for i in 0..300 {
        p_yes.push(0.97 + (i % 10) as f64 * 0.001);
        y.push(usize::from(!(i % 10 == 9 && i / 10 < false_yes)));
    }
    for i in 0..20 {
        p_yes.push(0.2 + i as f64 * 0.001);
        y.push(1);
    }
    for i in 0..200 {
        p_yes.push(0.41 + i as f64 * 0.0004);
        y.push(0);
    }
    let cf: Vec<f64> = p_yes.iter().map(|p| p.max(1.0 - p)).collect();
    let agree: Vec<bool> = p_yes.iter().zip(&y).map(|(p, y)| (*p > 0.5) == (*y == 1)).collect();
    let maj_yes = y.iter().filter(|&&c| c == 1).count() * 2 > y.len();
    let minor: Vec<bool> = y.iter().map(|&c| (c == 1) != maj_yes).collect();
    (p_yes, cf, agree, minor, y)
}

#[test]
fn one_sided_yes_graduates_when_only_confident_yes_is_answered_locally() {
    let (p_yes, cf, agree, minor, y) = yes_only_rows(0);
    // the symmetric scan fails (the 20 unsure yes rows disagree: agreement is 0.96 < 0.99)
    let (cov, mode, thr) = graduation(&cf, &agree, &minor, Some(&p_yes), &y, 0.99, 0.8);
    assert_eq!(mode, Some(Mode::OneSidedYes), "coverage {cov} thr {thr}");
    assert!((cov - 300.0 / 520.0).abs() < 1e-9 && thr > 0.5 && thr < 0.97, "{cov} {thr}");
    assert_eq!(Mode::OneSidedYes.name(), "one_sided_yes");
    // false "yes" among the local ones (3 of 300: upper bound above 2%) blocks it
    let (p_yes, cf, agree, minor, y) = yes_only_rows(3);
    let (_, mode, _) = graduation(&cf, &agree, &minor, Some(&p_yes), &y, 0.99, 0.8);
    assert_ne!(mode, Some(Mode::OneSidedYes));
    // abstentions (P(yes) above 1) are never local
    let (mut p_yes, cf, agree, minor, y) = yes_only_rows(0);
    p_yes.iter_mut().take(250).for_each(|p| *p = 2.0);
    let (_, mode, _) = graduation(&cf, &agree, &minor, Some(&p_yes), &y, 0.99, 0.8);
    assert_ne!(mode, Some(Mode::OneSidedYes), "only 50 local rows are left: too few for a 2% bound");
}

fn mode_fit(mode: Mode, thr: f64) -> Fit {
    Fit { mode: Some(mode), thr, ..Default::default() }
}

#[test]
fn one_sided_yes_is_demoted_while_the_false_yes_rate_is_too_high_and_recovers() {
    let mut s = Serving::default();
    s.install(mode_fit(Mode::OneSidedYes, 0.9), 0.97);
    // clean audits never demote
    for _ in 0..60 {
        assert!(s.audit(1, 1, 0.97));
    }
    assert!(!s.demoted);
    // the lead says "no" to local "yes" answers: the Wilson lower bound of the false-yes rate passes 2%
    for _ in 0..8 {
        s.audit(1, 0, 0.97);
    }
    assert!(s.demoted, "{}", s.audit_summary());
    assert_eq!(s.audit_summary()["misses"], 8);
    // a window of clean audits (the bad ones age out of the 100) promotes it again
    for _ in 0..100 {
        s.audit(1, 1, 0.97);
    }
    assert!(!s.demoted);
}

#[test]
fn rules_that_only_confirm_yes_graduate_as_one_sided_yes_end_to_end() {
    // a yes-majority question: the symmetric scan finds no minority rows among the answered ones, but the "yes"
    // rule is precise and everything else is forwarded
    let lines = stream(700, 5, 0.7);
    let (mut p, si) = tiered(UnderstudentKind::Rules, &[], RULES_YES_ONLY, Some(replay_lead(&lines)), |_| {});
    let (sum, rows) = run(&mut p, &lines);
    let serving = p.understudents[si].serving();
    assert_eq!(serving.mode(), Some(Mode::OneSidedYes), "{}", serving.fit.report);
    assert!(rows.iter().any(|d| d["mode"] == "one_sided_yes" && d["answered_by"] == "understudent" && d["label"] == 1), "{sum}");
    assert!(rows.iter().filter(|d| d["answered_by"] == "understudent").all(|d| d["label"] == 1), "only yes is local");
    assert_eq!(p.understudents[si].status()["mode"], "one_sided_yes");
    // probes round trip the mode
    let dir = tmp("yes");
    snapshot::save(&p, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[si].serving().mode(), Some(Mode::OneSidedYes));
    assert_eq!(q.understudents[si].serving().fit.thr, serving.fit.thr);
    let _ = std::fs::remove_dir_all(&dir);
}
