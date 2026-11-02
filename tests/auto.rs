//! `--understudent auto`: race the candidates at every refit and serve one. Near-ties go to the cheaper candidate, a
//! clearly higher coverage wins, a switch resets the audits, local answers compute only the serving understudent's
//! features, an unlearnable question becomes lead-only (and stops refitting), and replays are deterministic.

use anyhow::Result;
use understudy::encoder::{Encoder, Fake, Featurizer, Tokens};
use understudy::proxy::{Cand, Proxy, Settings, RECHECK_EVERY, model_cost_us, select_fastest, select_serving};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::understudent::UnderstudentKind;
use understudy::lead::{Replay, Lead};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const H: usize = 16;

#[derive(Default)]
struct Spy {
    calls: AtomicUsize,
    input_calls: AtomicUsize,
}

impl Encoder for Spy {
    fn hidden(&self) -> usize {
        H
    }
    fn fingerprint(&self) -> Value {
        Fake { hidden: H }.fingerprint()
    }
    fn encode(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<Tokens>> {
        self.calls.fetch_add(texts.len(), Ordering::SeqCst);
        Fake { hidden: H }.encode(texts, layers)
    }
    fn encode_input(&self, texts: &[&str]) -> Result<Vec<Tokens>> {
        self.input_calls.fetch_add(texts.len(), Ordering::SeqCst);
        Fake { hidden: H }.encode_input(texts)
    }
}

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap()
}

fn auto(enc: Arc<dyn Encoder>, lead: Option<Replay>, audit_rate: f64, tweak: impl FnOnce(&mut Settings)) -> (Proxy, usize) {
    let mut s = Settings::default();
    s.fit.understudent = UnderstudentKind::Auto;
    s.fit.tiers = UnderstudentKind::default_tiers();
    s.audit_rate = audit_rate;
    tweak(&mut s);
    let fz = Featurizer::for_understudent(UnderstudentKind::Auto, None, None, None, H).unwrap();
    let mut p = Proxy::new(s, fz, enc, lead.map(|t| Box::new(t) as Box<dyn Lead>));
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"];

fn text(r: &mut Xoshiro, marker: &str) -> String {
    let mut w: Vec<&str> = (0..2).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
    for _ in 0..3 {
        w.insert((r.next_u64() % (w.len() as u64 + 1)) as usize, marker);
    }
    w.join(" ")
}

/// `n` lines whose label follows the text's marker (`separable`) or is a coin flip.
fn stream(n: usize, seed: u64, separable: bool) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.3;
            let flip = r.uniform() < 0.3;
            let t = text(&mut r, if bad { "error" } else { "ok" });
            let yes = if separable { bad } else { flip };
            let py = if yes { 0.95 } else { 0.05 };
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
    let rows = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| {
            let mut v: Value = serde_json::from_str(l).unwrap();
            v.as_object_mut().unwrap().remove("understudent_us");
            v
        })
        .collect();
    (sum, rows)
}

fn cand(graduated: bool, coverage: f64, cost_us: f64) -> Cand {
    Cand { graduated, coverage, cost_us: Some(cost_us) }
}

#[test]
fn a_cheaper_candidate_wins_a_near_tie() {
    // n = 400: SE of coverage 0.80 is 0.02, so 0.79 ties and 0.70 does not
    let c = [cand(true, 0.79, 20.0), cand(true, 0.80, 300.0), cand(true, 0.80, 40_000.0), cand(true, 0.80, 400_000.0)];
    assert_eq!(select_serving(&c, 400), Some(0), "keywords is cheapest and within one SE");
    let c = [cand(true, 0.70, 20.0), cand(true, 0.80, 300.0), cand(true, 0.80, 40_000.0), cand(true, 0.80, 400_000.0)];
    assert_eq!(select_serving(&c, 400), Some(1), "keywords is more than one SE short: lite is the cheapest of the tied");
    // equal costs fall back to the list order
    let c = [cand(true, 0.5, 100.0), cand(true, 0.8, 100.0), cand(true, 0.8, 100.0)];
    assert_eq!(select_serving(&c, 400), Some(1));
    // unknown costs fall back to the list order too
    let c = [Cand { graduated: true, coverage: 0.8, cost_us: None }, Cand { graduated: true, coverage: 0.8, cost_us: Some(1.0) }];
    assert_eq!(select_serving(&c, 400), Some(0));
}

#[test]
fn a_higher_coverage_beyond_one_se_wins_and_only_graduates_race() {
    let c = [cand(true, 0.60, 20.0), cand(true, 0.62, 300.0), cand(true, 0.90, 40_000.0), cand(false, 0.99, 400_000.0)];
    assert_eq!(select_serving(&c, 400), Some(2));
    let c = [cand(false, 0.9, 1.0), cand(false, 0.9, 1.0)];
    assert_eq!(select_serving(&c, 400), None);
}

#[test]
fn it_races_every_candidate_at_a_refit_and_serves_one() {
    let lines = stream(260, 3, true);
    let (mut p, si) = auto(Arc::new(Spy::default()), Some(replay_lead(&lines)), 0.1, |s| s.cost_us = vec![20.0, 300.0, 40_000.0, 400_000.0]);
    let (sum, rows) = run(&mut p, &lines);
    let race = p.understudents[si].race.as_array().unwrap();
    assert_eq!(race.len(), 4);
    for (r, k) in race.iter().zip(["keywords", "lite", "probe", "deep"]) {
        assert_eq!(r["understudent"], k);
        assert!(r["cv_agreement"].is_number() && r["cv_coverage"].is_number() && r["cost_us"].is_number(), "{r}");
    }
    assert!(race.iter().all(|r| !r["mode"].is_null()), "all graduate on a separable stream: {race:?}");
    // equal coverage: the cheapest (keywords) serves
    assert_eq!(p.understudents[si].serving_name(), Some("keywords"));
    assert_eq!(sum["serving"], "keywords");
    assert_eq!(sum["understudent"], "auto");
    let hist = sum["serving_history"].as_array().unwrap();
    assert_eq!(hist.last().unwrap()["understudent"], "keywords");
    assert!(rows.iter().any(|r| r["readout"] == "keywords"));
    let refit = sum["refits"].as_array().unwrap().last().unwrap();
    assert_eq!(refit["race"].as_array().unwrap().len(), 4);
    assert_eq!(p.understudents[si].status()["serving"], "keywords");
}

#[test]
fn a_switch_resets_the_audit_window() {
    let lines = stream(260, 4, true);
    let (mut p, si) = auto(Arc::new(Spy::default()), Some(replay_lead(&lines)), 1.0, |s| s.cost_us = vec![20.0, 300.0, 40_000.0, 400_000.0]);
    run(&mut p, &lines);
    let st = &mut p.understudents[si];
    assert_eq!(st.serving_name(), Some("keywords"));
    assert!(!st.tiers[0].serving.audits.is_empty(), "the serving understudent was audited");
    assert!(st.tiers[1].serving.audits.is_empty(), "a candidate that doesn't serve isn't audited");
    st.switch_serving(Some(1));
    assert_eq!(st.serving_name(), Some("lite"));
    assert!(st.tiers.iter().all(|t| t.serving.audits.is_empty()));
    let mode = st.tiers[1].serving.mode().unwrap();
    st.tiers[1].serving.audits.push_back(understudy::understudent::Audit { agreed: true, missed: false, mode });
    st.switch_serving(Some(1));
    assert_eq!(st.tiers[1].serving.audits.len(), 1, "no switch, no reset");
    st.switch_serving(None);
    assert!(st.tiers[1].serving.audits.is_empty());
}

#[test]
fn local_answers_compute_only_the_serving_understudents_features() {
    let spy = Arc::new(Spy::default());
    let all = stream(300, 5, true);
    let lines = &all[..260];
    let (mut p, si) = auto(spy.clone(), Some(replay_lead(&all)), 0.0, |s| s.cost_us = vec![20.0, 300.0, 40_000.0, 400_000.0]);
    run(&mut p, lines);
    assert_eq!(p.understudents[si].serving_name(), Some("keywords"));
    let before = (spy.calls.load(Ordering::SeqCst), spy.input_calls.load(Ordering::SeqCst));
    let mut local = 0;
    for l in &all[260..290] {
        let o = p.answer(l["id"].as_str(), &l["state"], None).unwrap();
        local += usize::from(o.answered_by == "understudent");
    }
    assert!(local >= 20, "keywords answered {local} of 30");
    // forwarded items run every candidate's features, local ones none: the encoder only ran for the forwarded
    let after = (spy.calls.load(Ordering::SeqCst), spy.input_calls.load(Ordering::SeqCst));
    let forwarded = 30 - local;
    assert!(after.0 - before.0 <= forwarded * 2 * 3 && (forwarded > 0 || after == before), "{before:?} -> {after:?}, {forwarded} forwarded");
    // serve deep instead: its layers are computed for local answers, probe/lite are not asked
    p.understudents[si].switch_serving(Some(3));
    let (c0, i0) = (spy.calls.load(Ordering::SeqCst), spy.input_calls.load(Ordering::SeqCst));
    let o = p.answer(all[290]["id"].as_str(), &all[290]["state"], None).unwrap();
    if o.answered_by == "understudent" {
        assert!(spy.calls.load(Ordering::SeqCst) > c0, "deep ran the model");
        assert_eq!(spy.input_calls.load(Ordering::SeqCst), i0, "lite's input embeddings were not computed");
    }
}

#[test]
fn random_labels_end_lead_only_and_stop_the_refits() {
    // states repeat, and a repeated state is one example: ~85% of the lines are new
    let n = 3000;
    let lines = stream(n, 6, false);
    let spy = Arc::new(Spy::default());
    let (mut p, si) = auto(spy.clone(), Some(replay_lead(&lines)), 0.1, |_| {});
    let (sum, rows) = run(&mut p, &lines[..900]);
    let why = p.understudents[si].lead_only.clone().unwrap_or_else(|| panic!("lead-only after 600 random examples: {} {}", p.understudents[si].race, p.understudents[si].examples.len()));
    assert!(why.contains("target"), "{why}");
    assert_eq!(sum["lead_only"].as_str(), Some(why.as_str()));
    assert_eq!(p.understudents[si].state(), "lead_only");
    assert_eq!(p.understudents[si].status()["lead_only"].as_str(), Some(why.as_str()));
    assert!(rows.iter().all(|r| r["answered_by"] == "lead"));
    let refits = |p: &Proxy| p.understudents[si].race.clone();
    let race = refits(&p);
    // further items: no features computed (the encoder stays idle), no refits until the re-check
    let (c0, i0) = (spy.calls.load(Ordering::SeqCst), spy.input_calls.load(Ordering::SeqCst));
    let (sum2, _) = run(&mut p, &lines[900..1300]);
    assert_eq!(sum2["refits"].as_array().map_or(0, Vec::len), 0, "refits stopped");
    assert_eq!((spy.calls.load(Ordering::SeqCst), spy.input_calls.load(Ordering::SeqCst)), (c0, i0), "no features while lead-only");
    assert_eq!(refits(&p), race);
    // after RECHECK_EVERY new examples one re-check refits every candidate (and still gives up)
    let (sum3, _) = run(&mut p, &lines[1300..]);
    let rechecks = sum3["refits"].as_array().unwrap().len();
    assert!((1..=2).contains(&rechecks), "{rechecks} re-checks in ~{} new examples", RECHECK_EVERY * 3 / 2);
    assert!(p.understudents[si].lead_only.is_some());
}

#[test]
fn auto_replays_are_deterministic() {
    let lines = stream(300, 8, true);
    let once = || {
        let (mut p, _) = auto(Arc::new(Spy::default()), Some(replay_lead(&lines)), 0.1, |s| s.cost_us = vec![20.0, 300.0, 40_000.0, 400_000.0]);
        let (mut sum, rows) = run(&mut p, &lines);
        for k in ["secs", "wall_secs", "mean_understudent_us", "refit_secs", "refit_secs_by_tier", "fit_secs"] {
            sum.as_object_mut().unwrap().remove(k);
        }
        sum["refits"].as_array_mut().map(|a| a.iter_mut().for_each(|r| { r.as_object_mut().unwrap().remove("secs"); r.as_object_mut().unwrap().remove("fit_secs"); r.as_object_mut().unwrap().remove("tiers"); }));
        (sum.to_string(), rows)
    };
    let (a, b) = (once(), once());
    assert_eq!(a.1, b.1);
}

#[test]
fn prefer_fast_serves_the_cheapest_candidate_within_eps_of_the_best_coverage() {
    // coverage within 0.05 of the best (0.80) and graduated: the cheapest of them, regardless of the standard error
    let c = [cand(true, 0.70, 20.0), cand(true, 0.76, 300.0), cand(true, 0.80, 40_000.0), cand(true, 0.80, 400_000.0)];
    assert_eq!(select_fastest(&c, 0.05), Some(1));
    assert_eq!(select_fastest(&c, 0.02), Some(2), "a tighter margin leaves only the expensive ones");
    assert_eq!(select_fastest(&c, 0.0), Some(2), "eps 0: the best coverage, the cheaper of the equals");
    assert_eq!(select_fastest(&c, 0.5), Some(0));
    // a candidate that did not graduate never serves, however cheap or covering
    let c = [cand(false, 0.99, 1.0), cand(true, 0.50, 300.0), cand(true, 0.52, 100.0)];
    assert_eq!(select_fastest(&c, 0.05), Some(2));
    assert_eq!(select_fastest(&[cand(false, 0.9, 1.0)], 0.05), None);
    // equal costs: the earlier candidate
    assert_eq!(select_fastest(&[cand(true, 0.8, 5.0), cand(true, 0.8, 5.0)], 0.05), Some(0));
}

#[test]
fn the_cost_model_depends_on_the_design_layers_and_tokens_only() {
    use understudy::understudent::UnderstudentKind::*;
    assert_eq!(model_cost_us(Keywords, 0, 1000.0), model_cost_us(Keywords, 0, 10.0), "a cheap design does not scale with tokens");
    let (a, b) = (model_cost_us(Probe, 8, 300.0), model_cost_us(Deep, 12, 300.0));
    assert!(b > a && a > 10.0 * model_cost_us(Keywords, 0, 300.0), "{a} {b}");
    assert!(model_cost_us(Probe, 8, 600.0) > model_cost_us(Probe, 8, 300.0), "more tokens cost more");
    assert_eq!(model_cost_us(Deep, 12, 300.0), model_cost_us(Deep, 12, 300.0));
}

#[test]
fn prefer_fast_records_the_modelled_costs_and_a_replay_repeats() {
    let lines = stream(260, 3, true);
    let mut costs = vec![];
    for _ in 0..2 {
        // no fixed `cost_us`: the costs come from the model, not a clock
        let (mut p, si) = auto(Arc::new(Spy::default()), Some(replay_lead(&lines)), 0.1, |s| s.prefer_fast = Some(0.05));
        let (sum, rows) = run(&mut p, &lines);
        let race: Vec<f64> = p.understudents[si].race.as_array().unwrap().iter().map(|r| r["cost_us"].as_f64().unwrap()).collect();
        assert!(race[0] < race[1] && race[1] < race[2] && race[2] < race[3], "{race:?}");
        assert_eq!(sum["serving"], "keywords", "all graduate with equal coverage: the cheapest serves");
        costs.push((race, rows));
    }
    assert_eq!(costs[0], costs[1], "the same stream gives the same costs and decisions");
}

#[test]
fn prefer_fast_cost_of_probe_depends_on_the_state_length() {
    // probe keeps no token states, so its encoding records no token count: the modelled cost must still follow the text
    let cost = |words: usize| {
        let lines: Vec<Value> = stream(260, 3, true)
            .into_iter()
            .map(|mut l| {
                let t = l["state"]["output"].as_str().unwrap().to_string();
                l["state"]["output"] = json!(vec![t; words].join(" "));
                l
            })
            .collect();
        let (mut p, si) = auto(Arc::new(Spy::default()), Some(replay_lead(&lines)), 0.1, |s| {
            s.prefer_fast = Some(0.05);
            s.fit.tiers = vec![UnderstudentKind::Keywords, UnderstudentKind::Probe];
        });
        run(&mut p, &lines);
        let race = p.understudents[si].race.as_array().unwrap();
        assert_eq!(race.len(), 2);
        race[1]["cost_us"].as_f64().unwrap()
    };
    let (short, long) = (cost(1), cost(40));
    assert!(long > short * 2.0, "short {short} long {long}");
}
