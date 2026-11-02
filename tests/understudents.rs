//! The understudents (deep, probe, keywords, lite, neighbors, ladder) under the same routing and statistics, on a fake
//! encoder: each graduates on a separable stream and never on random labels; replays are deterministic; probes
//! round-trip; probe stops at layer 8; keywords never touches the encoder; a ladder answers from its cheapest eligible
//! tier without computing the later tiers' features, and a demoted tier hands over to the next.

use anyhow::Result;
use understudy::encoder::{Encoder, Fake, Featurizer, NoEncoder, Tokens};
use understudy::snapshot;
use understudy::proxy::{Kind, Proxy, Settings};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::understudent::{Mode, UnderstudentKind};
use understudy::lead::{Replay, Lead};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const H: usize = 16;
const ALL: [UnderstudentKind; 6] =
    [UnderstudentKind::Deep, UnderstudentKind::Probe, UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Neighbors, UnderstudentKind::Ladder];

/// A fake encoder that records how it was used.
#[derive(Default)]
struct Spy {
    inner: Option<Fake>,
    calls: AtomicUsize,
    input_calls: AtomicUsize,
    layers: Mutex<Vec<Vec<usize>>>,
}

impl Spy {
    fn new() -> Arc<Spy> {
        Arc::new(Spy { inner: Some(Fake { hidden: H }), ..Default::default() })
    }
}

impl Encoder for Spy {
    fn hidden(&self) -> usize {
        H
    }
    fn fingerprint(&self) -> Value {
        self.inner.as_ref().unwrap().fingerprint()
    }
    fn encode(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<Tokens>> {
        self.calls.fetch_add(texts.len(), Ordering::SeqCst);
        self.layers.lock().unwrap().push(layers.to_vec());
        self.inner.as_ref().unwrap().encode(texts, layers)
    }
    fn encode_input(&self, texts: &[&str]) -> Result<Vec<Tokens>> {
        self.input_calls.fetch_add(texts.len(), Ordering::SeqCst);
        self.inner.as_ref().unwrap().encode_input(texts)
    }
}

fn noul() -> Question {
    Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap()
}

fn proxy(kind: UnderstudentKind, enc: Arc<dyn Encoder>, lead: Option<Replay>, audit_rate: f64) -> (Proxy, usize) {
    let tiers = if kind == UnderstudentKind::Ladder { UnderstudentKind::default_tiers() } else { vec![] };
    proxy_t(kind, &tiers, enc, lead, audit_rate)
}

fn proxy_t(kind: UnderstudentKind, tiers: &[UnderstudentKind], enc: Arc<dyn Encoder>, lead: Option<Replay>, audit_rate: f64) -> (Proxy, usize) {
    let mut s = Settings::default();
    s.fit.understudent = kind;
    s.fit.tiers = tiers.to_vec();
    s.audit_rate = audit_rate;
    let hidden = enc.hidden();
    let fz = Featurizer::for_understudent(kind, None, None, None, hidden).unwrap();
    let mut p = Proxy::new(s, fz, enc, lead.map(|t| Box::new(t) as Box<dyn Lead>));
    let si = p.understudent("q", &noul());
    (p, si)
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"];

/// Two filler words and three copies of `marker`, shuffled: every understudent can read the marker.
fn text(r: &mut Xoshiro, marker: &str) -> String {
    let mut w: Vec<&str> = (0..2).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
    for _ in 0..3 {
        w.insert((r.next_u64() % (w.len() as u64 + 1)) as usize, marker);
    }
    w.join(" ")
}

fn add(p: &mut Proxy, si: usize, state: &Value, py: f64) {
    let (key, enc) = p.encode_many(&[state]).unwrap().pop().unwrap();
    p.add_example(si, Kind::Sampled, state, usize::from(py >= 0.5), Some(vec![1.0 - py, py]), key, enc);
}

/// `n` examples where "error" texts are yes (`separable`) or labels are unrelated to the text, then a refit.
fn fit_on(p: &mut Proxy, si: usize, n: usize, separable: bool, seed: u64) {
    let mut r = Xoshiro::new(seed);
    for _ in 0..n {
        let yes = r.uniform() < 0.35;
        let marker = if separable && yes { "error" } else { "ok" };
        let t = text(&mut r, marker);
        add(p, si, &json!(t), if yes { 0.95 } else { 0.05 });
    }
    p.refit(si);
}

fn modes(p: &Proxy, si: usize) -> Vec<Option<Mode>> {
    p.understudents[si].tiers.iter().map(|t| t.serving.mode()).collect()
}

fn stream(n: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.3;
            let t = text(&mut r, if bad { "error" } else { "ok" });
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

fn encoder_for(kind: UnderstudentKind) -> Arc<dyn Encoder> {
    if kind.needs_encoder() { Spy::new() } else { Arc::new(NoEncoder) }
}

#[test]
fn every_understudent_graduates_on_separable_and_never_on_random_labels() {
    for kind in ALL {
        let (mut p, si) = proxy(kind, encoder_for(kind), None, 0.0);
        fit_on(&mut p, si, 300, true, 1);
        for m in modes(&p, si) {
            assert_eq!(m, Some(Mode::Symmetric), "{kind:?} on separable data: {}", p.understudents[si].serving().fit.report);
        }
        let (mut p, si) = proxy(kind, encoder_for(kind), None, 0.0);
        fit_on(&mut p, si, 300, false, 2);
        for m in modes(&p, si) {
            assert_eq!(m, None, "{kind:?} on random labels: {}", p.understudents[si].serving().fit.report);
        }
    }
}

#[test]
fn replays_are_deterministic_for_every_understudent() {
    let lines = stream(260, 3);
    for kind in ALL {
        let run = || {
            let (mut p, _) = proxy(kind, encoder_for(kind), Some(replay_lead(&lines)), 0.1);
            run_replay(&mut p, &lines)
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "{kind:?}");
        assert!(a.iter().any(|d| d["answered_by"] == "understudent"), "{kind:?} never answered locally");
        assert!(a.iter().any(|d| d["audited"] == true), "{kind:?} never audited");
    }
}

#[test]
fn readout_names() {
    let lines = stream(260, 4);
    for (kind, want) in [(UnderstudentKind::Probe, "mean"), (UnderstudentKind::Keywords, "keywords"), (UnderstudentKind::Lite, "lite"), (UnderstudentKind::Neighbors, "neighbors")] {
        let (mut p, _) = proxy(kind, encoder_for(kind), Some(replay_lead(&lines)), 0.1);
        let d = run_replay(&mut p, &lines);
        assert!(d.iter().filter(|d| d["answered_by"] == "understudent").all(|d| d["readout"] == want), "{kind:?}");
    }
}

#[test]
fn probe_design_stops_at_layer_8_and_reads_the_mean_only() {
    let spy = Spy::new();
    let (mut p, si) = proxy(UnderstudentKind::Probe, spy.clone(), None, 0.0);
    fit_on(&mut p, si, 120, true, 5);
    assert!(spy.layers.lock().unwrap().iter().all(|l| l == &[8]), "{:?}", spy.layers.lock().unwrap());
    assert_eq!(p.fz.pool_dim(), H);
    assert_eq!(p.understudents[si].serving().fit.pool.as_ref().unwrap().lr.d, H);
    assert!(p.understudents[si].examples.iter().all(|e| e.enc.parts[0].tokens == 0), "no token states are kept");
}

#[test]
fn static_reads_input_embeddings_only() {
    let spy = Spy::new();
    let (mut p, si) = proxy(UnderstudentKind::Lite, spy.clone(), None, 0.0);
    fit_on(&mut p, si, 120, true, 6);
    assert_eq!(spy.calls.load(Ordering::SeqCst), 0, "no transformer layer runs");
    let n = spy.input_calls.load(Ordering::SeqCst);
    assert!((100..=120).contains(&n), "one input-embedding pass per distinct state, got {n}");
    assert_eq!(p.understudents[si].serving().fit.pool.as_ref().unwrap().lr.d, 2 * H, "mean and max");
}

#[test]
fn keywords_never_loads_or_calls_the_model() {
    // NoEncoder errors on any call, so a full replay (fits, audits, novelty) proves it
    let lines = stream(300, 7);
    let (mut p, _) = proxy(UnderstudentKind::Keywords, Arc::new(NoEncoder), Some(replay_lead(&lines)), 0.1);
    let d = run_replay(&mut p, &lines);
    assert!(d.iter().any(|d| d["answered_by"] == "understudent"));
    assert!(!UnderstudentKind::Keywords.needs_encoder());
    // and the real model is only built through needs_encoder
    let spy = Spy::new();
    let (mut p, _) = proxy(UnderstudentKind::Keywords, spy.clone(), Some(replay_lead(&lines)), 0.1);
    run_replay(&mut p, &lines);
    assert_eq!(spy.calls.load(Ordering::SeqCst) + spy.input_calls.load(Ordering::SeqCst), 0);
}

fn counters(spy: &Spy) -> (usize, usize) {
    (spy.calls.load(Ordering::SeqCst), spy.input_calls.load(Ordering::SeqCst))
}

#[test]
fn ladder_answers_from_the_cheapest_eligible_tier_without_computing_later_features() {
    let spy = Spy::new();
    let (mut p, si) = proxy(UnderstudentKind::Ladder, spy.clone(), None, 0.0);
    fit_on(&mut p, si, 300, true, 21);
    assert!(modes(&p, si).iter().all(Option::is_some), "every tier graduated: {:?}", modes(&p, si));
    let mut r = Xoshiro::new(22);
    let mut by_keywords = 0;
    for i in 0..30 {
        let t = text(&mut r, if i % 2 == 0 { "error" } else { "ok" });
        let before = counters(&spy);
        let Ok(o) = p.answer(Some("x"), &json!(t), None) else { continue };
        let d = &o.decisions[0].1;
        if d.readout.as_deref() == Some("keywords") && o.answered_by == "understudent" {
            by_keywords += 1;
            assert_eq!(counters(&spy), before, "keywords answered, so neither the model's layers nor its embeddings were touched");
            assert_eq!(d.label, usize::from(i % 2 == 0));
        }
    }
    assert!(by_keywords >= 20, "keywords answered only {by_keywords} of 30");
}

#[test]
fn a_demoted_tier_hands_its_answers_to_the_next() {
    let spy = Spy::new();
    let (mut p, si) = proxy(UnderstudentKind::Ladder, spy.clone(), None, 0.0);
    fit_on(&mut p, si, 300, true, 23);
    let mut r = Xoshiro::new(24);
    let order = ["keywords", "lite", "mean", "deep:"];
    for stage in 0..4 {
        let (mut mine, mut local) = (0, 0);
        for i in 0..30 {
            let t = text(&mut r, if i % 2 == 0 { "error" } else { "ok" });
            let layers_before = spy.calls.load(Ordering::SeqCst);
            let Ok(o) = p.answer(Some("x"), &json!(t), None) else { continue };
            let d = &o.decisions[0].1;
            assert_eq!(o.answered_by, "understudent");
            let rd = d.readout.as_deref().unwrap();
            let rank = order.iter().position(|n| rd.starts_with(n)).unwrap();
            if rd == "lite" || rd == "keywords" {
                assert_eq!(spy.calls.load(Ordering::SeqCst), layers_before, "{rd} runs no transformer layer");
            }
            assert!(rank >= stage, "stage {stage}: tier {rd} is demoted yet answered");
            local += 1;
            mine += usize::from(rank == stage);
        }
        assert!(local >= 20 && mine * 2 >= local, "stage {stage}: {mine} of {local} answered by {}", order[stage]);
        if stage < 3 {
            p.understudents[si].tiers[stage].serving.demoted = true;
        }
    }
}

#[test]
fn chunked_encoding_equals_plain_for_short_texts_and_combines_long_ones() {
    use understudy::encoder::{CLIP_CHARS, MAX_CHUNKS, TOKEN_CAP, chunk_text};
    let enc = Fake { hidden: H };
    let plain = Featurizer::new(vec![1, 2, 3], None, None, H).unwrap();
    let chunked = Featurizer { chunk: true, ..plain.clone() };
    let short = vec!["cargo build\nerror[E0308]: mismatched types\nFinished in 3.2s".to_string()];
    let (ta, tb) = (plain.field_texts(&short), chunked.field_texts(&short));
    assert_eq!(ta, tb);
    assert_eq!(plain.encode_fields(&enc, &[ta]).unwrap(), chunked.encode_fields(&enc, &[tb]).unwrap());

    let line = "alpha beta gamma delta epsilon zeta eta theta\n";
    let long = line.repeat(200);
    let chunks = chunk_text(&long);
    assert!((3..=MAX_CHUNKS).contains(&chunks.len()), "{}", chunks.len());
    assert!(chunks.iter().all(|c| c.chars().count() <= CLIP_CHARS));
    assert_eq!(chunks.concat(), long, "chunks are consecutive and lossless");
    assert!(chunks[..chunks.len() - 1].iter().all(|c| c.ends_with('\n')), "cut at line breaks");
    assert_ne!(plain.field_texts(&[long.clone()]), chunked.field_texts(&[long.clone()]), "plain clips, chunked doesn't");
    // a text that would need more than MAX_CHUNKS soft chunks is still cut into at most MAX_CHUNKS
    let huge = "x".repeat(23_000);
    assert!(chunk_text(&huge).len() <= MAX_CHUNKS);

    let part = &chunked.encode_fields(&enc, &[chunked.field_texts(&[long.clone()])]).unwrap()[0][0];
    let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
    let toks = enc.encode(&refs, &[1, 2, 3]).unwrap();
    let n: usize = toks.iter().map(|t| t.tokens).sum();
    for li in 0..3 {
        for j in 0..H {
            let rows = toks.iter().flat_map(|t| t.states[li].chunks_exact(H));
            let (mut sum, mut max) = (0f64, f32::NEG_INFINITY);
            for row in rows {
                sum += row[j] as f64;
                max = max.max(row[j]);
            }
            let (mean, mx) = (part.pool[li * 2 * H + j], part.pool[li * 2 * H + H + j]);
            assert!((mean as f64 - sum / n as f64).abs() < 1e-4, "token-count-weighted mean");
            assert_eq!(mx, max, "max over chunks");
        }
    }
    assert!(part.tokens <= TOKEN_CAP && part.tokens > TOKEN_CAP * 3 / 4, "{}", part.tokens);
    assert_eq!(part.states.len(), part.tokens * H);
}

#[test]
fn pair_features_flip_a_two_option_comparison_when_a_and_b_swap() {
    for kind in [UnderstudentKind::Lite, UnderstudentKind::Keywords] {
        let enc: Arc<dyn Encoder> = if kind == UnderstudentKind::Keywords { Arc::new(NoEncoder) } else { Spy::new() };
        let fz = Featurizer::for_understudent(kind, None, None, Some(("a".into(), "b".into())), H).unwrap();
        let mut s = Settings::default();
        s.fit.understudent = kind;
        let mut p = Proxy::new(s, fz, enc, None);
        let si = p.understudent("q", &noul());
        let mut r = Xoshiro::new(31);
        let pair = |r: &mut Xoshiro| {
            let a_good = r.uniform() < 0.5;
            let (a, b) = (text(r, if a_good { "good" } else { "poor" }), text(r, if a_good { "poor" } else { "good" }));
            (a_good, a, b)
        };
        for _ in 0..400 {
            let (a_good, a, b) = pair(&mut r);
            add(&mut p, si, &json!({"a": a, "b": b}), if a_good { 0.95 } else { 0.05 });
        }
        p.refit(si);
        assert!(p.understudents[si].serving().mode().is_some(), "{kind:?} graduates on pair data");
        let p_yes = |p: &mut Proxy, state: Value| {
            let e = p.encode_many(&[&state]).unwrap().pop().unwrap().1;
            let t = &p.understudents[si].tiers[0];
            t.serving.predict(&t.fz, &e, 2).p.unwrap()[1]
        };
        for _ in 0..20 {
            let (a_good, a, b) = pair(&mut r);
            let (ab, ba) = (p_yes(&mut p, json!({"a": a, "b": b})), p_yes(&mut p, json!({"a": b, "b": a})));
            assert_eq!(ab > 0.5, a_good, "{kind:?} reads the pair: {ab}");
            assert_eq!(ba > 0.5, !a_good, "{kind:?} flips when a and b swap: {ba}");
        }
    }
}

#[test]
fn ladder_refits_cheap_tiers_more_often_and_reports_per_tier() {
    let lines = stream(320, 41);
    // every local answer is audited, so examples keep arriving after the cheap tiers graduate
    let (mut p, _) = proxy(UnderstudentKind::Ladder, Spy::new(), Some(replay_lead(&lines)), 1.0);
    let input: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    let sum = understudy::replay::run(&mut p, input.as_bytes(), &mut Vec::new()).unwrap();
    let refits = sum["refits"].as_array().unwrap();
    let kind = |k: &str| -> Vec<&Value> { refits.iter().filter(|r| r["refit"] == k).collect() };
    let (cheap, all) = (kind("cheap"), kind("all"));
    assert!(!cheap.is_empty() && !all.is_empty(), "{refits:?}");
    assert!(cheap.len() > all.len());
    for r in &cheap {
        let tiers: Vec<&String> = r["tiers"].as_object().unwrap().keys().collect();
        assert_eq!(tiers, ["keywords", "lite"], "a cheap refit fits keywords and lite only");
    }
    assert!(all.iter().all(|r| r["tiers"].as_object().unwrap().len() == 4));
    for k in ["coverage_by_tier", "graduated_at_item_by_tier", "refit_secs_by_tier"] {
        assert_eq!(sum[k].as_object().unwrap().len(), 4, "{k}");
    }
    assert_eq!(sum["understudent"], "ladder");
}

#[test]
fn probe_then_deep_encode_like_each_alone() {
    // a short state shares one forward pass (probe's, continued by deep); a long one can't (probe clips, deep chunks)
    for state in [json!("alpha beta error gamma"), json!("alpha beta gamma delta epsilon zeta eta theta\n".repeat(150))] {
        let (mut p, _) = proxy(UnderstudentKind::Ladder, Spy::new(), None, 0.0);
        let (mean_i, main_i) = (2, 3);
        let mut w = p.work(&state);
        p.ensure(&mut w, &[mean_i]).unwrap();
        assert!(!w.enc.mean.is_empty() && w.enc.parts.is_empty(), "deep's tier waits until it is needed");
        p.ensure(&mut w, &[main_i]).unwrap();
        let (fm, fr) = (p.tier_fz[mean_i].1.clone(), p.tier_fz[main_i].1.clone());
        let full = p.fz.full_texts(&state);
        let enc = Fake { hidden: H };
        assert_eq!(w.enc.mean, fm.encode_fields(&enc, &[fm.field_texts(&full)]).unwrap().pop().unwrap());
        assert_eq!(w.enc.parts, fr.encode_fields(&enc, &[fr.field_texts(&full)]).unwrap().pop().unwrap());
    }
}

#[test]
fn snapshots_round_trip_for_every_understudent() {
    for kind in ALL {
        let lines = stream(260, 13);
        let (mut p, si) = proxy(kind, encoder_for(kind), Some(replay_lead(&lines)), if kind == UnderstudentKind::Ladder { 1.0 } else { 0.1 });
        run_replay(&mut p, &lines);
        assert!(p.understudents[si].tiers.iter().all(|t| t.serving.fit.has_model()), "{kind:?}");
        let dir = std::env::temp_dir().join(format!("understudy-understudents-{kind:?}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        snapshot::save(&p, &dir).unwrap();
        let q = snapshot::load(&dir, encoder_for(kind), None, |_| {}).unwrap();
        let (a, b) = (&p.understudents[si], &q.understudents[si]);
        assert_eq!(*a.serving().fit, *b.serving().fit, "{kind:?}");
        assert_eq!(a.kind, b.kind);
        for (ta, tb) in a.tiers.iter().zip(&b.tiers) {
            assert_eq!(*ta.serving.fit, *tb.serving.fit, "{kind:?} tier {:?}", ta.kind);
            assert_eq!(ta.serving.audits, tb.serving.audits, "{kind:?} tier {:?}", ta.kind);
            assert_eq!(ta.serving.demoted, tb.serving.demoted);
        }
        assert!(a.examples.iter().zip(&b.examples).all(|(x, y)| *x.enc == *y.enc), "{kind:?}");
        let mut q = q;
        for l in stream(15, 14) {
            let (_, ea) = p.encode_many(&[&l["state"]]).unwrap().pop().unwrap();
            let (_, eb) = q.encode_many(&[&l["state"]]).unwrap().pop().unwrap();
            let (sa, sb) = (&p.understudents[si], &q.understudents[si]);
            for (ta, tb) in sa.tiers.iter().zip(&sb.tiers) {
                assert_eq!(ta.serving.predict(&ta.fz, &ea, 2).p, tb.serving.predict(&tb.fz, &eb, 2).p, "{kind:?} tier {:?}", ta.kind);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
