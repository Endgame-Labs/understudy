//! `serve`'s default `auto` never needs the embedding model to start: it races the model-free designs, and the encoder
//! designs join the race once the model is ready (fetched in the background, unless disabled).

use anyhow::{Result, anyhow};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use understudy::encoder::{Encoder, Fake, Featurizer, NoEncoder};
use understudy::lazyenc::LazyEncoder;
use understudy::lead::{Lead, Replay};
use understudy::proxy::{Proxy, Settings};
use understudy::question::Question;
use understudy::rng::Xoshiro;
use understudy::understudent::UnderstudentKind as K;

const H: usize = 16;

fn fake() -> Arc<dyn Encoder> {
    Arc::new(Fake { hidden: H })
}

fn lines(n: usize, seed: u64, from: usize) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (from..from + n)
        .map(|i| {
            let bad = r.uniform() < 0.3;
            let word = ["alpha", "beta", "gamma", "delta"][(r.next_u64() % 4) as usize];
            let text = format!("{word} {} {} run", if bad { "error" } else { "ok" }, if bad { "error" } else { "ok" });
            json!({"id": format!("i{i:04}"), "state": {"call": "run", "output": text}, "lead_probs": [if bad { 0.05 } else { 0.95 }, if bad { 0.95 } else { 0.05 }]})
        })
        .collect()
}

fn replay(p: &mut Proxy, ls: &[Value]) {
    let input: String = ls.iter().map(|l| l.to_string() + "\n").collect();
    let mut out = vec![];
    understudy::replay::run(p, input.as_bytes(), &mut out).unwrap();
}

#[test]
fn an_auto_race_without_the_model_gains_the_encoder_designs_later() {
    let all: Vec<Value> = lines(500, 3, 0);
    let probs: HashMap<String, Vec<f64>> = all.iter().map(|l| (l["id"].as_str().unwrap().into(), l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())).collect();
    let lead = Replay { probs, source: "test".into() };
    let mut s = Settings::default();
    s.fit.understudent = K::Auto;
    s.fit.tiers = vec![K::Keywords, K::Shape, K::Checklist, K::Memo];
    let fz = Featurizer::for_understudent(K::Auto, None, None, None, NoEncoder.hidden()).unwrap();
    let mut p = Proxy::new(s, fz, Arc::new(NoEncoder), Some(Box::new(lead) as Box<dyn Lead>));
    let q = Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap();
    let si = p.understudent("q", &q);
    replay(&mut p, &all[..250]);
    assert_eq!(p.understudents[si].race.as_array().unwrap().len(), 4, "only the model-free designs race");
    // the model arrives: lite, probe and deep are added after the existing candidates; nothing learned is lost
    let before = p.understudents[si].examples.len();
    assert!(p.add_encoder_tiers(fake(), &[K::Lite, K::Probe, K::Deep]).unwrap());
    assert!(!p.add_encoder_tiers(fake(), &[K::Lite, K::Probe, K::Deep]).unwrap(), "a second call changes nothing");
    assert_eq!(p.understudents[si].examples.len(), before);
    assert_eq!(p.understudents[si].tiers.iter().map(|t| t.kind).collect::<Vec<_>>(), vec![K::Keywords, K::Shape, K::Checklist, K::Memo, K::Lite, K::Probe, K::Deep]);
    replay(&mut p, &all[250..]);
    assert_eq!(p.understudents[si].race.as_array().unwrap().len(), 7, "the encoder designs are in the next race");
}

#[test]
fn the_model_is_fetched_in_the_background_and_a_failure_never_stops_serving() {
    let loads = Arc::new(AtomicUsize::new(0));
    let fetched = Arc::new(AtomicUsize::new(0));
    let (l, f) = (loads.clone(), fetched.clone());
    // missing at first, there after the fetch
    let lazy = LazyEncoder::with(
        true,
        Box::new(move || if l.fetch_add(1, Ordering::SeqCst) == 0 { Err(anyhow!("no EmbeddingGemma 2 model found")) } else { Ok(fake()) }),
        Box::new(move || { f.fetch_add(1, Ordering::SeqCst); std::thread::sleep(std::time::Duration::from_millis(100)); Ok(()) }),
    );
    lazy.start();
    assert!(lazy.ready().is_none(), "start returns at once while the fetch runs");
    assert!(lazy.wait().is_ok());
    assert_eq!((fetched.load(Ordering::SeqCst), lazy.ready().is_some()), (1, true));
    // a failed fetch: no model, a clear error, no panic
    let lazy = LazyEncoder::with(true, Box::new(|| Err(anyhow!("no EmbeddingGemma 2 model found"))), Box::new(|| Err(anyhow!("offline"))));
    lazy.start();
    let e = lazy.wait().err().expect("no model").to_string();
    assert!(e.contains("understudy model fetch") && e.contains("offline"), "{e}");
    assert!(lazy.ready().is_none());
}

#[test]
fn no_model_download_never_fetches() {
    let fetched = Arc::new(AtomicUsize::new(0));
    let f = fetched.clone();
    let lazy = LazyEncoder::with(false, Box::new(|| Err(anyhow!("missing"))), Box::new(move || { f.fetch_add(1, Ordering::SeqCst); Ok(()) }));
    lazy.start();
    assert!(lazy.wait().is_err());
    assert_eq!(fetched.load(Ordering::SeqCst), 0);
    let _: Result<()> = Ok(());
}

#[test]
fn examples_learned_before_the_model_arrived_train_the_encoder_designs_at_the_next_refit() {
    let mut all: Vec<Value> = lines(400, 5, 0);
    // distinct states (identical ones collapse into one example)
    for (i, l) in all.iter_mut().enumerate() {
        let t = format!("{} w{} x{}", l["state"]["output"].as_str().unwrap(), i, i * 7 % 13);
        l["state"]["output"] = json!(t);
    }
    let probs: HashMap<String, Vec<f64>> = all.iter().map(|l| (l["id"].as_str().unwrap().into(), l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())).collect();
    let lead = Replay { probs, source: "test".into() };
    let mut s = Settings::default();
    s.fit.understudent = K::Auto;
    s.fit.tiers = vec![K::Keywords, K::Shape, K::Checklist, K::Memo];
    let fz = Featurizer::for_understudent(K::Auto, None, None, None, NoEncoder.hidden()).unwrap();
    let mut p = Proxy::new(s, fz, Arc::new(NoEncoder), Some(Box::new(lead) as Box<dyn Lead>));
    let q = Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap();
    let si = p.understudent("q", &q);
    for l in &all[..250] {
        p.answer(l["id"].as_str(), &l["state"], None).unwrap();
    }
    let before = p.understudents[si].examples.len();
    assert!(before >= 60, "{before}");
    assert!(p.understudents[si].tiers.iter().any(|t| t.serving.mode().is_some()), "a model-free design graduated");
    assert!(p.add_encoder_tiers(fake(), &[K::Lite, K::Probe, K::Deep]).unwrap());
    // the next refit completes every earlier example with the encoder features and fits every design on all of them
    p.refit(si);
    let u = &p.understudents[si];
    assert_eq!(u.examples.len(), before);
    assert!(u.examples.iter().all(|e| !e.enc.stat.is_empty() && !e.enc.mean.is_empty() && !e.enc.parts.is_empty()), "an earlier example was left without the encoder features");
    assert!(u.tiers.iter().all(|t| t.serving.fit.report.get("cv_agreement").is_some_and(|v| !v.is_null())), "every design fitted");
    assert!(u.tiers.iter().any(|t| t.serving.mode().is_some()), "graduation survives the upgrade");
}
