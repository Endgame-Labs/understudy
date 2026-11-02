//! `--graduation sequential`: alpha-spending across a question's graduation attempts. The default (pointwise) is unchanged.

use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use understudy::encoder::{Encoder, Fake, Featurizer};
use understudy::lead::Replay;
use understudy::snapshot;
use understudy::proxy::{Proxy, Settings};
use understudy::question::Question;
use understudy::readout::ReadoutChoice;
use understudy::rng::Xoshiro;
use understudy::stats::sequential_z;
use understudy::understudent::{Graduation, graduation, graduation_z};

const H: usize = 16;
const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"];

fn proxy(g: Graduation, lines: &[Value]) -> Proxy {
    let mut s = Settings::default();
    s.fit.readout = ReadoutChoice::Pool;
    s.fit.graduation = g;
    let fz = Featurizer::new(vec![1, 2, 3], None, None, H).unwrap();
    let probs: HashMap<String, Vec<f64>> = lines
        .iter()
        .map(|l| (l["id"].as_str().unwrap().to_string(), l["lead_probs"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()))
        .collect();
    let lead = Replay { probs, source: "test".into() };
    let enc: Arc<dyn Encoder> = Arc::new(Fake { hidden: H });
    let mut p = Proxy::new(s, fz, enc, Some(Box::new(lead)));
    p.understudent("q", &Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap());
    p
}

fn stream(n: usize, seed: u64) -> Vec<Value> {
    let mut r = Xoshiro::new(seed);
    (0..n)
        .map(|i| {
            let bad = r.uniform() < 0.3;
            let mut w: Vec<&str> = (0..4).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
            w.insert((r.next_u64() % 5) as usize, if bad { "error" } else { "ok" });
            let py = if bad { 0.95 } else { 0.05 };
            json!({"id": format!("i{i:04}"), "state": {"call": "run", "output": w.join(" ")}, "lead_probs": [1.0 - py, py]})
        })
        .collect()
}

fn replay(p: &mut Proxy, lines: &[Value]) -> (Vec<Value>, Value) {
    let input: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    let mut out = vec![];
    let sum = understudy::replay::run(p, input.as_bytes(), &mut out).unwrap();
    let d = String::from_utf8(out).unwrap().lines().map(|l| {
        let mut v: Value = serde_json::from_str(l).unwrap();
        v.as_object_mut().unwrap().remove("understudent_us");
        v
    }).collect();
    (d, sum)
}

#[test]
fn z_at_one_is_the_pointwise_z_for_the_scan_helper() {
    // graduation() is graduation_z at 1.96, bit for bit
    let n = 400;
    let cf: Vec<f64> = (0..n).map(|i| 0.5 + 0.5 * (i as f64 / n as f64)).collect();
    let agree: Vec<bool> = (0..n).map(|i| i % 25 != 0).collect();
    let y: Vec<usize> = (0..n).map(|i| usize::from(i % 4 == 0)).collect();
    let minor: Vec<bool> = y.iter().map(|&c| c == 1).collect();
    assert_eq!(graduation(&cf, &agree, &minor, None, &y, 0.9, 0.8), graduation_z(&cf, &agree, &minor, None, &y, 0.9, 0.8, 1.96));
    // a larger z is never more permissive
    let (c1, m1, _) = graduation_z(&cf, &agree, &minor, None, &y, 0.9, 0.8, 1.96);
    let (c2, _, _) = graduation_z(&cf, &agree, &minor, None, &y, 0.9, 0.8, 4.0);
    assert!(m1.is_some() && c2 <= c1);
    let (c3, m3, _) = graduation_z(&cf, &agree, &minor, None, &y, 0.9, 0.8, 40.0);
    assert!(m3.is_none() && c3 == 0.0);
}

#[test]
fn pointwise_is_the_default_and_stores_nothing_new() {
    let lines = stream(300, 21);
    let mut a = proxy(Graduation::default(), &lines);
    let mut b = proxy(Graduation::Pointwise, &lines);
    let (da, sa) = replay(&mut a, &lines);
    let (db, _) = replay(&mut b, &lines);
    assert_eq!(da, db);
    assert!(sa["graduated_at_item"].is_number());
    assert_eq!(a.understudents[0].attempts, 0, "pointwise counts no attempts");
    let dir = std::env::temp_dir().join(format!("understudy-test-seq-pw-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    snapshot::save(&a, &dir).unwrap();
    let text = std::fs::read_to_string(dir.join("snapshot.json")).unwrap();
    assert!(!text.contains("attempts") && !text.contains("graduation"), "a pointwise snapshot is byte-identical to before");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sequential_counts_attempts_raises_z_and_persists_the_count() {
    let lines = stream(400, 22);
    let mut p = proxy(Graduation::Sequential, &lines);
    let (_, sum) = replay(&mut p, &lines);
    let refits = sum["refits"].as_array().unwrap();
    assert!(!refits.is_empty());
    assert_eq!(refits[0]["graduation"]["attempt"], 1);
    assert_eq!(refits[0]["graduation"]["z"].as_f64().unwrap(), sequential_z(1, 1));
    assert_eq!(p.understudents[0].attempts, refits.len(), "each refit that ran the test is an attempt");
    let dir = std::env::temp_dir().join(format!("understudy-test-seq-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    snapshot::save(&p, &dir).unwrap();
    let q = snapshot::load(&dir, Arc::new(Fake { hidden: H }), None, |_| {}).unwrap();
    assert_eq!(q.understudents[0].attempts, p.understudents[0].attempts);
    assert_eq!(q.settings.fit.graduation, Graduation::Sequential);
    // a later refit (a demoted question re-entering graduation included) continues the count
    let mut q = q;
    let n = q.understudents[0].attempts;
    q.refit(0);
    assert_eq!(q.understudents[0].attempts, n + 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sequential_never_graduates_before_pointwise() {
    let lines = stream(500, 23);
    let (mut a, mut b) = (proxy(Graduation::Pointwise, &lines), proxy(Graduation::Sequential, &lines));
    let (_, sa) = replay(&mut a, &lines);
    let (_, sb) = replay(&mut b, &lines);
    let (ga, gb) = (sa["graduated_at_item"].as_u64(), sb["graduated_at_item"].as_u64());
    assert!(ga.is_some(), "an easy stream graduates pointwise");
    assert!(gb.is_none_or(|g| g >= ga.unwrap()), "{ga:?} {gb:?}");
}
