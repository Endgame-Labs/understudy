//! `train --max-examples`: the CLI keeps up to that many sampled examples (default 2000, oldest evicted).

fn train_n(extra: &[&str]) -> u64 {
    let dir = std::env::temp_dir().join(format!("understudy-train-cap-{}-{}", std::process::id(), extra.len()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    let q = r#"{"name":"cap","category":"real_judging","type":"noul","options":["false","true"],"question":{"type":"noul","instructions":"Is it best?","criteria":{"true":"y","false":"n"}},"pair":["instruction","response"],"notes":""}"#;
    std::fs::write(dir.join("question.json"), q).unwrap();
    let mut ex = String::new();
    for i in 0..2300 {
        let (a, b) = (i * 7919 % 10007, i * 104729 % 10009);
        ex += &format!("{{\"id\":\"it{i:05}#1\",\"state\":{{\"instruction\":\"task w{a} x{b}\",\"response\":\"resp y{b} z{a}\"}},\"lead_probs\":[{},{}]}}\n", 0.8 - 0.6 * (i % 2) as f64, 0.2 + 0.6 * (i % 2) as f64);
    }
    std::fs::write(dir.join("ex.jsonl"), ex).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_understudy"))
        .args(["train", "--understudent", "keywords", "--snapshot"]).arg(dir.join("snapshot")).arg("--question").arg(dir.join("question.json"))
        .arg("--examples").arg(dir.join("ex.jsonl")).args(extra).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    v["n"].as_u64().unwrap()
}

#[test]
fn max_examples_raises_the_sampled_cap_and_the_default_stays_2000() {
    assert_eq!(train_n(&[]), 2000, "default: the oldest 300 are evicted");
    assert_eq!(train_n(&["--max-examples", "2300"]), 2300, "raised cap: every example is fit");
}
