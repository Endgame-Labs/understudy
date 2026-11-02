//! Loads EmbeddingGemma 2 from the HF cache and embeds two texts. Ignored by default (needs the checkpoint):
//! `cargo test --release --test real_model -- --ignored --nocapture`.

use understudy::encoder::{Eg2, Encoder, Featurizer};
use std::time::Instant;

#[test]
#[ignore]
fn embeds_two_texts() {
    let t = Instant::now();
    let enc = Eg2::load_default().expect("EmbeddingGemma 2 checkpoint");
    eprintln!("load: {:.2}s", t.elapsed().as_secs_f64());
    let fz = Featurizer::new(vec![4, 8, 12], None, None, enc.hidden()).unwrap();
    let texts = vec![
        vec!["cargo build\nerror[E0308]: mismatched types in main.rs\nFinished in 3.2s".to_string()],
        vec!["git status\nOn branch main\nnothing to commit, working tree clean".to_string()],
    ];
    let e = fz.encode(&enc, &texts).unwrap();
    assert_eq!(e.len(), 2);
    for x in &e {
        let p = &x.parts[0];
        assert_eq!(p.pool.len(), 2 * 512 * 3);
        assert!(p.tokens > 5 && p.states.len() == p.tokens * 512);
        assert!(p.pool.iter().all(|v| v.is_finite()));
    }
    let (a, b) = (fz.novelty(&e[0]), fz.novelty(&e[1]));
    let cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    eprintln!("tokens {} / {}, cosine of middle-layer means {cos:.4}", e[0].parts[0].tokens, e[1].parts[0].tokens);
    assert!(cos < 0.9999);
    // timing: one text at a time, as replay encodes
    let long = "Compiling server.go ... done\n".repeat(20);
    for (name, text) in [("short", texts[0][0].clone()), ("~200 tokens", long)] {
        let _ = fz.encode(&enc, &[vec![text.clone()]]).unwrap();
        let t = Instant::now();
        for _ in 0..10 {
            fz.encode(&enc, &[vec![text.clone()]]).unwrap();
        }
        eprintln!("{name}: {:.1} ms per text", t.elapsed().as_secs_f64() * 100.0);
    }
}

/// sha256 of a decisions file with `understudent_us` removed, each line re-serialized by serde_json.
fn canon_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for l in text.lines() {
        let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
        v.as_object_mut().unwrap().remove("understudent_us");
        h.update(serde_json::to_string(&v).unwrap().as_bytes());
        h.update(b"\n");
    }
    understudy::question::hex(&h.finalize())
}

/// `UNDERSTUDY_HASH_FILE=decisions.jsonl cargo test --release --test real_model -- --ignored hash_file --nocapture`
/// prints a decisions file's canonical hash (how `DEEP_SMOKE250` was taken from the pre-understudents binary's output).
#[test]
#[ignore]
fn hash_file() {
    if let Ok(p) = std::env::var("UNDERSTUDY_HASH_FILE") {
        eprintln!("canonical hash: {}", canon_hash(&std::fs::read_to_string(p).unwrap()));
    }
}

/// The decisions of the pre-understudents binary (default deep) on the 250-item smoke stream.
const DEEP_SMOKE250: &str = "055e5ba59cf8b225f19aca2689594e6f658f5cce57999f10494f9741e052b73c";

/// deep's replay is byte-identical to before the understudents were added. Needs the checkpoint and the smoke stream:
/// `UNDERSTUDY_SMOKE_DIR=<dir with question.json and stream.jsonl> cargo test --release --test real_model -- --ignored deep_unchanged`.
#[test]
#[ignore]
fn deep_unchanged() {
    let Ok(dir) = std::env::var("UNDERSTUDY_SMOKE_DIR") else { return };
    let dir = std::path::PathBuf::from(dir);
    let qf = understudy::question::QuestionFile::load(&dir.join("question.json")).unwrap();
    let enc: std::sync::Arc<dyn Encoder> = std::sync::Arc::new(Eg2::load_default().unwrap());
    let stream = dir.join("stream.jsonl");
    let fz = Featurizer::new(vec![4, 8, 12], None, qf.pair.clone(), enc.hidden()).unwrap();
    let lead = understudy::lead::Replay::from_stream(&stream).unwrap();
    let mut p = understudy::proxy::Proxy::new(understudy::proxy::Settings::default(), fz, enc, Some(Box::new(lead)));
    p.understudent(&qf.name, &qf.question().unwrap());
    let mut out = vec![];
    understudy::replay::run(&mut p, std::io::BufReader::new(std::fs::File::open(&stream).unwrap()), &mut out).unwrap();
    assert_eq!(canon_hash(&String::from_utf8(out).unwrap()), DEEP_SMOKE250);
}
