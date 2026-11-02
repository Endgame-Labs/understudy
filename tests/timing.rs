//! Refit cost at full scale (2,000 examples, hidden 512, three layers, ~400 tokens each) on the fake encoder.
//! Ignored by default: `cargo test --release --test timing -- --ignored --nocapture`.

use understudy::encoder::{Fake, Featurizer};
use understudy::question::QType;
use understudy::readout::ReadoutChoice;
use understudy::rng::Xoshiro;
use understudy::understudent::{Row, fit_understudent};
use std::time::Instant;

#[test]
#[ignore]
fn refit_at_2000() {
    let fz = Featurizer::new(vec![4, 8, 12], None, None, 512).unwrap();
    let enc = Fake { hidden: 512 };
    let vocab: Vec<String> = (0..300).map(|i| format!("w{i}")).collect();
    let mut r = Xoshiro::new(0);
    let n = 2000;
    let mut texts = vec![];
    let mut y = vec![];
    for _ in 0..n {
        let bad = r.uniform() < 0.3;
        let mut w: Vec<&str> = (0..400).map(|_| vocab[(r.next_u64() % 300) as usize].as_str()).collect();
        if bad {
            w[(r.next_u64() % 400) as usize] = "error";
        }
        texts.push(vec![w.join(" ")]);
        y.push(usize::from(bad));
    }
    let t = Instant::now();
    let encs = fz.encode(&enc, &texts).unwrap();
    eprintln!("fake-encoded {n} texts in {:.1}s", t.elapsed().as_secs_f64());
    let rows: Vec<Row> = encs.iter().zip(&y).map(|(e, &y)| Row { enc: e, y, w: 0.95, group: None }).collect();
    let mut s = understudy::proxy::Settings::default().fit;
    for (readout, m) in [(ReadoutChoice::Pool, n), (ReadoutChoice::Attn, n), (ReadoutChoice::Auto, n), (ReadoutChoice::Auto, 600)] {
        s.readout = readout;
        let t = Instant::now();
        let f = fit_understudent(&fz, QType::Noul, 2, &rows[..m], &s);
        eprintln!(
            "{readout:?} n={m}: {:.1}s wall; job secs {}; readout {:?} mode {:?} cv_agreement {} epochs {}",
            t.elapsed().as_secs_f64(), f.report["fit_secs"], f.readout, f.mode, f.report["cv_agreement"], f.report["attn_epochs"]
        );
    }
}

/// keywords' refit at full scale: 2,000 examples of ~40 words (the smoke streams' ~150-character tool outputs) and of
/// ~400 words (a worst case: ~2,400 characters, 7,000 hashed features each). Prints the 5-fold + final fit's wall time.
#[test]
#[ignore]
fn keywords_refit_at_2000() {
    use understudy::encoder::Encoding;
    use understudy::understudent::UnderstudentKind;
    let fz = Featurizer::new(vec![4, 8, 12], None, None, 512).unwrap();
    let vocab: Vec<String> = (0..300).map(|i| format!("w{i}")).collect();
    let n = 2000;
    for words in [40usize, 400] {
        let mut r = Xoshiro::new(0);
        let (mut encs, mut y) = (vec![], vec![]);
        for _ in 0..n {
            let bad = r.uniform() < 0.3;
            let mut w: Vec<&str> = (0..words).map(|_| vocab[(r.next_u64() % 300) as usize].as_str()).collect();
            if bad {
                w[(r.next_u64() % words as u64) as usize] = "error";
            }
            encs.push(Encoding { sparse: Some(understudy::keywords::featurize(&[w.join(" ")])), ..Default::default() });
            y.push(usize::from(bad));
        }
        let rows: Vec<Row> = encs.iter().zip(&y).map(|(e, &y)| Row { enc: e, y, w: 0.95, group: None }).collect();
        let mut s = understudy::proxy::Settings::default().fit;
        s.understudent = UnderstudentKind::Keywords;
        let t = Instant::now();
        let f = fit_understudent(&fz, QType::Noul, 2, &rows, &s);
        eprintln!(
            "keywords n={n} words={words}: {:.3}s wall; fit secs {} iters {}; mode {:?} cv_agreement {}",
            t.elapsed().as_secs_f64(), f.report["fit_secs"], f.report["keywords_iters"], f.mode, f.report["cv_agreement"]
        );
    }
}
