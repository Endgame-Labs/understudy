//! Synthetic harness for `--graduation sequential`: a question whose true agreement with the lead is known is tested at
//! every refit (every 100 new examples, a rotating window of 2,000) by the real graduation scan, `pointwise` (z = 1.96)
//! against `sequential` (alpha-spending over attempts, Bonferroni over candidates). Prints a Markdown table.
//!
//!     cargo run --release --example sequential_sim [trials]
//!
//! Rows: 30% minority class; each row agrees with the lead with probability `p` independently of its confidence.
//! `flat`: every row has confidence 0.9 (one threshold). `scan`: confidence uniform in [0.5, 1) (the 0.50..0.99 threshold scan
//! runs too; alpha-spending does not cover that within-attempt scan). `x4`: four independent candidates race, the question
//! graduates if any does (a ladder or auto), each tested at the Bonferroni-corrected z.
use understudy::rng::Xoshiro;
use understudy::stats::{sequential_z, wilson_z};
use understudy::understudent::graduation_z;

const TARGET: f64 = 0.97;
const MINORITY: f64 = 0.80;
const REFITS: usize = 30;
const WINDOW: usize = 2000;

struct Row {
    cf: f64,
    agree: bool,
    y: usize,
}

fn unif(r: &mut Xoshiro) -> f64 {
    (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

fn row(r: &mut Xoshiro, p: f64, scan: bool) -> Row {
    let y = usize::from(unif(r) < 0.3);
    let agree = unif(r) < p;
    let cf = if scan { 0.5 + 0.5 * unif(r) } else { 0.9 };
    Row { cf, agree, y }
}

/// One candidate's stream tested at every refit; returns per refit (graduated, coverage).
fn run(r: &mut Xoshiro, p: f64, scan: bool, zs: &[f64]) -> Vec<(bool, f64)> {
    let mut rows: Vec<Row> = vec![];
    let mut out = vec![];
    for k in 1..=REFITS {
        for _ in 0..100 {
            rows.push(row(r, p, scan));
        }
        let w = &rows[rows.len().saturating_sub(WINDOW)..];
        let cf: Vec<f64> = w.iter().map(|x| x.cf).collect();
        let agree: Vec<bool> = w.iter().map(|x| x.agree).collect();
        let y: Vec<usize> = w.iter().map(|x| x.y).collect();
        let minor: Vec<bool> = y.iter().map(|&c| c == 1).collect();
        let (cov, mode, _) = graduation_z(&cf, &agree, &minor, None, &y, TARGET, MINORITY, zs[k - 1]);
        out.push((mode.is_some(), cov));
    }
    out
}

fn main() {
    let trials: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    println!("target {TARGET}, minority bar {MINORITY}, {REFITS} refits (100 examples apart, window {WINDOW}), {trials} trials per cell");
    println!("z by attempt (1 candidate): sequential {:.2} at 1, {:.2} at 5, {:.2} at 10, {:.2} at 30; pointwise 1.96", sequential_z(1, 1), sequential_z(5, 1), sequential_z(10, 1), sequential_z(30, 1));
    println!("(wilson lower of 0.99 over 1000 rows: z=1.96 {:.4}, z=4.14 {:.4})\n", wilson_z(990, 1000, 1.96).0, wilson_z(990, 1000, 4.14).0);
    println!("| true agreement | confidence | candidates | policy | P(ever graduates) | mean first graduation (refit) | share of refits graduated | coverage at refit 30 |");
    println!("|---|---|---|---|---|---|---|---|");
    let cells: Vec<(f64, bool, usize)> = vec![
        (0.95, false, 1), (0.955, false, 1), (0.96, false, 1), (0.965, false, 1),
        (0.96, true, 1), (0.965, true, 1), (0.96, false, 4), (0.965, false, 4),
        (0.99, false, 1), (0.99, true, 1), (0.99, false, 4), (0.995, false, 1),
    ];
    for (p, scan, m) in cells {
        for seq in [false, true] {
            let zs: Vec<f64> = (1..=REFITS).map(|k| if seq { sequential_z(k, m) } else { 1.96 }).collect();
            let mut r = Xoshiro::new(0x5eed ^ (p * 1e6) as u64 ^ (m as u64) << 40 ^ u64::from(scan) << 50);
            let (mut ever, mut first_sum, mut first_n, mut shr, mut cov30) = (0usize, 0.0, 0usize, 0.0, 0.0);
            for _ in 0..trials {
                // the candidates' streams are independent; the question is graduated at refit k if any candidate is
                let runs: Vec<Vec<(bool, f64)>> = (0..m).map(|_| run(&mut r, p, scan, &zs)).collect();
                let at = |k: usize| runs.iter().map(|c| c[k]).filter(|x| x.0).fold(None, |a: Option<f64>, x| Some(a.map_or(x.1, |a| a.max(x.1))));
                let first = (0..REFITS).find(|&k| at(k).is_some());
                if let Some(f) = first {
                    ever += 1;
                    first_sum += (f + 1) as f64;
                    first_n += 1;
                }
                shr += (0..REFITS).filter(|&k| at(k).is_some()).count() as f64 / REFITS as f64;
                cov30 += at(REFITS - 1).unwrap_or(0.0);
            }
            let t = trials as f64;
            println!(
                "| {p} | {} | {m} | {} | {:.1}% | {} | {:.1}% | {:.1}% |",
                if scan { "scan" } else { "flat" },
                if seq { "sequential" } else { "pointwise" },
                100.0 * ever as f64 / t,
                if first_n > 0 { format!("{:.1}", first_sum / first_n as f64) } else { "-".into() },
                100.0 * shr / t,
                100.0 * cov30 / t
            );
        }
    }
}
