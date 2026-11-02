//! A question's understudent: fitting (5-fold OOF, readout choice, graduation, novelty floor) and serving
//! (prediction, eligibility, audits, demotion), following hunch's `fit_understudent` and `Understudent`.

use crate::abstain::{self, Abstain};
use crate::checklist::{self, Checklist};
use crate::encoder::{Encoding, Featurizer};
use crate::shape::{self, BlendStd};
use crate::linear;
use crate::memo::{self, MemoModel};
use crate::netdesign::{self, NetModel};
use crate::keywords::{self, Neighbors, SparseLr};
use crate::question::QType;
use crate::rules::RuleSet;
use crate::readout::{AttnModel, AttnParams, PoolModel, Readout, ReadoutChoice, Standardizer, token_standardizer};
use crate::abstain::BAR_Z;
use crate::stats::{argmax, group_kfold, kfold, quantile, wilson, wilson_z};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

/// A one-sided question's bar: the share of local "no" answers the lead calls "yes".
pub const MISS_RATE: f64 = 0.02;
/// One-sided graduation: lower bound of "yes" recall among the forwarded (`one_sided_yes`: of "no" recall).
pub const ONE_SIDED_RECALL: f64 = 0.85;
pub const AUDIT_WINDOW: usize = 100;
pub const AUDIT_MIN: usize = 20;
pub const FOLDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Symmetric,
    OneSided,
    /// Mirror of `OneSided` (noul only): answer locally only a confident "yes" (P(yes) above the threshold), forward the rest.
    OneSidedYes,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Symmetric => "symmetric",
            Mode::OneSided => "one_sided",
            Mode::OneSidedYes => "one_sided_yes",
        }
    }
}

/// Which understudent answers a question. They share all the routing and statistics (graduation, audits, novelty,
/// demotion) and differ in features and model; see SPEC.md "Understudents".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum UnderstudentKind {
    /// Mean+max pooling or attention over layers 4/8/12 (readout auto).
    #[default]
    #[serde(rename = "deep")]
    #[value(name = "deep")]
    Deep,
    /// hunch's understudent: layer-8 token mean, standardized, logistic regression.
    #[serde(rename = "probe")]
    #[value(name = "probe")]
    Probe,
    /// Hashed character and word n-grams, sparse logistic regression; no neural network.
    #[serde(rename = "keywords")]
    #[value(name = "keywords")]
    Keywords,
    /// Input token embeddings (no layer): mean and max, standardized, logistic regression.
    #[serde(rename = "lite")]
    #[value(name = "lite")]
    Lite,
    /// Cosine 7-nearest-neighbour vote over the layer-8 mean.
    #[serde(rename = "neighbors")]
    #[value(name = "neighbors")]
    Neighbors,
    /// Decision rules an LLM wrote as regexes, validated on held-out examples (`rules.rs`); no model, no encoder.
    #[serde(rename = "rules")]
    #[value(name = "rules")]
    Rules,
    /// Logistic regression over cheap structural features of the state (lengths, shares, markdown, markers, JSON types,
    /// echoed numbers); no encoder.
    #[serde(rename = "shape")]
    #[value(name = "shape")]
    Shape,
    /// One sparse+dense logistic regression over the hashed n-grams, the structural features and the input embeddings'
    /// mean and max (only the embedding table is read).
    #[serde(rename = "blend")]
    #[value(name = "blend")]
    Blend,
    /// Retrieval over past lead answers: answers only an exact or near-duplicate (char 5-gram MinHash) state; else
    /// abstains. No model, no encoder.
    #[serde(rename = "memo")]
    #[value(name = "memo")]
    Memo,
    /// A small trained network: one hidden layer over [deep's pooled encoder features] ++ [`shape`'s features]
    /// (`netdesign.rs`); the encoder runs exactly as for deep.
    #[serde(rename = "net")]
    #[value(name = "net")]
    Net,
    /// An automatically learned decision list over sparse binary features (hashed n-grams and `shape`'s indicators);
    /// answers only when a rule fires (`checklist.rs`). No encoder.
    #[serde(rename = "checklist")]
    #[value(name = "checklist")]
    Checklist,
    /// Tiers in increasing cost (`FitSettings::tiers`, default `keywords,lite,probe,deep`), each with its own fit,
    /// graduation, audits and demotion; the cheapest eligible tier answers.
    #[serde(rename = "ladder")]
    #[value(name = "ladder")]
    Ladder,
    /// Race the candidates (`FitSettings::tiers`, default `keywords,lite,probe,deep`) at every refit and serve ONE:
    /// the best graduate by coverage, ties to the cheaper. See SPEC.md "Understudents".
    #[serde(rename = "auto")]
    #[value(name = "auto")]
    Auto,
}

impl UnderstudentKind {
    pub fn name(self) -> &'static str {
        match self {
            UnderstudentKind::Deep => "deep",
            UnderstudentKind::Probe => "probe",
            UnderstudentKind::Keywords => "keywords",
            UnderstudentKind::Lite => "lite",
            UnderstudentKind::Neighbors => "neighbors",
            UnderstudentKind::Rules => "rules",
            UnderstudentKind::Shape => "shape",
            UnderstudentKind::Blend => "blend",
            UnderstudentKind::Memo => "memo",
            UnderstudentKind::Net => "net",
            UnderstudentKind::Checklist => "checklist",
            UnderstudentKind::Ladder => "ladder",
            UnderstudentKind::Auto => "auto",
        }
    }

    /// Whether the question holds several fitted tiers (a ladder's tiers, or auto's candidates).
    pub fn is_tiered(self) -> bool {
        matches!(self, UnderstudentKind::Ladder | UnderstudentKind::Auto)
    }

    /// Whether answering needs the model loaded (everything but `keywords`, `rules`, `shape` and `memo`; a ladder's `tiers` decide).
    pub fn needs_encoder(self) -> bool {
        !matches!(self, UnderstudentKind::Keywords | UnderstudentKind::Rules | UnderstudentKind::Shape | UnderstudentKind::Memo | UnderstudentKind::Checklist)
    }

    /// The default ladder tiers (and auto candidates), cheapest first. Adding a candidate is one entry here (plus its
    /// arm in `check_tiers` and `Featurizer::tier`).
    pub fn default_tiers() -> Vec<UnderstudentKind> {
        vec![UnderstudentKind::Keywords, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep]
    }
}

/// The `readout` a decision reports for a (non-ladder) understudent's fit.
pub fn readout_label(kind: UnderstudentKind, r: Readout) -> &'static str {
    match kind {
        UnderstudentKind::Probe => "mean",
        UnderstudentKind::Lite => "lite",
        _ => r.name(),
    }
}

/// What a refit needs besides the examples.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FitSettings {
    pub target: f64,
    pub minority: f64,
    pub min_examples: usize,
    pub novelty_refs: usize,
    pub readout: ReadoutChoice,
    pub attn: AttnParams,
    pub seed: u64,
    #[serde(default)]
    pub understudent: UnderstudentKind,
    /// A ladder's tiers or auto's candidates, cheapest first (empty for any other understudent).
    #[serde(default)]
    pub tiers: Vec<UnderstudentKind>,
    /// How out-of-fold predictions become a graduation threshold (`abstain.rs`); the default is hunch's single threshold.
    #[serde(default, skip_serializing_if = "Abstain::is_default")]
    pub abstain: Abstain,
    /// How a refit's graduation test treats repeated looks (`--graduation`).
    #[serde(default, skip_serializing_if = "Graduation::is_default")]
    pub graduation: Graduation,
    /// The graduation attempt this fit is (1-based; set by the proxy for each refit, never stored with the settings).
    #[serde(skip)]
    pub attempt: usize,
}

impl FitSettings {
    /// The Wilson z of this fit's graduation bars: 1.96, or the attempt's spent z under `--graduation sequential`
    /// (Bonferroni over the `tiers` raced in the attempt).
    pub fn grad_z(&self) -> f64 {
        match self.graduation {
            Graduation::Pointwise => BAR_Z,
            Graduation::Sequential => crate::stats::sequential_z(self.attempt, self.tiers.len()),
        }
    }
}

/// How graduation accounts for a question being tested at every refit (`--graduation`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum Graduation {
    /// Every refit tests the pointwise 95% Wilson bounds (z = 1.96).
    #[default]
    #[serde(rename = "pointwise")]
    #[value(name = "pointwise")]
    Pointwise,
    /// Alpha-spending across a question's attempts: attempt k tests at the two-sided level 0.05 * 6 / (pi^2 k^2) / m.
    #[serde(rename = "sequential")]
    #[value(name = "sequential")]
    Sequential,
}

impl Graduation {
    pub fn is_default(&self) -> bool {
        *self == Graduation::Pointwise
    }
}

/// One training row.
pub struct Row<'a> {
    pub enc: &'a Encoding,
    pub y: usize,
    /// hunch's weight: 2.0 for a label, else the lead's confidence (max probability).
    pub w: f64,
    /// Rows sharing a group (e.g. the per-response sub-items of one item, ids `<item>#<k>`) never straddle CV folds.
    pub group: Option<u64>,
}

/// An understudent's fitted state. A refit builds a new one and the understudent swaps it in whole.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fit {
    /// The readout that answers (and whose OOF predictions graded graduation).
    pub readout: Option<Readout>,
    pub pool: Option<PoolModel>,
    pub attn: Option<AttnModel>,
    pub sparse: Option<SparseLr>,
    pub neighbors: Option<Neighbors>,
    /// The `rules` understudent's rule set (validated on held-out rows).
    pub rules: Option<Arc<RuleSet>>,
    /// The `blend` understudent's block standardizers (its model is `sparse`, over the stacked columns).
    pub blend: Option<BlendStd>,
    /// The `memo` understudent's stored keys and lead answers.
    pub memo: Option<MemoModel>,
    /// The `net` understudent's network.
    pub net: Option<NetModel>,
    /// The `checklist` understudent's rules.
    pub checklist: Option<Checklist>,
    /// `--abstain calibrated`: the per-class confidence thresholds a symmetric answer must reach (indexed by the predicted
    /// class); None answers at `thr` whatever the class.
    pub class_thr: Option<Vec<f64>>,
    pub mode: Option<Mode>,
    pub thr: f64,
    /// Normalized novelty vectors of the newest examples, [refs x ref_dim].
    pub refs: Vec<f32>,
    pub ref_dim: usize,
    pub nn_floor: f64,
    pub report: Value,
}

impl Fit {
    pub fn has_model(&self) -> bool {
        match self.readout {
            Some(Readout::Pool) => self.pool.is_some(),
            Some(Readout::Attn) => self.attn.is_some(),
            Some(Readout::Keywords) => self.sparse.is_some(),
            Some(Readout::Neighbors) => self.neighbors.as_ref().is_some_and(|k| !k.is_empty()),
            Some(Readout::Rules) => self.rules.is_some(),
            Some(Readout::Shape) => self.pool.is_some(),
            Some(Readout::Blend) => self.sparse.is_some() && self.blend.is_some(),
            Some(Readout::Memo) => self.memo.as_ref().is_some_and(|m| !m.is_empty()),
            Some(Readout::Net) => self.net.is_some(),
            Some(Readout::Checklist) => self.checklist.is_some(),
            None => false,
        }
    }

    pub fn proba(&self, fz: &Featurizer, e: &Encoding, k: usize) -> Option<Vec<f64>> {
        match self.readout? {
            Readout::Pool => Some(self.pool.as_ref()?.proba(&fz.pool_features(e), k)),
            Readout::Attn => Some(self.attn.as_ref()?.proba(fz, e, k)),
            Readout::Keywords => Some(self.sparse.as_ref()?.proba(e.sparse.as_ref()?, k)),
            Readout::Neighbors => Some(self.neighbors.as_ref()?.proba(&fz.novelty(e), k, None)),
            Readout::Rules => self.rules.as_ref()?.proba(e.rules.as_ref()?, k),
            Readout::Shape => Some(shape::proba(self.pool.as_ref()?, e.shape.as_ref()?, k)),
            Readout::Blend => {
                let (sp, st) = (self.sparse.as_ref()?, self.blend.as_ref()?);
                Some(sp.proba(&shape::blend_row(e.sparse.as_ref()?, e.shape.as_ref()?, &fz.pool_features(e), st), k))
            }
            Readout::Memo => self.memo.as_ref()?.proba(e.memo.as_ref()?, k),
            Readout::Net => Some(self.net.as_ref()?.proba(&fz.pool_features(e), e.shape.as_ref()?, k)),
            Readout::Checklist => self.checklist.as_ref()?.proba(&checklist::row_ids(e.sparse.as_ref()?, e.shape.as_ref()?), k),
        }
    }

    /// The vector novelty is judged on: the projected n-gram vector for `keywords`, else the layer mean.
    pub fn novelty_vec(&self, fz: &Featurizer, e: &Encoding) -> Option<Vec<f32>> {
        match self.readout {
            Some(Readout::Keywords | Readout::Blend | Readout::Checklist) => Some(keywords::project(e.sparse.as_ref()?)),
            Some(Readout::Rules | Readout::Memo) => None,
            Some(Readout::Shape) => Some(shape::novelty(&self.pool.as_ref()?.std, e.shape.as_ref()?)),
            _ => Some(fz.novelty(e)),
        }
    }

    /// Whether `e` is novel: its nearest reference is less similar than the floor. False without references.
    pub fn novel(&self, fz: &Featurizer, e: &Encoding) -> bool {
        if self.ref_dim == 0 || self.refs.len() / self.ref_dim < 2 {
            return false;
        }
        let Some(v) = self.novelty_vec(fz, e) else { return false };
        let best = self.refs.chunks_exact(self.ref_dim).map(|r| crate::simd::dot(r, &v)).fold(f32::NEG_INFINITY, f32::max);
        (best as f64) < self.nn_floor
    }
}

/// The CV folds of `fit_understudent` (and of memo's inner threshold search). Without any group prefix: plain
/// `FOLDS`-fold `kfold`, as ever. With groups (some row has `Some(group)`; a `None` row counts as its own group):
/// `GroupKFold` with `min(FOLDS, groups)` folds, so siblings never straddle a split even with fewer than `FOLDS`
/// groups; None (decline to graduate) when there are fewer than 2 groups. Never an ungrouped split of grouped rows.
fn cv_folds(groups: &[Option<u64>], seed: u32) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
    let n = groups.len();
    if groups.iter().all(Option::is_none) {
        return Some(kfold(n, FOLDS, seed));
    }
    let distinct = groups.iter().filter_map(|g| *g).collect::<std::collections::HashSet<_>>().len() + groups.iter().filter(|g| g.is_none()).count();
    let k = distinct.min(FOLDS);
    (k >= 2).then(|| group_kfold(groups, k, seed))
}

/// Probabilities from each fold's model for its test rows (zeros where a fold saw a single class, as hunch).
type Oof = Vec<Vec<f64>>;

/// Fit a question's understudent (`s.understudent`, never `Ladder`: the proxy fits a ladder's tiers separately) on its examples
/// (oldest first) and decide graduation by 5-fold CV.
pub fn fit_understudent(fz: &Featurizer, qtype: QType, k: usize, rows: &[Row], s: &FitSettings) -> Fit {
    let t0 = Instant::now();
    let n = rows.len();
    let leaf = s.understudent;
    assert!(!leaf.is_tiered(), "a ladder or auto is fitted tier by tier");
    let mut f = Fit { report: json!({"n": n}), nn_floor: -1.0, ..Default::default() };
    if n == 0 {
        return f;
    }
    let y: Vec<usize> = rows.iter().map(|r| r.y).collect();
    let w: Vec<f64> = rows.iter().map(|r| r.w).collect();
    let encs: Vec<&Encoding> = rows.iter().map(|r| r.enc).collect();

    let mut classes = y.clone();
    classes.sort_unstable();
    classes.dedup();
    let groups: Vec<Option<u64>> = rows.iter().map(|r| r.group).collect();
    // `--abstain calibrated`: a frozen model certified on rows it never saw (`fit_certified`); declined to the CV path
    // below (which then never graduates) only when the groups cannot fill a train / select / certify split
    let split = (s.abstain == Abstain::Calibrated && n >= s.min_examples && classes.len() >= 2).then(|| abstain::three_way(&groups, s.seed)).flatten();
    if let Some(split) = split {
        return fit_certified(fz, qtype, k, rows, s, &split, t0);
    }

    // novelty: the newest `novelty_refs` examples and the 5th percentile of their nearest-neighbour similarities
    // (`memo` has none: it only answers near-duplicates)
    let shape_std = build_novelty(&mut f, leaf, fz, &encs, s);
    let enough = n >= s.min_examples && classes.len() >= 2;
    // grouped folds when ids carry a group prefix (`cv_folds`); too few groups to hold anything out: decline
    let folds_opt = if enough { cv_folds(&groups, s.seed as u32) } else { None };
    let no_groups = enough && folds_opt.is_none();
    let enough = enough && !no_groups;
    let folds = folds_opt.unwrap_or_default();
    let oof = fit_leaf(&mut f, fz, leaf, &encs, &folds, &groups, &y, &w, k, qtype, s, enough, shape_std);
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("labelled".into(), json!(w.iter().filter(|&&x| x == 2.0).count()));
    rep.insert("readout".into(), json!(readout_label(leaf, f.readout.unwrap())));
    if !enough {
        rep.insert(
            "why_not".into(),
            json!(if n < s.min_examples {
                "too few examples"
            } else if no_groups {
                "fewer than 2 id groups: no grouped cross-validation is possible"
            } else {
                "only one answer seen"
            }),
        );
        rep.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
        return f;
    }
    let p = oof.as_ref().unwrap();

    // graduation, exactly as hunch's fit_understudent
    let pred: Vec<usize> = p.iter().map(|x| argmax(x)).collect();
    let cf: Vec<f64> = p.iter().map(|x| x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect();
    let agree: Vec<bool> = pred.iter().zip(&y).map(|(a, b)| a == b).collect();
    let mut counts = vec![0usize; k];
    y.iter().for_each(|&c| counts[c] += 1);
    let maj = argmax(&counts.iter().map(|&c| c as f64).collect::<Vec<_>>());
    let minor: Vec<bool> = y.iter().map(|&c| c != maj).collect();
    let n_minor = minor.iter().filter(|&&m| m).count();
    rep.insert("cv_agreement".into(), json!(agree.iter().filter(|&&a| a).count() as f64 / n as f64));
    rep.insert(
        "cv_minority_recall".into(),
        if n_minor > 0 {
            json!(agree.iter().zip(&minor).filter(|(a, m)| **a && **m).count() as f64 / n_minor as f64)
        } else {
            Value::Null
        },
    );
    rep.insert("majority_share".into(), json!(1.0 - n_minor as f64 / n as f64));
    if matches!(leaf, UnderstudentKind::Memo | UnderstudentKind::Checklist) {
        // an abstention is not a disagreement: memo's agreement is over the rows it answers (0.0 below 30 such rows,
        // so it never props up an auto race's give-up check)
        let answered: Vec<usize> = (0..n).filter(|&j| !p[j][0].is_nan()).collect();
        let hit = answered.iter().filter(|&&j| agree[j]).count();
        let (m_ans, m_hit) = (answered.iter().filter(|&&j| minor[j]).count(), answered.iter().filter(|&&j| minor[j] && agree[j]).count());
        rep.insert("answered".into(), json!(answered.len()));
        rep.insert("cv_agreement".into(), json!(if answered.len() >= 30 { hit as f64 / answered.len() as f64 } else { 0.0 }));
        rep.insert("cv_minority_recall".into(), if m_ans > 0 { json!(m_hit as f64 / m_ans as f64) } else { Value::Null });
    }
    let p_yes: Option<Vec<f64>> = (qtype == QType::Noul).then(|| p.iter().map(|x| x[1]).collect());
    let mut best = graduation_z(&cf, &agree, &minor, p_yes.as_deref(), &y, s.target, s.minority, s.grad_z());
    if s.abstain == Abstain::Calibrated {
        // too few id groups for a train / select / certify split: a calibrated policy cannot be certified, so no graduation
        rep.insert("abstain".into(), json!("calibrated"));
        best = (0.0, None, 0.0);
    }
    (f.mode, f.thr) = (best.1, best.2);
    rep.insert("mode".into(), json!(f.mode.map(Mode::name)));
    rep.insert("threshold".into(), json!(f.thr));
    rep.insert("cv_coverage".into(), json!(best.0));
    if f.mode.is_none() {
        rep.insert("why_not".into(), json!(if s.abstain == Abstain::Calibrated { "fewer than 3 id groups: no train / select / certify split is possible" } else { "CV agreement / minority recall below the graduation bars" }));
    }
    rep.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
    f
}

/// The novelty reference set (the newest `novelty_refs` of `encs`, oldest first) and floor (the 5th percentile of
/// their nearest-neighbour similarities) into `f`; returns `shape`'s standardizer, fitted on `encs`.
fn build_novelty(f: &mut Fit, leaf: UnderstudentKind, fz: &Featurizer, encs: &[&Encoding], s: &FitSettings) -> Option<Standardizer> {
    let n = encs.len();
    let shape_std = (leaf == UnderstudentKind::Shape).then(|| {
        let x: Vec<f32> = encs.iter().flat_map(|e| e.shape.as_ref().expect("shape needs features").iter().copied()).collect();
        Standardizer::fit(&x, shape::DIM)
    });
    let nov = |e: &Encoding| -> Vec<f32> {
        match leaf {
            UnderstudentKind::Keywords | UnderstudentKind::Blend | UnderstudentKind::Checklist => keywords::project(e.sparse.as_ref().expect("keywords needs hashed features")),
            UnderstudentKind::Shape => shape::novelty(shape_std.as_ref().unwrap(), e.shape.as_ref().expect("shape needs features")),
            _ => fz.novelty(e),
        }
    };
    if leaf != UnderstudentKind::Memo {
        let refs: Vec<Vec<f32>> = encs[n.saturating_sub(s.novelty_refs)..].iter().map(|e| nov(e)).collect();
        f.ref_dim = refs[0].len();
        f.refs = refs.concat();
        if refs.len() > 1 {
            let nn: Vec<f64> = (0..refs.len())
                .into_par_iter()
                .map(|i| {
                    (0..refs.len()).filter(|&j| j != i).map(|j| crate::simd::dot(&refs[i], &refs[j])).fold(f32::NEG_INFINITY, f32::max) as f64
                })
                .collect();
            f.nn_floor = quantile(&nn, 0.05);
        }
    }
    shape_std
}

/// Fit the leaf understudent's model(s) on `encs` into `f`; the out-of-fold probabilities over `folds` (None without folds).
#[allow(clippy::too_many_arguments)]
fn fit_leaf(f: &mut Fit, fz: &Featurizer, leaf: UnderstudentKind, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], groups: &[Option<u64>], y: &[usize],
            w: &[f64], k: usize, qtype: QType, s: &FitSettings, enough: bool, shape_std: Option<Standardizer>) -> Option<Oof> {
    match leaf {
        UnderstudentKind::Keywords => fit_keywords(f, encs, folds, y, w, k),
        UnderstudentKind::Neighbors => fit_neighbors(f, fz, encs, folds, y, k),
        UnderstudentKind::Shape => fit_shape(f, encs, folds, y, w, k, shape_std.unwrap()),
        UnderstudentKind::Blend => fit_stack(f, fz, encs, folds, y, w, k),
        UnderstudentKind::Memo => fit_memo(f, encs, folds, groups, y, w, k, qtype, s),
        UnderstudentKind::Net => fit_net(f, fz, encs, folds, groups, y, w, k, s),
        UnderstudentKind::Checklist => fit_checklist(f, encs, folds, y, k, qtype),
        _ => fit_neural(f, fz, encs, folds, y, w, k, s, enough),
    }
}

/// `--abstain calibrated`: fit on TRAIN alone, choose the per-class thresholds on SELECT, evaluate the exact frozen
/// serving predicate once on CERTIFY, and graduate only if that certificate meets the bars (`abstain.rs`). The fit that
/// is returned is the TRAIN model: nothing is refitted after certification.
fn fit_certified(fz: &Featurizer, qtype: QType, k: usize, rows: &[Row], s: &FitSettings, split: &abstain::Split, t0: Instant) -> Fit {
    let leaf = s.understudent;
    let pick = |idx: &[usize]| -> Vec<&Row> { idx.iter().map(|&i| &rows[i]).collect() };
    let train = pick(&split.train);
    let encs: Vec<&Encoding> = train.iter().map(|r| r.enc).collect();
    let y: Vec<usize> = train.iter().map(|r| r.y).collect();
    let w: Vec<f64> = train.iter().map(|r| r.w).collect();
    let groups: Vec<Option<u64>> = train.iter().map(|r| r.group).collect();
    let mut f = Fit { report: json!({"n": rows.len()}), nn_floor: -1.0, ..Default::default() };
    let shape_std = build_novelty(&mut f, leaf, fz, &encs, s);
    // no folds and `enough = false`: the final model alone (deep fits its pool readout unless attention is forced)
    fit_leaf(&mut f, fz, leaf, &encs, &[], &groups, &y, &w, k, qtype, s, false, shape_std);
    let tr_classes = {
        let mut c = y.clone();
        c.sort_unstable();
        c.dedup();
        c.len()
    };
    // the minority class is the one the lead answers least among TRAIN's rows (frozen: no SELECT or CERTIFY label shapes it)
    let mut counts = vec![0usize; k];
    y.iter().for_each(|&c| counts[c] += 1);
    let maj = argmax(&counts.iter().map(|&c| c as f64).collect::<Vec<_>>());

    let mut rep = std::mem::take(&mut f.report);
    {
        let r = rep.as_object_mut().unwrap();
        r.insert("labelled".into(), json!(rows.iter().filter(|r| r.w == 2.0).count()));
        r.insert("readout".into(), json!(readout_label(leaf, f.readout.unwrap())));
        r.insert("abstain".into(), json!("calibrated"));
        r.insert("split".into(), json!({"train": split.train.len(), "select": split.select.len(), "certify": split.certify.len()}));
    }
    let finish = |mut f: Fit, mut rep: Value, why: Option<&str>, cov: f64, agreement: Option<f64>| -> Fit {
        let r = rep.as_object_mut().unwrap();
        r.insert("mode".into(), json!(f.mode.map(Mode::name)));
        r.insert("threshold".into(), json!(f.thr));
        r.insert("cv_coverage".into(), json!(cov));
        r.insert("cv_agreement".into(), json!(agreement.unwrap_or(0.0)));
        if let Some(why) = why {
            r.insert("why_not".into(), json!(why));
        }
        r.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
        f.report = rep;
        f
    };
    if !f.has_model() || tr_classes < 2 {
        return finish(f, rep, Some("no model could be fitted on the training part"), 0.0, None);
    }

    // The frozen model's predictions, through the very predicate serving uses: a novel row or an abstention is forwarded.
    let frozen = Serving { fit: Arc::new(f), ..Default::default() };
    let scored = |idx: &[usize]| -> (Vec<usize>, Vec<f64>, Vec<bool>, Vec<bool>) {
        let out: Vec<(usize, f64)> = idx
            .par_iter()
            .map(|&j| match frozen.fit.proba(fz, rows[j].enc, k) {
                Some(p) if !p.iter().any(|x| x.is_nan()) && !frozen.fit.novel(fz, rows[j].enc) => (argmax(&p), p.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
                _ => (0, f64::NEG_INFINITY),
            })
            .collect();
        let agree = idx.iter().zip(&out).map(|(&j, o)| o.0 == rows[j].y).collect();
        let minor = idx.iter().map(|&j| rows[j].y != maj).collect();
        (out.iter().map(|o| o.0).collect(), out.iter().map(|o| o.1).collect(), agree, minor)
    };
    // SELECT: the thresholds, with the novelty gate applied
    let (sp, sc, sa, sm) = scored(&split.select);
    let sel_rows: Vec<usize> = (0..split.select.len()).collect();
    let thr = abstain::choose(&abstain::Preds { pred: &sp, cf: &sc, agree: &sa, minor: &sm }, &sel_rows, k, s.target, s.minority);
    let mut f = Arc::try_unwrap(frozen.fit).unwrap_or_else(|a| (*a).clone());
    let Some(thr) = thr else {
        let r = rep.as_object_mut().unwrap();
        r.insert("select".into(), json!({"rows": split.select.len(), "thresholds": null}));
        return finish(f, rep, Some("no per-class thresholds meet the graduation bars on the selection part"), 0.0, None);
    };
    let sel_t = abstain::Preds { pred: &sp, cf: &sc, agree: &sa, minor: &sm }.tally(&sel_rows, &thr);
    // CERTIFY: the exact frozen policy (model + thresholds + novelty floor), once. Several candidates share these
    // rows, so each is certified at the Bonferroni-corrected z.
    f.mode = Some(Mode::Symmetric);
    f.thr = thr.iter().cloned().filter(|t| t.is_finite()).fold(f64::INFINITY, f64::min);
    f.class_thr = Some(thr.clone());
    let z = match s.graduation {
        Graduation::Pointwise => abstain::bonferroni_z(s.tiers.len()),
        Graduation::Sequential => crate::stats::sequential_z(s.attempt, s.tiers.len()),
    };
    let sv = Serving { fit: Arc::new(f), ..Default::default() };
    let eligible: Vec<(bool, bool, bool)> = split
        .certify
        .par_iter()
        .map(|&j| (sv.predict(fz, rows[j].enc, k).eligible, sv.fit.proba(fz, rows[j].enc, k).is_some_and(|p| argmax(&p) == rows[j].y), rows[j].y != maj))
        .collect();
    let mut cert = abstain::Tally::default();
    for &(ok, agree, minor) in &eligible {
        if ok {
            cert.add(agree, minor);
        }
    }
    let mut f = Arc::try_unwrap(sv.fit).unwrap_or_else(|a| (*a).clone());
    let passed = cert.passes_z(s.target, s.minority, z);
    let (agr, low) = if cert.sel > 0 { (cert.agree as f64 / cert.sel as f64, crate::stats::wilson_z(cert.agree, cert.sel, z).0) } else { (0.0, 0.0) };
    let cov = cert.sel as f64 / split.certify.len() as f64;
    {
        let r = rep.as_object_mut().unwrap();
        r.insert("select".into(), json!({"rows": split.select.len(), "answered": sel_t.sel, "agreed": sel_t.agree, "thresholds": thr}));
        r.insert("certificate".into(), json!({
            "rows": split.certify.len(), "answered": cert.sel, "agreed": cert.agree, "agreement": agr, "wilson_low": low, "z": z,
            "minority_answered": cert.ms, "minority_agreed": cert.ms_agree, "coverage": cov, "passed": passed,
        }));
        r.insert("answered".into(), json!(cert.sel));
    }
    if !passed {
        (f.mode, f.thr, f.class_thr) = (None, 0.0, None);
        return finish(f, rep, Some("the frozen policy's certificate on held-out rows is below the graduation bars"), cov, Some(agr));
    }
    finish(f, rep, None, cov, Some(agr))
}

/// hunch's graduation scan over per-row predictions (shared by every understudent): `cf` the confidence of each row's
/// prediction (-inf for a row the understudent abstains on), `agree` whether it matched the lead, `minor` whether the
/// lead's class is not the majority, `p_yes` (noul only) the predicted P(yes) (above 1 for an abstention). Returns
/// (coverage, mode, threshold): the symmetric scan first, then (noul, >= 15 lead "yes") the one-sided scan, taken
/// when its coverage is larger, then its mirror (`one_sided_yes`: only a confident "yes" is local, bars mirrored) likewise.
pub fn graduation(cf: &[f64], agree: &[bool], minor: &[bool], p_yes: Option<&[f64]>, y: &[usize], target: f64, minority: f64)
                  -> (f64, Option<Mode>, f64) {
    graduation_z(cf, agree, minor, p_yes, y, target, minority, BAR_Z)
}

/// `graduation` with every Wilson bound at `z` (1.96 is `graduation`; `--graduation sequential` raises it with the attempt).
#[allow(clippy::too_many_arguments)]
pub fn graduation_z(cf: &[f64], agree: &[bool], minor: &[bool], p_yes: Option<&[f64]>, y: &[usize], target: f64, minority: f64, z: f64)
                    -> (f64, Option<Mode>, f64) {
    let wilson = |k: usize, m: usize| wilson_z(k, m, z);
    let n = y.len();
    let mut best: (f64, Option<Mode>, f64) = (0.0, None, 0.0);
    for i in 0..50 {
        let t = 0.5 + 0.01 * i as f64; // np.arange(0.5, 0.995, 0.01)
        let (mut sel, mut sel_agree, mut ms, mut ms_agree) = (0, 0, 0, 0);
        for j in 0..n {
            if cf[j] >= t {
                sel += 1;
                sel_agree += usize::from(agree[j]);
                if minor[j] {
                    ms += 1;
                    ms_agree += usize::from(agree[j]);
                }
            }
        }
        if sel >= 30 && wilson(sel_agree, sel).0 >= target && ms >= 15 && wilson(ms_agree, ms).0 >= minority {
            best = (sel as f64 / n as f64, Some(Mode::Symmetric), t);
            break;
        }
    }
    let yes_total = y.iter().filter(|&&c| c == 1).count();
    if let (Some(p_yes), true) = (p_yes, yes_total >= 15) {
        let mut taus: Vec<f64> = p_yes.iter().filter(|x| **x <= 1.0).map(|x| (x * 1e4).round_ties_even() / 1e4).collect();
        taus.sort_by(|a, b| b.total_cmp(a));
        taus.dedup();
        for tau in taus {
            let (mut local, mut local_yes, mut fwd_yes) = (0, 0, 0);
            for j in 0..n {
                if p_yes[j] < tau {
                    local += 1;
                    local_yes += usize::from(y[j] == 1);
                } else {
                    fwd_yes += usize::from(y[j] == 1);
                }
            }
            if local >= 30 && wilson(fwd_yes, yes_total).0 >= ONE_SIDED_RECALL && wilson(local_yes, local).1 <= MISS_RATE {
                let cov = local as f64 / n as f64;
                if cov > best.0 {
                    best = (cov, Some(Mode::OneSided), tau);
                }
                break;
            }
        }
    }
    // the mirror: answer locally only a confident "yes" (p_yes > thr; an abstention never), forward the rest. The
    // thresholds are the observed P(yes) just below (1e-4 under) each value, so the least confident local "yes"
    // stays local; none under 0.5, so a local answer is the model's own argmax. Smallest threshold (most coverage) first.
    let no_total = y.iter().filter(|&&c| c == 0).count();
    if let (Some(p_yes), true, true) = (p_yes, yes_total >= 15, no_total > 0) {
        let mut thrs: Vec<f64> = p_yes.iter().filter(|x| **x <= 1.0).map(|x| (x * 1e4).round_ties_even() / 1e4).filter(|t| *t >= 0.5001).map(|t| t - 1e-4).collect();
        thrs.sort_by(f64::total_cmp);
        thrs.dedup();
        for thr in thrs {
            let (mut local, mut local_no, mut fwd_no) = (0, 0, 0);
            for j in 0..n {
                if p_yes[j] <= 1.0 && p_yes[j] > thr {
                    local += 1;
                    local_no += usize::from(y[j] == 0);
                } else {
                    fwd_no += usize::from(y[j] == 0);
                }
            }
            if local >= 30 && wilson(fwd_no, no_total).0 >= ONE_SIDED_RECALL && wilson(local_no, local).1 <= MISS_RATE {
                let cov = local as f64 / n as f64;
                if cov > best.0 {
                    best = (cov, Some(Mode::OneSidedYes), thr);
                }
                break;
            }
        }
    }
    best
}

/// The neural understudents (`deep`, `probe`, `lite`): pool and/or attention readouts over the encodings. Fills `f`'s models and
/// readout; returns the chosen readout's out-of-fold probabilities (None when too few examples for CV).
#[allow(clippy::too_many_arguments)]
fn fit_neural(f: &mut Fit, fz: &Featurizer, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize], w: &[f64], k: usize,
              s: &FitSettings, enough: bool) -> Option<Oof> {
    let n = encs.len();
    // deep reads either readout (auto); the others are pool only
    let (want_pool, want_attn) = match s.understudent {
        UnderstudentKind::Deep => (s.readout != ReadoutChoice::Attn, s.readout != ReadoutChoice::Pool),
        _ => (true, false),
    };
    // Before graduation is possible only the final model is fitted: pool, unless attn is forced.
    let (do_pool, do_attn) = if enough { (want_pool, want_attn) } else { (want_pool, !want_pool) };

    let d = fz.pool_dim();
    let (pool_std, z) = if do_pool {
        let x: Vec<f32> = encs.par_iter().flat_map_iter(|e| fz.pool_features(e)).collect();
        let st = Standardizer::fit(&x, d);
        let mut z = vec![0f32; n * d];
        z.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| st.apply(r, o));
        (Some(st), z)
    } else {
        (None, vec![])
    };
    let tok_std = do_attn.then(|| token_standardizer(fz, encs, fz.hidden));
    let all: Vec<usize> = (0..n).collect();

    // Jobs: (readout, fold or None for the final model); all run in parallel, each single-threaded.
    let mut jobs: Vec<(Readout, Option<usize>)> = vec![];
    for (r, on) in [(Readout::Pool, do_pool), (Readout::Attn, do_attn)] {
        if on {
            jobs.push((r, None));
            jobs.extend((0..folds.len()).map(|i| (r, Some(i))));
        }
    }
    enum Model {
        Pool(Option<linear::Logistic>),
        Attn(Option<AttnModel>),
    }
    let timed: Vec<(Model, f64)> = jobs
        .par_iter()
        .map(|&(r, fold)| {
            let t = Instant::now();
            let idx = fold.map_or(&all, |i| &folds[i].0);
            let m = match r {
                Readout::Pool => Model::Pool(linear::fit(&z, d, idx, y, w, linear::C)),
                Readout::Attn => {
                    let seed = s.seed.wrapping_mul(1000).wrapping_add(fold.map_or(0, |i| i as u64 + 1));
                    Model::Attn(AttnModel::train(fz, encs, idx, y, w, tok_std.as_ref().unwrap(), &s.attn, seed))
                }
                _ => unreachable!("the neural fit trains pool and attn only"),
            };
            (m, t.elapsed().as_secs_f64())
        })
        .collect();

    let mut oof: [Option<Oof>; 2] = [None, None];
    let mut secs = [0.0f64; 2];
    for ((r, fold), (m, t)) in jobs.iter().zip(timed) {
        let ri = *r as usize;
        secs[ri] += t;
        match (fold, m) {
            (None, Model::Pool(m)) => f.pool = m.map(|lr| PoolModel { std: pool_std.clone().unwrap(), lr }),
            (None, Model::Attn(m)) => f.attn = m,
            (Some(i), m) => {
                let o = oof[ri].get_or_insert_with(|| vec![vec![0.0; k]; n]);
                let te = &folds[*i].1;
                let probs: Vec<Vec<f64>> = match &m {
                    Model::Pool(Some(lr)) => te.par_iter().map(|&j| lr.proba(&z[j * d..(j + 1) * d], k)).collect(),
                    Model::Attn(Some(am)) => te.par_iter().map(|&j| am.proba(fz, encs[j], k)).collect(),
                    _ => vec![vec![0.0; k]; te.len()],
                };
                for (&j, p) in te.iter().zip(probs) {
                    o[j] = p;
                }
            }
        }
    }
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("fit_secs".into(), json!({"pool": secs[0], "attn": secs[1]}));
    if do_attn {
        rep.insert("attn_epochs".into(), json!(s.attn.epochs(n)));
    }
    if !enough {
        f.readout = Some(if do_pool { Readout::Pool } else { Readout::Attn });
        return None;
    }

    // readout choice: OOF agreement, one standard error in favour of pool
    let agreement = |p: &Oof| p.iter().zip(y).filter(|(p, y)| argmax(p) == **y).count() as f64 / n as f64;
    let mut agree_by = serde_json::Map::new();
    for (r, o) in [(Readout::Pool, &oof[0]), (Readout::Attn, &oof[1])] {
        if let Some(o) = o {
            agree_by.insert(r.name().into(), json!(agreement(o)));
        }
    }
    let chosen = match (&oof[0], &oof[1]) {
        (Some(p), Some(a)) => {
            let (ap, aa) = (agreement(p), agreement(a));
            let se = (aa * (1.0 - aa) / n as f64).sqrt();
            if aa > ap + se { Readout::Attn } else { Readout::Pool }
        }
        (Some(_), None) => Readout::Pool,
        _ => Readout::Attn,
    };
    f.readout = Some(chosen);
    rep.insert("cv_agreement_by_readout".into(), Value::Object(agree_by));
    oof[chosen as usize].take()
}

/// The `keywords` understudent: sparse logistic regression on the hashed features.
fn fit_keywords(f: &mut Fit, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize], w: &[f64], k: usize) -> Option<Oof> {
    let n = encs.len();
    let xs: Vec<&keywords::Sparse> = encs.iter().map(|e| e.sparse.as_ref().expect("keywords needs hashed features")).collect();
    let all: Vec<usize> = (0..n).collect();
    let jobs: Vec<Option<usize>> = std::iter::once(None).chain((0..folds.len()).map(Some)).collect();
    let timed: Vec<(Option<SparseLr>, f64)> = jobs
        .par_iter()
        .map(|&fold| {
            let t = Instant::now();
            (keywords::fit(&xs, fold.map_or(&all, |i| &folds[i].0), y, w, keywords::C), t.elapsed().as_secs_f64())
        })
        .collect();
    f.readout = Some(Readout::Keywords);
    let mut oof: Option<Oof> = None;
    let mut secs = 0.0;
    for (fold, (m, t)) in jobs.iter().zip(timed) {
        secs += t;
        match fold {
            None => f.sparse = m,
            Some(i) => {
                let o = oof.get_or_insert_with(|| vec![vec![0.0; k]; n]);
                let te = &folds[*i].1;
                let probs: Vec<Vec<f64>> = match &m {
                    Some(lr) => te.iter().map(|&j| lr.proba(xs[j], k)).collect(),
                    None => vec![vec![0.0; k]; te.len()],
                };
                for (&j, p) in te.iter().zip(probs) {
                    o[j] = p;
                }
            }
        }
    }
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("fit_secs".into(), json!({"keywords": secs}));
    rep.insert("keywords_iters".into(), json!(f.sparse.as_ref().map(|m| m.iters)));
    oof
}

/// The `neighbors` understudent: the training vectors are the L2-normalized layer means; OOF votes come from each fold's
/// training rows.
fn fit_neighbors(f: &mut Fit, fz: &Featurizer, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize], k: usize) -> Option<Oof> {
    let t = Instant::now();
    let n = encs.len();
    let vecs: Vec<Vec<f32>> = encs.par_iter().map(|e| fz.novelty(e)).collect();
    let d = vecs[0].len();
    let m = Neighbors::new(d, vecs.concat(), y.iter().map(|&c| c as u32).collect());
    f.readout = Some(Readout::Neighbors);
    let oof = (!folds.is_empty()).then(|| {
        let mut o: Oof = vec![vec![0.0; k]; n];
        for (train, test) in folds {
            let ps: Vec<Vec<f64>> = test.par_iter().map(|&j| m.proba(&vecs[j], k, Some(train))).collect();
            for (&j, p) in test.iter().zip(ps) {
                o[j] = p;
            }
        }
        o
    });
    f.neighbors = Some(m);
    f.report.as_object_mut().unwrap().insert("fit_secs".into(), json!({"neighbors": t.elapsed().as_secs_f64()}));
    oof
}

/// The `shape` understudent: standardized (clipped) structural features, `linear.rs`'s logistic regression at its C (0.5).
/// Each fold's standardizer is fitted on that fold's TRAINING rows only, so the out-of-fold predictions never see a
/// held-out row's statistics; the final model's (`std`) is fitted on all rows.
fn fit_shape(f: &mut Fit, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize], w: &[f64], k: usize, std: Standardizer) -> Option<Oof> {
    let t = Instant::now();
    let (n, d) = (encs.len(), shape::DIM);
    let raw: Vec<f32> = encs.iter().flat_map(|e| e.shape.as_ref().expect("shape needs features").iter().copied()).collect();
    let all: Vec<usize> = (0..n).collect();
    let jobs: Vec<Option<usize>> = std::iter::once(None).chain((0..folds.len()).map(Some)).collect();
    // (the final model, or a fold's model with its test rows' probabilities)
    let models: Vec<(Option<linear::Logistic>, Vec<Vec<f64>>)> = jobs
        .par_iter()
        .map(|&fold| {
            let (st, train) = match fold {
                None => (std.clone(), &all),
                Some(i) => {
                    let x: Vec<f32> = folds[i].0.iter().flat_map(|&j| raw[j * d..(j + 1) * d].iter().copied()).collect();
                    (Standardizer::fit(&x, d), &folds[i].0)
                }
            };
            let mut z = vec![0f32; n * d];
            z.par_chunks_mut(d).zip(raw.par_chunks(d)).for_each(|(o, r)| o.copy_from_slice(&shape::standardize(&st, r)));
            let lr = linear::fit(&z, d, train, y, w, linear::C);
            let probs = fold.map_or_else(Vec::new, |i| {
                folds[i].1.iter().map(|&j| lr.as_ref().map_or_else(|| vec![0.0; k], |m| m.proba(&z[j * d..(j + 1) * d], k))).collect()
            });
            (lr, probs)
        })
        .collect();
    f.readout = Some(Readout::Shape);
    let mut oof: Option<Oof> = None;
    let mut final_lr = None;
    for (fold, (m, probs)) in jobs.iter().zip(models) {
        match fold {
            None => final_lr = m,
            Some(i) => {
                let o = oof.get_or_insert_with(|| vec![vec![0.0; k]; n]);
                for (&j, p) in folds[*i].1.iter().zip(probs) {
                    o[j] = p;
                }
            }
        }
    }
    f.pool = final_lr.map(|lr| PoolModel { std, lr });
    f.report.as_object_mut().unwrap().insert("fit_secs".into(), json!({"shape": t.elapsed().as_secs_f64()}));
    oof
}

/// The `blend` understudent: one sparse+dense logistic regression over [hashed n-grams] ++ [structural features] ++
/// [input embeddings' mean and max], dense blocks normalized to unit norm per row (`shape::blend_row`), `keywords`'s
/// solver and C. Each fold's two dense standardizers are fitted on that fold's TRAINING rows only; the final model's
/// (`f.blend`) on all rows.
fn fit_stack(f: &mut Fit, fz: &Featurizer, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize], w: &[f64], k: usize) -> Option<Oof> {
    let t = Instant::now();
    let n = encs.len();
    let fd = shape::DIM;
    let fx: Vec<f32> = encs.iter().flat_map(|e| e.shape.as_ref().expect("blend needs features").iter().copied()).collect();
    let embs: Vec<Vec<f32>> = encs.par_iter().map(|e| fz.pool_features(e)).collect();
    let de = embs[0].len();
    let ex: Vec<f32> = embs.iter().flatten().copied().collect();
    let dim = shape::blend_dim(fd, de);
    let all: Vec<usize> = (0..n).collect();
    let fit_std = |rows: &[usize]| -> BlendStd {
        let fxs: Vec<f32> = rows.iter().flat_map(|&j| fx[j * fd..(j + 1) * fd].iter().copied()).collect();
        let exs: Vec<f32> = rows.iter().flat_map(|&j| ex[j * de..(j + 1) * de].iter().copied()).collect();
        BlendStd { shape: Standardizer::fit(&fxs, fd), emb: Standardizer::fit(&exs, de) }
    };
    let final_std = fit_std(&all);
    let jobs: Vec<Option<usize>> = std::iter::once(None).chain((0..folds.len()).map(Some)).collect();
    let models: Vec<(Option<SparseLr>, Vec<Vec<f64>>)> = jobs
        .par_iter()
        .map(|&fold| {
            let (st, train) = match fold {
                None => (final_std.clone(), &all),
                Some(i) => (fit_std(&folds[i].0), &folds[i].0),
            };
            let rows: Vec<keywords::Sparse> = (0..n)
                .into_par_iter()
                .map(|i| shape::blend_row(encs[i].sparse.as_ref().expect("blend needs hashed features"), encs[i].shape.as_ref().unwrap(), &embs[i], &st))
                .collect();
            let xs: Vec<&keywords::Sparse> = rows.iter().collect();
            let m = keywords::fit_dim(&xs, train, y, w, keywords::C, dim);
            let probs = fold.map_or_else(Vec::new, |i| {
                folds[i].1.iter().map(|&j| m.as_ref().map_or_else(|| vec![0.0; k], |lr| lr.proba(xs[j], k))).collect()
            });
            (m, probs)
        })
        .collect();
    f.readout = Some(Readout::Blend);
    let mut oof: Option<Oof> = None;
    for (fold, (m, probs)) in jobs.iter().zip(models) {
        match fold {
            None => f.sparse = m,
            Some(i) => {
                let o = oof.get_or_insert_with(|| vec![vec![0.0; k]; n]);
                for (&j, p) in folds[*i].1.iter().zip(probs) {
                    o[j] = p;
                }
            }
        }
    }
    f.blend = Some(final_std);
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("fit_secs".into(), json!({"blend": t.elapsed().as_secs_f64()}));
    rep.insert("blend_dim".into(), json!(dim));
    rep.insert("keywords_iters".into(), json!(f.sparse.as_ref().map(|m| m.iters)));
    oof
}

/// The `net` understudent: a one-hidden-layer network (`netdesign.rs`) over [deep's pooled encoder features] ++ [`shape`'s
/// features]. Each job splits its training rows into fitting and early-stopping rows FIRST; its two standardizers are
/// fitted on the fitting rows only (never the early-stopping rows), as is the network; the final model's job runs on all rows. One single-threaded, seeded training per job.
#[allow(clippy::too_many_arguments)]
fn fit_net(f: &mut Fit, fz: &Featurizer, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], groups: &[Option<u64>], y: &[usize], w: &[f64], k: usize,
           s: &FitSettings) -> Option<Oof> {
    let t = Instant::now();
    let n = encs.len();
    let dp = fz.pool_dim();
    let df = shape::DIM;
    let d = dp + df;
    let pool: Vec<f32> = encs.par_iter().flat_map_iter(|e| fz.pool_features(e)).collect();
    let fx: Vec<f32> = encs.iter().flat_map(|e| e.shape.as_ref().expect("net needs features").iter().copied()).collect();
    let all: Vec<usize> = (0..n).collect();
    let jobs: Vec<Option<usize>> = std::iter::once(None).chain((0..folds.len()).map(Some)).collect();
    let models: Vec<(Option<NetModel>, Vec<Vec<f64>>, f64)> = jobs
        .par_iter()
        .map(|&fold| {
            let t0 = Instant::now();
            let train = fold.map_or(&all, |i| &folds[i].0);
            let seed = s.seed.wrapping_mul(1000).wrapping_add(fold.map_or(0, |i| i as u64 + 1));
            let (fit_rows, val_rows) = netdesign::inner_split(train, groups, y, seed);
            let gather = |src: &[f32], dim: usize| -> Vec<f32> { fit_rows.iter().flat_map(|&j| src[j * dim..(j + 1) * dim].iter().copied()).collect() };
            // split FIRST: the standardizers see the fitting rows only, never the early-stopping rows
            let (ps, fs) = (Standardizer::fit(&gather(&pool, dp), dp), Standardizer::fit(&gather(&fx, df), df));
            let mut z = vec![0f32; n * d];
            z.par_chunks_mut(d).enumerate().for_each(|(j, o)| o.copy_from_slice(&netdesign::input(&ps, &fs, &pool[j * dp..(j + 1) * dp], &fx[j * df..(j + 1) * df])));
            let model = netdesign::train_split(&z, d, train, fit_rows, val_rows, y, w, seed)
                .map(|t| NetModel { pool_std: ps, shape_std: fs, h: t.h, w1: t.w1, b1: t.b1, w2: t.w2, b2: t.b2, classes: t.classes, epochs: t.epochs });
            let probs = fold.map_or_else(Vec::new, |i| {
                folds[i].1.iter().map(|&j| model.as_ref().map_or_else(|| vec![0.0; k], |m| m.proba(&pool[j * dp..(j + 1) * dp], &fx[j * df..(j + 1) * df], k))).collect()
            });
            (model, probs, t0.elapsed().as_secs_f64())
        })
        .collect();
    f.readout = Some(Readout::Net);
    let mut oof: Option<Oof> = None;
    let mut epochs = vec![];
    for (fold, (m, probs, _)) in jobs.iter().zip(models) {
        match fold {
            None => {
                epochs.push(m.as_ref().map(|m| m.epochs));
                f.net = m;
            }
            Some(i) => {
                epochs.push(m.as_ref().map(|m| m.epochs));
                let o = oof.get_or_insert_with(|| vec![vec![0.0; k]; n]);
                for (&j, p) in folds[*i].1.iter().zip(probs) {
                    o[j] = p;
                }
            }
        }
    }
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("fit_secs".into(), json!({"net": t.elapsed().as_secs_f64()}));
    rep.insert("net_epochs".into(), json!(epochs));
    rep.insert("net_dim".into(), json!(d));
    oof
}

/// The `checklist` understudent: a greedily induced decision list per job (`checklist::induce`) over the binary features of
/// [`checklist::row_ids`]; out-of-fold, a row is answered by the first rule that fires in its fold's list and is an
/// abstention (a NaN row) when none does. The final list's rules are named (hashed n-grams by their text, from the
/// training states' texts when the encodings carry them) and printed into the report's `listing`.
fn fit_checklist(f: &mut Fit, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize], k: usize, qtype: QType) -> Option<Oof> {
    let t = Instant::now();
    let n = encs.len();
    let ids: Vec<Vec<u32>> = encs
        .par_iter()
        .map(|e| checklist::row_ids(e.sparse.as_ref().expect("checklist needs hashed features"), e.shape.as_ref().expect("checklist needs features")))
        .collect();
    let all: Vec<usize> = (0..n).collect();
    let jobs: Vec<Option<usize>> = std::iter::once(None).chain((0..folds.len()).map(Some)).collect();
    let lists: Vec<Checklist> = jobs.par_iter().map(|&fold| checklist::induce(&ids, fold.map_or(&all, |i| &folds[i].0), y, k)).collect();
    f.readout = Some(Readout::Checklist);
    let mut oof: Option<Oof> = None;
    let mut final_list = Checklist::default();
    for (fold, list) in jobs.iter().zip(lists) {
        match fold {
            None => final_list = list,
            Some(i) => {
                let o = oof.get_or_insert_with(|| vec![vec![f64::NAN; k]; n]);
                for &j in &folds[*i].1 {
                    if let Some(p) = list.proba(&ids[j], k) {
                        o[j] = p;
                    }
                }
            }
        }
    }
    let auxes: Vec<&crate::encoder::Aux> = encs.iter().filter_map(|e| e.aux.as_deref()).collect();
    let states: Vec<&[String]> = auxes.iter().map(|a| a.texts.as_slice()).collect();
    checklist::name_rules(&mut final_list, &states);
    // a structural slot is named by the field that most often holds it
    for name in final_list.names.iter_mut() {
        if let Some(rest) = name.strip_prefix("slot") {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(sl) = digits.parse::<usize>() {
                let mut seen: std::collections::BTreeMap<&str, usize> = Default::default();
                for a in &auxes {
                    if let Some(nm) = a.slots.get(sl).filter(|s| !s.is_empty()) {
                        *seen.entry(nm.as_str()).or_default() += 1;
                    }
                }
                if let Some((nm, _)) = seen.into_iter().max_by_key(|(nm, c)| (*c, std::cmp::Reverse(*nm))) {
                    *name = format!("[{nm}]{}", &rest[digits.len()..]);
                }
            }
        }
    }
    let class_names: Vec<String> = if qtype == QType::Noul { vec!["no".into(), "yes".into()] } else { (0..k).map(|c| format!("class {c}")).collect() };
    let listing = final_list.describe(&class_names);
    let rules: Vec<Value> = final_list
        .rules
        .iter()
        .zip(&final_list.names)
        .map(|(r, name)| json!({"feature": name, "class": class_names.get(r.class), "confidence": r.conf, "rows": r.n, "hits": r.hits}))
        .collect();
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("fit_secs".into(), json!({"checklist": t.elapsed().as_secs_f64()}));
    rep.insert("checklist".into(), json!({"rules": rules, "default": class_names.get(final_list.default), "listing": listing}));
    f.checklist = Some(final_list);
    oof
}

/// Graduation over out-of-fold predictions, as `fit_understudent` does it (an abstention is a NaN row).
fn graduate_oof(p: &Oof, y: &[usize], qtype: QType, k: usize, s: &FitSettings) -> (f64, Option<Mode>, f64) {
    let pred: Vec<usize> = p.iter().map(|x| argmax(x)).collect();
    let cf: Vec<f64> = p.iter().map(|x| x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).collect();
    let agree: Vec<bool> = pred.iter().zip(y).map(|(a, b)| a == b).collect();
    let mut counts = vec![0usize; k];
    y.iter().for_each(|&c| counts[c] += 1);
    let maj = argmax(&counts.iter().map(|&c| c as f64).collect::<Vec<_>>());
    let minor: Vec<bool> = y.iter().map(|&c| c != maj).collect();
    let p_yes: Option<Vec<f64>> = (qtype == QType::Noul).then(|| p.iter().map(|x| x[1]).collect());
    graduation_z(&cf, &agree, &minor, p_yes.as_deref(), y, s.target, s.minority, s.grad_z())
}

/// The `memo` understudent: the stored keys and lead answers, and the similarity threshold the fit's own CV picks
/// (`memo::GRID`: the value whose answers graduate with the most coverage, ties to the stricter; else
/// `memo::DEFAULT_SIM`). Out-of-fold, a row is answered from the other folds' rows alone; an abstention is a NaN row.
///
/// The threshold is chosen by NESTED grouped CV: for each outer fold, the threshold is picked from inner folds
/// within that fold's training rows only (`cv_folds` again), and the outer fold's test rows are answered at that
/// threshold, so the returned out-of-fold predictions (which graduation certifies) come from a threshold chosen
/// without the evaluated rows. The deployed threshold (`model.thr_sim`) is the same scan over all outer
/// out-of-fold rows (`memo_sim_grid`); the per-fold nested picks are reported as `memo_sim_nested`.
#[allow(clippy::too_many_arguments)]
fn fit_memo(f: &mut Fit, encs: &[&Encoding], folds: &[(Vec<usize>, Vec<usize>)], groups: &[Option<u64>], y: &[usize], w: &[f64], k: usize, qtype: QType,
            s: &FitSettings) -> Option<Oof> {
    let t = Instant::now();
    let n = encs.len();
    let keys: Vec<&memo::MemoKey> = encs.iter().map(|e| e.memo.as_deref().expect("memo needs keys")).collect();
    let mut model = MemoModel::new(&keys, y, w, memo::DEFAULT_SIM);
    f.readout = Some(Readout::Memo);
    let mut oof = None;
    let mut tried = vec![];
    let mut nested = vec![];
    if !folds.is_empty() {
        let answer = |nb: &[(f64, usize)], thr: f64| model.answer(nb, thr, k).unwrap_or_else(|| vec![f64::NAN; k]);
        let mut neigh: Vec<Vec<(f64, usize)>> = vec![vec![]; n];
        for (train, test) in folds {
            let found: Vec<Vec<(f64, usize)>> = test.par_iter().map(|&j| model.neighbours(keys[j], memo::MIN_SIM, Some(train))).collect();
            for (&j, v) in test.iter().zip(found) {
                neigh[j] = v;
            }
        }
        // the threshold whose out-of-fold answers over `rows` (their neighbours from `nb`) graduate with the most coverage
        let pick = |rows: &[usize], nb: &dyn Fn(usize) -> Vec<(f64, usize)>, record: &mut Vec<Value>| -> f64 {
            let ys: Vec<usize> = rows.iter().map(|&j| y[j]).collect();
            let cache: Vec<Vec<(f64, usize)>> = rows.iter().map(|&j| nb(j)).collect();
            let mut best: Option<(f64, f64)> = None;
            for thr in memo::GRID {
                let p: Oof = cache.iter().map(|c| answer(c, thr)).collect();
                let (cov, mode, _) = graduate_oof(&p, &ys, qtype, k, s);
                record.push(json!({"sim": thr, "coverage": cov, "graduated": mode.is_some()}));
                if mode.is_some() && best.is_none_or(|b| cov >= b.0) {
                    best = Some((cov, thr));
                }
            }
            best.map_or(memo::DEFAULT_SIM, |b| b.1)
        };
        let mut o: Oof = vec![vec![f64::NAN; k]; n];
        for (fi, (train, test)) in folds.iter().enumerate() {
            // inner folds over this fold's training rows alone
            let sub: Vec<Option<u64>> = train.iter().map(|&j| groups[j]).collect();
            let chosen = match cv_folds(&sub, (s.seed as u32).wrapping_add(1 + fi as u32)) {
                Some(inner) => {
                    let mut inner_neigh: std::collections::HashMap<usize, Vec<(f64, usize)>> = Default::default();
                    for (itr, ite) in &inner {
                        let itr_g: Vec<usize> = itr.iter().map(|&a| train[a]).collect();
                        let ite_g: Vec<usize> = ite.iter().map(|&a| train[a]).collect();
                        let found: Vec<Vec<(f64, usize)>> = ite_g.par_iter().map(|&j| model.neighbours(keys[j], memo::MIN_SIM, Some(&itr_g))).collect();
                        inner_neigh.extend(ite_g.into_iter().zip(found));
                    }
                    pick(train, &|j| inner_neigh[&j].clone(), &mut vec![])
                }
                None => memo::DEFAULT_SIM,
            };
            nested.push(chosen);
            for &j in test {
                o[j] = answer(&neigh[j], chosen);
            }
        }
        // the deployed threshold: the same scan over every outer out-of-fold row
        let all: Vec<usize> = (0..n).collect();
        let chosen = pick(&all, &|j| neigh[j].clone(), &mut tried);
        oof = Some(o);
        model.thr_sim = chosen;
    }
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("memo_sim".into(), json!(model.thr_sim));
    rep.insert("memo_sim_grid".into(), json!(tried));
    rep.insert("memo_sim_nested".into(), json!(nested));
    rep.insert("fit_secs".into(), json!({"memo": t.elapsed().as_secs_f64()}));
    f.memo = Some(model);
    oof
}

/// One audit: whether the understudent agreed with the lead, whether it missed a "yes" (one-sided), and the mode it
/// answered in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Audit {
    pub agreed: bool,
    pub missed: bool,
    pub mode: Mode,
}

/// A prediction and what it allows.
#[derive(Clone, Debug)]
pub struct Pred {
    /// None before any fit with a model.
    pub p: Option<Vec<f64>>,
    /// Graduated, confident per its mode, and not novel.
    pub eligible: bool,
    pub novel: bool,
    /// Eligible and not demoted.
    pub local: bool,
}

/// The serving half of an understudent: its current fit, its audit window and demotion.
#[derive(Clone, Debug, Default)]
pub struct Serving {
    pub fit: Arc<Fit>,
    pub audits: VecDeque<Audit>,
    pub demoted: bool,
}

impl Serving {
    pub fn mode(&self) -> Option<Mode> {
        self.fit.mode
    }

    /// Swap in a new fit. The audit window carries over while the mode stays; a fit answering in another mode
    /// starts an empty window.
    pub fn install(&mut self, fit: Fit, target: f64) {
        self.fit = Arc::new(fit);
        if let Some(m) = self.mode() {
            if self.audits.iter().any(|a| a.mode != m) {
                self.audits.clear();
            }
        }
        self.demoted = self.failing(target);
    }

    pub fn predict(&self, fz: &Featurizer, e: &Encoding, k: usize) -> Pred {
        let f = &self.fit;
        let p = f.proba(fz, e, k);
        let novel = p.is_some() && f.novel(fz, e);
        let (Some(mode), Some(pv)) = (f.mode, p.as_ref()) else {
            return Pred { p, eligible: false, novel, local: false };
        };
        let eligible = !novel
            && match mode {
                Mode::OneSided => pv[1] < f.thr,
                Mode::OneSidedYes => pv[1] > f.thr,
                Mode::Symmetric => {
                    let top = pv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    match &f.class_thr {
                        Some(t) => top >= t[argmax(pv)],
                        None => top >= f.thr,
                    }
                }
            };
        Pred { p, eligible, novel, local: eligible && !self.demoted }
    }

    /// Record the lead's answer to a state this understudent would have answered locally.
    pub fn audit(&mut self, predicted: usize, answer: usize, target: f64) -> bool {
        let Some(mode) = self.mode() else { return false };
        if self.audits.len() == AUDIT_WINDOW {
            self.audits.pop_front();
        }
        self.audits.push_back(Audit { agreed: predicted == answer, missed: match mode {
            Mode::OneSided => answer == 1,
            Mode::OneSidedYes => answer == 0,
            Mode::Symmetric => false,
        }, mode });
        self.demoted = self.failing(target);
        true
    }

    fn window(&self) -> Vec<Audit> {
        let m = self.mode();
        self.audits.iter().filter(|a| Some(a.mode) == m).copied().collect()
    }

    fn failing(&self, target: f64) -> bool {
        let a = self.window();
        match self.mode() {
            Some(_) if a.len() < AUDIT_MIN => false,
            None => false,
            Some(Mode::OneSided | Mode::OneSidedYes) => wilson(a.iter().filter(|x| x.missed).count(), a.len()).0 > MISS_RATE,
            Some(Mode::Symmetric) => wilson(a.iter().filter(|x| x.agreed).count(), a.len()).1 < target,
        }
    }

    pub fn audit_summary(&self) -> Value {
        let a = self.window();
        json!({
            "n": a.len(),
            "agreement": if a.is_empty() { Value::Null } else { json!(a.iter().filter(|x| x.agreed).count() as f64 / a.len() as f64) },
            "misses": if matches!(self.mode(), Some(Mode::OneSided | Mode::OneSidedYes)) { json!(a.iter().filter(|x| x.missed).count()) } else { Value::Null },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Part;
    use crate::rng::Xoshiro;
    use std::sync::Arc;

    fn shape_encs(n: usize, seed: u64) -> (Vec<Encoding>, Vec<usize>) {
        shape_encs_sep(n, seed, 1.5)
    }

    /// `n` rows whose first structural feature separates the two classes by `shift` on each side.
    fn shape_encs_sep(n: usize, seed: u64, shift: f64) -> (Vec<Encoding>, Vec<usize>) {
        let mut r = Xoshiro::new(seed);
        let mut y = vec![];
        let encs = (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..shape::DIM).map(|_| r.normal() as f32).collect();
                let c = usize::from(r.uniform() < 0.5);
                v[0] += if c == 1 { shift as f32 } else { -(shift as f32) };
                y.push(c);
                let sp = keywords::featurize(&[format!("row {}", y.len())]);
                let pool: Vec<f32> = (0..8).map(|_| r.normal() as f32).collect();
                Encoding {
                    shape: Some(Arc::new(v)),
                    sparse: Some(sp),
                    parts: vec![Part { pool, tokens: 1, states: vec![] }],
                    ..Default::default()
                }
            })
            .collect();
        (encs, y)
    }

    fn perturbed(encs: &[Encoding], j: usize) -> Vec<Encoding> {
        let mut out = encs.to_vec();
        let mut v = out[j].shape.as_ref().unwrap().to_vec();
        v[3] = 1e7;
        v[5] = -1e7;
        out[j].shape = Some(Arc::new(v));
        out
    }

    /// A held-out row's features must not reach the preprocessing of the model that predicts its fold: with the
    /// standardizer fitted on training rows only, the other held-out rows' predictions ignore row j entirely.
    fn assert_fold_ignores_held_out_row(fit: impl Fn(&[Encoding], &[(Vec<usize>, Vec<usize>)], &[usize]) -> Oof) {
        let (encs, y) = shape_encs(120, 4);
        let folds = kfold(120, FOLDS, 0);
        let j = folds[0].1[0];
        let clean = fit(&encs, &folds, &y);
        let dirty = fit(&perturbed(&encs, j), &folds, &y);
        for &i in &folds[0].1 {
            if i != j {
                assert_eq!(clean[i], dirty[i], "row {i} saw held-out row {j}'s features through the preprocessing");
            }
        }
        // (and the leak would be visible: a row of another fold, which trained on j, does change)
        let other = folds[1].1[0];
        assert_ne!(clean[other], dirty[other], "the outlier must matter where it is a training row");
    }

    #[test]
    fn shape_oof_standardizer_is_fitted_on_training_rows_only() {
        assert_fold_ignores_held_out_row(|encs, folds, y| {
            let refs: Vec<&Encoding> = encs.iter().collect();
            let x: Vec<f32> = refs.iter().flat_map(|e| e.shape.as_ref().unwrap().iter().copied()).collect();
            let std = Standardizer::fit(&x, shape::DIM);
            let w = vec![2.0; y.len()];
            let mut f = Fit { report: json!({}), ..Default::default() };
            fit_shape(&mut f, &refs, folds, y, &w, 2, std).unwrap()
        });
    }

    #[test]
    fn blend_oof_standardizers_are_fitted_on_training_rows_only() {
        let fz = Featurizer::for_understudent(UnderstudentKind::Blend, None, None, None, 4).unwrap();
        assert_fold_ignores_held_out_row(|encs, folds, y| {
            let refs: Vec<&Encoding> = encs.iter().collect();
            let w = vec![2.0; y.len()];
            let mut f = Fit { report: json!({}), ..Default::default() };
            fit_stack(&mut f, &fz, &refs, folds, y, &w, 2).unwrap()
        });
    }

    #[test]
    fn grouped_folds_use_fewer_folds_or_decline_never_plain_kfold() {
        let rows = 60;
        let ungrouped = vec![None; rows];
        assert_eq!(cv_folds(&ungrouped, 7).unwrap(), kfold(rows, FOLDS, 7), "no groups: byte-identical to plain kfold");
        // 3 groups of 20 siblings, interleaved: k = 3 grouped folds, no group split
        let g3: Vec<Option<u64>> = (0..rows).map(|i| Some((i % 3) as u64)).collect();
        let folds = cv_folds(&g3, 0).unwrap();
        assert_eq!(folds.len(), 3);
        for (tr, te) in &folds {
            for g in 0..3u64 {
                let in_test = te.iter().filter(|&&i| g3[i] == Some(g)).count();
                let in_train = tr.iter().filter(|&&i| g3[i] == Some(g)).count();
                assert!(in_test == 0 || in_train == 0, "group {g} straddles a split");
            }
        }
        // 2 groups: 2 folds; 1 group: decline
        let g2: Vec<Option<u64>> = (0..rows).map(|i| Some((i % 2) as u64)).collect();
        assert_eq!(cv_folds(&g2, 0).unwrap().len(), 2);
        assert!(cv_folds(&vec![Some(1); rows], 0).is_none());
        // 5+ groups: unchanged group_kfold with FOLDS folds
        let g9: Vec<Option<u64>> = (0..rows).map(|i| Some((i % 9) as u64)).collect();
        assert_eq!(cv_folds(&g9, 3).unwrap(), group_kfold(&g9, FOLDS, 3));
        // a single group beside ungrouped rows still has other units to hold out
        let mixed: Vec<Option<u64>> = (0..rows).map(|i| if i < 30 { Some(1) } else { None }).collect();
        assert_eq!(cv_folds(&mixed, 0).unwrap().len(), FOLDS);
    }

    fn memo_inputs(n: usize, groups: impl Fn(usize) -> Option<u64>) -> (Vec<Encoding>, Vec<usize>, Vec<Option<u64>>) {
        let mut r = Xoshiro::new(2);
        let bases: Vec<String> = (0..6).map(|t| format!("template {t}: {}", (0..30).map(|_| format!("w{}", r.next_u64() % 400)).collect::<Vec<_>>().join(" "))).collect();
        let mut y = vec![];
        let encs = (0..n)
            .map(|i| {
                let t = (r.next_u64() % 6) as usize;
                y.push(t % 2);
                Encoding { memo: Some(Arc::new(memo::MemoKey::from_text(&format!("{} (seen {i})", bases[t])))), ..Default::default() }
            })
            .collect();
        (encs, y, (0..n).map(groups).collect())
    }

    #[test]
    fn memo_threshold_is_chosen_without_the_evaluated_rows() {
        let n = 200;
        let (encs, y, groups) = memo_inputs(n, |_| None);
        let folds = cv_folds(&groups, 0).unwrap();
        let s = crate::proxy::Settings::default().fit;
        let run = |y: &[usize]| {
            let refs: Vec<&Encoding> = encs.iter().collect();
            let w = vec![2.0; n];
            let mut f = Fit { report: json!({}), ..Default::default() };
            let oof = fit_memo(&mut f, &refs, &folds, &groups, y, &w, 2, QType::Noul, &s).unwrap();
            (f.report["memo_sim_nested"].clone(), oof)
        };
        let (nested, oof) = run(&y);
        assert_eq!(nested.as_array().unwrap().len(), FOLDS, "one nested pick per outer fold");
        // flip the labels of fold 0's evaluated rows: fold 0's threshold, and every row's out-of-fold answer there, cannot change
        let mut flipped = y.clone();
        for &j in &folds[0].1 {
            flipped[j] = 1 - flipped[j];
        }
        let (nested2, oof2) = run(&flipped);
        assert_eq!(nested[0], nested2[0], "fold 0's threshold depended on its own evaluated labels");
        for &j in &folds[0].1 {
            assert_eq!(oof[j].iter().map(|x| x.is_nan()).collect::<Vec<_>>(), oof2[j].iter().map(|x| x.is_nan()).collect::<Vec<_>>());
            if !oof[j][0].is_nan() {
                assert_eq!(oof[j], oof2[j]);
            }
        }
    }

    #[test]
    fn memo_inner_folds_never_split_groups() {
        // 3 groups, each with its own template family: no sibling is ever a held-out row's neighbour, so memo answers nothing
        let n = 150;
        let (encs, y, groups) = memo_inputs(n, |i| Some((i % 3) as u64));
        let folds = cv_folds(&groups, 0).unwrap();
        assert_eq!(folds.len(), 3);
        let refs: Vec<&Encoding> = encs.iter().collect();
        let s = crate::proxy::Settings::default().fit;
        let mut f = Fit { report: json!({}), ..Default::default() };
        let w = vec![2.0; n];
        let oof = fit_memo(&mut f, &refs, &folds, &groups, &y, &w, 2, QType::Noul, &s).unwrap();
        assert_eq!(oof.len(), n);
        // each outer fold's inner split has 2 groups -> 2 inner folds, no panic and a threshold per outer fold
        assert_eq!(f.report["memo_sim_nested"].as_array().unwrap().len(), 3);
    }

    fn bits(o: &Oof) -> Vec<Vec<u64>> {
        o.iter().map(|r| r.iter().map(|x| x.to_bits()).collect()).collect()
    }

    fn net_fz() -> Featurizer {
        // one layer, hidden 4: a mean+max pool of 8 dims, as `shape_encs` builds
        Featurizer::for_understudent(UnderstudentKind::Net, Some(vec![4]), None, None, 4).unwrap()
    }

    fn run_net(encs: &[Encoding], folds: &[(Vec<usize>, Vec<usize>)], y: &[usize]) -> (Oof, Fit) {
        let refs: Vec<&Encoding> = encs.iter().collect();
        let w = vec![2.0; y.len()];
        let groups = vec![None; y.len()];
        let s = crate::proxy::Settings::default().fit;
        let mut f = Fit { report: json!({}), ..Default::default() };
        let oof = fit_net(&mut f, &net_fz(), &refs, folds, &groups, y, &w, 2, &s).unwrap();
        (oof, f)
    }

    #[test]
    fn net_oof_preprocessing_is_fitted_on_training_rows_only() {
        // a held-out row's structural features (an outlier in the standardizer's statistics)...
        assert_fold_ignores_held_out_row(|encs, folds, y| run_net(encs, folds, y).0);
        // ...and its pooled encoder features
        let (encs, y) = shape_encs(120, 4);
        let folds = kfold(120, FOLDS, 0);
        let j = folds[0].1[0];
        let mut dirty = encs.clone();
        dirty[j].parts[0].pool[2] = 1e7;
        let (clean, noisy) = (run_net(&encs, &folds, &y).0, run_net(&dirty, &folds, &y).0);
        for &i in &folds[0].1 {
            if i != j {
                assert_eq!(bits(&vec![clean[i].clone()]), bits(&vec![noisy[i].clone()]), "row {i} saw held-out row {j}'s pooled features");
            }
        }
    }

    #[test]
    fn net_and_checklist_ignore_the_held_out_labels() {
        // flipping the labels of fold 0's evaluated rows cannot change a prediction made for fold 0
        let (encs, y) = shape_encs(150, 9);
        let folds = kfold(150, FOLDS, 0);
        let mut flipped = y.clone();
        folds[0].1.iter().for_each(|&j| flipped[j] = 1 - flipped[j]);
        let (a, b) = (run_net(&encs, &folds, &y).0, run_net(&encs, &folds, &flipped).0);
        for &j in &folds[0].1 {
            assert_eq!(bits(&vec![a[j].clone()]), bits(&vec![b[j].clone()]), "net row {j}");
        }
        let refs: Vec<&Encoding> = encs.iter().collect();
        let dl = |y: &[usize]| {
            let mut f = Fit { report: json!({}), ..Default::default() };
            fit_checklist(&mut f, &refs, &folds, y, 2, QType::Noul).unwrap()
        };
        let (a, b) = (dl(&y), dl(&flipped));
        for &j in &folds[0].1 {
            assert_eq!(bits(&vec![a[j].clone()]), bits(&vec![b[j].clone()]), "checklist row {j}");
        }
        // and the other folds do see them (the flip is not a no-op)
        let other = folds[1].1.iter().any(|&j| bits(&vec![dl(&y)[j].clone()]) != bits(&vec![dl(&flipped)[j].clone()]));
        assert!(other, "the flipped labels must matter where they are training rows");
    }

    #[test]
    fn net_fit_is_deterministic_and_bounded() {
        let (encs, y) = shape_encs(150, 3);
        let folds = kfold(150, FOLDS, 0);
        let t = Instant::now();
        let (a, fa) = run_net(&encs, &folds, &y);
        let secs = t.elapsed().as_secs_f64();
        let (b, fb) = run_net(&encs, &folds, &y);
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(fa.net, fb.net, "same data, same network");
        assert!(secs < 30.0, "a 150-row fit took {secs}s");
        assert!(fa.net.as_ref().unwrap().epochs >= 1);
    }

    fn cal_settings(kind: UnderstudentKind) -> FitSettings {
        let mut s = crate::proxy::Settings::default().fit;
        s.understudent = kind;
        s.abstain = Abstain::Calibrated;
        s
    }

    fn paired_rows<'a>(encs: &'a [Encoding], y: &[usize]) -> Vec<Row<'a>> {
        encs.iter().zip(y).enumerate().map(|(i, (e, &y))| Row { enc: e, y, w: 0.95, group: Some((i / 2) as u64) }).collect()
    }

    /// The certificate is a function of the frozen model and the CERTIFY rows alone: relabelling the CERTIFY rows changes
    /// the verdict but not the model, the novelty set or the thresholds; relabelling SELECT leaves the model; relabelling
    /// TRAIN changes the model but never which rows are certified.
    #[test]
    fn certification_uses_only_the_certify_rows() {
        let fz = Featurizer::for_understudent(UnderstudentKind::Shape, None, None, None, 4).unwrap();
        let (encs, y) = shape_encs_sep(900, 11, 8.0);
        let s = cal_settings(UnderstudentKind::Shape);
        let groups: Vec<Option<u64>> = (0..900).map(|i| Some((i / 2) as u64)).collect();
        let split = abstain::three_way(&groups, s.seed).unwrap();
        let fit = |labels: &[usize]| fit_understudent(&fz, QType::Noul, 2, &paired_rows(&encs, labels), &s);
        let base = fit(&y);
        assert_eq!(base.mode, Some(Mode::Symmetric), "separable data graduates: {}", base.report);
        assert_eq!(base.report["certificate"]["rows"], split.certify.len());
        assert_eq!(base.report["select"]["rows"], split.select.len());
        // the certificate is the exact serving predicate over the CERTIFY rows
        let sv = Serving { fit: Arc::new(base.clone()), ..Default::default() };
        let answered = split.certify.iter().filter(|&&j| sv.predict(&fz, &encs[j], 2).eligible).count();
        let agreed = split.certify.iter().filter(|&&j| sv.predict(&fz, &encs[j], 2).eligible && argmax(&sv.fit.proba(&fz, &encs[j], 2).unwrap()) == y[j]).count();
        assert_eq!(base.report["certificate"]["answered"], answered);
        assert_eq!(base.report["certificate"]["agreed"], agreed);
        assert!(answered >= 30 && base.report["certificate"]["passed"] == true);

        let flip = |part: &[usize]| {
            let mut z = y.clone();
            part.iter().for_each(|&j| z[j] = 1 - z[j]);
            z
        };
        // CERTIFY labels: same model, same thresholds, same answered rows; the verdict follows the labels
        let c = fit(&flip(&split.certify));
        assert_eq!((&c.pool, &c.refs, c.nn_floor), (&base.pool, &base.refs, base.nn_floor), "CERTIFY labels reached the model");
        assert_eq!(c.report["select"]["thresholds"], base.report["select"]["thresholds"], "CERTIFY labels reached the thresholds");
        assert_eq!(c.report["certificate"]["answered"], base.report["certificate"]["answered"]);
        assert_ne!(c.report["certificate"]["agreed"], base.report["certificate"]["agreed"]);
        assert!(c.mode.is_none() && c.class_thr.is_none(), "a certificate on mislabelled rows must refuse: {}", c.report["certificate"]);
        // SELECT labels: the model is untouched (only the thresholds may move)
        let sel = fit(&flip(&split.select));
        assert_eq!((&sel.pool, &sel.refs, sel.nn_floor), (&base.pool, &base.refs, base.nn_floor), "SELECT labels reached the model");
        // TRAIN labels: the model changes, the certified rows do not
        let tr = fit(&flip(&split.train));
        assert_ne!(tr.pool, base.pool);
        assert_eq!(tr.report["split"]["certify"], split.certify.len());
        assert_eq!(tr.report["split"], base.report["split"]);
        // the served model is the TRAIN model: no refit on all rows (its standardizer saw 360 rows, not 600)
        assert_eq!(base.pool.as_ref().unwrap().std.mu.len(), shape::DIM);
        assert_eq!(base.report["split"]["train"], split.train.len());
        assert!(base.refs.len() / base.ref_dim <= split.train.len());
    }

    /// A class with no SELECT support gets an infinite threshold and is never answered.
    #[test]
    fn unsupported_classes_are_never_answered() {
        let fz = Featurizer::for_understudent(UnderstudentKind::Shape, None, None, None, 4).unwrap();
        let (encs, mut y) = shape_encs_sep(900, 3, 8.0);
        // a rare third answer on a handful of rows
        for j in (0..900).step_by(97) {
            y[j] = 2;
        }
        let s = cal_settings(UnderstudentKind::Shape);
        let f = fit_understudent(&fz, QType::Choice, 3, &paired_rows(&encs, &y), &s);
        let Some(thr) = f.class_thr.clone() else { panic!("the two common classes graduate: {}", f.report) };
        assert!(thr[2].is_infinite(), "{thr:?}");
        let sv = Serving { fit: Arc::new(f), ..Default::default() };
        for e in &encs {
            let p = sv.predict(&fz, e, 3);
            if p.eligible {
                assert_ne!(argmax(p.p.as_ref().unwrap()), 2);
            }
        }
        // the infinite threshold survives the snapshot's JSON as a null
        #[derive(Serialize, Deserialize)]
        struct T(#[serde(with = "crate::abstain::thr_serde")] Option<Vec<f64>>);
        let back: T = serde_json::from_str(&serde_json::to_string(&T(Some(thr.clone()))).unwrap()).unwrap();
        assert_eq!(back.0, Some(thr));
    }

    /// The default policy is the unchanged CV path: all rows, no certificate, no class thresholds.
    #[test]
    fn the_default_policy_is_unchanged() {
        let fz = Featurizer::for_understudent(UnderstudentKind::Shape, None, None, None, 4).unwrap();
        let (encs, y) = shape_encs_sep(400, 5, 6.0);
        let rows: Vec<Row> = encs.iter().zip(&y).map(|(e, &y)| Row { enc: e, y, w: 0.95, group: None }).collect();
        let mut s = crate::proxy::Settings::default().fit;
        s.understudent = UnderstudentKind::Shape;
        let a = fit_understudent(&fz, QType::Noul, 2, &rows, &s);
        let b = fit_understudent(&fz, QType::Noul, 2, &rows, &s);
        assert_eq!((&a.pool, a.mode, a.thr, &a.refs, a.nn_floor), (&b.pool, b.mode, b.thr, &b.refs, b.nn_floor));
        assert!(a.class_thr.is_none() && a.report.get("certificate").is_none() && a.report.get("split").is_none() && a.report.get("abstain").is_none());
        assert!(a.report["cv_agreement"].is_number() && a.report["cv_coverage"].is_number());
        // refit on every row: its standardizer and novelty set come from all 400, not a part
        assert_eq!(a.refs.len() / a.ref_dim, s.novelty_refs.min(400));
        // the default policy is the one the plain scan gives on its own out-of-fold predictions
        let (cov, mode, thr) = (a.report["cv_coverage"].as_f64().unwrap(), a.mode, a.thr);
        assert!(mode.is_some() && cov > 0.0 && (0.5..=0.995).contains(&thr));
        let mut c = s.clone();
        c.abstain = Abstain::Single;
        assert_eq!(fit_understudent(&fz, QType::Noul, 2, &rows, &c).thr, thr);
    }

    /// Too few groups to fill the three parts: calibrated cannot be certified, so it never graduates.
    #[test]
    fn calibrated_without_a_split_never_graduates() {
        let fz = Featurizer::for_understudent(UnderstudentKind::Shape, None, None, None, 4).unwrap();
        let (encs, y) = shape_encs_sep(300, 5, 5.0);
        let rows: Vec<Row> = encs.iter().zip(&y).enumerate().map(|(i, (e, &y))| Row { enc: e, y, w: 0.95, group: Some((i % 2) as u64) }).collect();
        let f = fit_understudent(&fz, QType::Noul, 2, &rows, &cal_settings(UnderstudentKind::Shape));
        assert!(f.mode.is_none() && f.class_thr.is_none(), "{}", f.report);
    }

    #[test]
    fn checklist_prints_its_rules_by_name() {
        // class 1 states carry the word "zzq"; the hashed n-grams that fire name themselves from the states' texts
        let mut r = Xoshiro::new(5);
        let (mut encs, mut y) = (vec![], vec![]);
        for i in 0..120usize {
            let c = usize::from(i % 2 == 1);
            let text = format!("row {} {}{}", r.next_u64() % 1000, if c == 1 { "zzq " } else { "" }, ["alpha", "beta", "gamma", "delta"][(r.next_u64() % 4) as usize]);
            let state = json!({"answer": text});
            let texts = vec![text];
            encs.push(Encoding {
                sparse: Some(keywords::featurize(&texts)),
                shape: Some(Arc::new(shape::features(&state, None))),
                aux: Some(Arc::new(crate::encoder::Aux { texts, slots: shape::slot_names(&state) })),
                ..Default::default()
            });
            y.push(c);
        }
        let refs: Vec<&Encoding> = encs.iter().collect();
        let folds = kfold(120, FOLDS, 0);
        let mut f = Fit { report: json!({}), ..Default::default() };
        let oof = fit_checklist(&mut f, &refs, &folds, &y, 2, QType::Noul).unwrap();
        let listing = f.report["checklist"]["listing"].as_str().unwrap();
        assert!(listing.contains("IF ") && listing.contains(" THEN yes"), "{listing}");
        assert!(listing.contains("zq") || listing.contains("zz"), "a rule should name the marker n-gram: {listing}");
        assert!(!listing.contains("gram#"), "known n-grams are printed as text: {listing}");
        assert!(listing.contains("default (abstain, never answers)"));
        // rows no rule fires on abstain
        let answered = oof.iter().filter(|p| !p[0].is_nan()).count();
        assert!(answered >= 60, "the marker rule answers the class-1 rows: {answered}");
        assert!(f.report["checklist"]["rules"].as_array().unwrap().iter().all(|r| r["feature"].is_string()));
        // a structural slot is named by the field that holds it
        let (state, texts) = (json!({"answer": "I'm sorry, I cannot help"}), vec!["I'm sorry, I cannot help".to_string()]);
        let e = Encoding { sparse: Some(keywords::featurize(&texts)), shape: Some(Arc::new(shape::features(&state, None))), ..Default::default() };
        let ids = checklist::row_ids(e.sparse.as_ref().unwrap(), e.shape.as_ref().unwrap());
        let names: Vec<String> = ids.iter().map(|&i| checklist::feature_name(i, None)).collect();
        assert!(names.iter().any(|n| n.ends_with(".refusal")) && names.iter().any(|n| n.ends_with(".apology")), "{names:?}");
    }
}
