//! The router, as hunch's `Proxy`: a request is answered locally when every question in it may be (graduated,
//! confident, not novel, not demoted), else forwarded whole to the lead, whose answers become examples and
//! audit every understudent that would have answered. A share of local answers is audited too. A question refits
//! (synchronously) once it has `refit_every` new examples.
//!
//! A question's understudent is a list of **tiers** (one for every understudent but the `ladder`). Every tier has its own fit,
//! graduation, audit window and demotion; a request is answered by the cheapest tier that may answer, and a tier's
//! features are computed only when a request reaches it (`Proxy::ensure`).

use crate::diskcache::DiskCache;
use crate::encoder::{Encoder, Encoding, Featurizer, Part, Resume, Tokens, state_hash, state_text, text_hash};
use crate::memo::MemoKey;
use crate::question::Question;
use crate::rng::Mt19937;
use crate::rules::{self, RuleInput, RuleRow, RulesClient, RulesCtx, RulesSettings, RulesState};
use crate::stats::argmax;
use crate::understudent::{Fit, FitSettings, Graduation, Mode, Pred, Row, Serving, UnderstudentKind, fit_understudent, readout_label};
use crate::lead::Lead;
use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Labelled,
    Sampled,
}

/// One example: a state (by hash), its class and the lead's probabilities, and its encoding.
#[derive(Clone, Debug)]
pub struct Example {
    pub seq: u64,
    pub kind: Kind,
    pub state: String,
    pub label: usize,
    pub probs: Option<Vec<f64>>,
    pub enc_key: String,
    pub enc: Arc<Encoding>,
    /// Byte length of the state's text (`state` itself is only a hash): the cost model's token estimate when the
    /// encoding recorded no token count. 0 for examples from an older snapshot.
    pub text_len: usize,
    /// Hash of the id's prefix before `Settings::group_sep` (e.g. `<item>` of `<item>#<k>`): examples sharing it never
    /// straddle CV folds. None for ids without the separator (or no id).
    pub group: Option<u64>,
}

impl Example {
    /// hunch's weight: 2.0 for a label, else the confidence (max probability; 1.0 for a bare class).
    pub fn weight(&self) -> f64 {
        match (self.kind, &self.probs) {
            (Kind::Labelled, _) => 2.0,
            (Kind::Sampled, Some(p)) => p.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            (Kind::Sampled, None) => 1.0,
        }
    }
}

/// Router settings (hunch's defaults).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub fit: FitSettings,
    pub audit_rate: f64,
    pub refit_every: usize,
    /// A ladder's cheap tiers (`keywords`, `lite`) refit this often (all tiers refit every `refit_every`).
    #[serde(default = "default_cheap_every")]
    pub cheap_refit_every: usize,
    pub sampled_cap: usize,
    pub labelled_cap: usize,
    /// `auto`: examples after which a question where no candidate graduated and none got within `GIVE_UP_MARGIN` of the
    /// target is declared `lead_only`.
    #[serde(default = "default_give_up")]
    pub give_up_after: usize,
    /// `auto`: the candidates' per-item costs in microseconds, in candidate order, instead of the measured ones (tests,
    /// reproducible replays). Empty: measure.
    #[serde(default)]
    pub cost_us: Vec<f64>,
    /// `auto --prefer-fast EPS`: among the graduating candidates serve the cheapest by the deterministic cost model
    /// (`model_cost_us`) whose coverage is within EPS (absolute) of the best, instead of the standard-error tie rule.
    /// None (the default) keeps the standard rule, and the snapshot unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefer_fast: Option<f64>,
    /// The LLM that writes the `rules` understudent's rules.
    #[serde(default)]
    pub rules: RulesSettings,
    /// Group examples by their id's prefix before this character (the per-response sub-items `<item>#<k>` of one item
    /// share `<item>`): cross-validation and the rules understudent's held-out split never separate a group. Ids without
    /// the character are ungrouped. None turns grouping off.
    #[serde(default = "default_group_sep")]
    pub group_sep: Option<char>,
}

fn default_group_sep() -> Option<char> {
    Some('#')
}

/// The group key of an id: a hash of everything before the last `sep`.
pub fn group_of(sep: Option<char>, id: Option<&str>) -> Option<u64> {
    let (prefix, _) = id?.rsplit_once(sep?)?;
    let h = Sha256::digest(prefix.as_bytes());
    Some(u64::from_be_bytes(h[..8].try_into().unwrap()))
}

fn default_cheap_every() -> usize {
    25
}

fn default_give_up() -> usize {
    600
}

/// `auto`: a `lead_only` question re-checks (one refit of every candidate) after this many new examples.
pub const RECHECK_EVERY: usize = 1000;
/// `auto`: no candidate within this of the target's agreement means the question is not learnable cheaply.
pub const GIVE_UP_MARGIN: f64 = 0.05;
/// `auto`: examples whose prediction time is measured at each refit.
const COST_SAMPLE: usize = 32;

/// One candidate of an `auto` race.
#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub graduated: bool,
    pub coverage: f64,
    /// Measured (or given) encode + predict microseconds per item; None when never measured.
    pub cost_us: Option<f64>,
}

/// The cost class two candidates must differ in to count as cheaper: decades of microseconds. Raw timings are noisy
/// and a replay must repeat, so 1.9 ms and 2.1 ms are the same class and the list order (cheapest first) decides.
fn cost_class(us: f64) -> i64 {
    us.max(1.0).log10().floor() as i64
}

/// The serving candidate: among the graduated, the highest `coverage`; candidates within one standard error of that
/// coverage proportion (over `n` examples) are tied and the cheaper cost class wins, then the earlier candidate. If a
/// tied candidate's cost is unknown the earlier (cheaper by list order) one wins. None when none graduated.
pub fn select_serving(c: &[Cand], n: usize) -> Option<usize> {
    let best = c.iter().filter(|x| x.graduated).map(|x| x.coverage).fold(f64::NEG_INFINITY, f64::max);
    if best == f64::NEG_INFINITY {
        return None;
    }
    let se = (best * (1.0 - best) / n.max(1) as f64).sqrt();
    let tied: Vec<usize> = (0..c.len()).filter(|&i| c[i].graduated && c[i].coverage >= best - se).collect();
    if tied.iter().all(|&i| c[i].cost_us.is_some()) {
        tied.iter().copied().min_by_key(|&i| (cost_class(c[i].cost_us.unwrap()), i))
    } else {
        tied.first().copied()
    }
}

/// `--prefer-fast`'s cost model: what one item costs a candidate, in microseconds, from the design alone and the
/// number of tokens it encodes, never from a clock, so the choice repeats in a replay. The cheap designs have a fixed
/// cost (their measured p50 in `runs/latency-jev`, rounded); an encoder design costs `MODEL_US_PER_TOKEN_LAYER` (about
/// 50 us on the benchmark machine: probe ran 37-69 ms for 88-150 tokens, deep 61-115 ms) per
/// token per encoder layer run (probe reads layer 8, deep and net run to their deepest layer, 12 by default) on top of
/// its fixed cost (the readout; net also runs the structural features).
pub const MODEL_US_PER_TOKEN_LAYER: f64 = 50.0;
/// Tokens assumed per item before any example has been encoded.
pub const MODEL_DEFAULT_TOKENS: f64 = 256.0;

/// The fixed part of a design's modelled cost, microseconds.
pub fn model_fixed_us(kind: UnderstudentKind) -> f64 {
    match kind {
        UnderstudentKind::Rules => 50.0,
        UnderstudentKind::Shape => 80.0,
        UnderstudentKind::Memo => 100.0,
        UnderstudentKind::Keywords => 130.0,
        UnderstudentKind::Checklist => 170.0,
        UnderstudentKind::Lite => 190.0,
        UnderstudentKind::Blend => 350.0,
        UnderstudentKind::Probe => 200.0,
        UnderstudentKind::Deep => 400.0,
        UnderstudentKind::Net => 600.0,
        _ => 0.0,
    }
}

/// `--prefer-fast`'s cost of one item: the fixed cost, plus for an encoder design `MODEL_US_PER_TOKEN_LAYER` x tokens
/// x the deepest layer it runs.
pub fn model_cost_us(kind: UnderstudentKind, deepest_layer: usize, tokens: f64) -> f64 {
    let enc = matches!(kind, UnderstudentKind::Probe | UnderstudentKind::Deep | UnderstudentKind::Net);
    model_fixed_us(kind) + if enc { MODEL_US_PER_TOKEN_LAYER * tokens * deepest_layer as f64 } else { 0.0 }
}

/// The `--prefer-fast` serving candidate: among the graduated (which met the agreement and minority bars on their own
/// out-of-fold predictions), those whose coverage is within `eps` of the best; the cheapest by `cost_us` wins, then the
/// earlier in the list. None when none graduated or a candidate in the band has no cost.
pub fn select_fastest(c: &[Cand], eps: f64) -> Option<usize> {
    let best = c.iter().filter(|x| x.graduated).map(|x| x.coverage).fold(f64::NEG_INFINITY, f64::max);
    if best == f64::NEG_INFINITY {
        return None;
    }
    let band: Vec<usize> = (0..c.len()).filter(|&i| c[i].graduated && c[i].coverage >= best - eps - 1e-12).collect();
    if band.iter().any(|&i| c[i].cost_us.is_none()) {
        return band.first().copied();
    }
    band.iter().copied().min_by(|&a, &b| c[a].cost_us.unwrap().total_cmp(&c[b].cost_us.unwrap()).then(a.cmp(&b)))
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            fit: FitSettings {
                target: 0.90,
                minority: 0.80,
                min_examples: 60,
                novelty_refs: 1000,
                readout: crate::readout::ReadoutChoice::Auto,
                attn: Default::default(),
                seed: 0,
                understudent: UnderstudentKind::Deep,
                tiers: vec![],
                abstain: Default::default(),
                graduation: Default::default(),
                attempt: 0,
            },
            audit_rate: 0.10,
            refit_every: 100,
            cheap_refit_every: default_cheap_every(),
            sampled_cap: 2000,
            labelled_cap: 5000,
            give_up_after: default_give_up(),
            cost_us: vec![],
            prefer_fast: None,
            rules: RulesSettings::default(),
            group_sep: default_group_sep(),
        }
    }
}

/// A ladder's tiers must be among these, each at most once.
pub fn check_tiers(tiers: &[UnderstudentKind]) -> Result<()> {
    ensure!(!tiers.is_empty(), "a ladder needs at least one tier");
    for (i, t) in tiers.iter().enumerate() {
        ensure!(matches!(t, UnderstudentKind::Rules | UnderstudentKind::Keywords | UnderstudentKind::Lite | UnderstudentKind::Probe | UnderstudentKind::Deep
            | UnderstudentKind::Shape | UnderstudentKind::Blend | UnderstudentKind::Memo | UnderstudentKind::Net | UnderstudentKind::Checklist), "{} can't be a ladder tier (tiers: rules, keywords, lite, probe, deep, shape, blend, memo, net, checklist)", t.name());
        ensure!(!tiers[..i].contains(t), "tier {} twice", t.name());
        ensure!(*t != UnderstudentKind::Rules || i == 0, "the rules tier must come first (tiers go cheapest first)");
    }
    Ok(())
}

/// Whether a tier is refitted on the short schedule.
fn cheap(kind: UnderstudentKind) -> bool {
    matches!(kind, UnderstudentKind::Rules | UnderstudentKind::Keywords | UnderstudentKind::Lite | UnderstudentKind::Shape | UnderstudentKind::Blend | UnderstudentKind::Memo | UnderstudentKind::Checklist)
}

/// One tier of a question's understudent: what it reads, and its serving state.
#[derive(Clone, Debug)]
pub struct Tier {
    pub kind: UnderstudentKind,
    pub fz: Featurizer,
    pub serving: Serving,
}

/// One question: its spec, examples and tiers.
#[derive(Clone, Debug)]
pub struct Understudent {
    pub name: String,
    pub key: String,
    pub question: Question,
    /// `Ladder`, or the one understudent all of whose (single) tier is.
    pub kind: UnderstudentKind,
    pub tiers: Vec<Tier>,
    /// New examples since the last refit of every tier.
    pub since_refit: usize,
    /// New examples since the last refit of any tier (a ladder's cheap tiers refit on a shorter schedule).
    pub since_cheap: usize,
    /// Oldest first (by `seq`).
    pub examples: Vec<Example>,
    /// `auto`: the candidate that serves (the only one that answers and is audited), if any graduated.
    pub serving_idx: Option<usize>,
    /// `auto`: why the question is lead-only (it forwards at once, computes no features and refits only every
    /// `RECHECK_EVERY` examples).
    pub lead_only: Option<String>,
    /// `auto`: the last refit's race, one entry per candidate.
    pub race: Value,
    /// `auto`: states of examples recorded without features while lead-only (by `seq`), encoded at the re-check.
    pub raw: HashMap<u64, Value>,
    /// The `rules` tier's proposal bookkeeping.
    pub rules: RulesState,
    /// `--graduation sequential`: graduation attempts so far (refits that ran the test), across demotions. 0 otherwise.
    pub attempts: usize,
}

impl Understudent {
    pub fn new(name: &str, question: Question, kind: UnderstudentKind, tiers: Vec<Tier>) -> Self {
        Understudent { name: name.into(), key: question.key(), question, kind, tiers, since_refit: 0, since_cheap: 0, examples: vec![],
                  rules: RulesState::default(), serving_idx: None, lead_only: None, race: Value::Null, raw: HashMap::new(), attempts: 0 }
    }

    pub fn is_ladder(&self) -> bool {
        self.kind == UnderstudentKind::Ladder
    }

    pub fn is_auto(&self) -> bool {
        self.kind == UnderstudentKind::Auto
    }

    /// A ladder or an auto race: several tiers/candidates, each with its own fit.
    pub fn is_tiered(&self) -> bool {
        self.kind.is_tiered()
    }

    /// Whether tier `t` may predict for a request: always in a ladder; in `auto` only the serving candidate, and
    /// nothing while lead-only.
    pub fn tries(&self, t: usize) -> bool {
        !self.is_auto() || (self.lead_only.is_none() && self.serving_idx == Some(t))
    }

    /// The serving candidate's name, if any.
    pub fn serving_name(&self) -> Option<&'static str> {
        self.serving_idx.map(|i| self.tiers[i].kind.name())
    }

    /// Make candidate `new` the serving one. A switch (to another candidate, or to none) starts empty audit windows
    /// for both candidates, as hunch does on a mode change: audits of the old understudent say nothing about the new one.
    pub fn switch_serving(&mut self, new: Option<usize>) {
        if new == self.serving_idx {
            return;
        }
        for t in [self.serving_idx, new].into_iter().flatten() {
            self.tiers[t].serving.audits.clear();
            self.tiers[t].serving.demoted = false;
        }
        self.serving_idx = new;
    }

    /// The last (most capable) tier; the only one unless a ladder.
    pub fn serving(&self) -> &Serving {
        &self.tiers.last().unwrap().serving
    }

    pub fn serving_mut(&mut self) -> &mut Serving {
        &mut self.tiers.last_mut().unwrap().serving
    }

    /// The tier that stands for the question: the first graduated and not demoted, else the first graduated, else
    /// the last.
    pub fn lead(&self) -> usize {
        let t = &self.tiers;
        if self.is_auto() {
            // the serving candidate, else the one nearest to graduating
            let agree = |i: usize| t[i].serving.fit.report.get("cv_agreement").and_then(Value::as_f64).unwrap_or(-1.0);
            return self.serving_idx.unwrap_or_else(|| (0..t.len()).rev().max_by(|&a, &b| agree(a).total_cmp(&agree(b))).unwrap_or(0));
        }
        t.iter()
            .position(|t| t.serving.mode().is_some() && !t.serving.demoted)
            .or_else(|| t.iter().position(|t| t.serving.mode().is_some()))
            .unwrap_or(t.len() - 1)
    }

    /// The `readout` of a decision answered by tier `ti`: `keywords`, `lite`, `mean` (probe) or `deep:pool|attn` in a
    /// ladder, else the understudent's own (`pool`/`attn` for deep, `mean`, `lite`, `neighbors`, `keywords`).
    pub fn label(&self, ti: usize) -> Option<String> {
        let t = &self.tiers[ti];
        let r = t.serving.fit.readout?;
        Some(match (self.kind, t.kind) {
            (UnderstudentKind::Ladder | UnderstudentKind::Auto, UnderstudentKind::Deep) => format!("deep:{}", r.name()),
            (UnderstudentKind::Ladder | UnderstudentKind::Auto, k) => readout_label(k, r).to_string(),
            (k, _) => readout_label(k, r).to_string(),
        })
    }

    fn add(&mut self, ex: Example, sampled_cap: usize, labelled_cap: usize) {
        self.examples.retain(|e| !(e.kind == ex.kind && e.state == ex.state));
        let kind = ex.kind;
        self.examples.push(ex);
        self.since_refit += 1;
        self.since_cheap += 1;
        // trim: the newest `sampled_cap` sampled examples, the newest `labelled_cap` labels per class
        let mut keep = vec![true; self.examples.len()];
        let mut seen: HashMap<Option<usize>, usize> = HashMap::new();
        for (i, e) in self.examples.iter().enumerate().rev() {
            if e.kind != kind {
                continue;
            }
            let (group, cap) = match kind {
                Kind::Sampled => (None, sampled_cap),
                Kind::Labelled => (Some(e.label), labelled_cap),
            };
            let c = seen.entry(group).or_default();
            *c += 1;
            if *c > cap {
                keep[i] = false;
            }
        }
        let mut it = keep.into_iter();
        self.examples.retain(|_| it.next().unwrap());
    }

    /// Training rows, oldest first, without sampled answers for states that have a label (hunch's
    /// `_drop_contested`).
    pub fn training(&self) -> Vec<&Example> {
        let labelled: std::collections::HashSet<&str> =
            self.examples.iter().filter(|e| e.kind == Kind::Labelled).map(|e| e.state.as_str()).collect();
        self.examples.iter().filter(|e| e.kind == Kind::Labelled || !labelled.contains(e.state.as_str())).collect()
    }

    pub fn state(&self) -> &'static str {
        if self.lead_only.is_some() {
            return "lead_only";
        }
        tier_state(&self.tiers[self.lead()].serving)
    }

    pub fn status(&self) -> Value {
        let li = self.lead();
        let lead = &self.tiers[li].serving;
        let r = &lead.fit.report;
        json!({
            "name": self.name, "key": self.key, "type": self.question.qtype, "state": self.state(),
            "examples": self.examples.len(),
            "labelled": self.examples.iter().filter(|e| e.kind == Kind::Labelled).count(),
            "since_refit": self.since_refit,
            "mode": lead.mode().map(Mode::name), "threshold": lead.fit.thr,
            "readout": if self.is_auto() && self.serving_idx.is_none() { None } else { self.label(li) },
            "understudent": self.kind.name(),
            "serving": self.serving_name(), "lead_only": self.lead_only, "race": self.race,
            "tiers": self.is_tiered().then(|| self.tiers.iter().enumerate().map(|(i, t)| json!({
                "tier": t.kind.name(), "state": tier_state(&t.serving),
                "mode": t.serving.mode().map(Mode::name), "threshold": t.serving.fit.thr, "readout": self.label(i),
                "cv_agreement": t.serving.fit.report.get("cv_agreement"), "cv_coverage": t.serving.fit.report.get("cv_coverage"),
                "audits": t.serving.audit_summary(),
                "rules": t.serving.fit.rules.as_ref().map(|r| rules_json(r)),
            })).collect::<Vec<_>>()),
            "cv_agreement": r.get("cv_agreement"), "cv_minority_recall": r.get("cv_minority_recall"),
            "cv_coverage": r.get("cv_coverage"), "why_not": r.get("why_not"),
            "audits": lead.audit_summary(), "report": r,
            "rules": self.tiers.iter().find_map(|t| t.serving.fit.rules.as_ref()).map(|r| rules_json(r)),
        })
    }
}

/// A rule set for status output: its JSON and a readable listing.
fn rules_json(r: &rules::RuleSet) -> Value {
    let mut v = serde_json::to_value(r).unwrap_or(Value::Null);
    v["listing"] = json!(r.describe());
    v
}

fn tier_state(s: &Serving) -> &'static str {
    match (s.mode(), s.demoted) {
        (None, _) => "learning",
        (Some(_), true) => "demoted",
        (Some(_), false) => "graduated",
    }
}

/// A decision for one question on one item, in the SPEC's decisions.jsonl shape.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Decision {
    pub id: Option<String>,
    pub answered_by: String,
    pub label: usize,
    pub probs: Vec<f64>,
    pub understudent_label: Option<usize>,
    pub understudent_probs: Option<Vec<f64>>,
    pub eligible: bool,
    pub novel: bool,
    pub demoted: bool,
    pub mode: Option<String>,
    pub audited: bool,
    pub readout: Option<String>,
    pub understudent_us: u64,
}

/// A request's outcome: who answered and each question's decision (in the request's question order).
#[derive(Clone, Debug)]
pub struct Outcome {
    pub answered_by: &'static str,
    pub decisions: Vec<(String, Decision)>,
    /// Refits this request triggered: (question name, seconds, report).
    pub refits: Vec<(String, f64, Value)>,
}

/// Recently used encodings, by cache key (the hash of the state's texts).
#[derive(Default)]
pub struct EncCache {
    map: HashMap<String, Arc<Encoding>>,
    order: VecDeque<String>,
    pub cap: usize,
}

impl EncCache {
    pub fn get(&self, k: &str) -> Option<Arc<Encoding>> {
        self.map.get(k).cloned()
    }

    pub fn put(&mut self, k: String, e: Arc<Encoding>) {
        if self.map.insert(k.clone(), e).is_none() {
            self.order.push_back(k);
            while self.order.len() > self.cap.max(1) {
                let old = self.order.pop_front().unwrap();
                self.map.remove(&old);
            }
        }
    }
}

/// One state being answered: its texts and the encoding computed so far. `ensure` fills in tiers lazily.
pub struct Work {
    pub key: String,
    pub enc: Arc<Encoding>,
    full: Vec<String>,
    /// What the `rules` tier reads (when the proxy has one).
    rules: Option<Arc<RuleInput>>,
    /// The structural features (a `shape` or `blend` tier) and the retrieval key (a `memo` tier).
    shape: Option<Arc<Vec<f32>>>,
    memo: Option<Arc<MemoKey>>,
    /// What names a `checklist` tier's rules (kept in the encoding, in memory).
    aux: Option<Arc<crate::encoder::Aux>>,
    /// A probe pass kept so that deep can continue it (only for short states, whose probe and deep texts agree).
    held: Option<Held>,
}

struct Held {
    /// One text per field (each a single chunk).
    texts: Vec<String>,
    /// The layers (a prefix of deep's) the pass has run, and their states, per field.
    layers: Vec<usize>,
    tokens: Vec<Tokens>,
    resumes: Vec<Option<Resume>>,
}

pub struct Proxy {
    pub settings: Settings,
    /// The base featurizer: a single understudent's, or a ladder's deep tier's (layers, attention layer, pair, chunking).
    pub fz: Featurizer,
    pub encoder: Arc<dyn Encoder>,
    pub lead: Option<Box<dyn Lead>>,
    pub understudents: Vec<Understudent>,
    /// The audit draws: CPython's `random.Random(seed)`, as hunch's proxy.
    pub rng: Mt19937,
    pub seq: u64,
    pub cache: EncCache,
    /// Every understudent's tiers, cheapest first, with their featurizers.
    pub tier_fz: Vec<(UnderstudentKind, Featurizer)>,
    /// `auto`: each tier's measured encode microseconds per item (moving average over the batches that computed it).
    pub tier_cost: Vec<Option<f64>>,
    /// The rules LLM behind its cache (when a tier is `rules`); tests put a mock here.
    pub rules: Option<Arc<RulesClient>>,
    /// The on-disk encode cache (`--encode-cache`), consulted before the encoder runs.
    pub disk: Option<Arc<DiskCache>>,
    /// Frozen (`judge --frozen`): answers are scored by the fits as they are, and nothing is learned: no example is
    /// recorded, no audit counted and no refit run, so a held-out stream never trains the understudent. Forwarded items
    /// still ask the lead (the decision carries its answer). Not saved in snapshots; off by default.
    pub frozen: bool,
    /// `serve`: learn from forwarded answers without auditing the current fit against them (the caller audits the
    /// prediction it actually served). Not saved in snapshots; off by default.
    pub no_audit: bool,
}

impl Proxy {
    pub fn new(mut settings: Settings, fz: Featurizer, encoder: Arc<dyn Encoder>, lead: Option<Box<dyn Lead>>) -> Self {
        let rng = Mt19937::python(settings.fit.seed);
        let tier_fz = if settings.fit.understudent.is_tiered() {
            if settings.fit.tiers.is_empty() {
                settings.fit.tiers = UnderstudentKind::default_tiers();
            }
            check_tiers(&settings.fit.tiers).expect("valid ladder tiers");
            settings.fit.tiers.iter().map(|&k| (k, fz.tier(k).expect("a ladder tier"))).collect()
        } else {
            vec![(settings.fit.understudent, fz.clone())]
        };
        let rules = tier_fz
            .iter()
            .any(|(k, _)| *k == UnderstudentKind::Rules)
            .then(|| Arc::new(RulesClient::new(Box::new(rules::ChatLlm::new(&settings.rules.url).with_key(rules_key(&settings.rules.key_env))))));
        Proxy { settings, fz, encoder, lead, understudents: vec![], rng, seq: 0, cache: EncCache { cap: 4096, ..Default::default() }, tier_cost: vec![None; tier_fz.len()], tier_fz, rules, disk: None, frozen: false, no_audit: false }
    }

    /// Serve and fill encodings from the on-disk cache in `dir` (see `diskcache`).
    pub fn set_encode_cache(&mut self, dir: &std::path::Path) -> Result<()> {
        self.disk = Some(Arc::new(DiskCache::open(dir, &self.encoder.fingerprint())?));
        Ok(())
    }

    fn has_rules_tier(&self) -> bool {
        self.tier_fz.iter().any(|(k, _)| *k == UnderstudentKind::Rules)
    }

    fn has_tier(&self, pred: impl Fn(UnderstudentKind) -> bool) -> bool {
        self.tier_fz.iter().any(|(k, _)| pred(*k))
    }

    /// Fresh (unfitted) tiers for a new question.
    pub fn new_tiers(&self) -> Vec<Tier> {
        self.tier_fz.iter().map(|(k, fz)| Tier { kind: *k, fz: fz.clone(), serving: Serving::default() }).collect()
    }

    /// Grow an `auto` or ladder proxy that started without an encoder: switch to `encoder` and append the designs `kinds`
    /// (those that read it) as new, unfitted candidates after the existing ones. Existing fits, examples and the serving
    /// candidate are kept; the new candidates join the race at the next refit (the examples' missing features are
    /// computed then). Returns false, changing nothing, when every kind is already a tier.
    pub fn add_encoder_tiers(&mut self, encoder: Arc<dyn Encoder>, kinds: &[UnderstudentKind]) -> Result<bool> {
        ensure!(self.settings.fit.understudent.is_tiered(), "only a ladder or auto has tiers to add");
        let new: Vec<UnderstudentKind> = kinds.iter().copied().filter(|k| !self.tier_fz.iter().any(|(t, _)| t == k)).collect();
        if new.is_empty() {
            return Ok(false);
        }
        let hidden = encoder.hidden();
        self.encoder = encoder;
        self.fz.hidden = hidden;
        for (_, f) in &mut self.tier_fz {
            f.hidden = hidden;
        }
        for s in &mut self.understudents {
            for t in &mut s.tiers {
                t.fz.hidden = hidden;
            }
        }
        for k in new {
            let fz = self.fz.tier(k)?;
            self.tier_fz.push((k, fz.clone()));
            self.tier_cost.push(None);
            self.settings.fit.tiers.push(k);
            for s in &mut self.understudents {
                s.tiers.push(Tier { kind: k, fz: fz.clone(), serving: Serving::default() });
            }
        }
        self.disk = None;
        Ok(true)
    }

    /// The understudent for a question, added if new. A known key keeps its first name.
    pub fn understudent(&mut self, name: &str, q: &Question) -> usize {
        let key = q.key();
        if let Some(i) = self.understudents.iter().position(|s| s.key == key) {
            return i;
        }
        let tiers = self.new_tiers();
        self.understudents.push(Understudent::new(name, q.clone(), self.settings.fit.understudent, tiers));
        self.understudents.len() - 1
    }

    fn all_tiers(&self) -> Vec<usize> {
        (0..self.tier_fz.len()).collect()
    }

    /// Encode states for every tier of this proxy's understudents (through the cache, then the examples, then the encoder
    /// in one batch).
    pub fn encode_many(&mut self, states: &[&Value]) -> Result<Vec<(String, Arc<Encoding>)>> {
        let tiers = self.all_tiers();
        self.encode_with(states, &tiers)
    }

    /// Encode states, computing only the tiers `tiers` (indices into `tier_fz`) read that the cache or examples
    /// don't already hold.
    pub fn encode_with(&mut self, states: &[&Value], tiers: &[usize]) -> Result<Vec<(String, Arc<Encoding>)>> {
        let mut works: Vec<Work> = states.iter().map(|s| self.work(s)).collect();
        self.ensure_many(&mut works, tiers)?;
        Ok(works.into_iter().map(|w| (w.key, w.enc)).collect())
    }

    /// Start work on a state: its texts and whatever encoding the cache or the examples hold already.
    pub fn work(&self, state: &Value) -> Work {
        let full = self.fz.full_texts(state);
        let mut key = self.fz.cache_key(&full);
        let rules = self.has_rules_tier().then(|| Arc::new(RuleInput::from_state(state)));
        if let Some(r) = &rules {
            key = text_hash(&format!("{key}\u{0}rules\u{0}{}", r.text));
        }
        // the structural features and the retrieval key read the state itself, not just its (capped) texts
        let (want_shape, want_memo) = (self.has_tier(|k| matches!(k, UnderstudentKind::Shape | UnderstudentKind::Blend | UnderstudentKind::Net | UnderstudentKind::Checklist)), self.has_tier(|k| k == UnderstudentKind::Memo));
        let shape = want_shape.then(|| Arc::new(crate::shape::features(state, self.fz.pair.as_ref())));
        let memo = want_memo.then(|| Arc::new(MemoKey::from_text(&state_text(state))));
        if want_shape || want_memo {
            key = text_hash(&format!("{key}\u{0}state\u{0}{}", crate::pyjson::dumps(state)));
        }
        let enc = self.lookup(&key).unwrap_or_default();
        let aux = self.has_tier(|k| k == UnderstudentKind::Checklist).then(|| Arc::new(crate::encoder::Aux { texts: full.clone(), slots: crate::shape::slot_names(state) }));
        Work { key, enc, full, rules, shape, memo, aux, held: None }
    }

    /// Whether `e` holds what tier `t` reads.
    fn has(&self, t: usize, e: &Encoding) -> bool {
        let (kind, fz) = &self.tier_fz[t];
        match *kind {
            UnderstudentKind::Keywords => e.sparse.is_some(),
            UnderstudentKind::Rules => e.rules.is_some(),
            UnderstudentKind::Shape => e.shape.is_some(),
            UnderstudentKind::Memo => e.memo.is_some(),
            UnderstudentKind::Checklist => e.sparse.is_some() && e.shape.is_some(),
            UnderstudentKind::Net => e.shape.is_some() && !fz.parts(e).is_empty(),
            UnderstudentKind::Blend => e.sparse.is_some() && e.shape.is_some() && !fz.parts(e).is_empty(),
            _ => !fz.parts(e).is_empty(),
        }
    }

    pub fn ensure(&mut self, w: &mut Work, tiers: &[usize]) -> Result<()> {
        self.ensure_many(std::slice::from_mut(w), tiers)
    }

    /// Fill in what tiers `tiers` read, for every work, in one encoder batch per tier. In a ladder, probe's pass over a
    /// short state is kept (in `Work::held`) and continued to deep's deepest layer when deep's tier is computed.
    pub fn ensure_many(&mut self, works: &mut [Work], tiers: &[usize]) -> Result<()> {
        for &t in tiers {
            let (kind, fz) = self.tier_fz[t].clone();
            // each distinct state once (duplicates in a batch take the first's result)
            let mut first: HashMap<String, usize> = HashMap::new();
            let mut todo: Vec<usize> = vec![];
            let mut copies: Vec<(usize, usize)> = vec![];
            for i in 0..works.len() {
                if self.has(t, &works[i].enc) {
                    continue;
                }
                match first.get(&works[i].key) {
                    Some(&f) => copies.push((i, f)),
                    None => {
                        first.insert(works[i].key.clone(), i);
                        todo.push(i);
                    }
                }
            }
            // net reads the structural features beside deep's encoding: fill them in first, so a disk-cache hit for the
            // encoding still leaves the row complete
            if kind == UnderstudentKind::Net {
                for &i in &todo {
                    if works[i].enc.shape.is_none() {
                        let v = works[i].shape.clone();
                        Arc::make_mut(&mut works[i].enc).shape = v;
                    }
                }
            }
            // the disk cache serves states whose every field was encoded before (under this featurizer)
            let disk = self.disk.clone().filter(|_| !matches!(kind, UnderstudentKind::Keywords | UnderstudentKind::Rules | UnderstudentKind::Shape | UnderstudentKind::Memo | UnderstudentKind::Blend | UnderstudentKind::Checklist));
            if let Some(dc) = &disk {
                let (mut rest, mut hit_cost) = (vec![], vec![]);
                for &i in &todo {
                    let got: Vec<Option<(Part, f64)>> = works[i].full.iter().map(|t| dc.get(&dc.key(&fz, t))).collect();
                    if got.iter().all(Option::is_some) {
                        hit_cost.push(got.iter().flatten().map(|(_, c)| *c).fold(0.0, f64::max));
                        let parts: Vec<Part> = got.into_iter().flatten().map(|(p, _)| p).collect();
                        Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, parts);
                    } else {
                        rest.push(i);
                    }
                }
                if !hit_cost.is_empty() {
                    let per_item = hit_cost.iter().sum::<f64>() / hit_cost.len() as f64;
                    self.tier_cost[t] = Some(self.tier_cost[t].map_or(per_item, |c| 0.8 * c + 0.2 * per_item));
                }
                todo = rest;
            }
            if todo.is_empty() {
                for (i, f) in copies {
                    let e = works[f].enc.clone();
                    works[i].enc = e;
                }
                continue;
            }
            let clock = Instant::now();
            match kind {
                UnderstudentKind::Keywords => {
                    for &i in &todo {
                        let sp = crate::keywords::featurize(&works[i].full);
                        Arc::make_mut(&mut works[i].enc).sparse = Some(sp);
                    }
                }
                UnderstudentKind::Rules => {
                    for &i in &todo {
                        let r = works[i].rules.clone();
                        Arc::make_mut(&mut works[i].enc).rules = r;
                    }
                }
                UnderstudentKind::Shape => {
                    for &i in &todo {
                        let v = works[i].shape.clone();
                        Arc::make_mut(&mut works[i].enc).shape = v;
                    }
                }
                UnderstudentKind::Memo => {
                    for &i in &todo {
                        let m = works[i].memo.clone();
                        Arc::make_mut(&mut works[i].enc).memo = m;
                    }
                }
                UnderstudentKind::Checklist => {
                    for &i in &todo {
                        if works[i].enc.sparse.is_none() {
                            let sp = crate::keywords::featurize(&works[i].full);
                            Arc::make_mut(&mut works[i].enc).sparse = Some(sp);
                        }
                        if works[i].enc.shape.is_none() {
                            let v = works[i].shape.clone();
                            Arc::make_mut(&mut works[i].enc).shape = v;
                        }
                    }
                }
                UnderstudentKind::Blend => {
                    // hashed n-grams and structural features (whatever the encoding lacks), then the embedding table
                    for &i in &todo {
                        if works[i].enc.sparse.is_none() {
                            let sp = crate::keywords::featurize(&works[i].full);
                            Arc::make_mut(&mut works[i].enc).sparse = Some(sp);
                        }
                        if works[i].enc.shape.is_none() {
                            let v = works[i].shape.clone();
                            Arc::make_mut(&mut works[i].enc).shape = v;
                        }
                    }
                    let need: Vec<usize> = todo.iter().copied().filter(|&i| fz.parts(&works[i].enc).is_empty()).collect();
                    if !need.is_empty() {
                        let texts: Vec<Vec<Vec<String>>> = need.iter().map(|&i| fz.field_texts(&works[i].full)).collect();
                        let parts = fz.encode_fields(self.encoder.as_ref(), &texts)?;
                        for (&i, p) in need.iter().zip(parts) {
                            Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, p);
                        }
                    }
                }
                UnderstudentKind::Probe if self.main_tier().is_some() => self.encode_mean_staged(works, &todo, &fz)?,
                UnderstudentKind::Deep | UnderstudentKind::Net if todo.iter().any(|&i| works[i].held.is_some()) => self.encode_main_resumed(works, &todo, &fz)?,
                _ => {
                    let texts: Vec<Vec<Vec<String>>> = todo.iter().map(|&i| fz.field_texts(&works[i].full)).collect();
                    let parts = fz.encode_fields(self.encoder.as_ref(), &texts)?;
                    for (&i, p) in todo.iter().zip(parts) {
                        Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, p);
                    }
                }
            }
            let per_item = clock.elapsed().as_secs_f64() * 1e6 / todo.len() as f64;
            if let Some(dc) = &disk {
                for &i in &todo {
                    for (text, part) in works[i].full.iter().zip(works[i].enc.slot(fz.slot)) {
                        if let Err(e) = dc.put(&dc.key(&fz, text), part, per_item) {
                            eprintln!("understudy: encode cache write failed: {e:#}");
                        }
                    }
                }
            }
            self.tier_cost[t] = Some(self.tier_cost[t].map_or(per_item, |c| 0.8 * c + 0.2 * per_item));
            for (i, f) in copies {
                let e = works[f].enc.clone();
                works[i].enc = e;
            }
        }
        for w in works.iter_mut() {
            if let (Some(a), None) = (&w.aux, &w.enc.aux) {
                Arc::make_mut(&mut w.enc).aux = Some(a.clone());
            }
        }
        for w in works.iter() {
            self.cache.put(w.key.clone(), w.enc.clone());
        }
        Ok(())
    }

    /// The ladder's deep tier, if it has one (index, featurizer).
    fn main_tier(&self) -> Option<(usize, &Featurizer)> {
        if !self.settings.fit.understudent.is_tiered() {
            return None;
        }
        self.tier_fz.iter().position(|(k, _)| matches!(*k, UnderstudentKind::Deep | UnderstudentKind::Net)).map(|i| (i, &self.tier_fz[i].1))
    }

    /// probe's tier when deep has one too: states whose probe and deep texts agree (short ones, one chunk per field)
    /// run `encode_resumable` through deep's layers up to probe's, keep the pass in the work for deep to continue,
    /// and give probe its layer's mean from it; the others are encoded plainly.
    fn encode_mean_staged(&mut self, works: &mut [Work], todo: &[usize], fz: &Featurizer) -> Result<()> {
        let (mi, main) = self.main_tier().expect("a deep tier");
        let main = main.clone();
        let l = fz.layers[0];
        let mut stage1: Vec<usize> = main.layers.iter().copied().filter(|&x| x <= l).collect();
        stage1.sort_unstable();
        stage1.dedup();
        let can = stage1.contains(&l);
        let mut staged: Vec<usize> = vec![];
        let mut plain: Vec<usize> = vec![];
        for &i in todo {
            let mean_texts = fz.field_texts(&works[i].full);
            let reuse = can && !self.has(mi, &works[i].enc) && main.field_texts(&works[i].full) == mean_texts;
            if reuse { staged.push(i) } else { plain.push(i) }
        }
        if !plain.is_empty() {
            let texts: Vec<Vec<Vec<String>>> = plain.iter().map(|&i| fz.field_texts(&works[i].full)).collect();
            let parts = fz.encode_fields(self.encoder.as_ref(), &texts)?;
            for (&i, p) in plain.iter().zip(parts) {
                Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, p);
            }
        }
        if !staged.is_empty() {
            let texts: Vec<String> = staged.iter().flat_map(|&i| fz.field_texts(&works[i].full).into_iter().map(|mut c| c.remove(0))).collect();
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let outs = self.encoder.encode_resumable(&refs, &stage1)?;
            let pos = stage1.iter().position(|&x| x == l).unwrap();
            let mut it = outs.into_iter();
            let mut tx = texts.into_iter();
            for &i in &staged {
                let nf = works[i].full.len();
                let (mut parts, mut tokens, mut resumes, mut own) = (vec![], vec![], vec![], vec![]);
                for _ in 0..nf {
                    let (t, r) = it.next().expect("one output per text");
                    parts.push(fz.part_chunks(&[t.select(&[pos])]));
                    tokens.push(t);
                    resumes.push(r);
                    own.push(tx.next().unwrap());
                }
                Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, parts);
                works[i].held = Some(Held { texts: own, layers: stage1.clone(), tokens, resumes });
            }
        }
        Ok(())
    }

    /// deep's tier for works that hold a probe pass: continue those through deep's deeper layers (the others are
    /// encoded plainly).
    fn encode_main_resumed(&mut self, works: &mut [Work], todo: &[usize], fz: &Featurizer) -> Result<()> {
        let held: Vec<usize> = todo.iter().copied().filter(|&i| works[i].held.is_some()).collect();
        let plain: Vec<usize> = todo.iter().copied().filter(|&i| works[i].held.is_none()).collect();
        if !plain.is_empty() {
            let texts: Vec<Vec<Vec<String>>> = plain.iter().map(|&i| fz.field_texts(&works[i].full)).collect();
            let parts = fz.encode_fields(self.encoder.as_ref(), &texts)?;
            for (&i, p) in plain.iter().zip(parts) {
                Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, p);
            }
        }
        let have = works[held[0]].held.as_ref().unwrap().layers.clone();
        let stage2: Vec<usize> = fz.layers.iter().copied().filter(|l| !have.contains(l)).collect();
        let mut hs: Vec<Held> = held.iter().map(|&i| works[i].held.take().unwrap()).collect();
        let resumed: Vec<Tokens> = if stage2.is_empty() {
            vec![]
        } else {
            let from: Vec<Option<Resume>> = hs.iter_mut().flat_map(|h| h.resumes.drain(..)).collect();
            let texts: Vec<&str> = hs.iter().flat_map(|h| h.texts.iter().map(String::as_str)).collect();
            self.encoder.resume(&texts, from, &stage2)?
        };
        let mut more = resumed.into_iter();
        for (&i, h) in held.iter().zip(&hs) {
            let mut parts = vec![];
            for f in 0..h.tokens.len() {
                let t = &h.tokens[f];
                let deep = if stage2.is_empty() { None } else { Some(more.next().expect("one output per text")) };
                let states = fz
                    .layers
                    .iter()
                    .map(|l| match h.layers.iter().position(|x| x == l) {
                        Some(p) => t.states[p].clone(),
                        None => deep.as_ref().unwrap().states[stage2.iter().position(|x| x == l).unwrap()].clone(),
                    })
                    .collect();
                parts.push(fz.part_chunks(&[Tokens { tokens: t.tokens, hidden: t.hidden, states }]));
            }
            Arc::make_mut(&mut works[i].enc).set_slot(fz.slot, parts);
        }
        Ok(())
    }

    fn lookup(&self, key: &str) -> Option<Arc<Encoding>> {
        self.cache.get(key).or_else(|| {
            self.understudents.iter().flat_map(|s| s.examples.iter()).find(|e| e.enc_key == key).map(|e| e.enc.clone())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record(&mut self, si: usize, kind: Kind, state: String, text_len: usize, label: usize, probs: Option<Vec<f64>>, enc_key: String, enc: Arc<Encoding>, id: Option<&str>) {
        self.seq += 1;
        let group = group_of(self.settings.group_sep, id);
        let ex = Example { seq: self.seq, kind, state, label, probs, enc_key, enc, text_len, group };
        let (sc, lc) = (self.settings.sampled_cap, self.settings.labelled_cap);
        self.understudents[si].add(ex, sc, lc);
    }

    /// Add a labelled example (ground truth) or a sampled one (a lead's answer) without routing.
    #[allow(clippy::too_many_arguments)]
    pub fn add_example(&mut self, si: usize, kind: Kind, state: &Value, label: usize, probs: Option<Vec<f64>>, enc_key: String, enc: Arc<Encoding>) {
        self.add_example_with_id(si, kind, state, label, probs, enc_key, enc, None);
    }

    /// `add_example` for an item with an id (which decides its CV group, see `Settings::group_sep`).
    #[allow(clippy::too_many_arguments)]
    pub fn add_example_with_id(&mut self, si: usize, kind: Kind, state: &Value, label: usize, probs: Option<Vec<f64>>, enc_key: String, enc: Arc<Encoding>, id: Option<&str>) {
        self.record(si, kind, state_hash(state), state_text(state).len(), label, probs, enc_key, enc, id);
    }

    /// Refit every tier of one understudent now. Returns (seconds, report): the last tier's report, and for a ladder every
    /// tier's under "tiers".
    pub fn refit(&mut self, si: usize) -> (f64, Value) {
        self.refit_tiers(si, false)
    }

    /// Refit an understudent's tiers: all of them, or (`cheap_only`) a ladder's `keywords` and `lite`.
    fn refit_tiers(&mut self, si: usize, cheap_only: bool) -> (f64, Value) {
        let t = Instant::now();
        if self.understudents[si].is_auto() {
            self.complete_examples(si);
        }
        let st = &self.understudents[si];
        // an auto race needs every candidate's features on every row (examples recorded while lead-only and not yet
        // completed, or loaded without them, sit out)
        let nt = self.tier_fz.len();
        // an auto candidate is fitted on the rows that hold its features (a row recorded before a design joined, with no
        // state kept to complete it, sits out of that design's fit only)
        let rows_for = |t: usize| -> Vec<Row> {
            st.training().into_iter().filter(|e| !st.is_auto() || self.has(t, &e.enc)).map(|e| Row { enc: &e.enc, y: e.label, w: e.weight(), group: e.group }).collect()
        };
        let per_tier: Vec<Vec<Row>> = (0..nt.max(st.tiers.len())).map(|t| if t < st.tiers.len() { rows_for(t) } else { vec![] }).collect();
        let rows: &Vec<Row> = per_tier.iter().max_by_key(|r| r.len()).expect("a tier");
        let n_rows = rows.len();
        let (qtype, k) = (st.question.qtype, st.question.k());
        let which: Vec<usize> = (0..st.tiers.len()).filter(|&i| !cheap_only || cheap(st.tiers[i].kind)).collect();
        // `--graduation sequential`: a refit that runs the graduation test is the question's next attempt (a demoted question
        // re-entering graduation continues the count, so it never starts over)
        let sequential = self.settings.fit.graduation == Graduation::Sequential;
        let tests = n_rows >= self.settings.fit.min_examples && rows.iter().any(|r| r.y != rows[0].y);
        let attempt = if sequential && tests { st.attempts + 1 } else { st.attempts };
        let grad_z = FitSettings { attempt, ..self.settings.fit.clone() }.grad_z();
        let mut rules_state: Option<RulesState> = None;
        let fits: Vec<(usize, Fit)> = which
            .iter()
            .map(|&i| {
                let tier = &st.tiers[i];
                if tier.kind == UnderstudentKind::Rules {
                    let rr: Vec<RuleRow> = st
                        .training()
                        .into_iter()
                        .filter_map(|e| Some(RuleRow { hash: &e.state, input: e.enc.rules.as_deref()?, y: e.label, seq: e.seq, group: e.group }))
                        .collect();
                    let f = &self.settings.fit;
                    let ctx = RulesCtx { client: self.rules.as_deref(), settings: &self.settings.rules, question: &st.question, target: f.target,
                                         minority: f.minority, min_examples: f.min_examples, seed: f.seed, z: grad_z };
                    let failing = tier.serving.fit.mode.is_none() || tier.serving.demoted;
                    let (fit, state) = rules::fit_rules(&ctx, &rr, tier.serving.fit.rules.as_ref(), failing, &st.rules, self.seq);
                    rules_state = Some(state);
                    return (i, fit);
                }
                let s = FitSettings { understudent: tier.kind, attempt, ..self.settings.fit.clone() };
                (i, fit_understudent(&tier.fz, qtype, k, &per_tier[i], &s))
            })
            .collect();
        let tiered = st.is_tiered();
        let mut report = fits.last().map_or(Value::Null, |(_, f)| f.report.clone());
        if tiered {
            let m: Map<String, Value> = fits.iter().map(|(i, f)| (self.understudents[si].tiers[*i].kind.name().to_string(), f.report.clone())).collect();
            report["tiers"] = Value::Object(m);
            report["refit"] = json!(if cheap_only { "cheap" } else { "all" });
        }
        if sequential && tests {
            report["graduation"] = json!({"policy": "sequential", "attempt": attempt, "z": grad_z, "candidates": st.tiers.len().max(1)});
        }
        let target = self.settings.fit.target;
        let st = &mut self.understudents[si];
        st.attempts = attempt;
        if let Some(rs) = rules_state {
            st.rules = rs;
        }
        st.since_cheap = 0;
        if !cheap_only {
            st.since_refit = 0;
        }
        for (i, f) in fits {
            st.tiers[i].serving.install(f, target);
        }
        if self.understudents[si].is_auto() {
            self.finish_race(si, n_rows, &mut report);
        }
        (t.elapsed().as_secs_f64(), report)
    }

    /// Whether an auto question runs without the embedding model so far (its examples keep their states, so the encoder
    /// designs that join later can be fitted on them too).
    fn awaiting_encoder(&self, si: usize) -> bool {
        self.understudents[si].is_auto() && self.encoder.hidden() == 0
    }

    /// Drop kept states whose example has been evicted.
    fn prune_raw(&mut self, si: usize) {
        let st = &mut self.understudents[si];
        if st.raw.len() > st.examples.len() * 2 + 16 {
            let live: std::collections::HashSet<u64> = st.examples.iter().map(|e| e.seq).collect();
            st.raw.retain(|k, _| live.contains(k));
        }
    }

    /// Encode, for an auto question that was lead-only, the examples recorded without features (from their kept
    /// states), in batches. Examples that can't be completed (loaded from a snapshot without their states) stay out of
    /// the fit.
    fn complete_examples(&mut self, si: usize) {
        let nt = self.tier_fz.len();
        let st = &self.understudents[si];
        let need: Vec<usize> =
            (0..st.examples.len()).filter(|&i| st.raw.contains_key(&st.examples[i].seq) && (0..nt).any(|t| !self.has(t, &st.examples[i].enc))).collect();
        let all = self.all_tiers();
        for chunk in need.chunks(64) {
            let states: Vec<Value> = chunk.iter().map(|&i| self.understudents[si].raw[&self.understudents[si].examples[i].seq].clone()).collect();
            let mut works: Vec<Work> = states.iter().map(|s| self.work(s)).collect();
            if let Err(e) = self.ensure_many(&mut works, &all) {
                eprintln!("understudy: encoding the lead-only examples failed: {e:#}");
                return;
            }
            for (&i, w) in chunk.iter().zip(works) {
                self.understudents[si].examples[i].enc = w.enc;
            }
        }
        if !self.awaiting_encoder(si) {
            self.understudents[si].raw.clear();
        }
    }

    /// An auto question's refit tail: each candidate's cost, the race record, the serving candidate (see
    /// `select_serving`) and the lead-only verdict. `n` is the number of rows the candidates were fitted on.
    fn finish_race(&mut self, si: usize, n: usize, report: &mut Value) {
        let nt = self.tier_fz.len();
        let costs: Vec<Option<f64>> = (0..nt).map(|i| self.candidate_cost(si, i)).collect();
        let st = &self.understudents[si];
        let race: Vec<Value> = st
            .tiers
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let r = &t.serving.fit.report;
                json!({"understudent": t.kind.name(), "cv_agreement": r.get("cv_agreement"), "cv_coverage": r.get("cv_coverage"),
                       "mode": t.serving.mode().map(Mode::name), "why_not": r.get("why_not"), "cost_us": costs[i]})
            })
            .collect();
        let cands: Vec<Cand> = st
            .tiers
            .iter()
            .zip(&costs)
            .map(|(t, &cost_us)| Cand {
                graduated: t.serving.mode().is_some(),
                coverage: t.serving.fit.report.get("cv_coverage").and_then(Value::as_f64).unwrap_or(0.0),
                cost_us,
            })
            .collect();
        // coverage is out-of-sample for every candidate: out-of-fold over the `n` rows for the fitted ones, over the
        // held-out rows (those the LLM never saw, `heldout_n` of them) for `rules`. A smaller evaluation set means a
        // noisier coverage, so the tie band uses the smallest evaluation set among the graduated candidates.
        let n_eval = st
            .tiers
            .iter()
            .filter(|t| t.kind == UnderstudentKind::Rules && t.serving.mode().is_some())
            .filter_map(|t| t.serving.fit.report.get("heldout_n").and_then(Value::as_u64))
            .fold(n, |a, h| a.min(h as usize));
        let pick = match self.settings.prefer_fast {
            Some(eps) => select_fastest(&cands, eps),
            None => select_serving(&cands, n_eval),
        };
        let best_agree = st
            .tiers
            .iter()
            .filter_map(|t| t.serving.fit.report.get("cv_agreement").and_then(Value::as_f64))
            .fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x))));
        let target = self.settings.fit.target;
        let verdict = match (pick, best_agree) {
            (None, Some(a)) if n >= self.settings.give_up_after && a < target - GIVE_UP_MARGIN => Some(format!(
                "no candidate graduated and the best cv_agreement is {a:.3}, more than {GIVE_UP_MARGIN} under the target {target} after {n} examples"
            )),
            _ => None,
        };
        let keep_raw = self.awaiting_encoder(si);
        let st = &mut self.understudents[si];
        st.switch_serving(pick);
        if verdict.is_none() && !keep_raw {
            st.raw.clear();
        }
        st.lead_only = verdict;
        st.race = json!(race);
        report["race"] = json!(race);
        report["serving"] = json!(st.serving_name());
        report["lead_only"] = json!(st.lead_only);
    }

    /// An auto candidate's cost in microseconds per item: the given `Settings::cost_us`, else its measured encode time
    /// plus the measured prediction time on recent examples. deep's encode time adds probe's when both race: a shared
    /// pass is timed in two pieces (probe's to layer 8, deep's rest), while a lone serving deep runs all of it. None
    /// when the encode time was never measured (examples that came from a snapshot or `train`).
    fn candidate_cost(&self, si: usize, i: usize) -> Option<f64> {
        if let Some(&c) = self.settings.cost_us.get(i) {
            return Some(c);
        }
        let kind = self.tier_fz[i].0;
        if self.settings.prefer_fast.is_some() {
            // the deterministic cost model: the design, the deepest layer it runs and the tokens it encodes
            let fz = &self.tier_fz[i].1;
            let st = &self.understudents[si];
            // token counts live in the main slot (deep, net) or, with the token states off, are 0 in probe's: take the larger
            let toks: Vec<f64> = st.examples.iter().rev().filter(|e| self.has(i, &e.enc)).take(COST_SAMPLE).map(|e| match [&e.enc.parts, &e.enc.mean].iter().map(|ps| ps.iter().map(|p| p.tokens).sum::<usize>()).max().unwrap_or(0) {
                0 => (e.text_len / 4).max(1) as f64, // no token count recorded (a race without deep): about 4 bytes a token
                t => t as f64,
            }).collect();
            let tokens = if toks.is_empty() { MODEL_DEFAULT_TOKENS } else { toks.iter().sum::<f64>() / toks.len() as f64 };
            return Some(model_cost_us(kind, fz.layers.iter().copied().max().unwrap_or(0), tokens));
        }
        let mut enc = self.tier_cost[i]?;
        if kind == UnderstudentKind::Deep {
            if let Some(e) = self.tier_fz.iter().position(|(k, _)| *k == UnderstudentKind::Probe) {
                enc += self.tier_cost[e].unwrap_or(0.0);
            }
        }
        let st = &self.understudents[si];
        let tier = &st.tiers[i];
        if !tier.serving.fit.has_model() {
            return Some(enc);
        }
        let k = st.question.k();
        let sample: Vec<&Example> = st.examples.iter().rev().filter(|e| self.has(i, &e.enc)).take(COST_SAMPLE).collect();
        if sample.is_empty() {
            return Some(enc);
        }
        let clock = Instant::now();
        for e in &sample {
            std::hint::black_box(tier.serving.predict(&tier.fz, &e.enc, k));
        }
        Some(enc + clock.elapsed().as_secs_f64() * 1e6 / sample.len() as f64)
    }

    /// Answer one state for the understudents `which` (default: all). `id` reaches the lead (replay leads key
    /// by it) and the decisions.
    ///
    /// Per question, tiers are tried cheapest first (a tier's features are computed only when reached) and the first
    /// that may answer (graduated, confident, not novel, not demoted) does. The request is local when every question
    /// has such a tier. Otherwise it is forwarded: every tier is then computed (the example needs them) and predicts,
    /// and every tier that would have answered is audited by the lead's answer. A local request is audited at
    /// `audit_rate`: the answering tier and any other tier whose prediction was computed are audited.
    pub fn answer(&mut self, id: Option<&str>, state: &Value, which: Option<&[usize]>) -> Result<Outcome> {
        let sis: Vec<usize> = which.map(<[usize]>::to_vec).unwrap_or_else(|| (0..self.understudents.len()).collect());
        if sis.is_empty() {
            bail!("no questions to answer");
        }
        let nt = self.tier_fz.len();
        let t0 = Instant::now();
        // every question lead-only: forward at once, with no features and no cache or example lookup
        let all_lead_only = sis.iter().all(|&i| self.understudents[i].lead_only.is_some());
        let mut w = if all_lead_only {
            let full = self.fz.full_texts(state);
            Work { key: self.fz.cache_key(&full), enc: Default::default(), full, rules: None, shape: None, memo: None, aux: None, held: None }
        } else {
            self.work(state)
        };
        let mut preds: Vec<Vec<Option<Pred>>> = vec![vec![None; nt]; sis.len()];
        let mut chosen: Vec<Option<usize>> = vec![None; sis.len()];
        for t in 0..nt {
            let live: Vec<usize> =
                (0..sis.len()).filter(|&q| chosen[q].is_none() && self.understudents[sis[q]].tries(t) && self.understudents[sis[q]].tiers[t].serving.fit.has_model()).collect();
            if live.is_empty() {
                continue;
            }
            self.ensure(&mut w, &[t])?;
            for q in live {
                let s = &self.understudents[sis[q]];
                let tier = &s.tiers[t];
                let p = tier.serving.predict(&tier.fz, &w.enc, s.question.k());
                if p.local {
                    chosen[q] = Some(t);
                }
                preds[q][t] = Some(p);
            }
            if chosen.iter().all(Option::is_some) {
                break;
            }
        }
        let local = chosen.iter().all(Option::is_some);
        if !local {
            // forwarded: the example needs every tier's features, and every tier predicts (to be audited)
            if !all_lead_only {
                let all = self.all_tiers();
                self.ensure(&mut w, &all)?;
            }
            for q in 0..sis.len() {
                let s = &self.understudents[sis[q]];
                for t in 0..nt {
                    if preds[q][t].is_none() && s.tries(t) && s.tiers[t].serving.fit.has_model() {
                        let tier = &s.tiers[t];
                        preds[q][t] = Some(tier.serving.predict(&tier.fz, &w.enc, s.question.k()));
                    }
                }
            }
        }
        let understudent_us = t0.elapsed().as_micros() as u64;
        // the tier whose prediction a decision reports: the answering one, else the first that was eligible, else
        // the last that predicted at all
        let shown: Vec<usize> = (0..sis.len())
            .map(|q| {
                chosen[q]
                    .or_else(|| (0..nt).find(|&t| preds[q][t].as_ref().is_some_and(|p| p.eligible)))
                    .or_else(|| (0..nt).rev().find(|&t| preds[q][t].is_some()))
                    .unwrap_or(0)
            })
            .collect();
        let none = Pred { p: None, eligible: false, novel: false, local: false };
        let mut decisions: Vec<(String, Decision)> = sis
            .iter()
            .enumerate()
            .map(|(q, &i)| {
                let s = &self.understudents[i];
                let ti = shown[q];
                let p = preds[q][ti].as_ref().unwrap_or(&none);
                let sp = p.p.clone();
                let serving = &s.tiers[ti].serving;
                (s.name.clone(), Decision {
                    id: id.map(String::from),
                    answered_by: "understudent".into(),
                    label: sp.as_ref().map_or(0, |x| argmax(x)),
                    probs: sp.clone().unwrap_or_default(),
                    understudent_label: sp.as_ref().map(|x| argmax(x)),
                    understudent_probs: sp,
                    eligible: p.eligible,
                    novel: p.novel,
                    demoted: serving.demoted,
                    mode: serving.mode().map(|m| m.name().to_string()),
                    audited: false,
                    readout: if s.is_auto() && (s.serving_idx.is_none() || s.lead_only.is_some()) { None } else { s.label(ti) },
                    understudent_us,
                })
            })
            .collect();
        let mut refits = vec![];
        if local {
            if self.lead.is_some() && !self.frozen && self.rng.random() < self.settings.audit_rate {
                match self.ask(id, state, &sis) {
                    Ok(parsed) => {
                        // the audited item also becomes an example, so every tier needs its features
                        let all = self.all_tiers();
                        match self.ensure(&mut w, &all) {
                            Ok(()) => {
                                let audited = self.learn(&sis, &preds, &parsed, state, &w, id, &mut refits);
                                for ((_, d), a) in decisions.iter_mut().zip(audited) {
                                    d.audited = a;
                                }
                            }
                            Err(e) => eprintln!("understudy: audit failed: {e:#}"),
                        }
                    }
                    Err(e) => eprintln!("understudy: audit failed: {e:#}"),
                }
            }
            return Ok(Outcome { answered_by: "understudent", decisions, refits });
        }
        if self.lead.is_none() {
            bail!("not every question may be answered locally, and there is no lead");
        }
        let parsed = self.ask(id, state, &sis)?;
        let audited = if self.frozen { vec![false; sis.len()] } else { self.learn(&sis, &preds, &parsed, state, &w, id, &mut refits) };
        for (((_, d), (lab, p)), a) in decisions.iter_mut().zip(&parsed).zip(audited) {
            d.answered_by = "lead".into();
            d.label = *lab;
            d.probs = p.clone();
            d.audited = a;
        }
        Ok(Outcome { answered_by: "lead", decisions, refits })
    }

    /// The lead's answers for understudents `sis`, parsed: (class, probabilities, one-hot for a bare class).
    fn ask(&self, id: Option<&str>, state: &Value, sis: &[usize]) -> Result<Vec<(usize, Vec<f64>)>> {
        let qs: Map<String, Value> = sis.iter().map(|&i| (self.understudents[i].name.clone(), self.understudents[i].question.wire())).collect();
        let answers = self.lead.as_ref().unwrap().ask(id, state, &qs)?;
        sis.iter()
            .map(|&i| {
                let s = &self.understudents[i];
                let a = answers.get(&s.name).ok_or_else(|| anyhow!("the lead's response has no answer for {:?}", s.name))?;
                let (lab, p) = s.question.parse_answer(a).map_err(|e| anyhow!("the lead's answer for {:?} is malformed: {e}", s.name))?;
                let p = p.unwrap_or_else(|| (0..s.question.k()).map(|c| if c == lab { 1.0 } else { 0.0 }).collect());
                Ok((lab, p))
            })
            .collect()
    }

    /// Store the lead's answers as sampled examples, audit every tier that predicted and was eligible, and refit
    /// the understudents with enough new examples (a ladder's cheap tiers more often). Returns which understudents recorded an
    /// audit.
    fn learn(&mut self, sis: &[usize], preds: &[Vec<Option<Pred>>], parsed: &[(usize, Vec<f64>)], state: &Value, w: &Work, id: Option<&str>,
             refits: &mut Vec<(String, f64, Value)>) -> Vec<bool> {
        let h = state_hash(state);
        let text_len = state_text(state).len();
        let target = self.settings.fit.target;
        let mut audited = vec![];
        for ((&i, p), (lab, probs)) in sis.iter().zip(preds).zip(parsed) {
            self.record(i, Kind::Sampled, h.clone(), text_len, *lab, Some(probs.clone()), w.key.clone(), w.enc.clone(), id);
            if self.understudents[i].lead_only.is_some() || self.awaiting_encoder(i) {
                // no features were computed (lead-only), or the model isn't loaded yet: keep the state to encode it at the
                // re-check, or when the model arrives
                let seq = self.seq;
                self.understudents[i].raw.insert(seq, state.clone());
                self.prune_raw(i);
            }
            let mut a = false;
            for (t, tp) in p.iter().enumerate().filter(|_| !self.no_audit) {
                if let Some(tp) = tp.as_ref().filter(|tp| tp.eligible) {
                    a |= self.understudents[i].tiers[t].serving.audit(argmax(tp.p.as_ref().unwrap()), *lab, target);
                }
            }
            audited.push(a);
            let st = &self.understudents[i];
            let kind = if st.lead_only.is_some() {
                (st.since_refit >= RECHECK_EVERY).then_some(false)
            } else if st.since_refit >= self.settings.refit_every {
                Some(false)
            } else if st.is_ladder() && st.since_cheap >= self.settings.cheap_refit_every {
                Some(true)
            } else {
                None
            };
            if let Some(cheap_only) = kind {
                let (secs, rep) = self.refit_tiers(i, cheap_only);
                refits.push((self.understudents[i].name.clone(), secs, rep));
            }
        }
        audited
    }
}

/// The rules LLM's bearer key from the environment variable named by `key_env` (none when unnamed or unset: a
/// keyless endpoint, or the endpoint says so with a 401).
fn rules_key(key_env: &str) -> Option<String> {
    std::env::var(key_env).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v != "implicit")
}
