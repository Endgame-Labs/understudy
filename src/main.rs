//! The `understudy` command: `replay` (benchmark a stream), `judge` (production: answer or delegate, learn, persist),
//! `train` (optional seeding of a snapshot) and `status`.

use anyhow::{Context, Result, anyhow, bail, ensure};
use understudy::encoder::{Eg2, Encoder, Featurizer, NoEncoder};
use understudy::snapshot;
use understudy::proxy::{Proxy, Settings, check_tiers};
use understudy::question::QuestionFile;
use understudy::readout::ReadoutChoice;
use understudy::replay;
use understudy::understudent::UnderstudentKind;
use understudy::config::{Config, LeadKind};
use understudy::lead::{self, LeadFlags, Replay, Lead};
use clap::{Args, Parser, Subcommand};
use serde_json::{Value, json};
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "understudy", version, about = "A CPU reflex layer in front of a systemone decision model")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Replay a stream (`{"id","state","lead_probs"}` lines) deterministically; decisions to --out, summary to
    /// stderr.
    Replay {
        #[arg(long)]
        question: PathBuf,
        #[arg(long)]
        stream: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        opts: UnderstudentOpts,
    },
    /// Production: answer `{"id","state"}` (or systemone request) lines on stdin locally when the understudent may,
    /// else delegate to the lead; learn from every lead answer; keep auditing; save the snapshot.
    Judge {
        #[arg(long)]
        snapshot: PathBuf,
        /// Required when the snapshot doesn't exist yet; must match it otherwise.
        #[arg(long)]
        question: Option<PathBuf>,
        #[command(flatten)]
        lead: LeadOpts,
        /// Save the snapshot every this many items (and at the end of input).
        #[arg(long, default_value_t = 100)]
        save_every: usize,
        /// Score the stream with the snapshot's fits as they are and learn nothing from it (no examples, audits or
        /// refits; the snapshot is not saved): a fixed held-out test. Forwarded items still ask the lead.
        #[arg(long)]
        frozen: bool,
        #[command(flatten)]
        opts: UnderstudentOpts,
    },
    /// Seed a snapshot (new or used) with examples: `{"id","state","lead_probs"}` (a lead answer),
    /// `{"id","state","truth"}` (ground truth, weight 2) or `{"id","state"}` (the lead is asked); then refit and
    /// save. Prints the fit report.
    Train {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        question: Option<PathBuf>,
        #[arg(long)]
        examples: PathBuf,
        /// Keep up to this many sampled examples (lead answers) instead of the default 2000, whose oldest are evicted.
        /// Stored in the snapshot. Memory and fit time grow with it.
        #[arg(long)]
        max_examples: Option<usize>,
        #[command(flatten)]
        lead: LeadOpts,
        #[command(flatten)]
        opts: UnderstudentOpts,
    },
    /// A drop-in `POST /v1/systemone` HTTP proxy: forward to Jev/Rune, learn from the answers, and (in `serve` mode)
    /// answer graduated questions locally. See SPEC.md, "serve".
    Serve {
        #[command(flatten)]
        args: understudy::serve::ServeArgs,
        #[command(flatten)]
        opts: UnderstudentOpts,
    },
    /// Download or inspect the EmbeddingGemma 2 checkpoint.
    #[command(subcommand)]
    Model(ModelCmd),
    /// A snapshot's state per question, without loading the encoder.
    Status {
        #[arg(long)]
        snapshot: PathBuf,
    },
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Download config.json, tokenizer.json and model.safetensors from the Hugging Face hub (no token needed),
    /// verified by SHA-256, into the hub cache (or --dir). `HF_ENDPOINT` changes the host; `HF_TOKEN` is sent if set.
    Fetch {
        /// A full commit hash (default: the pinned revision, the only one with built-in checksums).
        #[arg(long)]
        revision: Option<String>,
        /// Write the three files here instead of the Hugging Face cache layout.
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Show which model directory is used and whether its files match the pinned hashes (hashes ~1.5 GB).
    Status,
}

/// `--understudent`: which understudent to run.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum UnderstudentArg {
    #[value(name = "deep")]
    Deep,
    #[value(name = "probe")]
    Probe,
    #[value(name = "keywords")]
    Keywords,
    #[value(name = "lite")]
    Lite,
    #[value(name = "neighbors")]
    Neighbors,
    #[value(name = "rules")]
    Rules,
    #[value(name = "shape")]
    Shape,
    #[value(name = "blend")]
    Blend,
    #[value(name = "memo")]
    Memo,
    #[value(name = "net")]
    Net,
    #[value(name = "checklist")]
    Checklist,
    #[value(name = "ladder")]
    Ladder,
    #[value(name = "auto")]
    Auto,
}

#[derive(Args, Clone)]
struct UnderstudentOpts {
    /// Which understudent: deep, probe, keywords, lite, neighbors, rules, shape, blend, memo, net, checklist, ladder, auto (race the candidates, serve one) [default: deep].
    #[arg(long, value_enum)]
    understudent: Option<UnderstudentArg>,
    /// A ladder's tiers, cheapest first, from rules (first only), keywords, lite, probe, deep, shape, blend, memo, net, checklist [default: keywords,lite,probe,deep].
    #[arg(long, value_delimiter = ',', value_enum)]
    tiers: Option<Vec<UnderstudentKind>>,
    /// Auto's candidates, cheapest first, from rules (first only), keywords, lite, probe, deep, shape, blend, memo, net, checklist [default: keywords,lite,probe,deep].
    #[arg(long, value_delimiter = ',', value_enum)]
    candidates: Option<Vec<UnderstudentKind>>,
    /// Graduation's abstention policy: `default` (one confidence threshold) or `calibrated` (a frozen model with one threshold per
    /// class: fitted on a train part, thresholds chosen on a select part, the exact policy certified once on a third,
    /// group-disjoint part; see SPEC.md) [default: default].
    #[arg(long, value_enum)]
    abstain: Option<understudy::abstain::Abstain>,
    /// How graduation treats being re-tested at every refit: `pointwise` (every refit tests z = 1.96) or `sequential`
    /// (alpha-spending across a question's attempts, Bonferroni over a ladder's or auto's candidates; see SPEC.md)
    /// [default: pointwise].
    #[arg(long, value_enum)]
    graduation: Option<understudy::understudent::Graduation>,
    /// Auto: examples after which a question where no candidate graduated or came within 0.05 of --target becomes
    /// lead_only [default: 600].
    #[arg(long)]
    give_up_after: Option<usize>,
    /// Auto: among the graduating candidates serve the cheapest (by a deterministic cost model: design, encoder layers
    /// run, tokens; never wall-clock) whose cv_coverage is within EPS (absolute, e.g. 0.05) of the best. Off by default.
    #[arg(long, value_name = "EPS")]
    prefer_fast: Option<f64>,
    /// Encoder layers read by the readouts (deep, ladder) [default: 4,8,12].
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,
    /// The attention readout's layer (one of --layers) [default: the middle one].
    #[arg(long)]
    attn_layer: Option<usize>,
    /// Clip long states for deep (and a ladder's deep tier) like hunch instead of encoding them in chunks.
    #[arg(long)]
    no_chunk: bool,
    /// Graduation's minority-recall bar [default: 0.8].
    #[arg(long)]
    minority: Option<f64>,
    /// [default: auto]
    #[arg(long, value_enum)]
    readout: Option<ReadoutChoice>,
    /// Share of local answers audited by the lead [default: 0.1].
    #[arg(long)]
    audit_rate: Option<f64>,
    /// Graduation's agreement bar [default: 0.9].
    #[arg(long)]
    target: Option<f64>,
    /// Seed for folds, audits and attention training [default: 0].
    #[arg(long)]
    seed: Option<u64>,
    /// Examples before graduation is tried [default: 60, hunch's].
    #[arg(long)]
    min_examples: Option<usize>,
    /// The `rules` tier's LLM endpoint (chat completions) [default: the lead config file's `[rules]`, else Fireworks'
    /// public endpoint].
    #[arg(long)]
    rules_llm_url: Option<String>,
    /// The `rules` tier's LLM model [default: accounts/fireworks/models/glm-5p3-flash].
    #[arg(long)]
    rules_llm_model: Option<String>,
    /// The environment variable holding the `rules` LLM's bearer key [default: `[rules]` key_env, else FIREWORKS_API_KEY].
    #[arg(long, value_name = "VAR")]
    rules_llm_key_env: Option<String>,
    /// Group examples by their id's prefix before this character, so cross-validation (and the rules understudent's
    /// held-out split) never separates the sub-items `<item>#<k>` of one item; `none` turns it off [default: #].
    #[arg(long, value_name = "CHAR|none")]
    group_by_id_prefix: Option<String>,
    /// A directory caching encoded texts on disk (one entry per text and featurizer: what probe, deep and lite read),
    /// shared across runs and systems; results are identical with and without it.
    #[arg(long)]
    encode_cache: Option<PathBuf>,
    /// A JSONL file caching the rules LLM's requests and responses: loaded first, live calls appended. With a warm
    /// cache a replay is offline and deterministic.
    #[arg(long)]
    rules_cache: Option<PathBuf>,
}

#[derive(Args, Clone)]
struct LeadOpts {
    /// A named lead from the lead config file ($UNDERSTUDY_LEADS or ~/.config/understudy/leads.toml; see
    /// examples/leads.example.toml).
    #[arg(long)]
    lead: Option<String>,
    /// The lead's endpoint (overrides the preset's). A chat lead takes a base URL or the full /chat/completions URL; a clef
    /// lead takes the API base (default https://api.cloudflare.com, or $CLOUDFLARE_API_BASE).
    #[arg(long)]
    lead_url: Option<String>,
    #[arg(long)]
    lead_model: Option<String>,
    /// What the lead speaks: systemone, chat or clef [default: the preset's, else systemone].
    #[arg(long, value_parser = LeadKind::parse)]
    lead_kind: Option<LeadKind>,
    /// Send `Authorization: Bearer $VAR` to the lead, VAR being the name of an environment variable (the key is
    /// never taken from argv, nor logged or written to a snapshot). Overrides the preset's `key_env`.
    #[arg(long, value_name = "VAR")]
    lead_key_env: Option<String>,
    /// Answer from a stream.jsonl's lead_probs, by id (offline, deterministic).
    #[arg(long, conflicts_with_all = ["lead", "lead_url", "lead_model", "lead_kind", "lead_key_env"])]
    replay_lead: Option<PathBuf>,
}

impl LeadOpts {
    fn flags(&self) -> LeadFlags<'_> {
        LeadFlags { name: self.lead.as_deref(), url: self.lead_url.as_deref(), model: self.lead_model.as_deref(), kind: self.lead_kind, key_env: self.lead_key_env.as_deref() }
    }

    /// The lead the flags name, else the lead a snapshot recorded (its URL and model; the key comes from
    /// `--lead-key-env`), else none.
    fn build(&self, recorded: Option<&Value>) -> Result<Option<Box<dyn Lead>>> {
        if let Some(p) = &self.replay_lead {
            return Ok(Some(Box::new(Replay::from_stream(p)?)));
        }
        if self.lead.is_some() || self.lead_url.is_some() || self.lead_model.is_some() || self.lead_kind.is_some() {
            return Ok(Some(lead::build(&Config::load, &self.flags())?));
        }
        if let Some(r) = recorded.filter(|r| r["kind"] == "http" || r["kind"] == "chat" || r["kind"] == "clef") {
            let (u, m) = (r["url"].as_str().unwrap_or_default(), r["model"].as_str().unwrap_or_default());
            let kind = match r["kind"].as_str() {
                Some("chat") => LeadKind::Chat,
                Some("clef") => LeadKind::Clef,
                _ => LeadKind::SystemOne,
            };
            let f = LeadFlags { url: Some(u), model: Some(m), kind: Some(kind), key_env: self.lead_key_env.as_deref(), ..Default::default() };
            return Ok(Some(lead::build(&Config::load, &f)?));
        }
        Ok(None)
    }
}

impl UnderstudentOpts {
    fn settings(&self) -> Settings {
        let mut s = Settings::default();
        s.fit.understudent = self.kind();
        s.fit.tiers = self.tier_list();
        if let Some(m) = self.minority {
            s.fit.minority = m;
        }
        if let Some(r) = self.readout {
            s.fit.readout = r;
        }
        if let Some(a) = self.audit_rate {
            s.audit_rate = a;
        }
        if let Some(t) = self.target {
            s.fit.target = t;
        }
        if let Some(x) = self.seed {
            s.fit.seed = x;
        }
        if let Some(m) = self.min_examples {
            s.fit.min_examples = m;
        }
        if let Some(a) = self.abstain {
            s.fit.abstain = a;
        }
        if let Some(g) = self.graduation {
            s.fit.graduation = g;
        }
        if let Some(g) = self.give_up_after {
            s.give_up_after = g;
        }
        if let Some(e) = self.prefer_fast {
            s.prefer_fast = Some(e);
        }
        if let Some(g) = &self.group_by_id_prefix {
            s.group_sep = if g == "none" { None } else { g.chars().next() };
        }
        match Config::load() {
            Ok(c) => s.rules = s.rules.with_config(&c),
            Err(e) => eprintln!("understudy: ignoring the lead config for the rules LLM: {e:#}"),
        }
        if let Some(u) = &self.rules_llm_url {
            s.rules.url = understudy::chat::endpoint(u);
        }
        if let Some(k) = &self.rules_llm_key_env {
            s.rules.key_env = k.clone();
        }
        if let Some(m) = &self.rules_llm_model {
            s.rules.model = m.clone();
        }
        s
    }

    /// Attach the rules LLM's cache files: the snapshot's (when `dir`) and `--rules-cache`.
    fn attach_rules_cache(&self, p: &Proxy, dir: Option<&Path>) -> Result<()> {
        let Some(c) = &p.rules else { return Ok(()) };
        if let Some(f) = &self.rules_cache {
            c.attach(f)?;
        }
        if let Some(d) = dir {
            std::fs::create_dir_all(d)?;
            c.attach(&d.join("rules_llm.jsonl"))?;
        }
        Ok(())
    }

    fn attach_encode_cache(&self, p: &mut Proxy) -> Result<()> {
        if let Some(d) = &self.encode_cache {
            p.set_encode_cache(d)?;
        }
        Ok(())
    }

    fn kind(&self) -> UnderstudentKind {
        match self.understudent {
            None | Some(UnderstudentArg::Deep) => UnderstudentKind::Deep,
            Some(UnderstudentArg::Probe) => UnderstudentKind::Probe,
            Some(UnderstudentArg::Keywords) => UnderstudentKind::Keywords,
            Some(UnderstudentArg::Lite) => UnderstudentKind::Lite,
            Some(UnderstudentArg::Neighbors) => UnderstudentKind::Neighbors,
            Some(UnderstudentArg::Rules) => UnderstudentKind::Rules,
            Some(UnderstudentArg::Shape) => UnderstudentKind::Shape,
            Some(UnderstudentArg::Blend) => UnderstudentKind::Blend,
            Some(UnderstudentArg::Memo) => UnderstudentKind::Memo,
            Some(UnderstudentArg::Net) => UnderstudentKind::Net,
            Some(UnderstudentArg::Checklist) => UnderstudentKind::Checklist,
            Some(UnderstudentArg::Ladder) => UnderstudentKind::Ladder,
            Some(UnderstudentArg::Auto) => UnderstudentKind::Auto,
        }
    }

    /// A ladder's tiers (empty for other understudents).
    fn tier_list(&self) -> Vec<UnderstudentKind> {
        match (self.understudent, &self.tiers) {
            (Some(UnderstudentArg::Ladder), Some(t)) => t.clone(),
            (Some(UnderstudentArg::Ladder), None) => UnderstudentKind::default_tiers(),
            (Some(UnderstudentArg::Auto), _) => self.candidates.clone().unwrap_or_else(UnderstudentKind::default_tiers),
            _ => vec![],
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.tiers.is_none() || self.understudent == Some(UnderstudentArg::Ladder), "--tiers only applies to --understudent ladder");
        ensure!(self.candidates.is_none() || self.understudent == Some(UnderstudentArg::Auto), "--candidates only applies to --understudent auto");
        ensure!(self.give_up_after.is_none() || self.understudent == Some(UnderstudentArg::Auto), "--give-up-after only applies to --understudent auto");
        ensure!(self.prefer_fast.is_none() || self.understudent == Some(UnderstudentArg::Auto), "--prefer-fast only applies to --understudent auto");
        ensure!(self.prefer_fast.is_none_or(|e| (0.0..=1.0).contains(&e)), "--prefer-fast wants a coverage margin between 0 and 1");
        if self.kind().is_tiered() {
            check_tiers(&self.tier_list())?;
        }
        let deep = match self.kind() {
            UnderstudentKind::Deep | UnderstudentKind::Net => true,
            UnderstudentKind::Ladder | UnderstudentKind::Auto => self.tier_list().iter().any(|t| matches!(t, UnderstudentKind::Deep | UnderstudentKind::Net)),
            _ => false,
        };
        ensure!(deep || (self.layers.is_none() && self.attn_layer.is_none() && self.readout.is_none()), "--layers, --attn-layer and --readout only apply to deep, net and a ladder with a deep or net tier");
        Ok(())
    }

    fn featurizer(&self, pair: Option<(String, String)>, hidden: usize) -> Result<Featurizer> {
        self.validate()?;
        let mut f = Featurizer::for_understudent(self.kind(), self.layers.clone(), self.attn_layer, pair, hidden)?;
        if self.no_chunk && matches!(self.kind(), UnderstudentKind::Deep | UnderstudentKind::Net | UnderstudentKind::Ladder | UnderstudentKind::Auto) {
            f.chunk = false;
        }
        Ok(f)
    }

    /// Flags that would change a stored snapshot's features or fits must agree with it (the audit rate may change).
    fn check_against(&self, p: &Proxy) -> Result<()> {
        let s = &p.settings;
        ensure!(self.understudent.is_none() || self.kind() == s.fit.understudent, "--understudent differs from the snapshot's {}", s.fit.understudent.name());
        ensure!(self.tiers.is_none() || self.tier_list() == s.fit.tiers, "--tiers differs from the snapshot's {:?}", s.fit.tiers);
        ensure!(self.candidates.is_none() || self.tier_list() == s.fit.tiers, "--candidates differs from the snapshot's {:?}", s.fit.tiers);
        ensure!(!self.no_chunk || !p.fz.chunk || !matches!(s.fit.understudent, UnderstudentKind::Deep | UnderstudentKind::Ladder | UnderstudentKind::Auto), "--no-chunk differs from the snapshot's featurizer (chunking on)");
        ensure!(self.minority.is_none_or(|m| m == s.fit.minority), "--minority differs from the snapshot's {}", s.fit.minority);
        ensure!(self.layers.as_ref().is_none_or(|l| *l == p.fz.layers), "--layers differs from the snapshot's {:?}", p.fz.layers);
        ensure!(self.attn_layer.is_none_or(|l| l == p.fz.attn_layer), "--attn-layer differs from the snapshot's {}", p.fz.attn_layer);
        ensure!(self.readout.is_none_or(|r| r == s.fit.readout), "--readout differs from the snapshot's {:?}", s.fit.readout);
        ensure!(self.abstain.is_none_or(|a| a == s.fit.abstain), "--abstain differs from the snapshot's {:?}", s.fit.abstain);
        ensure!(self.graduation.is_none_or(|g| g == s.fit.graduation), "--graduation differs from the snapshot's {:?}", s.fit.graduation);
        ensure!(self.prefer_fast.is_none() || self.prefer_fast == s.prefer_fast, "--prefer-fast differs from the snapshot's {:?}", s.prefer_fast);
        ensure!(self.target.is_none_or(|t| t == s.fit.target), "--target differs from the snapshot's {}", s.fit.target);
        ensure!(self.seed.is_none_or(|x| x == s.fit.seed), "--seed differs from the snapshot's {}", s.fit.seed);
        ensure!(self.min_examples.is_none_or(|m| m == s.fit.min_examples), "--min-examples differs from the snapshot's");
        Ok(())
    }
}

/// The model, unless the understudent never reads it (`keywords`, or a ladder of keywords alone, starts without loading
/// EmbeddingGemma at all).
fn load_encoder(kind: UnderstudentKind, tiers: &[UnderstudentKind]) -> Result<Arc<dyn Encoder>> {
    let needs = match kind {
        UnderstudentKind::Ladder | UnderstudentKind::Auto => tiers.iter().any(|t| t.needs_encoder()),
        k => k.needs_encoder(),
    };
    if !needs {
        return Ok(Arc::new(NoEncoder));
    }
    Ok(Arc::new(Eg2::load_default()?))
}

/// Open the snapshot at `dir`, or create one for `question`. Returns the proxy and the question's understudent index.
fn open_snapshot(dir: &Path, question: Option<&Path>, opts: &UnderstudentOpts, lead: &LeadOpts) -> Result<(Proxy, Option<usize>)> {
    let qf = question.map(QuestionFile::load).transpose()?;
    if snapshot::exists(dir) {
        let meta = snapshot::read_meta(dir)?;
        let stored: UnderstudentKind = serde_json::from_value(meta["settings"]["fit"].get("understudent").cloned().unwrap_or_default()).unwrap_or_default();
        let stored_tiers: Vec<UnderstudentKind> = serde_json::from_value(meta["settings"]["fit"]["tiers"].clone()).unwrap_or_default();
        let encoder = load_encoder(stored, &stored_tiers)?;
        let t = lead.build(meta.get("lead"))?;
        let audit = opts.audit_rate;
        let p = snapshot::load(dir, encoder, t, |s| {
            if let Some(a) = audit {
                s.audit_rate = a;
            }
        })?;
        opts.check_against(&p)?;
        let mut p = p;
        if let Some(g) = &opts.group_by_id_prefix {
            p.settings.group_sep = if g == "none" { None } else { g.chars().next() };
        }
        opts.attach_encode_cache(&mut p)?;
        opts.attach_rules_cache(&p, Some(dir))?;
        let si = match &qf {
            Some(qf) => {
                let q = qf.question()?;
                ensure!(qf.pair == p.fz.pair, "question.json's pair {:?} differs from the snapshot's {:?}", qf.pair, p.fz.pair);
                Some(p.understudents.iter().position(|s| s.key == q.key()).ok_or_else(|| {
                    anyhow!("{} (key {}) is not this snapshot's question; it has {:?}", qf.name, q.key(),
                            p.understudents.iter().map(|s| (&s.name, &s.key)).collect::<Vec<_>>())
                })?)
            }
            None => None,
        };
        return Ok((p, si));
    }
    let qf = qf.ok_or_else(|| anyhow!("{} holds no snapshot yet: give --question to create one", dir.display()))?;
    opts.validate()?;
    let encoder = load_encoder(opts.kind(), &opts.tier_list())?;
    let fz = opts.featurizer(qf.pair.clone(), encoder.hidden())?;
    let mut p = Proxy::new(opts.settings(), fz, encoder, lead.build(None)?);
    opts.attach_encode_cache(&mut p)?;
    let si = p.understudent(&qf.name, &qf.question()?);
    opts.attach_rules_cache(&p, Some(dir))?;
    Ok((p, Some(si)))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Replay { question, stream, out, opts } => {
            let qf = QuestionFile::load(&question)?;
            opts.validate()?;
            let encoder = load_encoder(opts.kind(), &opts.tier_list())?;
            let fz = opts.featurizer(qf.pair.clone(), encoder.hidden())?;
            let lead = Replay::from_stream(&stream)?;
            let mut p = Proxy::new(opts.settings(), fz, encoder, Some(Box::new(lead)));
            opts.attach_encode_cache(&mut p)?;
            p.understudent(&qf.name, &qf.question()?);
            opts.attach_rules_cache(&p, None)?;
            let input = std::io::BufReader::new(std::fs::File::open(&stream).with_context(|| stream.display().to_string())?);
            let output = BufWriter::new(std::fs::File::create(&out).with_context(|| out.display().to_string())?);
            let summary = replay::run(&mut p, input, output)?;
            eprintln!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Cmd::Judge { snapshot: dir, question, lead, save_every, frozen, opts } => judge(&dir, question.as_deref(), &lead, save_every, frozen, &opts)?,
        Cmd::Train { snapshot: dir, question, examples, max_examples, lead, opts } => train(&dir, question.as_deref(), &examples, max_examples, &lead, &opts)?,
        Cmd::Model(ModelCmd::Fetch { revision, dir }) => {
            let d = understudy::modelfetch::fetch(revision.as_deref(), dir.as_deref())?;
            println!("{}", d.display());
        }
        Cmd::Model(ModelCmd::Status) => {
            if !understudy::modelfetch::status()? {
                std::process::exit(1);
            }
        }
        Cmd::Status { snapshot: dir } => status(&dir)?,
        Cmd::Serve { args, opts } => serve(args, opts)?,
    }
    Ok(())
}

/// `serve`: one `Proxy` per upstream/model namespace, built (or loaded from `--snapshot-dir`) by the factory below.
fn serve(args: understudy::serve::ServeArgs, mut opts: UnderstudentOpts) -> Result<()> {
    if opts.understudent.is_none() {
        opts.understudent = Some(UnderstudentArg::Auto);
    }
    if opts.kind() == UnderstudentKind::Auto {
        return serve_auto(args, opts);
    }
    opts.validate()?;
    let encoder = load_encoder(opts.kind(), &opts.tier_list())?;
    let audit = opts.settings().audit_rate;
    let o = opts.clone();
    let factory: understudy::serve::ProxyFactory = Arc::new(move |dir, lead| {
        if let Some(d) = dir.filter(|d| snapshot::exists(d)) {
            let a = o.audit_rate;
            let mut p = snapshot::load(d, encoder.clone(), Some(lead), |s| {
                if let Some(a) = a {
                    s.audit_rate = a;
                }
            })?;
            o.check_against(&p)?;
            if let Some(g) = &o.group_by_id_prefix {
                p.settings.group_sep = if g == "none" { None } else { g.chars().next() };
            }
            o.attach_encode_cache(&mut p)?;
            o.attach_rules_cache(&p, Some(d))?;
            return Ok(p);
        }
        let fz = o.featurizer(None, encoder.hidden())?;
        let mut p = Proxy::new(o.settings(), fz, encoder.clone(), Some(lead));
        o.attach_encode_cache(&mut p)?;
        o.attach_rules_cache(&p, dir)?;
        Ok(p)
    });
    understudy::serve::run(args.config(audit, factory)?)
}

/// `serve --understudent auto` (the default): the race starts with the designs that need no model (keywords, shape,
/// checklist, memo, and rules when a rules LLM is configured) so the server never waits on, or fails for, the
/// embedding model. The model is loaded when it is on disk and otherwise fetched in the background (unless
/// `--no-model-download`); once it is ready the encoder designs (lite, probe, deep) join every namespace's race.
fn serve_auto(args: understudy::serve::ServeArgs, opts: UnderstudentOpts) -> Result<()> {
    let lazy = understudy::lazyenc::LazyEncoder::eg2(!args.no_model_download);
    understudy::serve::run(auto_config(args, opts, lazy)?)
}

/// The `serve` configuration for `--understudent auto` over the given lazily loaded model.
fn auto_config(args: understudy::serve::ServeArgs, opts: UnderstudentOpts, lazy: Arc<understudy::lazyenc::LazyEncoder>) -> Result<understudy::serve::ServeConfig> {
    let rules_on = opts.rules_llm_url.is_some() || opts.rules_llm_model.is_some() || Config::load().ok().is_some_and(|c| c.rules.is_some());
    let wanted: Vec<UnderstudentKind> = opts.candidates.clone().unwrap_or_else(|| {
        let mut v = vec![];
        if rules_on {
            v.push(UnderstudentKind::Rules);
        }
        v.extend([UnderstudentKind::Keywords, UnderstudentKind::Shape, UnderstudentKind::Checklist, UnderstudentKind::Memo, UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep]);
        v
    });
    let (free, later): (Vec<_>, Vec<_>) = wanted.iter().copied().partition(|k| !k.needs_encoder());
    let mut full = opts.clone();
    full.candidates = Some(wanted.clone());
    full.validate()?;
    let mut chk = opts.clone();
    chk.candidates = None; // a snapshot's candidates may have grown since it was made
    if free.is_empty() {
        // nothing to start with: the encoder is needed up front (a clear error names `understudy model fetch`)
        let encoder = load_encoder(UnderstudentKind::Auto, &wanted)?;
        return serve_config(args, full, chk, encoder);
    }
    let mut base = opts.clone();
    base.candidates = Some(free);
    let audit = opts.settings().audit_rate;
    let (lz, b, f, c) = (lazy.clone(), base, full, chk.clone());
    let factory: understudy::serve::ProxyFactory = Arc::new(move |dir, lead| {
        lz.start();
        if let Some(d) = dir.filter(|d| snapshot::exists(d)) {
            let meta = snapshot::read_meta(d)?;
            let enc: Arc<dyn Encoder> = if meta["encoder"] == NoEncoder.fingerprint() { Arc::new(NoEncoder) } else { lz.wait()? };
            return load_for_serve(d, enc, lead, &c);
        }
        let (enc, o): (Arc<dyn Encoder>, &UnderstudentOpts) = match lz.ready() {
            Some(e) => (e, &f),
            None => (Arc::new(NoEncoder), &b),
        };
        let fz = o.featurizer(None, enc.hidden())?;
        let mut p = Proxy::new(o.settings(), fz, enc, Some(lead));
        o.attach_encode_cache(&mut p)?;
        o.attach_rules_cache(&p, dir)?;
        Ok(p)
    });
    let mut cfg = args.config(audit, factory)?;
    let lz = lazy.clone();
    cfg.upgrade = Some(Arc::new(move |p: &mut Proxy| match lz.ready() {
        Some(e) if p.settings.fit.understudent == UnderstudentKind::Auto => p.add_encoder_tiers(e, &later),
        _ => Ok(false),
    }));
    Ok(cfg)
}

/// Load a namespace's saved snapshot for `serve`.
fn load_for_serve(d: &Path, encoder: Arc<dyn Encoder>, lead: Box<dyn Lead>, o: &UnderstudentOpts) -> Result<Proxy> {
    let a = o.audit_rate;
    let mut p = snapshot::load(d, encoder, Some(lead), |s| {
        if let Some(a) = a {
            s.audit_rate = a;
        }
    })?;
    o.check_against(&p)?;
    if let Some(g) = &o.group_by_id_prefix {
        p.settings.group_sep = if g == "none" { None } else { g.chars().next() };
    }
    o.attach_encode_cache(&mut p)?;
    o.attach_rules_cache(&p, Some(d))?;
    Ok(p)
}

fn serve_config(args: understudy::serve::ServeArgs, opts: UnderstudentOpts, chk: UnderstudentOpts, encoder: Arc<dyn Encoder>) -> Result<understudy::serve::ServeConfig> {
    let audit = opts.settings().audit_rate;
    let factory: understudy::serve::ProxyFactory = Arc::new(move |dir, lead| {
        if let Some(d) = dir.filter(|d| snapshot::exists(d)) {
            return load_for_serve(d, encoder.clone(), lead, &chk);
        }
        let fz = opts.featurizer(None, encoder.hidden())?;
        let mut p = Proxy::new(opts.settings(), fz, encoder.clone(), Some(lead));
        opts.attach_encode_cache(&mut p)?;
        opts.attach_rules_cache(&p, dir)?;
        Ok(p)
    });
    args.config(audit, factory)
}

fn judge(dir: &Path, question: Option<&Path>, lead: &LeadOpts, save_every: usize, frozen: bool, opts: &UnderstudentOpts) -> Result<()> {
    let (mut p, _) = open_snapshot(dir, question, opts, lead)?;
    p.frozen = frozen;
    ensure!(p.lead.is_some(), "judge needs a lead: --lead (a preset in the lead config file), --lead-url/--lead-model or --replay-lead");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut since_save = 0;
    let t0 = std::time::Instant::now();
    let mut summary = replay::Summary::default();
    for (n, line) in std::io::stdin().lock().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let res = (|| -> Result<Value> {
            let v: Value = serde_json::from_str(&line).context("not JSON")?;
            let (out, o) = understudy::ops::judge_line(&mut p, &v)?;
            summary.add(summary.items, &o.decisions[0].1, None);
            summary.items += 1;
            for (name, secs, rep) in &o.refits {
                eprintln!("understudy: refit {name} in {secs:.2}s: n={} readout={} mode={} cv_coverage={}", rep["n"], rep["readout"], rep["mode"], rep["cv_coverage"]);
            }
            Ok(out)
        })();
        let v = res.unwrap_or_else(|e| {
            let id = serde_json::from_str::<Value>(&line).ok().and_then(|v| v.get("id").cloned()).unwrap_or(Value::Null);
            json!({"id": id, "error": format!("{e:#}"), "line": n + 1})
        });
        serde_json::to_writer(&mut out, &v)?;
        out.write_all(b"\n")?;
        out.flush()?;
        since_save += 1;
        if since_save >= save_every && !frozen {
            snapshot::save(&p, dir)?;
            since_save = 0;
        }
    }
    if !frozen {
        snapshot::save(&p, dir)?;
    }
    let st: Vec<Value> = p.understudents.iter().map(|s| s.status()).collect();
    let mut sum = summary.finish(t0.elapsed().as_secs_f64(), st.first().cloned().unwrap_or(Value::Null));
    sum["final"] = json!(st);
    sum["understudent"] = json!(p.settings.fit.understudent.name());
    eprintln!("{}", serde_json::to_string(&sum)?);
    Ok(())
}

fn train(dir: &Path, question: Option<&Path>, examples: &Path, max_examples: Option<usize>, lead: &LeadOpts, opts: &UnderstudentOpts) -> Result<()> {
    let (mut p, si) = open_snapshot(dir, question, opts, lead)?;
    if let Some(m) = max_examples {
        ensure!(m > 0, "--max-examples must be positive");
        p.settings.sampled_cap = m;
    }
    let si = match si {
        Some(i) => i,
        None if p.understudents.len() == 1 => 0,
        None => bail!("this snapshot has several questions: give --question"),
    };
    let text = std::fs::read_to_string(examples).with_context(|| examples.display().to_string())?;
    let rows: Vec<Value> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{}:{}", examples.display(), i + 1)))
        .collect::<Result<_>>()?;
    let added = understudy::ops::seed(&mut p, si, &rows)?;
    let (secs, report) = p.refit(si);
    snapshot::save(&p, dir)?;
    let st = p.understudents[si].status();
    println!("{}", serde_json::to_string_pretty(&json!({
        "snapshot": dir, "question": st["name"], "key": st["key"], "state": st["state"],
        "added": added,
        "n": report["n"], "cv_agreement": report["cv_agreement"], "cv_minority_recall": report["cv_minority_recall"],
        "mode": report["mode"], "threshold": report["threshold"], "cv_coverage": report["cv_coverage"],
        "readout": report["readout"], "cv_agreement_by_readout": report["cv_agreement_by_readout"],
        "why_not": report["why_not"], "refit_secs": secs,
    }))?);
    Ok(())
}

fn status(dir: &Path) -> Result<()> {
    let m = snapshot::read_meta(dir)?;
    let questions: Vec<Value> = m["understudents"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|s| {
            // a ladder's lead tier stands for the question: the first graduated and not demoted, else the first
            // graduated, else the last
            let tiers = s["tiers"].as_array().cloned().unwrap_or_default();
            let tier_state = |t: &Value| if t["fit"]["mode"].is_null() { "learning" } else if t["demoted"] == true { "demoted" } else { "graduated" };
            let serving = s["serving"].as_u64().and_then(|i| tiers.get(i as usize));
            let lead: Value = serving
                .or_else(|| tiers.iter().find(|t| tier_state(t) == "graduated"))
                .or_else(|| tiers.iter().find(|t| !t["fit"]["mode"].is_null()))
                .or(tiers.last())
                .cloned()
                .unwrap_or_else(|| s.clone());
            let tier_list: Value = if tiers.is_empty() { Value::Null } else {
                json!(tiers.iter().map(|t| json!({"tier": t["kind"], "state": tier_state(t), "mode": t["fit"]["mode"], "threshold": t["fit"]["thr"],
                    "readout": t["fit"]["readout"], "cv_agreement": t["fit"]["report"]["cv_agreement"], "cv_coverage": t["fit"]["report"]["cv_coverage"],
                    "audits": t["audits"].as_array().map_or(0, Vec::len), "rules": t["fit"]["rules"]})).collect::<Vec<_>>())
            };
            let (serving_name, lead_only, race) = (serving.map(|t| t["kind"].clone()), s["lead_only"].clone(), s["race"].clone());
            let s = &json!({"name": s["name"], "key": s["key"], "question": s["question"], "examples": s["examples"], "since_refit": s["since_refit"],
                            "lead_only": s["lead_only"],
                            "fit": lead["fit"], "demoted": lead["demoted"], "audits": lead["audits"]});
            let mode = s["fit"]["mode"].clone();
            let demoted = s["demoted"].as_bool().unwrap_or(false);
            let state = if !s["lead_only"].is_null() { "lead_only" } else if mode.is_null() { "learning" } else if demoted { "demoted" } else { "graduated" };
            let window: Vec<&Value> = s["audits"].as_array().map(|a| a.iter().filter(|x| x["mode"] == mode).collect()).unwrap_or_default();
            let agreed = window.iter().filter(|x| x["agreed"] == true).count();
            let missed = window.iter().filter(|x| x["missed"] == true).count();
            let r = &s["fit"]["report"];
            json!({
                "name": s["name"], "key": s["key"], "type": s["question"]["type"], "state": state,
                "examples": s["examples"].as_array().map_or(0, Vec::len),
                "labelled": s["examples"].as_array().map_or(0, |a| a.iter().filter(|e| e["kind"] == "labelled").count()),
                "since_refit": s["since_refit"],
                "mode": mode, "threshold": s["fit"]["thr"], "readout": s["fit"]["readout"],
                "cv_agreement": r["cv_agreement"], "cv_minority_recall": r["cv_minority_recall"], "cv_coverage": r["cv_coverage"],
                "why_not": r["why_not"], "tiers": tier_list,
                "serving": serving_name, "lead_only": lead_only, "race": race,
                "rules": tiers.iter().find(|t| !t["fit"]["rules"].is_null()).map_or(s["fit"]["rules"].clone(), |t| t["fit"]["rules"].clone()),
                "checklist": tiers.iter().map(|t| &t["fit"]["report"]["checklist"]).find(|d| !d.is_null()).cloned().unwrap_or_else(|| r["checklist"].clone()),
                "abstain": r["abstain"], "class_thr": s["fit"]["class_thr"],
                "audits": {"n": window.len(), "agreement": if window.is_empty() { Value::Null } else { json!(agreed as f64 / window.len() as f64) },
                           "misses": if mode == "one_sided" || mode == "one_sided_yes" { json!(missed) } else { Value::Null }},
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&json!({
        "snapshot": dir, "format": m["format"], "lead": m["lead"], "encoder": m["encoder"],
        "featurizer": m["featurizer"], "audit_rate": m["settings"]["audit_rate"], "questions": questions,
    }))?);
    // a decision list's top rules, as feature names (stderr, so the JSON above stays parseable)
    for q in &questions {
        if let Some(l) = q["checklist"]["listing"].as_str() {
            eprintln!("understudy: {} decision list (top rules first):\n{l}", q["name"].as_str().unwrap_or("question"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use understudy::encoder::Fake;
    use understudy::lazyenc::LazyEncoder;
    use understudy::rng::Xoshiro;
    use understudy::serve::http;

    /// A lead that answers a yes/no choice from the word "error" in the state.
    fn lead() -> http::Listener {
        let h: http::Handler = Arc::new(|rq: http::Request| {
            let v: Value = serde_json::from_slice(&rq.body).unwrap_or(Value::Null);
            let err = v["state"].as_str().unwrap_or("").contains("error");
            let (yes, no) = if err { (0.97, 0.03) } else { (0.03, 0.97) };
            let body = json!({"model": "m", "answers": {"d": {"type": "choice", "choice": if err { "a_yes" } else { "b_no" }, "confidence": 0.97, "probabilities": {"a_yes": yes, "b_no": no}}}, "usage": {}});
            (http::Response { status: 200, headers: vec![("content-type".into(), "application/json".into())], body: serde_json::to_vec(&body).unwrap(), stream: None }, None)
        });
        http::listen("127.0.0.1:0", 100, http::Limits::new(64 << 20), h).unwrap()
    }

    fn state(r: &mut Xoshiro, error: bool) -> String {
        const W: [&str; 8] = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"];
        let mut w: Vec<&str> = (0..2).map(|_| W[(r.next_u64() % 8) as usize]).collect();
        for _ in 0..3 {
            w.insert((r.next_u64() % (w.len() as u64 + 1)) as usize, if error { "error" } else { "ok" });
        }
        w.join(" ")
    }

    fn send(addr: std::net::SocketAddr, st: &str) {
        let body = json!({"model": "any-model", "state": st, "questions": {"d": {"type": "choice", "instructions": "Did it fail?", "criteria": {"a_yes": "failed", "b_no": "fine"}}}});
        let r = ureq::post(&format!("http://{addr}/v1/systemone")).set("content-type", "application/json").send_string(&body.to_string()).unwrap();
        assert_eq!(r.status(), 200);
    }

    /// (tiers, examples, graduated tier names, tiers with a cross-validated fit)
    fn look(srv: &understudy::serve::Server) -> (Vec<UnderstudentKind>, usize, Vec<String>, Vec<UnderstudentKind>) {
        let ns = srv.status()["namespaces"][0].clone();
        let (route, model) = (ns["route"].as_str().unwrap().to_string(), ns["model"].as_str().unwrap().to_string());
        srv.with_proxy(&route, &model, |p| {
            let u = &p.understudents[0];
            let grad: Vec<String> = u.tiers.iter().filter(|t| t.serving.mode().is_some()).map(|t| t.kind.name().to_string()).collect();
            let fitted = u.tiers.iter().filter(|t| t.serving.fit.report.get("cv_agreement").is_some_and(|v| !v.is_null())).map(|t| t.kind).collect();
            (u.tiers.iter().map(|t| t.kind).collect(), u.examples.len(), grad, fitted)
        })
        .expect("the namespace exists")
    }

    fn teach(srv: &understudy::serve::Server, n: usize, seed: u64) {
        let mut r = Xoshiro::new(seed);
        for i in 0..n {
            let e = r.uniform() < 0.35;
            send(srv.addr, &state(&mut r, e));
            if i % 25 == 24 {
                assert!(srv.wait_idle(Duration::from_secs(60)));
            }
        }
        assert!(srv.wait_idle(Duration::from_secs(60)));
    }

    #[test]
    fn default_auto_learns_and_graduates_without_the_model_and_the_encoder_designs_join_when_it_loads() {
        // no ambient lead config (it could switch the rules design on)
        let empty = std::env::temp_dir().join(format!("understudy-empty-leads-{}.toml", std::process::id()));
        std::fs::write(&empty, "").unwrap();
        unsafe { std::env::set_var("UNDERSTUDY_LEADS", &empty) };
        let up = lead();
        let url = format!("http://{}/v1/systemone", up.addr);
        let Cli { cmd: Cmd::Serve { args, opts } } = Cli::try_parse_from(["understudy", "serve", "--no-default-routes", "--lead-url", &url, "--no-snapshots", "--no-capture", "--bind", "127.0.0.1:0"]).ok().unwrap() else { panic!("serve") };
        assert!(opts.understudent.is_none(), "the default is auto, set by serve");
        let mut opts = opts;
        opts.understudent = Some(UnderstudentArg::Auto);
        // the model is absent until `ready` flips (the fetch waits for it)
        let ready = Arc::new(AtomicBool::new(false));
        let (r1, r2) = (ready.clone(), ready.clone());
        let lazy = LazyEncoder::with(
            true,
            Box::new(move || if r1.load(Ordering::SeqCst) { Ok(Arc::new(Fake { hidden: 16 }) as Arc<dyn Encoder>) } else { Err(anyhow::anyhow!("no model")) }),
            Box::new(move || {
                while !r2.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(())
            }),
        );
        let srv = understudy::serve::start(auto_config(args, opts, lazy.clone()).unwrap()).unwrap();
        teach(&srv, 300, 1);
        let (tiers, n, grad, fitted) = look(&srv);
        let free = vec![UnderstudentKind::Keywords, UnderstudentKind::Shape, UnderstudentKind::Checklist, UnderstudentKind::Memo];
        assert_eq!(tiers, free, "without the model the race is the model-free designs");
        assert!(n >= 250, "examples accumulate without the model: {n}");
        assert!(!grad.is_empty(), "a model-free design graduates: fitted {fitted:?}");
        assert_eq!(fitted, free, "every model-free design fits");
        assert!(srv.status()["namespaces"][0]["error"].is_null(), "{}", srv.status());
        // the model arrives: the encoder designs join, are fitted on what was learned before, and race
        ready.store(true, Ordering::SeqCst);
        lazy.wait().unwrap();
        teach(&srv, 300, 2);
        let (tiers, n2, grad, fitted) = look(&srv);
        let all = [free.clone(), vec![UnderstudentKind::Lite, UnderstudentKind::Probe, UnderstudentKind::Deep]].concat();
        assert_eq!(tiers, all);
        assert!(n2 > n);
        assert_eq!(fitted, all, "the encoder designs actually fit (graduated: {grad:?})");
        assert!(!grad.is_empty());
        let ns = srv.status()["namespaces"][0].clone();
        assert!(ns["error"].is_null(), "{ns}");
    }
}
