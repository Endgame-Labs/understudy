//! The `rules` understudent: decision rules written by an LLM as regexes, checked statistically against the lead's
//! answers. It is the cheapest tier (microseconds, no model), human-readable and auditable.
//!
//! A **rule set** is an ordered list of rules `{pattern, field?, label, why}` plus a default label (or "abstain").
//! The first rule whose regex is found in the state's text (or in the named top-level state field) answers with
//! its label; with no match the default answers, or the set abstains (the item is not eligible for local answering).
//!
//! Where hunch's understudents validate by out-of-fold predictions, rules are validated on **held-out** examples: those
//! the LLM never saw. A proposal shows the LLM a stratified sample of up to `SAMPLE_SIZE` labelled examples from
//! the older half of the example window; every other example of the window is held out. On the held-out rows:
//! - each rule's confidence is the Wilson lower bound of its precision among the held-out rows it answers (rows
//!   answered by an earlier rule don't count for it); the default's likewise over the rows no rule answers;
//! - a rule whose confidence is below `RULE_MIN_LB` (0.5, the lowest threshold graduation scans) is dropped and the
//!   rest re-evaluated until none is; a default below it becomes "abstain";
//! - graduation is hunch's scan (`understudy::graduation`) over the held-out rows where the set answers (an abstention
//!   is never local), with the answering rule's confidence as the prediction's confidence. Coverage is the share of
//!   held-out rows answered locally.
//!
//! Safety: patterns compile with the `regex` crate (finite automata, linear time: no catastrophic backtracking, no
//! look-around or backreferences) under a size limit; a pattern longer than `MAX_PATTERN` characters, one that
//! doesn't compile, one with an unknown label, and one that matches `TOO_BROAD` (95%) or more of **all** the window's
//! examples is rejected before validation.
//!
//! Every LLM request and response is cached (`RulesClient`, keyed by the SHA-256 of the request body), in memory and
//! in JSONL files (the snapshot's `rules_llm.jsonl`, `--rules-cache`), so a replay with a warm cache is offline and
//! deterministic.

use crate::encoder::{full_text, state_text};
use crate::pyjson;
use crate::question::{Question, QType};
use crate::rng::Xoshiro;
use crate::stats::wilson;
use crate::understudent::{Fit, Mode};
use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Examples shown to the LLM in one proposal (at most).
pub const SAMPLE_SIZE: usize = 40;
/// Characters of one example's text shown to the LLM (at most), and the whole prompt's budget for examples.
pub const EXAMPLE_CHARS: usize = 6000;
pub const PROMPT_CHARS: usize = 160_000;
/// Proposals per question (at most), and the new examples a failing set waits for before the next one.
pub const MAX_PROPOSALS: usize = 5;
pub const REPROPOSE_AFTER: usize = 100;
/// A set answering under this share of the held-out rows may still be widened by a re-proposal.
pub const LOW_COVERAGE: f64 = 0.5;
/// Examples the previous rule set got wrong that a re-proposal shows in addition (at most).
pub const HARD_ROWS: usize = 24;
/// A rule matching this share of all examples is too broad to be a rule.
pub const TOO_BROAD: f64 = 0.95;
/// A rule's (and the default's) held-out confidence must reach this Wilson lower bound to stay.
pub const RULE_MIN_LB: f64 = 0.5;
pub const MAX_PATTERN: usize = 400;
pub const MAX_RULES: usize = 24;
const REGEX_SIZE_LIMIT: usize = 1 << 20;

/// Which LLM writes the rules: any OpenAI-compatible chat endpoint. The default is Fireworks' public endpoint with
/// the key in `$FIREWORKS_API_KEY`; the `[rules]` table of the lead config file and the `--rules-llm-*` flags override it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RulesSettings {
    pub url: String,
    pub model: String,
    pub reasoning_effort: String,
    /// The environment variable holding the bearer key (its name only; the key is never stored). Empty: send none.
    #[serde(default)]
    pub key_env: String,
}

impl RulesSettings {
    /// The defaults with the config file's `[rules]` table applied.
    pub fn with_config(mut self, c: &crate::config::Config) -> Self {
        if let Some(r) = &c.rules {
            if let Some(u) = &r.url {
                self.url = crate::chat::endpoint(u);
            }
            if let Some(m) = &r.model {
                self.model = m.clone();
            }
            if let Some(e) = &r.reasoning_effort {
                self.reasoning_effort = e.clone();
            }
            if let Some(k) = &r.key_env {
                self.key_env = k.clone();
            }
        }
        self
    }
}

impl Default for RulesSettings {
    fn default() -> Self {
        RulesSettings {
            url: "https://api.fireworks.ai/inference/v1/chat/completions".into(),
            model: "accounts/fireworks/models/glm-5p3-flash".into(),
            reasoning_effort: "low".into(),
            key_env: "FIREWORKS_API_KEY".into(),
        }
    }
}

// ---------------------------------------------------------------- what rules read

/// A state as rules read it: its whole text (hunch's state text, capped at 20,000 characters) and, for an object, each
/// top-level field's text (a string as is, else Python JSON), which a rule can name.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuleInput {
    pub text: String,
    pub fields: Vec<(String, String)>,
}

impl RuleInput {
    pub fn from_state(state: &Value) -> Self {
        let text = full_text(&state_text(state));
        let fields = match state {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| (k.clone(), full_text(&if let Value::String(s) = v { s.clone() } else { pyjson::dumps(v) })))
                .collect(),
            _ => vec![],
        };
        RuleInput { text, fields }
    }

    fn field(&self, name: &str) -> Option<&str> {
        self.fields.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------- rule sets

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub pattern: String,
    /// A top-level state field the pattern is searched in; none = the whole text.
    #[serde(default)]
    pub field: Option<String>,
    pub label: usize,
    pub label_name: String,
    #[serde(default)]
    pub why: String,
    /// Held-out rows this rule answered (first match), and how many of them the lead answered the same way.
    #[serde(default)]
    pub heldout_matches: usize,
    #[serde(default)]
    pub heldout_correct: usize,
    /// Wilson lower bound of `heldout_correct / heldout_matches`: the confidence of this rule's answers.
    #[serde(default)]
    pub confidence: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dropped {
    pub pattern: String,
    pub label: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeldOut {
    pub n: usize,
    pub answered: usize,
    pub agreement: Option<f64>,
    pub coverage: f64,
}

/// The compiled patterns of a rule set's rules, in order. Not serialized (a loaded set recompiles).
#[derive(Clone, Default)]
pub struct Compiled(Vec<Regex>);

impl std::fmt::Debug for Compiled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Compiled({})", self.0.len())
    }
}

impl PartialEq for Compiled {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && self.0.iter().zip(&other.0).all(|(a, b)| a.as_str() == b.as_str())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuleSet {
    pub rules: Vec<Rule>,
    /// The label for a text no rule matches; none = abstain.
    pub default: Option<usize>,
    pub default_label: String,
    pub default_matches: usize,
    pub default_correct: usize,
    pub default_confidence: f64,
    /// State hashes of the examples the LLM was shown (everything else in the window is held out).
    pub shown: Vec<String>,
    /// Rules rejected before validation or dropped by it.
    pub dropped: Vec<Dropped>,
    /// Which proposal of the question wrote this set (1-based), and the hash of its LLM request.
    pub proposal: usize,
    pub request_key: String,
    /// The latest validation on held-out rows.
    pub heldout: HeldOut,
    #[serde(skip)]
    pub compiled: Compiled,
}

/// What a rule set says about one state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Answer {
    pub label: usize,
    pub confidence: f64,
    /// The answering rule's index; none for the default.
    pub rule: Option<usize>,
}

fn compile(pattern: &str) -> Result<Regex> {
    if pattern.chars().count() > MAX_PATTERN {
        bail!("pattern longer than {MAX_PATTERN} characters");
    }
    Ok(RegexBuilder::new(pattern).size_limit(REGEX_SIZE_LIMIT).dfa_size_limit(REGEX_SIZE_LIMIT * 4).build()?)
}

fn hit(rule: &Rule, re: &Regex, inp: &RuleInput) -> bool {
    match &rule.field {
        Some(f) => inp.field(f).is_some_and(|t| re.is_match(t)),
        None => re.is_match(&inp.text),
    }
}

impl RuleSet {
    /// Recompile the patterns (after loading). A pattern that no longer compiles is an error.
    pub fn compile(&mut self) -> Result<()> {
        let v: Result<Vec<Regex>> = self.rules.iter().map(|r| compile(&r.pattern).with_context(|| r.pattern.clone())).collect();
        self.compiled = Compiled(v?);
        Ok(())
    }

    /// The first matching rule's answer, else the default's, else None (abstain).
    pub fn apply(&self, inp: &RuleInput) -> Option<Answer> {
        for (i, (r, re)) in self.rules.iter().zip(&self.compiled.0).enumerate() {
            if hit(r, re, inp) {
                return Some(Answer { label: r.label, confidence: r.confidence, rule: Some(i) });
            }
        }
        self.default.map(|label| Answer { label, confidence: self.default_confidence, rule: None })
    }

    /// Class probabilities: the answer's confidence on its label, the rest split evenly. None on abstention.
    pub fn proba(&self, inp: &RuleInput, k: usize) -> Option<Vec<f64>> {
        let a = self.apply(inp)?;
        let rest = (1.0 - a.confidence) / (k.max(2) - 1) as f64;
        Some((0..k).map(|c| if c == a.label { a.confidence } else { rest }).collect())
    }

    /// A human-readable listing.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for (i, r) in self.rules.iter().enumerate() {
            s.push_str(&format!(
                "{:>2}. /{}/{} -> {}  [heldout {}/{} conf {:.3}]  {}\n",
                i + 1, r.pattern, r.field.as_ref().map_or(String::new(), |f| format!(" in {f}")), r.label_name,
                r.heldout_correct, r.heldout_matches, r.confidence, r.why
            ));
        }
        s.push_str(&format!("    default -> {}  [heldout {}/{} conf {:.3}]\n", self.default_label, self.default_correct, self.default_matches, self.default_confidence));
        s
    }
}

/// One training example as rules see it.
pub struct RuleRow<'a> {
    pub hash: &'a str,
    pub input: &'a RuleInput,
    pub y: usize,
    pub seq: u64,
    /// Rows of one group (see `Settings::group_sep`) are held out together: a row whose group has a row shown to the
    /// LLM is not held out.
    pub group: Option<u64>,
}

/// The groups that have a row among the shown hashes.
fn shown_groups(rows: &[RuleRow], shown: &HashSet<&str>) -> HashSet<u64> {
    rows.iter().filter(|r| shown.contains(r.hash)).filter_map(|r| r.group).collect()
}

fn is_seen(r: &RuleRow, shown: &HashSet<&str>, groups: &HashSet<u64>) -> bool {
    shown.contains(r.hash) || r.group.is_some_and(|g| groups.contains(&g))
}

// ---------------------------------------------------------------- the LLM

/// Something that completes a chat-completions request body, returning the first choice's message content.
pub trait RulesLlm: Send + Sync {
    fn complete(&self, body: &Value) -> Result<String>;
}

/// An OpenAI-style chat-completions endpoint over HTTP. The bearer key, if any, lives in memory only.
pub struct ChatLlm {
    pub url: String,
    key: Option<String>,
    agent: ureq::Agent,
    pub attempts: usize,
}

impl ChatLlm {
    pub fn new(url: &str) -> Self {
        ChatLlm { url: url.into(), key: None, agent: ureq::AgentBuilder::new().timeout(Duration::from_secs(300)).build(), attempts: 4 }
    }

    pub fn with_key(mut self, key: Option<String>) -> Self {
        self.key = key;
        self
    }
}

impl RulesLlm for ChatLlm {
    fn complete(&self, body: &Value) -> Result<String> {
        let mut wait = Duration::from_secs(3);
        let mut last = None;
        for attempt in 0..self.attempts {
            if attempt > 0 {
                std::thread::sleep(wait);
                wait *= 2;
            }
            let mut rq = self.agent.post(&self.url);
            if let Some(k) = &self.key {
                rq = rq.set("authorization", &format!("Bearer {k}"));
            }
            match rq.send_json(body) {
                Ok(r) => {
                    let v: Value = r.into_json().context("the rules LLM's response is not JSON")?;
                    return match v["choices"][0]["message"]["content"].as_str() {
                        Some(c) if !c.trim().is_empty() => Ok(c.to_string()),
                        _ => bail!("the rules LLM returned no content (finish_reason {})", v["choices"][0]["finish_reason"]),
                    };
                }
                Err(ureq::Error::Status(code, r)) if code == 408 || code == 429 || code >= 500 => {
                    last = Some(anyhow!("rules LLM HTTP {code}: {}", r.into_string().unwrap_or_default()));
                }
                Err(ureq::Error::Status(code, r)) => bail!("rules LLM HTTP {code}: {}", r.into_string().unwrap_or_default()),
                Err(e) => last = Some(anyhow!("rules LLM transport: {e}")),
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("rules LLM failed")))
    }
}

/// An LLM behind a cache of every request and response: memory, plus JSONL files `{"key","request","content"}` that
/// a live call is appended to. A cached request never reaches the LLM.
pub struct RulesClient {
    llm: Box<dyn RulesLlm>,
    mem: Mutex<HashMap<String, String>>,
    files: Mutex<Vec<PathBuf>>,
    /// Calls that reached the LLM, and calls answered from the cache.
    pub live: AtomicUsize,
    pub cached: AtomicUsize,
}

pub fn request_key(body: &Value) -> String {
    crate::question::hex(&Sha256::digest(serde_json::to_string(body).unwrap_or_default().as_bytes()))
}

impl RulesClient {
    pub fn new(llm: Box<dyn RulesLlm>) -> Self {
        RulesClient { llm, mem: Mutex::new(HashMap::new()), files: Mutex::new(vec![]), live: AtomicUsize::new(0), cached: AtomicUsize::new(0) }
    }

    /// Load a JSONL cache file (if it exists) and append later live calls to it. Returns the entries loaded.
    pub fn attach(&self, path: &Path) -> Result<usize> {
        let mut n = 0;
        if path.exists() {
            let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
            let mut mem = self.mem.lock().unwrap();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    if let (Some(k), Some(c)) = (v["key"].as_str(), v["content"].as_str()) {
                        n += usize::from(mem.insert(k.to_string(), c.to_string()).is_none());
                    }
                }
            }
        }
        let mut files = self.files.lock().unwrap();
        if !files.iter().any(|f| f == path) {
            files.push(path.to_path_buf());
        }
        Ok(n)
    }

    /// The content for a request: from the cache, else from the LLM (and cached).
    pub fn call(&self, body: &Value) -> Result<(String, String)> {
        let key = request_key(body);
        if let Some(c) = self.mem.lock().unwrap().get(&key) {
            self.cached.fetch_add(1, Ordering::SeqCst);
            return Ok((key, c.clone()));
        }
        let content = self.llm.complete(body)?;
        self.live.fetch_add(1, Ordering::SeqCst);
        self.mem.lock().unwrap().insert(key.clone(), content.clone());
        let line = serde_json::to_string(&json!({"key": key, "request": body, "content": content}))?;
        for f in self.files.lock().unwrap().iter() {
            let mut h = std::fs::OpenOptions::new().create(true).append(true).open(f).with_context(|| f.display().to_string())?;
            writeln!(h, "{line}")?;
        }
        Ok((key, content))
    }
}

// ---------------------------------------------------------------- the prompt

fn clip(text: &str, budget: usize) -> String {
    let n = text.chars().count();
    if n <= budget {
        return text.to_string();
    }
    let head = budget * 2 / 3;
    let tail = budget - head;
    let h: String = text.chars().take(head).collect();
    let t: String = text.chars().skip(n - tail).collect();
    format!("{h}\n...[clipped]...\n{t}")
}

/// An example as the LLM sees it: an object state as `[field]` blocks (so a rule can name a field), else the text.
fn render(inp: &RuleInput, cap: usize) -> String {
    if inp.fields.is_empty() {
        return clip(&inp.text, cap);
    }
    let total: usize = inp.fields.iter().map(|(_, v)| v.chars().count()).sum::<usize>().max(1);
    inp.fields
        .iter()
        .map(|(k, v)| {
            let share = (cap * v.chars().count() / total).max(200);
            format!("[{k}]\n{}", clip(v, share))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// FROZEN WORDING: the prompt text below (and in `build_request`) deliberately still says "teacher", as before the
// rename to "lead". `request_key` hashes the whole request, so any change would miss every existing rules cache and
// could change the rule sets (and so the decisions) an LLM writes. Do not reword it; the test
// `request_key_is_pinned_to_the_pre_rename_prompt` pins it.
const SYSTEM: &str = "You write decision rules, as regular expressions, for a fast and auditable classifier that stands in for a slow, \
accurate model (the teacher). You see labelled examples: the teacher's answers. Write rules that reproduce the teacher's answers on \
examples you have not seen, not just on these. Answer with one JSON object and nothing else.";

/// What a re-proposal adds to the prompt: the previous rule set and the examples it got wrong (class, what it said,
/// text).
pub struct Feedback<'a> {
    pub listing: String,
    pub errors: Vec<(usize, Option<usize>, &'a RuleInput)>,
}

/// The chat-completions request body asking for a rule set.
pub fn build_request(settings: &RulesSettings, q: &Question, sample: &[(usize, &RuleInput)], base_rates: &[usize], fields: &[String],
                     feedback: Option<&Feedback>) -> Value {
    let criteria = match &q.criteria {
        Value::Object(m) => m.iter().map(|(k, v)| format!("- {k}: {}", v.as_str().map_or_else(|| v.to_string(), String::from))).collect::<Vec<_>>().join("\n"),
        Value::Array(a) => a.iter().enumerate().map(|(i, v)| format!("- {i}: {}", v.as_str().map_or_else(|| v.to_string(), String::from))).collect::<Vec<_>>().join("\n"),
        other => other.to_string(),
    };
    let options: Vec<String> = q.options.iter().map(|o| format!("{o:?}")).collect();
    let rates: Vec<String> = q.options.iter().zip(base_rates).map(|(o, n)| format!("{o}: {n}")).collect();
    let cap = (PROMPT_CHARS / (sample.len() + feedback.map_or(0, |f| f.errors.len())).max(1)).min(EXAMPLE_CHARS);
    let examples: String = sample
        .iter()
        .enumerate()
        .map(|(i, (y, inp))| format!("### Example {} -- teacher's answer: {}\n{}\n", i + 1, q.options[*y], render(inp, cap)))
        .collect::<Vec<_>>()
        .join("\n");
    let hard: String = match feedback {
        None => String::new(),
        Some(f) => format!(
            "\n# Your previous rule set, measured on held-out examples\n{}\nIt got the \"hard examples\" at the end wrong (or abstained on them). \
             Return the complete improved set: copy every previous rule that is precise (each one that has a high held-out match count and \
             no mistakes is valuable, don't lose it), add rules for the forms you missed (several distinct patterns are fine), and tighten or \
             drop only the rules that misfire.\n\n# Hard examples ({})\n{}",
            f.listing,
            f.errors.len(),
            f.errors
                .iter()
                .enumerate()
                .map(|(i, (y, said, inp))| format!(
                    "### Hard example {} -- teacher's answer: {}; the previous rules said: {}\n{}\n",
                    i + 1, q.options[*y], said.map_or("abstain".to_string(), |s| q.options[s].clone()), render(inp, cap)
                ))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    };
    let field_note = if fields.is_empty() {
        "States are plain text.".to_string()
    } else {
        format!(
            "States are objects with the fields {}. Below, each example's fields are shown as [field] blocks. A rule may name one \
             of them in \"field\" (the pattern is then searched only in that field's text); with no field it is searched in the whole \
             text, which is the `call` (or `tool_call`) field, a line break, the `output` field, then the other fields as \"name: value\" lines.",
            fields.join(", ")
        )
    };
    let user = format!(
        "# The question the teacher answers\n{instr}\n\nType: {ty:?}. The answer is one of {opts}.\nCriteria:\n{criteria}\n\n\
         # How rules work\n{field_note}\nA rule is {{\"pattern\": <regex>, \"field\": <field name or null>, \"label\": <one of the options>, \"why\": <short reason>}}. \
         Rules are tried in order and the first whose pattern is found (searched anywhere, not anchored) in the text answers with its label. \
         If none matches, the default label answers, or \"abstain\" (no local answer: the teacher is asked instead; prefer this over a weak default).\n\
         Patterns use the Rust `regex` crate syntax: no look-around, no backreferences. Use (?i) for case-insensitive, (?m) so ^ and $ match at \
         line breaks, (?s) so . matches newlines. In JSON, escape backslashes (\\\\d, \\\\b).\n\
         Each rule is later measured on held-out examples: its precision there is its confidence, and a rule below 50% precision is dropped, \
         so write precise rules and leave out guesses. Order from most specific to most general. Watch for decoys: patterns that look like the \
         cue but whose teacher answer says otherwise. A pattern that matches 95% or more of all examples is rejected as too broad. \
         Rules written from a handful of examples tend to be too narrow: where the examples show a family of formats (token prefixes, \
         name=value assignments, connection strings), generalize with loose length bounds, optional separators and case-insensitive names \
         rather than copying one example's exact shape; the held-out check will drop what misfires. At most 16 rules.\n\n\
         # Output\n{{\"rules\": [{{\"pattern\": \"...\", \"field\": null, \"label\": \"{first}\", \"why\": \"...\"}}], \"default\": \"{first}\" or \"abstain\"}}\n\n\
         # Labelled examples ({n}, stratified by answer; the window's answer counts are {rates})\n{examples}{hard}",
        instr = q.instructions,
        ty = q.qtype,
        opts = options.join(", "),
        first = q.options[0],
        n = sample.len(),
        rates = rates.join(", "),
    );
    json!({
        "model": settings.model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 8000,
        "reasoning_effort": settings.reasoning_effort,
        "response_format": {"type": "json_object"},
    })
}

// ---------------------------------------------------------------- parsing a proposal

/// The class a label value names: an option's name (any case), noul's yes/no, a bool, or an index.
pub fn label_index(q: &Question, v: &Value) -> Option<usize> {
    match v {
        Value::String(s) => {
            let t = s.trim();
            if let Some(i) = q.options.iter().position(|o| o == t) {
                return Some(i);
            }
            if let Some(i) = q.options.iter().position(|o| o.eq_ignore_ascii_case(t)) {
                return Some(i);
            }
            match (q.qtype, t.to_lowercase().as_str()) {
                (QType::Noul, "yes") => Some(1),
                (QType::Noul, "no") => Some(0),
                _ => None,
            }
        }
        Value::Bool(b) if q.qtype == QType::Noul => Some(usize::from(*b)),
        Value::Number(n) => n.as_u64().filter(|&i| (i as usize) < q.options.len()).map(|i| i as usize),
        _ => None,
    }
}

/// A parsed, not yet compiled or validated, proposal.
#[derive(Debug)]
pub struct Proposal {
    pub rules: Vec<Rule>,
    pub default: Option<usize>,
    pub rejected: Vec<Dropped>,
}

/// The JSON object in an LLM's reply (fences and chatter around it are ignored).
fn json_object(content: &str) -> Result<Value> {
    let a = content.find('{').ok_or_else(|| anyhow!("no JSON object in the reply"))?;
    let b = content.rfind('}').ok_or_else(|| anyhow!("no JSON object in the reply"))?;
    serde_json::from_str(&content[a..=b]).context("the reply is not valid JSON")
}

pub fn parse_proposal(content: &str, q: &Question) -> Result<Proposal> {
    let v = json_object(content)?;
    let list = v.get("rules").and_then(Value::as_array).ok_or_else(|| anyhow!("the reply has no \"rules\" array"))?;
    let (mut rules, mut rejected) = (vec![], vec![]);
    for item in list {
        let pattern = item.get("pattern").and_then(Value::as_str).unwrap_or("").to_string();
        let name = item.get("label").map_or_else(String::new, |l| l.as_str().map_or_else(|| l.to_string(), String::from));
        if pattern.is_empty() {
            rejected.push(Dropped { pattern, label: name, reason: "no pattern".into() });
            continue;
        }
        let Some(label) = item.get("label").and_then(|l| label_index(q, l)) else {
            rejected.push(Dropped { pattern, label: name, reason: "unknown label".into() });
            continue;
        };
        let field = item.get("field").and_then(Value::as_str).filter(|f| !f.is_empty() && *f != "null").map(String::from);
        let why = item.get("why").and_then(Value::as_str).unwrap_or("").to_string();
        rules.push(Rule { pattern, field, label, label_name: q.options[label].clone(), why, heldout_matches: 0, heldout_correct: 0, confidence: 0.0 });
    }
    if rules.len() > MAX_RULES {
        for r in rules.split_off(MAX_RULES) {
            rejected.push(Dropped { pattern: r.pattern, label: r.label_name, reason: format!("more than {MAX_RULES} rules") });
        }
    }
    let default = v.get("default").and_then(|d| label_index(q, d));
    Ok(Proposal { rules, default, rejected })
}

// ---------------------------------------------------------------- validation

/// What a rule set did on a window: the set (rules kept, confidences set, drops recorded) and, per held-out row, its
/// answer.
pub struct Validated {
    pub set: RuleSet,
    /// Indices into the rows of the held-out ones, and each one's answer (label, confidence) or abstention.
    pub held: Vec<usize>,
    pub answers: Vec<Option<(usize, f64)>>,
}

/// Compile `rules`, reject the too-broad ones (by all `rows`), then measure on the held-out rows (those whose hash isn't in
/// `shown`) and drop rules below `RULE_MIN_LB` until stable. `set` carries the metadata (shown, proposal, ...) in.
pub fn validate(rules: Vec<Rule>, default: Option<usize>, rows: &[RuleRow], mut set: RuleSet, q: &Question) -> Validated {
    let shown: HashSet<&str> = set.shown.iter().map(String::as_str).collect();
    let mut dropped = std::mem::take(&mut set.dropped);
    // compile, and the match matrix over every row
    let mut live: Vec<(Rule, Regex)> = vec![];
    for r in rules {
        match compile(&r.pattern) {
            Ok(re) => live.push((r, re)),
            Err(e) => dropped.push(Dropped { pattern: r.pattern, label: r.label_name, reason: format!("does not compile: {e}") }),
        }
    }
    let hits: Vec<Vec<bool>> = rows.par_iter().map(|row| live.iter().map(|(r, re)| hit(r, re, row.input)).collect()).collect();
    let n = rows.len().max(1);
    let mut keep: Vec<usize> = vec![];
    for (i, (r, _)) in live.iter().enumerate() {
        let share = hits.iter().filter(|h| h[i]).count() as f64 / n as f64;
        if share >= TOO_BROAD {
            dropped.push(Dropped { pattern: r.pattern.clone(), label: r.label_name.clone(), reason: format!("too broad: matches {:.0}% of all examples", share * 100.0) });
        } else {
            keep.push(i);
        }
    }
    let sg = shown_groups(rows, &shown);
    let held: Vec<usize> = (0..rows.len()).filter(|&j| !is_seen(&rows[j], &shown, &sg)).collect();
    // measure; drop rules below RULE_MIN_LB until none is
    let (mut m, mut c);
    loop {
        (m, c) = (vec![0usize; live.len()], vec![0usize; live.len()]);
        for &j in &held {
            if let Some(&i) = keep.iter().find(|&&i| hits[j][i]) {
                m[i] += 1;
                c[i] += usize::from(live[i].0.label == rows[j].y);
            }
        }
        let bad: Vec<usize> = keep.iter().copied().filter(|&i| wilson(c[i], m[i]).0 < RULE_MIN_LB).collect();
        if bad.is_empty() {
            break;
        }
        for &i in &bad {
            let r = &live[i].0;
            dropped.push(Dropped {
                pattern: r.pattern.clone(),
                label: r.label_name.clone(),
                reason: format!("held-out precision {}/{} (Wilson lower bound {:.2} < {RULE_MIN_LB})", c[i], m[i], wilson(c[i], m[i]).0),
            });
        }
        keep.retain(|i| !bad.contains(i));
    }
    // the default: the rows no kept rule answers
    let (mut dm, mut dc) = (0, 0);
    if let Some(d) = default {
        for &j in &held {
            if !keep.iter().any(|&i| hits[j][i]) {
                dm += 1;
                dc += usize::from(d == rows[j].y);
            }
        }
    }
    let dconf = wilson(dc, dm).0;
    let default = default.filter(|_| dconf >= RULE_MIN_LB);
    let mut out_rules = vec![];
    let mut compiled = vec![];
    for &i in &keep {
        let (mut r, re) = live[i].clone();
        (r.heldout_matches, r.heldout_correct, r.confidence) = (m[i], c[i], wilson(c[i], m[i]).0);
        out_rules.push(r);
        compiled.push(re);
    }
    set.rules = out_rules;
    set.compiled = Compiled(compiled);
    set.default = default;
    set.default_label = default.map_or_else(|| "abstain".to_string(), |d| q.options[d].clone());
    (set.default_matches, set.default_correct, set.default_confidence) = (dm, dc, dconf);
    set.dropped = dropped;
    let answers: Vec<Option<(usize, f64)>> = held.iter().map(|&j| set.apply(rows[j].input).map(|a| (a.label, a.confidence))).collect();
    let answered = answers.iter().flatten().count();
    let right = held.iter().zip(&answers).filter(|(j, a)| a.is_some_and(|(l, _)| l == rows[**j].y)).count();
    set.heldout = HeldOut {
        n: held.len(),
        answered,
        agreement: (answered > 0).then(|| right as f64 / answered as f64),
        coverage: if held.is_empty() { 0.0 } else { answered as f64 / held.len() as f64 },
    };
    Validated { set, held, answers }
}

// ---------------------------------------------------------------- fitting

/// What persists between a question's refits.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RulesState {
    /// Proposals asked for so far (a failed LLM call or an unusable reply counts), and the proxy's example counter then.
    pub proposals: usize,
    pub last_seq: u64,
    pub last_error: Option<String>,
    /// How the latest re-proposal compared with the set it competed with.
    #[serde(default)]
    pub last_note: Option<String>,
}

pub struct RulesCtx<'a> {
    pub client: Option<&'a RulesClient>,
    pub settings: &'a RulesSettings,
    pub question: &'a Question,
    pub target: f64,
    pub minority: f64,
    pub min_examples: usize,
    pub seed: u64,
    /// The Wilson z of the graduation bars (1.96 unless `--graduation sequential`).
    pub z: f64,
}

/// A stratified sample of rows (round robin over the classes, each shuffled by a seeded generator) from the older half.
fn sample(rows: &[RuleRow], k: usize, seed: u64) -> Vec<usize> {
    let older = rows.len() / 2;
    let mut by: Vec<Vec<usize>> = vec![vec![]; k];
    for (j, r) in rows[..older].iter().enumerate() {
        by[r.y].push(j);
    }
    let mut rng = Xoshiro::new(seed);
    by.iter_mut().for_each(|v| rng.shuffle(v));
    let mut out = vec![];
    let mut at = 0;
    while out.len() < SAMPLE_SIZE && by.iter().any(|v| at < v.len()) {
        for v in &by {
            if at < v.len() && out.len() < SAMPLE_SIZE {
                out.push(v[at]);
            }
        }
        at += 1;
    }
    out
}

/// Up to `n` of the rows `prev` got wrong or abstained on, among the rows its LLM never saw (stratified by class,
/// shuffled by a seeded generator).
fn hard_rows(rows: &[RuleRow], prev: &RuleSet, k: usize, n: usize, seed: u64) -> Vec<(usize, Option<usize>)> {
    let seen: HashSet<&str> = prev.shown.iter().map(String::as_str).collect();
    let sg = shown_groups(rows, &seen);
    let mut by: Vec<Vec<(usize, Option<usize>)>> = vec![vec![]; k];
    for (j, r) in rows.iter().enumerate() {
        if is_seen(r, &seen, &sg) {
            continue;
        }
        let said = prev.apply(r.input).map(|a| a.label);
        if said != Some(r.y) {
            by[r.y].push((j, said));
        }
    }
    let mut rng = Xoshiro::new(seed ^ 0x9e37);
    by.iter_mut().for_each(|v| rng.shuffle(v));
    let mut out = vec![];
    let mut at = 0;
    while out.len() < n && by.iter().any(|v| at < v.len()) {
        for v in &by {
            if at < v.len() && out.len() < n {
                out.push(v[at]);
            }
        }
        at += 1;
    }
    out
}

/// Ask the LLM for a rule set from a sample of `rows`. A re-proposal (`prev`) also shows the previous set and
/// `HARD_ROWS` of the examples it got wrong, which join the shown ones (so they are held out of nothing: the new set is
/// still validated only on rows its LLM never saw). Returns the validated set, or why there is none.
fn propose(ctx: &RulesCtx, rows: &[RuleRow], proposal: usize, prev: Option<&RuleSet>) -> Result<Validated> {
    let client = ctx.client.ok_or_else(|| anyhow!("no rules LLM client"))?;
    let q = ctx.question;
    let k = q.k();
    let idx = sample(rows, k, ctx.seed.wrapping_mul(1000).wrapping_add(proposal as u64));
    if idx.is_empty() {
        bail!("no examples to show");
    }
    let mut base = vec![0usize; k];
    rows.iter().for_each(|r| base[r.y] += 1);
    let mut fields: Vec<String> = vec![];
    for &j in &idx {
        for (f, _) in &rows[j].input.fields {
            if !fields.contains(f) {
                fields.push(f.clone());
            }
        }
    }
    let shown: Vec<(usize, &RuleInput)> = idx.iter().map(|&j| (rows[j].y, rows[j].input)).collect();
    let hard: Vec<(usize, Option<usize>)> = prev.map_or(vec![], |p| hard_rows(rows, p, k, HARD_ROWS, ctx.seed.wrapping_add(proposal as u64)));
    let feedback = prev.map(|p| Feedback { listing: p.describe(), errors: hard.iter().map(|&(j, said)| (rows[j].y, said, rows[j].input)).collect() });
    let body = build_request(ctx.settings, q, &shown, &base, &fields, feedback.as_ref());
    let (key, content) = client.call(&body)?;
    let p = parse_proposal(&content, q)?;
    let mut hashes: Vec<String> = idx.iter().chain(hard.iter().map(|(j, _)| j)).map(|&j| rows[j].hash.to_string()).collect();
    hashes.sort();
    hashes.dedup();
    let set = RuleSet { shown: hashes, dropped: p.rejected, proposal, request_key: key, ..Default::default() };
    Ok(validate(p.rules, p.default, rows, set, q))
}

/// A validated set's graduation on its held-out rows.
struct Grade {
    cov: f64,
    mode: Option<Mode>,
    thr: f64,
    nh: usize,
    answered: usize,
    right: usize,
    n_minor: usize,
    minor_right: usize,
}

fn grade(ctx: &RulesCtx, rows: &[RuleRow], v: &Validated) -> Grade {
    let q = ctx.question;
    let y: Vec<usize> = v.held.iter().map(|&j| rows[j].y).collect();
    let nh = y.len();
    let mut counts = vec![0usize; q.k()];
    y.iter().for_each(|&c| counts[c] += 1);
    let maj = crate::stats::argmax(&counts.iter().map(|&c| c as f64).collect::<Vec<_>>());
    let minor: Vec<bool> = y.iter().map(|&c| c != maj).collect();
    let cf: Vec<f64> = v.answers.iter().map(|a| a.map_or(f64::NEG_INFINITY, |(_, c)| c)).collect();
    let agree: Vec<bool> = v.answers.iter().zip(&y).map(|(a, y)| a.is_some_and(|(l, _)| l == *y)).collect();
    // P(yes) of an answer: the confidence on a "yes", else its complement; above 1 for an abstention (never local)
    let p_yes: Option<Vec<f64>> = (q.qtype == QType::Noul).then(|| {
        v.answers.iter().map(|a| match a {
            None => 2.0,
            Some((1, c)) => *c,
            Some((_, c)) => 1.0 - c,
        }).collect()
    });
    let (cov, mode, thr) = if nh > 0 { crate::understudent::graduation_z(&cf, &agree, &minor, p_yes.as_deref(), &y, ctx.target, ctx.minority, ctx.z) } else { (0.0, None, 0.0) };
    Grade {
        cov, mode, thr, nh,
        answered: v.answers.iter().flatten().count(),
        right: agree.iter().filter(|&&a| a).count(),
        n_minor: minor.iter().filter(|&&m| m).count(),
        minor_right: agree.iter().zip(&minor).filter(|(a, m)| **a && **m).count(),
    }
}

/// Revalidate `set` (its rules as they stand) on `rows`, treating `shown` as the examples its LLM saw.
fn revalidate(set: &RuleSet, shown: Vec<String>, rows: &[RuleRow], q: &Question) -> Validated {
    let base = RuleSet { rules: vec![], default: None, shown, ..set.clone() };
    validate(set.rules.clone(), set.default, rows, base, q)
}

/// Refit the rules understudent: maybe ask for a new rule set, validate the current one on the window's held-out rows and
/// decide graduation. `current` is the installed set and `failing` whether it is not graduated or demoted. Returns the
/// fit and the state to keep.
///
/// A proposal is asked for when the window has `min_examples` rows of at least two classes, fewer than `MAX_PROPOSALS`
/// were asked, and either none was yet, or `REPROPOSE_AFTER` examples arrived since the last one while the set is
/// failing or answers under `LOW_COVERAGE` of the held-out rows (a graduated set that abstains on most rows can still be
/// widened). A re-proposal competes with the set in place: both are validated on the rows neither's LLM saw, and the
/// better one stays (graduated beats not; then higher coverage; a tie goes to the new one).
pub fn fit_rules(ctx: &RulesCtx, rows: &[RuleRow], current: Option<&Arc<RuleSet>>, failing: bool, state: &RulesState, seq_now: u64) -> (Fit, RulesState) {
    let t0 = Instant::now();
    let q = ctx.question;
    let n = rows.len();
    let mut st = state.clone();
    let mut f = Fit { report: json!({"n": n}), nn_floor: -1.0, readout: Some(crate::readout::Readout::Rules), ..Default::default() };
    let mut classes: Vec<usize> = rows.iter().map(|r| r.y).collect();
    classes.sort_unstable();
    classes.dedup();
    let enough = n >= ctx.min_examples && classes.len() >= 2;
    let live0 = ctx.client.map_or(0, |c| c.live.load(Ordering::SeqCst));
    let mut chosen: Option<(Validated, Grade)> = None;
    let mut proposed = false;
    let new_since = rows.iter().filter(|r| r.seq > st.last_seq).count();
    let improvable = failing || current.is_none_or(|c| c.heldout.coverage < LOW_COVERAGE);
    if enough && st.proposals < MAX_PROPOSALS && (st.proposals == 0 || (improvable && new_since >= REPROPOSE_AFTER)) && ctx.client.is_some() {
        proposed = true;
        st.proposals += 1;
        st.last_seq = seq_now;
        match propose(ctx, rows, st.proposals, current.map(|c| &**c)) {
            Ok(v) if !v.set.rules.is_empty() => {
                st.last_error = None;
                chosen = Some(match current {
                    None => {
                        let g = grade(ctx, rows, &v);
                        (v, g)
                    }
                    Some(cur) => {
                        let mut shown: Vec<String> = cur.shown.iter().chain(&v.set.shown).cloned().collect();
                        shown.sort();
                        shown.dedup();
                        let (old, new) = (revalidate(cur, shown.clone(), rows, q), revalidate(&v.set, shown, rows, q));
                        let (go, gn) = (grade(ctx, rows, &old), grade(ctx, rows, &new));
                        let keep_new = (gn.mode.is_some(), gn.cov) >= (go.mode.is_some(), go.cov);
                        st.last_note = Some(format!(
                            "proposal {}: new set {} (coverage {:.3}) vs current {} (coverage {:.3}) on {} common held-out rows: kept the {}",
                            st.proposals, gn.mode.map_or("not graduated", Mode::name), gn.cov, go.mode.map_or("not graduated", Mode::name), go.cov, gn.nh,
                            if keep_new { "new one" } else { "current one" }
                        ));
                        if keep_new { (new, gn) } else { (old, go) }
                    }
                });
            }
            Ok(v) => st.last_error = Some(format!("no usable rules (rejected or dropped: {})", v.set.dropped.len())),
            Err(e) => st.last_error = Some(format!("{e:#}")),
        }
    }
    if chosen.is_none() && enough {
        if let Some(cur) = current {
            let v = revalidate(cur, cur.shown.clone(), rows, q);
            let g = grade(ctx, rows, &v);
            chosen = Some((v, g));
        }
    }
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("readout".into(), json!("rules"));
    rep.insert("proposals".into(), json!(st.proposals));
    rep.insert("proposed_now".into(), json!(proposed));
    rep.insert("llm_calls".into(), json!(ctx.client.map_or(0, |c| c.live.load(Ordering::SeqCst)) - live0));
    rep.insert("llm_error".into(), json!(st.last_error));
    rep.insert("last_note".into(), json!(st.last_note));
    rep.insert("labelled".into(), json!(0));
    let Some((v, g)) = chosen.filter(|(v, _)| !v.set.rules.is_empty()) else {
        rep.insert("why_not".into(), json!(if !enough { "too few examples" } else { "no rule set yet" }));
        rep.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
        return (f, st);
    };
    f.mode = g.mode;
    f.thr = g.thr;
    let rep = f.report.as_object_mut().unwrap();
    rep.insert("heldout_n".into(), json!(g.nh));
    rep.insert("heldout_answered".into(), json!(g.answered));
    rep.insert("cv_agreement".into(), if g.answered > 0 { json!(g.right as f64 / g.answered as f64) } else { Value::Null });
    rep.insert("cv_minority_recall".into(), if g.n_minor > 0 { json!(g.minor_right as f64 / g.n_minor as f64) } else { Value::Null });
    rep.insert("majority_share".into(), json!(if g.nh > 0 { 1.0 - g.n_minor as f64 / g.nh as f64 } else { 0.0 }));
    rep.insert("mode".into(), json!(g.mode.map(Mode::name)));
    rep.insert("threshold".into(), json!(g.thr));
    rep.insert("cv_coverage".into(), json!(g.cov));
    rep.insert("rule_count".into(), json!(v.set.rules.len()));
    rep.insert("proposal".into(), json!(v.set.proposal));
    if g.mode.is_none() {
        rep.insert("why_not".into(), json!("held-out agreement / minority recall below the graduation bars"));
    }
    rep.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
    f.rules = Some(Arc::new(v.set));
    (f, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q() -> Question {
        Question::from_wire(&json!({"type": "noul", "instructions": "Does it leak?"}), None).unwrap()
    }

    fn rule(p: &str, label: usize) -> Rule {
        Rule { pattern: p.into(), field: None, label, label_name: if label == 1 { "true" } else { "false" }.into(), why: String::new(), heldout_matches: 0, heldout_correct: 0, confidence: 0.0 }
    }

    #[test]
    fn parses_proposals_with_chatter_and_bad_items() {
        let reply = "Sure!\n```json\n{\"rules\": [{\"pattern\": \"AKIA[0-9A-Z]{16}\", \"label\": \"true\", \"why\": \"aws\"},\
            {\"pattern\": \"x\", \"label\": \"maybe\"}, {\"pattern\": \"y\", \"label\": \"no\", \"field\": \"output\"}, {\"label\": true}], \"default\": \"abstain\"}\n```";
        let p = parse_proposal(reply, &q()).unwrap();
        assert_eq!(p.rules.len(), 2);
        assert_eq!((p.rules[0].label, p.rules[1].label, p.rules[1].field.as_deref()), (1, 0, Some("output")));
        assert_eq!(p.rejected.len(), 2);
        assert_eq!(p.default, None);
        assert!(parse_proposal("no json here", &q()).is_err());
        assert!(parse_proposal("{\"default\": \"true\"}", &q()).is_err());
        assert_eq!(parse_proposal("{\"rules\": [], \"default\": \"true\"}", &q()).unwrap().default, Some(1));
    }

    #[test]
    fn first_match_wins_and_abstains() {
        let mut set = RuleSet::default();
        set.rules = vec![rule("secret", 1), rule("sec", 0)];
        set.rules.iter_mut().for_each(|r| r.confidence = 0.9);
        set.compile().unwrap();
        let inp = |t: &str| RuleInput { text: t.into(), fields: vec![("output".into(), t.into())] };
        assert_eq!(set.apply(&inp("a secret")).unwrap().label, 1);
        assert_eq!(set.apply(&inp("a sect")).unwrap().label, 0);
        assert!(set.apply(&inp("nothing")).is_none());
        assert!(set.proba(&inp("nothing"), 2).is_none());
        set.default = Some(0);
        set.default_confidence = 0.8;
        assert_eq!(set.apply(&inp("nothing")).unwrap().label, 0);
        let p = set.proba(&inp("a secret"), 2).unwrap();
        assert!((p[1] - 0.9).abs() < 1e-12 && (p[0] - 0.1).abs() < 1e-12);
        // a field rule searches only its field
        set.rules = vec![Rule { field: Some("call".into()), ..rule("secret", 1) }];
        set.compile().unwrap();
        assert!(set.apply(&inp("a secret")).is_some()); // the default answers: the field is missing
        assert_eq!(set.apply(&inp("a secret")).unwrap().rule, None);
    }

    fn rows_of(texts: &[(String, usize)]) -> (Vec<RuleInput>, Vec<String>) {
        let inputs = texts.iter().map(|(t, _)| RuleInput { text: t.clone(), fields: vec![] }).collect();
        let hashes = texts.iter().enumerate().map(|(i, _)| format!("h{i}")).collect();
        (inputs, hashes)
    }

    #[test]
    fn validation_is_heldout_drops_bad_rules_and_rejects_broad_ones() {
        // 60 rows: "key" -> 1 (30), the rest 0; a decoy word "hash" on half of both classes
        let texts: Vec<(String, usize)> = (0..60).map(|i| (format!("line {i} {} {}", if i % 2 == 0 { "hash" } else { "ok" }, if i % 3 == 0 { "key" } else { "plain" }), usize::from(i % 3 == 0))).collect();
        let (inputs, hashes) = rows_of(&texts);
        let rows: Vec<RuleRow> = texts.iter().enumerate().map(|(i, (_, y))| RuleRow { hash: &hashes[i], input: &inputs[i], y: *y, seq: i as u64, group: None }).collect();
        let set = RuleSet { shown: hashes[..10].to_vec(), ..Default::default() };
        let rules = vec![rule("key", 1), rule("hash", 1), rule("line", 0), rule("(unclosed", 1), rule("plain", 0)];
        let v = validate(rules, Some(0), &rows, set, &q());
        let kept: Vec<&str> = v.set.rules.iter().map(|r| r.pattern.as_str()).collect();
        assert_eq!(kept, vec!["key", "plain"]); // "hash" is 50% precise (its lower bound is < 0.5), "line" matches everything
        let reasons: Vec<&str> = v.set.dropped.iter().map(|d| d.reason.as_str()).collect();
        assert!(reasons.iter().any(|r| r.starts_with("does not compile")));
        assert!(reasons.iter().any(|r| r.starts_with("too broad")));
        assert!(reasons.iter().any(|r| r.starts_with("held-out precision")));
        // held-out = the 50 rows not shown
        assert_eq!(v.held.len(), 50);
        assert!(v.held.iter().all(|&j| j >= 10));
        assert_eq!(v.set.heldout.n, 50);
        let r0 = &v.set.rules[0];
        assert_eq!(r0.heldout_matches, r0.heldout_correct);
        assert!(r0.confidence > 0.8);
        // the answers agree with the lead on the held-out rows
        assert_eq!(v.set.heldout.agreement, Some(1.0));
    }

    #[test]
    fn graduation_from_heldout_rows() {
        let texts: Vec<(String, usize)> = (0..200).map(|i| (format!("row {i} {}", if i % 3 == 0 { "TOKEN" } else { "benign" }), usize::from(i % 3 == 0))).collect();
        let (inputs, hashes) = rows_of(&texts);
        let rows: Vec<RuleRow> = texts.iter().enumerate().map(|(i, (_, y))| RuleRow { hash: &hashes[i], input: &inputs[i], y: *y, seq: i as u64, group: None }).collect();
        let reply = r#"{"rules": [{"pattern": "TOKEN", "label": "true", "why": "t"}, {"pattern": "benign", "label": "false", "why": "b"}], "default": "abstain"}"#;
        let client = RulesClient::new(Box::new(Fixed(reply.to_string())));
        let settings = RulesSettings::default();
        let question = q();
        let ctx = RulesCtx { client: Some(&client), settings: &settings, question: &question, target: 0.9, minority: 0.8, min_examples: 60, seed: 0, z: 1.96 };
        let (fit, st) = fit_rules(&ctx, &rows, None, true, &RulesState::default(), 200);
        assert_eq!(st.proposals, 1);
        assert_eq!(fit.mode, Some(Mode::Symmetric), "{}", fit.report);
        assert!(fit.report["cv_coverage"].as_f64().unwrap() > 0.99);
        let set = fit.rules.as_ref().unwrap();
        assert_eq!(set.shown.len(), SAMPLE_SIZE);
        // a graduated set is revalidated, never re-proposed
        let cur = fit.rules.clone();
        let (fit2, st2) = fit_rules(&ctx, &rows, cur.as_ref(), false, &st, 300);
        assert_eq!(st2.proposals, 1);
        assert_eq!(fit2.mode, Some(Mode::Symmetric));
        assert_eq!(client.live.load(Ordering::SeqCst), 1);
        // too few examples: no proposal
        let (fit3, st3) = fit_rules(&ctx, &rows[..30], None, true, &RulesState::default(), 30);
        assert!(fit3.rules.is_none() && st3.proposals == 0);
    }

    /// An LLM that always says the same thing.
    pub struct Fixed(pub String);
    impl RulesLlm for Fixed {
        fn complete(&self, _: &Value) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn cache_replays_without_the_llm() {
        let dir = std::env::temp_dir().join(format!("understudy-rules-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.jsonl");
        let _ = std::fs::remove_file(&path);
        let a = RulesClient::new(Box::new(Fixed("hello {}".into())));
        a.attach(&path).unwrap();
        let body = json!({"model": "m", "messages": [{"role": "user", "content": "x"}]});
        assert_eq!(a.call(&body).unwrap().1, "hello {}");
        assert_eq!(a.call(&body).unwrap().1, "hello {}");
        assert_eq!((a.live.load(Ordering::SeqCst), a.cached.load(Ordering::SeqCst)), (1, 1));
        struct Boom;
        impl RulesLlm for Boom {
            fn complete(&self, _: &Value) -> Result<String> {
                bail!("offline")
            }
        }
        let b = RulesClient::new(Box::new(Boom));
        assert_eq!(b.attach(&path).unwrap(), 1);
        assert_eq!(b.call(&body).unwrap().1, "hello {}");
        assert!(b.call(&json!({"other": 1})).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sample_is_stratified_older_half_and_deterministic() {
        let texts: Vec<(String, usize)> = (0..300).map(|i| (format!("r{i}"), usize::from(i % 10 == 0))).collect();
        let (inputs, hashes) = rows_of(&texts);
        let rows: Vec<RuleRow> = texts.iter().enumerate().map(|(i, (_, y))| RuleRow { hash: &hashes[i], input: &inputs[i], y: *y, seq: i as u64, group: None }).collect();
        let s = sample(&rows, 2, 7);
        assert_eq!(s.len(), SAMPLE_SIZE);
        assert!(s.iter().all(|&j| j < 150));
        assert_eq!(s.iter().filter(|&&j| rows[j].y == 1).count(), 15); // only 15 positives in the older half
        assert_eq!(s, sample(&rows, 2, 7));
    }

    #[test]
    fn request_key_is_pinned_to_the_pre_rename_prompt() {
        let q = Question::from_wire(&json!({"type": "noul", "instructions": "Does it leak?"}), None).unwrap();
        let (a, b) = (RuleInput { text: "token AKIA1234".into(), fields: vec![] }, RuleInput { text: "hello".into(), fields: vec![] });
        let sample = vec![(1usize, &a), (0usize, &b)];
        let fb = Feedback { listing: "1. /AKIA/ -> true".into(), errors: vec![(1usize, None, &a)] };
        let s = RulesSettings::default();
        let first = request_key(&build_request(&s, &q, &sample, &[1, 1], &[], None));
        let again = request_key(&build_request(&s, &q, &sample, &[1, 1], &[], Some(&fb)));
        assert_eq!(first, "2e9dc2a1df76d862f44b64a007e539e65d12d73b09e6b932bbbb640a9c68b475");
        assert_eq!(again, "f95e8e4e01585707770da2d40835edc049cde20ff4ee5e7a5f6b6631ed1df1c8");
    }
}
