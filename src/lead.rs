//! Leads: anything that answers systemone requests. `Http` speaks the System One API to any such endpoint (they differ
//! only by URL and model); `Chat` drives an OpenAI-compatible chat endpoint through the adapter in `chat.rs`;
//! `Replay` answers from a stream's recorded `lead_probs`, offline and deterministically. Endpoints are named in the
//! lead config file (`config.rs`), never built in.

use crate::chat::{self, ChatSpec};
use crate::clef::{self, ClefError, ClefSpec};
use crate::config::{self, Config, LeadKind};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::time::Duration;

pub trait Lead: Send + Sync {
    /// The answers object (`{question name: answer}`) for one state. `id` is the item's id when it has one.
    fn ask(&self, id: Option<&str>, state: &Value, questions: &Map<String, Value>) -> Result<Map<String, Value>>;
    /// What to record in a snapshot: `{"kind", "url", "model"}`.
    fn describe(&self) -> Value;
}

/// A systemone endpoint over HTTP. A bearer key, where the endpoint needs one, is held in memory only: it is never
/// logged, printed, put in `describe()` or written to a snapshot.
pub struct Http {
    pub url: String,
    pub model: String,
    key: Option<String>,
    agent: ureq::Agent,
    pub attempts: usize,
}

impl Http {
    pub fn new(url: &str, model: &str) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(120)).build();
        Http { url: url.into(), model: model.into(), key: None, agent, attempts: 4 }
    }

    pub fn with_key(mut self, key: Option<String>) -> Self {
        self.key = key;
        self
    }
}

/// A chat endpoint (OpenAI-compatible `/chat/completions`) as a lead, through the adapter in `chat.rs`.
pub struct Chat {
    pub spec: ChatSpec,
    /// The model sent (the spec's, else this).
    pub model: String,
    key: Option<String>,
    agent: ureq::Agent,
    pub attempts: usize,
}

impl Chat {
    pub fn new(spec: ChatSpec, model: &str) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(300)).build();
        Chat { spec, model: model.into(), key: None, agent, attempts: 4 }
    }

    pub fn with_key(mut self, key: Option<String>) -> Self {
        self.key = key;
        self
    }
}

impl Lead for Chat {
    fn ask(&self, _id: Option<&str>, state: &Value, questions: &Map<String, Value>) -> Result<Map<String, Value>> {
        let auth = self.key.as_ref().map(|k| format!("Bearer {k}"));
        let mut wait = Duration::from_secs(2);
        let mut last = None;
        for attempt in 0..self.attempts {
            if attempt > 0 {
                std::thread::sleep(wait);
                wait *= 2;
            }
            match chat::ask(&self.agent, &self.spec, auth.as_deref(), &self.model, state, questions) {
                Ok(r) => return Ok(r.answers),
                Err(e) if e.retryable() => last = Some(e),
                Err(e) => bail!("{e}"),
            }
        }
        Err(last.map_or_else(|| anyhow!("lead failed"), |e| anyhow!("{e}")))
    }

    fn describe(&self) -> Value {
        json!({"kind": "chat", "url": self.spec.url, "model": self.spec.model.as_deref().unwrap_or(&self.model)})
    }
}

/// Cloudflare Workers AI Clef / Clef Flash as a lead, through `clef.rs`: the System One request goes unchanged to the
/// model's URL and the reply is unwrapped from Cloudflare's envelope. The key is held in memory only.
pub struct Clef {
    pub spec: ClefSpec,
    pub model: String,
    key: Option<String>,
    agent: ureq::Agent,
    pub attempts: usize,
}

impl Clef {
    pub fn new(spec: ClefSpec, model: &str) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(120)).build();
        Clef { spec, model: model.into(), key: None, agent, attempts: 4 }
    }

    pub fn with_key(mut self, key: Option<String>) -> Self {
        self.key = key;
        self
    }
}

impl Lead for Clef {
    fn ask(&self, _id: Option<&str>, state: &Value, questions: &Map<String, Value>) -> Result<Map<String, Value>> {
        let url = self.spec.url(&self.model).map_err(|e| anyhow!("{e}"))?;
        let auth = self.key.as_ref().map(|k| format!("Bearer {k}"));
        let body = serde_json::to_vec(&json!({"model": clef::model_name(&self.model).unwrap_or(&self.model), "questions": questions, "state": state}))?;
        let mut wait = Duration::from_secs(2);
        let mut last = None;
        for attempt in 0..self.attempts {
            if attempt > 0 {
                std::thread::sleep(wait);
                wait *= 2;
            }
            match clef::call(&self.agent, &url, auth.as_deref(), &body, clef::DEFAULT_MAX_RESPONSE) {
                Ok(b) => {
                    let v: Value = serde_json::from_slice(&b).context("lead response is not JSON")?;
                    return match v.get("answers") {
                        Some(Value::Object(a)) => Ok(a.clone()),
                        _ => bail!("the lead's response has no answers object: {v}"),
                    };
                }
                Err(e @ ClefError::Http { .. } | e @ ClefError::Transport { .. }) if e.retryable() => last = Some(e),
                Err(e) => bail!("{e}"),
            }
        }
        Err(last.map_or_else(|| anyhow!("lead failed"), |e| anyhow!("{e}")))
    }

    /// The API base, not the URL: the account id is never recorded.
    fn describe(&self) -> Value {
        json!({"kind": "clef", "url": self.spec.base, "model": self.model})
    }
}

/// What the lead flags name. `name` is a preset from the lead config file; `url`, `model` and `kind` override or
/// stand in for it; `key_env` names the environment variable holding a bearer key.
#[derive(Default, Clone, Debug)]
pub struct LeadFlags<'a> {
    pub name: Option<&'a str>,
    pub url: Option<&'a str>,
    pub model: Option<&'a str>,
    pub kind: Option<LeadKind>,
    pub key_env: Option<&'a str>,
}

/// Build a lead from flags over `cfg` (only consulted when a preset name is given).
pub fn build(cfg: &dyn Fn() -> Result<Config>, f: &LeadFlags) -> Result<Box<dyn Lead>> {
    let entry = match f.name {
        Some(n) => Some(cfg()?.lead(n)?.clone()),
        None => None,
    };
    let kind = f.kind.or(entry.as_ref().map(|e| e.kind)).unwrap_or_default();
    let url = f.url.map(str::to_string).or(entry.as_ref().map(|e| e.url.clone()).filter(|u| !u.is_empty())).or_else(|| (kind == LeadKind::Clef).then(String::new)).ok_or_else(|| anyhow!("a lead needs --lead (a preset in the lead config file) or --lead-url"))?;
    let model = f.model.map(str::to_string).or(entry.as_ref().and_then(|e| e.model.clone()));
    let key_env = f.key_env.or(entry.as_ref().and_then(|e| e.key_env.as_deref()));
    let key = match key_env {
        Some(v) => config::key_from_env(v)?,
        // the documented default for a clef lead; with none set, the endpoint (or a proxy) supplies its own credentials
        None if kind == LeadKind::Clef => std::env::var(clef::TOKEN_ENV).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v != "implicit"),
        None => None,
    };
    Ok(match kind {
        LeadKind::SystemOne => {
            let model = model.ok_or_else(|| anyhow!("a systemone lead needs a model: --lead-model, or `model` in its preset"))?;
            Box::new(Http::new(&url, &model).with_key(key))
        }
        LeadKind::Chat => {
            let mut spec = entry.as_ref().map_or_else(|| ChatSpec::new(&url), ChatSpec::from_entry);
            if f.url.is_some() {
                spec.url = chat::endpoint(&url);
            }
            let model = model.ok_or_else(|| anyhow!("a chat lead needs a model: --lead-model, or `model` in its preset"))?;
            spec.model = Some(model.clone());
            Box::new(Chat::new(spec, &model).with_key(key))
        }
        LeadKind::Clef => {
            let model = model.ok_or_else(|| anyhow!("a clef lead needs a model: --lead-model (clef or clef-flash), or `model` in its preset"))?;
            let spec = ClefSpec::resolve(Some(&url), entry.as_ref().and_then(|e| e.account.as_deref()));
            spec.url(&model).map_err(|e| anyhow!("{e}"))?; // the account and model are checked now, not at the first call
            Box::new(Clef::new(spec, &model).with_key(key))
        }
    })
}

impl Lead for Http {
    fn ask(&self, _id: Option<&str>, state: &Value, questions: &Map<String, Value>) -> Result<Map<String, Value>> {
        let body = json!({"model": self.model, "questions": questions, "state": state});
        let mut wait = Duration::from_secs(2);
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
            match rq.send_json(&body) {
                Ok(r) => {
                    let v: Value = r.into_json().context("lead response is not JSON")?;
                    return match v.get("answers") {
                        Some(Value::Object(a)) => Ok(a.clone()),
                        _ => bail!("the lead's response has no answers object: {v}"),
                    };
                }
                Err(ureq::Error::Status(code, r)) if code == 408 || code == 429 || code >= 500 => {
                    last = Some(anyhow!("HTTP {code}: {}", r.into_string().unwrap_or_default()));
                }
                Err(ureq::Error::Status(code, r)) => bail!("lead HTTP {code}: {}", r.into_string().unwrap_or_default()),
                Err(e) => last = Some(anyhow!("lead transport: {e}")),
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("lead failed")))
    }

    fn describe(&self) -> Value {
        json!({"kind": "http", "url": self.url, "model": self.model})
    }
}

/// Recorded answers, by item id: a stream's `lead_probs`. Every question in a request gets the item's
/// probabilities (streams are single-question).
pub struct Replay {
    pub probs: HashMap<String, Vec<f64>>,
    pub source: String,
}

impl Replay {
    /// From stream.jsonl lines `{"id", "lead_probs", ...}`.
    pub fn from_stream(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let mut probs = HashMap::new();
        for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), i + 1))?;
            let id = v.get("id").and_then(Value::as_str).ok_or_else(|| anyhow!("{}:{}: no id", path.display(), i + 1))?;
            if let Some(p) = v.get("lead_probs").and_then(Value::as_array) {
                probs.insert(id.to_string(), p.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect());
            }
        }
        Ok(Replay { probs, source: path.display().to_string() })
    }
}

impl Lead for Replay {
    fn ask(&self, id: Option<&str>, _state: &Value, questions: &Map<String, Value>) -> Result<Map<String, Value>> {
        let id = id.ok_or_else(|| anyhow!("the replay lead needs item ids"))?;
        let p = self.probs.get(id).ok_or_else(|| anyhow!("the replay lead has no answer for {id:?}"))?;
        Ok(questions.keys().map(|q| (q.clone(), json!(p))).collect())
    }

    fn describe(&self) -> Value {
        json!({"kind": "replay", "source": self.source})
    }
}
