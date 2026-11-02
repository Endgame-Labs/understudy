//! The lead config file: named lead presets (and the rules LLM) kept outside the code, so no endpoint is built in.
//!
//! The file is `$UNDERSTUDY_LEADS` if set, else `$XDG_CONFIG_HOME/understudy/leads.toml`, else
//! `~/.config/understudy/leads.toml`. A missing default file is an empty config; a missing `$UNDERSTUDY_LEADS` file is an
//! error. See `examples/leads.example.toml`.

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What a lead speaks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LeadKind {
    /// The System One decision API: `{model, questions, state}` -> `{answers}`.
    #[default]
    SystemOne,
    /// An OpenAI-compatible `POST /chat/completions`, driven through the chat adapter (`chat.rs`).
    Chat,
}

impl LeadKind {
    pub fn name(self) -> &'static str {
        match self {
            LeadKind::SystemOne => "systemone",
            LeadKind::Chat => "chat",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "systemone" | "system_one" | "http" => Ok(LeadKind::SystemOne),
            "chat" => Ok(LeadKind::Chat),
            other => bail!("unknown lead kind {other:?}: systemone or chat"),
        }
    }
}

/// One `[leads.NAME]` table.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeadEntry {
    #[serde(default)]
    pub kind: LeadKind,
    pub url: String,
    #[serde(default)]
    pub model: Option<String>,
    /// For `serve`: the request models this lead answers (exact, or a prefix ending in `*`). Default: its `model`.
    #[serde(default)]
    pub route: Option<String>,
    /// The environment variable holding the bearer key (never the key itself). Omit for an endpoint that needs none.
    #[serde(default)]
    pub key_env: Option<String>,
    /// For `kind = "chat"`: sent as `reasoning_effort` when set.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// For `kind = "chat"`: send `response_format` with a JSON schema (default true; falls back by itself on a 4xx).
    #[serde(default)]
    pub json_schema: Option<bool>,
    /// For `kind = "chat"`: ask for logprobs to derive probabilities (default true).
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// For `kind = "chat"`: the fixed confidence used when no logprobs come back (default 0.9).
    #[serde(default)]
    pub confidence: Option<f64>,
    /// For `kind = "chat"`: `max_tokens` (default 2048, room for a reasoning model's thinking).
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// The `[rules]` table: the chat LLM that writes the `rules` understudent's rules.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RulesEntry {
    pub url: Option<String>,
    pub model: Option<String>,
    pub key_env: Option<String>,
    pub reasoning_effort: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub leads: BTreeMap<String, LeadEntry>,
    #[serde(default)]
    pub rules: Option<RulesEntry>,
    /// Where it was read from (none: no file).
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

/// The config file the environment names (it may not exist).
pub fn default_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("UNDERSTUDY_LEADS").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let base = std::env::var_os("XDG_CONFIG_HOME").filter(|p| !p.is_empty()).map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("understudy").join("leads.toml"))
}

impl Config {
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).map_err(|e| anyhow!("{e}"))
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading lead config {}", path.display()))?;
        let mut c = Self::parse(&text).with_context(|| format!("lead config {}", path.display()))?;
        c.path = Some(path.to_path_buf());
        Ok(c)
    }

    /// `$UNDERSTUDY_LEADS` (which must exist) or the default file (which may not: then empty).
    pub fn load() -> Result<Self> {
        if let Some(p) = std::env::var_os("UNDERSTUDY_LEADS").filter(|p| !p.is_empty()) {
            return Self::from_file(Path::new(&p));
        }
        match default_path() {
            Some(p) if p.is_file() => Self::from_file(&p),
            _ => Ok(Config::default()),
        }
    }

    /// A preset by name; the error says where presets come from.
    pub fn lead(&self, name: &str) -> Result<&LeadEntry> {
        self.leads.get(name).ok_or_else(|| {
            let known: Vec<&str> = self.leads.keys().map(String::as_str).collect();
            match &self.path {
                Some(p) if !known.is_empty() => anyhow!("unknown lead {name:?}: {} defines {}; add a [leads.{name}] table or pass --lead-url and --lead-model", p.display(), known.join(", ")),
                Some(p) => anyhow!("unknown lead {name:?}: {} defines no leads; add a [leads.{name}] table or pass --lead-url and --lead-model", p.display()),
                None => anyhow!(
                    "unknown lead {name:?}: no lead config file found (set $UNDERSTUDY_LEADS or create {}); see examples/leads.example.toml, or pass --lead-url and --lead-model",
                    default_path().map_or("~/.config/understudy/leads.toml".into(), |p| p.display().to_string())
                ),
            }
        })
    }
}

/// The bearer key an environment variable holds. Empty or `implicit` means none (an integration injects it). Never
/// printed; an unset variable is an error, so a missing key is not mistaken for an open endpoint.
pub fn key_from_env(var: &str) -> Result<Option<String>> {
    match std::env::var(var) {
        Ok(v) => {
            let v = v.trim().to_string();
            Ok((!v.is_empty() && v != "implicit").then_some(v))
        }
        Err(_) => bail!("the environment variable {var} (named for a lead key) is not set"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOML: &str = r#"
[leads.sys]
url = "https://example.com/v1/systemone"
model = "m1"
key_env = "SYS_KEY"

[leads.chatty]
kind = "chat"
url = "https://api.example.com/v1"
model = "small"
reasoning_effort = "low"
logprobs = false
confidence = 0.8

[rules]
url = "https://api.example.com/v1/chat/completions"
model = "writer"
"#;

    #[test]
    fn parses_leads_and_rules() {
        let c = Config::parse(TOML).unwrap();
        let s = c.lead("sys").unwrap();
        assert_eq!((s.kind, s.url.as_str(), s.model.as_deref(), s.key_env.as_deref()), (LeadKind::SystemOne, "https://example.com/v1/systemone", Some("m1"), Some("SYS_KEY")));
        let ch = c.lead("chatty").unwrap();
        assert_eq!((ch.kind, ch.logprobs, ch.confidence, ch.reasoning_effort.as_deref()), (LeadKind::Chat, Some(false), Some(0.8), Some("low")));
        assert_eq!(c.rules.as_ref().unwrap().model.as_deref(), Some("writer"));
    }

    #[test]
    fn unknown_lead_and_bad_files_are_clear_errors() {
        let c = Config::parse(TOML).unwrap();
        assert!(c.lead("nope").unwrap_err().to_string().contains("unknown lead \"nope\""));
        let empty = Config::default();
        assert!(empty.lead("rune").unwrap_err().to_string().contains("no lead config file"));
        // a typo'd field and a missing url are rejected, not ignored
        assert!(Config::parse("[leads.x]\nurl = \"u\"\nmodle = \"m\"\n").is_err());
        assert!(Config::parse("[leads.x]\nmodel = \"m\"\n").is_err());
        assert!(Config::parse("[leads.x]\nkind = \"grpc\"\nurl = \"u\"\n").is_err());
    }

    #[test]
    fn file_loading_and_keys() {
        let dir = std::env::temp_dir().join(format!("understudy-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("leads.toml");
        std::fs::write(&p, TOML).unwrap();
        let c = Config::from_file(&p).unwrap();
        assert_eq!(c.path.as_deref(), Some(p.as_path()));
        assert!(c.lead("nope").unwrap_err().to_string().contains("sys, chatty") || c.lead("nope").unwrap_err().to_string().contains("chatty, sys"));
        assert!(Config::from_file(&dir.join("missing.toml")).is_err());
        unsafe { std::env::set_var("UNDERSTUDY_CFG_TEST_KEY", " k ") };
        assert_eq!(key_from_env("UNDERSTUDY_CFG_TEST_KEY").unwrap().as_deref(), Some("k"));
        assert!(key_from_env("UNDERSTUDY_CFG_TEST_UNSET").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
