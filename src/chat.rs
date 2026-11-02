//! The chat adapter: an OpenAI-compatible `POST /chat/completions` endpoint used as a lead.
//!
//! Callers (and probes) speak System One. The adapter turns a decision request `{model, questions, state}` into chat
//! calls and the replies back into System One answers, so a chat model can stand in anywhere a System One lead can
//! (`judge`, `replay`, `train`, and a `serve` route with `kind = "chat"`).
//!
//! * **One chat call per question** (not one per request). Each call has a single, small answer schema, a failure
//!   costs one question and not the whole request, and the answer's token carries that question's logprobs, which a
//!   combined answer would mix. The state is repeated per question; a provider's prompt cache absorbs most of that.
//!   Questions of one request run concurrently.
//! * The prompt is fixed ([`prompt`], documented in `docs/api.md`). The reply is requested as `{"answer": ...}` under
//!   a JSON-schema `response_format`; if the endpoint rejects that (HTTP 400/422) the call is retried once plain and
//!   the JSON is parsed out of the text instead.
//! * **Probabilities.** System One returns a probability per class; a chat API does not. When logprobs are returned,
//!   the probability of each class is the (renormalized) mass of the first answer token's top alternatives that
//!   identify that class. Otherwise the chosen class gets a fixed confidence ([`DEFAULT_CONFIDENCE`], 0.9) and the
//!   rest share the remainder. The response says which, per question, in `understudy.probabilities`.

use crate::config::LeadEntry;
use crate::question::{QType, Question};
use serde_json::{Map, Value, json};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

/// The probability put on the chosen class when the endpoint gives no logprobs. A fixed, documented stand-in, not a
/// calibrated estimate.
pub const DEFAULT_CONFIDENCE: f64 = 0.9;
/// Alternatives asked for at the answer token (`top_logprobs`); 5 is what the widest-spread vendors accept.
pub const TOP_LOGPROBS: u32 = 5;
pub const DEFAULT_MAX_TOKENS: u32 = 2048;
/// Share of probability mass the matched alternatives must hold for logprobs to be trusted.
const MIN_MATCHED_MASS: f64 = 0.5;
/// Floor under a class's probability, so none is exactly zero.
const PROB_FLOOR: f64 = 1e-4;
/// Lowest fixed confidence: strictly above one half, so the chosen class of a two-class question stays the argmax
/// after the wire's four-decimal rounding (0.5 would be a tie that flips `true`/`false` and the first choice option).
pub const MIN_CONFIDENCE: f64 = 0.55;
/// Questions of one request in flight at once: a request with thousands of questions neither spawns thousands of
/// threads nor floods the endpoint.
pub const MAX_CONCURRENCY: usize = 8;
/// Largest chat response read, in bytes (a `serve` route lowers it to `--max-response-mib`).
pub const DEFAULT_MAX_RESPONSE: usize = 16 << 20;

pub const SYSTEM_PROMPT: &str = "You are a precise decision function. You are given a question with its criteria and a state. \
Judge the state only against the question and its criteria. Reply with one JSON object of the form {\"answer\": ...} \
and nothing else: no prose and no code fence.";

/// How to call one chat endpoint. It holds no key: the caller passes the `Authorization` value per call.
#[derive(Clone, Debug)]
pub struct ChatSpec {
    /// The full `.../chat/completions` URL.
    pub url: String,
    /// The model to send; none means the request's own `model`.
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub json_schema: bool,
    pub logprobs: bool,
    pub confidence: f64,
    pub max_tokens: u32,
    /// Largest response body read; a bigger one is [`ChatError::TooLarge`], never truncated. None: [`DEFAULT_MAX_RESPONSE`]
    /// (`understudy serve` substitutes its own `--max-response-mib` here unless a route sets a limit).
    pub max_response: Option<usize>,
    /// Set once a plain retry succeeded where the full request was refused: later calls start plain.
    plain: Arc<AtomicBool>,
}

impl ChatSpec {
    pub fn new(url: &str) -> Self {
        ChatSpec {
            url: endpoint(url),
            model: None,
            reasoning_effort: None,
            json_schema: true,
            logprobs: true,
            confidence: DEFAULT_CONFIDENCE,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_response: None,
            plain: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn from_entry(e: &LeadEntry) -> Self {
        let mut s = ChatSpec::new(&e.url);
        s.model = e.model.clone();
        s.reasoning_effort = e.reasoning_effort.clone();
        s.json_schema = e.json_schema.unwrap_or(true);
        s.logprobs = e.logprobs.unwrap_or(true);
        s.confidence = e.confidence.unwrap_or(DEFAULT_CONFIDENCE).clamp(MIN_CONFIDENCE, 1.0);
        s.max_tokens = e.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        s
    }
}

/// `url` as a chat-completions endpoint: a URL already ending in `/chat/completions` is kept, a base URL gets it.
pub fn endpoint(url: &str) -> String {
    let u = url.trim_end_matches('/');
    if u.ends_with("/chat/completions") { u.to_string() } else { format!("{u}/chat/completions") }
}

/// Why a chat call gave no answer.
#[derive(Debug)]
pub enum ChatError {
    /// The endpoint answered with a non-2xx status: relayed as it is.
    Http { status: u16, content_type: Option<String>, retry_after: Option<String>, body: Vec<u8> },
    /// The response (success or error body) is over the size limit; it is not read further.
    TooLarge(usize),
    /// No response (connection, timeout).
    Transport { msg: String, timed_out: bool },
    /// A 2xx whose answer could not be read (no content, no JSON, an answer outside the question's options).
    Unusable(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Http { status, body, .. } => write!(f, "chat lead HTTP {status}: {}", String::from_utf8_lossy(body)),
            ChatError::TooLarge(n) => write!(f, "chat lead response is over the {n}-byte limit"),
            ChatError::Transport { msg, .. } => write!(f, "chat lead transport: {msg}"),
            ChatError::Unusable(m) => write!(f, "chat lead answer unusable: {m}"),
        }
    }
}

impl std::error::Error for ChatError {}

impl ChatError {
    /// Worth another attempt: a transient status, a transport failure or an unreadable answer.
    pub fn retryable(&self) -> bool {
        match self {
            ChatError::Http { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            ChatError::TooLarge(_) => false,
            _ => true,
        }
    }
}

/// The System One response for a request, and how each question's probabilities were made.
#[derive(Debug)]
pub struct ChatResponse {
    pub model: String,
    pub answers: Map<String, Value>,
    /// Per question: `"logprobs"` or `"fixed"`.
    pub sources: Map<String, Value>,
    pub usage: Value,
}

impl ChatResponse {
    /// The System One body: `{model, answers, usage, understudy: {adapter, probabilities}}`.
    pub fn to_systemone(&self) -> Value {
        json!({"model": self.model, "answers": self.answers, "usage": self.usage,
               "understudy": {"adapter": "chat", "probabilities": self.sources}})
    }
}

// ---------------------------------------------------------------- the prompt

/// The state as the prompt shows it: a string as it is, anything else as indented JSON.
fn state_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

fn describe(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The user message for one question and state. The exact template is in `docs/api.md`.
pub fn prompt(q: &Question, state: &Value) -> String {
    let mut s = String::new();
    s.push_str("Question: ");
    s.push_str(if q.instructions.trim().is_empty() { "(none given)" } else { q.instructions.trim() });
    s.push_str("\n\n");
    match (q.qtype, &q.criteria) {
        (QType::Noul, Value::Object(m)) if !m.is_empty() => {
            s.push_str("Criteria:\n");
            for k in ["true", "false"] {
                if let Some(d) = m.get(k) {
                    s.push_str(&format!("- {k}: {}\n", describe(d)));
                }
            }
        }
        (QType::Noul, _) => s.push_str("Criteria:\n- true: yes\n- false: no\n"),
        (QType::Choice, Value::Object(m)) => {
            s.push_str("Criteria:\n");
            for (k, d) in m {
                s.push_str(&format!("- {k}: {}\n", describe(d)));
            }
        }
        (QType::Score, Value::Array(levels)) => {
            s.push_str("Criteria (levels, lowest first):\n");
            for (i, d) in levels.iter().enumerate() {
                s.push_str(&format!("- {i}: {}\n", describe(d)));
            }
        }
        _ => {}
    }
    s.push_str("\nAnswer format: ");
    s.push_str(&match q.qtype {
        QType::Noul => "{\"answer\": true} or {\"answer\": false} (a JSON boolean).".to_string(),
        QType::Choice => format!("{{\"answer\": \"<option>\"}} where <option> is exactly one of: {}.", q.options.join(", ")),
        QType::Score => format!("{{\"answer\": <level>}} where <level> is an integer from 0 to {}.", q.k() - 1),
    });
    s.push_str("\n\nState:\n");
    s.push_str(&state_text(state));
    s
}

/// The `response_format` JSON schema of a question's answer.
pub fn schema(q: &Question) -> Value {
    let inner = match q.qtype {
        QType::Noul => json!({"type": "boolean"}),
        QType::Choice => json!({"type": "string", "enum": q.options}),
        QType::Score => json!({"type": "integer", "enum": (0..q.k()).collect::<Vec<_>>()}),
    };
    json!({"type": "json_schema", "json_schema": {"name": "answer", "strict": true,
        "schema": {"type": "object", "properties": {"answer": inner}, "required": ["answer"], "additionalProperties": false}}})
}

/// The chat-completions body for one question.
pub fn build_body(spec: &ChatSpec, model: &str, q: &Question, state: &Value, plain: bool) -> Value {
    let mut b = json!({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt(q, state)}],
        "temperature": 0,
        "max_tokens": spec.max_tokens,
    });
    let o = b.as_object_mut().unwrap();
    if let Some(e) = &spec.reasoning_effort {
        o.insert("reasoning_effort".into(), json!(e));
    }
    if !plain {
        if spec.json_schema {
            o.insert("response_format".into(), schema(q));
        }
        if spec.logprobs {
            o.insert("logprobs".into(), json!(true));
            o.insert("top_logprobs".into(), json!(TOP_LOGPROBS.min(q.k() as u32).max(2)));
        }
    }
    b
}

// ---------------------------------------------------------------- reading a reply

/// The first JSON value in `text`: the whole text, a fenced block, or the first balanced `{...}` that parses.
pub fn extract_json(text: &str) -> Option<Value> {
    let t = text.trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return Some(v);
    }
    let bytes = t.as_bytes();
    for (start, _) in t.match_indices('{') {
        let (mut depth, mut in_str, mut esc) = (0usize, false, false);
        for (i, &c) in bytes.iter().enumerate().skip(start) {
            if in_str {
                match (esc, c) {
                    (true, _) => esc = false,
                    (false, b'\\') => esc = true,
                    (false, b'"') => in_str = false,
                    _ => {}
                }
                continue;
            }
            match c {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v) = serde_json::from_str::<Value>(&t[start..=i]) {
                            return Some(v);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// The class a parsed reply names: `{"answer": x}` (or a lone field, or a bare scalar).
pub fn class_of(q: &Question, v: &Value) -> Result<usize, String> {
    let a = match v {
        Value::Object(m) => match m.get("answer") {
            Some(a) => a,
            None if m.len() == 1 => m.values().next().unwrap(),
            None => return Err(format!("no \"answer\" field in {v}")),
        },
        other => other,
    };
    let bad = || format!("{a} is not an answer to a {:?} question (options {:?})", q.qtype, q.options);
    match q.qtype {
        QType::Noul => match a {
            Value::Bool(b) => Ok(usize::from(*b)),
            Value::String(s) => match s.trim().to_lowercase().as_str() {
                "true" | "yes" | "1" => Ok(1),
                "false" | "no" | "0" => Ok(0),
                _ => Err(bad()),
            },
            Value::Number(n) => match n.as_i64() {
                Some(0) => Ok(0),
                Some(1) => Ok(1),
                _ => Err(bad()),
            },
            _ => Err(bad()),
        },
        QType::Choice => {
            let s = a.as_str().ok_or_else(bad)?.trim();
            q.options.iter().position(|o| o == s).or_else(|| q.options.iter().position(|o| o.eq_ignore_ascii_case(s))).ok_or_else(bad)
        }
        QType::Score => {
            let i = match a {
                Value::Number(n) => n.as_u64(),
                Value::String(s) => s.trim().parse::<u64>().ok(),
                _ => None,
            };
            i.filter(|&i| (i as usize) < q.k()).map(|i| i as usize).ok_or_else(bad)
        }
    }
}

/// The fixed-confidence probabilities for chosen class `c`.
pub fn fixed_probs(k: usize, c: usize, confidence: f64) -> Vec<f64> {
    if k == 1 {
        return vec![1.0];
    }
    let confidence = confidence.clamp(MIN_CONFIDENCE, 1.0);
    let rest = (1.0 - confidence) / (k - 1) as f64;
    (0..k).map(|i| if i == c { confidence } else { rest }).collect()
}

/// Class probabilities from the answer token's top alternatives. `content` is the chat response's
/// `choices[0].logprobs.content` array. None when the tokens can't be read, the answer token's alternatives don't
/// identify the options, they hold under half the probability, or the chosen class isn't their argmax.
pub fn probs_from_logprobs(q: &Question, chosen: usize, content: &Value) -> Option<Vec<f64>> {
    let toks = content.as_array()?;
    let mut text = String::new();
    let mut spans = vec![];
    for t in toks {
        let s = t["token"].as_str()?;
        spans.push((text.len(), text.len() + s.len()));
        text.push_str(s);
    }
    // the first character of the answer value: after `"answer"`, the colon and an opening quote
    let k = text.rfind("\"answer\"")? + "\"answer\"".len();
    let rest = &text[k..];
    let mut off = k + (rest.len() - rest.trim_start().len());
    if !text[off..].starts_with(':') {
        return None;
    }
    off += 1;
    off += text[off..].len() - text[off..].trim_start().len();
    if text[off..].starts_with('"') {
        off += 1;
    }
    let i = spans.iter().position(|&(s, e)| s <= off && off < e)?;
    let alts = toks[i]["top_logprobs"].as_array()?;
    let mut mass = vec![0.0f64; q.k()];
    for a in alts {
        let (tok, lp) = (a["token"].as_str()?, a["logprob"].as_f64()?);
        let t = tok.trim().trim_matches('"').trim().to_lowercase();
        if t.is_empty() {
            continue;
        }
        let hits: Vec<usize> = (0..q.k()).filter(|&j| q.options[j].to_lowercase().starts_with(&t)).collect();
        if let [j] = hits[..] {
            mass[j] += lp.exp();
        }
    }
    let total: f64 = mass.iter().sum();
    if total < MIN_MATCHED_MASS || mass[chosen] <= 0.0 || crate::stats::argmax(&mass) != chosen {
        return None;
    }
    let mut p: Vec<f64> = mass.iter().map(|m| (m / total).max(PROB_FLOOR)).collect();
    let s: f64 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= s);
    Some(p)
}

/// One question's answer from a 2xx chat response.
struct Read {
    answer: Value,
    source: &'static str,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
}

fn content_text(resp: &Value) -> Option<String> {
    match &resp["choices"][0]["message"]["content"] {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")),
        _ => None,
    }
}

fn read_reply(spec: &ChatSpec, q: &Question, resp: &Value) -> Result<Read, ChatError> {
    let text = content_text(resp).filter(|t| !t.trim().is_empty()).ok_or_else(|| {
        ChatError::Unusable(format!("no content in the reply (finish_reason {})", resp["choices"][0]["finish_reason"]))
    })?;
    let v = extract_json(&text).ok_or_else(|| ChatError::Unusable(format!("no JSON in the reply: {}", text.chars().take(200).collect::<String>())))?;
    let c = class_of(q, &v).map_err(ChatError::Unusable)?;
    let lp = spec.logprobs.then(|| probs_from_logprobs(q, c, &resp["choices"][0]["logprobs"]["content"])).flatten();
    // the answer must read back (as the learners will read it) as the class the text chose
    let keeps_class = |p: &[f64]| q.parse_answer(&q.answer_object_normalized(p).0).is_ok_and(|(k, _)| k == c);
    let (p, source) = match lp.filter(|p| keeps_class(p)) {
        Some(p) => (p, "logprobs"),
        None => (fixed_probs(q.k(), c, spec.confidence), "fixed"),
    };
    let u = &resp["usage"];
    Ok(Read {
        answer: q.answer_object_normalized(&p).0,
        source,
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
        cached_tokens: u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
    })
}

// ---------------------------------------------------------------- calling

fn post(agent: &ureq::Agent, spec: &ChatSpec, auth: Option<&str>, body: &Value) -> Result<Value, ChatError> {
    let mut rq = agent.post(&spec.url).set("content-type", "application/json");
    if let Some(a) = auth {
        rq = rq.set("authorization", a);
    }
    let limit = spec.max_response.unwrap_or(DEFAULT_MAX_RESPONSE);
    // the body, read to at most `limit` bytes: one more means it is over
    let read = |r: ureq::Response| -> Result<Vec<u8>, ChatError> {
        let mut buf = vec![];
        std::io::Read::read_to_end(&mut std::io::Read::take(r.into_reader(), limit as u64 + 1), &mut buf)
            .map_err(|e| ChatError::Transport { timed_out: e.kind() == std::io::ErrorKind::TimedOut, msg: format!("reading the response: {e}") })?;
        if buf.len() > limit { Err(ChatError::TooLarge(limit)) } else { Ok(buf) }
    };
    match rq.send_json(body) {
        Ok(r) => serde_json::from_slice::<Value>(&read(r)?).map_err(|e| ChatError::Unusable(format!("the response is not JSON: {e}"))),
        Err(ureq::Error::Status(status, r)) => {
            let content_type = r.header("content-type").map(str::to_string);
            let retry_after = r.header("retry-after").map(str::to_string);
            Err(ChatError::Http { status, content_type, retry_after, body: read(r)? })
        }
        Err(ureq::Error::Transport(t)) => {
            let msg = t.to_string();
            Err(ChatError::Transport { timed_out: msg.contains("timed out") || msg.contains("timeout"), msg })
        }
    }
}

fn ask_one(agent: &ureq::Agent, spec: &ChatSpec, auth: Option<&str>, model: &str, q: &Question, state: &Value) -> Result<Read, ChatError> {
    let want_plain = spec.plain.load(Ordering::Relaxed);
    match post(agent, spec, auth, &build_body(spec, model, q, state, want_plain)) {
        Ok(resp) => read_reply(spec, q, &resp),
        // the endpoint refused response_format or logprobs: once, ask plainly and read the JSON out of the text
        Err(ChatError::Http { status: 400 | 422, .. }) if !want_plain && (spec.json_schema || spec.logprobs) => {
            match post(agent, spec, auth, &build_body(spec, model, q, state, true)) {
                Ok(resp) => {
                    spec.plain.store(true, Ordering::Relaxed);
                    read_reply(spec, q, &resp)
                }
                // the fallback's own failure is the one that matters (a 429 there must stay retryable)
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

/// Answer a System One request through the chat endpoint: one call per question, concurrently. `auth` is the whole
/// `Authorization` header value (never logged); `request_model` is used when the spec names no model.
pub fn ask(agent: &ureq::Agent, spec: &ChatSpec, auth: Option<&str>, request_model: &str, state: &Value, questions: &Map<String, Value>) -> Result<ChatResponse, ChatError> {
    let model = spec.model.as_deref().unwrap_or(request_model);
    let mut qs = vec![];
    for (name, wire) in questions {
        let q = Question::from_wire(wire, None).map_err(|e| ChatError::Unusable(format!("question {name:?}: {e}")))?;
        qs.push((name, q));
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<Result<Read, ChatError>>>> = qs.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..qs.len().min(MAX_CONCURRENCY) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((_, q)) = qs.get(i) else { break };
                *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(ask_one(agent, spec, auth, model, q, state));
            });
        }
    });
    let reads: Vec<Result<Read, ChatError>> = slots
        .into_iter()
        .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()).unwrap_or_else(|| Err(ChatError::Unusable("the chat worker panicked".into()))))
        .collect();
    let mut out = ChatResponse { model: model.to_string(), answers: Map::new(), sources: Map::new(), usage: Value::Null };
    let (mut pt, mut ct, mut cp) = (0, 0, 0);
    for ((name, _), r) in qs.iter().zip(reads) {
        let r = r?;
        out.answers.insert((*name).clone(), r.answer);
        out.sources.insert((*name).clone(), json!(r.source));
        pt += r.prompt_tokens;
        cp += r.cached_tokens;
        ct += r.completion_tokens;
    }
    out.usage = json!({"input_tokens": pt, "cached_input_tokens": cp, "output_tokens": ct});
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: Value) -> Question {
        Question::from_wire(&v, None).unwrap()
    }

    #[test]
    fn endpoint_accepts_base_or_full_urls() {
        assert_eq!(endpoint("https://h/v1"), "https://h/v1/chat/completions");
        assert_eq!(endpoint("https://h/v1/"), "https://h/v1/chat/completions");
        assert_eq!(endpoint("https://h/v1/chat/completions"), "https://h/v1/chat/completions");
    }

    #[test]
    fn json_extraction_handles_fences_prose_and_nesting() {
        assert_eq!(extract_json("{\"answer\": true}").unwrap()["answer"], true);
        assert_eq!(extract_json("```json\n{\"answer\": \"bug\"}\n```").unwrap()["answer"], "bug");
        assert_eq!(extract_json("Sure! Here you go: {\"answer\": 2} Hope that helps {").unwrap()["answer"], 2);
        assert_eq!(extract_json("{\"x\": \"}\", \"answer\": 1}").unwrap()["answer"], 1);
        assert_eq!(extract_json("true").unwrap(), json!(true));
        assert!(extract_json("I cannot decide.").is_none());
    }

    #[test]
    fn answers_map_to_classes() {
        let n = q(json!({"type": "noul", "instructions": "i"}));
        assert_eq!(class_of(&n, &json!({"answer": true})), Ok(1));
        assert_eq!(class_of(&n, &json!({"answer": "No"})), Ok(0));
        assert!(class_of(&n, &json!({"answer": "maybe"})).is_err());
        let c = q(json!({"type": "choice", "instructions": "i", "criteria": {"billing": "b", "bug": "x"}}));
        assert_eq!(class_of(&c, &json!({"answer": "bug"})), Ok(1));
        assert_eq!(class_of(&c, &json!({"answer": "BUG"})), Ok(1));
        assert!(class_of(&c, &json!({"answer": "other"})).is_err());
        let s = q(json!({"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]}));
        assert_eq!(class_of(&s, &json!({"answer": 2})), Ok(2));
        assert!(class_of(&s, &json!({"answer": 3})).is_err());
    }

    #[test]
    fn logprobs_become_probabilities_or_nothing() {
        let c = q(json!({"type": "choice", "instructions": "i", "criteria": {"billing": "b", "bug": "x", "howto": "h"}}));
        let content = json!([
            {"token": "{\"", "logprob": 0.0, "top_logprobs": []},
            {"token": "answer", "logprob": 0.0, "top_logprobs": []},
            {"token": "\":", "logprob": 0.0, "top_logprobs": []},
            {"token": " \"", "logprob": 0.0, "top_logprobs": []},
            {"token": "bug", "logprob": -0.2231, "top_logprobs": [
                {"token": "bug", "logprob": -0.2231}, {"token": "bill", "logprob": -1.6094}, {"token": "ho", "logprob": -4.6}, {"token": "x", "logprob": -9.0}]},
            {"token": "\"}", "logprob": 0.0, "top_logprobs": []}
        ]);
        let p = probs_from_logprobs(&c, 1, &content).unwrap();
        assert!(p[1] > 0.75 && p[0] > 0.15 && p[2] < 0.05, "{p:?}");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // the text's class is not the logprobs' argmax: no logprob probabilities
        assert!(probs_from_logprobs(&c, 0, &content).is_none());
        assert!(probs_from_logprobs(&c, 1, &json!([])).is_none());
        let fx = fixed_probs(3, 1, 0.9);
        assert!(fx[1] == 0.9 && (fx[0] - 0.05).abs() < 1e-12 && (fx[2] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn body_carries_schema_and_logprobs_unless_plain() {
        let spec = ChatSpec::new("https://h/v1");
        let n = q(json!({"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "t", "false": "f"}}));
        let b = build_body(&spec, "m", &n, &json!("hello"), false);
        assert_eq!(b["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"], "boolean");
        assert_eq!(b["logprobs"], true);
        assert_eq!(b["temperature"], 0);
        assert!(b["messages"][1]["content"].as_str().unwrap().ends_with("State:\nhello"));
        let b = build_body(&spec, "m", &n, &json!("hello"), true);
        assert!(b.get("response_format").is_none() && b.get("logprobs").is_none());
    }
}
