//! The Clef adapter: Cloudflare Workers AI's `@cf/cloudflare/clef` and `clef-flash` as a lead.
//!
//! Both models take a System One request body unchanged. The only differences are the URL (one per model, under an
//! account) and the reply, which Cloudflare wraps in its API envelope:
//! `{"result": {"model", "answers", "usage"}, "success": true, "errors": [], "messages": []}`.
//! This module builds the URL, sends the body as it is, unwraps `result` into the System One response, and turns a
//! failed envelope into an error that keeps Cloudflare's status, `Retry-After` and messages.

use serde_json::{Value, json};

/// Where the API is, unless the config or `$CLOUDFLARE_API_BASE` says otherwise.
pub const DEFAULT_BASE: &str = "https://api.cloudflare.com";
/// The environment variable holding the account id (never hardcoded).
pub const ACCOUNT_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";
/// The environment variable overriding the API base.
pub const BASE_ENV: &str = "CLOUDFLARE_API_BASE";
/// The environment variable holding the default bearer token of a clef route.
pub const TOKEN_ENV: &str = "CLOUDFLARE_API_TOKEN";
/// The prefix of a full Workers AI model name.
pub const MODEL_PREFIX: &str = "@cf/cloudflare/";
/// The response size limit when the caller sets none.
pub const DEFAULT_MAX_RESPONSE: usize = 16 << 20;
/// What a route's URL shows in place of the account id (in `/status`, namespaces and snapshots).
pub const ACCOUNT_PLACEHOLDER: &str = "<account_id>";

/// The model name a request's `model` stands for: `clef`, `clef-flash` or `@cf/cloudflare/<model>`. None if it has
/// characters that do not belong in a path segment.
pub fn model_name(model: &str) -> Option<&str> {
    let m = model.strip_prefix(MODEL_PREFIX).unwrap_or(model);
    let mut chars = m.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_alphanumeric()) && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    valid.then_some(m)
}

/// The request body for Cloudflare: the caller's bytes unchanged, except that a `model` spelled `@cf/cloudflare/<name>`
/// is rewritten to `<name>`, because the API accepts only `clef` or `clef-flash` in the body.
pub fn request_body(body: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if let Ok(mut v) = serde_json::from_slice::<Value>(body) {
        if let Some(name) = v["model"].as_str().and_then(|m| m.strip_prefix(MODEL_PREFIX)).map(str::to_string) {
            v["model"] = Value::String(name);
            if let Ok(b) = serde_json::to_vec(&v) {
                return b.into();
            }
        }
    }
    body.into()
}

/// Which account and API base a clef lead calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClefSpec {
    /// The API base, without a trailing slash.
    pub base: String,
    /// The account id; None until resolved (an error at the first call that needs it).
    pub account: Option<String>,
}

impl ClefSpec {
    /// From a configured base and account, else `$CLOUDFLARE_API_BASE` / `$CLOUDFLARE_ACCOUNT_ID`, else the default base.
    /// A missing account is not an error here; [`ClefSpec::url`] raises it.
    pub fn resolve(base: Option<&str>, account: Option<&str>) -> Self {
        let env = |v: &str| std::env::var(v).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let base = base.map(str::to_string).filter(|s| !s.trim().is_empty()).or_else(|| env(BASE_ENV)).unwrap_or_else(|| DEFAULT_BASE.into());
        let account = account.map(str::to_string).filter(|s| !s.trim().is_empty()).or_else(|| env(ACCOUNT_ENV));
        ClefSpec { base: base.trim_end_matches('/').to_string(), account }
    }

    /// The URL for `model` (any accepted spelling).
    pub fn url(&self, model: &str) -> Result<String, String> {
        let name = model_name(model).ok_or_else(|| format!("{model:?} is not a Clef model name (clef, clef-flash or {MODEL_PREFIX}<model>)"))?;
        let account = self.account.as_deref().ok_or_else(|| format!("a clef lead needs the Cloudflare account id: set `account` in its lead config or ${ACCOUNT_ENV}"))?;
        if account.is_empty() || !account.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err("the Cloudflare account id has characters that do not belong in a URL path".into());
        }
        Ok(format!("{}/client/v4/accounts/{account}/ai/run/{MODEL_PREFIX}{name}", self.base))
    }

    /// A URL for display and identity: the route's, with the account id as a placeholder.
    pub fn display_url(&self) -> String {
        format!("{}/client/v4/accounts/{ACCOUNT_PLACEHOLDER}/ai/run/{}", self.base, MODEL_PREFIX.trim_end_matches('/'))
    }
}

/// Why a clef call gave no System One answer.
#[derive(Debug)]
pub enum ClefError {
    /// A failure to relay: Cloudflare's HTTP status when it sent a non-2xx one, else 502 for a failed envelope. The
    /// body is a JSON error carrying Cloudflare's messages.
    Http { status: u16, retry_after: Option<String>, body: Vec<u8> },
    /// The response is over the size limit; it is not read further.
    TooLarge(usize),
    /// No response (connection, timeout, or the request could not be built).
    Transport { msg: String, timed_out: bool },
    /// A 2xx that is not an envelope with an object `result`.
    Unusable(String),
}

impl std::fmt::Display for ClefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClefError::Http { status, body, .. } => write!(f, "clef lead HTTP {status}: {}", String::from_utf8_lossy(body)),
            ClefError::TooLarge(n) => write!(f, "clef lead response is over the {n}-byte limit"),
            ClefError::Transport { msg, .. } => write!(f, "clef lead transport: {msg}"),
            ClefError::Unusable(m) => write!(f, "clef lead answer unusable: {m}"),
        }
    }
}

impl std::error::Error for ClefError {}

impl ClefError {
    /// Worth another attempt: a transient status or a transport failure.
    pub fn retryable(&self) -> bool {
        match self {
            ClefError::Http { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            ClefError::Transport { .. } => true,
            ClefError::TooLarge(_) | ClefError::Unusable(_) => false,
        }
    }
}

/// The messages of an envelope's `errors` array: each is `{"code", "message"}`, or a bare string.
fn error_messages(errors: &Value) -> Vec<String> {
    errors
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| match (e["code"].as_i64(), e["message"].as_str().or(e.as_str())) {
                    (Some(c), Some(m)) => format!("{m} (code {c})"),
                    (None, Some(m)) => m.to_string(),
                    _ => e.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn failure(status: u16, retry_after: Option<String>, messages: Vec<String>, errors: Value) -> ClefError {
    let message = if messages.is_empty() { "Cloudflare reported a failure with no message".to_string() } else { messages.join("; ") };
    let body = serde_json::to_vec(&json!({"error": {"type": "upstream_error", "message": format!("Cloudflare Workers AI: {message}"), "cloudflare_errors": errors}})).unwrap_or_default();
    ClefError::Http { status, retry_after, body }
}

/// Unwrap a Cloudflare response into the System One body (the envelope's `result`, as JSON bytes).
///
/// * 2xx with `success` not false, no `errors`, and an object `result`: the result.
/// * 2xx with `success: false` or a non-empty `errors`: [`ClefError::Http`] with status 502.
/// * non-2xx: that status, with the envelope's messages when the body has them, else a snippet of the body.
pub fn unwrap_envelope(status: u16, retry_after: Option<String>, body: &[u8]) -> Result<Vec<u8>, ClefError> {
    let v: Option<Value> = serde_json::from_slice(body).ok();
    if !(200..300).contains(&status) {
        let (msgs, errs) = match &v {
            Some(v) if !error_messages(&v["errors"]).is_empty() => (error_messages(&v["errors"]), v["errors"].clone()),
            _ => (vec![String::from_utf8_lossy(&body[..body.len().min(300)]).trim().to_string()].into_iter().filter(|s| !s.is_empty()).collect(), json!([])),
        };
        return Err(failure(status, retry_after, msgs, errs));
    }
    let v = v.ok_or_else(|| ClefError::Unusable("the response is not JSON".into()))?;
    let msgs = error_messages(&v["errors"]);
    if v["success"] == json!(false) || !msgs.is_empty() {
        return Err(failure(502, retry_after, msgs, v["errors"].clone()));
    }
    match v.get("result") {
        Some(r @ Value::Object(_)) => Ok(serde_json::to_vec(r).unwrap_or_default()),
        _ => Err(ClefError::Unusable("the response has no result object".into())),
    }
}

/// POST `body` (a System One request, as it is) to `url` and return the unwrapped System One response body.
/// `auth` is the whole `Authorization` header value, if any. The response is read to at most `max_response` bytes.
pub fn call(agent: &ureq::Agent, url: &str, auth: Option<&str>, body: &[u8], max_response: usize) -> Result<Vec<u8>, ClefError> {
    let body = request_body(body);
    let mut rq = agent.post(url).set("content-type", "application/json").set("accept-encoding", "identity");
    if let Some(a) = auth {
        rq = rq.set("authorization", a);
    }
    let read = |r: ureq::Response| -> Result<Vec<u8>, ClefError> {
        let mut buf = vec![];
        std::io::Read::read_to_end(&mut std::io::Read::take(r.into_reader(), max_response as u64 + 1), &mut buf)
            .map_err(|e| ClefError::Transport { timed_out: e.kind() == std::io::ErrorKind::TimedOut, msg: format!("reading the response: {e}") })?;
        if buf.len() > max_response { Err(ClefError::TooLarge(max_response)) } else { Ok(buf) }
    };
    match rq.send_bytes(&body) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => {
            let status = r.status();
            let retry_after = r.header("retry-after").map(str::to_string);
            unwrap_envelope(status, retry_after, &read(r)?)
        }
        Err(ureq::Error::Transport(t)) => {
            let msg = t.to_string();
            Err(ClefError::Transport { timed_out: msg.contains("timed out") || msg.contains("timeout"), msg })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_and_urls() {
        assert_eq!(model_name("clef"), Some("clef"));
        assert_eq!(model_name("@cf/cloudflare/clef-flash"), Some("clef-flash"));
        assert_eq!(model_name("clef/../x"), None);
        assert_eq!(model_name(""), None);
        let s = ClefSpec { base: "https://api.example.test".into(), account: Some("acct1".into()) };
        assert_eq!(s.url("clef-flash").unwrap(), "https://api.example.test/client/v4/accounts/acct1/ai/run/@cf/cloudflare/clef-flash");
        assert_eq!(s.url("@cf/cloudflare/clef").unwrap(), "https://api.example.test/client/v4/accounts/acct1/ai/run/@cf/cloudflare/clef");
        let none = ClefSpec { base: "https://x".into(), account: None };
        assert!(none.url("clef").unwrap_err().contains(ACCOUNT_ENV));
        assert!(!s.display_url().contains("acct1"));
        assert_eq!(&*request_body(b"{\"model\":\"@cf/cloudflare/clef\",\"state\":1}"), b"{\"model\":\"clef\",\"state\":1}");
        let plain = b"{ \"model\": \"clef\" }";
        assert_eq!(&*request_body(plain), plain);
    }

    #[test]
    fn envelopes() {
        let ok = br#"{"result":{"model":"clef","answers":{"q":{"type":"noul","noul":0.5}}},"success":true,"errors":[],"messages":[]}"#;
        let v: Value = serde_json::from_slice(&unwrap_envelope(200, None, ok).unwrap()).unwrap();
        assert_eq!(v["answers"]["q"]["noul"], 0.5);
        let bad = br#"{"result":null,"success":false,"errors":[{"code":7003,"message":"no route"}],"messages":[]}"#;
        match unwrap_envelope(200, None, bad) {
            Err(ClefError::Http { status: 502, body, .. }) => assert!(String::from_utf8_lossy(&body).contains("no route")),
            other => panic!("{other:?}"),
        }
        match unwrap_envelope(429, Some("5".into()), br#"{"success":false,"errors":[{"code":2,"message":"slow down"}]}"#) {
            Err(ClefError::Http { status: 429, retry_after, body }) => {
                assert_eq!(retry_after.as_deref(), Some("5"));
                assert!(String::from_utf8_lossy(&body).contains("slow down"));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(unwrap_envelope(200, None, b"{\"success\":true}"), Err(ClefError::Unusable(_))));
        assert!(matches!(unwrap_envelope(200, None, b"nope"), Err(ClefError::Unusable(_))));
    }
}
