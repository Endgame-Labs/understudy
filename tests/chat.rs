//! The chat adapter against a local fake OpenAI-compatible server (no network, no paid calls): one test per question
//! type for schema output, the text fallback and error passthrough; the `Chat` lead; keys.

use serde_json::{Map, Value, json};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use understudy::chat::{self, ChatError, ChatSpec};
use understudy::lead::{Chat, Lead};
use understudy::serve::http;

#[derive(Clone, Debug)]
struct Seen {
    path: String,
    body: Value,
    auth: Option<String>,
}

type Handler = Arc<dyn Fn(&Seen) -> (u16, String) + Send + Sync>;

struct Fake {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    _l: http::Listener,
}

impl Fake {
    fn start(h: Handler) -> Fake {
        let seen = Arc::new(Mutex::new(vec![]));
        let sn = seen.clone();
        let handler: http::Handler = Arc::new(move |rq: http::Request| {
            let auth = rq.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("authorization")).map(|(_, v)| v.clone());
            let s = Seen { path: rq.target, body: serde_json::from_slice(&rq.body).unwrap_or(Value::Null), auth };
            sn.lock().unwrap().push(s.clone());
            let (status, body) = h(&s);
            (http::Response { status, headers: vec![("content-type".into(), "application/json".into())], body: body.into_bytes(), stream: None }, None)
        });
        let l = http::listen("127.0.0.1:0", 100, http::Limits::new(1 << 20), handler).unwrap();
        Fake { addr: l.addr, seen, _l: l }
    }

    fn base(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn completion(content: &str) -> String {
    json!({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 40, "completion_tokens": 6}})
    .to_string()
}

fn completion_lp(content: &str, tokens: Value) -> String {
    json!({"choices": [{"message": {"content": content}, "finish_reason": "stop", "logprobs": {"content": tokens}}],
           "usage": {"prompt_tokens": 40, "completion_tokens": 6, "prompt_tokens_details": {"cached_tokens": 8}}})
    .to_string()
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(Duration::from_secs(10)).build()
}

fn qs(name: &str, q: Value) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert(name.into(), q);
    m
}

fn noul() -> Value {
    json!({"type": "noul", "instructions": "Is this support ticket urgent?", "criteria": {"true": "cannot work", "false": "can wait"}})
}

fn choice() -> Value {
    json!({"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "charges", "bug": "broken", "howto": "how do I"}})
}

fn score() -> Value {
    json!({"type": "score", "instructions": "How polite?", "criteria": ["rude", "curt", "warm"]})
}

/// Tokens of `{"answer": <value>}` whose value token has the given alternatives.
fn answer_tokens(value_tok: &str, alts: Value) -> Value {
    json!([
        {"token": "{\"", "logprob": 0.0, "top_logprobs": []},
        {"token": "answer", "logprob": 0.0, "top_logprobs": []},
        {"token": "\":", "logprob": 0.0, "top_logprobs": []},
        {"token": " ", "logprob": 0.0, "top_logprobs": []},
        {"token": value_tok, "logprob": -0.1, "top_logprobs": alts},
        {"token": "}", "logprob": 0.0, "top_logprobs": []}
    ])
}

// ---- noul ---------------------------------------------------------------------------------------------------------

#[test]
fn noul_schema_output_with_and_without_logprobs() {
    // with logprobs: the probability comes from the answer token's alternatives
    let f = Fake::start(Arc::new(|_| (200, completion_lp("{\"answer\": true}", answer_tokens("true", json!([{"token": "true", "logprob": -0.1054}, {"token": "false", "logprob": -2.3026}]))))));
    let spec = ChatSpec::new(&f.base());
    let r = chat::ask(&agent(), &spec, None, "any-model", &json!({"subject": "locked out"}), &qs("urgent", noul())).unwrap();
    let a = &r.answers["urgent"];
    assert_eq!(a["type"], "noul");
    let p = a["noul"].as_f64().unwrap();
    assert!((0.89..0.92).contains(&p), "{a}");
    assert_eq!(r.sources["urgent"], "logprobs");
    assert_eq!(r.model, "any-model");
    assert_eq!(r.usage, json!({"input_tokens": 40, "cached_input_tokens": 8, "output_tokens": 6}));
    let s = &f.seen()[0];
    assert_eq!(s.path, "/v1/chat/completions");
    assert_eq!(s.body["response_format"]["type"], "json_schema");
    assert_eq!(s.body["logprobs"], true);
    assert_eq!(s.body["model"], "any-model");
    let user = s.body["messages"][1]["content"].as_str().unwrap();
    assert!(user.contains("Question: Is this support ticket urgent?") && user.contains("- true: cannot work") && user.contains("\"subject\": \"locked out\""), "{user}");
    // without logprobs: the documented fixed confidence
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": false}"))));
    let r = chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("hi"), &qs("urgent", noul())).unwrap();
    assert_eq!(r.answers["urgent"]["noul"], 0.1);
    assert_eq!(r.sources["urgent"], "fixed");
}

#[test]
fn noul_text_fallback_when_response_format_is_refused() {
    let f = Fake::start(Arc::new(|s| {
        if s.body.get("response_format").is_some() || s.body.get("logprobs").is_some() {
            (400, json!({"error": {"message": "response_format is not supported by this model"}}).to_string())
        } else {
            (200, completion("Sure thing!\n```json\n{\"answer\": true}\n```"))
        }
    }));
    let spec = ChatSpec::new(&f.base());
    let r = chat::ask(&agent(), &spec, None, "m", &json!("x"), &qs("q", noul())).unwrap();
    assert_eq!(r.answers["q"]["noul"], 0.9);
    assert_eq!(f.seen().len(), 2, "one refused call, one plain retry");
    // the refusal is remembered: the next call goes plain at once
    chat::ask(&agent(), &spec, None, "m", &json!("x"), &qs("q", noul())).unwrap();
    assert_eq!(f.seen().len(), 3);
}

#[test]
fn noul_error_passthrough() {
    let f = Fake::start(Arc::new(|_| (429, "{\"error\":\"slow down\"}".into())));
    match chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("q", noul())) {
        Err(ChatError::Http { status: 429, body, .. }) => assert_eq!(body, b"{\"error\":\"slow down\"}"),
        other => panic!("{other:?}"),
    }
    // a malformed answer is "unusable", not a guess
    let f = Fake::start(Arc::new(|_| (200, completion("I am not sure."))));
    assert!(matches!(chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("q", noul())), Err(ChatError::Unusable(_))));
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": \"perhaps\"}"))));
    assert!(matches!(chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("q", noul())), Err(ChatError::Unusable(_))));
}

// ---- choice -------------------------------------------------------------------------------------------------------

#[test]
fn choice_schema_output_probabilities_and_enum() {
    let f = Fake::start(Arc::new(|_| {
        (200, completion_lp("{\"answer\": \"bug\"}", answer_tokens("bug", json!([{"token": "bug", "logprob": -0.2231}, {"token": "bill", "logprob": -1.6094}, {"token": "how", "logprob": -4.6}]))))
    }));
    let r = chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("it crashes"), &qs("route", choice())).unwrap();
    let a = &r.answers["route"];
    assert_eq!(a["choice"], "bug");
    let p = &a["probabilities"];
    let sum: f64 = ["billing", "bug", "howto"].iter().map(|k| p[*k].as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9 && p["bug"].as_f64().unwrap() > p["billing"].as_f64().unwrap() && p["billing"].as_f64().unwrap() > p["howto"].as_f64().unwrap(), "{a}");
    assert_eq!(a["confidence"], p["bug"]);
    assert_eq!(f.seen()[0].body["response_format"]["json_schema"]["schema"]["properties"]["answer"]["enum"], json!(["billing", "bug", "howto"]));
    assert!(f.seen()[0].body["messages"][1]["content"].as_str().unwrap().contains("exactly one of: billing, bug, howto"));
}

#[test]
fn choice_text_fallback_and_fixed_probabilities() {
    let f = Fake::start(Arc::new(|_| (200, completion("The answer is {\"answer\": \"howto\"} I think."))));
    let r = chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("route", choice())).unwrap();
    let a = &r.answers["route"];
    assert_eq!(a["choice"], "howto");
    assert_eq!(a["probabilities"], json!({"billing": 0.05, "bug": 0.05, "howto": 0.9}));
    assert_eq!(r.sources["route"], "fixed");
}

#[test]
fn choice_error_passthrough() {
    let f = Fake::start(Arc::new(|_| (503, "upstream down".into())));
    match chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("route", choice())) {
        Err(ChatError::Http { status: 503, body, .. }) => assert_eq!(body, b"upstream down"),
        other => panic!("{other:?}"),
    }
    // an answer outside the options is unusable
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": \"refund\"}"))));
    assert!(matches!(chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("route", choice())), Err(ChatError::Unusable(_))));
}

// ---- score --------------------------------------------------------------------------------------------------------

#[test]
fn score_schema_output_and_expected_level() {
    let f = Fake::start(Arc::new(|_| {
        (200, completion_lp("{\"answer\": 2}", answer_tokens("2", json!([{"token": "2", "logprob": -0.2231}, {"token": "1", "logprob": -1.6094}]))))
    }));
    let r = chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("thanks so much"), &qs("polite", score())).unwrap();
    let a = &r.answers["polite"];
    assert_eq!(a["type"], "score");
    let (p1, p2) = (a["probabilities"]["1"].as_f64().unwrap(), a["probabilities"]["2"].as_f64().unwrap());
    assert!(p2 > 0.75 && p1 > 0.15 && a["probabilities"]["0"].as_f64().unwrap() < 0.01, "{a}");
    let score = a["score"].as_f64().unwrap();
    assert!(score > 1.7 && score < 2.0, "{a}");
    let b = &f.seen()[0].body;
    assert_eq!(b["response_format"]["json_schema"]["schema"]["properties"]["answer"], json!({"type": "integer", "enum": [0, 1, 2]}));
    assert!(b["messages"][1]["content"].as_str().unwrap().contains("- 2: warm"));
}

#[test]
fn score_text_fallback_accepts_a_numeric_string() {
    let f = Fake::start(Arc::new(|_| (200, completion("```\n{\"answer\": \"1\"}\n```"))));
    let r = chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("polite", score())).unwrap();
    assert_eq!(r.answers["polite"]["probabilities"]["1"], 0.9);
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": 7}"))));
    assert!(matches!(chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("polite", score())), Err(ChatError::Unusable(_))));
}

#[test]
fn score_error_passthrough() {
    let f = Fake::start(Arc::new(|_| (401, "{\"error\":\"bad key\"}".into())));
    match chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("polite", score())) {
        // a 401 is not a response_format refusal: relayed, and asked once
        Err(ChatError::Http { status: 401, .. }) => assert_eq!(f.seen().len(), 1),
        other => panic!("{other:?}"),
    }
}

// ---- several questions, the lead, keys ----------------------------------------------------------------------------

#[test]
fn one_call_per_question_and_the_spec_model_wins() {
    let f = Fake::start(Arc::new(|s| {
        let user = s.body["messages"][1]["content"].as_str().unwrap();
        (200, completion(if user.contains("Which queue?") { "{\"answer\": \"bug\"}" } else { "{\"answer\": true}" }))
    }));
    let mut spec = ChatSpec::new(&f.base());
    spec.model = Some("upstream-model".into());
    let mut questions = qs("route", choice());
    questions.insert("urgent".into(), noul());
    let r = chat::ask(&agent(), &spec, None, "caller-model", &json!("x"), &questions).unwrap();
    assert_eq!(f.seen().len(), 2);
    assert!(f.seen().iter().all(|s| s.body["model"] == "upstream-model"));
    assert_eq!((r.answers["route"]["choice"].as_str(), r.model.as_str()), (Some("bug"), "upstream-model"));
    assert_eq!(r.answers.len(), 2);
    assert_eq!(r.to_systemone()["understudy"]["adapter"], "chat");
}

#[test]
fn chat_lead_sends_the_key_only_as_a_bearer_header_and_never_describes_it() {
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": true}"))));
    let mut spec = ChatSpec::new(&f.base());
    spec.model = Some("m".into());
    let lead = Chat::new(spec, "m").with_key(Some("s3cr3t-key".into()));
    let a = lead.ask(Some("id1"), &json!("x"), &qs("urgent", noul())).unwrap();
    assert_eq!(a["urgent"]["type"], "noul");
    let seen = f.seen();
    assert_eq!(seen[0].auth.as_deref(), Some("Bearer s3cr3t-key"));
    assert!(!seen[0].body.to_string().contains("s3cr3t-key"));
    let d = lead.describe().to_string();
    assert!(!d.contains("s3cr3t-key") && d.contains("\"kind\":\"chat\""), "{d}");
    // no key: no header
    let lead = Chat::new(ChatSpec::new(&f.base()), "m");
    lead.ask(None, &json!("x"), &qs("urgent", noul())).unwrap();
    assert_eq!(f.seen().last().unwrap().auth, None);
}

#[test]
fn chat_lead_retries_transient_errors_and_fails_fast_on_client_errors() {
    let n = Arc::new(Mutex::new(0));
    let n2 = n.clone();
    let f = Fake::start(Arc::new(move |_| {
        let mut n = n2.lock().unwrap();
        *n += 1;
        if *n == 1 { (500, "boom".into()) } else { (200, completion("{\"answer\": false}")) }
    }));
    let mut lead = Chat::new(ChatSpec::new(&f.base()), "m");
    lead.attempts = 3;
    // the first wait is two seconds
    assert!(lead.ask(None, &json!("x"), &qs("urgent", noul())).is_ok());
    assert_eq!(f.seen().len(), 2);
    let f = Fake::start(Arc::new(|_| (404, "no such model".into())));
    let lead = Chat::new(ChatSpec::new(&f.base()), "m");
    let e = lead.ask(None, &json!("x"), &qs("urgent", noul())).unwrap_err().to_string();
    assert!(e.contains("404") && e.contains("no such model"), "{e}");
    // not a 400/422 refusal, so no plain retry either
    assert_eq!(f.seen().len(), 1);
}

// ---- review fixes -------------------------------------------------------------------------------------------------

#[test]
fn questions_run_with_bounded_concurrency() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    let (now, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let (n2, p2) = (now.clone(), peak.clone());
    let f = Fake::start(Arc::new(move |_| {
        let cur = n2.fetch_add(1, SeqCst) + 1;
        p2.fetch_max(cur, SeqCst);
        std::thread::sleep(Duration::from_millis(40));
        n2.fetch_sub(1, SeqCst);
        (200, completion("{\"answer\": true}"))
    }));
    let mut questions = Map::new();
    for i in 0..40 {
        questions.insert(format!("q{i}"), noul());
    }
    let r = chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &questions).unwrap();
    assert_eq!(r.answers.len(), 40);
    assert_eq!(f.seen().len(), 40);
    let peak = peak.load(SeqCst);
    assert!(peak >= 2 && peak <= chat::MAX_CONCURRENCY, "peak in flight {peak}");
}

#[test]
fn oversized_responses_are_refused_not_read_or_truncated() {
    let big = "x".repeat(5000);
    let b2 = big.clone();
    let f = Fake::start(Arc::new(move |_| (200, completion(&b2))));
    let mut spec = ChatSpec::new(&f.base());
    spec.max_response = Some(1000);
    assert!(matches!(chat::ask(&agent(), &spec, None, "m", &json!("x"), &qs("q", noul())), Err(ChatError::TooLarge(1000))));
    // an error body over the limit is also refused, not silently cut
    let f = Fake::start(Arc::new(move |_| (500, big.clone())));
    let mut spec = ChatSpec::new(&f.base());
    spec.max_response = Some(1000);
    assert!(matches!(chat::ask(&agent(), &spec, None, "m", &json!("x"), &qs("q", noul())), Err(ChatError::TooLarge(_))));
    assert!(!ChatError::TooLarge(1).retryable());
}

#[test]
fn a_confidence_of_one_half_never_flips_the_answer() {
    let mut spec;
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": false}"))));
    spec = ChatSpec::new(&f.base());
    spec.confidence = 0.5;
    let r = chat::ask(&agent(), &spec, None, "m", &json!("x"), &qs("q", noul())).unwrap();
    assert!(r.answers["q"]["noul"].as_f64().unwrap() < 0.5, "{}", r.answers["q"]);
    let f = Fake::start(Arc::new(|_| (200, completion("{\"answer\": \"b\"}"))));
    spec = ChatSpec::new(&f.base());
    spec.confidence = 0.5;
    let two = json!({"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}});
    let r = chat::ask(&agent(), &spec, None, "m", &json!("x"), &qs("q", two)).unwrap();
    assert_eq!(r.answers["q"]["choice"], "b");
    // the preset's value is clamped on load too
    let mut e = understudy::config::LeadEntry::default();
    e.url = "https://h/v1".into();
    e.confidence = Some(0.5);
    assert!(ChatSpec::from_entry(&e).confidence > 0.5);
}

#[test]
fn a_failed_plain_fallback_reports_its_own_error() {
    let f = Fake::start(Arc::new(|s| {
        if s.body.get("response_format").is_some() {
            (400, "{\"error\":\"no response_format\"}".into())
        } else {
            (429, "{\"error\":\"slow down\"}".into())
        }
    }));
    match chat::ask(&agent(), &ChatSpec::new(&f.base()), None, "m", &json!("x"), &qs("q", noul())) {
        Err(e @ ChatError::Http { status: 429, .. }) => assert!(e.retryable()),
        other => panic!("{other:?}"),
    }
    assert_eq!(f.seen().len(), 2);
}
