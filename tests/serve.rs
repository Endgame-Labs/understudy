//! `understudy serve` against a local fake upstream (no network, no paid calls): byte-exact pass-through, errors,
//! routing, learning off the hot path, local answers after graduation, the capture log and persistence.

use clap::Parser;
use serde_json::{Map, Value, json};
use std::io::Read;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use understudy::config::Config;
use understudy::encoder::{Encoder, Featurizer, NoEncoder};
use understudy::snapshot;
use understudy::proxy::{Proxy, Settings};
use understudy::rng::Xoshiro;
use understudy::serve::{http, ProxyFactory, Route, Server, ServeArgs, ServeConfig, ServeMode, start};
use understudy::understudent::UnderstudentKind;

const RUNE: &str = "accounts/fireworks/routers/rune";

// ---- a fake upstream --------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Seen {
    path: String,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    delay: Duration,
}

type Handler = Arc<dyn Fn(&Seen) -> Reply + Send + Sync>;

struct Fake {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    _listener: http::Listener,
}

impl Fake {
    fn start(h: Handler) -> Fake {
        let seen = Arc::new(Mutex::new(vec![]));
        let sn = seen.clone();
        let handler: http::Handler = Arc::new(move |rq: http::Request| {
            let headers = rq.headers.iter().map(|(k, v)| (k.to_ascii_lowercase(), v.clone())).collect();
            let s = Seen { path: rq.target, body: rq.body, headers };
            sn.lock().unwrap().push(s.clone());
            let r = h(&s);
            std::thread::sleep(r.delay);
            (http::Response { status: r.status, headers: r.headers, body: r.body, stream: None }, None)
        });
        let l = http::listen("127.0.0.1:0", 1000, http::Limits::new(64 << 20), handler).unwrap();
        Fake { addr: l.addr, seen, _listener: l }
    }

    fn url(&self) -> String {
        format!("http://{}/v1/systemone", self.addr)
    }

    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

fn reply(status: u16, headers: &[(&str, &str)], body: impl Into<Vec<u8>>) -> Reply {
    Reply { status, headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body: body.into(), delay: Duration::ZERO }
}

/// The oracle lead: for a choice question, the smallest criteria key when the state mentions "error", else the
/// largest (so the answer follows the label, not its position); noul: 0.95 / 0.05; any other type: an opaque
/// answer. `probs` chooses probability-bearing answers over choice-only ones.
fn oracle(probs: bool, delay: Duration) -> Handler {
    Arc::new(move |s: &Seen| {
        let v: Value = serde_json::from_slice(&s.body).unwrap();
        let state = v["state"].as_str().unwrap_or("");
        let err = state.contains("error");
        let mut answers = Map::new();
        for (name, q) in v["questions"].as_object().unwrap() {
            let a = match q["type"].as_str() {
                Some("choice") => {
                    let mut keys: Vec<&String> = q["criteria"].as_object().unwrap().keys().collect();
                    keys.sort();
                    let c = if err { keys[0] } else { keys[keys.len() - 1] };
                    if probs {
                        let p: Map<String, Value> = keys.iter().map(|k| ((*k).clone(), json!(if *k == c { 0.97 } else { 0.03 / (keys.len() - 1).max(1) as f64 }))).collect();
                        json!({"type": "choice", "choice": c, "confidence": 0.97, "probabilities": p})
                    } else {
                        json!({"type": "choice", "choice": c, "confidence": 0.9})
                    }
                }
                Some("noul") => json!({"type": "noul", "noul": if err { 0.95 } else { 0.05 }}),
                _ => json!({"type": "mystery", "value": 7}),
            };
            answers.insert(name.clone(), a);
        }
        let body = json!({"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 4}, "extra_field": {"kept": true}});
        let mut r = reply(200, &[("content-type", "application/json"), ("fireworks-server-processing-time", "12")], serde_json::to_vec(&body).unwrap());
        r.delay = delay;
        r
    })
}

// ---- a client ----------------------------------------------------------------------------------------------------

struct Out {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Out {
    fn h(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(&self.body)))
    }
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().redirects(0).timeout(Duration::from_secs(20)).build()
}

fn post_with(agent: &ureq::Agent, addr: SocketAddr, path: &str, body: &[u8], extra: &[(&str, &str)]) -> Out {
    let mut rq = agent.post(&format!("http://{addr}{path}")).set("content-type", "application/json");
    for (k, v) in extra {
        rq = rq.set(k, v);
    }
    let resp = match rq.send_bytes(body) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(e) => panic!("transport: {e}"),
    };
    let status = resp.status();
    let headers = resp.headers_names().into_iter().map(|n| (n.to_ascii_lowercase(), resp.header(&n).unwrap().to_string())).collect();
    let mut buf = vec![];
    resp.into_reader().read_to_end(&mut buf).unwrap();
    Out { status, headers, body: buf }
}

fn post(addr: SocketAddr, body: &[u8]) -> Out {
    post_with(&agent(), addr, "/v1/systemone", body, &[])
}

// ---- requests and servers ----------------------------------------------------------------------------------------

fn choice(instructions: &str, criteria: &[(&str, &str)]) -> Value {
    let c: Map<String, Value> = criteria.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
    json!({"type": "choice", "instructions": instructions, "criteria": c})
}

fn request(model: &str, state: &str, questions: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({"model": model, "state": state, "questions": questions})).unwrap()
}

const FILLER: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"];

/// Two filler words and three copies of the marker, shuffled.
fn state_text(r: &mut Xoshiro, error: bool) -> String {
    let marker = if error { "error" } else { "ok" };
    let mut w: Vec<&str> = (0..2).map(|_| FILLER[(r.next_u64() % FILLER.len() as u64) as usize]).collect();
    for _ in 0..3 {
        w.insert((r.next_u64() % (w.len() as u64 + 1)) as usize, marker);
    }
    w.join(" ")
}

/// A small keywords learner: graduates on a few dozen separable examples.
fn factory() -> ProxyFactory {
    Arc::new(|dir, lead| {
        let enc: Arc<dyn Encoder> = Arc::new(NoEncoder);
        if let Some(d) = dir.filter(|d| snapshot::exists(d)) {
            return snapshot::load(d, enc, Some(lead), |_| {});
        }
        let mut s = Settings::default();
        s.fit.understudent = UnderstudentKind::Keywords;
        s.fit.min_examples = 40;
        s.refit_every = 40;
        s.audit_rate = 0.0;
        let fz = Featurizer::for_understudent(UnderstudentKind::Keywords, None, None, None, enc.hidden())?;
        Ok(Proxy::new(s, fz, enc, Some(lead)))
    })
}

fn routes(rune: &str, jev: &str) -> Vec<Route> {
    vec![
        Route { name: "rune".into(), pattern: RUNE.into(), url: rune.into(), bearer: None, chat: None, clef: None },
        Route { name: "jev".into(), pattern: "jev-*".into(), url: jev.into(), bearer: None, chat: None, clef: None },
    ]
}

fn config(mode: ServeMode, rune: &str, jev: &str) -> ServeConfig {
    let mut c = ServeConfig::new(factory());
    c.bind = "127.0.0.1:0".into();
    c.mode = mode;
    c.routes = routes(rune, jev);
    c.audit_rate = 0.0;
    c.timeout = Duration::from_secs(10);
    c
}

fn serve(mode: ServeMode, rune: &Fake, jev: &Fake) -> Server {
    start(config(mode, &rune.url(), &jev.url())).unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("understudy-serve-{name}-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Learner states of a namespace: (spec hash, state, examples).
fn learners(srv: &Server, up: &str, model: &str) -> Vec<(String, String, usize)> {
    srv.with_proxy(up, model, |p| p.understudents.iter().map(|u| (u.key.clone(), u.state().to_string(), u.examples.len())).collect()).unwrap_or_default()
}

/// Send `n` states (35% "error") for the question(s) and wait for the learner to take them in.
fn teach(srv: &Server, model: &str, n: usize, seed: u64, questions: &dyn Fn() -> Value) {
    let ag = agent();
    let mut r = Xoshiro::new(seed);
    for i in 0..n {
        let error = r.uniform() < 0.35;
        let st = state_text(&mut r, error);
        let out = post_with(&ag, srv.addr, "/v1/systemone", &request(model, &st, questions()), &[]);
        assert_eq!(out.status, 200);
        if i % 20 == 19 {
            assert!(srv.wait_idle(Duration::from_secs(30)));
        }
    }
    assert!(srv.wait_idle(Duration::from_secs(30)));
}

fn spec() -> Value {
    json!({"decision": choice("Which way?", &[("Héal ✨", "heal up"), ("Tune up", "upgrade a card")])})
}

// ---- tests -------------------------------------------------------------------------------------------------------

#[test]
fn forward_mode_is_byte_exact_both_ways() {
    // an upstream whose response has odd formatting, unknown fields, Unicode and extra headers
    let canned = "{ \"answers\" :{\"decision\":{\"choice\":\"A\",\"confidence\":0.8,\"probabilities\":{\"A\":0.8,\"B\":0.2}}},\n \"zz_unknown\": [1,2,{\"é\":\"✨\"}], \"model\":\"jev-1.14.0\",\"usage\":{\"input_tokens\":12,\"output_tokens\":3} }\n";
    let up = Fake::start(Arc::new(move |_| reply(200, &[("content-type", "application/json; charset=utf-8"), ("fireworks-server-processing-time", "42"), ("x-request-id", "abc")], canned)));
    let srv = serve(ServeMode::Forward, &up, &up);
    // a request with odd whitespace, key order, unknown top-level fields and an unknown question type
    let req = b"{\"questions\" : {\"decision\":{\"type\":\"choice\",\"instructions\":\"Which?\",\"criteria\":{\"B\":\"b\",\"A\":\"a\"}},\n  \"extra\":{\"type\":\"embedding\",\"weird\":[1,2,3]}},\n \"future_field\":{\"x\":null},\"state\":\"s \\u00e9\",\"model\":\"accounts/fireworks/routers/rune\",\"reasoning_effort\":\"high\"}";
    for path in ["/v1/systemone", "/systemone"] {
        let out = post_with(&agent(), srv.addr, path, req, &[]);
        assert_eq!(out.status, 200);
        assert_eq!(out.body, canned.as_bytes(), "response bytes");
        assert_eq!(out.h("content-type"), Some("application/json; charset=utf-8"));
        assert_eq!(out.h("fireworks-server-processing-time"), Some("42"));
        assert_eq!(out.h("x-request-id"), Some("abc"));
        assert_eq!(out.h("x-understudy"), Some("lead"));
    }
    let seen = up.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|s| s.body == req.to_vec()), "request bytes reach the upstream unchanged");
    assert!(seen.iter().all(|s| s.path == "/v1/systemone"));
    assert!(seen.iter().all(|s| s.headers.iter().any(|(k, v)| k == "content-type" && v == "application/json")));
    drop(seen);
    assert!(srv.status()["namespaces"].as_array().unwrap().is_empty(), "forward mode learns nothing");
}

#[test]
fn every_error_passes_through_unchanged_in_every_mode() {
    let date = "Wed, 21 Oct 2037 07:28:00 GMT";
    let cases: Vec<(u16, Vec<(&str, &str)>, Vec<u8>)> = vec![
        (400, vec![("content-type", "application/json")], br#"{"error":{"type":"max_tokens_exceeded","message":"input too long: 130000 tokens"}}"#.to_vec()),
        (402, vec![("content-type", "application/json")], br#"{"error":{"type":"billing_error","message":"no available typesafe api credits"}}"#.to_vec()),
        (413, vec![("content-type", "text/plain")], b"Request Entity Too Large\n".to_vec()),
        (422, vec![("content-type", "application/json")], br#"{"detail":"context length exceeded"}"#.to_vec()),
        (408, vec![("content-type", "application/json")], br#"{"error":"timeout"}"#.to_vec()),
        (429, vec![("content-type", "application/json"), ("retry-after", "7")], br#"{"error":"rate limited"}"#.to_vec()),
        (429, vec![("content-type", "application/json"), ("retry-after", date)], br#"{"error":"rate limited"}"#.to_vec()),
        (503, vec![("content-type", "application/json"), ("retry-after", "1.5")], br#"{"error":"overloaded"}"#.to_vec()),
        (503, vec![("content-type", "application/json"), ("retry-after", date)], br#"{"error":"overloaded"}"#.to_vec()),
        (500, vec![("content-type", "text/html")], b"<html>boom \xff\xfe</html>".to_vec()),
        (502, vec![], b"bad gateway".to_vec()),
        (504, vec![("content-type", "text/plain")], b"upstream timed out".to_vec()),
        // a 200 that is not valid JSON, and one that is JSON without answers
        (200, vec![("content-type", "application/json")], b"{\"answers\": {\"decision\": ".to_vec()),
        (200, vec![("content-type", "application/json")], br#"{"nothing":"here"}"#.to_vec()),
        (200, vec![("content-type", "application/json")], br#"{"answers":{"decision":{"choice":"not an option"}}}"#.to_vec()),
    ];
    for mode in [ServeMode::Forward, ServeMode::Shadow, ServeMode::Serve, ServeMode::Race] {
        for (status, headers, body) in &cases {
            let st = *status;
            let (h2, b2) = (headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<Vec<_>>(), body.clone());
            let up = Fake::start(Arc::new(move |_| Reply { status: st, headers: h2.clone(), body: b2.clone(), delay: Duration::ZERO }));
            let srv = serve(mode, &up, &up);
            let out = post(srv.addr, &request(RUNE, "some state", spec()));
            assert_eq!(out.status, *status, "{mode:?}");
            assert_eq!(&out.body, body, "{mode:?} {status}");
            for (k, v) in headers {
                assert_eq!(out.h(k), Some(*v), "{mode:?} {status} {k}");
            }
            assert_eq!(out.h("x-understudy"), Some("lead"));
            assert_eq!(up.calls(), 1, "the proxy never retries");
            assert!(srv.wait_idle(Duration::from_secs(5)));
            // nothing was learned from a failed or unusable answer
            for (_, _, n) in learners(&srv, "rune", RUNE) {
                assert_eq!(n, 0, "{mode:?} {status}");
            }
        }
    }
}

#[test]
fn transport_failures_are_502_or_504_and_marked() {
    let dead = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    };
    let srv = start(config(ServeMode::Shadow, &format!("http://{dead}/v1/systemone"), &format!("http://{dead}/v1/systemone"))).unwrap();
    let out = post(srv.addr, &request(RUNE, "s", spec()));
    assert_eq!(out.status, 502);
    assert_eq!(out.h("x-understudy"), Some("error"));
    assert_eq!(out.json()["error"]["type"], "upstream_unreachable");
    // a timeout is a 504
    let slow = Fake::start(oracle(true, Duration::from_millis(1500)));
    let mut c = config(ServeMode::Shadow, &slow.url(), &slow.url());
    c.timeout = Duration::from_millis(300);
    let srv = start(c).unwrap();
    let out = post(srv.addr, &request(RUNE, "s", spec()));
    assert_eq!(out.status, 504);
    assert_eq!(out.h("x-understudy"), Some("error"));
}

#[test]
fn bad_client_requests_and_the_utility_endpoints() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve(ServeMode::Shadow, &up, &up);
    assert_eq!(post(srv.addr, b"not json").status, 400);
    assert_eq!(post(srv.addr, br#"{"state":"x","questions":{}}"#).status, 400);
    let out = post(srv.addr, &request("gpt-nope", "s", spec()));
    assert_eq!((out.status, out.json()["error"]["type"].clone()), (400, json!("unknown_model")));
    assert_eq!(up.calls(), 0);
    let ag = agent();
    let r = ag.get(&format!("http://{}/healthz", srv.addr)).call().unwrap();
    assert_eq!(r.status(), 200);
    let r: Value = ag.get(&format!("http://{}/status", srv.addr)).call().unwrap().into_json().unwrap();
    assert_eq!(r["mode"], "shadow");
    assert!(matches!(ag.get(&format!("http://{}/nope", srv.addr)).call(), Err(ureq::Error::Status(404, _))));
}

#[test]
fn routing_by_model_and_credentials() {
    let (rune, jev) = (Fake::start(oracle(true, Duration::ZERO)), Fake::start(oracle(true, Duration::ZERO)));
    let mut c = config(ServeMode::Forward, &rune.url(), &jev.url());
    c.routes[1].bearer = Some("k-jev".into());
    let srv = start(c).unwrap();
    post(srv.addr, &request(RUNE, "s", spec()));
    post(srv.addr, &request("jev-latest", "s", spec()));
    post(srv.addr, &request("jev-preview", "s", spec()));
    assert_eq!((rune.calls(), jev.calls()), (1, 2));
    // the configured key goes upstream; a client's own Authorization is forwarded when no key is configured
    let auth = |f: &Fake, i: usize| f.seen.lock().unwrap()[i].headers.iter().find(|(k, _)| k == "authorization").map(|(_, v)| v.clone());
    assert_eq!(auth(&jev, 0).as_deref(), Some("Bearer k-jev"));
    assert_eq!(auth(&rune, 0), None);
    post_with(&agent(), srv.addr, "/v1/systemone", &request(RUNE, "s", spec()), &[("authorization", "Bearer client-key")]);
    assert_eq!(auth(&rune, 1).as_deref(), Some("Bearer client-key"));
}

#[test]
fn flags_build_routes_and_read_keys_from_the_environment_only() {
    #[derive(Parser)]
    struct W {
        #[command(flatten)]
        a: ServeArgs,
    }
    unsafe { std::env::set_var("UNDERSTUDY_TEST_KEY", " sekrit ") };
    let lead_cfg = Config::parse(
        r#"
[leads.rune]
url = "http://r.example/v1/systemone"
model = "accounts/fireworks/routers/rune"

[leads.jev]
url = "http://j.example/v1/systemone"
model = "jev-1.14.0"
route = "jev-*"
key_env = "UNDERSTUDY_TEST_KEY"

[leads.glm]
kind = "chat"
url = "http://c.example/v1"
model = "small-model"
route = "glm-*"
"#,
    )
    .unwrap();
    let w = W::parse_from(["x", "--route", "my-model-*=http://localhost:9/v1/systemone", "--route-key-env", "my-model-*=UNDERSTUDY_TEST_UNSET", "--chat-route", "chatty=http://localhost:9/v1"]);
    let c = w.a.config_with(&lead_cfg, 0.25, factory()).unwrap();
    assert_eq!(c.mode, ServeMode::Serve);
    assert_eq!(c.bind, "127.0.0.1:8787");
    let names: Vec<_> = c.routes.iter().map(|r| (r.pattern.as_str(), r.url.as_str(), r.bearer.as_deref(), r.chat.is_some())).collect();
    assert_eq!(names, vec![
        ("my-model-*", "http://localhost:9/v1/systemone", None, false),
        ("chatty", "http://localhost:9/v1/chat/completions", None, true),
        ("glm-*", "http://c.example/v1/chat/completions", None, true),
        ("jev-*", "http://j.example/v1/systemone", Some("sekrit"), false),
        (RUNE, "http://r.example/v1/systemone", None, false),
    ]);
    assert!(c.routes[0].matches("my-model-7") && !c.routes[0].matches("my-model") && c.routes[4].matches(RUNE));
    // with no config and no --route there is nothing to serve to: an explicit error, not a built-in default
    let w = W::parse_from(["x"]);
    let e = w.a.config_with(&Config::default(), 0.1, factory()).err().expect("no routes").to_string();
    assert!(e.contains("--route") && e.contains("leads.toml"), "{e}");
    let w = W::parse_from(["x", "--no-default-routes"]);
    assert!(w.a.config_with(&lead_cfg, 0.1, factory()).is_err());
}

#[test]
fn shadow_learns_per_upstream_and_model_and_keeps_request_names_apart() {
    let (rune, jev) = (Fake::start(oracle(true, Duration::ZERO)), Fake::start(oracle(true, Duration::ZERO)));
    let srv = serve(ServeMode::Shadow, &rune, &jev);
    // Rust's `decision`, Go's `action`, and both at once under identical specs
    let q = || json!({"decision": choice("Which way?", &[("A", "heal"), ("B", "tune")])});
    let go = || json!({"action": choice("Which way?", &[("A", "heal"), ("B", "tune")])});
    let both = || json!({"decision": choice("Which way?", &[("A", "heal"), ("B", "tune")]), "action": choice("Which way?", &[("A", "heal"), ("B", "tune")])});
    let ag = agent();
    for (i, mk) in [&q as &dyn Fn() -> Value, &go, &both].iter().enumerate() {
        let out = post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, "an error happened", mk()), &[]);
        let names: Vec<_> = out.json()["answers"].as_object().unwrap().keys().cloned().collect();
        assert_eq!(names.len(), if i == 2 { 2 } else { 1 });
    }
    post_with(&ag, srv.addr, "/v1/systemone", &request("jev-latest", "an error happened", q()), &[]);
    post_with(&ag, srv.addr, "/v1/systemone", &request("jev-preview", "all ok here", q()), &[]);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    // one learner per spec within a namespace, however it was named; the same spec under another model is separate
    let r = learners(&srv, "rune", RUNE);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].2, 1, "one distinct state, however many names");
    assert_eq!(learners(&srv, "jev", "jev-latest").len(), 1);
    assert_eq!(learners(&srv, "jev", "jev-preview").len(), 1);
    assert_eq!(rune.calls(), 3);
    assert_eq!(jev.calls(), 2);
    let st = srv.status();
    assert_eq!(st["namespaces"].as_array().unwrap().len(), 3);
    assert_eq!(st["learned"], 5);
}

/// Teach a question to graduation, then check what serve mode answers locally.
fn local_after_graduation(probs: bool, names: &[&str], model: &str, criteria: &[(&str, &str)]) {
    let up = Fake::start(oracle(probs, Duration::ZERO));
    let srv = serve(ServeMode::Serve, &up, &up);
    let questions = || Value::Object(names.iter().map(|n| (n.to_string(), choice("Which way?", criteria))).collect());
    let up_name = if model.starts_with("jev") { "jev" } else { "rune" };
    // before graduation everything is forwarded
    let first = post(srv.addr, &request(model, "an error happened", questions()));
    assert_eq!(first.h("x-understudy"), Some("lead"));
    teach(&srv, model, 220, 3, &questions);
    let l = learners(&srv, up_name, model);
    assert_eq!(l.len(), 1, "identical specs under different names share one learner");
    assert_eq!(l[0].1, "graduated", "{l:?}");
    // local answers: no upstream call, the exact envelope, the right class
    let calls = up.calls();
    let mut r = Xoshiro::new(99);
    let (mut local, mut wrong, total) = (0, 0, 100);
    let mut keys: Vec<&str> = criteria.iter().map(|c| c.0).collect();
    keys.sort();
    for _ in 0..total {
        let error = r.uniform() < 0.35;
        let st = state_text(&mut r, error);
        let out = post(srv.addr, &request(model, &st, questions()));
        assert_eq!(out.status, 200);
        if out.h("x-understudy") != Some("local") {
            continue;
        }
        local += 1;
        let v = out.json();
        assert_eq!(v["model"], model);
        assert_eq!(v["usage"], json!({"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}));
        let answers = v["answers"].as_object().unwrap();
        let mut got: Vec<&str> = answers.keys().map(String::as_str).collect();
        got.sort();
        let mut want = names.to_vec();
        want.sort();
        assert_eq!(got, want, "answers keys are exactly the submitted names");
        for a in answers.values() {
            assert_eq!(a["type"], "choice");
            let c = a["choice"].as_str().unwrap();
            assert!(keys.contains(&c), "choice {c:?} is one of the criteria keys");
            let want = if error { keys[0] } else { keys[keys.len() - 1] };
            wrong += usize::from(c != want);
            let p = a["probabilities"].as_object().unwrap();
            assert_eq!(p.len(), keys.len());
            assert!(p.keys().all(|k| keys.contains(&k.as_str())));
            let units: i64 = p.values().map(|x| (x.as_f64().unwrap() * 1e4).round() as i64).sum();
            assert_eq!(units, 10_000, "the printed probabilities sum to exactly 1: {p:?}");
            assert_eq!(a["confidence"].as_f64().unwrap(), p[c].as_f64().unwrap());
        }
    }
    assert!(local >= 80, "graduated and eligible requests are served locally: {local}/{total}");
    assert_eq!(wrong, 0, "the local answers follow the lead");
    assert_eq!(up.calls(), calls + (total - local), "local answers cost no upstream call");
}

#[test]
fn serve_answers_locally_only_after_graduation() {
    local_after_graduation(true, &["decision"], RUNE, &[("A", "heal"), ("B", "tune")]);
}

#[test]
fn serve_handles_go_names_unicode_labels_and_aliases() {
    local_after_graduation(true, &["action"], "jev-latest", &[("Héal ✨ (+20 HP)", "heal up"), ("Tune up — 调整", "upgrade a card"), ("Leave", "walk away")]);
    local_after_graduation(true, &["decision", "action"], RUNE, &[("A", "heal"), ("B", "tune")]);
}

#[test]
fn choice_only_upstream_answers_teach_just_as_well() {
    local_after_graduation(false, &["decision"], RUNE, &[("A", "heal"), ("B", "tune")]);
}

#[test]
fn serve_forwards_whole_requests_that_are_not_entirely_local() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve(ServeMode::Serve, &up, &up);
    let known = || spec();
    teach(&srv, RUNE, 200, 5, &known);
    assert_eq!(learners(&srv, "rune", RUNE)[0].1, "graduated");
    let st = "alpha error beta error error";
    let local = |req: Vec<u8>| post(srv.addr, &req);
    assert_eq!(local(request(RUNE, st, known())).h("x-understudy"), Some("local"));
    // the criteria in another order: the same question
    let reordered = json!({"decision": choice("Which way?", &[("Tune up", "upgrade a card"), ("Héal ✨", "heal up")])});
    let out = local(request(RUNE, st, reordered));
    assert_eq!(out.h("x-understudy"), Some("local"));
    assert_eq!(out.json()["answers"]["decision"]["choice"], "Héal ✨");
    // another criteria set, another instruction, another model: different questions, never answered locally
    let extra = json!({"decision": choice("Which way?", &[("Héal ✨", "heal up"), ("Tune up", "upgrade a card"), ("Leave", "go")])});
    assert_eq!(local(request(RUNE, st, extra)).h("x-understudy"), Some("lead"));
    let other = json!({"decision": choice("Which way, really?", &[("Héal ✨", "heal up"), ("Tune up", "upgrade a card")])});
    assert_eq!(local(request(RUNE, st, other)).h("x-understudy"), Some("lead"));
    assert_eq!(local(request("jev-latest", st, known())).h("x-understudy"), Some("lead"));
    // one graduated question next to an unknown-type one: the whole request goes to the lead, opaque part and all
    let mixed = json!({"decision": choice("Which way?", &[("Héal ✨", "heal up"), ("Tune up", "upgrade a card")]), "other": {"type": "embedding", "instructions": "?", "criteria": {"x": "y"}}});
    let body = request(RUNE, st, mixed);
    let calls = up.calls();
    let out = local(body.clone());
    assert_eq!(out.h("x-understudy"), Some("lead"));
    assert_eq!(out.json()["answers"]["other"]["type"], "mystery", "the lead's answer to the unknown type is returned as it came");
    assert_eq!(up.calls(), calls + 1);
    assert_eq!(up.seen.lock().unwrap().last().unwrap().body, body);
    // an unsupported type alone creates no learner
    let only = request("jev-latest", st, json!({"o": {"type": "embedding", "instructions": "?", "criteria": {"x": "y"}}}));
    assert_eq!(local(only).status, 200);
    assert!(srv.wait_idle(Duration::from_secs(5)));
    assert_eq!(learners(&srv, "jev", "jev-latest").len(), 1, "only the supported question of the earlier request");
}

#[test]
fn requests_during_a_refit_neither_wait_nor_fail() {
    let up = Fake::start(oracle(true, Duration::from_millis(20)));
    let srv = serve(ServeMode::Serve, &up, &up);
    teach(&srv, RUNE, 200, 8, &spec);
    assert_eq!(learners(&srv, "rune", RUNE)[0].1, "graduated");
    // hold the namespace the way a long refit does: serve forwards, shadow-style learning jobs queue (and drop)
    let t = Instant::now();
    srv.with_proxy("rune", RUNE, |_| {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..24)
                .map(|i| {
                    let addr = srv.addr;
                    s.spawn(move || {
                        let mut r = Xoshiro::new(500 + i);
                        let error = r.uniform() < 0.5;
                        let st = state_text(&mut r, error);
                        post(addr, &request(RUNE, &st, spec()))
                    })
                })
                .collect();
            for h in hs {
                let out = h.join().unwrap();
                assert_eq!(out.status, 200);
                assert_eq!(out.h("x-understudy"), Some("lead"), "a busy namespace forwards");
            }
        });
    })
    .unwrap();
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    assert!(srv.status()["local_busy"].as_u64().unwrap() >= 24);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    // and afterwards it serves locally again
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!(out.h("x-understudy"), Some("local"));
}

#[test]
fn a_full_learning_queue_drops_and_counts_but_never_blocks_a_response() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
    c.queue = 2;
    let srv = start(c).unwrap();
    post(srv.addr, &request(RUNE, "seed error", spec()));
    assert!(srv.wait_idle(Duration::from_secs(5)));
    let t = Instant::now();
    srv.with_proxy("rune", RUNE, |_| {
        for i in 0..20 {
            let out = post(srv.addr, &request(RUNE, &format!("state {i} error"), spec()));
            assert_eq!(out.status, 200);
        }
    })
    .unwrap();
    assert!(t.elapsed() < Duration::from_secs(5));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let st = srv.status();
    assert!(st["learn_dropped"].as_u64().unwrap() >= 10, "{st}");
}

#[test]
fn concurrent_requests_do_not_serialize_behind_upstream_calls() {
    let up = Fake::start(oracle(true, Duration::from_millis(250)));
    for mode in [ServeMode::Forward, ServeMode::Shadow, ServeMode::Serve] {
        let srv = serve(mode, &up, &up);
        let t = Instant::now();
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..24).map(|i| s.spawn(move || post(srv.addr, &request(RUNE, &format!("state {i} error"), spec())))).collect();
            for h in hs {
                let o = h.join().unwrap();
                assert_eq!(o.status, 200, "{mode:?}: {}", String::from_utf8_lossy(&o.body));
            }
        });
        // serialized it would take 24 x 250 ms = 6 s
        assert!(t.elapsed() < Duration::from_millis(2500), "{mode:?}: {:?}", t.elapsed());
    }
}

#[test]
fn the_capture_log_records_each_request_and_optionally_the_bodies() {
    let up = Fake::start(oracle(true, Duration::from_millis(5)));
    for bodies in [false, true] {
        let dir = tmp("capture");
        let log = dir.join("capture.jsonl");
        let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
        c.capture = Some(log.clone());
        c.capture_bodies = bodies;
        let srv = start(c).unwrap();
        let qs = json!({"decision": choice("Which way?", &[("A", "heal"), ("B", "tune")]), "action": choice("Which way?", &[("A", "heal"), ("B", "tune")]), "w": {"type": "embedding"}});
        let req = request(RUNE, "an error", qs);
        let ok = post(srv.addr, &req);
        let fail = post(srv.addr, &request("jev-latest", "x", spec()));
        assert_eq!((ok.status, fail.status), (200, 200));
        assert!(srv.wait_idle(Duration::from_secs(5)));
        srv.stop();
        let lines: Vec<Value> = std::fs::read_to_string(&log).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        let e = lines.iter().find(|l| l["upstream"] == "rune").unwrap();
        for k in ["ts", "model", "upstream", "mode", "routed", "status", "latency_ms", "upstream_ms", "usage", "state_sha", "questions"] {
            assert!(!e[k].is_null(), "{k} in {e}");
        }
        assert_eq!((e["model"].as_str(), e["mode"].as_str(), e["routed"].as_str(), e["status"].as_u64()), (Some(RUNE), Some("shadow"), Some("lead"), Some(200)));
        assert!(e["latency_ms"].as_f64().unwrap() >= e["upstream_ms"].as_f64().unwrap() && e["upstream_ms"].as_f64().unwrap() >= 5.0);
        assert_eq!(e["usage"]["input_tokens"], 100);
        let qs = e["questions"].as_array().unwrap();
        let names: Vec<_> = qs.iter().map(|q| q["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["decision", "action", "w"]);
        assert_eq!(qs[0]["spec_hash"], qs[1]["spec_hash"], "identical specs hash alike under different names");
        assert_eq!(qs[0]["spec_hash"].as_str().unwrap().len(), 16);
        assert_eq!((qs[0]["opaque"].clone(), qs[2]["opaque"].clone()), (json!(false), json!(true)));
        assert_eq!(qs[0]["understudent"]["eligible"], false);
        assert_eq!(qs[0]["lead"], "A");
        assert_eq!(e.get("request").is_some(), bodies);
        assert_eq!(e.get("response").is_some(), bodies);
        if bodies {
            assert_eq!(e["request"]["state"], "an error");
            assert_eq!(e["response"]["answers"]["decision"]["choice"], "A");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn would_be_answers_and_agreement_are_logged_once_the_understudent_has_graduated() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let dir = tmp("shadowlog");
    let log = dir.join("c.jsonl");
    let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
    c.capture = Some(log.clone());
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 200, 11, &spec);
    for i in 0..20 {
        let mut r = Xoshiro::new(900 + i);
        let error = r.uniform() < 0.4;
        let st = state_text(&mut r, error);
        post(srv.addr, &request(RUNE, &st, spec()));
    }
    assert!(srv.wait_idle(Duration::from_secs(10)));
    srv.stop();
    let lines: Vec<Value> = std::fs::read_to_string(&log).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let tail: Vec<&Value> = lines.iter().rev().take(20).collect();
    let eligible = tail.iter().filter(|l| l["questions"][0]["understudent"]["would_answer_local"] == true).count();
    assert!(eligible >= 15, "{eligible}");
    assert!(tail.iter().all(|l| l["questions"][0]["agree"] != json!(false)), "the understudent agrees with the lead");
    assert!(tail.iter().all(|l| l["questions"][0]["understudent"]["probs"].is_object()));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn learners_survive_a_restart_through_snapshot_dir() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let dir = tmp("probes");
    let mut c = config(ServeMode::Serve, &up.url(), &up.url());
    c.snapshot_dir = Some(dir.join("p"));
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 200, 4, &spec);
    assert_eq!(learners(&srv, "rune", RUNE)[0].1, "graduated");
    srv.stop(); // saves
    let mut c = config(ServeMode::Serve, &up.url(), &up.url());
    c.snapshot_dir = Some(dir.join("p"));
    let srv = start(c).unwrap();
    // the namespace is built by the first request that names it; the restored learner serves at once
    let first = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let l = learners(&srv, "rune", RUNE);
    assert_eq!(l.len(), 1);
    assert!(l[0].2 >= 60, "{l:?}");
    let _ = first;
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!(out.h("x-understudy"), Some("local"));
    assert_eq!(out.json()["answers"]["decision"]["choice"], "Héal ✨");
    drop(srv);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn audits_of_local_answers_reach_the_lead_in_the_background() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let mut c = config(ServeMode::Serve, &up.url(), &up.url());
    c.audit_rate = 1.0;
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 200, 6, &spec);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let (before, audits) = (up.calls(), srv.status()["audits"].as_u64().unwrap());
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!(out.h("x-understudy"), Some("local"));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert_eq!(up.calls(), before + 1, "the audit call");
    assert_eq!(srv.status()["audits"].as_u64().unwrap(), audits + 1);
}

/// Shadow overhead against a direct call to a fake upstream with a fixed 100 ms latency. `UNDERSTUDY_OVERHEAD_N`
/// sets the number of calls per side (default 30; the reported numbers use 1000, run with `--nocapture`).
#[test]
fn shadow_adds_under_a_couple_of_milliseconds() {
    let n: usize = std::env::var("UNDERSTUDY_OVERHEAD_N").ok().and_then(|x| x.parse().ok()).unwrap_or(30);
    let up = Fake::start(oracle(true, Duration::from_millis(100)));
    let srv = serve(ServeMode::Shadow, &up, &up);
    // a typical caller's request: a few KB of state, a handful of options
    let mut r = Xoshiro::new(1);
    let state: String = (0..600).map(|_| FILLER[(r.next_u64() % 12) as usize]).collect::<Vec<_>>().join(" ");
    let crit: Vec<(String, String)> = (0..8).map(|i| (format!("Play card {i} on enemy"), format!("deals {} damage; you would lose about {} HP", i * 3, 10 - i))).collect();
    let crit_ref: Vec<(&str, &str)> = crit.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let qs = json!({"decision": choice("Combat: you have 3 of 3 Energy left.", &crit_ref)});
    let (direct_ag, proxy_ag) = (agent(), agent());
    let (mut d, mut p, mut fails) = (vec![], vec![], 0);
    for i in 0..n + 5 {
        // a new state each time, so the proxy has real work to queue
        let body = request(RUNE, &format!("{state} step {i}"), qs.clone());
        let t = Instant::now();
        let a = post_with(&direct_ag, up.addr, "/v1/systemone", &body, &[]);
        let td = t.elapsed();
        let t = Instant::now();
        let b = post_with(&proxy_ag, srv.addr, "/v1/systemone", &body, &[]);
        let tp = t.elapsed();
        fails += usize::from(a.status != 200 || b.status != 200 || a.body != b.body);
        if i >= 5 {
            d.push(td.as_secs_f64() * 1e3);
            p.push(tp.as_secs_f64() * 1e3);
        }
    }
    assert_eq!(fails, 0, "no added failure modes: same status and bytes on every call");
    let q = |v: &[f64], f: f64| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        s[((s.len() - 1) as f64 * f).round() as usize]
    };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!("shadow overhead over {n} calls with a 100 ms upstream (ms): direct p50 {:.3} p95 {:.3} p99 {:.3} | via serve p50 {:.3} p95 {:.3} p99 {:.3} | added p50 {:.3} p95 {:.3} p99 {:.3} mean {:.3}",
        q(&d, 0.5), q(&d, 0.95), q(&d, 0.99), q(&p, 0.5), q(&p, 0.95), q(&p, 0.99), q(&p, 0.5) - q(&d, 0.5), q(&p, 0.95) - q(&d, 0.95), q(&p, 0.99) - q(&d, 0.99), mean(&p) - mean(&d));
    assert!(q(&p, 0.5) - q(&d, 0.5) < 10.0, "added p50 {:.3} ms", q(&p, 0.5) - q(&d, 0.5));
}

// ---- race ---------------------------------------------------------------------------------------------------------

/// The oracle with a delay (ms) that a test can change while the server runs.
fn slowable(delay_ms: &Arc<AtomicU64>) -> Handler {
    let inner = oracle(true, Duration::ZERO);
    let d = delay_ms.clone();
    Arc::new(move |s: &Seen| {
        let mut r = inner(s);
        r.delay = Duration::from_millis(d.load(Ordering::Relaxed));
        r
    })
}

fn capture_lines(log: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(log).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// A race server whose learner has graduated on `spec()`, with the fake's delay at `delay_ms` afterwards.
fn graduated_race(delay_ms: &Arc<AtomicU64>, cap: usize) -> (Fake, Server, PathBuf) {
    let up = Fake::start(slowable(delay_ms));
    let dir = tmp("race");
    let log = dir.join("capture.jsonl");
    let mut c = config(ServeMode::Race, &up.url(), &up.url());
    c.capture = Some(log.clone());
    c.race_max_inflight = cap;
    let srv = start(c).unwrap();
    delay_ms.store(0, Ordering::Relaxed);
    teach(&srv, RUNE, 220, 3, &spec);
    assert_eq!(learners(&srv, "rune", RUNE)[0].1, "graduated");
    (up, srv, log)
}

#[test]
fn race_forwarded_calls_add_no_latency_to_the_lead() {
    let delay = Arc::new(AtomicU64::new(100));
    let up = Fake::start(slowable(&delay));
    let srv = serve(ServeMode::Race, &up, &up);
    let (direct, via) = (agent(), agent());
    let (mut d, mut p) = (vec![], vec![]);
    for i in 0..25 {
        // never graduated (one fresh state each time), so every call is forwarded
        let body = request(RUNE, &format!("alpha beta gamma step {i}"), spec());
        let t = Instant::now();
        let a = post_with(&direct, up.addr, "/v1/systemone", &body, &[]);
        let td = t.elapsed();
        let t = Instant::now();
        let b = post_with(&via, srv.addr, "/v1/systemone", &body, &[]);
        let tp = t.elapsed();
        assert_eq!((a.status, &a.body), (b.status, &b.body), "the lead's bytes, unchanged");
        assert_eq!(b.h("x-understudy"), Some("lead"));
        if i >= 5 {
            d.push(td.as_secs_f64() * 1e3);
            p.push(tp.as_secs_f64() * 1e3);
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let (md, mp) = (med(&mut d), med(&mut p));
    assert!(mp - md < 2.0, "race adds {:.3} ms at p50 (direct {md:.3}, via {mp:.3})", mp - md);
}

#[test]
fn race_returns_a_local_answer_before_a_slow_lead_and_records_the_lead_as_an_audit() {
    let delay = Arc::new(AtomicU64::new(0));
    let (up, srv, log) = graduated_race(&delay, 64);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    delay.store(600, Ordering::Relaxed);
    let before = up.calls();
    let t = Instant::now();
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    let took = t.elapsed();
    assert_eq!(out.h("x-understudy"), Some("local"));
    assert_eq!(out.json()["answers"]["decision"]["choice"], "Héal ✨");
    assert!(took < Duration::from_millis(300), "the local answer does not wait for the lead: {took:?}");
    // the lead was called at once, and its answer is still audited afterwards
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert_eq!(up.calls(), before + 1);
    assert!(srv.status()["race_local"].as_u64().unwrap() >= 1);
    srv.stop();
    let lines = capture_lines(&log);
    let req = lines.iter().rev().find(|l| l["kind"] == "request" && l["routed"] == "local").expect("the local request line");
    let aud = lines.iter().rev().find(|l| l["kind"] == "audit" && l["race_id"] == req["race_id"]).expect("its audit line");
    assert_eq!((req["mode"].as_str(), req["status"].as_u64()), (Some("race"), Some(200)));
    assert!(req["local_ms"].as_f64().is_some());
    assert!(req["latency_ms"].as_f64().unwrap() < 300.0);
    assert!(aud["lead_ms"].as_f64().unwrap() >= 600.0, "{aud}");
    assert_eq!(aud["routed"], "local");
    assert_eq!(aud["questions"][0]["agree"], true, "{aud}");
    assert_eq!(aud["questions"][0]["lead"], "Héal ✨");
    assert_eq!(aud["questions"][0]["served"]["label"], "Héal ✨");
}

#[test]
fn race_forwards_what_is_not_local_and_logs_local_and_lead_times() {
    let delay = Arc::new(AtomicU64::new(0));
    let (up, srv, log) = graduated_race(&delay, 64);
    delay.store(20, Ordering::Relaxed);
    let before = up.calls();
    // a question the learner has never seen: forwarded
    let other = json!({"decision": choice("Another question?", &[("X", "x"), ("Y", "y")])});
    let out = post(srv.addr, &request(RUNE, "an error", other));
    assert_eq!(out.h("x-understudy"), Some("lead"));
    assert_eq!(up.calls(), before + 1);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    srv.stop();
    let lines = capture_lines(&log);
    let e = lines.iter().rev().find(|l| l["kind"] == "request" && l["routed"] == "lead").unwrap();
    assert_eq!(e["mode"], "race");
    assert!(e["lead_ms"].as_f64().unwrap() >= 20.0);
    assert!(e.get("local_ms").is_some());
}

#[test]
fn race_passes_errors_through_and_only_logs_a_lead_error_after_a_local_answer() {
    // errors pass through unchanged when nothing local is returned
    let up = Fake::start(Arc::new(|_| reply(429, &[("content-type", "application/json"), ("retry-after", "7")], br#"{"error":"rate limited"}"#.to_vec())));
    let srv = serve(ServeMode::Race, &up, &up);
    let out = post(srv.addr, &request(RUNE, "some state", spec()));
    assert_eq!((out.status, out.h("retry-after"), out.h("x-understudy")), (429, Some("7"), Some("lead")));
    assert_eq!(out.body, br#"{"error":"rate limited"}"#);
    drop(srv);
    // a transport failure is a 502, as in shadow
    let dead = "127.0.0.1:1";
    let srv = start(config(ServeMode::Race, &format!("http://{dead}/v1/systemone"), &format!("http://{dead}/v1/systemone"))).unwrap();
    let out = post(srv.addr, &request(RUNE, "some state", spec()));
    assert_eq!((out.status, out.h("x-understudy")), (502, Some("error")));
    drop(srv);
    // the lead fails after the local answer went out: the client still has the local answer, the log has the error
    let fail = Arc::new(AtomicU64::new(0));
    let (f2, inner) = (fail.clone(), oracle(true, Duration::ZERO));
    let up = Fake::start(Arc::new(move |s: &Seen| {
        if f2.load(Ordering::Relaxed) == 1 {
            let mut r = reply(500, &[("content-type", "text/plain")], b"boom".to_vec());
            r.delay = Duration::from_millis(100);
            r
        } else {
            inner(s)
        }
    }));
    let dir = tmp("race-err");
    let log = dir.join("capture.jsonl");
    let mut c = config(ServeMode::Race, &up.url(), &up.url());
    c.capture = Some(log.clone());
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 220, 3, &spec);
    fail.store(1, Ordering::Relaxed);
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!((out.status, out.h("x-understudy")), (200, Some("local")));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    srv.stop();
    let lines = capture_lines(&log);
    let aud = lines.iter().rev().find(|l| l["kind"] == "audit").expect("the failed audit is logged");
    assert_eq!(aud["status"], 500);
}

#[test]
fn race_caps_background_lead_calls_and_then_behaves_like_serve() {
    let delay = Arc::new(AtomicU64::new(0));
    let (up, srv, _log) = graduated_race(&delay, 1);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    delay.store(700, Ordering::Relaxed);
    let before = up.calls();
    let capped0 = srv.status()["race_capped"].as_u64().unwrap();
    let first = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!(first.h("x-understudy"), Some("local"));
    // the first lead call is still in flight: the cap is reached, so this request is handled as in serve mode
    // (answered locally, no lead call of its own)
    let second = post(srv.addr, &request(RUNE, "gamma error error eta error", spec()));
    assert_eq!(second.h("x-understudy"), Some("local"));
    assert_eq!(srv.status()["race_capped"].as_u64().unwrap(), capped0 + 1);
    assert_eq!(srv.status()["race_inflight"], 1);
    // and an ungraduated question past the cap is simply forwarded
    let other = json!({"decision": choice("Another question?", &[("X", "x"), ("Y", "y")])});
    delay.store(0, Ordering::Relaxed);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert_eq!(up.calls(), before + 1, "only the first request called the lead");
    assert_eq!(srv.status()["race_inflight"], 0);
    let out = post(srv.addr, &request(RUNE, "an error", other));
    assert_eq!(out.h("x-understudy"), Some("lead"));
}


fn raw_exchange(addr: SocketAddr, wire: &[u8], reads: usize) -> String {
    use std::io::Write;
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.write_all(wire).unwrap();
    let mut out = vec![];
    let mut buf = [0u8; 8192];
    for _ in 0..reads {
        let n = s.read(&mut buf).unwrap();
        out.extend_from_slice(&buf[..n]);
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[test]
fn the_http_layer_reads_chunked_bodies_honours_expect_and_reuses_connections() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve(ServeMode::Forward, &up, &up);
    let body = request(RUNE, "an error", spec());
    // a chunked request body is passed upstream whole (with a length)
    let (a, b) = body.split_at(10);
    let mut wire = b"POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
    for c in [a, b] {
        wire.extend_from_slice(format!("{:x}\r\n", c.len()).as_bytes());
        wire.extend_from_slice(c);
        wire.extend_from_slice(b"\r\n");
    }
    wire.extend_from_slice(b"0\r\n\r\n");
    let r = raw_exchange(srv.addr, &wire, 1);
    assert!(r.starts_with("HTTP/1.1 200 OK"), "{r}");
    assert_eq!(up.seen.lock().unwrap()[0].body, body);
    // Expect: 100-continue gets an interim response, then the real one
    let head = format!("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nExpect: 100-continue\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    let mut s = std::net::TcpStream::connect(srv.addr).unwrap();
    use std::io::Write;
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.write_all(head.as_bytes()).unwrap();
    let mut buf = [0u8; 64];
    let n = s.read(&mut buf).unwrap();
    assert!(String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 100 Continue"));
    s.write_all(&body).unwrap();
    let mut rest = String::new();
    s.read_to_string(&mut rest).unwrap();
    assert!(rest.starts_with("HTTP/1.1 200 OK"), "{rest}");
    // garbage and oversize requests are refused without hurting the server
    assert!(raw_exchange(srv.addr, b"NOT HTTP\r\n\r\n", 1).starts_with("HTTP/1.1 400"));
    let r = raw_exchange(srv.addr, b"POST /v1/systemone HTTP/1.1\r\nContent-Length: 999999999999\r\n\r\n", 1);
    assert!(r.starts_with("HTTP/1.1 413"), "{r}");
    // many sequential requests on one connection use one upstream connection's worth of proxy threads
    let ag = agent();
    for i in 0..50 {
        assert_eq!(post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, &format!("s{i}"), spec()), &[]).status, 200);
    }
    // the connection cap answers 503 instead of queueing
    let mut c = config(ServeMode::Forward, &up.url(), &up.url());
    c.max_conns = 1;
    let capped = start(c).unwrap();
    let _held = std::net::TcpStream::connect(capped.addr).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let r = raw_exchange(capped.addr, b"GET /healthz HTTP/1.1\r\nConnection: close\r\n\r\n", 1);
    assert!(r.starts_with("HTTP/1.1 503"), "{r}");
}


// ---- the review's findings ---------------------------------------------------------------------------------------

use std::io::Write;
use std::sync::atomic::AtomicBool;

/// Everything the server sends until it closes the connection (a test that hangs here means it did not close).
fn raw_all(addr: SocketAddr, wire: &[u8]) -> String {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
    s.write_all(wire).unwrap();
    let mut out = vec![];
    match s.read_to_end(&mut out) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
        Err(e) => panic!("the connection stayed open: {e} (got {:?})", String::from_utf8_lossy(&out)),
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn head_of(r: &str) -> String {
    r.split("\r\n\r\n").next().unwrap_or("").to_ascii_lowercase()
}

/// A one-shot-per-connection upstream that sends exactly the bytes it is given (the in-crate `http` can't write
/// `Connection` headers or other raw oddities).
struct RawUp {
    addr: SocketAddr,
}

fn raw_upstream(response: Vec<u8>) -> RawUp {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let response = response.clone();
            std::thread::spawn(move || {
                let mut buf = vec![];
                let mut b = [0u8; 4096];
                let end = loop {
                    let n = s.read(&mut b).unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&b[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..end]).into_owned();
                let want: usize = head.to_ascii_lowercase().lines().find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                while buf.len() < end + want {
                    let n = s.read(&mut b).unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&b[..n]);
                }
                let _ = head;
                let _ = s.write_all(&response);
            });
        }
    });
    RawUp { addr }
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

/// A valid gzip stream (one stored deflate block) of `data`.
fn gzip(data: &[u8]) -> Vec<u8> {
    let n = data.len() as u16;
    let mut o = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff, 1];
    o.extend_from_slice(&n.to_le_bytes());
    o.extend_from_slice(&(!n).to_le_bytes());
    o.extend_from_slice(data);
    o.extend_from_slice(&crc32(data).to_le_bytes());
    o.extend_from_slice(&(data.len() as u32).to_le_bytes());
    o
}

fn json_post_wire(extra_headers: &str, body: &[u8]) -> Vec<u8> {
    let mut w = format!("POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra_headers}\r\n", body.len()).into_bytes();
    w.extend_from_slice(body);
    w
}

fn ns_list(srv: &Server) -> Vec<Value> {
    srv.status()["namespaces"].as_array().unwrap().clone()
}

// #1

#[test]
fn a_snapshot_from_another_upstream_is_refused_and_never_reused() {
    let a = Fake::start(oracle(true, Duration::ZERO));
    let b = Fake::start(oracle(true, Duration::ZERO));
    let dir = tmp("ident");
    let mut c = config(ServeMode::Serve, &a.url(), &a.url());
    c.snapshot_dir = Some(dir.join("p"));
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 200, 4, &spec);
    assert_eq!(learners(&srv, "rune", RUNE)[0].1, "graduated");
    let id = ns_list(&srv)[0]["identity"].clone();
    assert_eq!(id["upstream"], json!(a.url()), "{id}");
    srv.stop();
    let saved = std::fs::read_to_string(std::fs::read_dir(dir.join("p")).unwrap().next().unwrap().unwrap().path().join("snapshot.json")).unwrap();
    assert!(saved.contains(&a.url()), "the snapshot records its upstream");

    // the same upstream spelled differently (trailing slash) is the same namespace: its learners serve at once
    let mut c = config(ServeMode::Serve, &(a.url() + "/"), &(a.url() + "/"));
    c.snapshot_dir = Some(dir.join("p"));
    let srv = start(c).unwrap();
    post(srv.addr, &request(RUNE, "alpha error", spec()));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert!(ns_list(&srv)[0]["error"].is_null(), "{:?}", ns_list(&srv));
    assert_eq!(post(srv.addr, &request(RUNE, "alpha error error beta error", spec())).h("x-understudy"), Some("local"));
    srv.stop();

    // another upstream with the same snapshot directory: refused with a clear error, answered by its own lead, no local
    let mut c = config(ServeMode::Serve, &b.url(), &b.url());
    c.snapshot_dir = Some(dir.join("p"));
    let srv = start(c).unwrap();
    for _ in 0..3 {
        let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
        assert_eq!(out.h("x-understudy"), Some("lead"), "another upstream's learners must not answer");
        assert_eq!(out.status, 200);
    }
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let ns = ns_list(&srv);
    let err = ns[0]["error"].as_str().unwrap_or_else(|| panic!("no error in {ns:?}"));
    assert!(err.contains("refusing to reuse") && err.contains(&a.url()) && err.contains(&b.url()), "{err}");
    assert!(learners(&srv, "rune", RUNE).is_empty());
    assert_eq!(b.calls(), 3);
    drop(srv);
    std::fs::remove_dir_all(dir).unwrap();
}

// #2

#[test]
fn namespaces_and_learners_are_capped_and_made_only_by_successful_answers() {
    // upstream errors and non-answers create nothing
    let bad = Fake::start(Arc::new(|s: &Seen| {
        let v: Value = serde_json::from_slice(&s.body).unwrap();
        if v["model"].as_str().unwrap().contains("500") { reply(500, &[("content-type", "application/json")], br#"{"error":"no"}"#.to_vec()) } else { reply(200, &[("content-type", "application/json")], br#"{"hello":"world"}"#.to_vec()) }
    }));
    let srv = serve(ServeMode::Shadow, &bad, &bad);
    for i in 0..6 {
        assert_eq!(post(srv.addr, &request(&format!("jev-500-{i}"), "an error", spec())).status, 500);
        assert_eq!(post(srv.addr, &request(&format!("jev-noanswer-{i}"), "an error", spec())).status, 200);
    }
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert!(ns_list(&srv).is_empty(), "no namespace without a successful answer: {:?}", ns_list(&srv));
    drop(srv);

    // namespaces: capped, the rest forward-only and counted
    let up = Fake::start(oracle(true, Duration::ZERO));
    let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
    c.max_namespaces = 3;
    let srv = start(c).unwrap();
    for i in 0..10 {
        let out = post(srv.addr, &request(&format!("jev-anything-{i}"), "an error", spec()));
        assert_eq!((out.status, out.h("x-understudy")), (200, Some("lead")));
    }
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let st = srv.status();
    assert_eq!(st["namespaces"].as_array().unwrap().len(), 3, "{st}");
    assert_eq!(st["namespaces_refused"], 7);
    drop(srv);

    // learners per namespace: capped and counted
    let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
    c.max_learners = 2;
    let srv = start(c).unwrap();
    for i in 0..5 {
        let q = json!({"decision": choice(&format!("Which way {i}?"), &[("a", "x"), ("b", "y")])});
        assert_eq!(post(srv.addr, &request(RUNE, "an error", q)).status, 200);
        assert!(srv.wait_idle(Duration::from_secs(10)));
    }
    assert_eq!(learners(&srv, "rune", RUNE).len(), 2);
    assert_eq!(srv.status()["learners_refused"], 3);
}

// #3

#[test]
fn a_dripping_client_hits_the_total_deadline_and_a_stalled_reader_the_write_timeout() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let mut c = config(ServeMode::Forward, &up.url(), &up.url());
    c.read_deadline = Duration::from_secs(1);
    c.write_timeout = Duration::from_secs(1);
    let srv = start(c).unwrap();
    // bytes dripped well inside any per-read timeout still run out of total time
    let mut s = std::net::TcpStream::connect(srv.addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(8))).unwrap();
    s.write_all(b"POST /v1/systemone HTTP/1.1\r\nContent-Length: 1000\r\n\r\n").unwrap();
    let t = Instant::now();
    let mut closed = false;
    while t.elapsed() < Duration::from_secs(6) {
        if s.write_all(b"x").is_err() {
            closed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    let mut got = String::new();
    let _ = s.read_to_string(&mut got);
    assert!(closed || got.starts_with("HTTP/1.1 408") || t.elapsed() < Duration::from_secs(6), "{got}");
    assert!(t.elapsed() < Duration::from_secs(5), "the request had one second in total: {:?}", t.elapsed());
    assert!(got.is_empty() || got.starts_with("HTTP/1.1 408"), "{got}");
    // more than a few trailers
    let mut w = b"POST /v1/systemone HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n".to_vec();
    for i in 0..40 {
        w.extend_from_slice(format!("X-T{i}: 1\r\n").as_bytes());
    }
    w.extend_from_slice(b"\r\n");
    assert!(raw_all(srv.addr, &w).starts_with("HTTP/1.1 431"));
    drop(srv);

    // a client that never reads a big response frees its connection slot after the write timeout
    let big = Fake::start(Arc::new(|_: &Seen| reply(200, &[("content-type", "application/json")], vec![b' '; 48 << 20])));
    let mut c = config(ServeMode::Forward, &big.url(), &big.url());
    c.write_timeout = Duration::from_secs(1);
    c.max_conns = 1;
    let srv = start(c).unwrap();
    let mut s = std::net::TcpStream::connect(srv.addr).unwrap();
    s.write_all(&json_post_wire("", &request(RUNE, "an error", spec()))).unwrap();
    std::thread::sleep(Duration::from_secs(4)); // never reads
    let r = raw_all(srv.addr, b"GET /healthz HTTP/1.1\r\nConnection: close\r\n\r\n");
    assert!(r.starts_with("HTTP/1.1 200"), "the stalled connection must have been dropped: {}", &r[..r.len().min(80)]);
}

// #4

#[test]
fn ambiguous_request_framing_is_refused_and_closes_the_connection() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve(ServeMode::Forward, &up, &up);
    let body = request(RUNE, "an error", spec());
    let n = body.len();
    let smuggled = format!("POST /v1/systemone HTTP/1.1\r\nContent-Length: {n}\r\n\r\n{}", String::from_utf8_lossy(&body));
    let hdr = "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n";
    let cases: Vec<(String, &str, String)> = vec![
        ("duplicate identical Content-Length".into(), "HTTP/1.1 400", format!("{hdr}Content-Length: {n}\r\nContent-Length: {n}\r\n\r\n")),
        ("conflicting Content-Length".into(), "HTTP/1.1 400", format!("{hdr}Content-Length: {n}\r\nContent-Length: 3\r\n\r\n")),
        ("Content-Length list".into(), "HTTP/1.1 400", format!("{hdr}Content-Length: {n}, {n}\r\n\r\n")),
        ("signed Content-Length".into(), "HTTP/1.1 400", format!("{hdr}Content-Length: +{n}\r\n\r\n")),
        ("TE and CL".into(), "HTTP/1.1 400", format!("{hdr}Content-Length: {n}\r\nTransfer-Encoding: chunked\r\n\r\n")),
        ("non-chunked TE".into(), "HTTP/1.1 501", format!("{hdr}Transfer-Encoding: gzip\r\n\r\n")),
        ("TE ending in chunked".into(), "HTTP/1.1 501", format!("{hdr}Transfer-Encoding: gzip, chunked\r\n\r\n")),
        ("substring TE".into(), "HTTP/1.1 501", format!("{hdr}Transfer-Encoding: xchunkedx\r\n\r\n")),
        ("two TE headers".into(), "HTTP/1.1 400", format!("{hdr}Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")),
        ("chunk without CRLF".into(), "HTTP/1.1 400", format!("{hdr}Transfer-Encoding: chunked\r\n\r\n4\r\nabcdXX0\r\n\r\n")),
        ("chunk size with a sign".into(), "HTTP/1.1 400", format!("{hdr}Transfer-Encoding: chunked\r\n\r\n+4\r\nabcd\r\n0\r\n\r\n")),
        ("bare LF line".into(), "HTTP/1.1 400", format!("{hdr}Content-Length: {n}\n\r\n")),
        ("space before the colon".into(), "HTTP/1.1 400", format!("{hdr}Content-Length : {n}\r\n\r\n")),
        ("folded header".into(), "HTTP/1.1 400", format!("{hdr}X-A: 1\r\n  folded\r\n\r\n")),
    ];
    for (what, want, wire) in cases {
        // the body (and a whole second request) follow; neither may ever reach the upstream
        let mut w = wire.into_bytes();
        w.extend_from_slice(&body);
        w.extend_from_slice(smuggled.as_bytes());
        let r = raw_all(srv.addr, &w);
        assert!(r.starts_with(want), "{what}: {}", &r[..r.len().min(120)]);
        assert!(head_of(&r).contains("connection: close"), "{what}: {r}");
        assert_eq!(up.calls(), 0, "{what}");
    }
    // well-formed chunking is still fine, with trailers and extensions
    let mut w = format!("{hdr}Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n").into_bytes();
    w.extend_from_slice(format!("{:x};ext=1\r\n", body.len()).as_bytes());
    w.extend_from_slice(&body);
    w.extend_from_slice(b"\r\n0\r\nX-Trailer: 1\r\n\r\n");
    assert!(raw_all(srv.addr, &w).starts_with("HTTP/1.1 200"));
    assert_eq!(up.seen.lock().unwrap()[0].body, body);
}

// #5 and #6

#[test]
fn encoded_bodies_and_repeated_headers_pass_through_untouched() {
    let payload = gzip(br#"{"model":"jev-1","answers":{}}"#);
    let gz = payload.clone();
    let up = Fake::start(Arc::new(move |_: &Seen| {
        reply(200, &[("content-type", "application/json"), ("content-encoding", "gzip"), ("set-cookie", "a=1"), ("set-cookie", "b=2"), ("x-multi", "1"), ("x-multi", "2"), ("vary", "accept"), ("vary", "accept-encoding")], gz.clone())
    }));
    for mode in [ServeMode::Forward, ServeMode::Shadow, ServeMode::Serve] {
        let srv = serve(mode, &up, &up);
        let r = raw_all(srv.addr, &json_post_wire("Accept-Encoding: gzip\r\n", &request(RUNE, "an error", spec())));
        let (head, body) = r.split_once("\r\n\r\n").unwrap();
        let head = head.to_ascii_lowercase();
        assert!(head.contains("content-encoding: gzip"), "{mode:?}: Content-Encoding is end-to-end: {head}");
        for want in ["set-cookie: a=1", "set-cookie: b=2", "x-multi: 1", "x-multi: 2", "vary: accept\r", "vary: accept-encoding"] {
            assert!(head.contains(want), "{mode:?}: {want:?} missing from {head}");
        }
        assert_eq!(head.matches("set-cookie").count(), 2);
        assert!(head.contains(&format!("content-length: {}", payload.len())));
        // the bytes are the upstream's, still encoded (compare as bytes: they are not UTF-8)
        let mut s = std::net::TcpStream::connect(srv.addr).unwrap();
        s.write_all(&json_post_wire("Accept-Encoding: gzip\r\n", &request(RUNE, "an error", spec()))).unwrap();
        let mut raw = vec![];
        s.read_to_end(&mut raw).unwrap();
        assert!(raw.ends_with(&payload), "{mode:?}: the gzip bytes were altered");
        let _ = body;
    }
    // a caller's Accept-Encoding goes upstream; without one the proxy asks for identity (so it can learn)
    let seen = up.seen.lock().unwrap();
    assert!(seen.iter().any(|s| s.headers.contains(&("accept-encoding".into(), "gzip".into()))));
}

// #7 and #8

#[test]
fn end_to_end_request_headers_go_upstream_and_connection_nominated_ones_do_not() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve(ServeMode::Forward, &up, &up);
    let extra = "Idempotency-Key: abc-123\r\nAccept: application/vnd.x+json\r\nAccept: text/plain\r\nX-Request-Id: req-9\r\nIf-None-Match: \"v1\"\r\nAuthorization: Bearer caller\r\nCookie: a=1\r\nCookie: b=2\r\nX-Secret: hide-me\r\nKeep-Alive: timeout=5\r\nTE: trailers\r\nProxy-Authorization: no\r\nConnection: close, X-Secret\r\n";
    let r = raw_all(srv.addr, &json_post_wire(extra, &request(RUNE, "an error", spec())));
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
    let seen = up.seen.lock().unwrap()[0].clone();
    let h = |n: &str| -> Vec<String> { seen.headers.iter().filter(|(k, _)| k == n).map(|(_, v)| v.clone()).collect() };
    assert_eq!(h("idempotency-key"), ["abc-123"]);
    assert_eq!(h("x-request-id"), ["req-9"]);
    assert_eq!(h("if-none-match"), ["\"v1\""]);
    assert_eq!(h("authorization"), ["Bearer caller"]);
    assert_eq!(h("accept"), ["application/vnd.x+json, text/plain"], "repeated values are all kept");
    assert_eq!(h("cookie"), ["a=1; b=2"]);
    for gone in ["x-secret", "keep-alive", "te", "proxy-authorization", "expect"] {
        assert!(h(gone).is_empty(), "{gone} is connection-specific: {:?}", seen.headers);
    }
    assert_ne!(h("host"), ["x"], "Host is the upstream's");

    // an upstream's Connection-nominated headers stay behind
    let raw = raw_upstream(b"HTTP/1.1 200 OK\r\nConnection: close, X-Internal\r\nX-Internal: secret\r\nX-Public: ok\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}".to_vec());
    let url = format!("http://{}/v1/systemone", raw.addr);
    let srv = start(config(ServeMode::Forward, &url, &url)).unwrap();
    let r = raw_all(srv.addr, &json_post_wire("", &request(RUNE, "an error", spec())));
    let head = head_of(&r);
    assert!(head.contains("x-public: ok"), "{head}");
    assert!(!head.contains("x-internal"), "{head}");
}

// #9

#[test]
fn an_over_limit_upstream_response_fails_explicitly() {
    let up = Fake::start(Arc::new(|s: &Seen| {
        let v: Value = serde_json::from_slice(&s.body).unwrap();
        let n = if v["state"] == "big" { 5000 } else { 1000 };
        reply(200, &[("content-type", "application/octet-stream")], vec![b'z'; n])
    }));
    let dir = tmp("big");
    let mut c = config(ServeMode::Forward, &up.url(), &up.url());
    c.max_response = 1000;
    c.capture = Some(dir.join("cap.jsonl"));
    let srv = start(c).unwrap();
    let exact = post(srv.addr, &request(RUNE, "ok", spec()));
    assert_eq!((exact.status, exact.body.len()), (200, 1000), "a response at the limit is relayed whole");
    let big = post(srv.addr, &request(RUNE, "big", spec()));
    assert_eq!(big.status, 502, "never a truncated 200");
    assert_eq!(big.json()["error"]["type"], "upstream_response_too_large");
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert_eq!(srv.status()["response_too_large"], 1);
    let log = std::fs::read_to_string(dir.join("cap.jsonl")).unwrap();
    let e: Vec<Value> = log.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    // Capture lines are written by two helper threads after the responses go out, so their order is not the requests':
    // under load the first request's line can land second. Find the failed request's line by what it says.
    assert_eq!(e.len(), 2, "{e:?}");
    let failed: Vec<&Value> = e.iter().filter(|x| x["status"] == 502).collect();
    assert_eq!(failed.len(), 1, "exactly one failed line: {e:?}");
    assert!(failed[0]["error"].as_str().unwrap().contains("limit"), "{}", failed[0]);
    assert_eq!(failed[0]["response_too_large"], true);
    assert!(e.iter().any(|x| x["status"] == 200 && x.get("response_too_large").is_none()), "the exact-limit request logged as served: {e:?}");
    drop(srv);
    std::fs::remove_dir_all(dir).unwrap();
}

// #10

#[test]
fn an_audit_compares_the_prediction_that_was_served_at_the_fit_that_served_it() {
    let gate = Arc::new(AtomicBool::new(true));
    let g = gate.clone();
    let inner = oracle(true, Duration::ZERO);
    let up = Fake::start(Arc::new(move |s: &Seen| {
        // an audit carrying the hold header waits at the gate (as a slow lead would)
        if s.headers.iter().any(|(k, v)| k == "x-audit-gate" && v == "hold") {
            while !g.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        inner(s)
    }));
    let dir = tmp("audit");
    let mut c = config(ServeMode::Serve, &up.url(), &up.url());
    c.audit_rate = 1.0;
    c.capture = Some(dir.join("cap.jsonl"));
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 200, 6, &spec);
    let audits = || -> Vec<Value> { std::fs::read_to_string(dir.join("cap.jsonl")).unwrap().lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).filter(|e| e["kind"] == "audit").collect() };
    let n0 = audits().len();
    let ag = agent();
    let hold = [("x-audit-gate", "hold")];

    // 1: no refit meanwhile: the served prediction is recorded, agrees, and counts toward demotion
    gate.store(false, Ordering::SeqCst);
    let out = post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, "alpha error error beta error", spec()), &hold);
    assert_eq!(out.h("x-understudy"), Some("local"));
    std::thread::sleep(Duration::from_millis(200));
    gate.store(true, Ordering::SeqCst);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let a = audits();
    let q = &a[n0]["questions"][0];
    assert_eq!(q["served"]["label"], "Héal ✨", "{}", a[n0]);
    assert_eq!(q["served"]["audit_counted"], true);
    assert_eq!(q["agree"], true);

    // 2: the learner refits while the audit is outstanding (more examples arrive through forwarded requests): the
    // audit still judges what was served, and says that the fit it judged is gone
    let v0 = ns_list(&srv)[0]["fit_version"].as_u64().unwrap();
    gate.store(false, Ordering::SeqCst);
    let out = post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, "alpha error error beta error", spec()), &hold);
    assert_eq!(out.h("x-understudy"), Some("local"));
    let learned0 = srv.status()["learned"].as_u64().unwrap();
    let mut r = Xoshiro::new(77);
    for _ in 0..60 {
        let error = r.uniform() < 0.5;
        let st = state_text(&mut r, error);
        let mut q = spec().as_object().unwrap().clone();
        q.insert("other".into(), json!({"type": "mystery"})); // an opaque sibling: forwarded, and still learned from
        assert_eq!(post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, &st, Value::Object(q)), &[]).h("x-understudy"), Some("lead"));
    }
    let t = Instant::now();
    while (srv.status()["learned"].as_u64().unwrap() < learned0 + 60 || ns_list(&srv)[0]["pending"] != 0) && t.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(ns_list(&srv)[0]["fit_version"].as_u64().unwrap() > v0, "a refit happened");
    gate.store(true, Ordering::SeqCst);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let a = audits();
    let q = &a[n0 + 1]["questions"][0];
    assert_eq!(q["served"]["label"], "Héal ✨", "{}", a[n0 + 1]);
    assert_eq!(q["served"]["fit_version"], v0);
    assert!(q["served"]["audit_counted"].as_str().unwrap().starts_with("stale"), "{q}");
    assert_eq!(q["agree"], true);
    drop(srv);
    std::fs::remove_dir_all(dir).unwrap();
}

// #11

#[test]
fn a_full_post_queue_drops_and_counts_and_never_runs_work_on_the_connection() {
    // the capture log is a pipe nobody reads: the helpers block in their first writes
    let dir = tmp("fifo");
    let fifo = dir.join("cap.fifo");
    assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
    let keep = std::fs::OpenOptions::new().read(true).write(true).open(&fifo).unwrap();
    let up = Fake::start(oracle(true, Duration::ZERO));
    let mut c = config(ServeMode::Forward, &up.url(), &up.url());
    c.capture = Some(fifo.clone());
    c.capture_bodies = true;
    c.post_queue = 2;
    let srv = start(c).unwrap();
    let filler = "alpha ".repeat(20_000);
    let ag = agent();
    let t = Instant::now();
    for i in 0..30 {
        let out = post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, &format!("{filler} {i}"), spec()), &[]);
        assert_eq!(out.status, 200, "request {i}");
    }
    assert!(t.elapsed() < Duration::from_secs(10), "{:?}: responses wait for no post-response work", t.elapsed());
    assert!(srv.status()["post_dropped"].as_u64().unwrap() > 0, "{}", srv.status());
    // let the helpers finish so the server can stop
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let drain = std::thread::spawn(move || {
        let mut k = keep;
        let mut b = vec![0u8; 1 << 16];
        while !s2.load(Ordering::SeqCst) {
            let _ = k.read(&mut b);
        }
    });
    drop(srv);
    stop.store(true, Ordering::SeqCst);
    let mut w = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();
    let _ = w.write_all(b"\n");
    let _ = drain.join();
    std::fs::remove_dir_all(dir).unwrap();
}

// #12

#[test]
fn no_length_on_204_and_the_upstreams_on_304_and_probabilities_sum_to_one() {
    let up = Fake::start(Arc::new(|s: &Seen| {
        let v: Value = serde_json::from_slice(&s.body).unwrap();
        match v["state"].as_str().unwrap() {
            "204" => reply(204, &[], vec![]),
            _ => reply(304, &[("etag", "\"v7\""), ("content-length", "1234")], vec![]),
        }
    }));
    let srv = serve(ServeMode::Forward, &up, &up);
    let r = raw_all(srv.addr, &json_post_wire("", &request(RUNE, "204", spec())));
    assert!(r.starts_with("HTTP/1.1 204"), "{r}");
    assert!(!head_of(&r).contains("content-length"), "{r}");
    assert!(r.ends_with("\r\n\r\n"));
    let r = raw_all(srv.addr, &json_post_wire("", &request(RUNE, "304", spec())));
    assert!(r.starts_with("HTTP/1.1 304"), "{r}");
    assert!(head_of(&r).contains("content-length: 1234"), "the 304 keeps the representation's length: {r}");
    assert!(r.ends_with("\r\n\r\n"), "and carries no body");
    assert!(head_of(&r).contains("etag: \"v7\""));
    // the proxy's own locally-made 204/304 don't exist; its 4xx replies keep an exact length
    let r = raw_all(srv.addr, b"GET /nope HTTP/1.1\r\nConnection: close\r\n\r\n");
    assert!(head_of(&r).contains("content-length: "));

    use understudy::question::Question;
    let q = Question::from_wire(&choice("Which?", &[("a", "x"), ("b", "y"), ("c", "z")]), None).unwrap();
    let units = |v: &Value| -> Vec<i64> { v["probabilities"].as_object().unwrap().values().map(|x| (x.as_f64().unwrap() * 1e4).round() as i64).collect() };
    let (v, best) = q.answer_object_normalized(&[0.90004, 0.04998, 0.04998]);
    assert_eq!((best, units(&v)), (0, vec![9000, 500, 500]), "{v}");
    assert_eq!(v["confidence"].as_f64().unwrap(), 0.9);
    // rounding that would sum to 1.0001 or 0.9999 is apportioned instead
    for p in [[0.33335, 0.33335, 0.3333], [0.45005, 0.45005, 0.0999], [1.0 / 3.0; 3], [0.0001, 0.0001, 0.9998]] {
        let (v, best) = q.answer_object_normalized(&p);
        let u = units(&v);
        assert_eq!(u.iter().sum::<i64>(), 10_000, "{p:?} -> {v}");
        let name = v["choice"].as_str().unwrap();
        assert_eq!(v["confidence"], v["probabilities"][name], "confidence matches the chosen class: {v}");
        assert_eq!(q.options[best], name);
    }
    // many small classes lose no mass
    let crit: Vec<(String, String)> = (0..30).map(|i| (format!("k{i:02}"), "d".to_string())).collect();
    let crit: Vec<(&str, &str)> = crit.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let big = Question::from_wire(&choice("Which?", &crit), None).unwrap();
    let (v, _) = big.answer_object_normalized(&[1.0 / 30.0; 30]);
    assert_eq!(units(&v).iter().sum::<i64>(), 10_000);
}

#[test]
fn shutdown_waits_for_the_outstanding_race_audit() {
    let delay = Arc::new(AtomicU64::new(0));
    let (_up, srv, log) = graduated_race(&delay, 64);
    assert!(srv.wait_idle(Duration::from_secs(10)));
    delay.store(500, Ordering::Relaxed);
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!(out.h("x-understudy"), Some("local"));
    // stop at once, with the lead's answer still on its way
    srv.stop();
    let lines = capture_lines(&log);
    let req = lines.iter().rev().find(|l| l["kind"] == "request" && l["routed"] == "local").expect("the local request line");
    assert!(lines.iter().any(|l| l["kind"] == "audit" && l["race_id"] == req["race_id"]), "the late audit was lost at shutdown");
}

// ---- a chat route: callers speak System One, the upstream speaks chat -----------------------------------------------

/// A fake chat endpoint: the oracle lead as an OpenAI-compatible completion. The state is read back out of the prompt.
fn chat_oracle() -> Handler {
    Arc::new(|s: &Seen| {
        let v: Value = serde_json::from_slice(&s.body).unwrap();
        let user = v["messages"][1]["content"].as_str().unwrap();
        let err = user.split("State:\n").nth(1).unwrap_or("").contains("error");
        let schema = &v["response_format"]["json_schema"]["schema"]["properties"]["answer"];
        let content = if let Some(opts) = schema["enum"].as_array() {
            let mut keys: Vec<&str> = opts.iter().map(|o| o.as_str().unwrap()).collect();
            keys.sort();
            json!({"answer": if err { keys[0] } else { keys[keys.len() - 1] }})
        } else {
            json!({"answer": err})
        };
        let body = json!({"model": v["model"], "choices": [{"message": {"content": content.to_string()}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 30, "completion_tokens": 4}});
        reply(200, &[("content-type", "application/json")], serde_json::to_vec(&body).unwrap())
    })
}

fn chat_server(mode: ServeMode, up: &Fake) -> Server {
    let mut c = config(mode, "http://127.0.0.1:9/unused", "http://127.0.0.1:9/unused");
    let spec = understudy::chat::ChatSpec::new(&format!("http://{}/v1", up.addr));
    c.routes = vec![Route { name: "chat".into(), pattern: "chatm-*".into(), url: spec.url.clone(), bearer: Some("chat-key".into()), chat: Some(spec), clef: None }];
    start(c).unwrap()
}

#[test]
fn a_chat_route_answers_in_system_one_and_learns_and_serves_locally() {
    let up = Fake::start(chat_oracle());
    let srv = chat_server(ServeMode::Serve, &up);
    let questions = || json!({"decision": choice("Which way?", &[("A", "heal"), ("B", "tune")]), "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "yes", "false": "no"}}});
    let first = post(srv.addr, &request("chatm-1", "an error happened", questions()));
    assert_eq!((first.status, first.h("x-understudy")), (200, Some("lead")));
    let v = first.json();
    assert_eq!(v["model"], "chatm-1");
    assert_eq!(v["answers"]["decision"]["choice"], "A");
    assert_eq!(v["answers"]["decision"]["probabilities"], json!({"A": 0.9, "B": 0.1}));
    assert_eq!(v["answers"]["urgent"]["noul"], 0.9);
    assert_eq!(v["understudy"]["probabilities"]["decision"], "fixed");
    // one chat call per question, the key as the route's bearer
    assert_eq!(up.calls(), 2);
    for s in up.seen.lock().unwrap().iter() {
        assert_eq!(s.path, "/v1/chat/completions");
        assert!(s.headers.iter().any(|(k, v)| k == "authorization" && v == "Bearer chat-key"));
    }
    // it learns from chat answers like from any lead's, then answers locally
    teach(&srv, "chatm-1", 220, 5, &questions);
    let l = learners(&srv, "chat", "chatm-1");
    assert_eq!(l.len(), 2, "{l:?}");
    assert!(l.iter().all(|x| x.1 == "graduated"), "{l:?}");
    let calls = up.calls();
    let st = state_text(&mut Xoshiro::new(99), true);
    let out = post(srv.addr, &request("chatm-1", &st, questions()));
    assert_eq!(out.h("x-understudy"), Some("local"));
    assert_eq!(out.json()["answers"]["decision"]["choice"], "A");
    assert_eq!(up.calls(), calls);
}

#[test]
fn a_chat_route_passes_chat_errors_through_and_rejects_non_decisions() {
    let up = Fake::start(Arc::new(|_| reply(429, &[("content-type", "application/json"), ("retry-after", "3")], "{\"error\":{\"message\":\"rate limited\"}}")));
    let srv = chat_server(ServeMode::Forward, &up);
    let out = post(srv.addr, &request("chatm-1", "x", spec()));
    assert_eq!(out.status, 429);
    assert_eq!(out.body, b"{\"error\":{\"message\":\"rate limited\"}}");
    assert_eq!(out.h("retry-after"), Some("3"));
    // a reply that is not an answer is a 502, not a made-up answer
    let up = Fake::start(Arc::new(|_| reply(200, &[("content-type", "application/json")], "{\"choices\":[{\"message\":{\"content\":\"hmm, hard to say\"}}]}")));
    let srv = chat_server(ServeMode::Forward, &up);
    let out = post(srv.addr, &request("chatm-1", "x", spec()));
    assert_eq!(out.status, 502);
    assert!(String::from_utf8_lossy(&out.body).contains("unusable"));
    // a body that is not a decision request
    let out = post(srv.addr, b"{\"model\":\"chatm-1\"}");
    assert_eq!(out.status, 400);
}

#[test]
fn a_chat_route_follows_the_serve_response_limit_not_the_spec_default() {
    // 17 MiB is over the chat spec's own 16 MiB default but under the configured 32 MiB
    const BIG: usize = 17 << 20;
    let serve_with = |up: &Fake| {
        let mut c = config(ServeMode::Forward, "http://127.0.0.1:9/unused", "http://127.0.0.1:9/unused");
        c.max_response = 32 << 20;
        let spec = understudy::chat::ChatSpec::new(&format!("http://{}/v1", up.addr));
        c.routes = vec![Route { name: "chat".into(), pattern: "chatm-*".into(), url: spec.url.clone(), bearer: None, chat: Some(spec), clef: None }];
        start(c).unwrap()
    };
    let one = || json!({"urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "yes", "false": "no"}}});
    // a valid completion whose content carries 17 MiB of text after the answer's JSON
    let up = Fake::start(Arc::new(|_| {
        let body = json!({"model": "m", "choices": [{"message": {"content": "{\"answer\": true}", "padding": "p".repeat(BIG)}, "finish_reason": "stop"}], "usage": {}});
        reply(200, &[("content-type", "application/json")], serde_json::to_vec(&body).unwrap())
    }));
    let srv = serve_with(&up);
    let out = post(srv.addr, &request("chatm-1", "x", one()));
    assert_eq!(out.status, 200, "{}", String::from_utf8_lossy(&out.body[..out.body.len().min(300)]));
    assert_eq!(out.json()["answers"]["urgent"]["noul"], 0.9);
    // a 17 MiB 429 body passes through byte-exact
    let big: Vec<u8> = (0..BIG).map(|i| b'a' + (i % 26) as u8).collect();
    let b2 = big.clone();
    let up = Fake::start(Arc::new(move |_| reply(429, &[("content-type", "text/plain")], b2.clone())));
    let srv = serve_with(&up);
    let out = post(srv.addr, &request("chatm-1", "x", one()));
    assert_eq!(out.status, 429);
    assert!(out.body == big, "17 MiB error body not preserved ({} bytes)", out.body.len());
    // an error body over 1 MiB and under the limit is preserved byte-exact too
    let mid: Vec<u8> = (0..(3 << 20)).map(|i| b'A' + (i % 26) as u8).collect();
    let m2 = mid.clone();
    let up = Fake::start(Arc::new(move |_| reply(500, &[("content-type", "text/plain")], m2.clone())));
    let srv = serve_with(&up);
    let out = post(srv.addr, &request("chatm-1", "x", one()));
    assert_eq!(out.status, 500);
    assert!(out.body == mid, "3 MiB error body not preserved ({} bytes)", out.body.len());
}

// ---- defaults, --lead-url and pass-through -----------------------------------------------------------------------

#[derive(Parser)]
struct Wrap {
    #[command(flatten)]
    a: ServeArgs,
}

#[test]
fn snapshots_are_on_by_default_and_no_snapshots_turns_them_off() {
    let c = Wrap::parse_from(["x", "--lead-url", "http://l.example/v1/systemone"]).a.config_with(&Config::default(), 0.1, factory()).unwrap();
    assert_eq!(c.snapshot_dir.as_deref(), Some(std::path::Path::new("understudy-snapshots")));
    let c = Wrap::parse_from(["x", "--lead-url", "http://l.example/v1/systemone", "--no-snapshots"]).a.config_with(&Config::default(), 0.1, factory()).unwrap();
    assert!(c.snapshot_dir.is_none());
    let c = Wrap::parse_from(["x", "--lead-url", "http://l.example/v1/systemone", "--snapshot-dir", "elsewhere"]).a.config_with(&Config::default(), 0.1, factory()).unwrap();
    assert_eq!(c.snapshot_dir.as_deref(), Some(std::path::Path::new("elsewhere")));
}

#[test]
fn a_restart_in_serve_mode_keeps_what_shadow_mode_learned() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let dir = tmp("shadow-then-serve");
    let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
    c.snapshot_dir = Some(dir.join("s"));
    let srv = start(c).unwrap();
    teach(&srv, RUNE, 200, 4, &spec);
    assert_eq!(learners(&srv, "rune", RUNE)[0].1, "graduated");
    srv.stop();
    let mut c = config(ServeMode::Serve, &up.url(), &up.url());
    c.snapshot_dir = Some(dir.join("s"));
    let srv = start(c).unwrap();
    // the first request builds the namespace from the snapshot (and is forwarded); then the restored learner serves
    post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    let calls = up.calls();
    let out = post(srv.addr, &request(RUNE, "alpha error error beta error", spec()));
    assert_eq!(out.h("x-understudy"), Some("local"), "the restarted server answers from the snapshot");
    assert_eq!(up.calls(), calls, "no lead call was needed");
    drop(srv);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn lead_url_is_a_catch_all_for_any_model_and_passes_the_callers_authorization() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let w = Wrap::parse_from(["x", "--lead-url", &up.url(), "--no-snapshots", "--no-capture"]);
    let mut c = w.a.config_with(&Config::default(), 0.0, factory()).unwrap();
    c.bind = "127.0.0.1:0".into();
    let srv = start(c).unwrap();
    let out = post_with(&agent(), srv.addr, "/v1/systemone", &request("some-model-nobody-configured", "s", spec()), &[("authorization", "Bearer caller-key")]);
    assert_eq!((out.status, out.h("x-understudy")), (200, Some("lead")));
    let seen = up.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let auth = seen[0].headers.iter().find(|(k, _)| k == "authorization").map(|(_, v)| v.as_str());
    assert_eq!(auth, Some("Bearer caller-key"));
    // a configured route key replaces it
    unsafe { std::env::set_var("UNDERSTUDY_TEST_LEAD_KEY", "route-key") };
    let w = Wrap::parse_from(["x", "--lead-url", &up.url(), "--route-key-env", "lead=UNDERSTUDY_TEST_LEAD_KEY"]);
    let c = w.a.config_with(&Config::default(), 0.0, factory()).unwrap();
    assert_eq!(c.routes[0].bearer.as_deref(), Some("route-key"));
    // presets from the config file are tried first
    let cfg = Config::parse("[leads.rune]\nurl = \"http://r.example/v1/systemone\"\nmodel = \"m\"\n").unwrap();
    let c = Wrap::parse_from(["x", "--lead-url", &up.url()]).a.config_with(&cfg, 0.0, factory()).unwrap();
    assert_eq!(c.routes.iter().map(|r| r.pattern.as_str()).collect::<Vec<_>>(), vec!["m", "*"]);
}

/// An upstream that answers any request with a chunked event stream: one event at once, one 600 ms later. It records
/// each request's method, target, headers and body.
fn sse_upstream() -> (SocketAddr, Arc<Mutex<Vec<(String, String, Vec<(String, String)>, Vec<u8>)>>>) {
    use std::io::Write;
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(vec![]));
    let sn = seen.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let sn = sn.clone();
            std::thread::spawn(move || {
                let mut buf = vec![];
                let mut b = [0u8; 4096];
                let end = loop {
                    let n = s.read(&mut b).unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&b[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..end]).into_owned();
                let mut lines = head.lines();
                let mut first = lines.next().unwrap().split(' ');
                let (method, target) = (first.next().unwrap().to_string(), first.next().unwrap().to_string());
                let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string())).collect();
                let want: usize = headers.iter().find(|(k, _)| k == "content-length").map(|(_, v)| v.parse().unwrap()).unwrap_or(0);
                while buf.len() < end + want {
                    let n = s.read(&mut b).unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&b[..n]);
                }
                sn.lock().unwrap().push((method, target, headers, buf[end..end + want].to_vec()));
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nx-upstream: yes\r\n\r\n");
                let _ = s.write_all(b"d\r\ndata: first\n\n\r\n");
                let _ = s.flush();
                std::thread::sleep(Duration::from_millis(600));
                let _ = s.write_all(b"e\r\ndata: second\n\n\r\n0\r\n\r\n");
                let _ = s.flush();
            });
        }
    });
    (addr, seen)
}

#[test]
fn an_unknown_path_is_a_404_and_nothing_is_forwarded() {
    let (up, seen) = sse_upstream();
    let w = Wrap::parse_from(["x", "--lead-url", &format!("http://{up}/v1/systemone"), "--no-snapshots", "--no-capture"]);
    let mut c = w.a.config_with(&Config::default(), 0.0, factory()).unwrap();
    c.bind = "127.0.0.1:0".into();
    let srv = start(c).unwrap();
    for wire in ["GET /v1/models HTTP/1.1\r\nhost: t\r\nauthorization: Bearer caller-key\r\nconnection: close\r\n\r\n", "GET @127.0.0.1:1/leak HTTP/1.1\r\nhost: t\r\nconnection: close\r\n\r\n", "POST /v1/chat/completions HTTP/1.1\r\nhost: t\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}"] {
        let r = raw_all(srv.addr, wire.as_bytes());
        let h = head_of(&r);
        assert!(h.starts_with("http/1.1 404") || h.starts_with("http/1.1 400"), "{r}");
        if h.starts_with("http/1.1 404") {
            assert!(r.contains("Understudy handles only /v1/systemone; this path is not forwarded"), "{r}");
        }
    }
    assert!(srv.wait_idle(Duration::from_secs(5)));
    assert!(seen.lock().unwrap().is_empty(), "nothing was forwarded");
}

#[test]
fn a_zstd_compressed_request_is_routed_and_learned_and_goes_upstream_as_sent() {
    // the upstream reads no body (a real one would decompress it): it answers a fixed choice
    let up = Fake::start(Arc::new(|_: &Seen| reply(200, &[("content-type", "application/json")], r#"{"model":"m","answers":{"decision":{"type":"choice","choice":"Tune up","confidence":0.9,"probabilities":{"Héal ✨":0.1,"Tune up":0.9}}},"usage":{}}"#.as_bytes().to_vec())));
    let srv = serve(ServeMode::Shadow, &up, &up);
    let plain = request(RUNE, &"alpha error ".repeat(200), spec());
    let packed = ruzstd::encoding::compress_to_vec(&plain[..], ruzstd::encoding::CompressionLevel::Fastest);
    let out = post_with(&agent(), srv.addr, "/v1/systemone", &packed, &[("content-encoding", "zstd")]);
    assert_eq!((out.status, out.h("x-understudy")), (200, Some("lead")));
    let seen = up.seen.lock().unwrap().clone();
    assert_eq!(seen[0].body, packed, "the upstream gets the compressed bytes");
    assert_eq!(seen[0].headers.iter().find(|(k, _)| k == "content-encoding").map(|(_, v)| v.as_str()), Some("zstd"));
    assert!(srv.wait_idle(Duration::from_secs(10)));
    assert_eq!(learners(&srv, "rune", RUNE).len(), 1, "it was read and learned from");
    let bad = post_with(&agent(), srv.addr, "/v1/systemone", b"not zstd", &[("content-encoding", "zstd")]);
    assert_eq!(bad.status, 400);
    let other = post_with(&agent(), srv.addr, "/v1/systemone", &plain, &[("content-encoding", "br")]);
    assert_eq!(other.status, 400);
}

#[test]
fn a_zstd_frame_may_not_declare_a_window_over_the_body_budget() {
    let up = Fake::start(Arc::new(|_: &Seen| reply(200, &[("content-type", "application/json")], r#"{"model":"m","answers":{"decision":{"type":"choice","choice":"Tune up","confidence":0.9,"probabilities":{"Héal ✨":0.1,"Tune up":0.9}}},"usage":{}}"#.as_bytes().to_vec())));
    let mut c = config(ServeMode::Shadow, &up.url(), &up.url());
    c.max_body = 1 << 20;
    let srv = start(c).unwrap();
    // magic, descriptor (no checksum, no size, multi-segment), window descriptor 0x88 = 2^(10+17) = 128 MiB, one last RLE block
    let frame = [0x28u8, 0xB5, 0x2F, 0xFD, 0x00, 0x88, 0x0B, 0x00, 0x00, b'x'];
    let out = post_with(&agent(), srv.addr, "/v1/systemone", &frame, &[("content-encoding", "zstd")]);
    assert_eq!(out.status, 400, "{}", String::from_utf8_lossy(&out.body));
    assert!(String::from_utf8_lossy(&out.body).contains("unreadable"), "{}", String::from_utf8_lossy(&out.body));
    assert_eq!(up.calls(), 0);
    // an ordinary compressed request still works under the same small budget
    let plain = request(RUNE, &"alpha error ".repeat(200), spec());
    let packed = ruzstd::encoding::compress_to_vec(&plain[..], ruzstd::encoding::CompressionLevel::Fastest);
    let ok = post_with(&agent(), srv.addr, "/v1/systemone", &packed, &[("content-encoding", "zstd")]);
    assert_eq!(ok.status, 200);
}

/// Parse the default flags (plus `extra`) into a config whose catch-all lead is `up`, and start it.
fn serve_from_flags(up: &Fake, extra: &[&str]) -> Server {
    #[derive(Parser)]
    struct W {
        #[command(flatten)]
        a: ServeArgs,
    }
    let url = up.url();
    let mut argv = vec!["x", "--lead-url", url.as_str(), "--bind", "127.0.0.1:0", "--no-snapshots"];
    argv.extend_from_slice(extra);
    let w = W::parse_from(argv);
    let mut c = w.a.config_with(&Config::default(), 0.0, factory()).unwrap();
    c.timeout = Duration::from_secs(10);
    start(c).unwrap()
}

#[test]
fn default_mode_forwards_until_graduation_then_answers_locally() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve_from_flags(&up, &[]);
    let questions = || spec();
    // a fresh start answers nothing locally: the first N requests of one question all reach the lead
    let n = 10;
    let ag = agent();
    let mut r = Xoshiro::new(5);
    for _ in 0..n {
        let st = state_text(&mut r, true);
        let out = post_with(&ag, srv.addr, "/v1/systemone", &request(RUNE, &st, questions()), &[]);
        assert_eq!(out.h("x-understudy"), Some("lead"));
    }
    assert_eq!(up.calls(), n);
    // enough examples graduate the question without a restart, and later requests are local
    teach(&srv, RUNE, 220, 3, &questions);
    assert_eq!(learners(&srv, "lead", RUNE)[0].1, "graduated");
    let mut r = Xoshiro::new(77);
    let local = (0..50)
        .filter(|_| {
            let st = state_text(&mut r, false);
            post(srv.addr, &request(RUNE, &st, questions())).h("x-understudy") == Some("local")
        })
        .count();
    assert!(local >= 40, "a graduated question is answered locally by default: {local}/50");
}

#[test]
fn shadow_mode_never_answers_locally() {
    let up = Fake::start(oracle(true, Duration::ZERO));
    let srv = serve_from_flags(&up, &["--mode", "shadow"]);
    let questions = || spec();
    teach(&srv, RUNE, 220, 3, &questions);
    assert_eq!(learners(&srv, "lead", RUNE)[0].1, "graduated");
    let mut r = Xoshiro::new(77);
    for _ in 0..50 {
        let st = state_text(&mut r, false);
        assert_eq!(post(srv.addr, &request(RUNE, &st, questions())).h("x-understudy"), Some("lead"));
    }
}
