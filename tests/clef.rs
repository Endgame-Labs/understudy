//! The Clef adapter against a local fake Cloudflare Workers AI (no network, no paid calls, no real account id): the
//! envelope's unwrap, failures, Retry-After, the size limit, auth, routing by model, and a shadow-mode `serve` route.

use serde_json::{Value, json};
use std::io::Read;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use understudy::clef::{self, ClefError, ClefSpec};
use understudy::config::Config;
use understudy::encoder::{Encoder, Featurizer, NoEncoder};
use understudy::proxy::{Proxy, Settings};
use understudy::serve::{ProxyFactory, Route, ServeArgs, ServeConfig, ServeMode, Server, http, start};
use understudy::understudent::UnderstudentKind;

const ACCOUNT: &str = "test-account";

#[derive(Clone, Debug)]
struct Seen {
    path: String,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
}

impl Seen {
    fn h(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn reply(status: u16, headers: &[(&str, &str)], body: impl Into<Vec<u8>>) -> Reply {
    Reply { status, headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body: body.into() }
}

struct Fake {
    addr: SocketAddr,
    seen: Arc<Mutex<Vec<Seen>>>,
    _listener: http::Listener,
}

impl Fake {
    fn start(h: impl Fn(&Seen) -> Reply + Send + Sync + 'static) -> Fake {
        let seen = Arc::new(Mutex::new(vec![]));
        let sn = seen.clone();
        let handler: http::Handler = Arc::new(move |rq: http::Request| {
            let headers = rq.headers.iter().map(|(k, v)| (k.to_ascii_lowercase(), v.clone())).collect();
            let s = Seen { path: rq.target, body: rq.body, headers };
            sn.lock().unwrap().push(s.clone());
            let r = h(&s);
            (http::Response { status: r.status, headers: r.headers, body: r.body, stream: None }, None)
        });
        let l = http::listen("127.0.0.1:0", 1000, http::Limits::new(64 << 20), handler).unwrap();
        Fake { addr: l.addr, seen, _listener: l }
    }

    fn spec(&self) -> ClefSpec {
        ClefSpec { base: format!("http://{}", self.addr), account: Some(ACCOUNT.into()) }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

/// A recorded Workers AI success: the System One answers inside the envelope.
fn envelope(model: &str, noul: f64) -> Vec<u8> {
    let b = json!({
        "result": {"model": model, "answers": {"urgent": {"type": "noul", "noul": noul}}, "usage": {"input_tokens": 41, "output_tokens": 3}},
        "success": true, "errors": [], "messages": []
    });
    serde_json::to_vec(&b).unwrap()
}

fn qs() -> Value {
    json!({"urgent": {"type": "noul", "instructions": "Is this support ticket urgent?"}})
}

fn body(model: &str, state: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"model": model, "state": state, "questions": qs()})).unwrap()
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().redirects(0).timeout(Duration::from_secs(20)).build()
}

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

fn post(srv: &Server, body: &[u8], extra: &[(&str, &str)]) -> Out {
    let mut rq = agent().post(&format!("http://{}/v1/systemone", srv.addr)).set("content-type", "application/json");
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

fn factory() -> ProxyFactory {
    Arc::new(|_dir, lead| {
        let enc: Arc<dyn Encoder> = Arc::new(NoEncoder);
        let mut s = Settings::default();
        s.fit.understudent = UnderstudentKind::Keywords;
        s.fit.min_examples = 40;
        s.audit_rate = 0.0;
        let fz = Featurizer::for_understudent(UnderstudentKind::Keywords, None, None, None, enc.hidden())?;
        Ok(Proxy::new(s, fz, enc, Some(lead)))
    })
}

fn clef_route(up: &Fake, bearer: Option<&str>) -> Route {
    let spec = up.spec();
    Route { name: "clef".into(), pattern: "clef*".into(), url: spec.display_url(), bearer: bearer.map(String::from), chat: None, clef: Some(spec) }
}

fn server(mode: ServeMode, route: Route, tweak: impl FnOnce(&mut ServeConfig)) -> Server {
    let mut c = ServeConfig::new(factory());
    c.bind = "127.0.0.1:0".into();
    c.mode = mode;
    c.routes = vec![route];
    c.audit_rate = 0.0;
    c.timeout = Duration::from_secs(10);
    tweak(&mut c);
    start(c).unwrap()
}

// ---- the adapter ---------------------------------------------------------------------------------------------------

#[test]
fn success_is_unwrapped_to_a_system_one_body() {
    let up = Fake::start(|_| reply(200, &[("content-type", "application/json")], envelope("clef-flash", 0.8735)));
    let url = up.spec().url("clef-flash").unwrap();
    let sent = body("clef-flash", "locked out of my account");
    let got = clef::call(&agent(), &url, None, &sent, 1 << 20).unwrap();
    let v: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(v["model"], "clef-flash");
    assert_eq!(v["answers"]["urgent"]["noul"], 0.8735);
    assert!(v.get("success").is_none() && v.get("result").is_none(), "{v}");
    let s = &up.seen()[0];
    assert_eq!(s.path, format!("/client/v4/accounts/{ACCOUNT}/ai/run/@cf/cloudflare/clef-flash"));
    assert_eq!(s.body, sent, "the request body goes unchanged");
}

#[test]
fn failures_keep_cloudflares_status_retry_after_and_messages() {
    let up = Fake::start(|s| match s.path.rsplit('/').next().unwrap() {
        "clef" => reply(200, &[], br#"{"result":null,"success":false,"errors":[{"code":7003,"message":"model unavailable"}],"messages":[]}"#.to_vec()),
        _ => reply(429, &[("retry-after", "7")], br#"{"result":null,"success":false,"errors":[{"code":3040,"message":"capacity temporarily exceeded"}],"messages":[]}"#.to_vec()),
    });
    // success:false on a 200: a 502 carrying the message
    match clef::call(&agent(), &up.spec().url("clef").unwrap(), None, &body("clef", "x"), 1 << 20) {
        Err(ClefError::Http { status: 502, body, .. }) => assert!(String::from_utf8_lossy(&body).contains("model unavailable")),
        other => panic!("{other:?}"),
    }
    // a 429: the status, Retry-After and message are kept (through serve, as a client sees them)
    let srv = server(ServeMode::Forward, clef_route(&up, None), |_| {});
    let out = post(&srv, &body("clef-flash", "x"), &[]);
    assert_eq!((out.status, out.h("retry-after"), out.h("content-type")), (429, Some("7"), Some("application/json")));
    let v = out.json();
    assert!(v["error"]["message"].as_str().unwrap().contains("capacity temporarily exceeded"), "{v}");
    let out = post(&srv, &body("clef", "x"), &[]);
    assert_eq!(out.status, 502);
    assert!(out.json()["error"]["message"].as_str().unwrap().contains("model unavailable"));
}

#[test]
fn an_oversized_response_is_an_error_not_a_truncation() {
    let big = {
        let b = json!({"result": {"model": "clef", "answers": {}, "pad": "p".repeat(5000)}, "success": true, "errors": [], "messages": []});
        serde_json::to_vec(&b).unwrap()
    };
    let up = Fake::start(move |_| reply(200, &[], big.clone()));
    let url = up.spec().url("clef").unwrap();
    assert!(matches!(clef::call(&agent(), &url, None, &body("clef", "x"), 1000), Err(ClefError::TooLarge(1000))));
    assert!(clef::call(&agent(), &url, None, &body("clef", "x"), 1 << 20).is_ok());
    let srv = server(ServeMode::Forward, clef_route(&up, None), |c| c.max_response = 1000);
    let out = post(&srv, &body("clef", "x"), &[]);
    assert_eq!(out.status, 502);
    assert!(String::from_utf8_lossy(&out.body).contains("limit"), "{}", String::from_utf8_lossy(&out.body));
}

#[test]
fn the_callers_authorization_passes_through_unless_a_key_is_configured() {
    let up = Fake::start(|_| reply(200, &[], envelope("clef", 0.5)));
    let pass = server(ServeMode::Forward, clef_route(&up, None), |_| {});
    post(&pass, &body("clef", "x"), &[("authorization", "Bearer caller-token")]);
    post(&pass, &body("clef", "x"), &[]);
    let keyed = server(ServeMode::Forward, clef_route(&up, Some("route-token")), |_| {});
    post(&keyed, &body("clef", "x"), &[("authorization", "Bearer caller-token")]);
    let seen = up.seen();
    assert_eq!(seen[0].h("authorization"), Some("Bearer caller-token"));
    assert_eq!(seen[1].h("authorization"), None);
    assert_eq!(seen[2].h("authorization"), Some("Bearer route-token"));
}

#[test]
fn models_route_to_their_own_paths() {
    let up = Fake::start(|s| {
        let m = s.path.rsplit('/').next().unwrap().to_string();
        reply(200, &[], envelope(&m, 0.5))
    });
    let srv = server(ServeMode::Forward, clef_route(&up, None), |_| {});
    for (model, path_model) in [("clef", "clef"), ("clef-flash", "clef-flash"), ("@cf/cloudflare/clef-flash", "clef-flash"), ("@cf/cloudflare/clef", "clef")] {
        let out = post(&srv, &body(model, "x"), &[]);
        assert_eq!(out.status, 200, "{model}");
        assert_eq!(out.json()["model"], path_model);
        assert_eq!(out.h("content-type"), Some("application/json"));
    }
    // Cloudflare takes only `clef` or `clef-flash` as the body's model, so a prefixed name is rewritten there
    let bodies: Vec<Value> = up.seen().iter().map(|s| serde_json::from_slice(&s.body).unwrap()).collect();
    assert_eq!(bodies.iter().map(|b| b["model"].as_str().unwrap()).collect::<Vec<_>>(), ["clef", "clef-flash", "clef-flash", "clef"]);
    let paths: Vec<_> = up.seen().iter().map(|s| s.path.clone()).collect();
    assert_eq!(paths[2], format!("/client/v4/accounts/{ACCOUNT}/ai/run/@cf/cloudflare/clef-flash"));
    // a model no route matches is not sent anywhere
    let n = up.seen().len();
    assert_eq!(post(&srv, &body("other-model", "x"), &[]).status, 400);
    assert_eq!(up.seen().len(), n);
}

#[test]
fn config_builds_clef_routes_and_a_missing_account_is_a_clear_error() {
    #[derive(clap::Parser)]
    struct W {
        #[command(flatten)]
        a: ServeArgs,
    }
    use clap::Parser;
    // SAFETY: this is the only test in this binary that touches the process environment
    unsafe { std::env::remove_var(clef::ACCOUNT_ENV) };
    let no_account = Config::parse("[leads.cf]\nkind = \"clef\"\nmodel = \"clef-flash\"\n").unwrap();
    let w = W::parse_from(["x"]);
    let e = w.a.config_with(&no_account, 0.1, factory()).err().expect("no account").to_string();
    assert!(e.contains("account") && e.contains(clef::ACCOUNT_ENV), "{e}");
    let with = Config::parse("[leads.cf]\nkind = \"clef\"\nurl = \"http://localhost:9/\"\naccount = \"acct-x\"\nroute = \"clef*\"\nkey_env = \"UNDERSTUDY_CLEF_TEST_UNSET\"\n").unwrap();
    let c = w.a.config_with(&with, 0.1, factory()).unwrap();
    let r = &c.routes[0];
    assert!(r.matches("clef") && r.matches("@cf/cloudflare/clef-flash") && !r.matches("other"));
    assert_eq!(r.clef.as_ref().unwrap().url("clef").unwrap(), "http://localhost:9/client/v4/accounts/acct-x/ai/run/@cf/cloudflare/clef");
    assert!(!r.url.contains("acct-x"), "the display URL carries no account id: {}", r.url);
    // the flag form
    let w = W::parse_from(["x", "--clef-route", "clef*"]);
    assert!(w.a.config_with(&Config::default(), 0.1, factory()).is_err());
    unsafe { std::env::set_var(clef::ACCOUNT_ENV, "env-acct") };
    let c = w.a.config_with(&Config::default(), 0.1, factory()).unwrap();
    assert_eq!(c.routes[0].clef.as_ref().unwrap().account.as_deref(), Some("env-acct"));
    unsafe { std::env::remove_var(clef::ACCOUNT_ENV) };
}

// ---- serve ---------------------------------------------------------------------------------------------------------

#[test]
fn shadow_mode_serves_the_system_one_body_and_learns_from_it() {
    let n = Arc::new(AtomicUsize::new(0));
    let count = n.clone();
    let up = Fake::start(move |s| {
        count.fetch_add(1, Ordering::Relaxed);
        let v: Value = serde_json::from_slice(&s.body).unwrap();
        let urgent = v["state"].as_str().unwrap_or("").contains("error");
        reply(200, &[("content-type", "application/json")], envelope("clef-flash", if urgent { 0.95 } else { 0.05 }))
    });
    let srv = server(ServeMode::Shadow, clef_route(&up, None), |_| {});
    for i in 0..60 {
        let state = if i % 3 == 0 { format!("error number {i} in the app") } else { format!("all fine number {i}") };
        let out = post(&srv, &body("clef-flash", &state), &[]);
        assert_eq!(out.status, 200);
        assert_eq!(out.h("x-understudy"), Some("lead"));
        let v = out.json();
        assert_eq!(v["model"], "clef-flash");
        assert!(v["answers"]["urgent"]["noul"].is_number() && v.get("result").is_none(), "{v}");
    }
    assert!(srv.wait_idle(Duration::from_secs(30)));
    assert_eq!(n.load(Ordering::Relaxed), 60);
    let st = srv.status();
    assert_eq!((st["lead"].as_u64(), st["local"].as_u64()), (Some(60), Some(0)), "shadow answers nothing locally");
    assert_eq!(st["learned"], 60);
    assert_eq!(st["namespaces"][0]["route"], "clef");
    let id = st["namespaces"][0]["identity"].to_string();
    assert!(!id.contains(ACCOUNT), "{id}");
}
