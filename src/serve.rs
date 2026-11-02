//! `understudy serve`: a drop-in `POST /v1/systemone` HTTP proxy (SPEC.md, "serve").
//!
//! The hot path is a small thread-per-connection HTTP/1.1 server over `std::net` (`http` below: keep-alive, chunked
//! request bodies, `Expect: 100-continue`; `tiny_http` was tried and starves new connections behind idle keep-alive
//! ones, which is exactly how a client with a pool of keep-alive connections behaves). A request is parsed only to route it by `model` and (for the
//! capture log and `serve` mode) to read its questions; the ORIGINAL bytes go to the upstream and the upstream's
//! status, body bytes, content type and `Retry-After` come back unchanged. Learning never runs on that path: every
//! successful lead answer becomes a job on a bounded queue (dropped and counted when it is full), and one worker
//! thread per upstream/model namespace owns that namespace's `Proxy` (its own mutex, its own snapshot directory), so
//! a refit of one namespace stalls neither the HTTP threads nor the other namespaces.
//!
//! Modes: `forward` (pass-through and capture), `shadow` (forward, learn, log what the understudent would have said)
//! and `serve` (answer locally when every question is graduated and eligible, else forward as in shadow).
//!
//! Learner identity is `namespace + question spec`: a learner is named by its spec hash and the request's own
//! names are kept apart (`QInfo::name`), so `answers` keys always match the submitted names exactly, identical
//! specs under different names share one learner, and a question the learners don't parse (an unknown `type`) is
//! opaque: forwarded, never learned.

use crate::encoder::state_hash;
use crate::chat::{self, ChatError, ChatSpec};
use crate::config::{self, Config, LeadKind};
use crate::lead::Lead;
use crate::proxy::Proxy;
use crate::question::{Question, hex};
use crate::understudent::Pred;
use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, ValueEnum};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, RandomState};
use std::io::{BufWriter, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ServeMode {
    /// Pure pass-through plus capture.
    Forward,
    /// Forward everything and return the lead's response; learn from it and log the understudent's would-be answers.
    Shadow,
    /// Answer locally when every question is graduated and eligible; otherwise forward as in shadow.
    Serve,
    /// Send every request to the lead at once and compute the local answer in parallel; the first usable one wins.
    Race,
}

impl ServeMode {
    pub fn name(self) -> &'static str {
        match self {
            ServeMode::Forward => "forward",
            ServeMode::Shadow => "shadow",
            ServeMode::Serve => "serve",
            ServeMode::Race => "race",
        }
    }
}

/// The command-line half of the configuration (the understudent flags come from `main.rs`'s `UnderstudentOpts`).
#[derive(Args, Clone, Debug)]
pub struct ServeArgs {
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:8787")]
    pub bind: String,
    /// forward, shadow, serve or race.
    #[arg(long, value_enum, default_value = "shadow")]
    pub mode: ServeMode,
    /// An extra System One route `MODEL=URL`; MODEL is an exact model name or a prefix ending in `*`. Repeatable;
    /// tried before the routes from the lead config file.
    #[arg(long = "route", value_name = "MODEL=URL")]
    pub routes: Vec<String>,
    /// An extra chat route `MODEL=URL`: callers still speak System One, and understudy turns each request (and each
    /// audit) into OpenAI-compatible chat calls to URL (a base URL or the full /chat/completions URL), sending
    /// the request's model. Repeatable. For a different upstream model, or tuning, use a `kind = "chat"` lead in the
    /// config file.
    #[arg(long = "chat-route", value_name = "MODEL=URL")]
    pub chat_routes: Vec<String>,
    /// Send `Authorization: Bearer $ENV` upstream for a route: `NAME_OR_MODEL=ENV_VAR` (the NAME is a lead's name in
    /// the config file). The key is read from the environment only, never from a file or argv. An endpoint that
    /// needs none takes none. Without a key, the client's own `Authorization` header is forwarded.
    #[arg(long = "route-key-env", value_name = "ROUTE=ENV_VAR")]
    pub route_key_env: Vec<String>,
    /// A catch-all System One lead for any model, such as `https://their-lead/v1/systemone`; needs no config file. Routes
    /// from the config file and `--route` are tried first. The caller's `Authorization` header goes to it unless a route
    /// key is set (`--route-key-env lead=VAR`).
    #[arg(long, value_name = "URL")]
    pub lead_url: Option<String>,
    /// Ignore the routes of the lead config file (use --route, --chat-route and --lead-url only).
    #[arg(long)]
    pub no_default_routes: bool,
    /// Persist each upstream/model's learners (one snapshot per namespace) under this directory, and reload them at
    /// the next start.
    #[arg(long, default_value = "understudy-snapshots")]
    pub snapshot_dir: Option<PathBuf>,
    /// Keep learning in memory only (nothing is saved or reloaded).
    #[arg(long)]
    pub no_snapshots: bool,
    /// Do not download the embedding model in the background (`--understudent auto` then races the model-free designs
    /// until you run `understudy model fetch`).
    #[arg(long)]
    pub no_model_download: bool,
    /// The capture log (JSONL, one line per request).
    #[arg(long, default_value = "understudy-capture.jsonl")]
    pub capture: PathBuf,
    /// Write no capture log.
    #[arg(long)]
    pub no_capture: bool,
    /// Put the full request and response bodies in the capture log (for building replay datasets).
    #[arg(long)]
    pub capture_bodies: bool,
    /// Most concurrent client connections (one thread each); further ones get a 503.
    #[arg(long, default_value_t = 512)]
    pub max_conns: usize,
    /// Learning queue length per namespace; when full, jobs are dropped (and counted), never waited for.
    #[arg(long, default_value_t = 1024)]
    pub queue: usize,
    /// Upstream request timeout in seconds. The proxy never retries: callers do.
    #[arg(long, default_value_t = 120)]
    pub timeout_secs: u64,
    /// Save snapshots every this many learned requests (and when a worker has been idle).
    #[arg(long, default_value_t = 200)]
    pub save_every: usize,
    /// `serve`: how long a request waits for a namespace's lock (held by learning) before it is forwarded, in ms.
    #[arg(long, default_value_t = 5)]
    pub local_wait_ms: u64,
    /// `serve`: concurrent background audit calls to the lead (extra calls are skipped).
    #[arg(long, default_value_t = 4)]
    pub max_audits: usize,
    /// `race`: most lead calls in flight at once (each call is started before the local answer is known and, when
    /// the local answer wins, finishes in the background as an audit). Past it a request behaves as in `serve`.
    #[arg(long, default_value_t = 64)]
    pub race_max_inflight: usize,
    /// Largest request body accepted, in MiB.
    #[arg(long, default_value_t = 64)]
    pub max_body_mib: usize,
    /// Largest upstream response relayed, in MiB. A larger one is a 502 (and a capture-log note), never truncated.
    #[arg(long, default_value_t = 256)]
    pub max_response_mib: usize,
    /// Most upstream/model namespaces (each a worker thread and a snapshot); beyond it a request is forward-only.
    #[arg(long, default_value_t = 16)]
    pub max_namespaces: usize,
    /// Most learners (distinct question specs) per namespace; beyond it a new spec is forwarded and never learned.
    #[arg(long, default_value_t = 256)]
    pub max_learners: usize,
    /// Total time a client has to send one request's headers and body, in seconds.
    #[arg(long, default_value_t = 60)]
    pub read_deadline_secs: u64,
    /// Total time a client has to take one response, in seconds.
    #[arg(long, default_value_t = 30)]
    pub write_timeout_secs: u64,
}

/// A way to build a namespace's `Proxy` (a fresh one, or the snapshot in the given directory) around the stash lead.
pub type ProxyFactory = Arc<dyn Fn(Option<&Path>, Box<dyn Lead>) -> Result<Proxy> + Send + Sync>;

#[derive(Clone)]
pub struct Route {
    /// Names the upstream in namespaces and the capture log (`rune`, `jev`, ...).
    pub name: String,
    /// Exact model, or a prefix ending in `*`.
    pub pattern: String,
    pub url: String,
    /// Sent as `Authorization: Bearer`; read from the environment by `ServeArgs::config`, never printed.
    pub bearer: Option<String>,
    /// A chat route: requests are translated to chat calls (`chat.rs`) and answered in System One. `url` is then
    /// the chat endpoint, and the spec's model, if any, replaces the request's.
    pub chat: Option<ChatSpec>,
}

impl Route {
    pub fn matches(&self, model: &str) -> bool {
        match self.pattern.strip_suffix('*') {
            Some(prefix) => model.starts_with(prefix),
            None => model == self.pattern,
        }
    }
}

pub struct ServeConfig {
    pub bind: String,
    pub mode: ServeMode,
    pub routes: Vec<Route>,
    pub snapshot_dir: Option<PathBuf>,
    pub capture: Option<PathBuf>,
    pub capture_bodies: bool,
    pub max_conns: usize,
    pub queue: usize,
    pub timeout: Duration,
    pub save_every: usize,
    pub local_wait: Duration,
    /// How long shutdown waits for outstanding race audits (late lead calls) before abandoning them.
    pub shutdown_drain: Duration,
    /// `serve`: the share of local answers also sent to the lead in the background to audit them.
    pub audit_rate: f64,
    pub max_audits: usize,
    /// `race`: lead calls in flight at once, before a request falls back to `serve` behaviour.
    pub race_max_inflight: usize,
    pub max_body: usize,
    pub max_response: usize,
    pub max_namespaces: usize,
    pub max_learners: usize,
    pub read_deadline: Duration,
    pub write_timeout: Duration,
    /// Post-response work queued for the helper threads before it is dropped (and counted).
    pub post_queue: usize,
    pub factory: ProxyFactory,
    /// Called under a namespace's lock before each job and when idle: lets an `auto` proxy grow once the encoder is
    /// ready. True when it changed the proxy.
    pub upgrade: Option<Upgrade>,
}

/// See `ServeConfig::upgrade`.
pub type Upgrade = Arc<dyn Fn(&mut Proxy) -> Result<bool> + Send + Sync>;

impl ServeConfig {
    /// Defaults (shadow, loopback port 8787, built-in routes, no capture file) around a proxy factory.
    pub fn new(factory: ProxyFactory) -> Self {
        ServeConfig {
            bind: "127.0.0.1:8787".into(),
            mode: ServeMode::Shadow,
            routes: vec![],
            snapshot_dir: None,
            capture: None,
            capture_bodies: false,
            max_conns: 512,
            queue: 1024,
            timeout: Duration::from_secs(120),
            save_every: 200,
            local_wait: Duration::from_millis(5),
            shutdown_drain: Duration::from_secs(30),
            audit_rate: 0.1,
            max_audits: 4,
            race_max_inflight: 64,
            max_body: 64 << 20,
            max_response: 256 << 20,
            max_namespaces: 16,
            max_learners: 256,
            read_deadline: Duration::from_secs(60),
            write_timeout: Duration::from_secs(30),
            post_queue: 8192,
            factory,
            upgrade: None,
        }
    }
}

/// The routes the lead config file defines: every lead with a `route` or a `model`. The route's name is the
/// lead's, its bearer key the lead's `key_env` (a warning, not an error, when that variable is unset).
pub fn routes_from_config(cfg: &Config) -> Vec<Route> {
    let mut out = vec![];
    for (name, e) in &cfg.leads {
        let Some(pattern) = e.route.clone().or_else(|| e.model.clone()) else { continue };
        let bearer = e.key_env.as_deref().and_then(|v| match config::key_from_env(v) {
            Ok(k) => k,
            Err(err) => {
                eprintln!("understudy serve: route {name}: {err:#}; sending no key");
                None
            }
        });
        let chat = (e.kind == LeadKind::Chat).then(|| ChatSpec::from_entry(e));
        out.push(Route { name: name.clone(), pattern, url: if chat.is_some() { chat::endpoint(&e.url) } else { e.url.clone() }, bearer, chat });
    }
    out
}

impl ServeArgs {
    /// The configuration these flags mean, with the routes of the lead config file (`$UNDERSTUDY_LEADS` or
    /// `~/.config/understudy/leads.toml`). Bearer keys come from the environment variables named by
    /// `--route-key-env` (empty or `implicit` means none).
    pub fn config(&self, audit_rate: f64, factory: ProxyFactory) -> Result<ServeConfig> {
        let file = if self.no_default_routes { Config::default() } else { Config::load()? };
        self.config_with(&file, audit_rate, factory)
    }

    /// [`ServeArgs::config`] over an explicit lead config.
    pub fn config_with(&self, file: &Config, audit_rate: f64, factory: ProxyFactory) -> Result<ServeConfig> {
        let mut routes = vec![];
        for r in &self.routes {
            let (pattern, url) = r.split_once('=').ok_or_else(|| anyhow!("--route wants MODEL=URL, got {r:?}"))?;
            routes.push(Route { name: slug(pattern), pattern: pattern.into(), url: url.into(), bearer: None, chat: None });
        }
        for r in &self.chat_routes {
            let (pattern, url) = r.split_once('=').ok_or_else(|| anyhow!("--chat-route wants MODEL=URL, got {r:?}"))?;
            let spec = ChatSpec::new(url);
            routes.push(Route { name: slug(pattern), pattern: pattern.into(), url: spec.url.clone(), bearer: None, chat: Some(spec) });
        }
        if !self.no_default_routes {
            routes.extend(routes_from_config(file));
        }
        if let Some(u) = &self.lead_url {
            routes.push(Route { name: "lead".into(), pattern: "*".into(), url: u.clone(), bearer: None, chat: None });
        }
        for k in &self.route_key_env {
            let (which, var) = k.split_once('=').ok_or_else(|| anyhow!("--route-key-env wants ROUTE=ENV_VAR, got {k:?}"))?;
            let r = routes.iter_mut().find(|r| r.name == which || r.pattern == which).ok_or_else(|| anyhow!("--route-key-env: no route {which:?}"))?;
            r.bearer = std::env::var(var).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v != "implicit");
        }
        if routes.is_empty() {
            bail!("no routes: pass --lead-url URL (or --route MODEL=URL, or --chat-route), or define leads in the lead config file ($UNDERSTUDY_LEADS or ~/.config/understudy/leads.toml; see examples/leads.example.toml)");
        }
        let mut c = ServeConfig::new(factory);
        c.bind = self.bind.clone();
        c.mode = self.mode;
        c.routes = routes;
        c.snapshot_dir = if self.no_snapshots { None } else { self.snapshot_dir.clone() };
        c.capture = (!self.no_capture).then(|| self.capture.clone());
        c.capture_bodies = self.capture_bodies;
        c.max_conns = self.max_conns.max(1);
        c.queue = self.queue.max(1);
        c.timeout = Duration::from_secs(self.timeout_secs);
        c.save_every = self.save_every.max(1);
        c.local_wait = Duration::from_millis(self.local_wait_ms);
        c.audit_rate = audit_rate;
        c.max_audits = self.max_audits;
        c.race_max_inflight = self.race_max_inflight.max(1);
        c.max_body = self.max_body_mib.max(1) << 20;
        c.max_response = self.max_response_mib.max(1) << 20;
        c.max_namespaces = self.max_namespaces;
        c.max_learners = self.max_learners;
        c.read_deadline = Duration::from_secs(self.read_deadline_secs.max(1));
        c.write_timeout = Duration::from_secs(self.write_timeout_secs.max(1));
        Ok(c)
    }
}

fn slug(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

// ---- shared state --------------------------------------------------------------------------------------------

#[derive(Default)]
struct Stats {
    requests: AtomicU64,
    local: AtomicU64,
    lead: AtomicU64,
    upstream_errors: AtomicU64,
    rejected: AtomicU64,
    queued: AtomicU64,
    dropped: AtomicU64,
    learned: AtomicU64,
    learn_errors: AtomicU64,
    audits: AtomicU64,
    audits_skipped: AtomicU64,
    /// `race`: requests answered locally while the lead call finished in the background.
    race_local: AtomicU64,
    /// `race`: requests answered with the lead's response.
    race_lead: AtomicU64,
    /// `race`: requests past `--race-max-inflight`, handled as in `serve`.
    race_capped: AtomicU64,
    local_busy: AtomicU64,
    /// A namespace was wanted but `--max-namespaces` was reached: the request stayed forward-only.
    ns_refused: AtomicU64,
    /// A question spec was wanted as a learner but `--max-learners` was reached: it was forwarded, never learned.
    learners_refused: AtomicU64,
    /// Post-response work (capture logging, queueing) dropped because the helpers' queue was full.
    post_dropped: AtomicU64,
    /// Upstream responses over `--max-response-mib` (answered 502, never truncated).
    response_too_large: AtomicU64,
}

/// What the stash lead answers: the upstream's answers for the request being learned, by learner name.
struct Stash(Arc<Mutex<Map<String, Value>>>, Value);

impl Lead for Stash {
    fn ask(&self, _id: Option<&str>, _state: &Value, _questions: &Map<String, Value>) -> Result<Map<String, Value>> {
        Ok(self.0.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    /// Recorded in the saved snapshot: which upstream this namespace's learners learned from.
    fn describe(&self) -> Value {
        json!({"kind": "serve", "namespace": self.1})
    }
}

/// A URL with its scheme and host lowercased, a default port, fragment and trailing slashes dropped, so equal
/// upstreams compare equal.
fn normalize_url(u: &str) -> String {
    let u = u.trim();
    let (scheme, rest) = u.split_once("://").unwrap_or(("", u));
    let rest = rest.split('#').next().unwrap_or("");
    let (auth, tail) = match rest.find(['/', '?']) {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };
    let scheme = scheme.to_ascii_lowercase();
    let mut auth = auth.to_ascii_lowercase();
    let default = match scheme.as_str() {
        "http" => ":80",
        "https" => ":443",
        _ => "\0",
    };
    if let Some(a) = auth.strip_suffix(default) {
        auth = a.to_string();
    }
    let (path, query) = match tail.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (tail, None),
    };
    let path = path.trim_end_matches('/');
    format!("{scheme}://{auth}{path}{}", query.map(|q| format!("?{q}")).unwrap_or_default())
}

/// What makes two namespaces the same one: route, model, normalized upstream URL and a hash of the route's
/// configuration (pattern, URL, whether it sends its own bearer key; never the key).
fn identity(route: &Route, model: &str) -> Value {
    let url = normalize_url(&route.url);
    let mut cfg = json!({"name": route.name, "pattern": route.pattern, "url": url, "bearer": route.bearer.is_some()});
    if let Some(c) = &route.chat {
        // a chat route answers differently from a System One one with the same URL; the key is absent for the rest
        cfg["chat"] = json!({"model": c.model, "reasoning_effort": c.reasoning_effort, "json_schema": c.json_schema, "logprobs": c.logprobs, "confidence": c.confidence});
    }
    let config_hash = hex(&Sha256::digest(serde_json::to_string(&cfg).unwrap_or_default().as_bytes()))[..16].to_string();
    json!({"route": route.name, "model": model, "upstream": url, "config_hash": config_hash})
}

fn identity_key(id: &Value) -> String {
    ["route", "model", "upstream", "config_hash"].iter().map(|k| id[k].as_str().unwrap_or("")).collect::<Vec<_>>().join("|")
}

/// One upstream/model's learners: the `Proxy` is built by the worker (loading a snapshot can take a while), so it is
/// None until then; a request that finds it busy or missing is forwarded.
struct NsCore {
    key: String,
    route: String,
    model: String,
    identity: Value,
    /// Why the namespace's proxy could not be built (e.g. a snapshot from another upstream), if it could not.
    error: Mutex<Option<String>>,
    /// Refits completed (the version of the fits serving now); bumped under the proxy lock.
    fit_version: AtomicU64,
    dir: Option<PathBuf>,
    proxy: Mutex<Option<Proxy>>,
    stash: Arc<Mutex<Map<String, Value>>>,
    /// Jobs queued or being processed.
    pending: AtomicUsize,
}

struct Namespace {
    core: Arc<NsCore>,
    tx: SyncSender<Job>,
}

struct Job {
    entry: Map<String, Value>,
    req: Arc<Value>,
    infos: Vec<QInfo>,
    status: u16,
    body: Vec<u8>,
    group: Option<String>,
    /// An audit of a local answer: what was served, then.
    served: Option<Served>,
}

/// The prediction a local answer was served from, kept so its audit compares against it, not against whatever the
/// learner says by the time the lead has answered.
#[derive(Clone)]
struct Served {
    fit_version: u64,
    per: Vec<ServedQ>,
}

#[derive(Clone)]
struct ServedQ {
    spec_hash: String,
    tier: Option<usize>,
    label: usize,
}

struct Shared {
    cfg: ServeConfig,
    agent: ureq::Agent,
    nss: Mutex<HashMap<String, Arc<Namespace>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    log: Option<Mutex<BufWriter<std::fs::File>>>,
    stats: Stats,
    audits_inflight: AtomicUsize,
    /// `race`: lead calls started and not yet finished.
    race_inflight: AtomicUsize,
    seq: AtomicU64,
    rand: RandomState,
    /// Post-response work (queueing, capture logging) handed to a couple of helper threads, so a client reusing its
    /// connection never waits for it.
    post: Mutex<Option<SyncSender<After>>>,
    /// Post-response work not yet finished.
    active: Arc<AtomicUsize>,
}

impl Shared {
    fn log(&self, entry: Map<String, Value>) {
        let Some(log) = &self.log else { return };
        let mut line = serde_json::to_vec(&Value::Object(entry)).unwrap_or_default();
        line.push(b'\n');
        let mut w = log.lock().unwrap_or_else(|e| e.into_inner());
        if w.write_all(&line).and_then(|_| w.flush()).is_err() {
            eprintln!("understudy: capture log write failed");
        }
    }

    /// Run `work` on a helper thread, never on the caller's: when the helpers' queue is full (or they are gone) the
    /// work is dropped and counted in `post_dropped`. The caller has already counted it in `active` (when the handler
    /// made it, before the response was written), so `wait_idle` never sees a gap between a response reaching its
    /// client and its post-response work being registered; this releases the count when the work finishes or is dropped.
    fn post_work(&self, work: After) {
        let tx = self.post.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let active = self.active.clone();
        let job: After = Box::new(move || {
            work();
            active.fetch_sub(1, Relaxed);
        });
        let sent = tx.is_some_and(|t| t.try_send(job).is_ok());
        if !sent {
            self.active.fetch_sub(1, Relaxed);
            self.stats.post_dropped.fetch_add(1, Relaxed);
        }
    }

    fn rand01(&self) -> f64 {
        (self.rand.hash_one(self.seq.fetch_add(1, Relaxed)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn route(&self, model: &str) -> Option<&Route> {
        self.cfg.routes.iter().find(|r| r.matches(model))
    }

    /// The namespace of a route/model, if it exists.
    fn find_namespace(&self, route: &Route, model: &str) -> Option<Arc<Namespace>> {
        let key = identity_key(&identity(route, model));
        self.nss.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned()
    }

    /// The namespace of a route/model, created (with its worker) on first use. None when `--max-namespaces` is
    /// reached (counted): the request then stays forward-only. Only a successful upstream answer for the model
    /// should get here.
    fn namespace(self: &Arc<Self>, route: &Route, model: &str) -> Option<Arc<Namespace>> {
        let id = identity(route, model);
        let key = identity_key(&id);
        let mut map = self.nss.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ns) = map.get(&key) {
            return Some(ns.clone());
        }
        if map.len() >= self.cfg.max_namespaces {
            self.stats.ns_refused.fetch_add(1, Relaxed);
            return None;
        }
        // the directory names route and model only: a changed upstream finds the old snapshot and is refused by the
        // worker, instead of quietly starting a second one
        let dir = self.cfg.snapshot_dir.as_ref().map(|d| d.join(format!("{}__{}-{}", slug(&route.name), slug(model), &hex(&Sha256::digest(format!("{}|{model}", route.name).as_bytes()))[..8])));
        let core = Arc::new(NsCore {
            key: key.clone(),
            route: route.name.clone(),
            model: model.to_string(),
            identity: id,
            error: Mutex::new(None),
            fit_version: AtomicU64::new(0),
            dir,
            proxy: Mutex::new(None),
            stash: Arc::new(Mutex::new(Map::new())),
            pending: AtomicUsize::new(0),
        });
        let (tx, rx) = sync_channel(self.cfg.queue);
        let (sh, c) = (self.clone(), core.clone());
        let h = std::thread::Builder::new().name(format!("understudy-learn-{}", route.name)).spawn(move || worker(sh, c, rx)).expect("spawn worker");
        self.workers.lock().unwrap_or_else(|e| e.into_inner()).push(h);
        let ns = Arc::new(Namespace { core, tx });
        map.insert(key, ns.clone());
        Some(ns)
    }

    /// Queue a learning job; when the queue is full (or the worker is gone) drop it, count it and log it now.
    fn enqueue(&self, ns: &Namespace, mut job: Job) {
        ns.core.pending.fetch_add(1, Relaxed);
        match ns.tx.try_send(job) {
            Ok(()) => {
                self.stats.queued.fetch_add(1, Relaxed);
            }
            Err(TrySendError::Full(j) | TrySendError::Disconnected(j)) => {
                ns.core.pending.fetch_sub(1, Relaxed);
                self.stats.dropped.fetch_add(1, Relaxed);
                job = j;
                job.entry.insert("learning".into(), json!("dropped"));
                job.entry.insert("questions".into(), json!(basic_questions(&job.infos)));
                self.log(job.entry);
            }
        }
    }
}

// ---- questions of a request ---------------------------------------------------------------------------------

/// A request question: its submitted name, its spec hash (the learner's identity within a namespace) and the parsed
/// question, or None when the learners don't understand it (it is then opaque: forwarded, never learned).
#[derive(Clone)]
struct QInfo {
    name: String,
    spec_hash: String,
    qtype: String,
    question: Option<Question>,
}

fn question_infos(req: &Value) -> Vec<QInfo> {
    let Some(qs) = req.get("questions").and_then(Value::as_object) else { return vec![] };
    qs.iter()
        .map(|(name, q)| {
            let qtype = q.get("type").and_then(Value::as_str).unwrap_or("?").to_string();
            match Question::from_wire(q, None) {
                Ok(question) => QInfo { name: name.clone(), spec_hash: question.key(), qtype, question: Some(question) },
                Err(_) => {
                    let h = Sha256::digest(serde_json::to_string(q).unwrap_or_default().as_bytes());
                    QInfo { name: name.clone(), spec_hash: hex(&h)[..16].to_string(), qtype, question: None }
                }
            }
        })
        .collect()
}

fn basic_questions(infos: &[QInfo]) -> Vec<Value> {
    infos.iter().map(|i| json!({"name": i.name, "spec_hash": i.spec_hash, "type": i.qtype, "opaque": i.question.is_none()})).collect()
}

/// A lead's answer in the form `Question::parse_answer` reads: a bare `{"choice": ...}` (no probabilities) is its label.
fn normalize(a: &Value) -> Value {
    if let Value::Object(m) = a {
        if !m.contains_key("probabilities") {
            if let Some(c @ Value::String(_)) = m.get("choice") {
                return c.clone();
            }
        }
    }
    a.clone()
}

// ---- predictions ---------------------------------------------------------------------------------------------

/// What an understudent says about one question of one request.
struct QPred {
    probs: Option<Vec<f64>>,
    eligible: bool,
    novel: bool,
    demoted: bool,
    /// Eligible and not demoted: it would answer.
    local: bool,
    mode: Option<String>,
    readout: Option<String>,
    /// The tier that predicted (the one a local answer is served from).
    tier: Option<usize>,
}

impl QPred {
    fn json(&self, q: &Question) -> Value {
        let best = self.probs.as_ref().map(|p| crate::stats::argmax(p));
        json!({
            "label": best.map(|b| &q.options[b]),
            "probs": self.probs.as_ref().map(|p| q.options.iter().zip(p).map(|(o, x)| (o.clone(), json!(x))).collect::<Map<_, _>>()),
            "eligible": self.eligible, "novel": self.novel, "demoted": self.demoted, "would_answer_local": self.local,
            "mode": self.mode, "readout": self.readout,
        })
    }
}

/// Predict learners `sis` for a state without learning or auditing: the same tier order and eligibility as
/// `Proxy::answer`. A local tier (graduated, confident, not novel, not demoted) is reported when there is one, else
/// the first eligible, else the last that predicted.
fn predict(p: &mut Proxy, state: &Value, sis: &[usize]) -> Result<Vec<QPred>> {
    let nt = p.tier_fz.len();
    let none = || QPred { probs: None, eligible: false, novel: false, demoted: false, local: false, mode: None, readout: None, tier: None };
    if sis.iter().all(|&i| p.understudents[i].lead_only.is_some()) {
        return Ok(sis.iter().map(|_| none()).collect());
    }
    let mut w = p.work(state);
    let mut preds: Vec<Vec<Option<Pred>>> = vec![vec![None; nt]; sis.len()];
    let mut chosen: Vec<Option<usize>> = vec![None; sis.len()];
    for t in 0..nt {
        let live: Vec<usize> = (0..sis.len())
            .filter(|&q| chosen[q].is_none() && p.understudents[sis[q]].tries(t) && p.understudents[sis[q]].tiers[t].serving.fit.has_model())
            .collect();
        if live.is_empty() {
            continue;
        }
        p.ensure(&mut w, &[t])?;
        for q in live {
            let s = &p.understudents[sis[q]];
            let tier = &s.tiers[t];
            let pr = tier.serving.predict(&tier.fz, &w.enc, s.question.k());
            if pr.local {
                chosen[q] = Some(t);
            }
            preds[q][t] = Some(pr);
        }
        if chosen.iter().all(Option::is_some) {
            break;
        }
    }
    Ok((0..sis.len())
        .map(|q| {
            let ti = chosen[q]
                .or_else(|| (0..nt).find(|&t| preds[q][t].as_ref().is_some_and(|p| p.eligible)))
                .or_else(|| (0..nt).rev().find(|&t| preds[q][t].is_some()));
            let (Some(ti), s) = (ti, &p.understudents[sis[q]]) else { return none() };
            let pr = preds[q][ti].as_ref().unwrap();
            let serving = &s.tiers[ti].serving;
            QPred {
                probs: pr.p.clone(),
                eligible: pr.eligible,
                novel: pr.novel,
                demoted: serving.demoted,
                local: pr.local,
                mode: serving.mode().map(|m| m.name().to_string()),
                readout: s.label(ti),
                tier: Some(ti),
            }
        })
        .collect())
}

// ---- upstream ------------------------------------------------------------------------------------------------

struct Upstream {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

/// A failed upstream call.
struct UpErr {
    msg: String,
    timed_out: bool,
    too_large: bool,
}

/// Headers that belong to one connection, not to the message (RFC 9110 7.6.1), plus `Host` and `Content-Length`,
/// which the proxy sets itself, and `Expect`, which it answers itself.
const HOP: [&str; 12] = ["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length", "expect"];

/// The header names the `Connection` header(s) nominate as hop-by-hop.
fn nominated(headers: &[(String, String)]) -> HashSet<String> {
    headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("connection")).flat_map(|(_, v)| v.split(',')).map(|t| t.trim().to_ascii_lowercase()).filter(|t| !t.is_empty()).collect()
}

/// The end-to-end headers of a message: everything except the hop-by-hop ones (fixed list and `Connection`'s).
fn end_to_end(headers: &[(String, String)]) -> Vec<(String, String)> {
    let nom = nominated(headers);
    headers.iter().filter(|(k, _)| {
        let k = k.to_ascii_lowercase();
        !HOP.contains(&k.as_str()) && !nom.contains(&k)
    }).cloned().collect()
}

/// What goes upstream with the request's original bytes: the caller's end-to-end headers (all values of repeated
/// ones), a default content type, `identity` encoding unless the caller chose one (an encoded answer can't be
/// learned from, but is relayed as it is), and the route's own bearer key in place of the caller's authorization.
fn upstream_headers(route: &Route, client: &[(String, String)]) -> Vec<(String, String)> {
    upstream_headers_with(route, client, true)
}

/// `upstream_headers`; with `defaults` false (pass-through) no header is added that the caller did not send, except the
/// route's own key.
fn upstream_headers_with(route: &Route, client: &[(String, String)], defaults: bool) -> Vec<(String, String)> {
    let mut h = end_to_end(client);
    if route.bearer.is_some() {
        h.retain(|(k, _)| !k.eq_ignore_ascii_case("authorization"));
    }
    if defaults && !h.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        h.push(("content-type".into(), "application/json".into()));
    }
    if defaults && !h.iter().any(|(k, _)| k.eq_ignore_ascii_case("accept-encoding")) {
        h.push(("accept-encoding".into(), "identity".into()));
    }
    if let Some(k) = &route.bearer {
        h.push(("authorization".into(), format!("Bearer {k}")));
    }
    // the HTTP client keeps one value per header name (except `x-*`): fold repeats into one field line, which for
    // list-valued fields is the same message (RFC 9110 5.3); cookies join with `; `
    let mut out: Vec<(String, String)> = vec![];
    for (k, v) in h {
        let k = k.to_ascii_lowercase();
        match out.iter_mut().find(|(n, _)| *n == k && !k.starts_with("x-")) {
            Some((_, old)) => {
                old.push_str(if k == "cookie" { "; " } else { ", " });
                old.push_str(&v);
            }
            None => out.push((k, v)),
        }
    }
    out
}

/// Send the original bytes upstream once. Statuses of 4xx and 5xx are responses, not errors; only a transport
/// failure (or a response over `max_response`, which is never truncated) is an Err.
fn call_upstream(agent: &ureq::Agent, route: &Route, headers: &[(String, String)], body: &[u8], max_response: usize) -> Result<Upstream, UpErr> {
    let err = |msg: String, timed_out: bool| UpErr { msg, timed_out, too_large: false };
    if let Some(spec) = &route.chat {
        return call_chat(agent, spec, headers, body, max_response);
    }
    let mut rq = agent.post(&route.url);
    for (k, v) in headers {
        rq = rq.set(k, v);
    }
    let resp = match rq.send_bytes(body) {
        Ok(r) | Err(ureq::Error::Status(_, r)) => r,
        Err(ureq::Error::Transport(t)) => {
            let m = t.to_string();
            let timed_out = m.contains("timed out") || m.contains("timeout");
            return Err(err(m, timed_out));
        }
    };
    let status = resp.status();
    let mut names: Vec<String> = vec![];
    for n in resp.headers_names() {
        if !names.iter().any(|x| x.eq_ignore_ascii_case(&n)) {
            names.push(n);
        }
    }
    let headers = names.iter().flat_map(|n| resp.all(n).into_iter().map(|v| (n.clone(), v.to_string()))).collect();
    let mut buf = vec![];
    resp.into_reader().take(max_response as u64 + 1).read_to_end(&mut buf).map_err(|e| err(format!("reading the upstream response: {e}"), e.kind() == std::io::ErrorKind::TimedOut))?;
    if buf.len() > max_response {
        return Err(UpErr { msg: format!("the upstream response is over the {max_response}-byte limit (--max-response-mib); it is not relayed truncated"), timed_out: false, too_large: true });
    }
    Ok(Upstream { status, headers, body: buf })
}

/// A chat route's "upstream": the System One request is answered by chat calls (`chat::ask`) and returned as the
/// System One response. A chat endpoint's own error status and body are relayed as they are.
fn call_chat(agent: &ureq::Agent, spec: &ChatSpec, headers: &[(String, String)], body: &[u8], max_response: usize) -> Result<Upstream, UpErr> {
    let json_headers = |extra: Option<String>| vec![("content-type".to_string(), extra.unwrap_or_else(|| "application/json".into()))];
    let reject = |status: u16, msg: &str| Ok(Upstream { status, headers: json_headers(None), body: error_body("invalid_request", msg) });
    let Ok(req) = serde_json::from_slice::<Value>(body) else { return reject(400, "the request body is not JSON") };
    let (Some(model), Some(questions)) = (req["model"].as_str(), req["questions"].as_object()) else {
        return reject(400, "a chat route needs a request with a string model and a questions object");
    };
    let auth = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("authorization")).map(|(_, v)| v.as_str());
    let mut spec = spec.clone();
    spec.max_response.get_or_insert(max_response);
    match chat::ask(agent, &spec, auth, model, &req["state"], questions) {
        Ok(r) => {
            let body = serde_json::to_vec(&r.to_systemone()).unwrap_or_default();
            if body.len() > max_response {
                return Err(UpErr { msg: format!("the chat response is over the {max_response}-byte limit"), timed_out: false, too_large: true });
            }
            Ok(Upstream { status: 200, headers: json_headers(None), body })
        }
        Err(ChatError::Http { status, content_type, retry_after, body }) => {
            let mut headers = json_headers(content_type);
            if let Some(ra) = retry_after {
                headers.push(("retry-after".to_string(), ra));
            }
            Ok(Upstream { status, headers, body })
        }
        Err(ChatError::TooLarge(n)) => Err(UpErr { msg: format!("the chat endpoint's response is over the {n}-byte limit (--max-response-mib); it is not relayed truncated"), timed_out: false, too_large: true }),
        Err(ChatError::Transport { msg, timed_out }) => Err(UpErr { msg, timed_out, too_large: false }),
        Err(ChatError::Unusable(m)) => Ok(Upstream { status: 502, headers: json_headers(None), body: error_body("bad_gateway", &format!("the chat lead's answer was unusable: {m}")) }),
    }
}

/// A response to the client: the upstream's end-to-end headers (every value), tagged `x-understudy`. Its own framing
/// is dropped (`Content-Length` stays only on a 304, where it describes the representation, RFC 9110 8.6).
fn reply(status: u16, headers: &[(String, String)], body: Vec<u8>, tag: &str) -> http::Response {
    let keep_len = status == 304;
    let mut h: Vec<(String, String)> = end_to_end(headers).into_iter().filter(|(k, _)| !k.eq_ignore_ascii_case("x-understudy")).collect();
    if keep_len {
        h.extend(headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).cloned());
    }
    h.push(("x-understudy".into(), tag.into()));
    http::Response { status, headers: h, body, stream: None }
}

fn json_reply(status: u16, body: Vec<u8>, tag: &str) -> http::Response {
    reply(status, &[("content-type".into(), "application/json".into())], body, tag)
}

fn error_body(kind: &str, message: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"error": {"type": kind, "message": message}})).unwrap()
}

// ---- the HTTP handler ----------------------------------------------------------------------------------------

fn iso_now() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let (days, rem) = ((d.as_secs() / 86400) as i64, d.as_secs() % 86400);
    // Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3600, rem % 3600 / 60, rem % 60, d.subsec_millis())
}

fn ms(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e6).round() / 1e3
}

fn lock_wait<T>(m: &Mutex<T>, wait: Duration) -> Option<MutexGuard<'_, T>> {
    let end = Instant::now() + wait;
    loop {
        match m.try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::Poisoned(_)) => return None,
            Err(TryLockError::WouldBlock) => {
                if Instant::now() >= end {
                    return None;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }
}

fn usage_of(body: &[u8]) -> Value {
    serde_json::from_slice::<Value>(body).ok().and_then(|v| v.get("usage").cloned()).unwrap_or(Value::Null)
}

fn bodies_json(raw: &[u8]) -> Value {
    serde_json::from_slice::<Value>(raw).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(raw).into_owned()))
}

type After = http::After;
type Handled = (http::Response, Option<After>);

fn handle(sh: &Arc<Shared>, rq: http::Request) -> Handled {
    let t0 = Instant::now();
    let path = rq.target.split('?').next().unwrap_or("").to_string();
    let post = rq.method == "POST";
    match (post, path.as_str()) {
        (false, "/healthz") => return (reply(200, &[("content-type".into(), "text/plain".into())], b"ok\n".to_vec(), "local"), None),
        (false, "/status" | "/understudy/status") => return (json_reply(200, serde_json::to_vec_pretty(&status_json(sh)).unwrap(), "local"), None),
        (true, "/v1/systemone" | "/systemone") => {}
        (_, "/healthz" | "/status" | "/understudy/status" | "/v1/systemone" | "/systemone") => return (json_reply(404, error_body("not_found", "understudy serves POST /v1/systemone (alias /systemone), GET /healthz and GET /status"), "error"), None),
        _ => {
            sh.stats.rejected.fetch_add(1, Relaxed);
            return (json_reply(404, error_body("not_found", "Understudy handles only /v1/systemone; this path is not forwarded"), "error"), None);
        }
    }
    sh.stats.requests.fetch_add(1, Relaxed);
    let mut client_headers = rq.headers.clone();
    let group = rq.header("x-understudy-group").map(|g| format!("{g}#{}", sh.seq.fetch_add(1, Relaxed)));
    let mut raw = rq.body;
    // A compressed body (the TypeSafe SDK zstd-compresses requests of 1 KiB or more) is read for routing and learning;
    // the upstream still gets the original bytes and the `Content-Encoding` header.
    let decoded = match decode_body(rq.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-encoding")).map(|(_, v)| v.as_str()), &raw, sh.cfg.max_body) {
        Ok(d) => d,
        Err(m) => {
            sh.stats.rejected.fetch_add(1, Relaxed);
            return (json_reply(400, error_body("invalid_request", &m), "error"), None);
        }
    };
    let parsed: Option<Value> = serde_json::from_slice(decoded.as_deref().unwrap_or(&raw)).ok();
    let model = parsed.as_ref().and_then(|v| v.get("model")).and_then(Value::as_str).map(String::from);
    let (Some(parsed), Some(model)) = (parsed, model) else {
        sh.stats.rejected.fetch_add(1, Relaxed);
        return (json_reply(400, error_body("invalid_request", "understudy routes by the request's model: the body must be a JSON object with a string model"), "error"), None);
    };
    let Some(route) = sh.route(&model).cloned() else {
        sh.stats.rejected.fetch_add(1, Relaxed);
        return (json_reply(400, error_body("unknown_model", &format!("no upstream is configured for model {model:?}")), "error"), None);
    };
    if let (Some(d), true) = (decoded, route.chat.is_some()) {
        // a chat route builds its own calls from the decoded request
        raw = d;
        client_headers.retain(|(k, _)| !k.eq_ignore_ascii_case("content-encoding"));
    }
    let fwd = upstream_headers(&route, &client_headers);
    let mode = sh.cfg.mode;
    let parsed = Arc::new(parsed);
    let mut entry = Map::new();
    entry.insert("ts".into(), json!(iso_now()));
    entry.insert("kind".into(), json!("request"));
    entry.insert("model".into(), json!(model));
    entry.insert("upstream".into(), json!(route.name));
    entry.insert("mode".into(), json!(mode.name()));
    entry.insert("request_bytes".into(), json!(raw.len()));
    if let Some(g) = &group {
        entry.insert("group".into(), json!(g));
    }
    // Work after the response is on the wire: capture logging, queueing for the learners, audits.
    let work = |sh: &Arc<Shared>, f: Box<dyn FnOnce() + Send>| -> Option<After> {
        sh.active.fetch_add(1, Relaxed); // counted now, not when it runs after the response: see `post_work`
        let sh = sh.clone();
        Some(Box::new(move || sh.post_work(f)))
    };

    // race: the lead call starts now; the local answer is computed beside it
    let mut raced: Option<Race> = None;
    if mode == ServeMode::Race {
        if sh.race_inflight.fetch_add(1, Relaxed) >= sh.cfg.race_max_inflight {
            sh.race_inflight.fetch_sub(1, Relaxed);
            sh.stats.race_capped.fetch_add(1, Relaxed);
            entry.insert("race_capped".into(), json!(true));
        } else {
            raced = Some(Race::start(sh, &route, &model, &fwd, &raw, &parsed));
        }
    }
    let mut local_ms: Option<f64> = None;
    let mut early: Option<(Result<Upstream, UpErr>, f64)> = None;
    if let Some(r) = &mut raced {
        let rid = sh.seq.fetch_add(1, Relaxed);
        entry.insert("race_id".into(), json!(rid));
        loop {
            match r.rx.recv() {
                Ok(Ev::Local(ans, took)) => {
                    local_ms = Some(took);
                    let Some((body, preds, served)) = ans else { continue };
                    sh.stats.local.fetch_add(1, Relaxed);
                    sh.stats.race_local.fetch_add(1, Relaxed);
                    let r = raced.take().unwrap();
                    entry.insert("routed".into(), json!("local"));
                    entry.insert("status".into(), json!(200));
                    entry.insert("latency_ms".into(), json!(ms(t0.elapsed())));
                    entry.insert("local_ms".into(), json!(local_ms));
                    entry.insert("upstream_ms".into(), Value::Null);
                    entry.insert("lead_ms".into(), Value::Null);
                    entry.insert("usage".into(), json!({"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}));
                    let resp = reply(200, &[("content-type".into(), "application/json".into())], body.clone(), "local");
                    let qi = r.infos;
                    // the lead's answer is still coming: it becomes the audit (counted as active work, so idle waits for it)
                    sh.active.fetch_add(1, Relaxed);
                    let (shf, route2, model2, parsed2, qi2, group2) = (sh.clone(), route.clone(), model.clone(), parsed.clone(), qi.clone(), group.clone());
                    let (rx, t_lead) = (r.rx, r.started);
                    std::thread::spawn(move || {
                        race_finish(&shf, route2, model2, rx, t_lead, parsed2, qi2, group2, served, rid, local_ms);
                        shf.active.fetch_sub(1, Relaxed);
                    });
                    let sh2 = sh.clone();
                    let after = work(sh, Box::new(move || {
                        let sh = sh2;
                        entry.insert("state_sha".into(), json!(state_hash(parsed.get("state").unwrap_or(&Value::Null))));
                        let qs: Vec<Value> = qi.iter().zip(&preds).map(|(i, p)| {
                            let mut v = json!({"name": i.name, "spec_hash": i.spec_hash, "type": i.qtype, "opaque": false});
                            v["understudent"] = p.json(i.question.as_ref().unwrap());
                            v
                        }).collect();
                        entry.insert("questions".into(), json!(qs));
                        if sh.cfg.capture_bodies {
                            entry.insert("request".into(), (*parsed).clone());
                            entry.insert("response".into(), bodies_json(&body));
                        }
                        sh.log(entry);
                    }));
                    return (resp, after);
                }
                Ok(Ev::Lead(res, up_ms)) => {
                    early = Some((res, up_ms));
                    break;
                }
                Err(_) => {
                    early = Some((Err(UpErr { msg: "the race's lead thread ended without an answer".into(), timed_out: false, too_large: false }), ms(t0.elapsed())));
                    break;
                }
            }
        }
        sh.stats.race_lead.fetch_add(1, Relaxed);
        entry.insert("local_ms".into(), json!(local_ms));
    }
    let race_infos = raced.take().map(|r| r.infos);

    // serve: answer locally when every question may be (race falls back to this past `--race-max-inflight`)
    let mut infos: Option<Vec<QInfo>> = race_infos;
    if mode == ServeMode::Serve || (mode == ServeMode::Race && entry.contains_key("race_capped")) {
        let qi = question_infos(&parsed);
        if let Some((body, preds, served)) = try_local(sh, &route, &model, &parsed, &qi) {
            sh.stats.local.fetch_add(1, Relaxed);
            entry.insert("routed".into(), json!("local"));
            entry.insert("status".into(), json!(200));
            entry.insert("latency_ms".into(), json!(ms(t0.elapsed())));
            entry.insert("upstream_ms".into(), Value::Null);
            entry.insert("usage".into(), json!({"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}));
            let audit = sh.cfg.audit_rate > 0.0 && sh.rand01() < sh.cfg.audit_rate;
            let resp = reply(200, &[("content-type".into(), "application/json".into())], body.clone(), "local");
            let (sh2, route2) = (sh.clone(), route.clone());
            let after = work(sh, Box::new(move || {
                let sh = sh2;
                entry.insert("state_sha".into(), json!(state_hash(parsed.get("state").unwrap_or(&Value::Null))));
                let qs: Vec<Value> = qi.iter().zip(&preds).map(|(i, p)| {
                    let mut v = json!({"name": i.name, "spec_hash": i.spec_hash, "type": i.qtype, "opaque": false});
                    v["understudent"] = p.json(i.question.as_ref().unwrap());
                    v
                }).collect();
                entry.insert("questions".into(), json!(qs));
                if sh.cfg.capture_bodies {
                    entry.insert("request".into(), (*parsed).clone());
                    entry.insert("response".into(), bodies_json(&body));
                }
                sh.log(entry);
                if audit {
                    spawn_audit(&sh, route2, model, raw, parsed, qi, fwd, group, served);
                }
            }));
            return (resp, after);
        }
        infos = Some(qi);
    }

    // forward the original bytes
    let (result, up_ms) = match early {
        Some(e) => e,
        None => {
            let t_up = Instant::now();
            let r = call_upstream(&sh.agent, &route, &fwd, &raw, sh.cfg.max_response);
            (r, ms(t_up.elapsed()))
        }
    };
    entry.insert("routed".into(), json!("lead"));
    entry.insert("latency_ms".into(), json!(ms(t0.elapsed())));
    entry.insert("upstream_ms".into(), json!(up_ms));
    if mode == ServeMode::Race {
        entry.insert("lead_ms".into(), json!(up_ms));
    }
    let up = match result {
        Ok(u) => u,
        Err(UpErr { msg, timed_out, too_large }) => {
            sh.stats.upstream_errors.fetch_add(1, Relaxed);
            let status = if timed_out { 504 } else { 502 };
            entry.insert("status".into(), json!(status));
            entry.insert("error".into(), json!(msg));
            if too_large {
                sh.stats.response_too_large.fetch_add(1, Relaxed);
                entry.insert("response_too_large".into(), json!(true));
            }
            let resp = json_reply(status, error_body(if too_large { "upstream_response_too_large" } else { "upstream_unreachable" }, &msg), "error");
            let sh2 = sh.clone();
            let after = work(sh, Box::new(move || {
                let qs = infos.unwrap_or_else(|| question_infos(&parsed));
                entry.insert("questions".into(), json!(basic_questions(&qs)));
                sh2.log(entry);
            }));
            return (resp, after);
        }
    };
    sh.stats.lead.fetch_add(1, Relaxed);
    entry.insert("status".into(), json!(up.status));
    let Upstream { status, headers, body } = up;
    let resp = reply(status, &headers, body.clone(), "lead");
    let (sh2, route2) = (sh.clone(), route);
    let after = work(sh, Box::new(move || {
        let sh = sh2;
        let infos = infos.unwrap_or_else(|| question_infos(&parsed));
        entry.insert("usage".into(), usage_of(&body));
        if sh.cfg.capture_bodies {
            entry.insert("request".into(), (*parsed).clone());
            entry.insert("response".into(), bodies_json(&body));
        }
        if mode == ServeMode::Forward {
            entry.insert("state_sha".into(), json!(state_hash(parsed.get("state").unwrap_or(&Value::Null))));
            entry.insert("questions".into(), json!(basic_questions(&infos)));
            sh.log(entry);
            return;
        }
        // shadow, serve and race learn from the lead's answers. A namespace is made only by a successful answer for this
        // model (one with an `answers` object); anything else is predicted from an existing namespace or just logged.
        let answered = (200..300).contains(&status) && serde_json::from_slice::<Value>(&body).ok().is_some_and(|v| v.get("answers").is_some_and(Value::is_object));
        let ns = if answered { sh.namespace(&route2, &model) } else { sh.find_namespace(&route2, &model) };
        let Some(ns) = ns else {
            entry.insert("state_sha".into(), json!(state_hash(parsed.get("state").unwrap_or(&Value::Null))));
            entry.insert("learning".into(), json!(if answered { "forward-only: --max-namespaces reached" } else { "not learned: the lead did not give an answer" }));
            entry.insert("questions".into(), json!(basic_questions(&infos)));
            sh.log(entry);
            return;
        };
        let body = if answered { body } else { vec![] };
        sh.enqueue(&ns, Job { entry, req: parsed, infos, status, body, group, served: None });
    }));
    (resp, after)
}

/// The request body decoded, when its `Content-Encoding` is `zstd` (None when it has none). Other codings are refused
/// with a message that says so; a decoded body over `max` bytes is refused too.
fn decode_body(encoding: Option<&str>, raw: &[u8], max: usize) -> Result<Option<Vec<u8>>, String> {
    let enc = encoding.map(|e| e.trim().to_ascii_lowercase()).unwrap_or_default();
    match enc.as_str() {
        "" | "identity" => Ok(None),
        "zstd" => {
            // the decoder holds a whole declared window before it yields a byte, so a frame may not declare one over the
            // body budget (or 8 MiB, the window every zstd encoder is allowed to use, so ordinary clients still work)
            let window = (max as u64).max(8 << 20);
            let mut dec = ruzstd::decoding::StreamingDecoder::new_with_max_window_size(raw, window).map_err(|e| format!("the zstd request body is unreadable: {e}"))?;
            let mut out = vec![];
            (&mut dec).take(max as u64 + 1).read_to_end(&mut out).map_err(|e| format!("the zstd request body is unreadable: {e}"))?;
            if out.len() > max {
                return Err("the decompressed request body is over --max-body-mib".into());
            }
            Ok(Some(out))
        }
        other => Err(format!("understudy reads request bodies with Content-Encoding zstd or none, not {other:?}")),
    }
}

/// What the race's two threads report to the request thread.
enum Ev {
    /// The local answer when every question may be answered locally (None otherwise), and the ms it took.
    Local(Option<(Vec<u8>, Vec<QPred>, Served)>, f64),
    /// The lead call's result and the ms it took.
    Lead(Result<Upstream, UpErr>, f64),
}

/// One `race` request in flight: the lead call and the local computation run on their own threads, so the request
/// thread adds nothing to the lead's latency and answers as soon as a usable result exists.
struct Race {
    rx: Receiver<Ev>,
    started: Instant,
    infos: Vec<QInfo>,
}

impl Race {
    /// Start the lead call (counted in `race_inflight` by the caller, released when it ends) and the local computation.
    fn start(sh: &Arc<Shared>, route: &Route, model: &str, fwd: &[(String, String)], raw: &[u8], parsed: &Arc<Value>) -> Race {
        let (tx, rx) = std::sync::mpsc::channel();
        let started = Instant::now();
        let infos = question_infos(parsed);
        {
            let (sh, route, fwd, raw, tx) = (sh.clone(), route.clone(), fwd.to_vec(), raw.to_vec(), tx.clone());
            std::thread::spawn(move || {
                let t = Instant::now();
                let r = call_upstream(&sh.agent, &route, &fwd, &raw, sh.cfg.max_response);
                let took = ms(t.elapsed());
                sh.race_inflight.fetch_sub(1, Relaxed);
                let _ = tx.send(Ev::Lead(r, took));
            });
        }
        {
            let (sh, route, model, parsed, infos) = (sh.clone(), route.clone(), model.to_string(), parsed.clone(), infos.clone());
            std::thread::spawn(move || {
                let t = Instant::now();
                let ans = try_local(&sh, &route, &model, &parsed, &infos);
                let _ = tx.send(Ev::Local(ans, ms(t.elapsed())));
            });
        }
        Race { rx, started, infos }
    }
}

/// A local answer was returned: wait for the lead call, which is only an audit and a learning example now (never
/// returned). A lead error is logged and nothing else.
#[allow(clippy::too_many_arguments)]
fn race_finish(sh: &Arc<Shared>, route: Route, model: String, rx: Receiver<Ev>, started: Instant, parsed: Arc<Value>, infos: Vec<QInfo>, group: Option<String>, served: Served, rid: u64, local_ms: Option<f64>) {
    let (r, took) = loop {
        match rx.recv() {
            Ok(Ev::Lead(r, took)) => break (r, took),
            Ok(Ev::Local(..)) => continue,
            Err(_) => break (Err(UpErr { msg: "the race's lead thread ended without an answer".into(), timed_out: false, too_large: false }), ms(started.elapsed())),
        }
    };
    let mut entry = Map::new();
    entry.insert("race_id".into(), json!(rid));
    entry.insert("mode".into(), json!("race"));
    entry.insert("routed".into(), json!("local"));
    entry.insert("local_ms".into(), json!(local_ms));
    entry.insert("lead_ms".into(), json!(took));
    complete_audit(sh, route, model, r, took, parsed, infos, group, served, entry);
}

/// `serve` mode's local answer: the response bytes, the predictions and what was served, when every question is a
/// known learner that is graduated and eligible. Waits at most `local_wait` for the namespace (learning holds it);
/// otherwise None and the request is forwarded. Never creates a namespace.
fn try_local(sh: &Arc<Shared>, route: &Route, model: &str, req: &Value, infos: &[QInfo]) -> Option<(Vec<u8>, Vec<QPred>, Served)> {
    if infos.is_empty() || infos.iter().any(|i| i.question.is_none()) {
        return None;
    }
    let state = req.get("state")?;
    let ns = sh.find_namespace(route, model)?;
    let Some(mut guard) = lock_wait(&ns.core.proxy, sh.cfg.local_wait) else {
        sh.stats.local_busy.fetch_add(1, Relaxed);
        return None;
    };
    let p = guard.as_mut()?;
    let fit_version = ns.core.fit_version.load(Relaxed);
    let mut sis: Vec<usize> = vec![];
    let mut at: Vec<usize> = vec![];
    for i in infos {
        let si = p.understudents.iter().position(|s| s.key == i.spec_hash)?;
        at.push(sis.iter().position(|&x| x == si).unwrap_or_else(|| {
            sis.push(si);
            sis.len() - 1
        }));
    }
    let preds = predict(p, state, &sis).ok()?;
    if !preds.iter().all(|q| q.local) {
        return None;
    }
    let mut answers = Map::new();
    let mut label_of: Vec<usize> = vec![0; sis.len()];
    for (k, &si) in sis.iter().enumerate() {
        label_of[k] = p.understudents[si].question.answer_object_normalized(preds[k].probs.as_ref()?).1;
    }
    for (i, &k) in infos.iter().zip(&at) {
        answers.insert(i.name.clone(), p.understudents[sis[k]].question.answer_object_normalized(preds[k].probs.as_ref()?).0);
    }
    let body = serde_json::to_vec(&json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
    }))
    .ok()?;
    let out = at.iter().map(|&k| QPred { probs: preds[k].probs.clone(), eligible: preds[k].eligible, novel: preds[k].novel, demoted: preds[k].demoted, local: true, mode: preds[k].mode.clone(), readout: preds[k].readout.clone(), tier: preds[k].tier }).collect();
    let served = Served { fit_version, per: sis.iter().enumerate().map(|(k, &si)| ServedQ { spec_hash: p.understudents[si].key.clone(), tier: preds[k].tier, label: label_of[k] }).collect() };
    Some((body, out, served))
}

/// Audit a local answer: send the same request to the lead in the background and learn from (and audit against)
/// its answer. At most `max_audits` run at once; the rest are skipped.
#[allow(clippy::too_many_arguments)]
fn spawn_audit(sh: &Arc<Shared>, route: Route, model: String, raw: Vec<u8>, parsed: Arc<Value>, infos: Vec<QInfo>, fwd: Vec<(String, String)>, group: Option<String>, served: Served) {
    if sh.audits_inflight.fetch_add(1, Relaxed) >= sh.cfg.max_audits {
        sh.audits_inflight.fetch_sub(1, Relaxed);
        sh.stats.audits_skipped.fetch_add(1, Relaxed);
        return;
    }
    sh.stats.audits.fetch_add(1, Relaxed);
    let sh = sh.clone();
    std::thread::spawn(move || {
        let t = Instant::now();
        let r = call_upstream(&sh.agent, &route, &fwd, &raw, sh.cfg.max_response);
        sh.audits_inflight.fetch_sub(1, Relaxed);
        let mut entry = Map::new();
        entry.insert("mode".into(), json!("serve"));
        entry.insert("routed".into(), json!("lead"));
        complete_audit(&sh, route, model, r, ms(t.elapsed()), parsed, infos, group, served, entry);
    });
}

/// Finish a background audit's capture entry and queue it as a learning job (an upstream error is only logged).
/// `entry` already holds the caller's `mode`, `routed` and any extra fields.
#[allow(clippy::too_many_arguments)]
fn complete_audit(sh: &Arc<Shared>, route: Route, model: String, r: Result<Upstream, UpErr>, took_ms: f64, parsed: Arc<Value>, infos: Vec<QInfo>, group: Option<String>, served: Served, mut entry: Map<String, Value>) {
    entry.insert("ts".into(), json!(iso_now()));
    entry.insert("kind".into(), json!("audit"));
    entry.insert("model".into(), json!(model));
    entry.insert("upstream".into(), json!(route.name));
    entry.insert("upstream_ms".into(), json!(took_ms));
    match r {
        Ok(u) => {
            entry.insert("status".into(), json!(u.status));
            entry.insert("usage".into(), usage_of(&u.body));
            match sh.find_namespace(&route, &model) {
                Some(ns) => sh.enqueue(&ns, Job { entry, req: parsed, infos, status: u.status, body: u.body, group, served: Some(served) }),
                None => {
                    entry.insert("learning".into(), json!("audit not learned: the namespace is gone"));
                    entry.insert("questions".into(), json!(basic_questions(&infos)));
                    sh.log(entry);
                }
            }
        }
        Err(e) => {
            entry.insert("error".into(), json!(e.msg));
            entry.insert("questions".into(), json!(basic_questions(&infos)));
            sh.log(entry);
        }
    }
}

fn status_json(sh: &Arc<Shared>) -> Value {
    let s = &sh.stats;
    let nss: Vec<Value> = sh
        .nss
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .map(|ns| {
            let learners = match ns.core.proxy.try_lock() {
                Ok(g) => g.as_ref().map(|p| p.understudents.iter().map(|u| json!({"spec_hash": u.key, "state": u.state(), "examples": u.examples.len(), "type": u.question.qtype})).collect::<Vec<_>>()).into(),
                Err(_) => json!("busy"),
            };
            json!({"namespace": ns.core.key, "route": ns.core.route, "model": ns.core.model, "identity": ns.core.identity, "error": ns.core.error.lock().unwrap_or_else(|e| e.into_inner()).clone(), "pending": ns.core.pending.load(Relaxed), "fit_version": ns.core.fit_version.load(Relaxed), "learners": learners})
        })
        .collect();
    json!({
        "mode": sh.cfg.mode.name(),
        "requests": s.requests.load(Relaxed), "local": s.local.load(Relaxed), "lead": s.lead.load(Relaxed),
        "upstream_errors": s.upstream_errors.load(Relaxed), "rejected": s.rejected.load(Relaxed),
        "learn_queued": s.queued.load(Relaxed), "learn_dropped": s.dropped.load(Relaxed), "learned": s.learned.load(Relaxed),
        "learn_errors": s.learn_errors.load(Relaxed), "audits": s.audits.load(Relaxed), "audits_skipped": s.audits_skipped.load(Relaxed),
        "race_local": s.race_local.load(Relaxed), "race_lead": s.race_lead.load(Relaxed), "race_capped": s.race_capped.load(Relaxed), "race_inflight": sh.race_inflight.load(Relaxed),
        "local_busy": s.local_busy.load(Relaxed), "namespaces_refused": s.ns_refused.load(Relaxed), "learners_refused": s.learners_refused.load(Relaxed),
        "post_dropped": s.post_dropped.load(Relaxed), "response_too_large": s.response_too_large.load(Relaxed), "namespaces": nss,
    })
}

// ---- the learning worker -------------------------------------------------------------------------------------

/// A saved snapshot is only for the upstream it learned from: refuse one whose recorded identity differs (or has none).
fn check_recorded_upstream(core: &NsCore) -> Result<()> {
    let Some(dir) = core.dir.as_deref().filter(|d| crate::snapshot::exists(d)) else { return Ok(()) };
    let meta = crate::snapshot::read_meta(dir)?;
    let rec = meta.get("lead").and_then(|l| l.get("namespace")).filter(|n| n.is_object());
    let Some(rec) = rec else {
        bail!("the snapshot in {} records no upstream, so it can't be shown to belong to {}; move it away to start fresh", dir.display(), core.identity["upstream"]);
    };
    if identity_key(rec) != core.key {
        bail!(
            "the snapshot in {} was learned from upstream {} (route {}, config {}), but this namespace is {} (route {}, config {}); refusing to reuse its learners: use a different --snapshot-dir or move it away",
            dir.display(), rec["upstream"], rec["route"], rec["config_hash"], core.identity["upstream"], core.identity["route"], core.identity["config_hash"]
        );
    }
    Ok(())
}

fn worker(sh: Arc<Shared>, core: Arc<NsCore>, rx: Receiver<Job>) {
    let built = check_recorded_upstream(&core).and_then(|()| (sh.cfg.factory)(core.dir.as_deref(), Box::new(Stash(core.stash.clone(), core.identity.clone()))));
    let user_audit = match built {
        Ok(mut p) => {
            // Every lead answer is free here (the caller already paid for it), so every one is audited and learned from.
            // The configured rate is kept for the saved snapshot and for serve's own background audits.
            let u = p.settings.audit_rate;
            p.settings.audit_rate = 1.0;
            *core.proxy.lock().unwrap_or_else(|e| e.into_inner()) = Some(p);
            u
        }
        Err(e) => {
            eprintln!("understudy: namespace {}: {e:#}", core.key);
            *core.error.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{e:#}"));
            for mut job in rx {
                job.entry.insert("learning".into(), json!(format!("namespace unavailable: {e:#}")));
                job.entry.insert("questions".into(), json!(basic_questions(&job.infos)));
                sh.log(job.entry);
                sh.stats.learn_errors.fetch_add(1, Relaxed);
                core.pending.fetch_sub(1, Relaxed);
            }
            return;
        }
    };
    let (mut dirty, mut since_save) = (false, 0usize);
    let save = |dirty: &mut bool, since: &mut usize| {
        *since = 0;
        let Some(dir) = &core.dir else { return };
        if !std::mem::take(dirty) {
            return;
        }
        let mut g = core.proxy.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = g.as_mut() {
            p.settings.audit_rate = user_audit;
            if let Err(e) = crate::snapshot::save(p, dir) {
                eprintln!("understudy: saving {}: {e:#}", dir.display());
            }
            p.settings.audit_rate = 1.0;
        }
    };
    loop {
        match rx.recv_timeout(Duration::from_secs(2)) {
            Ok(job) => {
                let entry = {
                    let mut g = core.proxy.lock().unwrap_or_else(|e| e.into_inner());
                    let p = g.as_mut().expect("proxy built");
                    upgrade(&sh, &core, p, &mut dirty);
                    // a panic in a fit must cost one job, not the namespace
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| process(&sh, &core, p, job)));
                    p.no_audit = false;
                    r
                };
                match entry {
                    Ok(e) => {
                        sh.log(e);
                        dirty = true;
                        since_save += 1;
                    }
                    Err(_) => {
                        sh.stats.learn_errors.fetch_add(1, Relaxed);
                        eprintln!("understudy: a learning job panicked");
                    }
                }
                core.pending.fetch_sub(1, Relaxed);
                if since_save >= sh.cfg.save_every {
                    save(&mut dirty, &mut since_save);
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                {
                    let mut g = core.proxy.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(p) = g.as_mut() {
                        upgrade(&sh, &core, p, &mut dirty);
                    }
                }
                save(&mut dirty, &mut since_save)
            }
            Err(RecvTimeoutError::Disconnected) => {
                save(&mut dirty, &mut since_save);
                return;
            }
        }
    }
}

/// Let the proxy grow (`ServeConfig::upgrade`), e.g. an `auto` race gaining the encoder designs once the model is ready.
fn upgrade(sh: &Shared, core: &NsCore, p: &mut Proxy, dirty: &mut bool) {
    let Some(up) = &sh.cfg.upgrade else { return };
    match up(p) {
        Ok(true) => {
            *dirty = true;
            core.fit_version.fetch_add(1, Relaxed);
            eprintln!("understudy: namespace {}: the encoder designs joined the race", core.key);
        }
        Ok(false) => {}
        Err(e) => eprintln!("understudy: namespace {}: could not add the encoder designs: {e:#}", core.key),
    }
}

/// Learn from one answered request (or only predict, for a failed one) and finish its capture entry.
fn process(sh: &Shared, core: &NsCore, p: &mut Proxy, job: Job) -> Map<String, Value> {
    let Job { mut entry, req, infos, status, body, group, served } = job;
    let state = req.get("state").cloned().unwrap_or(Value::Null);
    entry.insert("state_sha".into(), json!(state_hash(&state)));
    // the learner of every supported question (created on first sight)
    // (past `--max-learners` a new spec is not learned: it was forwarded and is counted)
    let learner: Vec<Option<usize>> = infos
        .iter()
        .map(|i| {
            let q = i.question.as_ref()?;
            if let Some(si) = p.understudents.iter().position(|s| s.key == i.spec_hash) {
                return Some(si);
            }
            if p.understudents.len() >= sh.cfg.max_learners {
                sh.stats.learners_refused.fetch_add(1, Relaxed);
                return None;
            }
            Some(p.understudent(&i.spec_hash, q))
        })
        .collect();
    let refused = infos.iter().zip(&learner).any(|(i, l)| i.question.is_some() && l.is_none());
    let resp: Option<Value> = (200..300).contains(&status).then(|| serde_json::from_slice(&body).ok()).flatten();
    let answers = resp.as_ref().and_then(|v| v.get("answers")).and_then(Value::as_object);
    // the lead's class per learner: the first submitted name whose answer parses
    let mut lead_of: HashMap<usize, (usize, Value)> = HashMap::new();
    let mut lead_label: Vec<Option<usize>> = vec![None; infos.len()];
    for (qi, (i, si)) in infos.iter().zip(&learner).enumerate() {
        let (Some(si), Some(a)) = (*si, answers.and_then(|a| a.get(&i.name))) else { continue };
        let a = normalize(a);
        if let Ok((lab, _)) = p.understudents[si].question.parse_answer(&a) {
            lead_label[qi] = Some(lab);
            lead_of.entry(si).or_insert((lab, a));
        }
    }
    let mut learn: Vec<usize> = lead_of.keys().copied().collect();
    learn.sort_unstable();
    let mut said: HashMap<usize, QPred> = HashMap::new();
    let mut note = json!("none");
    // an audit of a local answer: the served prediction against the lead's answer, at the fit that served it
    let mut served_audit: HashMap<usize, (usize, u64, Value)> = HashMap::new();
    if let Some(sv) = &served {
        let now = core.fit_version.load(Relaxed);
        let target = p.settings.fit.target;
        for sq in &sv.per {
            let Some(si) = p.understudents.iter().position(|s| s.key == sq.spec_hash) else { continue };
            let Some(&(lab, _)) = lead_of.get(&si) else { continue };
            let counted = if sv.fit_version != now {
                json!("stale: the fit was replaced after serving")
            } else if let Some(t) = sq.tier {
                json!(p.understudents[si].tiers[t].serving.audit(sq.label, lab, target))
            } else {
                json!(false)
            };
            served_audit.insert(si, (sq.label, sv.fit_version, counted));
        }
        // the audits above are the served ones; the proxy must not also audit the learner as it is now
        p.no_audit = true;
    }
    if !learn.is_empty() {
        {
            let mut st = core.stash.lock().unwrap_or_else(|e| e.into_inner());
            st.clear();
            for si in &learn {
                st.insert(p.understudents[*si].name.clone(), lead_of[si].1.clone());
            }
        }
        let res = p.answer(group.as_deref(), &state, Some(&learn));
        core.stash.lock().unwrap_or_else(|e| e.into_inner()).clear();
        match res {
            Ok(o) => {
                sh.stats.learned.fetch_add(1, Relaxed);
                note = json!("learned");
                for (si, (_, d)) in learn.iter().zip(&o.decisions) {
                    let local = d.eligible && !d.demoted;
                    said.insert(*si, QPred { probs: d.understudent_probs.clone(), eligible: d.eligible, novel: d.novel, demoted: d.demoted, local, mode: d.mode.clone(), readout: d.readout.clone(), tier: None });
                }
                if !o.refits.is_empty() {
                    core.fit_version.fetch_add(1, Relaxed);
                    entry.insert("refits".into(), json!(o.refits.iter().map(|(n, secs, r)| json!({"question": n, "secs": secs, "n": r["n"], "mode": r["mode"], "cv_agreement": r["cv_agreement"], "cv_coverage": r["cv_coverage"]})).collect::<Vec<_>>()));
                    for (n, secs, rep) in &o.refits {
                        eprintln!("understudy: refit {n} in {secs:.2}s: n={} readout={} mode={} cv_coverage={}", rep["n"], rep["readout"], rep["mode"], rep["cv_coverage"]);
                    }
                }
            }
            Err(e) => {
                sh.stats.learn_errors.fetch_add(1, Relaxed);
                note = json!(format!("learn error: {e:#}"));
            }
        }
    }
    // the rest: predictions only
    let rest: Vec<usize> = {
        let mut r: Vec<usize> = learner.iter().flatten().copied().filter(|si| !said.contains_key(si)).collect::<HashSet<_>>().into_iter().collect();
        r.sort_unstable();
        r
    };
    if !rest.is_empty() {
        match predict(p, &state, &rest) {
            Ok(ps) => said.extend(rest.iter().copied().zip(ps)),
            Err(e) => note = json!(format!("predict error: {e:#}")),
        }
    }
    if !(200..300).contains(&status) {
        note = json!("not learned: the lead did not answer 2xx");
    } else if resp.is_none() {
        note = json!("not learned: the lead's response is not JSON");
    } else if refused && learn.is_empty() {
        note = json!("not learned: --max-learners reached");
    }
    entry.insert("learning".into(), note);
    let qs: Vec<Value> = infos
        .iter()
        .enumerate()
        .map(|(qi, i)| {
            let mut v = json!({"name": i.name, "spec_hash": i.spec_hash, "type": i.qtype, "opaque": i.question.is_none()});
            if let Some(si) = learner[qi] {
                let q = &p.understudents[si].question;
                if let Some(s) = said.get(&si) {
                    v["understudent"] = s.json(q);
                    if let (Some(l), Some(pr)) = (lead_label[qi], &s.probs) {
                        v["agree"] = json!(crate::stats::argmax(pr) == l);
                    }
                }
                if let (Some((label, version, counted)), Some(l)) = (served_audit.get(&si), lead_label[qi]) {
                    // an audit's agreement is the prediction that was served, not the learner's answer now
                    v["served"] = json!({"label": q.options[*label], "fit_version": version, "audit_counted": counted});
                    v["agree"] = json!(*label == l);
                }
                v["lead"] = json!(lead_label[qi].map(|l| &q.options[l]));
                v["learner_state"] = json!(p.understudents[si].state());
                v["examples"] = json!(p.understudents[si].examples.len());
            }
            v
        })
        .collect();
    entry.insert("questions".into(), json!(qs));
    entry
}

// ---- lifecycle -----------------------------------------------------------------------------------------------

pub struct Server {
    /// The address actually bound (useful with port 0).
    pub addr: SocketAddr,
    shared: Arc<Shared>,
    listener: Option<http::Listener>,
    helpers: Vec<std::thread::JoinHandle<()>>,
}

pub fn start(cfg: ServeConfig) -> Result<Server> {
    let log = match &cfg.capture {
        Some(p) => Some(Mutex::new(BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(p).with_context(|| p.display().to_string())?))),
        None => None,
    };
    let agent = ureq::AgentBuilder::new()
        .timeout(cfg.timeout)
        .timeout_connect(Duration::from_secs(10))
        .redirects(0)
        .max_idle_connections(512)
        .max_idle_connections_per_host(256)
        .build();
    let (bind, max_conns) = (cfg.bind.clone(), cfg.max_conns);
    let limits = http::Limits { max_body: cfg.max_body, read_deadline: cfg.read_deadline, write_timeout: cfg.write_timeout };
    let (post_tx, post_rx) = sync_channel::<After>(cfg.post_queue);
    let shared = Arc::new(Shared {
        cfg,
        agent,
        nss: Mutex::new(HashMap::new()),
        workers: Mutex::new(vec![]),
        log,
        stats: Stats::default(),
        audits_inflight: AtomicUsize::new(0),
        race_inflight: AtomicUsize::new(0),
        seq: AtomicU64::new(0),
        rand: RandomState::new(),
        post: Mutex::new(Some(post_tx)),
        active: Arc::new(AtomicUsize::new(0)),
    });
    let post_rx = Arc::new(Mutex::new(post_rx));
    let helpers = (0..2)
        .map(|i| {
            let rx = post_rx.clone();
            std::thread::Builder::new()
                .name(format!("understudy-post-{i}"))
                .spawn(move || {
                    // the lock is held only while waiting for the next piece of work
                    while let Ok(w) = { let g = rx.lock().unwrap_or_else(|e| e.into_inner()); g.recv() } {
                        w();
                    }
                })
                .expect("spawn post thread")
        })
        .collect();
    let sh = shared.clone();
    let handler: http::Handler = Arc::new(move |rq| handle(&sh, rq));
    let listener = http::listen(&bind, max_conns, limits, handler).map_err(|e| anyhow!("binding {bind}: {e}"))?;
    Ok(Server { addr: listener.addr, shared, listener: Some(listener), helpers })
}

impl Server {
    /// Counters and namespaces, as `GET /status` returns them.
    pub fn status(&self) -> Value {
        status_json(&self.shared)
    }

    /// Whether every queued learning job has been processed.
    pub fn idle(&self) -> bool {
        self.shared.nss.lock().unwrap_or_else(|e| e.into_inner()).values().all(|n| n.core.pending.load(Relaxed) == 0)
    }

    /// Wait until the learning queues drain (and every namespace's proxy is built); false on timeout.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let end = Instant::now() + timeout;
        loop {
            let ready = self.shared.nss.lock().unwrap_or_else(|e| e.into_inner()).values().all(|n| n.core.proxy.try_lock().map(|g| g.is_some()).unwrap_or(true) || n.core.error.lock().unwrap_or_else(|e| e.into_inner()).is_some());
            if self.shared.active.load(Relaxed) == 0 && self.idle() && ready && self.shared.audits_inflight.load(Relaxed) == 0 && self.shared.race_inflight.load(Relaxed) == 0 {
                return true;
            }
            if Instant::now() >= end {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Run `f` with a namespace's proxy locked (as a refit holds it). None when the namespace doesn't exist or its
    /// proxy isn't built yet.
    pub fn with_proxy<R>(&self, upstream: &str, model: &str, f: impl FnOnce(&mut Proxy) -> R) -> Option<R> {
        let ns = self.shared.nss.lock().unwrap_or_else(|e| e.into_inner()).values().find(|n| n.core.route == upstream && n.core.model == model).cloned()?;
        let mut g = ns.core.proxy.lock().unwrap_or_else(|e| e.into_inner());
        g.as_mut().map(f)
    }

    /// Stop accepting, finish queued learning, save snapshots and wait for the threads.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if let Some(l) = self.listener.take() {
            l.stop();
        }
        // outstanding race audits (a late lead answer is a demotion audit and a training example) must land before
        // the namespaces go; wait for them, bounded, and say what is abandoned
        let end = Instant::now() + self.shared.cfg.shutdown_drain;
        loop {
            let (race, active) = (self.shared.race_inflight.load(Relaxed), self.shared.active.load(Relaxed));
            if race == 0 && active == 0 {
                break;
            }
            if Instant::now() >= end {
                eprintln!("understudy serve: shutdown abandoned {race} race lead call(s) and {active} unfinished audit/learning task(s) after {:?}", self.shared.cfg.shutdown_drain);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        // finish the post-response work, then let the workers drain their queues and save
        self.shared.post.lock().unwrap_or_else(|e| e.into_inner()).take();
        for t in self.helpers.drain(..) {
            let _ = t.join();
        }
        self.shared.nss.lock().unwrap_or_else(|e| e.into_inner()).clear();
        for w in self.shared.workers.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            let _ = w.join();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Serve until killed.
pub fn run(cfg: ServeConfig) -> Result<()> {
    let mode = cfg.mode;
    let routes: Vec<String> = cfg.routes.iter().map(|r| format!("{} -> {}{}{}", r.pattern, r.url, if r.chat.is_some() { " (chat)" } else { "" }, if r.bearer.is_some() { " (bearer)" } else { "" })).collect();
    let srv = start(cfg)?;
    eprintln!("understudy serve: {} mode on http://{} (POST /v1/systemone, GET /status)", mode.name(), srv.addr);
    for r in routes {
        eprintln!("understudy serve:   {r}");
    }
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

// ---- a small HTTP/1.1 server ---------------------------------------------------------------------------------

/// A thread-per-connection HTTP/1.1 server: keep-alive, `Content-Length` and chunked request bodies, `Expect:
/// 100-continue`. No pipelining, TLS or upgrades (the caller is a local client on loopback). Framing is strict: any
/// ambiguity (duplicate or conflicting lengths, `Transfer-Encoding` with `Content-Length`, a transfer coding other
/// than `chunked`, bare line feeds, a chunk not followed by CRLF) is an error and the connection is closed. A request
/// has a total time to arrive and a response a total time to be taken.
pub mod http {
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Per-connection limits.
    #[derive(Clone, Copy, Debug)]
    pub struct Limits {
        /// Largest request body.
        pub max_body: usize,
        /// Total time for one request's headers and body to arrive (not per read).
        pub read_deadline: Duration,
        /// Total time for one response to be written.
        pub write_timeout: Duration,
    }

    impl Limits {
        pub fn new(max_body: usize) -> Self {
            Limits { max_body, read_deadline: Duration::from_secs(60), write_timeout: Duration::from_secs(30) }
        }
    }

    pub struct Request {
        pub method: String,
        pub target: String,
        pub headers: Vec<(String, String)>,
        pub body: Vec<u8>,
    }

    impl Request {
        pub fn header(&self, name: &str) -> Option<&str> {
            self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
        }
    }

    /// A body read from a source as it is written (pass-through): its length when known, else sent chunked.
    pub struct Stream {
        pub reader: Box<dyn Read + Send>,
        pub len: Option<u64>,
    }

    pub struct Response {
        pub status: u16,
        pub headers: Vec<(String, String)>,
        pub body: Vec<u8>,
        /// When set, the body is read from here and written as it arrives (and `body` is ignored).
        pub stream: Option<Stream>,
    }

    /// Work to run once the response has been written.
    pub type After = Box<dyn FnOnce() + Send>;
    pub type Handler = Arc<dyn Fn(Request) -> (Response, Option<After>) + Send + Sync>;

    pub struct Listener {
        pub addr: SocketAddr,
        stop: Arc<AtomicBool>,
        conns: Arc<AtomicUsize>,
        accept: Option<std::thread::JoinHandle<()>>,
    }

    impl Listener {
        /// Stop accepting, let open connections finish their request and wait (briefly) for them.
        pub fn stop(mut self) {
            self.shutdown();
        }

        fn shutdown(&mut self) {
            self.stop.store(true, Relaxed);
            let _ = TcpStream::connect(self.addr); // wakes the blocked accept
            if let Some(a) = self.accept.take() {
                let _ = a.join();
            }
            let end = Instant::now() + Duration::from_secs(5);
            while self.conns.load(Relaxed) > 0 && Instant::now() < end {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    impl Drop for Listener {
        fn drop(&mut self) {
            self.shutdown();
        }
    }

    pub fn listen(bind: &str, max_conns: usize, limits: Limits, handler: Handler) -> io::Result<Listener> {
        let l = TcpListener::bind(bind)?;
        let addr = l.local_addr()?;
        let (stop, conns) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(0)));
        let (st, cn) = (stop.clone(), conns.clone());
        let accept = std::thread::Builder::new().name("understudy-accept".into()).spawn(move || {
            for s in l.incoming() {
                if st.load(Relaxed) {
                    break;
                }
                let Ok(mut s) = s else { continue };
                if cn.fetch_add(1, Relaxed) >= max_conns {
                    cn.fetch_sub(1, Relaxed);
                    let mut r = Response { status: 503, headers: vec![("content-type".into(), "application/json".into()), ("retry-after".into(), "1".into())], body: br#"{"error":{"type":"too_many_connections","message":"understudy is at --max-conns"}}"#.to_vec(), stream: None };
                    let _ = write_response(&mut s, &mut r, true, false, Duration::from_secs(1));
                    continue;
                }
                let (st2, cn2, h) = (st.clone(), cn.clone(), handler.clone());
                let r = std::thread::Builder::new().name("understudy-conn".into()).stack_size(1 << 20).spawn(move || {
                    connection(s, &st2, limits, &h);
                    cn2.fetch_sub(1, Relaxed);
                });
                if r.is_err() {
                    cn.fetch_sub(1, Relaxed);
                }
            }
        })?;
        Ok(Listener { addr, stop, conns, accept: Some(accept) })
    }

    fn reason(code: u16) -> &'static str {
        match code {
            100 => "Continue",
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            304 => "Not Modified",
            400 => "Bad Request",
            401 => "Unauthorized",
            402 => "Payment Required",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            408 => "Request Timeout",
            413 => "Payload Too Large",
            422 => "Unprocessable Entity",
            429 => "Too Many Requests",
            431 => "Request Header Fields Too Large",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            505 => "HTTP Version Not Supported",
            _ => "Status",
        }
    }

    /// Write every byte within `limit` in total (a peer that reads a trickle does not hold the connection forever).
    fn write_all_within(w: &mut TcpStream, mut buf: &[u8], limit: Duration) -> io::Result<()> {
        let end = Instant::now() + limit;
        while !buf.is_empty() {
            let now = Instant::now();
            if now >= end {
                return Err(io::ErrorKind::TimedOut.into());
            }
            w.set_write_timeout(Some(end - now))?;
            match w.write(buf) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => buf = &buf[n..],
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Err(io::ErrorKind::TimedOut.into()),
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Statuses that carry no body at all (RFC 9110 6.4.1): 1xx, 204, 304.
    fn bodiless(status: u16) -> bool {
        matches!(status, 100..=199 | 204 | 304)
    }

    fn write_response(w: &mut TcpStream, r: &mut Response, close: bool, head: bool, timeout: Duration) -> io::Result<()> {
        if let Some(st) = r.stream.take() {
            return write_stream(w, r, st, close, head, timeout);
        }
        let mut out = Vec::with_capacity(256 + r.body.len());
        write!(out, "HTTP/1.1 {} {}\r\n", r.status, reason(r.status))?;
        let mut len: Option<String> = None;
        for (k, v) in &r.headers {
            if k.contains(['\r', '\n']) || v.contains(['\r', '\n']) || k.eq_ignore_ascii_case("connection") {
                continue;
            }
            if k.eq_ignore_ascii_case("content-length") {
                // a 304 keeps the length of the representation it stands for; no response is framed by a repeat
                if r.status == 304 && len.is_none() && !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
                    len = Some(v.clone());
                }
                continue;
            }
            write!(out, "{k}: {v}\r\n")?;
        }
        // 204 (and 1xx) never carry Content-Length; 304 only the upstream's
        let len = match r.status {
            204 | 100..=199 => None,
            304 => len,
            _ => Some(r.body.len().to_string()),
        };
        if let Some(n) = len {
            write!(out, "content-length: {n}\r\n")?;
        }
        write!(out, "{}\r\n", if close { "connection: close\r\n" } else { "" })?;
        if !head && !bodiless(r.status) {
            out.extend_from_slice(&r.body);
        }
        write_all_within(w, &out, timeout)?;
        w.flush()
    }

    /// A response whose body is copied from its source as it arrives: with the source's length when it had one, else
    /// chunked. Every piece is flushed at once (an event stream's events reach the caller as they happen); the
    /// time limit applies to each write, not to the whole stream. A source that fails or ends early closes the
    /// connection without the terminating chunk, so the caller sees a broken stream, never a truncated "complete" one.
    fn write_stream(w: &mut TcpStream, r: &Response, mut st: Stream, close: bool, head: bool, timeout: Duration) -> io::Result<()> {
        let mut out = Vec::with_capacity(512);
        write!(out, "HTTP/1.1 {} {}\r\n", r.status, reason(r.status))?;
        for (k, v) in &r.headers {
            if k.contains(['\r', '\n']) || v.contains(['\r', '\n']) || k.eq_ignore_ascii_case("connection") || k.eq_ignore_ascii_case("content-length") || k.eq_ignore_ascii_case("transfer-encoding") {
                continue;
            }
            write!(out, "{k}: {v}\r\n")?;
        }
        let none = bodiless(r.status);
        let chunked = st.len.is_none() && !none && !head;
        match st.len {
            Some(n) if !none => write!(out, "content-length: {n}\r\n")?,
            _ if chunked => write!(out, "transfer-encoding: chunked\r\n")?,
            _ => {}
        }
        write!(out, "{}\r\n", if close { "connection: close\r\n" } else { "" })?;
        write_all_within(w, &out, timeout)?;
        w.flush()?;
        if head || none {
            return Ok(());
        }
        let mut left = st.len;
        let mut buf = vec![0u8; 16 * 1024];
        loop {
            let want = left.map_or(buf.len(), |l| l.min(buf.len() as u64) as usize);
            if want == 0 {
                break;
            }
            let n = match st.reader.read(&mut buf[..want]) {
                Ok(0) if left.is_some() => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if let Some(l) = &mut left {
                *l -= n as u64;
            }
            if chunked {
                let mut piece = format!("{n:x}\r\n").into_bytes();
                piece.extend_from_slice(&buf[..n]);
                piece.extend_from_slice(b"\r\n");
                write_all_within(w, &piece, timeout)?;
            } else {
                write_all_within(w, &buf[..n], timeout)?;
            }
            w.flush()?;
        }
        if chunked {
            write_all_within(w, b"0\r\n\r\n", timeout)?;
        }
        w.flush()
    }

    enum Bad {
        /// The peer went away or sent nonsense: just close.
        Close,
        Reply(u16, &'static str),
    }

    /// A read that fails: a timeout is a 408, anything else a closed connection.
    fn io_bad(e: io::Error) -> Bad {
        if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) {
            Bad::Reply(408, "request timeout")
        } else {
            Bad::Close
        }
    }

    /// The socket, with an optional deadline on all the reads that remain before it.
    struct DeadlineRead {
        s: TcpStream,
        deadline: Option<Instant>,
    }

    impl Read for DeadlineRead {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if let Some(d) = self.deadline {
                let now = Instant::now();
                if now >= d {
                    return Err(io::ErrorKind::TimedOut.into());
                }
                self.s.set_read_timeout(Some(d - now))?;
            }
            self.s.read(buf)
        }
    }

    type Reader = BufReader<DeadlineRead>;

    /// One CRLF-terminated line, without its CRLF, within the `left` byte budget (which it spends).
    fn line(rd: &mut Reader, buf: &mut Vec<u8>, left: &mut usize) -> Result<(), Bad> {
        buf.clear();
        let n = rd.by_ref().take(*left as u64 + 1).read_until(b'\n', buf).map_err(io_bad)?;
        if n > *left {
            return Err(Bad::Reply(431, "headers too large"));
        }
        if n == 0 || buf.last() != Some(&b'\n') {
            return Err(Bad::Close);
        }
        *left -= n;
        if buf.len() < 2 || buf[buf.len() - 2] != b'\r' {
            return Err(Bad::Reply(400, "a line ends in a bare LF"));
        }
        buf.truncate(buf.len() - 2);
        if buf.iter().any(|&b| b == b'\r' || b == 0) {
            return Err(Bad::Reply(400, "a bare CR or NUL in a line"));
        }
        Ok(())
    }

    fn token(name: &str) -> bool {
        !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
    }

    fn read_request(rd: &mut Reader, wr: &mut TcpStream, max_body: usize) -> Result<(Request, bool), Bad> {
        let (mut buf, mut left) = (vec![], 128 * 1024usize);
        line(rd, &mut buf, &mut left)?;
        let l = String::from_utf8_lossy(&buf).into_owned();
        let parts: Vec<&str> = l.split(' ').collect();
        let [method, target, version] = parts[..] else { return Err(Bad::Reply(400, "bad request line")) };
        if method.is_empty() || target.is_empty() {
            return Err(Bad::Reply(400, "bad request line"));
        }
        // origin-form only: an absolute-form or authority-form target (or one like `@host/x`) could name another host
        if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
            return Err(Bad::Reply(505, "HTTP/1.0 and HTTP/1.1 only"));
        }
        let mut headers = vec![];
        loop {
            line(rd, &mut buf, &mut left)?;
            if buf.is_empty() {
                break;
            }
            if matches!(buf[0], b' ' | b'\t') {
                return Err(Bad::Reply(400, "obsolete header line folding"));
            }
            let h = String::from_utf8_lossy(&buf).into_owned();
            let (k, v) = h.split_once(':').ok_or(Bad::Reply(400, "bad header"))?;
            if !token(k) {
                return Err(Bad::Reply(400, "bad header name"));
            }
            headers.push((k.to_string(), v.trim_matches([' ', '\t']).to_string()));
        }
        let mut rq = Request { method: method.into(), target: target.into(), headers, body: vec![] };
        let conn = rq.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("connection")).map(|(_, v)| v.to_ascii_lowercase()).collect::<Vec<_>>().join(",");
        let tokens: Vec<&str> = conn.split(',').map(str::trim).collect();
        let keep = if version == "HTTP/1.0" { tokens.contains(&"keep-alive") } else { !tokens.contains(&"close") };
        // framing: exactly one unambiguous way to find the end of the body
        let cls: Vec<&str> = rq.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).map(|(_, v)| v.as_str()).collect();
        let tes: Vec<&str> = rq.headers.iter().filter(|(k, _)| k.eq_ignore_ascii_case("transfer-encoding")).map(|(_, v)| v.as_str()).collect();
        if !cls.is_empty() && !tes.is_empty() {
            return Err(Bad::Reply(400, "both Transfer-Encoding and Content-Length"));
        }
        if cls.len() > 1 {
            return Err(Bad::Reply(400, "more than one Content-Length"));
        }
        if tes.len() > 1 {
            return Err(Bad::Reply(400, "more than one Transfer-Encoding"));
        }
        let chunked = match tes.first() {
            None => false,
            Some(t) if t.eq_ignore_ascii_case("chunked") => true,
            Some(_) => return Err(Bad::Reply(501, "only the chunked transfer coding is supported")),
        };
        let clen: Option<usize> = match cls.first() {
            None => None,
            Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => Some(n.parse().map_err(|_| Bad::Reply(413, "request body too large"))?),
            Some(_) => return Err(Bad::Reply(400, "bad content-length")),
        };
        if clen.is_some_and(|n| n > max_body) {
            return Err(Bad::Reply(413, "request body too large"));
        }
        if rq.header("expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue")) {
            wr.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(|_| Bad::Close)?;
        }
        if chunked {
            let mut body = vec![];
            loop {
                let mut left = 4096usize;
                line(rd, &mut buf, &mut left)?;
                let size_s = String::from_utf8_lossy(&buf).into_owned();
                let hex = size_s.split(';').next().unwrap_or("");
                if hex.is_empty() || hex.len() > 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(Bad::Reply(400, "bad chunk size"));
                }
                let size = usize::from_str_radix(hex, 16).map_err(|_| Bad::Reply(400, "bad chunk size"))?;
                if size == 0 {
                    // trailers: a few, in a small budget
                    let (mut left, mut count) = (8192usize, 0usize);
                    loop {
                        line(rd, &mut buf, &mut left)?;
                        if buf.is_empty() {
                            break;
                        }
                        count += 1;
                        if count > 32 {
                            return Err(Bad::Reply(431, "too many trailers"));
                        }
                    }
                    break;
                }
                if size > max_body - body.len() {
                    return Err(Bad::Reply(413, "request body too large"));
                }
                let got = rd.by_ref().take(size as u64).read_to_end(&mut body).map_err(io_bad)?;
                if got != size {
                    return Err(Bad::Close);
                }
                let mut crlf = [0u8; 2];
                rd.read_exact(&mut crlf).map_err(io_bad)?;
                if &crlf != b"\r\n" {
                    return Err(Bad::Reply(400, "a chunk is not followed by CRLF"));
                }
            }
            rq.body = body;
        } else if let Some(n) = clen {
            let mut body = Vec::new();
            let got = rd.by_ref().take(n as u64).read_to_end(&mut body).map_err(io_bad)?;
            if got != n {
                return Err(Bad::Close);
            }
            rq.body = body;
        }
        Ok((rq, keep))
    }

    fn connection(stream: TcpStream, stop: &AtomicBool, limits: Limits, handler: &Handler) {
        let _ = stream.set_nodelay(true);
        let Ok(rs) = stream.try_clone() else { return };
        let mut wr = stream;
        let _ = wr.set_write_timeout(Some(limits.write_timeout));
        let mut rd: Reader = BufReader::with_capacity(16 * 1024, DeadlineRead { s: rs, deadline: None });
        let mut idle = Instant::now();
        loop {
            // wait for the first byte of the next request, polling for shutdown and giving up on idle connections
            rd.get_mut().deadline = None;
            let _ = rd.get_ref().s.set_read_timeout(Some(Duration::from_millis(500)));
            match rd.fill_buf() {
                Ok([]) => return,
                Ok(_) => {}
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {
                    if stop.load(Relaxed) || idle.elapsed() > Duration::from_secs(300) {
                        return;
                    }
                    continue;
                }
                Err(_) => return,
            }
            // from the first byte, headers and body together have `read_deadline` to arrive
            rd.get_mut().deadline = Some(Instant::now() + limits.read_deadline);
            let (rq, keep) = match read_request(&mut rd, &mut wr, limits.max_body) {
                Ok(x) => x,
                Err(Bad::Close) => return,
                Err(Bad::Reply(code, msg)) => {
                    let body = format!("{{\"error\":{{\"type\":\"bad_request\",\"message\":\"{msg}\"}}}}").into_bytes();
                    let _ = write_response(&mut wr, &mut Response { status: code, headers: vec![("content-type".into(), "application/json".into())], body, stream: None }, true, false, limits.write_timeout);
                    let _ = wr.shutdown(Shutdown::Both);
                    return;
                }
            };
            rd.get_mut().deadline = None;
            let head = rq.method == "HEAD";
            let (mut resp, after) = handler(rq);
            let close = !keep || stop.load(Relaxed);
            let wrote = write_response(&mut wr, &mut resp, close, head, limits.write_timeout);
            if let Some(a) = after {
                a();
            }
            if wrote.is_err() || close {
                let _ = wr.shutdown(Shutdown::Both);
                return;
            }
            idle = Instant::now();
        }
    }
}
