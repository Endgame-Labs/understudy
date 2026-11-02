//! A persisted snapshot: everything a `Proxy` needs to resume, in a directory.
//!
//! ```text
//! DIR/snapshot.json                  format, encoder fingerprint, featurizer, settings, lead, audit RNG state, and
//!                                 per question: spec, key, fit metadata (readout, mode, threshold, novelty floor,
//!                                 CV report), audit window, and the example window (seq, kind, state hash,
//!                                 label, lead probabilities, encoding key)
//! DIR/fits/<key>.safetensors      the fit's arrays: pool.{mu,sd,coef,intercept}, attn.{mu,sd,q,w,b}, refs
//! DIR/states/<enc>.safetensors    one encoding per state: part<i>.pool (F32) and part<i>.tokens (BF16 [t, h])
//! ```
//! The safetensors files are standard (8-byte header length, JSON header, little-endian data), so numpy or the
//! `safetensors` package read them. Encodings are written once and removed when no example references them.

use crate::encoder::{Encoder, Encoding, Part, Slot};
use crate::shape::BlendStd;
use crate::linear::Logistic;
use crate::keywords::{Neighbors, Sparse, SparseLr};
use crate::proxy::{Example, Kind, Proxy, Settings, Understudent};
use crate::understudent::UnderstudentKind;
use crate::question::Question;
use crate::readout::{AttnModel, PoolModel, Readout, Standardizer};
use crate::understudent::{Audit, Fit, Mode, Serving};
use crate::lead::Lead;
use anyhow::{Context, Result, anyhow, bail, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

pub const FORMAT: &str = "understudy-snapshot/1";

// ---------------------------------------------------------------- safetensors

pub enum Arr<'a> {
    F32(&'a [f32]),
    Bf16(&'a [bf16]),
    U32(&'a [u32]),
}

/// Write a safetensors file (atomically: a temporary file, then a rename).
pub fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Arr)], meta: &Map<String, Value>) -> Result<()> {
    let mut header = Map::new();
    let meta: Map<String, Value> = meta.iter().map(|(k, v)| (k.clone(), Value::String(v.as_str().map_or_else(|| v.to_string(), String::from)))).collect();
    header.insert("__metadata__".into(), Value::Object(meta));
    let mut data: Vec<u8> = vec![];
    for (name, shape, arr) in tensors {
        let start = data.len();
        let dtype = match arr {
            Arr::F32(a) => {
                data.extend(a.iter().flat_map(|x| x.to_le_bytes()));
                "F32"
            }
            Arr::Bf16(a) => {
                data.extend(a.iter().flat_map(|x| x.to_bits().to_le_bytes()));
                "BF16"
            }
            Arr::U32(a) => {
                data.extend(a.iter().flat_map(|x| x.to_le_bytes()));
                "U32"
            }
        };
        header.insert((*name).into(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut h = serde_json::to_vec(&Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + data.len());
    out.extend((h.len() as u64).to_le_bytes());
    out.extend(h);
    out.extend(data);
    write_atomic(path, &out)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| tmp.display().to_string())?;
    std::fs::rename(&tmp, path).with_context(|| path.display().to_string())?;
    Ok(())
}

/// A safetensors file read whole.
pub struct SafeTensors {
    header: Map<String, Value>,
    data: Vec<u8>,
}

impl SafeTensors {
    pub fn read(path: &Path) -> Result<Self> {
        let b = std::fs::read(path).with_context(|| path.display().to_string())?;
        ensure!(b.len() >= 8, "{}: too short", path.display());
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        ensure!(b.len() >= 8 + n, "{}: truncated header", path.display());
        let header: Map<String, Value> = serde_json::from_slice(&b[8..8 + n])?;
        Ok(SafeTensors { header, data: b[8 + n..].to_vec() })
    }

    /// A string from the file's `__metadata__`.
    pub fn meta(&self, name: &str) -> Option<String> {
        self.header.get("__metadata__")?.get(name)?.as_str().map(String::from)
    }

    pub fn has(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    fn raw(&self, name: &str, dtype: &str) -> Result<(&[u8], Vec<usize>)> {
        let t = self.header.get(name).ok_or_else(|| anyhow!("no tensor {name}"))?;
        ensure!(t["dtype"] == dtype, "{name}: dtype {} not {dtype}", t["dtype"]);
        let o: Vec<usize> = serde_json::from_value(t["data_offsets"].clone())?;
        let shape: Vec<usize> = serde_json::from_value(t["shape"].clone())?;
        Ok((&self.data[o[0]..o[1]], shape))
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (b, _) = self.raw(name, "F32")?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn u32(&self, name: &str) -> Result<Vec<u32>> {
        let (b, _) = self.raw(name, "U32")?;
        Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn bf16(&self, name: &str) -> Result<(Vec<bf16>, Vec<usize>)> {
        let (b, shape) = self.raw(name, "BF16")?;
        Ok((b.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap()))).collect(), shape))
    }
}

// ---------------------------------------------------------------- snapshot.json

#[derive(Serialize, Deserialize)]
struct SnapshotFile {
    format: String,
    encoder: Value,
    featurizer: crate::encoder::Featurizer,
    settings: Settings,
    lead: Value,
    rng: crate::rng::Mt19937,
    seq: u64,
    understudents: Vec<UnderstudentFile>,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

#[derive(Serialize, Deserialize)]
struct UnderstudentFile {
    name: String,
    key: String,
    question: Question,
    since_refit: usize,
    demoted: bool,
    audits: Vec<Audit>,
    fit: FitMeta,
    /// A ladder's tiers, cheapest first (empty for a single understudent, whose one tier is the fields above).
    #[serde(default)]
    tiers: Vec<TierFile>,
    #[serde(default)]
    since_cheap: usize,
    /// `auto`: the serving candidate (index into `tiers`), the lead-only verdict and the last race.
    #[serde(default)]
    serving: Option<usize>,
    #[serde(default)]
    lead_only: Option<String>,
    #[serde(default)]
    race: Value,
    #[serde(default)]
    rules_state: crate::rules::RulesState,
    /// `--graduation sequential`: the graduation attempts spent so far.
    #[serde(default, skip_serializing_if = "is_zero")]
    attempts: usize,
    examples: Vec<ExampleFile>,
}

#[derive(Serialize, Deserialize)]
struct TierFile {
    kind: UnderstudentKind,
    demoted: bool,
    audits: Vec<Audit>,
    fit: FitMeta,
}

#[derive(Serialize, Deserialize)]
struct ExampleFile {
    seq: u64,
    kind: Kind,
    state: String,
    label: usize,
    probs: Option<Vec<f64>>,
    enc: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    text_len: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    group: Option<u64>,
}

#[derive(Serialize, Deserialize, Default)]
struct FitMeta {
    readout: Option<Readout>,
    mode: Option<Mode>,
    thr: f64,
    nn_floor: f64,
    ref_dim: usize,
    report: Value,
    pool: Option<PoolMeta>,
    attn: Option<AttnMeta>,
    sparse: Option<SparseMeta>,
    neighbors: Option<NeighborsMeta>,
    /// The `rules` understudent's rule set: patterns, labels, held-out stats (human-readable).
    #[serde(default)]
    rules: Option<crate::rules::RuleSet>,
    #[serde(default)]
    blend: Option<BlendMeta>,
    #[serde(default)]
    memo: Option<MemoMeta>,
    #[serde(default)]
    net: Option<NetMeta>,
    /// The `checklist` understudent's rules (small: kept whole in the metadata).
    #[serde(default)]
    checklist: Option<crate::checklist::Checklist>,
    #[serde(default, with = "crate::abstain::thr_serde")]
    class_thr: Option<Vec<f64>>,
}

#[derive(Serialize, Deserialize)]
struct NetMeta {
    dp: usize,
    df: usize,
    h: usize,
    classes: Vec<usize>,
    epochs: usize,
}

#[derive(Serialize, Deserialize)]
struct BlendMeta {
    shape_d: usize,
    emb_d: usize,
}

#[derive(Serialize, Deserialize)]
struct MemoMeta {
    thr_sim: f64,
}

#[derive(Serialize, Deserialize)]
struct SparseMeta {
    classes: Vec<usize>,
    iters: usize,
}

#[derive(Serialize, Deserialize)]
struct NeighborsMeta {
    d: usize,
}

fn fit_meta(fit: &Fit) -> FitMeta {
    FitMeta {
        readout: fit.readout,
        mode: fit.mode,
        thr: fit.thr,
        nn_floor: fit.nn_floor,
        ref_dim: fit.ref_dim,
        report: fit.report.clone(),
        pool: fit.pool.as_ref().map(|p| PoolMeta { d: p.lr.d, classes: p.lr.classes.clone(), iters: p.lr.iters }),
        attn: fit.attn.as_ref().map(|a| AttnMeta { hidden: a.hidden, queries: a.queries, pair: a.pair, classes: a.classes.clone(), epochs: a.epochs }),
        sparse: fit.sparse.as_ref().map(|m| SparseMeta { classes: m.classes.clone(), iters: m.iters }),
        neighbors: fit.neighbors.as_ref().map(|m| NeighborsMeta { d: m.d }),
        rules: fit.rules.as_ref().map(|r| (**r).clone()),
        blend: fit.blend.as_ref().map(|t| BlendMeta { shape_d: t.shape.mu.len(), emb_d: t.emb.mu.len() }),
        memo: fit.memo.as_ref().map(|m| MemoMeta { thr_sim: m.thr_sim }),
        net: fit.net.as_ref().map(|m| NetMeta { dp: m.pool_std.mu.len(), df: m.shape_std.mu.len(), h: m.h, classes: m.classes.clone(), epochs: m.epochs }),
        checklist: fit.checklist.clone(),
        class_thr: fit.class_thr.clone(),
    }
}

#[derive(Serialize, Deserialize)]
struct PoolMeta {
    d: usize,
    classes: Vec<usize>,
    iters: usize,
}

#[derive(Serialize, Deserialize)]
struct AttnMeta {
    hidden: usize,
    queries: usize,
    pair: bool,
    classes: Vec<usize>,
    epochs: usize,
}

pub fn exists(dir: &Path) -> bool {
    dir.join("snapshot.json").is_file()
}

/// Read snapshot.json without loading arrays or the encoder (for `status`).
pub fn read_meta(dir: &Path) -> Result<Value> {
    let p = dir.join("snapshot.json");
    let v: Value = serde_json::from_slice(&std::fs::read(&p).with_context(|| p.display().to_string())?)?;
    ensure!(matches!(v.get("format").and_then(Value::as_str), Some(FORMAT)), "{}: not a {FORMAT} snapshot", p.display());
    Ok(v)
}

/// Save the proxy's state to `dir`.
pub fn save(proxy: &Proxy, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir.join("fits"))?;
    std::fs::create_dir_all(dir.join("states"))?;
    let mut live: HashSet<String> = HashSet::new();
    for s in &proxy.understudents {
        for e in &s.examples {
            if live.insert(e.enc_key.clone()) {
                let p = dir.join("states").join(format!("{}.safetensors", e.enc_key));
                if !p.exists() {
                    write_encoding(&p, &e.enc)?;
                }
            }
        }
    }
    let mut understudents = vec![];
    for s in &proxy.understudents {
        let ladder = s.is_tiered();
        let one = s.serving();
        if ladder {
            for t in &s.tiers {
                write_fit(&dir.join("fits").join(format!("{}.{}.safetensors", s.key, t.kind.name())), &t.serving.fit)?;
            }
        } else {
            write_fit(&dir.join("fits").join(format!("{}.safetensors", s.key)), &one.fit)?;
        }
        understudents.push(UnderstudentFile {
            name: s.name.clone(),
            key: s.key.clone(),
            question: s.question.clone(),
            since_refit: s.since_refit,
            since_cheap: s.since_cheap,
            serving: s.serving_idx,
            lead_only: s.lead_only.clone(),
            race: s.race.clone(),
            rules_state: s.rules.clone(),
            attempts: s.attempts,
            demoted: if ladder { false } else { one.demoted },
            audits: if ladder { vec![] } else { one.audits.iter().copied().collect() },
            fit: if ladder { FitMeta::default() } else { fit_meta(&one.fit) },
            tiers: if ladder {
                s.tiers
                    .iter()
                    .map(|t| TierFile { kind: t.kind, demoted: t.serving.demoted, audits: t.serving.audits.iter().copied().collect(), fit: fit_meta(&t.serving.fit) })
                    .collect()
            } else {
                vec![]
            },
            examples: s
                .examples
                .iter()
                .map(|e| ExampleFile { seq: e.seq, kind: e.kind, state: e.state.clone(), label: e.label, probs: e.probs.clone(), enc: e.enc_key.clone(), text_len: e.text_len, group: e.group })
                .collect(),
        });
    }
    let pf = SnapshotFile {
        format: FORMAT.into(),
        encoder: proxy.encoder.fingerprint(),
        featurizer: proxy.fz.clone(),
        settings: proxy.settings.clone(),
        lead: proxy.lead.as_ref().map_or(Value::Null, |t| t.describe()),
        rng: proxy.rng.clone(),
        seq: proxy.seq,
        understudents,
    };
    write_atomic(&dir.join("snapshot.json"), &serde_json::to_vec_pretty(&pf)?)?;
    // drop encodings no example references any more
    for f in std::fs::read_dir(dir.join("states"))? {
        let p = f?.path();
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        if p.extension().is_some_and(|x| x == "safetensors") && !live.contains(&stem) {
            std::fs::remove_file(&p)?;
        }
    }
    Ok(())
}

/// The tensor name prefixes of an encoding's three part lists.
const SLOTS: [(&str, Slot); 3] = [("part", Slot::Main), ("mean", Slot::Mean), ("stat", Slot::Stat)];

fn write_encoding(path: &Path, e: &Encoding) -> Result<()> {
    let mut names: Vec<(String, String)> = vec![];
    let mut lists: Vec<&Part> = vec![];
    for (prefix, slot) in SLOTS {
        for (i, p) in e.slot(slot).iter().enumerate() {
            names.push((format!("{prefix}{i}.pool"), format!("{prefix}{i}.tokens")));
            lists.push(p);
        }
    }
    let mut ts = vec![];
    for (p, (a, b)) in lists.into_iter().zip(&names) {
        let h = if p.tokens > 0 { p.states.len() / p.tokens } else { 0 };
        ts.push((a.as_str(), vec![p.pool.len()], Arr::F32(&p.pool)));
        ts.push((b.as_str(), vec![p.tokens, h], Arr::Bf16(&p.states)));
    }
    if let Some(sp) = &e.sparse {
        ts.push(("sparse.idx", vec![sp.idx.len()], Arr::U32(&sp.idx)));
        ts.push(("sparse.val", vec![sp.val.len()], Arr::F32(&sp.val)));
    }
    if let Some(v) = &e.shape {
        ts.push(("shape.v", vec![v.len()], Arr::F32(v)));
    }
    // the retrieval key: the exact hash (two u32 halves) then the signature
    let memo_sig: Vec<u32> = e.memo.as_ref().map(|m| [(m.exact >> 32) as u32, m.exact as u32].into_iter().chain(m.sig.iter().copied()).collect()).unwrap_or_default();
    if e.memo.is_some() {
        ts.push(("memo.key", vec![memo_sig.len()], Arr::U32(&memo_sig)));
    }
    let mut meta = Map::new();
    meta.insert("format".into(), json!(FORMAT));
    meta.insert("parts".into(), json!(e.parts.len().to_string()));
    if let Some(r) = &e.rules {
        meta.insert("rules".into(), json!(serde_json::to_string(&**r)?));
    }
    write_safetensors(path, &ts, &meta)
}

fn read_encoding(path: &Path) -> Result<Encoding> {
    let st = SafeTensors::read(path)?;
    let mut enc = Encoding::default();
    for (prefix, slot) in SLOTS {
        let mut parts = vec![];
        for i in 0.. {
            if !st.has(&format!("{prefix}{i}.pool")) {
                break;
            }
            let pool = st.f32(&format!("{prefix}{i}.pool"))?;
            let (states, shape) = st.bf16(&format!("{prefix}{i}.tokens"))?;
            parts.push(Part { pool, tokens: shape[0], states });
        }
        enc.set_slot(slot, parts);
    }
    enc.sparse = if st.has("sparse.idx") { Some(Sparse { idx: st.u32("sparse.idx")?, val: st.f32("sparse.val")? }) } else { None };
    enc.rules = match st.meta("rules") {
        Some(s) => Some(Arc::new(serde_json::from_str(&s).with_context(|| format!("{}: rules input", path.display()))?)),
        None => None,
    };
    enc.shape = if st.has("shape.v") { Some(Arc::new(st.f32("shape.v")?)) } else { None };
    enc.memo = if st.has("memo.key") {
        let k = st.u32("memo.key")?;
        ensure!(k.len() == 2 + crate::memo::NSIG, "{}: memo key of length {}", path.display(), k.len());
        Some(Arc::new(crate::memo::MemoKey { exact: (k[0] as u64) << 32 | k[1] as u64, sig: k[2..].to_vec() }))
    } else {
        None
    };
    ensure!(!enc.parts.is_empty() || !enc.mean.is_empty() || !enc.stat.is_empty() || enc.sparse.is_some() || enc.rules.is_some() || enc.shape.is_some() || enc.memo.is_some(), "{}: no parts", path.display());
    Ok(enc)
}

fn write_fit(path: &Path, f: &Fit) -> Result<()> {
    let mut ts: Vec<(&str, Vec<usize>, Arr)> = vec![];
    if f.ref_dim > 0 {
        ts.push(("refs", vec![f.refs.len() / f.ref_dim, f.ref_dim], Arr::F32(&f.refs)));
    }
    if let Some(p) = &f.pool {
        let d = p.lr.d;
        ts.push(("pool.mu", vec![d], Arr::F32(&p.std.mu)));
        ts.push(("pool.sd", vec![d], Arr::F32(&p.std.sd)));
        ts.push(("pool.coef", vec![p.lr.intercept.len(), d], Arr::F32(&p.lr.coef)));
        ts.push(("pool.intercept", vec![p.lr.intercept.len()], Arr::F32(&p.lr.intercept)));
    }
    if let Some(a) = &f.attn {
        ts.push(("attn.mu", vec![a.hidden], Arr::F32(&a.std.mu)));
        ts.push(("attn.sd", vec![a.hidden], Arr::F32(&a.std.sd)));
        ts.push(("attn.q", vec![a.queries, a.hidden], Arr::F32(&a.q)));
        ts.push(("attn.w", vec![a.classes.len(), a.w.len() / a.classes.len()], Arr::F32(&a.w)));
        ts.push(("attn.b", vec![a.classes.len()], Arr::F32(&a.b)));
    }
    if let Some(m) = &f.sparse {
        ts.push(("sparse.coef", vec![m.intercept.len(), m.coef.len() / m.intercept.len()], Arr::F32(&m.coef)));
        ts.push(("sparse.intercept", vec![m.intercept.len()], Arr::F32(&m.intercept)));
    }
    let neighbors_y: Vec<f32> = f.neighbors.as_ref().map(|m| m.y.iter().map(|&c| c as f32).collect()).unwrap_or_default();
    if let Some(m) = &f.neighbors {
        ts.push(("neighbors.vecs", vec![m.len(), m.d], Arr::F32(&m.vecs)));
        ts.push(("neighbors.y", vec![m.len()], Arr::F32(&neighbors_y)));
    }
    if let Some(t) = &f.blend {
        ts.push(("blend.shape_mu", vec![t.shape.mu.len()], Arr::F32(&t.shape.mu)));
        ts.push(("blend.shape_sd", vec![t.shape.sd.len()], Arr::F32(&t.shape.sd)));
        ts.push(("blend.emb_mu", vec![t.emb.mu.len()], Arr::F32(&t.emb.mu)));
        ts.push(("blend.emb_sd", vec![t.emb.sd.len()], Arr::F32(&t.emb.sd)));
    }
    let memo_y: Vec<f32> = f.memo.as_ref().map(|m| m.y.iter().map(|&c| c as f32).collect()).unwrap_or_default();
    let memo_exact: Vec<u32> = f.memo.as_ref().map(|m| m.exact.iter().flat_map(|&h| [(h >> 32) as u32, h as u32]).collect()).unwrap_or_default();
    if let Some(m) = &f.memo {
        ts.push(("memo.sig", vec![m.len(), crate::memo::NSIG], Arr::U32(&m.sig)));
        ts.push(("memo.exact", vec![memo_exact.len()], Arr::U32(&memo_exact)));
        ts.push(("memo.y", vec![m.len()], Arr::F32(&memo_y)));
        ts.push(("memo.conf", vec![m.len()], Arr::F32(&m.conf)));
    }
    if let Some(m) = &f.net {
        ts.push(("net.pool_mu", vec![m.pool_std.mu.len()], Arr::F32(&m.pool_std.mu)));
        ts.push(("net.pool_sd", vec![m.pool_std.sd.len()], Arr::F32(&m.pool_std.sd)));
        ts.push(("net.shape_mu", vec![m.shape_std.mu.len()], Arr::F32(&m.shape_std.mu)));
        ts.push(("net.shape_sd", vec![m.shape_std.sd.len()], Arr::F32(&m.shape_std.sd)));
        ts.push(("net.w1", vec![m.h, m.d()], Arr::F32(&m.w1)));
        ts.push(("net.b1", vec![m.h], Arr::F32(&m.b1)));
        ts.push(("net.w2", vec![m.classes.len(), m.h], Arr::F32(&m.w2)));
        ts.push(("net.b2", vec![m.classes.len()], Arr::F32(&m.b2)));
    }
    let mut meta = Map::new();
    meta.insert("format".into(), json!(FORMAT));
    write_safetensors(path, &ts, &meta)
}

fn read_fit(path: &Path, m: FitMeta) -> Result<Fit> {
    let st = if path.exists() { Some(SafeTensors::read(path)?) } else { None };
    let get = |n: &str| -> Result<Vec<f32>> { st.as_ref().ok_or_else(|| anyhow!("{} is missing", path.display()))?.f32(n) };
    let pool = match m.pool {
        Some(p) => Some(PoolModel {
            std: Standardizer { mu: get("pool.mu")?, sd: get("pool.sd")? },
            lr: Logistic { d: p.d, coef: get("pool.coef")?, intercept: get("pool.intercept")?, classes: p.classes, iters: p.iters },
        }),
        None => None,
    };
    let attn = match m.attn {
        Some(a) => Some(AttnModel {
            hidden: a.hidden,
            queries: a.queries,
            pair: a.pair,
            std: Standardizer { mu: get("attn.mu")?, sd: get("attn.sd")? },
            q: get("attn.q")?,
            w: get("attn.w")?,
            b: get("attn.b")?,
            classes: a.classes,
            epochs: a.epochs,
        }),
        None => None,
    };
    let sparse = match m.sparse {
        Some(sp) => Some(SparseLr { coef: get("sparse.coef")?, intercept: get("sparse.intercept")?, classes: sp.classes, iters: sp.iters }),
        None => None,
    };
    let neighbors = match m.neighbors {
        Some(kn) => Some(Neighbors::new(kn.d, get("neighbors.vecs")?, get("neighbors.y")?.into_iter().map(|c| c as u32).collect())),
        None => None,
    };
    let refs = if m.ref_dim > 0 { get("refs")? } else { vec![] };
    let rules = match m.rules {
        Some(mut r) => {
            r.compile()?;
            Some(Arc::new(r))
        }
        None => None,
    };
    let blend = match m.blend {
        Some(_) => Some(BlendStd {
            shape: Standardizer { mu: get("blend.shape_mu")?, sd: get("blend.shape_sd")? },
            emb: Standardizer { mu: get("blend.emb_mu")?, sd: get("blend.emb_sd")? },
        }),
        None => None,
    };
    let memo = match m.memo {
        Some(mm) => {
            let t = st.as_ref().ok_or_else(|| anyhow!("{} is missing", path.display()))?;
            let exact: Vec<u64> = t.u32("memo.exact")?.chunks_exact(2).map(|c| (c[0] as u64) << 32 | c[1] as u64).collect();
            let y: Vec<u32> = get("memo.y")?.into_iter().map(|c| c as u32).collect();
            ensure!(exact.len() == y.len(), "{}: memo arrays disagree", path.display());
            Some(crate::memo::MemoModel { thr_sim: mm.thr_sim, exact, sig: t.u32("memo.sig")?, y, conf: get("memo.conf")? })
        }
        None => None,
    };
    let net = match m.net {
        Some(mm) => Some(crate::netdesign::NetModel {
            pool_std: Standardizer { mu: get("net.pool_mu")?, sd: get("net.pool_sd")? },
            shape_std: Standardizer { mu: get("net.shape_mu")?, sd: get("net.shape_sd")? },
            h: mm.h,
            w1: get("net.w1")?,
            b1: get("net.b1")?,
            w2: get("net.w2")?,
            b2: get("net.b2")?,
            classes: mm.classes,
            epochs: mm.epochs,
        }),
        None => None,
    };
    Ok(Fit { readout: m.readout, pool, attn, sparse, neighbors, rules, blend, memo, net, checklist: m.checklist, class_thr: m.class_thr, mode: m.mode, thr: m.thr, refs, ref_dim: m.ref_dim, nn_floor: m.nn_floor, report: m.report })
}

/// Load a snapshot. The encoder must match the one that computed its features. `lead` replaces the recorded one
/// (which is only a description); `settings_override` edits the stored settings (e.g. the audit rate).
pub fn load(dir: &Path, encoder: Arc<dyn Encoder>, lead: Option<Box<dyn Lead>>,
            settings_override: impl FnOnce(&mut Settings)) -> Result<Proxy> {
    let p = dir.join("snapshot.json");
    let pf: SnapshotFile = serde_json::from_slice(&std::fs::read(&p).with_context(|| p.display().to_string())?)
        .with_context(|| format!("{}: not a readable snapshot", p.display()))?;
    if pf.format != FORMAT {
        bail!("{}: format {:?}, this understudy reads {FORMAT}", p.display(), pf.format);
    }
    let fp = encoder.fingerprint();
    ensure!(pf.encoder == fp, "{}: the snapshot's features come from encoder {}, but this one is {}", p.display(), pf.encoder, fp);
    ensure!(pf.featurizer.hidden == encoder.hidden(), "hidden size mismatch");
    let mut settings = pf.settings;
    settings_override(&mut settings);
    let mut proxy = Proxy::new(settings, pf.featurizer, encoder, lead);
    proxy.rng = pf.rng;
    proxy.seq = pf.seq;
    let mut encs: HashMap<String, Arc<Encoding>> = HashMap::new();
    for sf in pf.understudents {
        let mut examples = Vec::with_capacity(sf.examples.len());
        for e in sf.examples {
            let enc = match encs.get(&e.enc) {
                Some(x) => x.clone(),
                None => {
                    let x = Arc::new(read_encoding(&dir.join("states").join(format!("{}.safetensors", e.enc)))?);
                    encs.insert(e.enc.clone(), x.clone());
                    x
                }
            };
            examples.push(Example { seq: e.seq, kind: e.kind, state: e.state, label: e.label, probs: e.probs, enc_key: e.enc, enc, text_len: e.text_len, group: e.group });
        }
        ensure!(sf.question.key() == sf.key, "question {} doesn't hash to its key {}", sf.name, sf.key);
        let mut tiers = proxy.new_tiers();
        if proxy.settings.fit.understudent.is_tiered() {
            ensure!(sf.tiers.len() == tiers.len(), "question {} has {} tiers, the snapshot's ladder {}", sf.name, sf.tiers.len(), tiers.len());
            for (t, tf) in tiers.iter_mut().zip(sf.tiers) {
                ensure!(t.kind == tf.kind, "question {}: tier {} where the ladder has {}", sf.name, tf.kind.name(), t.kind.name());
                let fit = read_fit(&dir.join("fits").join(format!("{}.{}.safetensors", sf.key, t.kind.name())), tf.fit)?;
                t.serving = Serving { fit: Arc::new(fit), audits: tf.audits.into_iter().collect(), demoted: tf.demoted };
            }
        } else {
            let fit = read_fit(&dir.join("fits").join(format!("{}.safetensors", sf.key)), sf.fit)?;
            tiers[0].serving = Serving { fit: Arc::new(fit), audits: sf.audits.into_iter().collect(), demoted: sf.demoted };
        }
        let mut st = Understudent::new(&sf.name, sf.question, proxy.settings.fit.understudent, tiers);
        (st.key, st.since_refit, st.since_cheap, st.examples) = (sf.key, sf.since_refit, sf.since_cheap, examples);
        (st.serving_idx, st.lead_only, st.race) = (sf.serving, sf.lead_only, sf.race);
        st.rules = sf.rules_state;
        st.attempts = sf.attempts;
        proxy.understudents.push(st);
    }
    Ok(proxy)
}
