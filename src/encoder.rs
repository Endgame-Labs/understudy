//! What the understudents read: hunch's state text, clipped like hunch's, run through an encoder that returns
//! per-token hidden states, reduced to an `Encoding` (pool vectors plus capped bf16 token states).

use crate::pyjson;
use anyhow::{Result, ensure};
use half::{bf16, slice::HalfFloatSliceExt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// The encoder reads at most this many characters of a state: longer ones keep their head and tail.
pub const CLIP_CHARS: usize = 3000;
/// Token states kept per text for the attention readout: the first two thirds and last third beyond this.
pub const TOKEN_CAP: usize = 512;
/// A state's text is read whole (by `keywords`, `lite`, chunked encoders) up to this many characters; longer ones keep
/// their first two thirds and last third of it.
pub const FULL_CHARS: usize = 20_000;
/// A chunked encoder splits a text over `CLIP_CHARS` into at most this many chunks of `CLIP_CHARS` characters.
pub const MAX_CHUNKS: usize = 8;

/// hunch's `state_text`: a string as is; a tool call and its output as "call\noutput" plus other fields as sorted
/// "name: value" lines; anything else as Python JSON.
pub fn state_text(state: &Value) -> String {
    if let Value::String(s) = state {
        return s.clone();
    }
    if let Value::Object(m) = state {
        let call = if m.contains_key("call") { m.get("call") } else { m.get("tool_call") };
        if let (Some(call), Some(out)) = (call.filter(|c| !c.is_null()), m.get("output")) {
            let mut text = format!("{}\n{}", pyjson::py_str(call), pyjson::py_str(out));
            let mut keys: Vec<&String> = m.keys().filter(|k| !matches!(k.as_str(), "call" | "tool_call" | "output")).collect();
            keys.sort_by(|a, b| a.chars().cmp(b.chars()));
            for k in keys {
                let v = &m[k];
                let s = if let Value::String(s) = v { s.clone() } else { pyjson::dumps(v) };
                text.push_str(&format!("\n{k}: {s}"));
            }
            return text;
        }
    }
    pyjson::dumps(state)
}

/// hunch's `encoder_text`: a text over `CLIP_CHARS` characters keeps its first two thirds and last third of that
/// budget, joined by "\n…\n".
pub fn encoder_text(text: &str) -> String {
    let n = text.chars().count();
    if n <= CLIP_CHARS {
        return text.to_string();
    }
    let head = CLIP_CHARS * 2 / 3;
    let tail = CLIP_CHARS - head;
    let h: String = text.chars().take(head).collect();
    let t: String = text.chars().skip(n - tail).collect();
    format!("{h}\n…\n{t}")
}

/// A state text capped at `FULL_CHARS` characters (head two thirds, tail one third, joined by "\n…\n").
pub fn full_text(text: &str) -> String {
    clip(text, FULL_CHARS)
}

fn clip(text: &str, budget: usize) -> String {
    let n = text.chars().count();
    if n <= budget {
        return text.to_string();
    }
    let head = budget * 2 / 3;
    let tail = budget - head;
    let h: String = text.chars().take(head).collect();
    let t: String = text.chars().skip(n - tail).collect();
    format!("{h}\n…\n{t}")
}

/// Split a text for chunked encoding: one chunk when it fits in `CLIP_CHARS`, else consecutive chunks of at most
/// `CLIP_CHARS` characters, each cut after the last line break in the final 30% of its window (a hard cut when there
/// is none), at most `MAX_CHUNKS` of them (a text that would need more is cut hard at `CLIP_CHARS`, then clipped to
/// `MAX_CHUNKS * CLIP_CHARS` characters like `encoder_text`).
pub fn chunk_text(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= CLIP_CHARS {
        return vec![text.to_string()];
    }
    let cut = |soft: bool| -> Vec<String> {
        let (mut out, mut at) = (vec![], 0);
        while at < chars.len() {
            let mut end = (at + CLIP_CHARS).min(chars.len());
            if soft && end < chars.len() {
                let lo = at + CLIP_CHARS * 7 / 10;
                if let Some(nl) = (lo..end).rev().find(|&i| chars[i] == '\n') {
                    end = nl + 1;
                }
            }
            out.push(chars[at..end].iter().collect());
            at = end;
        }
        out
    };
    let soft = cut(true);
    if soft.len() <= MAX_CHUNKS {
        return soft;
    }
    let mut hard = cut(false);
    hard.truncate(MAX_CHUNKS);
    hard
}

/// hunch's `text_hash`: the first 16 hex digits of the SHA-256 of the text.
pub fn text_hash(text: &str) -> String {
    crate::question::hex(&Sha256::digest(text.as_bytes()))[..16].to_string()
}

/// A state's identity (hunch's `state_hash`): the hash of its full state text.
pub fn state_hash(state: &Value) -> String {
    text_hash(&state_text(state))
}

/// Per-token hidden states of one text: `states[i]` is [tokens x hidden] after the i-th requested layer.
pub struct Tokens {
    pub tokens: usize,
    pub hidden: usize,
    pub states: Vec<Vec<f32>>,
}

/// Something that returns per-token hidden states: EmbeddingGemma 2 (`Eg2`) or a stand-in for tests (`Fake`).
pub trait Encoder: Send + Sync {
    fn hidden(&self) -> usize;
    /// What features depend on (checkpoint, precision); probes refuse to mix encoders.
    fn fingerprint(&self) -> Value;
    /// States after each of `layers` (in that order) for each text.
    fn encode(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<Tokens>>;
    /// The input token embeddings alone (no transformer layer), as one "layer" of states: the `lite` understudent.
    fn encode_input(&self, texts: &[&str]) -> Result<Vec<Tokens>>;
    /// Like `encode`, but each text also yields a handle `resume` can continue to deeper layers without recomputing
    /// the ones run. An encoder that can't resume returns `None` handles.
    fn encode_resumable(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<(Tokens, Option<Resume>)>> {
        Ok(self.encode(texts, layers)?.into_iter().map(|t| (t, None)).collect())
    }
    /// States after `layers` (each deeper than the ones `from` ran) for texts started by `encode_resumable`. Without
    /// handles it recomputes from the first layer.
    fn resume(&self, texts: &[&str], from: Vec<Option<Resume>>, layers: &[usize]) -> Result<Vec<Tokens>> {
        let _ = from;
        self.encode(texts, layers)
    }
}

/// A forward pass stopped part-way (opaque to everything but the encoder that made it).
pub struct Resume(pub Box<dyn std::any::Any + Send + Sync>);

/// No encoder at all (the `keywords` understudent never reads token states): every call is an error.
pub struct NoEncoder;

impl Encoder for NoEncoder {
    fn hidden(&self) -> usize {
        0
    }

    fn fingerprint(&self) -> Value {
        json!({"encoder": "none"})
    }

    fn encode(&self, _: &[&str], _: &[usize]) -> Result<Vec<Tokens>> {
        anyhow::bail!("this understudent has no encoder")
    }

    fn encode_input(&self, _: &[&str]) -> Result<Vec<Tokens>> {
        anyhow::bail!("this understudent has no encoder")
    }
}

/// EmbeddingGemma 2 through the vendored eg2cpu.
pub struct Eg2 {
    pub model: eg2cpu::Model,
}

impl Eg2 {
    pub fn load_default() -> Result<Self> {
        Ok(Eg2 { model: eg2cpu::Model::load_default()? })
    }
}

impl Encoder for Eg2 {
    fn hidden(&self) -> usize {
        self.model.cfg.hidden
    }

    fn fingerprint(&self) -> Value {
        let bf16 = std::env::var("EG2_BF16").unwrap_or_default().trim().to_lowercase();
        let precision = if matches!(bf16.as_str(), "" | "off" | "0" | "none") { "fp32".to_string() } else { format!("bf16:{bf16}") };
        json!({"encoder": "eg2cpu", "revision": self.model.revision(), "precision": precision})
    }

    fn encode(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<Tokens>> {
        let outs = self.model.embed_batch_tokens(texts, layers)?;
        Ok(outs
            .into_iter()
            .map(|o| {
                let states = layers.iter().map(|l| o.states.iter().find(|(x, _)| x == l).expect("requested layer").1.clone()).collect();
                Tokens { tokens: o.tokens, hidden: o.hidden, states }
            })
            .collect())
    }

    fn encode_resumable(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<(Tokens, Option<Resume>)>> {
        let outs = self.model.embed_batch_tokens_partial(texts, layers)?;
        Ok(outs
            .into_iter()
            .map(|mut o| {
                let taken = std::mem::take(&mut o.states);
                let states = layers.iter().map(|l| taken.iter().find(|(x, _)| x == l).expect("requested layer").1.clone()).collect();
                (Tokens { tokens: o.tokens, hidden: o.hidden, states }, Some(Resume(Box::new(o))))
            })
            .collect())
    }

    fn resume(&self, texts: &[&str], from: Vec<Option<Resume>>, layers: &[usize]) -> Result<Vec<Tokens>> {
        if from.iter().any(Option::is_none) {
            return self.encode(texts, layers);
        }
        let parts: Vec<eg2cpu::TokenPartial> = from
            .into_iter()
            .map(|r| *r.unwrap().0.downcast::<eg2cpu::TokenPartial>().expect("a handle from this encoder"))
            .collect();
        let outs = self.model.resume_batch_tokens(parts, layers)?;
        Ok(outs
            .into_iter()
            .map(|o| {
                let states = layers.iter().map(|l| o.states.iter().find(|(x, _)| x == l).expect("requested layer").1.clone()).collect();
                Tokens { tokens: o.tokens, hidden: o.hidden, states }
            })
            .collect())
    }

    fn encode_input(&self, texts: &[&str]) -> Result<Vec<Tokens>> {
        use rayon::prelude::*;
        let h = self.model.cfg.hidden;
        texts
            .par_iter()
            .map(|t| {
                let ids = self.model.tokenize(t)?;
                Ok(Tokens { tokens: ids.len(), hidden: h, states: vec![self.model.input_embeddings(&ids)] })
            })
            .collect()
    }
}

/// A deterministic stand-in: one token per whitespace-separated word (plus a start token), each word a fixed
/// pseudo-random vector; layer `l` scales it by `1 + l / 10` and adds a small position term.
pub struct Fake {
    pub hidden: usize,
}

impl Fake {
    pub fn word(&self, w: &str) -> Vec<f32> {
        let seed = u64::from_str_radix(&text_hash(w), 16).unwrap();
        let mut r = crate::rng::Xoshiro::new(seed);
        (0..self.hidden).map(|_| r.normal() as f32).collect()
    }
}

impl Encoder for Fake {
    fn hidden(&self) -> usize {
        self.hidden
    }

    fn fingerprint(&self) -> Value {
        json!({"encoder": "fake", "hidden": self.hidden})
    }

    fn encode_input(&self, texts: &[&str]) -> Result<Vec<Tokens>> {
        self.encode(texts, &[0])
    }

    fn encode(&self, texts: &[&str], layers: &[usize]) -> Result<Vec<Tokens>> {
        Ok(texts
            .iter()
            .map(|t| {
                let words: Vec<&str> = std::iter::once("<bos>").chain(t.split_whitespace()).collect();
                let vecs: Vec<Vec<f32>> = words.iter().map(|w| self.word(w)).collect();
                let states = layers
                    .iter()
                    .map(|&l| {
                        let s = 1.0 + l as f32 / 10.0;
                        vecs.iter().enumerate().flat_map(|(i, v)| v.iter().map(move |x| x * s + 0.01 * i as f32)).collect()
                    })
                    .collect();
                Tokens { tokens: words.len(), hidden: self.hidden, states }
            })
            .collect())
    }
}

/// Where in an `Encoding` a featurizer's neural parts live. A single understudent uses `Main`; a ladder keeps its `deep`
/// tier in `Main`, `probe`'s layer-8 mean in `Mean` and `lite`'s input-embedding pools in `Stat`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Slot {
    #[default]
    Main,
    Mean,
    Stat,
}

/// How states become features: the layers read, the attention readout's layer, and an optional pair of fields.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Featurizer {
    pub layers: Vec<usize>,
    pub attn_layer: usize,
    pub pair: Option<(String, String)>,
    pub hidden: usize,
    /// Pool features are the token mean only (no max): `probe`, `neighbors`.
    #[serde(default)]
    pub mean_only: bool,
    /// Keep the attention layer's token states (only the attention readout reads them).
    #[serde(default = "yes")]
    pub tokens: bool,
    /// Read the input embedding table instead of hidden states (`lite`; `layers` is then `[0]`).
    #[serde(default)]
    pub input: bool,
    /// Encode a text over `CLIP_CHARS` as consecutive chunks and combine them (see `Featurizer::part_chunks`)
    /// instead of clipping it like hunch.
    #[serde(default)]
    pub chunk: bool,
    #[serde(default)]
    pub slot: Slot,
}

fn yes() -> bool {
    true
}

/// One encoded text: per layer its token mean then token max ([2 x hidden x layers]), and the attention layer's
/// token states as bf16, capped at `TOKEN_CAP` tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub pool: Vec<f32>,
    pub tokens: usize,
    pub states: Vec<bf16>,
}

/// One state's encoding. Every field is optional work: an understudent fills only what its tiers read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Encoding {
    /// Neural parts (one per text, two for a pair) of a single understudent, or of a ladder's `deep` tier.
    pub parts: Vec<Part>,
    /// A ladder's `probe` tier: layer-8 token means.
    pub mean: Vec<Part>,
    /// A ladder's `lite` tier: input-embedding pools.
    pub stat: Vec<Part>,
    /// The hashed n-gram vector (`keywords`, a ladder's `keywords` tier).
    pub sparse: Option<crate::keywords::Sparse>,
    /// What the `rules` tier reads: the state's text and top-level fields.
    pub rules: Option<Arc<crate::rules::RuleInput>>,
    /// The structural feature vector (`shape`, `blend`).
    pub shape: Option<Arc<Vec<f32>>>,
    /// The retrieval key (`memo`).
    pub memo: Option<Arc<crate::memo::MemoKey>>,
    /// What names a `checklist` rule's features: the state's full texts and the field name each structural slot holds.
    /// In memory only (never saved in a snapshot): a reloaded example's rules fall back to `gram#<bucket>` names.
    pub aux: Option<Arc<Aux>>,
}

/// See `Encoding::aux`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Aux {
    pub texts: Vec<String>,
    pub slots: Vec<String>,
}

impl Encoding {
    pub fn slot(&self, s: Slot) -> &[Part] {
        match s {
            Slot::Main => &self.parts,
            Slot::Mean => &self.mean,
            Slot::Stat => &self.stat,
        }
    }

    pub fn set_slot(&mut self, s: Slot, parts: Vec<Part>) {
        match s {
            Slot::Main => self.parts = parts,
            Slot::Mean => self.mean = parts,
            Slot::Stat => self.stat = parts,
        }
    }
}

impl Tokens {
    /// The states of the layers at positions `idx` of `states`, in that order.
    pub fn select(&self, idx: &[usize]) -> Tokens {
        Tokens { tokens: self.tokens, hidden: self.hidden, states: idx.iter().map(|&i| self.states[i].clone()).collect() }
    }
}

/// How many token states each of several chunks keeps out of `cap`: every chunk its own length if all fit, else
/// the shortest chunks keep all theirs and the rest split what is left evenly (so chunks stay evenly represented).
fn budgets(lens: &[usize], cap: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| lens[i]);
    let (mut left, mut out) = (cap, vec![0; lens.len()]);
    for (r, &i) in order.iter().enumerate() {
        out[i] = lens[i].min(left / (lens.len() - r));
        left -= out[i];
    }
    out
}

impl Featurizer {
    pub fn new(layers: Vec<usize>, attn_layer: Option<usize>, pair: Option<(String, String)>, hidden: usize) -> Result<Self> {
        ensure!(!layers.is_empty(), "at least one layer");
        let attn_layer = attn_layer.unwrap_or(layers[layers.len() / 2]);
        ensure!(layers.contains(&attn_layer), "the attention layer {attn_layer} must be one of the layers {layers:?}");
        Ok(Featurizer { layers, attn_layer, pair, hidden, mean_only: false, tokens: true, input: false, chunk: false, slot: Slot::Main })
    }

    /// The featurizer a single understudent reads: `deep`/`ladder` the given layers (default 4,8,12) with mean+max pooling,
    /// token states and chunked encoding of long texts; `probe`/`neighbors` layer 8's mean only; `lite` the input
    /// embeddings' mean and max over chunks; `keywords` none (it only needs the state texts). A ladder's own tiers come
    /// from `tier`.
    pub fn for_understudent(kind: crate::understudent::UnderstudentKind, layers: Option<Vec<usize>>, attn_layer: Option<usize>, pair: Option<(String, String)>,
                       hidden: usize) -> Result<Self> {
        use crate::understudent::UnderstudentKind::*;
        let free = matches!(kind, Deep | Ladder | Auto | Net);
        ensure!(free || (layers.is_none() && attn_layer.is_none()), "--layers and --attn-layer only apply to deep and ladder, not {}", kind.name());
        let mut f = match kind {
            Deep | Ladder | Auto | Net => Featurizer::new(layers.unwrap_or_else(|| vec![4, 8, 12]), attn_layer, pair, hidden)?,
            Probe | Neighbors => Featurizer::new(vec![8], None, pair, hidden)?,
            Lite | Blend | Keywords | Rules | Shape | Memo | Checklist => Featurizer::new(vec![0], None, pair, hidden)?,
        };
        f.tokens = free;
        f.mean_only = matches!(kind, Probe | Neighbors);
        f.input = matches!(kind, Lite | Blend);
        f.chunk = matches!(kind, Deep | Ladder | Auto | Lite | Blend | Net);
        Ok(f)
    }

    /// A ladder tier's featurizer; `self` is the ladder's base (deep's: layers, attention layer, pair, chunking).
    pub fn tier(&self, kind: crate::understudent::UnderstudentKind) -> Result<Featurizer> {
        use crate::understudent::UnderstudentKind::*;
        Ok(match kind {
            // net reads deep's encoding (same featurizer, same slot, same encode-cache entries)
            Deep | Net => Featurizer { slot: Slot::Main, ..self.clone() },
            Probe => Featurizer { mean_only: true, tokens: false, slot: Slot::Mean, ..Featurizer::new(vec![8], None, self.pair.clone(), self.hidden)? },
            Lite | Blend => Featurizer { tokens: false, input: true, chunk: true, slot: Slot::Stat, ..Featurizer::new(vec![0], None, self.pair.clone(), self.hidden)? },
            Shape | Memo | Checklist => Featurizer { tokens: false, ..Featurizer::new(vec![0], None, self.pair.clone(), self.hidden)? },
            Keywords => Featurizer { tokens: false, ..Featurizer::new(vec![0], None, self.pair.clone(), self.hidden)? },
            Rules => Featurizer { tokens: false, ..Featurizer::new(vec![0], None, self.pair.clone(), self.hidden)? },
            other => anyhow::bail!("{} can't be a ladder tier (tiers: rules, keywords, lite, probe, deep, shape, blend, memo, net, checklist)", other.name()),
        })
    }

    fn attn_pos(&self) -> usize {
        self.layers.iter().position(|&l| l == self.attn_layer).unwrap()
    }

    /// A state's whole text(s), capped at `FULL_CHARS`: the one state text, or each pair field's.
    pub fn full_texts(&self, state: &Value) -> Vec<String> {
        match &self.pair {
            None => vec![full_text(&state_text(state))],
            Some((a, b)) => [a, b].iter().map(|f| full_text(&state_text(state.get(f.as_str()).unwrap_or(&Value::Null)))).collect(),
        }
    }

    /// The encoder texts of each field of a state (from `full_texts`): its chunks if this featurizer chunks, else the
    /// one text clipped like hunch's `encoder_text`.
    pub fn field_texts(&self, full: &[String]) -> Vec<Vec<String>> {
        full.iter().map(|t| if self.chunk { chunk_text(t) } else { vec![encoder_text(t)] }).collect()
    }

    /// What the encoding cache keys a state by: the hash of its (capped) texts. A short state's key is the same
    /// as before chunking existed (hunch's `text_hash` of its encoder text(s)).
    pub fn cache_key(&self, full: &[String]) -> String {
        text_hash(&full.join("\u{0}"))
    }

    /// Encode many states, each a list of fields, each a list of chunk texts, in one batch: one `Part` per field.
    pub fn encode_fields(&self, enc: &dyn Encoder, states: &[Vec<Vec<String>>]) -> Result<Vec<Vec<Part>>> {
        let flat: Vec<&str> = states.iter().flatten().flatten().map(String::as_str).collect();
        if flat.is_empty() {
            return Ok(vec![]);
        }
        let toks = if self.input { enc.encode_input(&flat)? } else { enc.encode(&flat, &self.layers)? };
        Ok(self.group(states, toks))
    }

    /// Group the token states of `states`' chunk texts (in order) into parts, one per field.
    pub fn group(&self, states: &[Vec<Vec<String>>], toks: Vec<Tokens>) -> Vec<Vec<Part>> {
        let mut it = toks.into_iter();
        states
            .iter()
            .map(|fields| {
                fields
                    .iter()
                    .map(|chunks| {
                        let ts: Vec<Tokens> = chunks.iter().map(|_| it.next().expect("one output per text")).collect();
                        self.part_chunks(&ts)
                    })
                    .collect()
            })
            .collect()
    }

    /// Encode many states (each a list of texts, one chunk per field) in one batch into encodings whose `slot` holds
    /// the parts.
    pub fn encode(&self, enc: &dyn Encoder, states: &[Vec<String>]) -> Result<Vec<Arc<Encoding>>> {
        let fields: Vec<Vec<Vec<String>>> = states.iter().map(|t| t.iter().map(|x| vec![x.clone()]).collect()).collect();
        Ok(self
            .encode_fields(enc, &fields)?
            .into_iter()
            .map(|parts| {
                let mut e = Encoding::default();
                e.set_slot(self.slot, parts);
                Arc::new(e)
            })
            .collect())
    }

    /// One field's part from its chunks' token states (`states[i]` per requested layer).
    ///
    /// - pool: per layer, the token mean is the token-count-weighted mean over chunks (a sum over all tokens divided
    ///   by their total), the token max is the max over chunks; so one chunk gives exactly the plain encoding.
    /// - attention tokens (layer `attn_layer`): the chunks' token states in order, at most `TOKEN_CAP` in all. Each
    ///   chunk keeps up to its fair share of the cap (`budgets`: short chunks keep all their tokens, longer ones
    ///   split the rest evenly), the first two thirds of its share from its head and the last third from its tail.
    pub fn part_chunks(&self, chunks: &[Tokens]) -> Part {
        let h = chunks[0].hidden;
        let n: usize = chunks.iter().map(|c| c.tokens).sum();
        let mut pool = Vec::with_capacity(self.per_layer() * self.layers.len());
        for li in 0..self.layers.len() {
            let mut mean = vec![0f64; h];
            let mut max = vec![f32::NEG_INFINITY; h];
            for c in chunks {
                for row in c.states[li].chunks_exact(h) {
                    for j in 0..h {
                        mean[j] += row[j] as f64;
                        max[j] = max[j].max(row[j]);
                    }
                }
            }
            pool.extend(mean.iter().map(|&m| (m / n as f64) as f32));
            if !self.mean_only {
                pool.extend(max);
            }
        }
        if !self.tokens {
            return Part { pool, tokens: 0, states: vec![] };
        }
        let ap = self.attn_pos();
        let share = budgets(&chunks.iter().map(|c| c.tokens).collect::<Vec<_>>(), TOKEN_CAP);
        let keeps: Vec<Vec<usize>> = chunks
            .iter()
            .zip(&share)
            .map(|(c, &b)| {
                if c.tokens <= b {
                    (0..c.tokens).collect()
                } else {
                    let head = b * 2 / 3;
                    (0..head).chain(c.tokens - (b - head)..c.tokens).collect()
                }
            })
            .collect();
        let kept: usize = keeps.iter().map(Vec::len).sum();
        let mut states = vec![bf16::ZERO; kept * h];
        let mut r = 0;
        for (c, keep) in chunks.iter().zip(&keeps) {
            let s = &c.states[ap];
            for &i in keep {
                states[r * h..(r + 1) * h].convert_from_f32_slice(&s[i * h..(i + 1) * h]);
                r += 1;
            }
        }
        Part { pool, tokens: kept, states }
    }

    /// The parts of an encoding this featurizer reads.
    pub fn parts<'a>(&self, e: &'a Encoding) -> &'a [Part] {
        e.slot(self.slot)
    }

    /// Pool features: a part's pool vector, or `[x, y, x − y, x ⊙ y]` of a pair's.
    pub fn pool_features(&self, e: &Encoding) -> Vec<f32> {
        match self.parts(e) {
            [p] => p.pool.clone(),
            [x, y] => {
                let (x, y) = (&x.pool, &y.pool);
                let mut f = Vec::with_capacity(4 * x.len());
                f.extend_from_slice(x);
                f.extend_from_slice(y);
                f.extend(x.iter().zip(y).map(|(a, b)| a - b));
                f.extend(x.iter().zip(y).map(|(a, b)| a * b));
                f
            }
            _ => unreachable!("one or two parts"),
        }
    }

    /// Pool features per layer: mean and max, or the mean alone.
    fn per_layer(&self) -> usize {
        self.hidden * if self.mean_only { 1 } else { 2 }
    }

    pub fn pool_dim(&self) -> usize {
        self.per_layer() * self.layers.len() * if self.pair.is_some() { 4 } else { 1 }
    }

    /// A part's token mean at the attention layer.
    pub fn mid_mean<'a>(&self, p: &'a Part) -> &'a [f32] {
        let o = self.per_layer() * self.attn_pos();
        &p.pool[o..o + self.hidden]
    }

    /// The novelty vector: the L2-normalized token mean at the attention (middle) layer, of the concatenated
    /// parts for a pair.
    pub fn novelty(&self, e: &Encoding) -> Vec<f32> {
        let mut v: Vec<f32> = self.parts(e).iter().flat_map(|p| self.mid_mean(p).iter().copied()).collect();
        let n = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32 + 1e-12;
        v.iter_mut().for_each(|x| *x /= n);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_text_like_hunch() {
        assert_eq!(state_text(&json!("hi")), "hi");
        let s = json!({"output": "ok", "call": "ls", "exit": 0, "cwd": "/tmp"});
        assert_eq!(state_text(&s), "ls\nok\ncwd: /tmp\nexit: 0");
        assert_eq!(state_text(&json!({"b": 1, "a": [1, "x"]})), r#"{"a": [1, "x"], "b": 1}"#);
        assert_eq!(state_text(&json!({"call": null, "tool_call": "x", "output": "y"})), r#"{"call": null, "output": "y", "tool_call": "x"}"#);
        let long: String = "a".repeat(2500) + &"b".repeat(2500);
        let t = encoder_text(&long);
        assert_eq!(t.chars().count(), 3003);
        assert!(t.starts_with("aaa") && t.ends_with("bbb") && t.contains("\n…\n"));
        assert_eq!(text_hash("hello"), "2cf24dba5fb0a30e");
    }

    #[test]
    fn part_caps_tokens() {
        let f = Featurizer::new(vec![1, 2], None, None, 4).unwrap();
        let words = vec!["w"; 700].join(" ");
        let e = f.encode(&Fake { hidden: 4 }, &[vec![words]]).unwrap();
        assert_eq!(e[0].parts[0].tokens, TOKEN_CAP);
        assert_eq!(e[0].parts[0].pool.len(), 2 * 4 * 2);
        assert_eq!(f.attn_layer, 2);
    }
}
