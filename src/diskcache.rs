//! A disk cache of encoded texts (`--encode-cache DIR`), shared across systems, runs and passes.
//!
//! What it stores is exactly what an understudent reads: one `Part` per text (a state field) for one featurizer
//! configuration, i.e. probe's layer-8 token mean, deep's pooled layers plus its capped bf16 token states, lite's
//! input-embedding pools. The key is the SHA-256 of the encoder's fingerprint (checkpoint, precision), the
//! featurizer's layers, attention layer, pooling/token/chunking switches, and the text; the slot and pair fields are
//! left out, so a lone probe and an `auto` race's probe tier share entries. Each entry also records the encode
//! microseconds per state of the batch that computed it, so a replay served from the cache reports the same costs.
//! Entries are written to a temporary file and renamed, so concurrent processes can share a directory.

use crate::encoder::{Featurizer, Part};
use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"BLNKENC1";

pub struct DiskCache {
    dir: PathBuf,
    prefix: String,
}

impl DiskCache {
    /// Open (creating) a cache directory for an encoder with this fingerprint.
    pub fn open(dir: &Path, fingerprint: &Value) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| dir.display().to_string())?;
        Ok(DiskCache { dir: dir.to_path_buf(), prefix: format!("{MAGIC:?}|{fingerprint}") })
    }

    /// The entry name of a text under a featurizer.
    pub fn key(&self, fz: &Featurizer, text: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.prefix.as_bytes());
        h.update(format!("|layers={:?}|attn={}|mean_only={}|tokens={}|input={}|chunk={}|hidden={}|", fz.layers, fz.attn_layer, fz.mean_only, fz.tokens, fz.input, fz.chunk, fz.hidden).as_bytes());
        h.update(text.as_bytes());
        crate::question::hex(&h.finalize())
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(&key[..2]).join(format!("{key}.part"))
    }

    /// A cached part and the per-state encode microseconds recorded with it.
    pub fn get(&self, key: &str) -> Option<(Part, f64)> {
        let b = std::fs::read(self.path(key)).ok()?;
        decode(&b).ok()
    }

    pub fn put(&self, key: &str, part: &Part, cost_us: f64) -> Result<()> {
        let path = self.path(key);
        if path.exists() {
            return Ok(());
        }
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, encode(part, cost_us))?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

fn encode(p: &Part, cost_us: f64) -> Vec<u8> {
    let mut b = Vec::with_capacity(40 + p.pool.len() * 4 + p.states.len() * 2);
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&cost_us.to_le_bytes());
    b.extend_from_slice(&(p.tokens as u64).to_le_bytes());
    b.extend_from_slice(&(p.pool.len() as u64).to_le_bytes());
    b.extend_from_slice(&(p.states.len() as u64).to_le_bytes());
    p.pool.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
    p.states.iter().for_each(|x| b.extend_from_slice(&x.to_bits().to_le_bytes()));
    b
}

fn decode(b: &[u8]) -> Result<(Part, f64)> {
    if b.len() < 40 || &b[..8] != MAGIC {
        bail!("not an understudy encode-cache entry");
    }
    let u = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    let cost = f64::from_le_bytes(b[8..16].try_into().unwrap());
    let (tokens, np, ns) = (u(16) as usize, u(24) as usize, u(32) as usize);
    if b.len() != 40 + np * 4 + ns * 2 {
        bail!("truncated encode-cache entry");
    }
    let pool = b[40..40 + np * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let states = b[40 + np * 4..].chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap()))).collect();
    Ok((Part { pool, tokens, states }, cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let p = Part { pool: vec![1.5, -2.0, f32::MIN_POSITIVE], tokens: 7, states: vec![bf16::from_f32(0.5), bf16::from_f32(-3.0)] };
        let (q, c) = decode(&encode(&p, 12.5)).unwrap();
        assert_eq!((q, c), (p, 12.5));
        assert!(decode(b"junk").is_err());
    }
}
