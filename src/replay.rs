//! Replay: a stream through a fresh proxy whose lead is the stream's recorded answers, one item at a time,
//! with synchronous refits, so a run is reproducible. Writes one decision per item and returns a summary.

use crate::proxy::{Decision, Proxy};
use crate::stats::argmax;
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

/// Run every line of `stream` (`{"id", "state", "lead_probs"}`) through `proxy` (which holds one question
/// and a replay lead), writing decision lines to `out`. Returns the summary.
pub fn run(proxy: &mut Proxy, stream: impl BufRead, mut out: impl Write) -> Result<Value> {
    let t0 = std::time::Instant::now();
    let mut s = Summary::default();
    let mut serving: Option<Option<&'static str>> = None;
    for (i, line) in stream.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).with_context(|| format!("stream line {}", i + 1))?;
        let id = v.get("id").and_then(Value::as_str).ok_or_else(|| anyhow!("stream line {}: no id", i + 1))?;
        let state = v.get("state").ok_or_else(|| anyhow!("stream line {}: no state", i + 1))?;
        let o = proxy.answer(Some(id), state, None)?;
        let d = &o.decisions[0].1;
        let lead = v.get("lead_probs").and_then(Value::as_array).map(|a| argmax(&a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect::<Vec<_>>()));
        s.add(s.items, d, lead);
        if proxy.understudents[0].is_auto() {
            // from the next item on, the serving candidate is what the refit (if any) just chose
            let now = proxy.understudents[0].serving_name();
            if serving != Some(now) {
                s.serving_history.push(json!({"item": s.items + usize::from(!o.refits.is_empty()), "understudent": now.unwrap_or("none")}));
                serving = Some(now);
            }
        }
        if proxy.understudents[0].is_tiered() {
            for t in &proxy.understudents[0].tiers {
                if t.serving.mode().is_some() {
                    s.tier_graduated.entry(t.kind.name().to_string()).or_insert(s.items);
                }
            }
        }
        for (_, secs, rep) in &o.refits {
            let mut r = json!({"item": s.items, "secs": secs, "n": rep["n"], "readout": rep["readout"], "mode": rep["mode"],
                               "cv_agreement": rep["cv_agreement"], "cv_coverage": rep["cv_coverage"], "fit_secs": rep["fit_secs"],
                               "cv_agreement_by_readout": rep["cv_agreement_by_readout"]});
            if let Some(g) = rep.get("graduation") {
                r["graduation"] = g.clone();
            }
            if proxy.understudents[0].is_auto() {
                r["race"] = rep["race"].clone();
                r["serving"] = rep["serving"].clone();
                r["lead_only"] = rep["lead_only"].clone();
            }
            if let Some(ts) = rep.get("tiers").and_then(Value::as_object) {
                r["refit"] = rep["refit"].clone();
                r["tiers"] = Value::Object(
                    ts.iter()
                        .map(|(k, f)| {
                            let e = s.tier_refit.entry(k.clone()).or_default();
                            e.0 += f["secs"].as_f64().unwrap_or(0.0);
                            e.1 += 1;
                            (k.clone(), json!({"secs": f["secs"], "n": f["n"], "mode": f["mode"], "readout": f["readout"], "cv_agreement": f["cv_agreement"],
                                               "cv_coverage": f["cv_coverage"], "fit_secs": f["fit_secs"]}))
                        })
                        .collect(),
                );
            }
            s.refits.push(r);
        }
        serde_json::to_writer(&mut out, d)?;
        out.write_all(b"\n")?;
        s.items += 1;
    }
    out.flush()?;
    let st = &proxy.understudents[0];
    let mut sum = s.finish(t0.elapsed().as_secs_f64(), st.status());
    sum["understudent"] = json!(proxy.settings.fit.understudent.name());
    if let Some(c) = &proxy.rules {
        use std::sync::atomic::Ordering::SeqCst;
        sum["rules_llm"] = json!({"live_calls": c.live.load(SeqCst), "cached_calls": c.cached.load(SeqCst)});
        sum["rules"] = sum["final"]["rules"].clone();
        if let Some(l) = sum["final"]["rules"]["listing"].as_str() {
            eprintln!("understudy: final rule set (proposal {}):\n{l}", sum["final"]["rules"]["proposal"]);
        }
    }
    if st.is_auto() {
        sum["candidates"] = json!(st.tiers.iter().map(|t| t.kind.name()).collect::<Vec<_>>());
        sum["serving_history"] = json!(s.serving_history);
        sum["serving"] = json!(st.serving_name());
        sum["lead_only"] = json!(st.lead_only);
    }
    if st.is_tiered() {
        sum["tiers"] = json!(st.tiers.iter().map(|t| t.kind.name()).collect::<Vec<_>>());
    }
    Ok(sum)
}

/// Running counts over a decision stream (shared by replay and judge).
#[derive(Default)]
pub struct Summary {
    pub items: usize,
    pub local: usize,
    pub local_agree: usize,
    pub audited: usize,
    pub demotions: usize,
    pub graduated_at: Option<usize>,
    pub first_local: Option<usize>,
    pub us: Vec<u64>,
    /// Understudent-answered decisions by their `readout` (a ladder's tiers: "keywords", "deep:attn", ...).
    pub by_readout: std::collections::BTreeMap<String, usize>,
    pub refits: Vec<Value>,
    /// A ladder's understudent-answered decisions by tier (probe is "probe", deep "deep").
    pub by_tier: std::collections::BTreeMap<String, usize>,
    /// The item at whose end each ladder tier first had graduated.
    pub tier_graduated: std::collections::BTreeMap<String, usize>,
    /// Each ladder tier's refits: (total seconds, count).
    pub tier_refit: std::collections::BTreeMap<String, (f64, usize)>,
    /// `auto`: `{"item", "understudent"}` each time the serving candidate changed; it serves from `item` on.
    pub serving_history: Vec<Value>,
    was_demoted: bool,
}

impl Summary {
    pub fn add(&mut self, item: usize, d: &Decision, lead_label: Option<usize>) {
        self.us.push(d.understudent_us);
        if d.answered_by == "understudent" {
            self.local += 1;
            let readout = d.readout.clone().unwrap_or_default();
            let tier = if readout.starts_with("deep") { "deep".to_string() } else if readout == "mean" { "probe".to_string() } else { readout.clone() };
            *self.by_tier.entry(tier).or_default() += 1;
            *self.by_readout.entry(readout).or_default() += 1;
            self.first_local.get_or_insert(item);
            if lead_label == Some(d.label) {
                self.local_agree += 1;
            }
        }
        if d.mode.is_some() {
            self.graduated_at.get_or_insert(item);
        }
        self.audited += usize::from(d.audited);
        if d.demoted && !self.was_demoted {
            self.demotions += 1;
        }
        self.was_demoted = d.demoted;
    }

    pub fn finish(&self, secs: f64, status: Value) -> Value {
        let share = |n: usize| if self.items > 0 { n as f64 / self.items as f64 } else { 0.0 };
        let ladder = !status["tiers"].is_null();
        let mut us = self.us.clone();
        us.sort_unstable();
        let pct = |q: f64| if us.is_empty() { 0 } else { us[((us.len() - 1) as f64 * q).round() as usize] };
        let mean = if us.is_empty() { 0.0 } else { us.iter().sum::<u64>() as f64 / us.len() as f64 };
        let mut out = json!({
            "items": self.items, "answered_by_understudent": self.local,
            "coverage": if self.items > 0 { self.local as f64 / self.items as f64 } else { 0.0 },
            "local_agreement_with_lead": if self.local > 0 { json!(self.local_agree as f64 / self.local as f64) } else { Value::Null },
            "graduated_at_item": self.graduated_at, "first_local_item": self.first_local,
            "coverage_by_readout": self.by_readout.iter().map(|(k, &n)| (k.clone(), json!(if self.items > 0 { n as f64 / self.items as f64 } else { 0.0 }))).collect::<serde_json::Map<_, _>>(),
            "audited": self.audited, "demotions": self.demotions,
            "mode": status["mode"], "readout": status["readout"], "state": status["state"],
            "understudent_us": {"mean": mean, "p50": pct(0.5), "p95": pct(0.95)},
            "refits": self.refits,
            "refit_secs_total": self.refits.iter().filter_map(|r| r["secs"].as_f64()).fold(0.0, |a, b| a + b),
            "total_secs": secs,
            "final": status,
        });
        if ladder {
            let tiers: Vec<String> = status["tiers"].as_array().unwrap().iter().map(|t| t["tier"].as_str().unwrap_or_default().to_string()).collect();
            out["coverage_by_tier"] = json!(tiers.iter().map(|t| (t.clone(), json!(share(self.by_tier.get(t).copied().unwrap_or(0))))).collect::<serde_json::Map<_, _>>());
            out["graduated_at_item_by_tier"] = json!(tiers.iter().map(|t| (t.clone(), json!(self.tier_graduated.get(t)))).collect::<serde_json::Map<_, _>>());
            out["refit_secs_by_tier"] = json!(tiers
                .iter()
                .map(|t| {
                    let (total, n) = self.tier_refit.get(t).copied().unwrap_or((0.0, 0));
                    (t.clone(), json!({"total": total, "count": n, "mean": if n > 0 { total / n as f64 } else { 0.0 }}))
                })
                .collect::<serde_json::Map<_, _>>());
        }
        out
    }
}
