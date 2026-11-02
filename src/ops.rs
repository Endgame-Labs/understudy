//! What `train` and `judge` do per input, shared by the CLI and the tests.

use crate::proxy::{Kind, Outcome, Proxy};
use crate::question::Question;
use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value, json};

/// Seed understudent `si` with example lines: `{"state","lead_probs"}` (sampled), `{"state","truth"}` (labelled),
/// both, or neither (the lead is asked). Encodes in batches of 32. Doesn't refit. Returns the counts.
pub fn seed(p: &mut Proxy, si: usize, rows: &[Value]) -> Result<Value> {
    let q = p.understudents[si].question.clone();
    let name = p.understudents[si].name.clone();
    let (mut sampled, mut labelled, mut asked) = (0, 0, 0);
    for chunk in rows.chunks(32) {
        let states: Vec<&Value> = chunk.iter().map(|r| r.get("state").ok_or_else(|| anyhow!("an example without a state"))).collect::<Result<_>>()?;
        let encs = p.encode_many(&states)?;
        for ((r, state), (key, enc)) in chunk.iter().zip(&states).zip(encs) {
            let id = r.get("id").and_then(Value::as_str);
            let truth = r.get("truth");
            if let Some(t) = truth {
                let (lab, _) = q.parse_answer(t).with_context(|| format!("truth of {id:?}"))?;
                p.add_example_with_id(si, Kind::Labelled, state, lab, None, key.clone(), enc.clone(), id);
                labelled += 1;
            }
            let answer = match r.get("lead_probs") {
                Some(tp) => Some(q.parse_answer(tp).with_context(|| format!("lead_probs of {id:?}"))?),
                None if truth.is_none() => {
                    let t = p.lead.as_ref().ok_or_else(|| anyhow!("{id:?} has no lead_probs or truth, and there is no lead"))?;
                    let qs: Map<String, Value> = [(name.clone(), q.wire())].into_iter().collect();
                    let a = t.ask(id, state, &qs)?;
                    asked += 1;
                    Some(q.parse_answer(a.get(&name).ok_or_else(|| anyhow!("the lead didn't answer {name}"))?)?)
                }
                None => None,
            };
            if let Some((lab, pr)) = answer {
                let pr = pr.unwrap_or_else(|| (0..q.k()).map(|c| if c == lab { 1.0 } else { 0.0 }).collect());
                p.add_example_with_id(si, Kind::Sampled, state, lab, Some(pr), key, enc, id);
                sampled += 1;
            }
        }
    }
    Ok(json!({"sampled": sampled, "labelled": labelled, "asked_lead": asked}))
}

/// Judge one input line: `{"id","state"}` (every question of the snapshot) or a systemone request
/// `{"id"?, "questions": {name: question}, "state"}` (questions not in the snapshot start learning). Returns the output
/// line (a decision with `id`; several under "answers") and the outcome.
pub fn judge_line(p: &mut Proxy, v: &Value) -> Result<(Value, Outcome)> {
    let id = v.get("id").and_then(Value::as_str).map(String::from);
    let state = v.get("state").ok_or_else(|| anyhow!("no state"))?;
    let which = match v.get("questions").and_then(Value::as_object) {
        Some(qs) => Some(qs.iter().map(|(name, q)| Ok(p.understudent(name, &Question::from_wire(q, None)?))).collect::<Result<Vec<_>>>()?),
        None => None,
    };
    let o = p.answer(id.as_deref(), state, which.as_deref())?;
    let line = if o.decisions.len() == 1 {
        serde_json::to_value(&o.decisions[0].1)?
    } else {
        let answers: Map<String, Value> = o.decisions.iter().map(|(k, d)| Ok((k.clone(), serde_json::to_value(d)?))).collect::<Result<_>>()?;
        json!({"id": id, "answered_by": o.answered_by, "answers": answers})
    };
    Ok((line, o))
}
