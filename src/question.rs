//! Questions and answers on the systemone wire, and the dataset's question.json.

use crate::{pyjson, stats::argmax};
use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha1::{Digest, Sha1};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Noul,
    Choice,
    Score,
}

/// A question as the lead reads it, plus its answer classes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QType,
    pub instructions: String,
    pub criteria: Value,
    /// Class names in index order: noul `["false", "true"]`, choice the criteria keys, score `"0".."k-1"`.
    pub options: Vec<String>,
}

impl Question {
    /// From a wire question `{"type", "instructions", "criteria"}`; `options` (question.json's explicit choice
    /// order) overrides the criteria key order.
    pub fn from_wire(q: &Value, options: Option<Vec<String>>) -> Result<Self> {
        let qtype = match q.get("type").and_then(Value::as_str) {
            Some("noul") => QType::Noul,
            Some("choice") => QType::Choice,
            Some("score") => QType::Score,
            other => bail!("question type must be one of noul, choice, score, got {other:?}"),
        };
        let criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
        let empty = match &criteria {
            Value::Null => true,
            Value::Object(m) => m.is_empty(),
            Value::Array(a) => a.is_empty(),
            Value::String(s) => s.is_empty(),
            _ => false,
        };
        ensure!(qtype == QType::Noul || !empty, "a {qtype:?} question needs criteria");
        let derived = match qtype {
            QType::Noul => vec!["false".into(), "true".into()],
            QType::Choice => criteria.as_object().ok_or_else(|| anyhow!("choice criteria must be an object"))?.keys().cloned().collect(),
            QType::Score => {
                let a = criteria.as_array().ok_or_else(|| anyhow!("score criteria must be an ordered array of levels"))?;
                (0..a.len()).map(|i| i.to_string()).collect()
            }
        };
        let options = match options {
            Some(o) => {
                let (mut a, mut b) = (o.clone(), derived.clone());
                a.sort();
                b.sort();
                ensure!(a == b, "options {o:?} don't match the question's classes {derived:?}");
                o
            }
            None => derived,
        };
        let instructions = q.get("instructions").and_then(Value::as_str).unwrap_or("").to_string();
        Ok(Question { qtype, instructions, criteria, options })
    }

    pub fn k(&self) -> usize {
        self.options.len()
    }

    /// The wire form, as sent to a lead.
    pub fn wire(&self) -> Value {
        json!({"type": self.qtype, "instructions": self.instructions, "criteria": self.criteria})
    }

    /// hunch's `question_key`: the first 16 hex digits of the SHA-1 of the spec's Python JSON.
    pub fn key(&self) -> String {
        let spec = json!({"type": self.qtype, "instructions": self.instructions, "criteria": self.criteria});
        let h = Sha1::digest(pyjson::dumps(&spec).as_bytes());
        hex(&h)[..16].to_string()
    }

    /// hunch's `parse_answer`: (class, probabilities if given) from an answer object, a probability list or a
    /// plain label.
    pub fn parse_answer(&self, a: &Value) -> Result<(usize, Option<Vec<f64>>)> {
        let opts = &self.options;
        match a {
            Value::Object(m) => {
                if let Some(probs) = m.get("probabilities") {
                    let pm = probs.as_object().ok_or_else(|| anyhow!("probabilities must be an object"))?;
                    ensure!(pm.len() == opts.len() && opts.iter().all(|o| pm.contains_key(o)),
                            "probabilities must name exactly the options {opts:?}, got {probs}");
                    let p = opts.iter().map(|o| probability(&pm[o])).collect::<Result<Vec<_>>>()?;
                    return Ok((argmax(&p), Some(p)));
                }
                if let (Some(v), QType::Noul) = (m.get("noul"), self.qtype) {
                    let p1 = probability(v)?;
                    return Ok((usize::from(p1 >= 0.5), Some(vec![1.0 - p1, p1])));
                }
                bail!("unrecognized answer object: {a}")
            }
            Value::Array(xs) => {
                let mut p = xs.iter().map(probability).collect::<Result<Vec<_>>>()?;
                if self.qtype == QType::Noul && p.len() == 1 {
                    p = vec![1.0 - p[0], p[0]];
                }
                ensure!(p.len() == opts.len(), "{} probabilities for {} options", p.len(), opts.len());
                Ok((argmax(&p), Some(p)))
            }
            Value::String(s) if self.qtype == QType::Noul => match s.to_lowercase().as_str() {
                "yes" | "true" | "1" => Ok((1, None)),
                "no" | "false" | "0" => Ok((0, None)),
                _ => bail!("noul label must be yes/no or true/false, got {s:?}"),
            },
            Value::Bool(b) if self.qtype == QType::Noul => Ok((usize::from(*b), None)),
            Value::Number(n) if self.qtype == QType::Noul => {
                if let Some(i) = n.as_i64().filter(|_| !n.is_f64()) {
                    ensure!(i == 0 || i == 1, "noul label must be 0 or 1, got {i}");
                    return Ok((i as usize, None));
                }
                let p1 = probability(a)?;
                Ok((usize::from(p1 >= 0.5), Some(vec![1.0 - p1, p1])))
            }
            Value::String(s) => {
                if let Some(i) = opts.iter().position(|o| o == s) {
                    return Ok((i, None));
                }
                if let (QType::Score, Some(levels)) = (self.qtype, self.criteria.as_array()) {
                    if let Some(i) = levels.iter().position(|l| l.as_str() == Some(s)) {
                        return Ok((i, None));
                    }
                }
                bail!("label {s:?} is not one of {opts:?}")
            }
            Value::Number(n) if n.as_u64().is_some_and(|i| (i as usize) < opts.len()) && !n.is_f64() => {
                Ok((n.as_u64().unwrap() as usize, None))
            }
            _ => bail!("unrecognized label {a} for a {:?} question", self.qtype),
        }
    }

    /// hunch's `answer_object`: an answer as the lead returns them, from class probabilities.
    pub fn answer_object(&self, p: &[f64]) -> Value {
        let r4 = |x: f64| (x * 1e4).round_ties_even() / 1e4;
        if self.qtype == QType::Noul {
            return json!({"type": "noul", "noul": r4(p[1])});
        }
        let best = argmax(p);
        let probs: Map<String, Value> = self.options.iter().zip(p).map(|(o, &x)| (o.clone(), json!(r4(x)))).collect();
        if self.qtype == QType::Choice {
            return json!({"type": "choice", "choice": self.options[best], "confidence": r4(p[best]), "probabilities": probs});
        }
        let score: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
        json!({"type": "score", "score": r4(score), "confidence": r4(p[best]), "probabilities": probs})
    }
}

impl Question {
    /// An answer for `serve`: like `answer_object`, but the four-decimal probabilities are apportioned (largest
    /// remainder) so they sum to exactly 1 at that precision, and the choice and its confidence come from the
    /// printed probabilities. Returns the answer and the chosen class.
    pub fn answer_object_normalized(&self, p: &[f64]) -> (Value, usize) {
        if self.qtype == QType::Noul {
            return (self.answer_object(p), argmax(p));
        }
        const UNITS: i64 = 10_000;
        let total: f64 = p.iter().map(|x| x.max(0.0)).sum();
        let q: Vec<f64> = if total > 0.0 { p.iter().map(|x| x.max(0.0) / total).collect() } else { vec![1.0 / p.len() as f64; p.len()] };
        let mut units: Vec<i64> = q.iter().map(|x| (x * UNITS as f64).floor() as i64).collect();
        let mut order: Vec<usize> = (0..q.len()).collect();
        let rem = |i: usize| q[i] * UNITS as f64 - units[i] as f64;
        order.sort_by(|&a, &b| rem(b).total_cmp(&rem(a)).then(q[b].total_cmp(&q[a])).then(a.cmp(&b)));
        let mut left = UNITS - units.iter().sum::<i64>();
        for &i in order.iter().cycle() {
            if left <= 0 {
                break;
            }
            units[i] += 1;
            left -= 1;
        }
        let fp: Vec<f64> = units.iter().map(|&u| u as f64 / UNITS as f64).collect();
        let best = argmax(&fp);
        let probs: Map<String, Value> = self.options.iter().zip(&fp).map(|(o, &x)| (o.clone(), json!(x))).collect();
        let answer = if self.qtype == QType::Choice {
            json!({"type": "choice", "choice": self.options[best], "confidence": fp[best], "probabilities": probs})
        } else {
            let score: f64 = q.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
            json!({"type": "score", "score": (score * 1e4).round_ties_even() / 1e4, "confidence": fp[best], "probabilities": probs})
        };
        (answer, best)
    }
}

/// A probability from an answer: a finite number in [0, 1].
fn probability(x: &Value) -> Result<f64> {
    let p = x.as_f64().filter(|_| !x.is_boolean()).ok_or_else(|| anyhow!("a probability must be a number, got {x}"))?;
    ensure!((0.0..=1.0).contains(&p), "a probability must be in [0, 1], got {x}");
    Ok(p)
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A dataset's question.json.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestionFile {
    pub name: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(rename = "type")]
    pub qtype: QType,
    pub options: Vec<String>,
    pub question: Value,
    #[serde(default)]
    pub pair: Option<(String, String)>,
    #[serde(default)]
    pub notes: Option<String>,
}

impl QuestionFile {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let qf: QuestionFile = serde_json::from_slice(&std::fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?)
            .map_err(|e| anyhow!("{}: {e}", path.display()))?;
        Ok(qf)
    }

    pub fn question(&self) -> Result<Question> {
        let q = Question::from_wire(&self.question, Some(self.options.clone()))?;
        ensure!(q.qtype == self.qtype, "question.json: type {:?} but question.type {:?}", self.qtype, q.qtype);
        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_and_parse() {
        let q = Question::from_wire(&json!({"type": "noul", "instructions": "Did the tool call fail?"}), None).unwrap();
        // hunch: question_key({"type": "noul", "instructions": "Did the tool call fail?"})
        //   = sha1('{"criteria": null, "instructions": "Did the tool call fail?", "type": "noul"}')[:16]
        let want = hex(&Sha1::digest(br#"{"criteria": null, "instructions": "Did the tool call fail?", "type": "noul"}"#))[..16].to_string();
        assert_eq!(q.key(), want);
        assert_eq!(q.parse_answer(&json!({"type": "noul", "noul": 0.8})).unwrap(), (1, Some(vec![0.19999999999999996, 0.8])));
        assert_eq!(q.parse_answer(&json!([0.3, 0.7])).unwrap().0, 1);
        assert_eq!(q.parse_answer(&json!(true)).unwrap(), (1, None));
        assert!(q.parse_answer(&json!(1.5)).is_err());
        let s = Question::from_wire(&json!({"type": "score", "criteria": ["bad", "ok", "good"]}), None).unwrap();
        let (c, p) = s.parse_answer(&json!({"type": "score", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}})).unwrap();
        assert_eq!((c, p.unwrap().len()), (2, 3));
        assert_eq!(s.parse_answer(&json!("ok")).unwrap(), (1, None));
        let c = Question::from_wire(&json!({"type": "choice", "criteria": {"b": "B", "a": "A"}}), None).unwrap();
        assert_eq!(c.options, vec!["b", "a"]);
    }
}
