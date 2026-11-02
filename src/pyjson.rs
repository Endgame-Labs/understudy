//! Python's renderings of JSON values, byte for byte where hunch hashes or reads them:
//! `json.dumps(v, sort_keys=True, ensure_ascii=False)` and `str(v)` of the value `json.loads` gives.

use serde_json::Value;
use std::fmt::Write;

/// `json.dumps(v, sort_keys=True, ensure_ascii=False)`: `", "` and `": "` separators, keys sorted by code point.
pub fn dumps(v: &Value) -> String {
    let mut s = String::new();
    dump(v, &mut s);
    s
}

fn dump(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => number(n, out),
        Value::String(s) => quote(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort_by(|a, b| a.chars().cmp(b.chars())); // Python compares code points, not UTF-16 units
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                quote(k, out);
                out.push_str(": ");
                dump(&m[k], out);
            }
            out.push('}');
        }
    }
}

/// A JSON string as `json.dumps(..., ensure_ascii=False)` writes it.
fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn number(n: &serde_json::Number, out: &mut String) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else {
        out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
    }
}

/// Python's `repr(float)`: the shortest round-tripping digits, scientific below 1e-4 and from 1e16.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` gives the shortest round-trip digits too: "d.ddde±x".
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let s = if exp >= n - 1 {
            format!("{digits}{}.0", "0".repeat((exp - (n - 1)) as usize))
        } else if exp >= 0 {
            format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// Python's `str(v)` for a value from `json.loads`: a string as is, else its repr (`True`, `None`, `{'a': 1}`).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => {
            let mut s = String::new();
            repr(v, &mut s);
            s
        }
    }
}

fn repr(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Value::Number(n) => {
            if n.is_f64() {
                let f = n.as_f64().unwrap_or(f64::NAN);
                out.push_str(&match float_repr(f).as_str() {
                    "NaN" => "nan".to_string(),
                    "Infinity" => "inf".to_string(),
                    "-Infinity" => "-inf".to_string(),
                    s => s.to_string(),
                });
            } else {
                number(n, out)
            }
        }
        Value::String(s) => str_repr(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                repr(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                str_repr(k, out);
                out.push_str(": ");
                repr(x, out);
            }
            out.push('}');
        }
    }
}

/// `repr(str)`: single quotes unless the string has one and no double quote. Non-printable characters are
/// approximated by their escapes (Python's `isprintable` table is not reproduced beyond ASCII and C1 controls).
fn str_repr(s: &str, out: &mut String) {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..0xa1).contains(&(c as u32)) => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(q);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_like_python() {
        let v = json!({"type": "noul", "instructions": "Did it fail?\n\"x\"", "criteria": {"true": "é", "false": 1.5}});
        assert_eq!(dumps(&v), r#"{"criteria": {"false": 1.5, "true": "é"}, "instructions": "Did it fail?\n\"x\"", "type": "noul"}"#);
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(float_repr(1.0), "1.0");
        assert_eq!(float_repr(123.25), "123.25");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(py_str(&json!({"a": [1, true, null], "b": "it's"})), r#"{'a': [1, True, None], 'b': "it's"}"#);
    }
}
