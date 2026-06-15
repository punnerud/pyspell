//! A tiny, path-directed JSON scalar extractor — `no_std`, no full DOM.
//!
//! `get(text, "a.b.0.c")` walks the JSON text following the path, skipping over
//! every non-matching branch without allocating it, and materializes only the
//! single scalar it lands on. That keeps RAM use to the size of the result, not
//! the document — important on a device with ~50 kB free heap parsing a ~50 kB
//! response. Path components are object keys or array indices (`0`, `1`, …).

use alloc::string::String;

use crate::error::DslError;
use crate::value::Value;

/// Extract the scalar at `path` (dot-separated) from a JSON `text`.
pub fn get(text: &str, path: &str) -> Result<Value, DslError> {
    let b = text.as_bytes();
    let mut p = skip_ws(b, 0);
    if !path.is_empty() {
        for comp in path.split('.') {
            p = enter(b, p, comp)?;
            p = skip_ws(b, p);
        }
    }
    let (v, _) = parse_scalar(b, p)?;
    Ok(v)
}

fn err(m: &str) -> DslError {
    DslError::Type(String::from(m))
}

/// Position `p` is at the start of a value; descend into member/element `comp`.
fn enter(b: &[u8], p: usize, comp: &str) -> Result<usize, DslError> {
    if p >= b.len() {
        return Err(err("json: unexpected end"));
    }
    match b[p] {
        b'{' => enter_object(b, p, comp),
        b'[' => enter_array(b, p, comp),
        _ => Err(err("json: cannot index a scalar with a path component")),
    }
}

fn enter_object(b: &[u8], mut p: usize, key: &str) -> Result<usize, DslError> {
    p += 1; // past '{'
    loop {
        p = skip_ws(b, p);
        if p >= b.len() {
            return Err(err("json: unterminated object"));
        }
        if b[p] == b'}' {
            return Err(err("json: key not found"));
        }
        if b[p] != b'"' {
            return Err(err("json: expected a key string"));
        }
        let (k, np) = parse_string(b, p)?;
        p = skip_ws(b, np);
        if p >= b.len() || b[p] != b':' {
            return Err(err("json: expected ':'"));
        }
        p = skip_ws(b, p + 1);
        if k == key {
            return Ok(p);
        }
        p = skip_value(b, p)?;
        p = skip_ws(b, p);
        match b.get(p) {
            Some(b',') => p += 1,
            Some(b'}') => return Err(err("json: key not found")),
            _ => return Err(err("json: malformed object")),
        }
    }
}

fn enter_array(b: &[u8], mut p: usize, idx_str: &str) -> Result<usize, DslError> {
    let target: usize = idx_str.parse().map_err(|_| err("json: array index not a number"))?;
    p += 1; // past '['
    let mut i = 0;
    loop {
        p = skip_ws(b, p);
        if p >= b.len() {
            return Err(err("json: unterminated array"));
        }
        if b[p] == b']' {
            return Err(err("json: array index out of range"));
        }
        if i == target {
            return Ok(p);
        }
        p = skip_value(b, p)?;
        p = skip_ws(b, p);
        match b.get(p) {
            Some(b',') => {
                p += 1;
                i += 1;
            }
            Some(b']') => return Err(err("json: array index out of range")),
            _ => return Err(err("json: malformed array")),
        }
    }
}

/// Skip exactly one JSON value, returning the position just past it.
fn skip_value(b: &[u8], p: usize) -> Result<usize, DslError> {
    let p = skip_ws(b, p);
    match b.get(p) {
        Some(b'{') | Some(b'[') => {
            let mut depth = 0i32;
            let mut q = p;
            while q < b.len() {
                match b[q] {
                    b'"' => q = skip_string_raw(b, q)?,
                    b'{' | b'[' => {
                        depth += 1;
                        q += 1;
                    }
                    b'}' | b']' => {
                        depth -= 1;
                        q += 1;
                        if depth == 0 {
                            return Ok(q);
                        }
                    }
                    _ => q += 1,
                }
            }
            Err(err("json: unterminated container"))
        }
        Some(b'"') => skip_string_raw(b, p),
        Some(_) => {
            // scalar token: advance to the next structural delimiter
            let mut q = p;
            while q < b.len() && !matches!(b[q], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n') {
                q += 1;
            }
            Ok(q)
        }
        None => Err(err("json: unexpected end")),
    }
}

/// Skip a quoted string (handling escapes), returning the position past the
/// closing quote. `b[p]` must be `"`.
fn skip_string_raw(b: &[u8], mut p: usize) -> Result<usize, DslError> {
    p += 1;
    while p < b.len() {
        match b[p] {
            b'\\' => p += 2,
            b'"' => return Ok(p + 1),
            _ => p += 1,
        }
    }
    Err(err("json: unterminated string"))
}

/// Parse a quoted string into a `String`, returning it + the position past the
/// closing quote. `b[p]` must be `"`.
fn parse_string(b: &[u8], mut p: usize) -> Result<(String, usize), DslError> {
    p += 1;
    let mut out = String::new();
    while p < b.len() {
        match b[p] {
            b'"' => return Ok((out, p + 1)),
            b'\\' => {
                p += 1;
                match b.get(p) {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'n') => out.push('\n'),
                    Some(b't') => out.push('\t'),
                    Some(b'r') => out.push('\r'),
                    Some(b'b') => out.push('\u{8}'),
                    Some(b'f') => out.push('\u{c}'),
                    Some(b'u') => {
                        let cp = hex4(b, p + 1)?;
                        p += 4;
                        if let Some(c) = char::from_u32(cp as u32) {
                            out.push(c);
                        } else {
                            out.push('\u{fffd}');
                        }
                    }
                    _ => return Err(err("json: bad escape")),
                }
                p += 1;
            }
            _ => {
                // copy the raw UTF-8 byte run up to the next quote/backslash
                let start = p;
                while p < b.len() && b[p] != b'"' && b[p] != b'\\' {
                    p += 1;
                }
                out.push_str(core::str::from_utf8(&b[start..p]).map_err(|_| err("json: bad utf-8"))?);
            }
        }
    }
    Err(err("json: unterminated string"))
}

fn hex4(b: &[u8], p: usize) -> Result<u16, DslError> {
    if p + 4 > b.len() {
        return Err(err("json: bad \\u escape"));
    }
    let mut v: u16 = 0;
    for &c in &b[p..p + 4] {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return Err(err("json: bad \\u digit")),
        };
        v = v * 16 + d as u16;
    }
    Ok(v)
}

/// Parse the scalar value at `p` (must already be past leading whitespace).
fn parse_scalar(b: &[u8], p: usize) -> Result<(Value, usize), DslError> {
    match b.get(p) {
        Some(b'"') => {
            let (s, np) = parse_string(b, p)?;
            Ok((Value::str(&s), np))
        }
        Some(b't') => Ok((Value::Bool(true), p + 4)),
        Some(b'f') => Ok((Value::Bool(false), p + 5)),
        Some(b'n') => Err(err("json: value is null")),
        Some(b'{') | Some(b'[') => Err(err("json: path leads to an object/array, not a scalar")),
        Some(_) => {
            let start = p;
            let mut q = p;
            let mut is_float = false;
            while q < b.len()
                && !matches!(b[q], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n')
            {
                if matches!(b[q], b'.' | b'e' | b'E') {
                    is_float = true;
                }
                q += 1;
            }
            let tok = core::str::from_utf8(&b[start..q]).map_err(|_| err("json: bad number"))?;
            if is_float {
                tok.parse::<f64>().map(|f| (Value::Float(f), q)).map_err(|_| err("json: bad float"))
            } else {
                tok.parse::<i64>().map(|n| (Value::Int(n), q)).map_err(|_| err("json: bad integer"))
            }
        }
        None => Err(err("json: unexpected end")),
    }
}

fn skip_ws(b: &[u8], mut p: usize) -> usize {
    while p < b.len() && matches!(b[p], b' ' | b'\t' | b'\r' | b'\n') {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"{"properties":{"timeseries":[{"data":{"instant":{"details":{"air_temperature":12.7,"humidity":80}}}},{"data":{"x":1}}]},"name":"Oslo","ok":true}"#;

    #[test]
    fn extracts_scalars_by_path() {
        assert_eq!(
            get(DOC, "properties.timeseries.0.data.instant.details.air_temperature").unwrap(),
            Value::Float(12.7)
        );
        assert_eq!(
            get(DOC, "properties.timeseries.0.data.instant.details.humidity").unwrap(),
            Value::Int(80)
        );
        assert_eq!(get(DOC, "name").unwrap(), Value::str("Oslo"));
        assert_eq!(get(DOC, "ok").unwrap(), Value::Bool(true));
    }

    #[test]
    fn errors_are_clean() {
        assert!(get(DOC, "properties.nope").is_err());
        assert!(get(DOC, "properties.timeseries.9").is_err()); // out of range
        assert!(get(DOC, "properties").is_err()); // non-scalar leaf
    }
}
