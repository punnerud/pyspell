//! PySpell web add-on for the in-tunnel HTTP server.
//!
//! This is the ESP-specific glue that ties the two clean dependencies together:
//! `tailscale-core` (networking) calls [`route`] via a `fn` pointer, and `route`
//! parses + evaluates the submitted code with `pyspell-core`, supplying the ESP
//! wall clock so a request can set a real timeout (e.g. 10 s).
//!
//! Routes:
//! * `GET /`     → a tiny single-segment web page (text box + run button).
//! * `GET  /run?lang=py|rs&timeout=<s>&code=<urlencoded>` → eval, `text/plain` result.
//! * `POST /run?lang=py|rs&timeout=<s>` with the program as the raw request body
//!   → same result. POST avoids URL-encoding overhead and URL length limits, so
//!   it fits more code in the single request segment.
//! Any other path returns `None`, falling back to the built-in control panel.

use esp_idf_svc::sys::esp_timer_get_time;

use pyspell_core::{eval, parse, value::Value, Lang, Limits, VecEnv};
use tailscale_core::tcp::HttpReply;

/// Largest accepted program (URL-decoded). Keeps a single request segment bounded.
const MAX_CODE: usize = 1024;

pub fn route(method: &str, path: &str, query: &str, body: &[u8]) -> Option<HttpReply> {
    match path {
        "/" => Some(HttpReply {
            content_type: "text/html; charset=utf-8",
            body: PAGE.as_bytes().to_vec(),
        }),
        "/run" => Some(HttpReply {
            content_type: "text/plain; charset=utf-8",
            body: run(method, query, body).into_bytes(),
        }),
        _ => None,
    }
}

fn run(method: &str, query: &str, body: &[u8]) -> String {
    let lang = match query_get(query, "lang").as_str() {
        "rs" | "rust" => Lang::Rust,
        _ => Lang::Python,
    };
    let timeout_s = query_get(query, "timeout").parse::<i64>().unwrap_or(10).clamp(1, 60);
    // POST → the raw request body is the program; GET → the URL-encoded `code` param.
    let code = if method.eq_ignore_ascii_case("POST") {
        String::from_utf8_lossy(body).trim().to_string()
    } else {
        url_decode(&query_get(query, "code"))
    };
    if code.is_empty() {
        return "error: empty program".into();
    }
    if code.len() > MAX_CODE {
        return "error: program too long".into();
    }

    let program = match parse(&code, lang) {
        Ok(p) => p,
        Err(e) => return format!("error: {e}"),
    };

    // Wall-clock deadline using the ESP timer (microseconds since boot).
    let start = unsafe { esp_timer_get_time() };
    let budget_us = timeout_s * 1_000_000;
    let deadline = move || unsafe { esp_timer_get_time() } - start > budget_us;

    let env = device_env();
    let net = crate::net::DeviceNet;
    let limits =
        Limits { max_steps: 2_000_000, deadline: Some(&deadline), net: Some(&net) };
    match eval::run_with(&program, &env, limits) {
        Ok(v) => show(&v),
        Err(e) => format!("error: {e}"),
    }
}

/// Live device variables a program may read.
fn device_env() -> VecEnv {
    let (free, min_free, uptime_us) = unsafe {
        (
            esp_idf_svc::sys::esp_get_free_heap_size() as i64,
            esp_idf_svc::sys::esp_get_minimum_free_heap_size() as i64,
            esp_timer_get_time(),
        )
    };
    VecEnv::new()
        .set("free_heap", free)
        .set("min_free_heap", min_free)
        .set("uptime_ms", uptime_us / 1000)
        .set("uptime_s", uptime_us / 1_000_000)
}

fn show(v: &Value) -> String {
    match v {
        Value::Int(n) => format!("{n}"),
        Value::Float(x) => format!("{x}"),
        Value::Bool(b) => format!("{b}"),
        Value::Str(s) => s.to_string(),
        Value::List(l) => {
            let mut s = String::from("[");
            for (i, it) in l.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&show(it));
            }
            s.push(']');
            s
        }
    }
}

// ---- tiny query helpers (self-contained; tailscale-core's are private) ----

fn query_get(query: &str, key: &str) -> String {
    for kv in query.split('&') {
        if let Some((k, v)) = kv.split_once('=') {
            if k == key {
                return v.into();
            }
        }
    }
    String::new()
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => match (hexval(b[i + 1]), hexval(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                }
                _ => {
                    out.push(b'%');
                    i += 1;
                }
            },
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// The web page. Deliberately tiny so it fits one TCP segment (the in-tunnel
/// server sends a single data+FIN segment).
const PAGE: &str = "<!doctype html><meta charset=utf-8>\
<meta name=viewport content=\"width=device-width,initial-scale=1\">\
<title>PySpell</title>\
<body style=\"font-family:sans-serif;background:#111;color:#eee;margin:1em\">\
<h3>PySpell on ESP32</h3>\
<textarea id=c rows=4 style=width:100%>free_heap > 100000</textarea>\
<p><select id=l><option value=py>Python</option><option value=rs>Rust</option></select> \
timeout <input id=t value=10 size=2>s \
<button onclick=r()>Run</button></p><pre id=o></pre>\
<script>function r(){fetch('/run?lang='+l.value+'&timeout='+t.value,{method:'POST',body:c.value})\
.then(x=>x.text()).then(s=>o.textContent=s).catch(e=>o.textContent=e)}</script>";
