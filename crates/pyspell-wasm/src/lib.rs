//! Browser-WASM binding for `pyspell-core`.
//!
//! The exact `no_std + alloc` sandboxed evaluator from `pyspell-core`, compiled to wasm32
//! and exposed to JS, so the GitHub Pages live demo can **run** the Python the tiny model
//! generates — entirely in the browser, no device. `show(…)`/`led(…)`/`flash()` side
//! effects are captured and returned so the page can animate a stylized ESP32 (screen +
//! LED). `fetch_json` is intentionally absent (no network capability in the offline demo).

use std::cell::RefCell;

use pyspell_core::{eval, parse, value::Value, Actuator, Display, DslError, Lang, Limits};
use wasm_bindgen::prelude::*;

/// Captures the device-action side effects of one evaluation so JS can mirror them on the
/// stylized ESP32 illustration.
struct Caps {
    show: RefCell<Option<String>>,
    led: RefCell<Option<String>>,
    flash: RefCell<bool>,
}

impl Display for Caps {
    fn show(&self, text: &str) -> Result<(), DslError> {
        *self.show.borrow_mut() = Some(text.to_string());
        Ok(())
    }
}

impl Actuator for Caps {
    fn led(&self, on: bool, color: Option<(u8, u8, u8)>) -> Result<(), DslError> {
        *self.led.borrow_mut() = Some(if !on {
            "off".to_string()
        } else {
            match color {
                Some((r, g, b)) => format!("#{r:02x}{g:02x}{b:02x}"),
                None => "white".to_string(),
            }
        });
        Ok(())
    }
    fn flash(&self) -> Result<(), DslError> {
        *self.flash.borrow_mut() = true;
        Ok(())
    }
}

/// Render an evaluated [`Value`] to text (matches the device's `show`).
fn render(v: &Value) -> String {
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
                s.push_str(&render(it));
            }
            s.push(']');
            s
        }
    }
}

/// Minimal JSON string escaping (no serde dependency).
fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Evaluate `code` (Python or Rust subset) in the sandbox and return JSON:
/// `{"ok":true,"result":"8","show":<str|null>,"led":<str|null>,"flash":<bool>}` or
/// `{"ok":false,"error":"…"}`. The same evaluator as host + device; `fetch_json` is not
/// available (no network capability in the browser demo).
#[wasm_bindgen]
pub fn run_spell(code: &str, lang: &str) -> String {
    let lang = if lang.eq_ignore_ascii_case("rs") || lang.eq_ignore_ascii_case("rust") {
        Lang::Rust
    } else {
        Lang::Python
    };
    let program = match parse(code, lang) {
        Ok(p) => p,
        Err(e) => return format!("{{\"ok\":false,\"error\":{}}}", jstr(&format!("{e}"))),
    };
    let caps = Caps {
        show: RefCell::new(None),
        led: RefCell::new(None),
        flash: RefCell::new(false),
    };
    // Plausible free-variable readings for the offline demo (no real device).
    let env = |name: &str| -> Option<Value> {
        match name {
            "free_heap" => Some(Value::Int(200_000)),
            "min_free_heap" => Some(Value::Int(150_000)),
            "uptime_ms" => Some(Value::Int(42_000)),
            "uptime_s" => Some(Value::Int(42)),
            _ => None,
        }
    };
    let limits = Limits {
        max_steps: 2_000_000,
        max_bytes: 262_144,
        deadline: None,
        net: None, // no fetch_json in the browser demo
        display: Some(&caps),
        actuator: Some(&caps),
    };
    match eval::run_with(&program, &env, limits) {
        Ok(v) => {
            let showj = match caps.show.borrow().as_deref() {
                Some(s) => jstr(s),
                None => "null".to_string(),
            };
            let ledj = match caps.led.borrow().as_deref() {
                Some(s) => jstr(s),
                None => "null".to_string(),
            };
            format!(
                "{{\"ok\":true,\"result\":{},\"show\":{},\"led\":{},\"flash\":{}}}",
                jstr(&render(&v)),
                showj,
                ledj,
                *caps.flash.borrow()
            )
        }
        Err(e) => format!("{{\"ok\":false,\"error\":{}}}", jstr(&format!("{e}"))),
    }
}
