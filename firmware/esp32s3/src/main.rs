//! PySpell ESP32-S3 firmware — a live, sandboxed evaluator.
//!
//! The device never parses source. The host (`pyspell` CLI) compiles Rust/Python
//! to IR and streams it here as a single hex line over USB-Serial-JTAG; we decode
//! it to a [`pyspell_core::ir::Program`], evaluate it against a live device
//! environment (free heap, uptime, …), and reply with the result. This is the
//! MicroPython-like "push code live" loop, with the parser kept on the host.
//!
//! Wire protocol (line-based, robust over the esp-idf console):
//!   host → device:  `<hex of postcard Program>\n`
//!   device → host:  `OK <hex of postcard Value>\n`  or  `ERR <message>\n`
//! The host ignores any other (log) lines, so esp-idf boot logs don't interfere.

use core::ffi::c_void;

use esp_idf_svc::sys::{
    esp_get_free_heap_size, esp_get_minimum_free_heap_size, esp_timer_get_time,
    usb_serial_jtag_driver_config_t, usb_serial_jtag_driver_install, usb_serial_jtag_read_bytes,
    usb_serial_jtag_write_bytes,
};

use pyspell_core::{env::VecEnv, eval, value::Value, wire};

/// Largest accepted hex request line (covers `wire::MAX_FRAME` doubled for hex).
const MAX_LINE: usize = 8 * 1024;

fn main() {
    // Required once for any esp-idf app linked from Rust.
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    // Take over the USB-Serial-JTAG with a real driver so reads block reliably
    // instead of returning immediately the way the bare console path does.
    install_usb_serial_jtag();

    log::info!("PySpell ESP32-S3 evaluator ready — send a compiled program (hex line).");
    reply_line("READY pyspell-esp32");

    let mut line: Vec<u8> = Vec::with_capacity(256);
    let mut buf = [0u8; 256];
    loop {
        let n = unsafe {
            usb_serial_jtag_read_bytes(buf.as_mut_ptr() as *mut c_void, buf.len() as u32, u32::MAX)
        };
        if n <= 0 {
            continue;
        }
        for &byte in &buf[..n as usize] {
            match byte {
                b'\n' | b'\r' => {
                    if !line.is_empty() {
                        handle_line(&line);
                        line.clear();
                    }
                }
                _ => {
                    if line.len() < MAX_LINE {
                        line.push(byte);
                    } else {
                        // Overflow: drop the runaway line and report.
                        line.clear();
                        reply_err("request line too long");
                    }
                }
            }
        }
    }
}

fn handle_line(hex_line: &[u8]) {
    let bytes = match hex_decode(hex_line) {
        Some(b) => b,
        None => return reply_err("malformed hex"),
    };
    let program = match wire::from_bytes(&bytes) {
        Ok(p) => p,
        Err(e) => return reply_err(&format!("decode: {e}")),
    };
    let env = device_env();
    match eval::run(&program, &env) {
        Ok(value) => match wire::encode_value(&value) {
            Ok(payload) => reply_ok(&payload),
            Err(e) => reply_err(&format!("encode: {e}")),
        },
        Err(e) => reply_err(&format!("eval: {e}")),
    }
}

/// Live values a program may read. Refreshed every evaluation so a pushed
/// expression sees the current device state — the whole point of the live loop.
fn device_env() -> VecEnv {
    let (free, min_free, uptime_us) = unsafe {
        (
            esp_get_free_heap_size() as i64,
            esp_get_minimum_free_heap_size() as i64,
            esp_timer_get_time(),
        )
    };
    VecEnv::new()
        .set("free_heap", free)
        .set("min_free_heap", min_free)
        .set("uptime_ms", uptime_us / 1000)
        .set("uptime_s", uptime_us / 1_000_000)
}

// ---- protocol I/O --------------------------------------------------------

fn reply_ok(value_bytes: &[u8]) {
    let mut s = String::with_capacity(value_bytes.len() * 2 + 4);
    s.push_str("OK ");
    hex_encode_into(value_bytes, &mut s);
    reply_line(&s);
}

fn reply_err(msg: &str) {
    let mut s = String::with_capacity(msg.len() + 5);
    s.push_str("ERR ");
    s.push_str(msg);
    reply_line(&s);
}

fn reply_line(s: &str) {
    let mut out = Vec::with_capacity(s.len() + 1);
    out.extend_from_slice(s.as_bytes());
    out.push(b'\n');
    unsafe {
        usb_serial_jtag_write_bytes(out.as_ptr() as *const c_void, out.len(), u32::MAX);
    }
}

fn install_usb_serial_jtag() {
    let cfg = usb_serial_jtag_driver_config_t { tx_buffer_size: 1024, rx_buffer_size: 1024 };
    unsafe {
        // Safe to ignore the error: if the driver is already installed the
        // console path still works for our line protocol.
        let _ = usb_serial_jtag_driver_install(&cfg as *const _ as *mut _);
    }
}

// ---- hex helpers (no extra dependency) -----------------------------------

fn hex_encode_into(bytes: &[u8], out: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
}

fn hex_decode(s: &[u8]) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < s.len() {
        let hi = hex_val(s[i])?;
        let lo = hex_val(s[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Some(out)
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// Keep an unused import note: Value is referenced via wire/eval generic paths.
#[allow(dead_code)]
fn _value_type_anchor(_v: Value) {}
