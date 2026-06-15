//! Device-side network capability for PySpell `fetch` / `fetch_json`.
//!
//! Streams the HTTP(S) response through a probe and stops the moment the wanted
//! JSON field is found — so a ~50 kB yr.no response never has to fit in the
//! ~50 kB free heap. Uses esp-idf's `esp_http_client` directly (TLS via the
//! certificate bundle) so we can abort and free the TLS context early. The host
//! allowlist (`config::FETCH_ALLOW_HOSTS`) is enforced here — policy stays in
//! the ESP layer, not in the pure `pyspell-core` evaluator.

use core::ffi::c_char;
use std::ffi::CString;

use esp_idf_svc::sys::{
    esp_crt_bundle_attach, esp_http_client_cleanup, esp_http_client_close, esp_http_client_config_t,
    esp_http_client_fetch_headers, esp_http_client_init, esp_http_client_open,
    esp_http_client_read, esp_http_client_set_header,
};
use pyspell_core::{value::Value, DslError, Net};

/// Hard cap on bytes buffered while looking for the field. The streaming probe
/// normally finds the value well before this; the cap just bounds worst-case RAM.
const MAX_BUFFER: usize = 16 * 1024;
const CHUNK: usize = 512;
const USER_AGENT: &str = "pyspell-esp32/0.1 github.com/punnerud/pyspell";

/// Concurrent TLS fetches are bounded to ONE. Measured on hardware: esp-idf's
/// mbedTLS / `esp_crt_bundle` stack fails ("connect failed" / "http client init
/// failed") at even 2 concurrent TLS sessions — NOT a heap limit (~260 kB free),
/// but esp-idf TLS-stack resource/thread-safety limits (the cert bundle has global
/// state). So fetches serialize through this gate (each waiting job holds only its
/// URL); they all succeed, just one at a time. Compute-only PySpell jobs never enter
/// `stream()` → they run fully parallel across the worker pool. (The lean embedded-tls
/// build did 2+ concurrent verified fetches — it handles concurrency better here.)
const FETCH_MAX: usize = 1;
static FETCH_PERMITS: std::sync::Mutex<usize> = std::sync::Mutex::new(FETCH_MAX);
static FETCH_CV: std::sync::Condvar = std::sync::Condvar::new();

/// RAII fetch permit: releases (and wakes a waiter) on drop, even on early return.
struct FetchPermit;
impl Drop for FetchPermit {
    fn drop(&mut self) {
        if let Ok(mut n) = FETCH_PERMITS.lock() {
            *n += 1;
            FETCH_CV.notify_one();
        }
    }
}
fn acquire_fetch() -> FetchPermit {
    let mut n = FETCH_PERMITS.lock().unwrap();
    while *n == 0 {
        n = FETCH_CV.wait(n).unwrap();
    }
    *n -= 1;
    FetchPermit
}

pub struct DeviceNet;

impl Net for DeviceNet {
    fn fetch(&self, url: &str) -> Result<String, DslError> {
        // Whole body (capped). Prefer fetch_json on-device; this is for small
        // endpoints. Accumulate via the streaming helper without early-stop.
        let mut body: Vec<u8> = Vec::new();
        stream(url, |chunk| {
            body.extend_from_slice(chunk);
            body.len() > MAX_BUFFER // stop if too large
        })?;
        if body.len() > MAX_BUFFER {
            return Err(DslError::Net(String::from("response too large for fetch()")));
        }
        Ok(String::from_utf8_lossy(&body).into_owned())
    }

    fn fetch_extract(
        &self,
        url: &str,
        probe: &dyn Fn(&[u8]) -> Option<Value>,
    ) -> Result<Value, DslError> {
        let mut acc: Vec<u8> = Vec::new();
        let mut found: Option<Value> = None;
        stream(url, |chunk| {
            acc.extend_from_slice(chunk);
            if let Some(v) = probe(&acc) {
                found = Some(v);
                return true; // stop early → TLS context freed promptly
            }
            acc.len() > MAX_BUFFER // also stop if we've buffered too much
        })?;
        found.ok_or_else(|| {
            if acc.len() > MAX_BUFFER {
                DslError::Net(String::from("field not found within buffer cap"))
            } else {
                DslError::Net(String::from("field not found in response"))
            }
        })
    }
}

/// Enforce the config allowlist before any connection is made.
fn check_allowed(url: &str) -> Result<(), DslError> {
    let host = url.split("://").nth(1).unwrap_or(url).split(['/', ':']).next().unwrap_or("");
    let ok = crate::config::FETCH_ALLOW_HOSTS
        .iter()
        .any(|h| host == *h || host.ends_with(&format!(".{h}")));
    if ok {
        Ok(())
    } else {
        Err(DslError::Net(format!("host `{host}` not in allowlist")))
    }
}

/// GET `url`, calling `on_chunk` with each received block. `on_chunk` returns
/// `true` to stop early. TLS via the cert bundle. The client is always cleaned
/// up (freeing the TLS context) before returning.
fn stream(url: &str, mut on_chunk: impl FnMut(&[u8]) -> bool) -> Result<(), DslError> {
    check_allowed(url)?;
    // Bound concurrent TLS sessions (a waiting job blocks here holding only the URL).
    let _permit = acquire_fetch();
    let url_c = CString::new(url).map_err(|_| DslError::Net(String::from("bad url")))?;

    let mut cfg: esp_http_client_config_t = unsafe { core::mem::zeroed() };
    cfg.url = url_c.as_ptr();
    cfg.crt_bundle_attach = Some(esp_crt_bundle_attach);
    cfg.timeout_ms = 15_000;
    cfg.buffer_size = 1024;
    cfg.buffer_size_tx = 1024;

    let client = unsafe { esp_http_client_init(&cfg) };
    if client.is_null() {
        return Err(DslError::Net(String::from("http client init failed")));
    }

    // Run the request inside a closure so we can always clean up afterwards.
    let result = (|| -> Result<(), DslError> {
        let h = CString::new("User-Agent").unwrap();
        let v = CString::new(USER_AGENT).unwrap();
        unsafe { esp_http_client_set_header(client, h.as_ptr(), v.as_ptr()) };

        if unsafe { esp_http_client_open(client, 0) } != 0 {
            return Err(DslError::Net(String::from("connect failed")));
        }
        let _ = unsafe { esp_http_client_fetch_headers(client) };

        let mut buf = [0u8; CHUNK];
        loop {
            let n = unsafe {
                esp_http_client_read(client, buf.as_mut_ptr() as *mut c_char, buf.len() as i32)
            };
            if n <= 0 {
                break; // 0 = done, <0 = error/closed
            }
            if on_chunk(&buf[..n as usize]) {
                break;
            }
        }
        Ok(())
    })();

    unsafe {
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
    }
    result
}
