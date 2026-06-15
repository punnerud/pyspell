//! `LeanNet` — the device `pyspell_core::Net` capability over the synchronous
//! lean stack (smoltcp + embedded-tls). This is what lets a PySpell program run
//! `fetch_json(url, path)` on-device: the evaluator calls `fetch_extract`, we do
//! the HTTPS GET, strip HTTP headers, and stream the body into the probe with
//! early abort (so we never buffer the whole forecast).

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use embedded_tls::blocking::TlsConnection;
use embedded_tls::{Aes128GcmSha256, TlsConfig, TlsContext, UnsecureProvider};
use esp_hal::rng::Rng;
use esp_println::println;

use pyspell_core::error::DslError;
use pyspell_core::eval::Net;
use pyspell_core::value::Value;

use crate::net::LeanStack;

/// TLS record read buffer. M3.1e finding: the met.no CDN sends application-data
/// records up to the TLS 1.3 max (16 KiB) and ignores the RFC 6066
/// max_fragment_length request, so this CANNOT be shrunk per-fetch — it must hold
/// a full record (16640). The route to ≥4 parallel small jobs is therefore a
/// *shared* pool of these read buffers + admission control (step 2), not a
/// smaller per-job buffer. The other buffers below are genuinely small.
const TLS_READ_BUF: usize = 16640;
const TLS_WRITE_BUF: usize = 2048;
/// TCP socket ring buffers — small: tiny request + early-abort body (proven OK).
const TCP_RX_BUF: usize = 4096;
const TCP_TX_BUF: usize = 2048;
/// Cap on buffered body bytes before giving up (early abort usually fires first).
const MAX_BODY: usize = 32 * 1024;

/// esp-hal hardware RNG as a `rand_core` 0.6 CSPRNG for embedded-tls.
struct TlsRng(Rng);

impl rand_core::RngCore for TlsRng {
    fn next_u32(&mut self) -> u32 {
        self.0.random()
    }
    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.0.read(&mut b);
        u64::from_le_bytes(b)
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.read(dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.0.read(dest);
        Ok(())
    }
}
impl rand_core::CryptoRng for TlsRng {}

/// The network capability the evaluator sees. Borrows the stack via `RefCell`
/// because `Net` is `&self` but the stack needs `&mut`.
pub struct LeanNet<'a, 'd> {
    stack: &'a RefCell<LeanStack<'d>>,
}

impl<'a, 'd> LeanNet<'a, 'd> {
    pub fn new(stack: &'a RefCell<LeanStack<'d>>) -> Self {
        Self { stack }
    }

    /// HTTPS GET `url`, then call `on_body(accumulated_body)` after each chunk of
    /// *body* (HTTP headers stripped). Stops early when `on_body` returns true.
    fn http_get(
        &self,
        url: &str,
        mut on_body: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), DslError> {
        let (host, path) = split_url(url)?;

        let mut stack = self.stack.borrow_mut();
        let ip = stack
            .resolve(host, 10_000)
            .ok_or_else(|| DslError::Net(format!("DNS failed for {host}")))?;
        let handle = stack
            .connect_tcp(ip, 443, 49600, TCP_RX_BUF, TCP_TX_BUF, 12_000)
            .ok_or_else(|| DslError::Net(String::from("TCP connect failed")))?;

        let mut read_buf = vec![0u8; TLS_READ_BUF];
        let mut write_buf = vec![0u8; TLS_WRITE_BUF];
        let conn = stack.tcp_conn(handle, 15_000);
        let mut tls: TlsConnection<_, Aes128GcmSha256> =
            TlsConnection::new(conn, &mut read_buf, &mut write_buf);

        // NOTE: RFC 6066 max_fragment_length was tested (M3.1e) — the met.no CDN
        // ignores it and still sends ~16 KiB records, so we don't request it and
        // size TLS_READ_BUF for a full record instead.
        let cfg = TlsConfig::new()
            .with_server_name(host)
            .enable_rsa_signatures();
        let rng = TlsRng(Rng::new());
        tls.open(TlsContext::new(
            &cfg,
            UnsecureProvider::new::<Aes128GcmSha256>(rng),
        ))
        .map_err(|e| DslError::Net(format!("TLS handshake: {e:?}")))?;
        println!(
            "[net] TLS up (read_buf={} tx_buf={}); heap {} B free (LIVE)",
            TLS_READ_BUF,
            TLS_WRITE_BUF,
            esp_alloc::HEAP.free()
        );

        // HTTP/1.0 → close-delimited body, no chunked transfer-encoding to undo.
        let req = format!(
            "GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: pyspell/0.1\r\n\r\n"
        );
        tls.write(req.as_bytes())
            .map_err(|e| DslError::Net(format!("TLS write: {e:?}")))?;
        tls.flush()
            .map_err(|e| DslError::Net(format!("TLS flush: {e:?}")))?;

        let mut raw: Vec<u8> = Vec::new();
        let mut body: Vec<u8> = Vec::new();
        let mut header_done = false;
        let mut chunk = [0u8; 512];
        loop {
            let n = tls
                .read(&mut chunk)
                .map_err(|e| DslError::Net(format!("TLS read: {e:?}")))?;
            if n == 0 {
                break; // EOF (server closed)
            }
            if header_done {
                body.extend_from_slice(&chunk[..n]);
                if on_body(&body) {
                    break;
                }
            } else {
                raw.extend_from_slice(&chunk[..n]);
                if let Some(pos) = find(&raw, b"\r\n\r\n") {
                    header_done = true;
                    body.extend_from_slice(&raw[pos + 4..]);
                    raw = Vec::new();
                    if on_body(&body) {
                        break;
                    }
                }
            }
            if body.len() > MAX_BODY {
                break;
            }
        }

        let _ = tls.close();
        Ok(())
    }
}

impl Net for LeanNet<'_, '_> {
    fn fetch(&self, url: &str) -> Result<String, DslError> {
        let out: RefCell<Vec<u8>> = RefCell::new(Vec::new());
        self.http_get(url, |b| {
            // Keep the latest full body; never early-stop.
            let mut o = out.borrow_mut();
            o.clear();
            o.extend_from_slice(b);
            false
        })?;
        Ok(String::from_utf8_lossy(&out.into_inner()).into_owned())
    }

    fn fetch_extract(
        &self,
        url: &str,
        probe: &dyn Fn(&[u8]) -> Option<Value>,
    ) -> Result<Value, DslError> {
        let found: RefCell<Option<Value>> = RefCell::new(None);
        self.http_get(url, |b| {
            if let Some(v) = probe(b) {
                *found.borrow_mut() = Some(v);
                true // early abort — field located
            } else {
                false
            }
        })?;
        found
            .into_inner()
            .ok_or_else(|| DslError::Net(String::from("field not found in response")))
    }
}

/// Split `https://host/path` → (`host`, `/path`). `http://` accepted too.
fn split_url(url: &str) -> Result<(&str, &str), DslError> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| DslError::Net(format!("unsupported URL scheme: {url}")))?;
    match rest.find('/') {
        Some(i) => Ok((&rest[..i], &rest[i..])),
        None => Ok((rest, "/")),
    }
}

/// First index of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
