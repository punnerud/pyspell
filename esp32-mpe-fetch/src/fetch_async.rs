//! `AsyncLeanNet` — the device `pyspell_core::AsyncNet` over embassy-net + async
//! embedded-tls. Many of these can run concurrently on the embassy executor: each
//! `fetch_extract` awaits the network, yielding to other jobs while it waits.
//!
//! `Stack<'static>` is a cheap `Copy` handle, so one `AsyncLeanNet` is shared by
//! all jobs. TLS cert verification is still off (UnsecureProvider) — S2.5 adds
//! SPKI pinning.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use embassy_net::tcp::TcpSocket;
use embassy_net::{IpEndpoint, Stack};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::semaphore::{GreedySemaphore, Semaphore as _};
use embedded_tls::pki::CertVerifier;
use embedded_tls::{
    Aes128GcmSha256, Certificate, CryptoProvider, CryptoRngCore, TlsClock, TlsConfig,
    TlsConnection, TlsContext, TlsError, TlsVerifier,
};
use embedded_io_async::Write as _;
use esp_hal::rng::Rng;

use pyspell_core::error::DslError;
use pyspell_core::eval_async::AsyncNet;
use pyspell_core::value::Value;

use crate::pinning::{PinnedVerifier, SpkiPin};

use embassy_net::dns::DnsQueryType;

const TLS_READ_BUF: usize = 16640;
const TLS_WRITE_BUF: usize = 2048;
const TCP_RX_BUF: usize = 4096;
const TCP_TX_BUF: usize = 2048;
const MAX_BODY: usize = 32 * 1024;
/// Buffer the verifier copies the server cert chain into (~4.5 kB for met.no).
const CERT_SIZE: usize = 6144;

/// CENTRALIZED FETCH COORDINATOR — a shared gate capping how many TLS sessions run
/// at once. Any number of jobs may request a fetch; each acquires a permit first
/// (holding only its tiny URL/program while it waits), and only `TLS_MAX_CONCURRENT`
/// allocate the heavy buffers (~16.6 kB read + ~6 kB cert + RSA workspace) at a time.
/// Peak TLS memory is therefore bounded regardless of how many jobs arrive. Buffers
/// are allocated on demand inside the permit (freed on release) — the RSA verify
/// transient needs the headroom, so we don't also pin pre-allocated buffers. With
/// verification, 2 fits the 192 kB heap. Shared by PySpell fetches now and by
/// tailscale TLS (control plane + DERP) in M3.2 — one global TLS-memory budget.
pub const TLS_MAX_CONCURRENT: usize = 1;

/// Minimum free heap before a PySpell fetch allocates its ~45 kB of TLS buffers.
/// Below this, tailscale's netmap burst is likely in flight — wait it out.
const HEAP_ADMISSION_MIN: usize = 58 * 1024;

static TLS_GATE: GreedySemaphore<CriticalSectionRawMutex> =
    GreedySemaphore::new(TLS_MAX_CONCURRENT);

/// Pinned trust anchors (public CA certs — safe to commit). Each anchors a host
/// we talk to: HARICA RootCA 2015 → api.met.no; ISRG Root X1 → controlplane.tailscale.com
/// (Let's Encrypt). Both are the issuer of the topmost cert their server sends.
static HARICA_ROOT: &[u8] = include_bytes!("harica_rootca_2015.der");
static ISRG_X1: &[u8] = include_bytes!("isrg_root_x1.der");

/// The pinned root CA for a host, or None if we don't talk to it. This is the
/// shared TLS trust map for both PySpell fetches and tailscale's control plane.
fn ca_for_host(host: &str) -> Option<&'static [u8]> {
    if host == "api.met.no" || host.ends_with(".met.no") {
        Some(HARICA_ROOT)
    } else if host == "controlplane.tailscale.com" {
        Some(ISRG_X1)
    } else {
        None
    }
}

/// No RTC on the board, so cert validity is checked against a fixed "now". Within
/// the leaf/root validity windows; update if it drifts past the leaf's not-after.
struct FixedClock;
impl TlsClock for FixedClock {
    fn now() -> Option<u64> {
        Some(1_781_568_000) // ~2026-06-15 UTC
    }
}

/// met.no's pinned leaf SPKI SHA-256 (RSA-2048, CN=*.api.met.no, valid to 2026-11-07).
/// `openssl x509 -pubkey -noout | openssl pkey -pubin -outform der | openssl dgst -sha256`.
/// Update when met.no rotates its cert (a later step auto-refreshes via root-CA fallback).
const MET_NO_SPKI_PIN: SpkiPin = SpkiPin([
    0xdb, 0xb2, 0xfe, 0x1c, 0x72, 0xac, 0xbe, 0xa4, 0x0a, 0x7f, 0xa3, 0x88, 0x65, 0x84, 0x76, 0x0d,
    0xa4, 0x02, 0x37, 0x1d, 0x86, 0xb1, 0x30, 0xe3, 0x0e, 0x42, 0x56, 0x97, 0x99, 0x4c, 0x3c, 0xb7,
]);

/// Per-host TLS trust: low-memory SPKI pinning where we have a pin (met.no), full
/// CA-chain validation otherwise (e.g. the tailscale control plane's ECDSA chain,
/// fetched once at boot). One enum so a single `CryptoProvider` covers both.
enum HostVerifier<'a> {
    Pinned(PinnedVerifier),
    Chain(CertVerifier<'a, Aes128GcmSha256, FixedClock, CERT_SIZE>),
}

impl TlsVerifier<Aes128GcmSha256> for HostVerifier<'_> {
    fn set_hostname_verification(&mut self, hostname: &str) -> Result<(), TlsError> {
        match self {
            Self::Pinned(v) => v.set_hostname_verification(hostname),
            Self::Chain(v) => v.set_hostname_verification(hostname),
        }
    }
    fn verify_certificate(
        &mut self,
        transcript: &sha2::Sha256,
        cert: embedded_tls::CertificateRef,
    ) -> Result<(), TlsError> {
        match self {
            Self::Pinned(v) => v.verify_certificate(transcript, cert),
            Self::Chain(v) => v.verify_certificate(transcript, cert),
        }
    }
    fn verify_signature(
        &mut self,
        verify: embedded_tls::CertificateVerifyRef,
    ) -> Result<(), TlsError> {
        match self {
            Self::Pinned(v) => v.verify_signature(verify),
            Self::Chain(v) => v.verify_signature(verify),
        }
    }
}

/// CryptoProvider feeding embedded-tls our RNG and the per-host verifier.
struct PinnedProvider<'a> {
    rng: TlsRng,
    verifier: HostVerifier<'a>,
}

impl<'a> PinnedProvider<'a> {
    fn new(rng: TlsRng, host: &str, ca_der: &'a [u8]) -> Self {
        let verifier = if host == "api.met.no" || host.ends_with(".met.no") {
            HostVerifier::Pinned(PinnedVerifier::new(MET_NO_SPKI_PIN))
        } else {
            HostVerifier::Chain(CertVerifier::new(Certificate::X509(ca_der)))
        };
        Self { rng, verifier }
    }
}

impl CryptoProvider for PinnedProvider<'_> {
    type CipherSuite = Aes128GcmSha256;
    // Only used by client-cert auth (`signer`), which we don't implement.
    type Signature = p256::ecdsa::DerSignature;

    fn rng(&mut self) -> impl CryptoRngCore {
        &mut self.rng
    }

    fn verifier(&mut self) -> Result<&mut impl TlsVerifier<Self::CipherSuite>, TlsError> {
        Ok(&mut self.verifier)
    }
}

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

#[derive(Clone, Copy)]
pub struct AsyncLeanNet {
    stack: Stack<'static>,
}

impl AsyncLeanNet {
    pub fn new(stack: Stack<'static>) -> Self {
        Self { stack }
    }

    pub fn stack(&self) -> Stack<'static> {
        self.stack
    }

    /// HTTPS GET `url`, calling `on_body(accumulated_body)` after each body chunk
    /// (HTTP headers stripped). Stops early when `on_body` returns true.
    ///
    /// Acquires a TLS read buffer from the shared pool FIRST — so a job blocks here
    /// until a slot is free, and at most `TLS_POOL_SIZE` fetches ever hold memory at
    /// once, no matter how many jobs call in. The buffer is always returned.
    async fn http_get(
        &self,
        url: &str,
        mut on_body: impl FnMut(&[u8]) -> bool,
    ) -> Result<(), DslError> {
        let (host, path) = split_url(url)?;
        // The device is not an open proxy — only allowlisted hosts.
        let allowed = crate::config::FETCH_ALLOW_HOSTS
            .iter()
            .any(|h| host == *h || host.ends_with(&format!(".{h}")));
        if !allowed {
            return Err(DslError::Net(format!("host `{host}` not in allowlist")));
        }
        // Acquire a coordinator permit BEFORE allocating anything big. While waiting
        // the job holds only `url`/`host`/`path` — no buffers materialized in the queue.
        let _permit = TLS_GATE.acquire(1).await;
        // Heap admission: when tailscale's persistent map stream is bursting (it
        // decrypts the whole netmap as one ~60 kB Noise record), free heap dips. Rather
        // than allocate our ~45 kB TLS buffers into that dip and OOM the device, wait
        // for the burst to drain (the h2 buffer shrinks as it's consumed). Bounded so a
        // genuinely starved heap still proceeds and fails gracefully instead of hanging.
        let mut waited = 0u32;
        while esp_alloc::HEAP.free() < HEAP_ADMISSION_MIN && waited < 200 {
            embassy_time::Timer::after(embassy_time::Duration::from_millis(25)).await;
            waited += 1;
        }
        let mut read_buf = vec![0u8; TLS_READ_BUF]; // on demand; freed with the permit
        self.do_fetch(host, path, &mut read_buf, &mut on_body).await
    }

    /// The actual fetch, using a caller-provided (pooled) TLS read buffer.
    async fn do_fetch(
        &self,
        host: &str,
        path: &str,
        read_buf: &mut [u8],
        on_body: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<(), DslError> {
        let ip = {
            let addrs = self
                .stack
                .dns_query(host, DnsQueryType::A)
                .await
                .map_err(|e| DslError::Net(format!("DNS {host}: {e:?}")))?;
            *addrs
                .first()
                .ok_or_else(|| DslError::Net(format!("DNS {host}: no A record")))?
        };

        let mut rx = vec![0u8; TCP_RX_BUF];
        let mut tx = vec![0u8; TCP_TX_BUF];
        let mut socket = TcpSocket::new(self.stack, &mut rx, &mut tx);
        socket
            .connect(IpEndpoint::new(ip, 443))
            .await
            .map_err(|e| DslError::Net(format!("TCP connect: {e:?}")))?;

        let mut write_buf = vec![0u8; TLS_WRITE_BUF];
        let mut tls: TlsConnection<_, Aes128GcmSha256> =
            TlsConnection::new(socket, read_buf, &mut write_buf);

        let ca = ca_for_host(host)
            .ok_or_else(|| DslError::Net(format!("no pinned CA for `{host}`")))?;
        let cfg = TlsConfig::new()
            .with_server_name(host)
            .enable_rsa_signatures();
        tls.open(TlsContext::new(
            &cfg,
            PinnedProvider::new(TlsRng(Rng::new()), host, ca),
        ))
        .await
        .map_err(|e| DslError::Net(format!("TLS handshake: {e:?}")))?;

        let req = format!(
            "GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: pyspell/0.1\r\n\r\n"
        );
        tls.write_all(req.as_bytes())
            .await
            .map_err(|e| DslError::Net(format!("TLS write: {e:?}")))?;
        tls.flush()
            .await
            .map_err(|e| DslError::Net(format!("TLS flush: {e:?}")))?;

        let mut raw: Vec<u8> = Vec::new();
        let mut body: Vec<u8> = Vec::new();
        let mut header_done = false;
        let mut chunk = [0u8; 512];
        loop {
            let n = tls
                .read(&mut chunk)
                .await
                .map_err(|e| DslError::Net(format!("TLS read: {e:?}")))?;
            if n == 0 {
                break;
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

        let _ = tls.close().await;
        Ok(())
    }
}

impl AsyncNet for AsyncLeanNet {
    async fn fetch(&self, url: &str) -> Result<String, DslError> {
        let out: RefCell<Vec<u8>> = RefCell::new(Vec::new());
        self.http_get(url, |b| {
            let mut o = out.borrow_mut();
            o.clear();
            o.extend_from_slice(b);
            false
        })
        .await?;
        Ok(String::from_utf8_lossy(&out.into_inner()).into_owned())
    }

    async fn fetch_extract(
        &self,
        url: &str,
        probe: &dyn Fn(&[u8]) -> Option<Value>,
    ) -> Result<Value, DslError> {
        let found: RefCell<Option<Value>> = RefCell::new(None);
        self.http_get(url, |b| {
            if let Some(v) = probe(b) {
                *found.borrow_mut() = Some(v);
                true
            } else {
                false
            }
        })
        .await?;
        found
            .into_inner()
            .ok_or_else(|| DslError::Net(String::from("field not found in response")))
    }
}

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

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
