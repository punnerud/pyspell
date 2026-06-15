//! M3.2: tailscale control-plane registration on the lean async stack.
//!
//! Reuses tailscale-core's async control path (`AsyncConn`/`AsyncH2`) over an
//! embassy-net TCP socket, plus the NVS-extracted authorized node identity
//! (`config::TS_*_KEY`) — so the node registers WITHOUT an auth key / browser
//! login (the tailnet already knows this node key) and we get its tailscale IP.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use pyspell_core::AsyncNet;

use embassy_net::dns::DnsQueryType;
use embassy_net::tcp::TcpSocket;
use embassy_net::{IpEndpoint, Stack};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;
use embassy_time::Duration;
use esp_println::println;

use tailscale_core::platform::AsyncByteStream;
use tailscale_core::{h2, noise, transport};

use crate::config;
use crate::fetch_async::AsyncLeanNet;

const CONTROL_HOST: &str = "controlplane.tailscale.com";
const KEY_VER: u32 = 130;
const CAP_VER: u32 = 90;
const TS2021_PORT: u16 = 80;

/// Our tailscale IPv4, published once the control session has it. `main` waits on
/// this to know registration succeeded; the session task then keeps running (holding
/// the `/machine/map` long-poll open) so the node stays **online**.
pub static TS_IP: Signal<CriticalSectionRawMutex, String> = Signal::new();

// --- key derivation + hex ---------------------------------------------------

struct Keypair {
    public: [u8; 32],
}
impl Keypair {
    fn from_private(private: [u8; 32]) -> Self {
        let secret = x25519_dalek::StaticSecret::from(private);
        Self { public: x25519_dalek::PublicKey::from(&secret).to_bytes() }
    }
    fn public_hex(&self) -> String {
        hex_lower(&self.public)
    }
}

fn hex_lower(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn hex_decode_32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let b = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = (hexval(b[i * 2])? << 4) | hexval(b[i * 2 + 1])?;
    }
    Some(out)
}
fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Find our `100.x.y.z` tailscale IPv4 in the raw map response.
fn scan_tailscale_ip(raw: &[u8]) -> Option<String> {
    let pat = b"\"100.";
    let pos = raw.windows(pat.len()).position(|w| w == pat)?;
    let start = pos + 1;
    let mut end = start;
    while end < raw.len() && raw[end] != b'/' && raw[end] != b'"' {
        end += 1;
    }
    let s = core::str::from_utf8(&raw[start..end]).ok()?;
    if s.len() >= 7 && s.chars().all(|c| c.is_ascii_digit() || c == '.') {
        Some(String::from(s))
    } else {
        None
    }
}

// --- control JSON (minimal; the node key authorizes, hostinfo is metadata) --

fn hostinfo() -> serde_json::Value {
    // Match the identity the node was originally registered with (the esp-idf demo),
    // so tailscale doesn't flag "OS changed / state copied between devices" and
    // withhold the address — we are deliberately reusing that node key.
    serde_json::json!({
        "Hostname": "tdongle-s3",
        "OS": "espidf",
        "OSVersion": "5.2.2",
        "GoArch": "xtensa",
        "NetInfo": {
            "WorkingIPv4": true,
            "WorkingIPv6": false,
            "PreferredDERP": 1,
            "LinkType": "wired"
        }
    })
}

fn register_json(machine_pub: &str, node_pub: &str, disco_pub: &str) -> String {
    serde_json::json!({
        "Version": CAP_VER,
        "NodeKey": format!("nodekey:{node_pub}"),
        "MachineKey": format!("mkey:{machine_pub}"),
        "DiscoKey": format!("discokey:{disco_pub}"),
        "Hostinfo": hostinfo(),
        "Endpoints": [],
        "Capabilities": [],
        "DeviceName": "tdongle-s3",
        "Ephemeral": false
    })
    .to_string()
}

fn map_json(node_pub: &str, disco_pub: &str) -> String {
    serde_json::json!({
        "Version": CAP_VER,
        "NodeKey": format!("nodekey:{node_pub}"),
        "DiscoKey": format!("discokey:{disco_pub}"),
        "Endpoints": [],
        "Hostinfo": hostinfo(),
        "Stream": true,
        "KeepAlive": true,
        "ReadOnly": false,
        "OmitPeers": true,
        "Compress": ""
    })
    .to_string()
}

// --- AsyncByteStream over an embassy-net TcpSocket --------------------------

struct TcpAsyncStream<'a> {
    sock: TcpSocket<'a>,
}
impl AsyncByteStream for TcpAsyncStream<'_> {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        embedded_io_async::Read::read(&mut self.sock, buf)
            .await
            .map_err(|_| ())
    }
    async fn write_all(&mut self, buf: &[u8]) -> Result<(), ()> {
        embedded_io_async::Write::write_all(&mut self.sock, buf)
            .await
            .map_err(|_| ())
    }
}

// --- orchestration ----------------------------------------------------------

/// Fetch the control plane's Noise static public key via verified TLS GET /key.
async fn fetch_control_key(net: &AsyncLeanNet) -> Result<[u8; 32], String> {
    let url = format!("https://{CONTROL_HOST}/key?v={KEY_VER}");
    // Use the streaming extract (early-abort) like fetch_json: it stops as soon as
    // `publicKey` is found, BEFORE the server closes the HTTP/1.0 connection — so we
    // never hit embedded-tls's ConnectionClosed that a read-to-EOF fetch would.
    let probe = |buf: &[u8]| -> Option<pyspell_core::Value> {
        let s = match core::str::from_utf8(buf) {
            Ok(s) => s,
            Err(e) => core::str::from_utf8(&buf[..e.valid_up_to()]).ok()?,
        };
        pyspell_core::json::get(s, "publicKey").ok()
    };
    let v = net
        .fetch_extract(&url, &probe)
        .await
        .map_err(|e| format!("/key fetch: {e:?}"))?;
    let mkey = match v {
        pyspell_core::Value::Str(s) => s,
        _ => return Err(String::from("/key publicKey not a string")),
    };
    let hex = mkey.strip_prefix("mkey:").unwrap_or(&mkey);
    hex_decode_32(hex).ok_or_else(|| String::from("/key bad mkey hex"))
}

/// Run the full control-plane bring-up, publish our tailscale IPv4 via [`TS_IP`],
/// then **stay in the `/machine/map` long-poll loop forever** so the control plane
/// keeps the node marked **online** (it considers a node online only while its map
/// stream is held open). Reuses the already-authorized node identity from
/// `config::TS_*_KEY` (no auth key needed). Returns `Err` only if the connection
/// drops or a step fails — the caller retries.
pub async fn run_control_session(net: &AsyncLeanNet, stack: Stack<'static>) -> Result<(), String> {
    println!("[ts] heap at start: {} B", esp_alloc::HEAP.free());
    let control_pub = fetch_control_key(net).await?;
    println!("[ts] control key fetched; heap {} B", esp_alloc::HEAP.free());

    let machine = Keypair::from_private(config::TS_MACHINE_KEY);
    let node = Keypair::from_private(config::TS_NODE_KEY);
    let disco = Keypair::from_private(config::TS_DISCO_KEY);

    let (hs, framed_init) =
        noise::start(&config::TS_MACHINE_KEY, &control_pub).map_err(|e| format!("noise start: {e}"))?;
    let header = transport::base64_std(&framed_init);

    let ip = {
        let a = stack
            .dns_query(CONTROL_HOST, DnsQueryType::A)
            .await
            .map_err(|e| format!("dns: {e:?}"))?;
        *a.first().ok_or_else(|| String::from("control DNS: no A"))?
    };

    let mut rx = vec![0u8; 4096];
    let mut tx = vec![0u8; 4096];
    let mut sock = TcpSocket::new(stack, &mut rx, &mut tx);
    // The map long-poll is held open indefinitely; the control plane sends keepalive
    // frames (~every minute, KeepAlive:true), so a generous read timeout detects a
    // truly dead link without tripping on the quiet gaps between keepalives.
    sock.set_timeout(Some(Duration::from_secs(120)));
    sock.connect(IpEndpoint::new(ip, TS2021_PORT))
        .await
        .map_err(|e| format!("ts2021 tcp: {e:?}"))?;
    println!("[ts] tcp :80 connected, upgrading ...");

    let mut conn = transport::connect_and_upgrade_async(TcpAsyncStream { sock }, CONTROL_HOST, &header)
        .await
        .map_err(|e| format!("upgrade: {e}"))?;

    let (typ, payload) = conn.read_frame().await.map_err(|e| format!("read resp: {e}"))?;
    if typ != noise::MSG_RESPONSE {
        return Err(format!("unexpected handshake frame {typ}"));
    }
    let tr = hs.complete(&payload).map_err(|e| format!("noise complete: {e}"))?;
    println!(
        "[ts] noise handshake complete; stack free here {} B",
        crate::stack_free_now()
    );

    let (mut sess, early) = h2::AsyncH2::start(conn, tr, String::from(CONTROL_HOST))
        .await
        .map_err(|e| format!("h2 start: {e}"))?;
    println!(
        "[ts] http2 up; early {} B; heap {} B",
        early.as_ref().map(|e| e.len()).unwrap_or(0),
        esp_alloc::HEAP.free()
    );

    // A cached map may arrive as the HTTP/2 early payload (this node was up before).
    // Grab the IP from it if present, but DON'T short-circuit: we still open the map
    // long-poll below and hold it open, which is what marks the node online.
    let mut have_ip = false;
    if let Some(ep) = &early {
        if let Some(ip) = scan_tailscale_ip(ep) {
            println!("[ts] >>> tailscale IP = {} (from early payload)", ip);
            TS_IP.signal(ip);
            have_ip = true;
        }
    }
    drop(early);

    let reg = register_json(&machine.public_hex(), &node.public_hex(), &disco.public_hex());
    let (rstatus, rresp) = sess
        .post_json("/machine/register", reg.as_bytes())
        .await
        .map_err(|e| format!("register: {e}"))?;
    // Scan the bytes (no serde Value tree — heap is tight). tailscale JSON is compact.
    let authorized = contains(&rresp, b"\"MachineAuthorized\":true");
    drop(rresp);
    println!("[ts] register status={rstatus} authorized={authorized}");

    // Streaming map, held open forever. Scan early frames for our 100.x address (it's
    // in the Node block near the start) to publish the IP; once we have it, stop
    // accumulating and just consume keepalive frames — staying connected is what keeps
    // the node online. We never buffer the whole netmap (the 128 kB heap can't hold it).
    println!("[ts] heap before map: {} B free", esp_alloc::HEAP.free());
    let mapj = map_json(&node.public_hex(), &disco.public_hex());
    let sid = sess
        .post_stream("/machine/map", mapj.as_bytes())
        .await
        .map_err(|e| format!("map stream: {e}"))?;
    let mut acc: Vec<u8> = Vec::new();
    let mut keepalives: u32 = 0;
    let mut frame_no: u32 = 0;
    loop {
        let chunk = match sess.read_data(sid).await {
            Ok(c) => c,
            Err(e) => return Err(format!("map read: {e}")),
        };
        frame_no += 1;
        if frame_no <= 12 {
            println!(
                "[ts] DIAG frame#{frame_no} len={} heap={} B",
                chunk.len(),
                esp_alloc::HEAP.free()
            );
        }
        if !have_ip {
            acc.extend_from_slice(&chunk);
            if let Some(ip) = scan_tailscale_ip(&acc) {
                println!(
                    "[ts] >>> tailscale IP = {} (holding map stream -> ONLINE); heap {} B",
                    ip,
                    esp_alloc::HEAP.free()
                );
                TS_IP.signal(ip);
                have_ip = true;
                acc = Vec::new(); // free the scan window; only keepalives from here
            } else if acc.len() > 12288 {
                let drop = acc.len() - 4096;
                acc.drain(..drop);
            }
        } else {
            // Keepalive / map-delta frame; we don't need its contents, just the link.
            keepalives = keepalives.wrapping_add(1);
            if keepalives <= 5 || keepalives % 10 == 1 {
                println!("[ts] online; map keepalive #{keepalives}; heap {} B", esp_alloc::HEAP.free());
            }
        }
    }
}
