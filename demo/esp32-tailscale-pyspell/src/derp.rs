//! Minimal DERP client.
//!
//! DERP is Tailscale's relay: clients connect out to a DERP server over TLS and
//! exchange opaque packets addressed by node public key. We use it as a fallback
//! transport for our WireGuard + disco packets when a direct path can't be
//! punched (the common remote/NAT case). The dongle connects to its home DERP
//! region (nyc) so packets peers send to us via DERP actually arrive.
//!
//! Wire: after an HTTP `Upgrade: DERP`, the stream carries frames
//! `[type u8][len u32 BE][payload]`. The server opens with frameServerKey
//! (magic + its NaCl public key); we reply with an encrypted frameClientInfo;
//! thereafter frameSendPacket/frameRecvPacket carry our tunnel traffic.

use anyhow::{bail, Context, Result};
use core::ffi::c_void;

use crypto_box::aead::{Aead, Nonce};
use crypto_box::{PublicKey, SalsaBox, SecretKey};

use esp_idf_svc::sys;

pub const FRAME_SERVER_KEY: u8 = 0x01;
pub const FRAME_CLIENT_INFO: u8 = 0x02;
pub const FRAME_SERVER_INFO: u8 = 0x03;
pub const FRAME_SEND_PACKET: u8 = 0x04;
pub const FRAME_RECV_PACKET: u8 = 0x05;
pub const FRAME_KEEPALIVE: u8 = 0x06;
pub const FRAME_NOTE_PREFERRED: u8 = 0x07;
pub const FRAME_PING: u8 = 0x0c;
pub const FRAME_PONG: u8 = 0x0d;

const MAGIC: &[u8] = b"DERP\xf0\x9f\x94\x91"; // "DERP🔑"
const MAX_FRAME: usize = 64 * 1024;

/// Pinned trust anchors for the DERP relay's TLS chain. The relay
/// (`derp1f.tailscale.com`) presents a Let's Encrypt leaf chaining up to ISRG:
/// `leaf ← YE2 ← Root YE ← ISRG Root X2 ← ISRG Root X1`. Pinning just these two
/// ISRG roots — instead of attaching the full Mozilla CA bundle — cuts handshake
/// RAM + CPU on every (re)connect, which is what lets the DERP responder coexist
/// with the lwIP bridge on ~64 kB of free heap. Both roots are embedded so a chain
/// that anchors at X1 (current) or shortens to X2 still validates. These are public
/// root CAs (safe to commit); refresh from https://letsencrypt.org/certs/ if the
/// DERP CA ever changes (TLS will simply fail to connect if the pin goes stale).
static DERP_CA_PEM: &[u8] = include_bytes!("../certs/derp-roots.pem");

/// Monotonic milliseconds since boot (for the in-tunnel TCP server's retransmit
/// timers and the peer-LRU table).
fn now_ms() -> u64 {
    (unsafe { sys::esp_timer_get_time() } / 1000) as u64
}

/// An established DERP connection (TLS socket after the protocol handshake).
pub struct Derp {
    tls: *mut sys::esp_tls,
}

// The esp_tls handle is owned exclusively by whoever holds Derp; we only use it
// from the single DERP thread.
unsafe impl Send for Derp {}

impl Derp {
    /// Connect to `host:443`, perform the HTTP upgrade + DERP handshake using our
    /// node key as the DERP identity.
    pub fn connect(host: &str, node_priv: &[u8; 32], node_pub: &[u8; 32]) -> Result<Self> {
        let tls = tls_connect(host, 443)?;
        let mut d = Derp { tls };
        d.http_upgrade(host).context("DERP http upgrade")?;
        let server_pub = d.read_server_key().context("DERP server key")?;
        d.send_client_info(node_priv, node_pub, &server_pub)
            .context("DERP client info")?;
        Ok(d)
    }

    fn http_upgrade(&mut self, host: &str) -> Result<()> {
        let req = format!(
            "GET /derp HTTP/1.1\r\n\
             Host: {host}\r\n\
             Connection: Upgrade\r\n\
             Upgrade: DERP\r\n\r\n"
        );
        self.write_all(req.as_bytes())?;

        // Read response headers up to \r\n\r\n.
        let mut head = Vec::new();
        let mut b = [0u8; 1];
        loop {
            let n = self.read_some(&mut b)?;
            if n == 0 {
                bail!("connection closed during upgrade");
            }
            head.push(b[0]);
            if head.ends_with(b"\r\n\r\n") {
                break;
            }
            if head.len() > 4096 {
                bail!("upgrade response too large");
            }
        }
        let status = String::from_utf8_lossy(&head);
        let line = status.lines().next().unwrap_or("");
        if !line.contains(" 101 ") {
            bail!("DERP upgrade failed: '{}'", line.trim());
        }
        Ok(())
    }

    fn read_server_key(&mut self) -> Result<[u8; 32]> {
        let (typ, payload) = self.read_frame()?;
        if typ != FRAME_SERVER_KEY {
            bail!("expected server key frame, got type {typ}");
        }
        if payload.len() < MAGIC.len() + 32 || &payload[..MAGIC.len()] != MAGIC {
            bail!("bad server key frame ({} bytes)", payload.len());
        }
        let mut k = [0u8; 32];
        k.copy_from_slice(&payload[MAGIC.len()..MAGIC.len() + 32]);
        Ok(k)
    }

    fn send_client_info(
        &mut self,
        node_priv: &[u8; 32],
        node_pub: &[u8; 32],
        server_pub: &[u8; 32],
    ) -> Result<()> {
        // ClientInfo JSON, NaCl-boxed to the server's key.
        let json = br#"{"version":2}"#;
        let bx = SalsaBox::new(&PublicKey::from(*server_pub), &SecretKey::from(*node_priv));
        let mut nbytes = [0u8; 24];
        fill_random(&mut nbytes);
        let nonce = Nonce::<SalsaBox>::clone_from_slice(&nbytes);
        let ct = bx
            .encrypt(&nonce, &json[..])
            .map_err(|_| anyhow::anyhow!("clientinfo seal"))?;

        let mut payload = Vec::with_capacity(32 + 24 + ct.len());
        payload.extend_from_slice(node_pub); // our DERP identity = node public key
        payload.extend_from_slice(&nbytes);
        payload.extend_from_slice(&ct);
        self.write_frame(FRAME_CLIENT_INFO, &payload)
    }

    /// Send a tunnel packet to a peer (addressed by node public key) via the relay.
    pub fn send_packet(&mut self, dst_node_pub: &[u8; 32], pkt: &[u8]) -> Result<()> {
        let mut payload = Vec::with_capacity(32 + pkt.len());
        payload.extend_from_slice(dst_node_pub);
        payload.extend_from_slice(pkt);
        self.write_frame(FRAME_SEND_PACKET, &payload)
    }

    /// Read the next frame: returns (type, payload).
    pub fn read_frame(&mut self) -> Result<(u8, Vec<u8>)> {
        let mut hdr = [0u8; 5];
        self.read_exact(&mut hdr)?;
        let typ = hdr[0];
        let len = u32::from_be_bytes([hdr[1], hdr[2], hdr[3], hdr[4]]) as usize;
        if len > MAX_FRAME {
            bail!("DERP frame too large: {len}");
        }
        let mut payload = vec![0u8; len];
        if len > 0 {
            self.read_exact(&mut payload)?;
        }
        Ok((typ, payload))
    }

    /// Like [`read_frame`](Self::read_frame) but reads the payload into a caller-owned
    /// buffer that is reused across frames (its capacity is retained), so the serve
    /// loop does not allocate a fresh `Vec` per received packet.
    pub fn read_frame_into(&mut self, payload: &mut Vec<u8>) -> Result<u8> {
        let mut hdr = [0u8; 5];
        self.read_exact(&mut hdr)?;
        let typ = hdr[0];
        let len = u32::from_be_bytes([hdr[1], hdr[2], hdr[3], hdr[4]]) as usize;
        if len > MAX_FRAME {
            bail!("DERP frame too large: {len}");
        }
        payload.clear();
        payload.resize(len, 0);
        if len > 0 {
            self.read_exact(payload)?;
        }
        Ok(typ)
    }

    /// Read one frame into `payload`, but return `Ok(None)` if no frame arrives
    /// within `timeout_ms` (so the serve loop can run retransmit ticks while idle).
    /// Uses `poll()` on the raw socket; if mbedTLS already has a buffered record
    /// (which `poll` can't see), it reads immediately instead.
    pub fn read_frame_timeout(
        &mut self,
        payload: &mut Vec<u8>,
        timeout_ms: i32,
    ) -> Result<Option<u8>> {
        let buffered = unsafe { sys::esp_tls_get_bytes_avail(self.tls) } > 0;
        if !buffered {
            let mut fd: i32 = -1;
            let r = unsafe { sys::esp_tls_get_conn_sockfd(self.tls, &mut fd) };
            if r == sys::ESP_OK && fd >= 0 {
                let mut pfd = sys::pollfd {
                    fd,
                    events: sys::POLLIN as i16,
                    revents: 0,
                };
                let pr = unsafe { sys::poll(&mut pfd as *mut _, 1 as sys::nfds_t, timeout_ms) };
                if pr == 0 {
                    return Ok(None); // idle: no data within the timeout
                }
                if pr < 0 {
                    bail!("DERP poll error");
                }
            }
            // If we couldn't get the fd, fall through to a blocking read.
        }
        self.read_frame_into(payload).map(Some)
    }

    fn write_frame(&mut self, typ: u8, payload: &[u8]) -> Result<()> {
        let mut hdr = [0u8; 5];
        hdr[0] = typ;
        hdr[1..5].copy_from_slice(&(payload.len() as u32).to_be_bytes());
        self.write_all(&hdr)?;
        if !payload.is_empty() {
            self.write_all(payload)?;
        }
        Ok(())
    }

    // --- raw TLS I/O ---

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut off = 0;
        while off < buf.len() {
            let n = self.read_some(&mut buf[off..])?;
            if n == 0 {
                bail!("DERP connection closed");
            }
            off += n;
        }
        Ok(())
    }

    fn read_some(&mut self, buf: &mut [u8]) -> Result<usize> {
        loop {
            let r =
                unsafe { sys::esp_tls_conn_read(self.tls, buf.as_mut_ptr() as *mut c_void, buf.len()) };
            if r > 0 {
                return Ok(r as usize);
            }
            if r == 0 {
                return Ok(0);
            }
            // ESP_TLS_ERR_SSL_WANT_READ/WRITE -> retry; other -> error.
            let want_read = sys::ESP_TLS_ERR_SSL_WANT_READ as isize;
            let want_write = sys::ESP_TLS_ERR_SSL_WANT_WRITE as isize;
            if r == want_read || r == want_write {
                continue;
            }
            bail!("esp_tls_conn_read error {r}");
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        let mut off = 0;
        while off < buf.len() {
            let r = unsafe {
                sys::esp_tls_conn_write(
                    self.tls,
                    buf[off..].as_ptr() as *const c_void,
                    buf.len() - off,
                )
            };
            if r > 0 {
                off += r as usize;
                continue;
            }
            let want_read = sys::ESP_TLS_ERR_SSL_WANT_READ as isize;
            let want_write = sys::ESP_TLS_ERR_SSL_WANT_WRITE as isize;
            if r == want_read || r == want_write {
                continue;
            }
            bail!("esp_tls_conn_write error {r}");
        }
        Ok(())
    }
}

impl Drop for Derp {
    fn drop(&mut self) {
        unsafe {
            sys::esp_tls_conn_destroy(self.tls);
        }
    }
}

/// Open a validated TLS connection to `host:port` using the cert bundle.
fn tls_connect(host: &str, port: u16) -> Result<*mut sys::esp_tls> {
    let tls = unsafe { sys::esp_tls_init() };
    if tls.is_null() {
        bail!("esp_tls_init failed");
    }
    let mut cfg: sys::esp_tls_cfg_t = unsafe { core::mem::zeroed() };
    // Pin the DERP relay CA (PEM, NUL-terminated) instead of attaching the full
    // bundle. `esp_tls_conn_new_sync` is synchronous and parses the CA during the
    // call, so this stack-local buffer only needs to outlive the connect below.
    let mut ca = Vec::with_capacity(DERP_CA_PEM.len() + 1);
    ca.extend_from_slice(DERP_CA_PEM);
    ca.push(0); // PEM buffer must be NUL-terminated; cacert_bytes includes the NUL
    cfg.__bindgen_anon_1.cacert_buf = ca.as_ptr();
    cfg.__bindgen_anon_2.cacert_bytes = ca.len() as u32;
    cfg.timeout_ms = 15000;

    let host_c = std::ffi::CString::new(host).unwrap();
    let r = unsafe {
        sys::esp_tls_conn_new_sync(
            host_c.as_ptr(),
            host.len() as i32,
            port as i32,
            &cfg,
            tls,
        )
    };
    if r != 1 {
        unsafe { sys::esp_tls_conn_destroy(tls) };
        bail!("esp_tls_conn_new_sync to {host}:{port} failed (r={r})");
    }
    Ok(tls)
}

fn fill_random(out: &mut [u8]) {
    unsafe {
        sys::esp_fill_random(out.as_mut_ptr() as *mut c_void, out.len());
    }
}

/// Run the DERP relay dataplane forever: connect to our home DERP region and act
/// as a WireGuard responder for any peer that reaches us over the relay (the
/// remote/NAT case). Reconnects on error. This is independent of the UDP
/// dataplane — a DERP peer is reached only via the relay, keyed by node key.
pub fn run(id: crate::node::Identity, upgrade: Option<crate::node::Upgrade>) {
    let host = "derp1f.tailscale.com";
    loop {
        match Derp::connect(host, &id.node_priv, &id.node_pub) {
            Ok(mut d) => {
                println!("*** DERP connected to {host} (relay responder up) ***");
                serve(&mut d, &id, upgrade.as_ref());
                println!("DERP: disconnected, reconnecting in 5s");
            }
            Err(e) => println!("DERP connect failed: {e:#}"),
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

/// One peer's tunnel state, reached via the relay.
struct DerpPeer {
    tun: crate::wg::Tunnel,
    /// Last time we handled a packet from this peer (for LRU eviction).
    last_ms: u64,
    #[cfg(feature = "http-server")]
    tcp: crate::tcp::TcpServer,
}

/// Few tailscale peers ever reach us over the relay; cap the table so DERP RAM is
/// bounded no matter how many node keys probe us (LRU-evict the oldest when full).
const MAX_PEERS: usize = 4;
/// Bound on the "already attempted a direct upgrade" set (cleared when it grows).
const MAX_UPGRADED: usize = 16;
/// Idle poll cadence; also the retransmit-tick granularity for in-tunnel TCP.
const IDLE_MS: i32 = 100;

fn serve(d: &mut Derp, id: &crate::node::Identity, upgrade: Option<&crate::node::Upgrade>) {
    use std::collections::HashMap;
    let mut peers: HashMap<[u8; 32], DerpPeer> = HashMap::new();
    let mut upgraded: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    // Reused across frames so we don't allocate a fresh Vec per received packet.
    let mut scratch: Vec<u8> = Vec::with_capacity(2048);

    loop {
        let now = now_ms();
        let typ = match d.read_frame_timeout(&mut scratch, IDLE_MS) {
            Ok(Some(t)) => t,
            Ok(None) => {
                // Idle: drive the in-tunnel TCP retransmit timers for each peer.
                #[cfg(feature = "http-server")]
                for (src, p) in peers.iter_mut() {
                    let src = *src;
                    p.tcp.tick(now, |seg| {
                        let out = p.tun.encrypt(seg);
                        let _ = d.send_packet(&src, &out);
                    });
                }
                continue;
            }
            Err(e) => {
                println!("DERP read error: {e:#}");
                return;
            }
        };
        if typ != FRAME_RECV_PACKET || scratch.len() <= 32 {
            continue; // keepalive, serverinfo, or empty packet
        }
        let mut src = [0u8; 32];
        src.copy_from_slice(&scratch[..32]);
        let pkt = &scratch[32..];

        // derp-upgrade: first time we hear from a peer over the relay, coordinate
        // a direct path — send it CALL_ME_MAYBE with our endpoints and ask the UDP
        // dataplane to probe its endpoints.
        if let Some(up) = upgrade {
            if upgraded.len() > MAX_UPGRADED {
                upgraded.clear(); // bounded; a re-attempt is harmless
            }
            if upgraded.insert(src) {
                try_upgrade(d, id, up, &src);
            }
        }

        // disco over DERP: PONG pings, and harvest the peer's CALL_ME_MAYBE
        // (its FRESH endpoints/ports) to drive a direct hole-punch.
        if crate::disco::is_disco(pkt) {
            if let Ok(msg) = crate::disco::open(&id.disco_priv, pkt) {
                if msg.msg_type == crate::disco::PING {
                    let any = std::net::SocketAddr::from(([0, 0, 0, 0], 0));
                    let pong = crate::disco::pong_plaintext(&msg.txid, any);
                    if let Ok(wire) =
                        crate::disco::seal(&id.disco_priv, &id.disco_pub, &msg.sender_disco_pub, &pong)
                    {
                        let _ = d.send_packet(&src, &wire);
                    }
                } else if msg.msg_type == crate::disco::CALL_ME_MAYBE && !msg.endpoints.is_empty() {
                    if let Some(up) = upgrade {
                        println!(
                            "DERP CALL_ME_MAYBE from {} -> {} fresh endpoint(s)",
                            hex8(&src),
                            msg.endpoints.len()
                        );
                        let _ = up.tx.send(crate::node::Target {
                            name: format!("derp:{}", hex8(&src)),
                            disco_pub: msg.sender_disco_pub,
                            node_pub: src,
                            endpoints: msg.endpoints,
                            spray: true,
                        });
                    }
                }
            }
            continue;
        }

        match pkt.first().copied() {
            Some(x) if x == crate::wg::MSG_INITIATION => {
                let our_index = crate::wg::random_index();
                match crate::wg::consume_initiation(&id.node_priv, &id.node_pub, pkt, our_index) {
                    Ok((resp, tun, _peer_static)) => {
                        let _ = d.send_packet(&src, &resp);
                        // Bound the table: evict the least-recently-used peer if full.
                        if peers.len() >= MAX_PEERS && !peers.contains_key(&src) {
                            if let Some(old) =
                                peers.iter().min_by_key(|(_, p)| p.last_ms).map(|(k, _)| *k)
                            {
                                peers.remove(&old);
                            }
                        }
                        peers.insert(
                            src,
                            DerpPeer {
                                tun,
                                last_ms: now,
                                #[cfg(feature = "http-server")]
                                tcp: crate::new_tcp_server(),
                            },
                        );
                        println!("*** DERP WG HANDSHAKE COMPLETE (responder) with {} ***", hex8(&src));
                    }
                    Err(e) => println!("DERP WG init failed: {e:#}"),
                }
            }
            Some(x) if x == crate::wg::MSG_TRANSPORT => {
                if let Some(p) = peers.get_mut(&src) {
                    p.last_ms = now;
                    match p.tun.decrypt(pkt) {
                        Ok(inner)
                            if !inner.is_empty()
                                && crate::node::src_allowed(&id.allowed_srcs, &inner) =>
                        {
                            handle_inner(d, p, &src, &inner, now);
                        }
                        Ok(_) => {} // keepalive / filtered
                        Err(e) => println!("DERP transport decrypt failed: {e:#}"),
                    }
                }
            }
            _ => {}
        }
    }
}

/// Coordinate a direct path with a relayed peer: send it a disco CALL_ME_MAYBE
/// (our endpoints) over DERP and tell the UDP dataplane to probe its endpoints.
fn try_upgrade(d: &mut Derp, id: &crate::node::Identity, up: &crate::node::Upgrade, src: &[u8; 32]) {
    let peer = match up.peers.iter().find(|p| &p.node_pub == src) {
        Some(p) => p,
        None => return, // unknown peer / no endpoints to try
    };
    if peer.endpoints.is_empty() && up.our_endpoints.is_empty() {
        return;
    }
    // Tell the peer where to reach us directly.
    let cmm = crate::disco::call_me_maybe_plaintext(&up.our_endpoints);
    if let Ok(wire) = crate::disco::seal(&id.disco_priv, &id.disco_pub, &peer.disco_pub, &cmm) {
        let _ = d.send_packet(src, &wire);
    }
    // Ask the UDP dataplane to probe + handshake this peer directly.
    let _ = up.tx.send(crate::node::Target {
        name: format!("derp:{}", hex8(src)),
        disco_pub: peer.disco_pub,
        node_pub: *src,
        endpoints: peer.endpoints.clone(),
        spray: true, // remote peer: may be behind symmetric NAT
    });
    println!("DERP: upgrade attempt -> {} ({} ep)", hex8(src), peer.endpoints.len());
}

/// Handle a decrypted inner IP packet from a DERP peer: ICMP echo reply and/or
/// the in-tunnel HTTP server, reflecting responses back through the relay. The
/// HTTP server streams each segment straight out (encrypt + relay per segment) so
/// peak RAM stays O(1) regardless of response size; `tick` (driven from the serve
/// loop) handles retransmits.
fn handle_inner(d: &mut Derp, p: &mut DerpPeer, src: &[u8; 32], inner: &[u8], now_ms: u64) {
    #[cfg(feature = "icmp")]
    if let Some(reply) = tailscale_core::icmp::echo_reply_any(inner) {
        let out = p.tun.encrypt(&reply);
        let _ = d.send_packet(src, &out);
        println!("DERP ICMP echo -> replied to {}", hex8(src));
        return;
    }
    #[cfg(feature = "http-server")]
    {
        p.tcp.handle_stream(inner, now_ms, |seg| {
            let out = p.tun.encrypt(seg);
            let _ = d.send_packet(src, &out);
        });
        if let Some(a) = p.tcp.take_action() {
            crate::ui::dispatch_tcp(a);
        }
    }
    let _ = (&mut *p, &*d, src, inner, now_ms); // some combos use a subset of these
}

fn hex8(b: &[u8]) -> String {
    let mut s = String::new();
    for x in b.iter().take(8) {
        s.push_str(&format!("{x:02x}"));
    }
    s
}
