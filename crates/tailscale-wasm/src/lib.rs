//! Browser-WASM binding for `tailscale-core`: join the user's OWN tailnet from a browser
//! tab, over WebSocket (browsers have no raw TCP/UDP). Reuses the crate's async control
//! path (`noise` → `AsyncConn` → `AsyncH2`) verbatim; the only new thing is a WebSocket-
//! backed `AsyncByteStream` and the browser glue (web-crypto RNG, localStorage keys).
//!
//! Control dial (reverse-engineered from tailscale `control/controlhttp/client_js.go`):
//!   wss://controlplane.tailscale.com/ts2021?tskey=<base64(noise_init)>  subprotocol `control`
//! After open the WS carries raw controlbase frames, so `AsyncConn::from_stream` + the
//! existing framing/h2 just work.
//!
//! Node identity (3 x25519 keypairs) is persisted in `localStorage`, so a browser refresh
//! reuses the SAME node — same pending registration / same AuthURL, not a fresh one.
//! Interactive auth: register with no auth key → the server returns an `AuthURL`; open it
//! to authorize, then re-register (same keys) until `MachineAuthorized` → fetch the IP.

use std::cell::RefCell;
use std::future::poll_fn;
use std::rc::Rc;
use std::task::{Poll, Waker};

use crypto_box::aead::{Aead, Nonce};
use crypto_box::{PublicKey, SalsaBox, SecretKey};
use tailscale_core::platform::AsyncByteStream;
use tailscale_core::{h2, noise, tcp, transport, wg};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

fn now_ms() -> u64 {
    js_sys::Date::now() as u64
}

// --- Phase 2: serve PySpell over the tunnel (the simulated ESP32) ------------
static NODE_PAGE: &[u8] = b"<!doctype html><meta charset=utf-8><title>PySpell \xc2\xb7 your browser ESP32</title><body style='margin:0;font-family:system-ui;background:#0d1117;color:#e6edf3;padding:1.4rem;line-height:1.5'><h2>\xf0\x9f\xa7\xaa Hello from your browser-simulated ESP32</h2><p>This page is served by a PySpell node running in <b>another browser tab on your tailnet</b> \xe2\x80\x94 reached over Tailscale (control + DERP + WireGuard, all WebAssembly, ~150&nbsp;kB). Try the API:</p><pre style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.8rem'>curl http://&lt;this-node&gt;/run -d '21*2'   # =&gt; 42\ncurl http://&lt;this-node&gt;/run -d 'max([3,9,5])'</pre></body>";

fn render_value(v: &pyspell_core::Value) -> String {
    use pyspell_core::Value;
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
                s.push_str(&render_value(it));
            }
            s.push(']');
            s
        }
    }
}
fn run_pyspell(code: &str) -> String {
    use pyspell_core::{eval, parse, Lang, Limits};
    let code = code.trim();
    if code.is_empty() {
        return "error: empty program".into();
    }
    let prog = match parse(code, Lang::Python) {
        Ok(p) => p,
        Err(e) => return format!("error: {e}"),
    };
    let env = |_: &str| None::<pyspell_core::Value>;
    let limits = Limits {
        max_steps: 1_000_000,
        max_bytes: 131_072,
        deadline: None,
        net: None,
        display: None,
        actuator: None,
    };
    match eval::run_with(&prog, &env, limits) {
        Ok(v) => render_value(&v),
        Err(e) => format!("error: {e}"),
    }
}
/// In-tunnel HTTP routes the browser node serves to tailnet peers (your phone).
fn route(method: &str, path: &str, query: &str, body: &[u8]) -> Option<tcp::HttpReply> {
    let _ = method;
    match path {
        "/" => Some(tcp::HttpReply::ok_owned("text/html; charset=utf-8", NODE_PAGE.to_vec())),
        "/run" => {
            let code = if !body.is_empty() {
                String::from_utf8_lossy(body).into_owned()
            } else {
                // allow ?code=... too
                query
                    .split('&')
                    .find_map(|kv| kv.strip_prefix("code="))
                    .map(|s| s.replace('+', " "))
                    .unwrap_or_default()
            };
            Some(tcp::HttpReply::ok_owned("text/plain; charset=utf-8", run_pyspell(&code).into_bytes()))
        }
        _ => None,
    }
}

const CONTROL_HOST: &str = "controlplane.tailscale.com";
const CAP_VER: u32 = 90;

// --- web-crypto getrandom (tailscale-core forces getrandom `custom`) ---------
fn web_getrandom(buf: &mut [u8]) -> Result<(), getrandom::Error> {
    let crypto = web_sys::window()
        .and_then(|w| w.crypto().ok())
        .ok_or(getrandom::Error::UNSUPPORTED)?;
    // get_random_values caps at 65536 bytes; our buffers are tiny (≤64 B).
    crypto
        .get_random_values_with_u8_array(buf)
        .map_err(|_| getrandom::Error::UNSUPPORTED)?;
    Ok(())
}
getrandom::register_custom_getrandom!(web_getrandom);

// --- WebSocket-backed AsyncByteStream ---------------------------------------
#[derive(Default)]
struct WsInner {
    rx: Vec<u8>,
    opened: bool,
    closed: bool,
    read_waker: Option<Waker>,
    open_waker: Option<Waker>,
}
impl WsInner {
    fn wake_read(&mut self) {
        if let Some(w) = self.read_waker.take() {
            w.wake();
        }
    }
    fn wake_open(&mut self) {
        if let Some(w) = self.open_waker.take() {
            w.wake();
        }
    }
}

struct WsStream {
    ws: web_sys::WebSocket,
    inner: Rc<RefCell<WsInner>>,
    // keep the JS closures alive for the socket's lifetime
    _onmsg: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _onopen: Closure<dyn FnMut()>,
    _onclose: Closure<dyn FnMut(web_sys::CloseEvent)>,
    _onerr: Closure<dyn FnMut(web_sys::ErrorEvent)>,
}

async fn ws_connect(url: &str, subproto: &str) -> Result<WsStream, String> {
    let ws = web_sys::WebSocket::new_with_str(url, subproto)
        .map_err(|e| format!("WebSocket::new: {e:?}"))?;
    ws.set_binary_type(web_sys::BinaryType::Arraybuffer);
    let inner = Rc::new(RefCell::new(WsInner::default()));

    let onmsg = {
        let inner = inner.clone();
        Closure::<dyn FnMut(_)>::new(move |e: web_sys::MessageEvent| {
            if let Ok(buf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                let bytes = js_sys::Uint8Array::new(&buf).to_vec();
                let mut g = inner.borrow_mut();
                g.rx.extend_from_slice(&bytes);
                g.wake_read();
            }
        })
    };
    let onopen = {
        let inner = inner.clone();
        Closure::<dyn FnMut()>::new(move || {
            let mut g = inner.borrow_mut();
            g.opened = true;
            g.wake_open();
        })
    };
    let onclose = {
        let inner = inner.clone();
        Closure::<dyn FnMut(_)>::new(move |_e: web_sys::CloseEvent| {
            let mut g = inner.borrow_mut();
            g.closed = true;
            g.wake_open();
            g.wake_read();
        })
    };
    let onerr = {
        let inner = inner.clone();
        Closure::<dyn FnMut(_)>::new(move |_e: web_sys::ErrorEvent| {
            let mut g = inner.borrow_mut();
            g.closed = true;
            g.wake_open();
            g.wake_read();
        })
    };
    ws.set_onmessage(Some(onmsg.as_ref().unchecked_ref()));
    ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
    ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
    ws.set_onerror(Some(onerr.as_ref().unchecked_ref()));

    // await open (or fail on close/error)
    {
        let inner = inner.clone();
        poll_fn(move |cx| {
            let mut g = inner.borrow_mut();
            if g.opened {
                Poll::Ready(Ok(()))
            } else if g.closed {
                Poll::Ready(Err("WebSocket closed before open".to_string()))
            } else {
                g.open_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await?;
    }
    Ok(WsStream { ws, inner, _onmsg: onmsg, _onopen: onopen, _onclose: onclose, _onerr: onerr })
}

impl AsyncByteStream for WsStream {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        let inner = self.inner.clone();
        poll_fn(move |cx| {
            let mut g = inner.borrow_mut();
            if !g.rx.is_empty() {
                let n = g.rx.len().min(buf.len());
                buf[..n].copy_from_slice(&g.rx[..n]);
                g.rx.drain(..n);
                if g.rx.is_empty() && g.rx.capacity() > 16384 {
                    g.rx.shrink_to_fit();
                }
                Poll::Ready(Ok(n))
            } else if g.closed {
                Poll::Ready(Ok(0)) // EOF
            } else {
                g.read_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await
    }
    async fn write_all(&mut self, buf: &[u8]) -> Result<(), ()> {
        // WebSocket.send buffers internally; our control frames are small.
        self.ws.send_with_u8_array(buf).map_err(|_| ())
    }
}

// --- keys (persisted in localStorage so refresh reuses the same node) --------
fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}
fn store_get(key: &str) -> Option<String> {
    storage()?.get_item(key).ok().flatten()
}
fn store_set(key: &str, val: &str) {
    if let Some(s) = storage() {
        let _ = s.set_item(key, val);
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
    let hv = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    for i in 0..32 {
        out[i] = (hv(b[i * 2])? << 4) | hv(b[i * 2 + 1])?;
    }
    Some(out)
}
fn pub_of(priv32: &[u8; 32]) -> [u8; 32] {
    let s = x25519_dalek::StaticSecret::from(*priv32);
    x25519_dalek::PublicKey::from(&s).to_bytes()
}
/// Load a persisted private key by `name`, or generate + persist a fresh one.
fn load_or_gen(name: &str) -> [u8; 32] {
    let k = format!("ts_{name}_priv");
    if let Some(hex) = store_get(&k) {
        if let Some(b) = hex_decode_32(&hex) {
            return b;
        }
    }
    let mut b = [0u8; 32];
    let _ = web_getrandom(&mut b);
    store_set(&k, &hex_lower(&b));
    b
}

// --- control JSON (hand-built; browser node, ephemeral) ---------------------
fn hostinfo() -> String {
    String::from("{\"Hostname\":\"pyspell-web\",\"OS\":\"browser\",\"OSVersion\":\"wasm\",\"GoArch\":\"wasm\",\"NetInfo\":{\"WorkingIPv4\":true,\"PreferredDERP\":1}}")
}
fn register_json(mpub: &str, npub: &str, dpub: &str, auth_key: &str) -> String {
    let auth = if auth_key.is_empty() {
        String::new()
    } else {
        format!(",\"Auth\":{{\"AuthKey\":\"{auth_key}\"}}")
    };
    format!(
        "{{\"Version\":{CAP_VER},\"NodeKey\":\"nodekey:{npub}\",\"MachineKey\":\"mkey:{mpub}\",\"DiscoKey\":\"discokey:{dpub}\",\"Hostinfo\":{hi},\"Endpoints\":[],\"Capabilities\":[],\"DeviceName\":\"pyspell-web\",\"Ephemeral\":true{auth}}}",
        hi = hostinfo()
    )
}
fn map_json(npub: &str, dpub: &str) -> String {
    format!(
        "{{\"Version\":{CAP_VER},\"NodeKey\":\"nodekey:{npub}\",\"DiscoKey\":\"discokey:{dpub}\",\"Endpoints\":[],\"Hostinfo\":{hi},\"Stream\":false,\"OmitPeers\":false,\"ReadOnly\":false,\"Compress\":\"\"}}",
        hi = hostinfo()
    )
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}
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
fn extract_str_field(raw: &[u8], field: &str) -> Option<String> {
    let pat = format!("\"{field}\":\"");
    let p = raw.windows(pat.len()).position(|w| w == pat.as_bytes())?;
    let start = p + pat.len();
    let mut end = start;
    while end < raw.len() && raw[end] != b'"' {
        end += 1;
    }
    core::str::from_utf8(&raw[start..end]).ok().map(String::from)
}

fn url_encode_b64(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '+' => o.push_str("%2B"),
            '/' => o.push_str("%2F"),
            '=' => o.push_str("%3D"),
            c => o.push(c),
        }
    }
    o
}

fn json_result(status: u16, authorized: bool, auth_url: &Option<String>, ip: &Option<String>) -> String {
    let au = match auth_url {
        Some(u) => format!("\"{}\"", u.replace('\\', "\\\\").replace('"', "\\\"")),
        None => "null".to_string(),
    };
    let ipj = match ip {
        Some(i) => format!("\"{i}\""),
        None => "null".to_string(),
    };
    format!("{{\"ok\":true,\"status\":{status},\"authorized\":{authorized},\"authUrl\":{au},\"ip\":{ipj}}}")
}
fn json_err(msg: &str) -> String {
    format!("{{\"ok\":false,\"error\":\"{}\"}}", msg.replace('\\', "\\\\").replace('"', "\\\""))
}

/// One control round-trip over WebSocket: handshake → register → (if authorized) map →
/// IP. Reuses the persisted node identity. `control_pub_hex` is the `/key` mkey (the JS
/// fetches it — CORS-OK). `auth_key` empty = interactive (returns an AuthURL to open).
/// Returns JSON `{ok,status,authorized,authUrl,ip}` or `{ok:false,error}`.
#[wasm_bindgen]
pub async fn register_once(control_pub_hex: String, auth_key: String) -> String {
    match register_inner(&control_pub_hex, &auth_key).await {
        Ok(s) => s,
        Err(e) => json_err(&e),
    }
}

async fn register_inner(control_pub_hex: &str, auth_key: &str) -> Result<String, String> {
    let control_pub = hex_decode_32(control_pub_hex).ok_or("bad control pubkey hex")?;
    let machine_priv = load_or_gen("machine");
    let node_priv = load_or_gen("node");
    let disco_priv = load_or_gen("disco");
    let (mpub, npub, dpub) = (
        hex_lower(&pub_of(&machine_priv)),
        hex_lower(&pub_of(&node_priv)),
        hex_lower(&pub_of(&disco_priv)),
    );

    let (hs, framed_init) =
        noise::start(&machine_priv, &control_pub).map_err(|e| format!("noise start: {e}"))?;
    // Exact browser control dial (tailscale control/controlhttp + controlhttpcommon):
    // query param = HandshakeHeaderName "X-Tailscale-Handshake", WS subprotocol =
    // UpgradeHeaderValue "tailscale-control-protocol". (base64-std init, url-encoded.)
    let hsval = url_encode_b64(&transport::base64_std(&framed_init));
    let url = format!("wss://{CONTROL_HOST}/ts2021?X-Tailscale-Handshake={hsval}");

    let stream = ws_connect(&url, "tailscale-control-protocol").await?;
    let mut conn = transport::AsyncConn::from_stream(stream);
    let (typ, payload) = conn.read_frame().await.map_err(|e| format!("read resp: {e}"))?;
    if typ != noise::MSG_RESPONSE {
        return Err(format!("unexpected handshake frame type {typ}"));
    }
    let tr = hs.complete(&payload).map_err(|e| format!("noise complete: {e}"))?;
    let (mut sess, early) = h2::AsyncH2::start(conn, tr, String::from(CONTROL_HOST))
        .await
        .map_err(|e| format!("h2 start: {e}"))?;

    let mut ip = early.as_deref().and_then(scan_tailscale_ip);

    let reg = register_json(&mpub, &npub, &dpub, auth_key);
    let (status, rresp) = sess
        .post_json("/machine/register", reg.as_bytes())
        .await
        .map_err(|e| format!("register: {e}"))?;
    let authorized = contains(&rresp, b"\"MachineAuthorized\":true");
    let auth_url = extract_str_field(&rresp, "AuthURL").filter(|s| !s.is_empty());

    if authorized && ip.is_none() {
        let mapj = map_json(&npub, &dpub);
        if let Ok((_s, mresp)) = sess.post_json("/machine/map", mapj.as_bytes()).await {
            ip = scan_tailscale_ip(&mresp);
        }
    }
    Ok(json_result(status, authorized, &auth_url, &ip))
}

// --- Phase 1: DERP data-plane client over WebSocket -------------------------
// DERP frames: [type u8][len u32 BE][payload]. Handshake: server sends FRAME_SERVER_KEY
// (magic + 32-byte pub); we reply FRAME_CLIENT_INFO (node_pub + nonce + NaCl-boxed JSON).
// Then peers' WireGuard/disco packets arrive as FRAME_RECV_PACKET (src_node_pub + pkt).
const FRAME_SERVER_KEY: u8 = 0x01;
const FRAME_CLIENT_INFO: u8 = 0x02;
const FRAME_SERVER_INFO: u8 = 0x03;
const FRAME_SEND_PACKET: u8 = 0x04;
const FRAME_RECV_PACKET: u8 = 0x05;
const FRAME_KEEPALIVE: u8 = 0x06;
const DERP_MAGIC: &[u8] = b"DERP\xf0\x9f\x94\x91"; // "DERP🔑"

thread_local! {
    static DERP_LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}
fn dlog(s: String) {
    DERP_LOG.with(|l| {
        let mut v = l.borrow_mut();
        v.push(s);
        if v.len() > 60 {
            let n = v.len() - 60;
            v.drain(..n);
        }
    });
}

/// Recent DERP log lines (newline-joined) for the UI to poll.
#[wasm_bindgen]
pub fn derp_log() -> String {
    DERP_LOG.with(|l| l.borrow().join("\n"))
}

fn hex8(b: &[u8]) -> String {
    b.iter().take(8).map(|x| format!("{x:02x}")).collect()
}

/// A framed DERP reader/writer over the WebSocket stream (5-byte header).
struct DerpConn {
    stream: WsStream,
    rx: Vec<u8>,
}
impl DerpConn {
    async fn fill_to(&mut self, n: usize) -> Result<(), String> {
        let mut tmp = [0u8; 4096];
        while self.rx.len() < n {
            let r = self.stream.read(&mut tmp).await.map_err(|_| "derp read".to_string())?;
            if r == 0 {
                return Err("derp connection closed".into());
            }
            self.rx.extend_from_slice(&tmp[..r]);
        }
        Ok(())
    }
    async fn read_frame(&mut self) -> Result<(u8, Vec<u8>), String> {
        self.fill_to(5).await?;
        let typ = self.rx[0];
        let len = u32::from_be_bytes([self.rx[1], self.rx[2], self.rx[3], self.rx[4]]) as usize;
        if len > 64 * 1024 {
            return Err(format!("derp frame too large: {len}"));
        }
        self.fill_to(5 + len).await?;
        let payload = self.rx[5..5 + len].to_vec();
        self.rx.drain(..5 + len);
        if self.rx.is_empty() && self.rx.capacity() > 16384 {
            self.rx.shrink_to_fit();
        }
        Ok((typ, payload))
    }
    async fn write_frame(&mut self, typ: u8, payload: &[u8]) -> Result<(), String> {
        let mut f = Vec::with_capacity(5 + payload.len());
        f.push(typ);
        f.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        f.extend_from_slice(payload);
        self.stream.write_all(&f).await.map_err(|_| "derp write".to_string())
    }
    /// Relay a packet to a peer (addressed by node public key) via the DERP server.
    async fn send_packet(&mut self, dst_node_pub: &[u8; 32], pkt: &[u8]) -> Result<(), String> {
        let mut payload = Vec::with_capacity(32 + pkt.len());
        payload.extend_from_slice(dst_node_pub);
        payload.extend_from_slice(pkt);
        self.write_frame(FRAME_SEND_PACKET, &payload).await
    }
}

/// Connect to a DERP relay over WebSocket, do the DERP handshake (our node key = DERP
/// identity), then loop reading frames — logging any peer packets that arrive (proof the
/// phone can reach this browser node over the tunnel). Phase 2 will WireGuard-decrypt and
/// serve them. Runs until the connection drops. `derp_host` e.g. "derp1f.tailscale.com".
#[wasm_bindgen]
pub async fn connect_derp(derp_host: String) -> String {
    match derp_inner(&derp_host).await {
        Ok(s) => s,
        Err(e) => {
            dlog(format!("✖ {e}"));
            json_err(&e)
        }
    }
}

async fn derp_inner(derp_host: &str) -> Result<String, String> {
    let node_priv = load_or_gen("node");
    let node_pub = pub_of(&node_priv);
    let url = format!("wss://{derp_host}/derp");
    dlog(format!("connecting {url} …"));
    let stream = ws_connect(&url, "derp").await?;
    let mut d = DerpConn { stream, rx: Vec::new() };

    let (typ, payload) = d.read_frame().await?;
    if typ != FRAME_SERVER_KEY || payload.len() < DERP_MAGIC.len() + 32 || &payload[..DERP_MAGIC.len()] != DERP_MAGIC {
        return Err(format!("bad server-key frame (type {typ}, {} B)", payload.len()));
    }
    let mut server_pub = [0u8; 32];
    server_pub.copy_from_slice(&payload[DERP_MAGIC.len()..DERP_MAGIC.len() + 32]);
    dlog("got server key; sending client info…".into());

    let json = br#"{"version":2}"#;
    let bx = SalsaBox::new(&PublicKey::from(server_pub), &SecretKey::from(node_priv));
    let mut nbytes = [0u8; 24];
    let _ = web_getrandom(&mut nbytes);
    let nonce = Nonce::<SalsaBox>::clone_from_slice(&nbytes);
    let ct = bx.encrypt(&nonce, &json[..]).map_err(|_| "clientinfo seal".to_string())?;
    let mut ci = Vec::with_capacity(32 + 24 + ct.len());
    ci.extend_from_slice(&node_pub);
    ci.extend_from_slice(&nbytes);
    ci.extend_from_slice(&ct);
    d.write_frame(FRAME_CLIENT_INFO, &ci).await?;
    dlog("✅ DERP session up — open your node's IP from another tailnet device".into());

    // Phase 2: be the WireGuard RESPONDER + serve PySpell over the tunnel.
    let mut tcpsrv = tcp::TcpServer::with_handler(route);
    let mut tuns: Vec<(u32, wg::Tunnel, [u8; 32])> = Vec::new(); // (our_index, tunnel, peer node)
    let mut npkt = 0u32;
    loop {
        let (typ, payload) = d.read_frame().await?;
        if typ == FRAME_KEEPALIVE {
            continue;
        }
        if typ == FRAME_SERVER_INFO {
            dlog("server info received".into());
            continue;
        }
        if typ != FRAME_RECV_PACKET || payload.len() < 32 {
            continue;
        }
        let mut src = [0u8; 32];
        src.copy_from_slice(&payload[..32]);
        let inner = payload[32..].to_vec();
        match inner.first().copied() {
            Some(1) => {
                // WireGuard initiation → respond, build the tunnel (we are responder).
                npkt += 1;
                let mut ib = [0u8; 4];
                let _ = web_getrandom(&mut ib);
                let our_index = u32::from_le_bytes(ib);
                match wg::consume_initiation(&node_priv, &node_pub, &inner, our_index) {
                    Ok((resp, tun, _peer)) => {
                        d.send_packet(&src, &resp).await?;
                        tuns.push((our_index, tun, src));
                        if tuns.len() > 4 {
                            tuns.remove(0);
                        }
                        dlog(format!("🤝 #{npkt} WG handshake from {} — tunnel up", hex8(&src)));
                    }
                    Err(e) => dlog(format!("WG init error: {e}")),
                }
            }
            Some(4) => {
                // WireGuard transport → decrypt, serve via tcp.rs, reply through DERP.
                if inner.len() < 8 {
                    continue;
                }
                let ri = u32::from_le_bytes([inner[4], inner[5], inner[6], inner[7]]);
                if let Some(pos) = tuns.iter().position(|t| t.0 == ri) {
                    let plain = match tuns[pos].1.decrypt(&inner) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    if plain.is_empty() {
                        continue; // keepalive
                    }
                    let now = now_ms();
                    let mut outs: Vec<Vec<u8>> = Vec::new();
                    {
                        let tun = &mut tuns[pos].1;
                        tcpsrv.handle_stream(&plain, now, |seg| outs.push(tun.encrypt(seg)));
                    }
                    let peer = tuns[pos].2;
                    if !outs.is_empty() {
                        npkt += 1;
                        dlog(format!("➡ #{npkt} served {} reply segment(s) to {}", outs.len(), hex8(&peer)));
                    }
                    for enc in &outs {
                        d.send_packet(&peer, enc).await?;
                    }
                }
            }
            _ => { /* disco / other — ignored (DERP-relayed, no direct path needed) */ }
        }
    }
}

/// Clear the persisted node identity (start fresh next time).
#[wasm_bindgen]
pub fn forget_node() {
    if let Some(s) = storage() {
        for k in ["ts_machine_priv", "ts_node_priv", "ts_disco_priv"] {
            let _ = s.remove_item(k);
        }
    }
}
