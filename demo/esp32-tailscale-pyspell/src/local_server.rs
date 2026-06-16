//! Local-LAN parallel PySpell server.
//!
//! A bounded pool of worker threads drains a queue of accepted TCP connections, so
//! several `POST /run` requests are evaluated concurrently — the demo's ~260 kB heap
//! has room that the single-connection in-tunnel server doesn't use. Each worker
//! reuses [`crate::pyspell_web::route`] (same parser/evaluator/Net as the tunnel),
//! so behaviour is identical; only the transport (plain HTTP on the WiFi LAN) and the
//! concurrency differ.
//!
//! Concurrency is bounded by memory, not logic: every worker thread costs its own
//! stack (esp-idf threads aren't cooperative like the lean embassy build), and a
//! `fetch_json` worker also holds an mbedTLS session (~32 kB) while it runs. So the
//! pool size is a memory budget: `N * (stack + peak-TLS)` must fit free heap.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tailscale_core::tcp::{parse_range, HttpReply};

const PORT: u16 = 8080;
const MAX_HEADER: usize = 8192;

/// Spawn the acceptor + worker pool on the default LAN port (8080, fetch rejected).
pub fn run(n_workers: usize, worker_stack: usize) {
    run_port(PORT, n_workers, worker_stack, false)
}

/// Spawn the acceptor + worker pool on `port`. Call from its own thread (it loops
/// forever on `accept`). `worker_stack` must be large enough for a PySpell mbedTLS
/// fetch (~32 kB) if `allow_fetch`. Used by the lwIP bridge to serve our 100.x on
/// :80 with full routing (the WG netif accepts here over real TCP).
pub fn run_port(port: u16, n_workers: usize, worker_stack: usize, allow_fetch: bool) {
    let listener = match TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => {
            println!("[local] bind :{port} failed: {e}");
            return;
        }
    };

    // Accept the burst fast (a rendezvous channel made the 8 simultaneous connections
    // overflow the listen backlog → timeouts). Socket EXHAUSTION is instead handled by
    // SO_LINGER=0 on each connection (freed immediately on close, no TIME_WAIT pile-up),
    // so an 8-burst holds ~8 sockets only momentarily and tailscale keeps its share.
    let (tx, rx) = mpsc::channel::<TcpStream>();
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..n_workers {
        let rx = Arc::clone(&rx);
        let _ = thread::Builder::new()
            .stack_size(worker_stack)
            .spawn(move || worker_loop(i, rx, allow_fetch));
    }
    println!("[local] PySpell pool up on :{port} ({n_workers} workers, {worker_stack}B stack, fetch={allow_fetch})");

    // The job queue holds only the accepted connection (minimal) — workers do the
    // heavy parse+eval, so a burst of requests queues instead of spawning unbounded.
    for stream in listener.incoming() {
        if let Ok(s) = stream {
            let peer = s.peer_addr().map(|a| a.to_string()).unwrap_or_default();
            println!("[srv:{port}] accept {peer}");
            let _ = tx.send(s);
        }
    }
}

fn worker_loop(id: usize, rx: Arc<Mutex<Receiver<TcpStream>>>, allow_fetch: bool) {
    loop {
        let stream = {
            let guard = match rx.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            guard.recv()
        };
        match stream {
            Ok(s) => handle(id, s, allow_fetch),
            Err(_) => return, // sender dropped
        }
    }
}

fn handle(id: usize, mut stream: TcpStream, allow_fetch: bool) {
    // Short read timeout: a connection stalled by a transient socket-pool peak — or a
    // browser speculative *preconnect* that opens a socket but sends no request —
    // frees the (single) worker quickly instead of blocking the queue behind it.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    // SO_LINGER=0: close() sends RST instead of FIN, so the socket is freed IMMEDIATELY
    // rather than sitting in TIME_WAIT (~minutes) holding an LWIP fd. Under a burst of
    // short request/response connections, TIME_WAIT fds otherwise pile up and starve
    // tailscale's sockets (online poll + DERP). The response is written+flushed before
    // close, so the RST loses no data.
    set_linger_zero(&stream);

    // Read until the header terminator.
    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    let mut tmp = [0u8; 1024];
    let header_end = loop {
        match stream.read(&mut tmp) {
            Ok(0) => return,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if let Some(p) = find(&buf, b"\r\n\r\n") {
                    break p;
                }
                if buf.len() > MAX_HEADER {
                    return;
                }
            }
            Err(_) => return,
        }
    };

    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split(' ');
    let method = parts.next().unwrap_or("GET");
    let target = parts.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    let mut content_len = 0usize;
    for l in lines {
        let lower = l.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_len = v.trim().parse().unwrap_or(0);
        }
    }

    // Body bytes already read, then top up to Content-Length.
    let mut body: Vec<u8> = buf[header_end + 4..].to_vec();
    while body.len() < content_len {
        match stream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
            Err(_) => break,
        }
    }

    // These workers have THIN stacks (compute only). A `fetch_json` would drive
    // mbedTLS and overflow the stack, so reject it here and point at the in-tunnel
    // server (which has a fetch-capable stack). Compute jobs run fully parallel.
    let n = if !allow_fetch && path == "/run" && find(&body, b"fetch").is_some() {
        let msg: &[u8] = b"error: fetch_json is not available on the parallel LAN pool (thin stacks); use the in-tunnel server\n";
        write_full(&mut stream, 200, "text/plain; charset=utf-8", msg);
        msg.len()
    } else {
        match crate::pyspell_web::route(method, path, query, &body) {
            // Stream the response straight from its BodySource (real lwIP TCP does the
            // windowing/retransmit), honouring `Range:` — so a large body (e.g. the
            // model file) serves with O(1) RAM + partial fetch, never materialised.
            Some(r) => stream_reply(&mut stream, r, head.as_bytes()),
            None => {
                write_full(&mut stream, 404, "text/plain; charset=utf-8", b"not found\n");
                0
            }
        }
    };
    println!("[srv] w{id} serve {method} {path} -> {n} B");
    let _ = stream.flush();
    println!("[srv] w{id} done {path}");
}

/// Write a complete small response (status line + headers + body) to the stream.
fn write_full(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let reason = if status == 404 { "404 Not Found" } else { "200 OK" };
    let head = format!(
        "HTTP/1.1 {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

/// Stream an [`HttpReply`] over a real TCP stream, honouring a `Range:` request
/// (`req_headers` = the raw request header bytes). The body is read from its
/// `BodySource` in chunks and written incrementally — never copied whole — so the
/// peak RAM is one chunk regardless of body size. Returns body bytes sent.
fn stream_reply(stream: &mut TcpStream, reply: HttpReply, req_headers: &[u8]) -> usize {
    let total = reply.source.total_len();
    let (status, base, len) = match parse_range(req_headers, total) {
        Ok(None) => (200u16, 0usize, total),
        Ok(Some((a, b))) => (206u16, a, b - a + 1),
        Err(()) => {
            let head = format!(
                "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{total}\r\nContent-Length: 0\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccept-Ranges: bytes\r\n\r\n"
            );
            let _ = stream.write_all(head.as_bytes());
            return 0;
        }
    };
    let reason = if status == 206 { "206 Partial Content" } else { "200 OK" };
    let mut head = format!(
        "HTTP/1.1 {reason}\r\nContent-Type: {}\r\nContent-Length: {len}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccept-Ranges: bytes\r\n",
        reply.content_type
    );
    if status == 206 {
        head.push_str(&format!(
            "Content-Range: bytes {}-{}/{}\r\n",
            base,
            base + len - 1,
            total
        ));
    }
    head.push_str("\r\n");
    if stream.write_all(head.as_bytes()).is_err() {
        return 0;
    }
    let end = base + len;
    let mut off = base;
    let mut chunk = [0u8; 2048];
    while off < end {
        let want = (end - off).min(chunk.len());
        let got = reply.source.read_at(off, &mut chunk[..want]);
        if got == 0 {
            break;
        }
        if stream.write_all(&chunk[..got]).is_err() {
            break;
        }
        off += got;
    }
    off - base
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Set SO_LINGER=0 so the socket is freed immediately on close (RST, no TIME_WAIT).
/// Done via the raw fd + lwIP setsockopt (esp-idf's std lacks a stable `set_linger`).
fn set_linger_zero(stream: &TcpStream) {
    use std::os::fd::AsRawFd;
    // lwIP socket option values (sockets.h): SOL_SOCKET=0xfff, SO_LINGER=0x0080.
    const SOL_SOCKET_LWIP: i32 = 0x0fff;
    const SO_LINGER_LWIP: i32 = 0x0080;
    let l = esp_idf_svc::sys::linger {
        l_onoff: 1,
        l_linger: 0,
    };
    unsafe {
        esp_idf_svc::sys::lwip_setsockopt(
            stream.as_raw_fd(),
            SOL_SOCKET_LWIP,
            SO_LINGER_LWIP,
            &l as *const _ as *const core::ffi::c_void,
            core::mem::size_of::<esp_idf_svc::sys::linger>() as esp_idf_svc::sys::socklen_t,
        );
    }
}
