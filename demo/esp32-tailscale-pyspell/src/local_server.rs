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

const PORT: u16 = 8080;
const MAX_HEADER: usize = 8192;

/// Spawn the acceptor + worker pool. Call from its own thread (it loops forever on
/// `accept`). `worker_stack` must be large enough for a PySpell mbedTLS fetch
/// (~32 kB proven on the dataplane thread).
pub fn run(n_workers: usize, worker_stack: usize) {
    let listener = match TcpListener::bind(("0.0.0.0", PORT)) {
        Ok(l) => l,
        Err(e) => {
            println!("[local] bind :{PORT} failed: {e}");
            return;
        }
    };

    // RENDEZVOUS channel (capacity 0): the acceptor only accept()s the next connection
    // once a worker takes the current one. An UNBOUNDED queue would accept every burst
    // connection immediately — and each accepted TcpStream holds an LWIP socket while it
    // waits — which starved tailscale's DERP socket (10/16-socket pool is shared). With
    // rendezvous, the pool holds at most ~N_WORKERS+1 sockets; the rest wait in the
    // listener backlog (no socket fd), reserving the remaining sockets for tailscale.
    let (tx, rx) = mpsc::sync_channel::<TcpStream>(0);
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..n_workers {
        let rx = Arc::clone(&rx);
        let _ = thread::Builder::new()
            .stack_size(worker_stack)
            .spawn(move || worker_loop(i, rx));
    }
    println!("[local] PySpell pool up on :{PORT} ({n_workers} workers, {worker_stack}B stack)");

    // The job queue holds only the accepted connection (minimal) — workers do the
    // heavy parse+eval, so a burst of requests queues instead of spawning unbounded.
    for stream in listener.incoming() {
        if let Ok(s) = stream {
            let _ = tx.send(s);
        }
    }
}

fn worker_loop(id: usize, rx: Arc<Mutex<Receiver<TcpStream>>>) {
    loop {
        let stream = {
            let guard = match rx.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            guard.recv()
        };
        match stream {
            Ok(s) => handle(id, s),
            Err(_) => return, // sender dropped
        }
    }
}

fn handle(id: usize, mut stream: TcpStream) {
    // Short read timeout: a connection stalled by a transient socket-pool peak frees
    // its worker quickly (instead of blocking it ~15 s), so the pool recovers fast.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(6)));

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
    let (ct, rbody) = if path == "/run" && find(&body, b"fetch").is_some() {
        (
            "text/plain; charset=utf-8",
            b"error: fetch_json is not available on the parallel LAN pool (thin stacks); use the in-tunnel server\n".to_vec(),
        )
    } else {
        match crate::pyspell_web::route(method, path, query, &body) {
            Some(r) => (r.content_type, r.body),
            None => ("text/plain; charset=utf-8", b"not found\n".to_vec()),
        }
    };

    let resp = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        rbody.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.write_all(&rbody);
    let _ = stream.flush();
    let _ = id;
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
