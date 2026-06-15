# ESP32 demo: PySpell over Tailscale

A T-Dongle **ESP32-S3** firmware that joins **Tailscale** and serves a **PySpell**
text window + `/run` API *inside the tunnel*. Open the device's Tailscale IP in a
browser, type a Rust/Python expression, set a timeout, and run it on the device.

This folder is a **fork** of [tailscale-mpe-rust](https://github.com/punnerud/tailscale-mpe-rust)
(upstream untouched) with one added module. The layering the project wants:

```
ESP32 demo (this crate) ── the ONLY ESP-specific layer: display, USB, WiFi glue,
   │                        the web page + /run handler (src/pyspell_web.rs)
   ├── tailscale-core ───── networking (WireGuard, disco, in-tunnel TCP). Clean:
   │                        it only gained a generic `RouteFn` hook, no PySpell.
   └── pyspell-core ─────── parser + sandboxed evaluator + timeout. Clean, no ESP.
```

The integration seam is tiny: `tailscale_core::tcp::TcpServer::with_handler(fn)`
takes a `fn(path, query) -> Option<HttpReply>`. The firmware installs
`pyspell_web::route` (via `new_tcp_server()` in `main.rs`), which parses the
submitted code with `pyspell-core` and evaluates it with the ESP timer driving a
wall-clock timeout. Drop the `pyspell` feature and you get the stock tailscale
firmware back.

## Routes (served on port 80, inside the tunnel)

- `GET /` — minimal page: textarea + language select + timeout + Run button.
- `POST /run?lang=py|rs&timeout=<seconds>` — program in the request body
  (preferred: more room, no URL-encoding). The page uses this.
- `GET /run?lang=py|rs&timeout=<seconds>&code=<urlencoded>` — same, code in the
  query. Both return `text/plain` (the result value, or `error: …`). `timeout`
  is clamped to 1–30 s and enforced as a real wall-clock deadline on the device.

Live variables a program may read: `free_heap`, `min_free_heap`, `uptime_ms`,
`uptime_s`.

## Build & flash

`src/config.rs` (WiFi + Tailscale auth key) is reused from the working tdongles3
checkout and is **gitignored** — never commit it.

```sh
cd demo/esp32-tailscale-pyspell
cargo build --release
espflash flash --release            # restores tailscale AND adds PySpell
```

Then find the device's Tailscale IP (shown on the dongle's screen / in logs) and,
from a machine on the same tailnet:

```sh
open http://100.x.y.z/                       # the text window
curl 'http://100.x.y.z/run?lang=py&timeout=10&code=free_heap%20%3E%20100000'   # → true
```

## Size note

"What PySpell adds" is measured as the app-image delta between a `--features`
build with and without `pyspell` (everything tailscale needs is excluded from the
count). See the repo README / the measurement in the build logs.

## Constraint

The in-tunnel TCP server replies in a **single segment**, so the page is kept
tiny (well under ~1.2 kB). A richer UI would need multi-segment TX added to the
(forked) tailscale-core.
