# Lean-stack roadmap: parallel PySpell on ESP32-S3 (esp32-mpe-core)

Goal: run **3–4 parallel PySpell processes (~50 kB each)** on the ESP32-S3 by
moving off esp-idf onto the pure-Rust esp-rs stack, then adding a small
concurrency coordinator. This is a multi-week/month effort; this doc breaks it
into executable milestones.

## Why (measured, 2026-06-15)

| Stack | WiFi RAM cost | Free heap for the app |
|---|---|---|
| esp-idf (deployed: tailscale + PySpell) | ~200 kB (lwIP + mbedTLS 16k×2 + glue) | **~62 kB** (75 kB after the conservative tuning pass) |
| esp-rs lean PoC (esp-hal + esp-wifi, bare) | **~43 kB** | **~250 kB** (320 DRAM − 15 static − 43 WiFi) |

The lean stack frees ~190 kB → enough for ~4–5 concurrent fetch-processes
(a TLS session ≈ 40 kB dominates) or many compute-only ones. esp-idf tuning only
buys ~13 kB (the big mbedTLS-IN cut breaks TLS), so it is a dead end for this
goal. Flash is ~400 kB either way (the WiFi PHY/MAC blob dominates and cannot be
rewritten — esp-wifi wraps it).

## The #1 risk: esp-rs version matrix

Building the PoC hit ~9 version/feature conflicts (e.g. esp-hal 1.0.0 needs
`xtensa-lx-rt` 0.21 but esp-wifi 0.15.x needs 0.20 — a `links` conflict; espflash
4.4 demands an app descriptor esp-hal 0.23 doesn't emit). **Pin one coherent set
up front** — generate it with `esp-generate` (the official template picks a
known-good matrix) rather than hand-picking versions. Treat version bumps as
deliberate, tested events. The build needs the xtensa GCC on PATH:
`export PATH="$HOME/.rustup/toolchains/esp/xtensa-esp-elf/<ver>/xtensa-esp-elf/bin:$PATH"`.
esp-hal 0.23 images flash with a locally-built `espflash` 3.x; esp-hal 1.0 images
embed `esp-bootloader-esp-idf::esp_app_desc!()` and flash with espflash 4.x.

## Step 3 — the lean port (the marathon)

`pyspell-core` is already `no_std + alloc` and ports as-is. The work is the
*platform* layer (today esp-idf-svc/lwIP/mbedTLS), re-built on esp-rs:

- **M3.1 — Lean fetch PoC.** esp-hal + esp-wifi + **smoltcp** (TCP/IP) +
  **embedded-tls** (TLS): connect WiFi, HTTPS GET yr.no, run `pyspell-core`
  `fetch_json` over it. Validates the trio + measures real free heap with a live
  TLS session. (Biggest unknown: embedded-tls interop with the met.no CDN.)
- **M3.2 — Tailscale control plane on smoltcp.** Port `tailscale-core`'s socket
  use (currently esp-idf-svc UDP/TCP) to smoltcp; the noise/WireGuard crypto is
  already pure Rust and `no_std`. Do the `/key` + `/machine/register` + netmap
  over embedded-tls.
- **M3.3 — Data plane.** WireGuard UDP via smoltcp + the existing disco/DERP
  logic; the in-tunnel TCP server (already in `tailscale-core::tcp`) runs on
  smoltcp.
- **M3.4 — PySpell over the lean tunnel.** Re-point the demo's `net.rs`
  (fetch) and `display.rs` and the `/run` handler at the lean platform. PySpell
  language code is unchanged.

Each milestone is independently flashable and measurable.

## Step 2 — the concurrency coordinator — DONE (2026-06-15, proven on hardware)

Built on the lean stack as **embassy async** (esp-rtos has no public user threads, so
cooperative async is the path to parallelism; PySpell programs stay synchronous —
`pyspell-core` gained an async eval path `run_async`/`AsyncNet`, the language is
unchanged). `esp32-mpe-fetch` now: embassy-net (esp-radio's `Interface` is an
`embassy-net-driver`) + async embedded-tls; `AsyncLeanNet` impls `AsyncNet`. A POST
`/run` server runs **4 worker tasks** (`#[task(pool_size=4)]`, `TcpSocket::accept(8080)`);
each request runs a PySpell job. **Admission** = a `GreedySemaphore` capping concurrent
TLS sessions (each leases a ~16 kB read buffer), so memory stays bounded.

**Measured:** 4 jobs concurrently → 1815 ms wall (vs serial sum 5533 ms); **5 concurrent
POSTs from a laptop → all `Float(18.0)`, 1.12 s total** (vs ~5 s serial). 192 kB heap held
4 concurrent TLS, ~149 kB free after. **≥4 parallel PySpell answering POST = achieved.**

**TLS certificate verification — DONE (S2.5).** A `PinnedProvider` (impl `CryptoProvider`)
returns `embedded_tls::pki::CertVerifier` (feature `rustpki`+`rsa`) anchored on the pinned
**HARICA RootCA 2015** DER (`include_bytes!`), with a fixed `TlsClock`. Real RSA chain
validation against the pin — proven enforced by a negative test (corrupt anchor → handshake
`DecodeError`; correct anchor → verified `18.x`). RSA-2048 verify ×chain makes the handshake
~1.7 s (vs ~0.8 s unverified) and is memory-heavy, so admission is capped at **2 concurrent
TLS** (3 OOM'd the 192 kB heap); the other jobs wait for a permit. **Step 2 is complete:
≥4 parallel PySpell over POST + verified TLS.**

### Original step-2 design notes (now implemented above)

Even with RAM, today's server is single-connection and eval is synchronous. Add:

- **Multi-connection in-tunnel server:** accept N concurrent in-tunnel TCP
  connections (smoltcp sockets) instead of one — or a small accept queue.
- **Job coordinator:** each `/run` becomes a job spawned on its own task
  (FreeRTOS via esp-hal, or an Embassy async task). Per-job: a **memory budget**
  (~50 kB), **admission control** (start only if `free_heap ≥ budget + margin`,
  else queue/503), and the existing step + wall-clock budgets.
- **Memory accounting, not isolation:** no MMU on ESP32, so this is cgroup-style
  accounting (a counting allocator per job), not Firecracker-grade isolation.

Target: 3–4 concurrent fetch-jobs or many compute-jobs, each capped at ~50 kB.

## Suggested order

1. **M3.1 lean fetch PoC** — proves the trio + the real free-heap number with TLS.
2. **Step 2 coordinator** prototyped against M3.1 (concurrency + admission) so the
   parallel machinery exists early and is testable.
3. **M3.2 → M3.4** — port tailscale onto the lean platform.
4. Spin up doc-only agents (the existing harness) to load-test N parallel jobs.

## Status

- esp-idf tuning (step 1): **done** — +13 kB (62→75 kB), connectivity intact;
  confirmed marginal.
- Lean baseline PoC: **done** — `esp32-mpe-core/`, WiFi costs ~43 kB, ~250 kB free.
- **M3.1 lean fetch PoC: DONE (2026-06-15)** — `esp32-mpe-fetch/`, on hardware.

### M3.1 result (the trio works, fully synchronous)

`esp-hal 1.1 + esp-rtos 0.3 + esp-radio 0.18 + smoltcp 0.13 + embedded-tls 0.19`
(pinned by `esp-generate 1.3`). **No embassy/async** except `block_on` around
esp-radio's `connect_async` (the one async API). A hand-written `smoltcp::phy::Device`
wraps esp-radio's Wi-Fi `Interface` (esp-radio only ships an embassy-net-driver). The
device runs `pyspell-core`'s `fetch_json(url, path)` unchanged: DNS → TCP →
embedded-tls TLS 1.3 (blocking) → HTTP/1.0 GET → header strip → streaming probe with
early abort → `json::get`. **Live: Oslo air_temperature from api.met.no over the lean
TLS stack.** (TLS cert verification is still off — `UnsecureProvider`; a follow-up.)

### M3.1 memory budget (measured; 192 kB heap reservation)

| Point | Free heap | Cost |
|---|---|---|
| Boot | 196 608 B | — |
| After WiFi + smoltcp (DHCP/DNS) | ~147 916 B | WiFi+stack ≈ 49 kB |
| **TLS fetch LIVE peak** | **121 404 B** | one fetch ≈ **26.5 kB** over baseline |
| After fetch (early-abort + close) | 140 124 B | TLS freed |

Per-fetch breakdown: TLS **read_buf 16 640** (mandatory) + write_buf 2 048 + TCP
rx/tx 4 096/2 048 + handshake/body state ~2 kB. **Key finding:** the met.no CDN sends
application records up to the TLS-1.3 max (~16 kB) and **ignores RFC 6066
`max_fragment_length`**, so the 16.6 kB read buffer **cannot be shrunk per-fetch**
(tested: 8 kB → `InsufficientSpace` on read; 6 kB + MFL → `InvalidCertificate`).

### What this means for step 2 (≥4 parallel)

- **Private per-fetch ≈ 10 kB** (write + TCP + state) — within the 15–20 kB target.
- **The 16.6 kB TLS read buffer is the shared, admission-gated resource** → a small
  **pool of K read buffers** leased per active TLS read, NOT one per job. Compute-only
  jobs are <5 kB. With the lean stack's ~250 kB free, 4 fetches (4×10 kB private +
  4×16.6 kB pooled = ~106 kB) fit comfortably; queue jobs when the pool is exhausted.

- M3.2 → M3.4 (port tailscale onto the lean platform): not started.
