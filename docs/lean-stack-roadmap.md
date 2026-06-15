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

## Step 2 — the concurrency coordinator (build on the lean stack)

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
- M3.1 onward: not started (the marathon).
