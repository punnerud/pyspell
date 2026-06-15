# esp32-mpe-fetch — lean parallel PySpell (Step 2 coordinator)

Runs **≥4 PySpell jobs in parallel** on one ESP32-S3, answering HTTPS-backed POST
requests, on the pure-Rust esp-rs stack (off esp-idf) — with verified TLS. This is
Step 2 of the [lean-stack roadmap](../docs/lean-stack-roadmap.md); M3.1 (single
synchronous fetch) proved the trio first.

## Architecture

`esp-hal 1.1` + `esp-rtos` (embassy executor) + `esp-radio` (Wi-Fi) +
`embassy-net` (TCP/IP — esp-radio's `Interface` *is* an `embassy-net-driver`) +
async `embedded-tls`. `pyspell-core` links unchanged.

PySpell programs stay **synchronous** (`fetch_json("https://…","a.b.0.c")`, no async
keywords). Concurrency is internal: `pyspell-core` gained an async eval path
(`run_async` + `AsyncNet`) so the one blocking effect — the network fetch — `.await`s,
letting many jobs overlap their network waits on one cooperative executor. (esp-rtos
exposes no user threads, so async is the path to parallelism.)

- `src/fetch_async.rs` — `AsyncLeanNet` (impl `AsyncNet`): DNS → TCP → async TLS →
  HTTP/1.0 GET → header strip → streaming probe with early abort. TLS is verified by
  a `PinnedProvider` using `embedded_tls::pki::CertVerifier` against a pinned root.
- `src/bin/main.rs` — embassy `main`; brings up Wi-Fi + embassy-net; a boot self-test
  runs 4 jobs concurrently (`join4`); a POST `/run` server (4 worker tasks on :8080)
  runs a job per connection. A `GreedySemaphore` caps concurrent TLS sessions
  (admission) so memory stays bounded.

## TLS verification

The met.no chain is RSA (`*.api.met.no` ← GEANT TLS RSA 1 ← HARICA TLS RSA Root CA
2021 ← **HARICA RootCA 2015**). We pin the root that signs the topmost cert the server
sends — **HARICA RootCA 2015** — embedded as `src/harica_rootca_2015.der`
(`include_bytes!`). It's a public CA cert (safe to commit), obtained from the macOS
trust store and verified to sign the chain:

```sh
security find-certificate -c "Hellenic Academic and Research Institutions RootCA 2015" \
  -p /System/Library/Keychains/SystemRootCertificates.keychain | openssl x509 -outform DER \
  -out src/harica_rootca_2015.der
```

Enforcement is proven by a negative test: corrupting the pinned anchor makes every
handshake fail (`DecodeError`); the correct anchor verifies and returns the temperature.
No RTC on the board, so cert-validity dates are checked against a fixed `TlsClock::now()`
(update it if it drifts past the leaf's not-after).

## Build & flash

```sh
export PATH="$HOME/.rustup/toolchains/esp/xtensa-esp-elf/esp-15.2.0_20250920/xtensa-esp-elf/bin:$PATH"
cp src/config.rs.example src/config.rs   # then edit WIFI_SSID / WIFI_PASS (gitignored)
cargo build --release
espflash flash --chip esp32s3 --port /dev/cu.usbmodem2101 \
  --before usb-reset --after hard-reset \
  target/xtensa-esp32s3-none-elf/release/esp32-mpe-fetch
espflash reset --port /dev/cu.usbmodem2101 --chip esp32s3 && cat /dev/cu.usbmodem2101
```

Test the POST server from a machine on the same LAN:

```sh
for i in 1 2 3 4 5; do curl -s -X POST http://<device-ip>:8080/run -d x & done; wait
```

## Measured (ESP32-S3, 192 kB heap)

- **Unverified TLS:** 4 jobs concurrently → 1815 ms wall (vs serial sum 5533 ms);
  5 concurrent POSTs → 1.12 s total. 4 concurrent TLS fit.
- **Verified TLS (RSA):** handshake ~1.7 s each (RSA-2048 ×chain on xtensa); 3
  concurrent verified handshakes OOM the 192 kB heap, so admission caps concurrent TLS
  at **2** — the other jobs wait for a permit (admission control in action). All 4 jobs
  still complete and verify.

## Not yet

No Tailscale on the lean stack yet — that's M3.2–M3.4 (port tailscale-core onto
smoltcp/embassy-net + embedded-tls; the control plane is de-risked by working
embedded-tls, the WireGuard data plane is the new work).
