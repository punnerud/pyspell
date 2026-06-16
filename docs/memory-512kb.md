# How we fit it into 512 kB SRAM (no PSRAM)

The ESP32-S3 T-Dongle has **512 kB of SRAM and no PSRAM**. On that budget it runs a
full **Tailscale** node (control plane *and* DERP data plane), the **PySpell**
evaluator, a browser **agent IDE** served straight off the chip, a native **MCP
server**, and **multi-segment TCP** serving — all over a real **TLS 1.3** link to
`api.met.no`.

None of that fits by accident. This doc collects every memory-saving trick we used,
why it works, and where it lives in the tree. The techniques are spread across two
firmwares — the lean pure-Rust stack (`esp32-mpe-fetch/`) and the deployed esp-idf
demo (`demo/esp32-tailscale-pyspell/`) — so this is the one place they're written
down together.

## The honest budget

Of the 512 kB SRAM, ~320 kB is usable DRAM after the ROM/cache reservations. The
network stack eats most of the rest:

| Stack | WiFi/net RAM cost | Free heap for the app |
|---|---|---|
| esp-idf (deployed: tailscale + PySpell) | ~200 kB (lwIP + mbedTLS 16k×2 + glue) | **~62 kB** (75 kB after tuning) |
| esp-rs lean (esp-hal + esp-wifi, bare) | **~43 kB** | **~250 kB** |

**Read the headline number with suspicion.** The demo's "~260 kB free" is a
*calm-moment* reading taken between requests. The number that actually matters is the
**worst-case peak free heap, which is ~60 kB** — measured during a TLS fetch while
the Tailscale control session is live. Every trick below exists to keep transient
allocation spikes under that ~60 kB ceiling.

The blunt consequence: an **8-way parallel PySpell pool and full Tailscale (online +
DERP) do not coexist** on the esp-idf stack. They contend for the same ~16 lwIP
sockets and the same ~60 kB peak. We chose Tailscale-stays-green over the parallel
pool, and disabled the pool on the demo. Cheap parallelism is a property of the lean
stack (see trick F), which is why the roadmap moves there.

---

## A. Crypto & TLS — cut the biggest transient peaks

### 1. SPKI leaf-key pinning instead of CA-chain validation
A normal TLS verify buffers the whole certificate chain (~6 kB) and runs 2–3 RSA
verifies up the chain. We pin the **leaf key's SPKI** instead: parse only the leaf,
SHA-256 its SubjectPublicKeyInfo, compare against a compiled-in hash, and do **one**
RSA-PSS-SHA256 verify. That drops the chain buffer and the extra verifies — a TLS
fetch falls from ~45 kB to ~30 kB.

- `esp32-mpe-fetch/src/pinning.rs` — `PinnedVerifier` (impl of `embedded_tls::TlsVerifier`),
  `SpkiPin`.
- `esp32-mpe-fetch/src/fetch_async.rs:91,141` — the pin constant `MET_NO_SPKI_PIN`
  and where the verifier is wired in.

The pin is a public key fingerprint, not a secret — safe to commit.

### 2. Admission control / heap gate
Concurrency is bounded so peak heap is `K × per-fetch`, never `N × per-fetch`. A job
spins waiting for headroom before it allocates its TLS context, holding only its URL
while it waits:

```rust
while esp_alloc::HEAP.free() < HEAP_ADMISSION_MIN { /* yield */ }
```

- `esp32-mpe-fetch/src/fetch_async.rs:52,56,226` — `TLS_MAX_CONCURRENT` (currently 1),
  `HEAP_ADMISSION_MIN` (58 kB), and the gate loop.
- Demo analog: `FETCH_MAX` in `demo/esp32-tailscale-pyspell/src/net.rs:34` — esp-idf's
  mbedTLS fails outright at 2+ concurrent sessions, so the gate is also a correctness
  requirement there, not just a memory one.

---

## B. Streaming instead of buffering — never hold the whole netmap or body

The Tailscale netmap (DERPMap + peers) is tens of kB. Buffering it would blow the
budget on its own, so nothing is ever fully materialized.

### 3. Streaming `fetch_peers` with `serde_json::from_reader`
We expose the HTTP/2 DATA frames as an `io::Read` adapter (`H2Body`) and hand it to
`serde_json::from_reader`. serde walks the JSON field-by-field and **skips** the huge
DERPMap field without ever buffering it. The netmap transient drops from ~60 kB to
roughly one 4 kB chunk.

- `demo/esp32-tailscale-pyspell/src/control/mod.rs:99,111` — `fetch_peers`, `H2Body`.
- `demo/esp32-tailscale-pyspell/core/src/h2.rs:167` — `read_response_chunk`
  (frame-by-frame, the engine behind `from_reader`).

### 4. `OmitPeers:true` + a bounded sliding window
We ask the control server to omit peers we don't need, and the accumulator that does
need to scan keeps only a bounded sliding window:

```rust
if acc.len() > 12288 { acc.drain(..drop); }
```

- `demo/esp32-tailscale-pyspell/src/control/mod.rs` — `build_map_json`, `OmitPeers` at `:362`.

### 5. Early-abort streaming probes
`fetch_extract` / `stream()` stop reading the moment the wanted value is found —
before the TLS close-notify. This avoids buffering the rest of the body *and* sidesteps
embedded-tls' `ConnectionClosed` error on an unread tail.

- `demo/esp32-tailscale-pyspell/src/net.rs:74,81` — `DeviceNet::fetch_extract`, `stream()`.

### 6. Manual byte-scan instead of serde Value trees
For the few fields we read out of a response, we scan raw bytes rather than building a
JSON DOM:

```rust
contains(&resp, b"\"MachineAuthorized\":true")   // is the node authorized?
scan_tailscale_ip(&resp)                          // find the "100." address
```

A `serde_json::Value` tree of the netmap would cost far more than the answer is worth.

- `esp32-mpe-fetch/src/ts_control.rs:83,281` — `contains`, the `MachineAuthorized` check.
- `demo/esp32-tailscale-pyspell/src/control/mod.rs:327` — `scan_tailscale_ip`.

---

## C. Serve big pages from flash, not RAM

### 7. `&'static str` content + multi-segment TCP
The agent IDE page (~4.3 kB) and other static content live in **flash** as
`&'static str` — zero heap. To send something larger than one TCP segment without
buffering the whole response, `handle()` splits it into 512-byte segments and only the
current segment is ever in RAM; the FIN rides the last one:

- `demo/esp32-tailscale-pyspell/core/src/tcp.rs:148-167` — `handle()`, `MSS = 512`.
- The page itself: `demo/esp32-tailscale-pyspell/src/pyspell_web.rs`.

### 8. Retransmit-by-re-reading-flash (design note — "Nivå B")
The current multi-segment path is **"Nivå A": no retransmit** — it assumes segments
arrive. The robust low-RAM upgrade is to lean on the fact that a flash-backed page is
**immutable**: on a retransmit you simply re-read the same flash slice, so there's no
RAM buffer for unacked segments. This is documented here as the planned direction, not
yet implemented.

---

## D. Allocator & buffer discipline

### 9. Heap-vs-stack is a single DRAM tradeoff
`esp_alloc::heap_allocator!(size: N)` is just a static `.bss` array; the task stack is
the adjacent linker region. So **+16 kB of heap = −16 kB of stack** — they come from
the same pool. Because the TLS crypto runs on the executor stack, the heap size had to
be tuned by hand (192 → 128 → 160 kB) to leave enough stack for the handshake without
starving the heap.

- Live stack headroom probes: `esp32-mpe-fetch/src/lib.rs:29,37` — `stack_free_now`,
  `stack_total`, reading the `_stack_start_cpu0` / `_stack_end_cpu0` linker symbols.

### 10. `shrink_to_fit` only on an *empty* buffer (an anti-pattern we hit)
Shrinking a buffer reallocates. Doing it on a partially-full buffer mid-drain copies
the live bytes into a new smaller allocation — a transient spike that caused OOM under
pressure. The fix: only `shrink_to_fit` when the buffer is already empty, where the
realloc is free.

- `demo/esp32-tailscale-pyspell/core/src/h2.rs:603` — guarded by the empty check.

---

## E. Socket lifetime — keep lwIP's ~16 sockets free

### 11. `SO_LINGER = 0` (RST on close)
Burst connections that close normally pile up in TIME_WAIT and hold lwIP sockets for
seconds. We set `SO_LINGER = 0` so close sends an RST and frees the socket immediately
— important when the socket pool is small and Tailscale needs its share.

- `demo/esp32-tailscale-pyspell/src/local_server.rs:166` — `set_linger_zero` via
  `lwip_setsockopt` (the symbol isn't in the high-level bindings).

### 12. Raised socket ceiling + reservation for Tailscale
The lwIP socket count is bumped from the default 10 to 16, and sockets are reserved
for Tailscale so a burst of PySpell requests can't starve the DERP data plane.

- `demo/esp32-tailscale-pyspell/sdkconfig.defaults:57` — `CONFIG_LWIP_MAX_SOCKETS=16`.

---

## F. Concurrency model — why parallelism belongs on the lean stack

### 13. Cooperative shared stack vs per-thread stacks
On the lean stack (embassy executor) all tasks **share one stack**, so spawning many
parallel jobs is nearly free. On the esp-idf demo each worker is an OS thread with its
**own 12–32 kB stack**, so even a handful of parallel jobs costs more RAM than the
~60 kB peak allows. That is the concrete reason the demo's parallel pool was disabled
(it broke Tailscale staying green) and why scalable parallelism is a lean-stack
feature. See `docs/lean-stack-roadmap.md`.

---

## Tiny telemetry that doesn't cost RAM

To tune any of the above you have to *see* the peaks. We track rolling job rates as a
`Mutex<VecDeque<i64>>` of timestamps with windowed counts (10 s / 60 s / 10 min /
60 min) — a few bytes per recent job, no per-request allocation:

- `demo/esp32-tailscale-pyspell/src/jobcount.rs` — `record`, `counts`.

---

## Net result

- Standalone PySpell firmware: **460.8 kB** flash (under the 500 kB target).
- In the combined demo, PySpell adds only **~62 kB** over Tailscale.
- Worst-case peak free heap: **~60 kB** — and the tricks above are precisely what keep
  every transient under it.
- The trade that defines the build: **Tailscale stays green; the 8-way parallel pool
  does not run alongside it.** Cheap parallelism waits for the lean stack.

See also: `docs/lean-stack-roadmap.md` (the move to the lean esp-rs stack) and
`docs/design.md` (the host-parses / device-evaluates security boundary).
