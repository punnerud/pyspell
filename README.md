# PySpell

> **Why learn to *spell* Python when PySpell can conjure it for you?**
>
> "Spell" as in a magic spell: say what you want in plain English and a tiny on-device
> language model turns it into runnable Python — **100% locally on a $10 ESP32 with
> 512 kB RAM** (no PSRAM, no cloud, no API key), reachable from anywhere over
> **Tailscale**. The chip serves a web agent IDE, a native **MCP server** and a **REST
> API**, runs the code in a sandbox (with live **web-request / `fetch`** support), and
> drives its own screen + LED — up to **8 parallel** PySpell processes on that same
> half-megabyte of RAM.

**Toward micro-containers for microcontrollers.** Write a small program in **Rust**
or **Python** syntax, compile it to a compact IR on a host, and **push it live to an
ESP32** — reachable over **Tailscale** — where a tiny native evaluator runs it in a
sandbox against live device state. The direction is lightweight, pushable
*"micro-containers"* for tiny chips: drop a bit of code onto a device and run it
safely, remotely.

> **Honest status — not full containers yet.** Today PySpell is a sandboxed
> *expression* evaluator: `let` bindings + one expression in a safe Python/Rust
> **subset** (arithmetic, comparisons, lists, strings, and a fixed set of builtins
> incl. `fetch_json` for live data). No `def`, loops, imports, or arbitrary I/O —
> that restriction *is* the security boundary, and the parser never runs on the
> device (only verified IR crosses). Isolation is language-level (one shared
> sandbox), not OS containers; truly parallel, isolated "containers" need more RAM
> than the ESP32-S3 has (no PSRAM). So: the micro-container *vision*, with a small,
> safe evaluator as the first step.

> PySpell started life as a constraint DSL inside an unrelated routing solver.
> This is a clean, standalone extraction: the domain-specific schema is gone and
> the evaluator is now generic over a host-supplied environment.

## Why

- **Two syntaxes, one IR.** Write `free_heap > 100000 and uptime_s < 60`
  (Python) or `count <= limit && distance < 50000` (Rust). Both lower to the
  same IR (`pyspell-core::ir`).
- **The parser stays on the host.** `syn` and `rustpython-parser` are large; they
  compile source to IR on your laptop. The device only ever sees the IR — that
  is the security boundary (no on-device parsing, no `eval`, no imports, no I/O).
- **Same evaluator everywhere.** `pyspell-core` is `no_std + alloc`, so the exact
  code that runs a program on the host also runs it on the ESP32-S3.
- **Sandboxed.** Deny-by-default grammar (pure expressions + `let` + a fixed set
  of pure builtins) and a per-evaluation instruction budget.

## Layout

```
crates/
  pyspell-core/   no_std+alloc: Value, IR, evaluator, Env trait, postcard wire format
  pyspell-lang/   host-only: Rust (syn) + Python (rustpython) front-ends → IR
  pyspell-cli/    host `pyspell` binary: compile / run / push / repl / ports
firmware/
  esp32s3/        standalone esp-idf firmware: receives IR, evaluates live
```

## Language

A program is `let` bindings followed by a single returned expression (Rust), or a
single expression (Python). It can use:

- literals (int, float, bool, **string**), arithmetic, comparison, boolean
  (`and`/`or`, short-circuiting), `if/else` (ternary), lists, indexing (negative ok),
- builtins: `len, abs, min, max, sum, any, all, round, int, float, bool, str,
  index, before, first, last`, and membership (`x in list`),
- **controlled capabilities** (host-granted, off by default): `fetch(url)` /
  `fetch_json(url, "a.b.0.c")` for an allowlisted HTTPS GET, and `json_get(text,
  "a.b.0.c")` — the only "I/O", and host-gated by an allowlist,
- **free identifiers**, resolved at evaluation time against the host `Env`
  (CLI `--set`, or live device readings like `free_heap`, `uptime_ms`).

Anything else (loops, `def`, attribute access, imports, arbitrary I/O) is rejected
at compile time — that deny-by-default surface is the security boundary.

## Host usage

```sh
# Evaluate locally, binding free variables:
cargo run -p pyspell-cli -- run examples/health.py --set free_heap=120000 --set uptime_ms=45000
# → true

cargo run -p pyspell-cli -- run examples/mem_pct.rs --set total=320000 --set free=80000
# → 75

# Compile to a portable IR blob:
cargo run -p pyspell-cli -- compile examples/health.py        # → examples/health.py.psb

# List serial ports:
cargo run -p pyspell-cli -- ports
```

## ESP32-S3 (T-Dongle S3)

The firmware exposes live device variables to programs: `free_heap`,
`min_free_heap`, `uptime_ms`, `uptime_s`.

```sh
cd firmware/esp32s3
cargo build
espflash flash             # flash without --monitor; the CLI is the client

# Then push code live, no reflashing:
cd ../..
cargo run -p pyspell-cli -- push examples/health.py --port /dev/cu.usbmodem2101 --baud 115200
# or an interactive REPL:
cargo run -p pyspell-cli -- repl --port /dev/cu.usbmodem2101 --lang python
pyspell(py)> free_heap > 100000
true
pyspell(py)> uptime_s
> 42
```

Wire protocol (line-based, robust over the esp-idf USB-Serial-JTAG console):
`host → device: <hex of postcard Program>\n`, `device → host: OK <hex Value>` or
`ERR <message>`.

> The device runs one firmware at a time. Flashing PySpell temporarily replaces
> whatever was on the dongle; reflash the other project to switch back.

## On-device parser

For the "type code in a browser and run it" flow there is also a tiny,
dependency-free parser in `pyspell-core::parse` (both Python and Rust subsets).
Unlike the host front-ends it is `no_std` and a few kB, so it runs on the device
— source still compiles to AST before running, just on-device by a small, safe
parser. Evaluation can take a wall-clock timeout via `eval::run_with` + `Limits`.

## Demo: PySpell over Tailscale

`demo/esp32-tailscale-pyspell/` is a fork of
[tailscale-mpe-rust](https://github.com/punnerud/tailscale-mpe-rust) (upstream
untouched) that adds a PySpell **web text window** + **`/run` API** served inside
the Tailscale tunnel, with a configurable timeout. The layering keeps both
dependencies clean: tailscale-core only gained a generic `RouteFn` hook, PySpell
has no ESP code, and all ESP/web glue lives in the demo (`src/pyspell_web.rs`).
See `demo/esp32-tailscale-pyspell/PYSPELL-DEMO.md`.

Verified live over Tailscale (device `tdongle-s3`): `GET /` serves the page;
`GET /run?lang=py&code=1%2B2*3` → `7`; `lang=rs&code=uptime_ms/1000` → uptime.

## On-device AI coding agent (plain English → live code)

`http://<dongle>/` is a Cursor-like agent: type **"flash the light"**, **"show the
text \"hei og hopp\""**, **"what is 7 plus 5"**, or **"reverse the word robot"** and a
**~0.45 M-parameter language model (< 500 kB, int8)** turns it into PySpell code,
**runs it live on the chip**, and shows the result — or the physical action (the
screen lights up, the RGB LED blinks). Runtime, model, tokenizer and dictionary are
all served **from the dongle, offline** — no cloud, no key (OpenAI is optional, behind
the ⚙).

A model that small only works because of a chain of tricks — **[`tech.md`](tech.md)**
has the full deep-dive; the headlines:

- **The model points, the browser copies.** A 0.45 M model can't reliably copy
  arbitrary tokens (numbers, strings, lists), so it isn't asked to. It emits tiny
  *semantic* directives; the browser copies the literal content verbatim. `calculate
  3 + 2` → `print(`**3 + 2**`)`; `print "hello world"` → `print("`**hello world**`")`;
  `change add to subtract` → `@@ + ==> -`. Quoted text is literal content (copied
  byte-for-byte, excluded from vocab checks).
- **The device serves the model; the browser runs it.** Inference is WebAssembly,
  client-side. The 0.5 MB model image streams **off flash a TCP segment at a time**
  (`BodySource::Flash` + HTTP Range) and is never resident in the chip's ~60 kB heap.
  Inverted edge inference: the constrained device serves + grades, the browser computes.
- **Frozen embeddings distilled from a bigger model.** The 512-token vocab is
  embedded with all-MiniLM (22 M params, via Ollama), PCA'd to 128 dims, folded with a
  part-of-speech vector, and **frozen** — so the tiny model starts with meaningful word
  geometry instead of spending params learning it.
- **The vocabulary is also the dictionary.** Those same 512 tokens + embeddings are
  served back to the browser for input validation ("outside the model's vocabulary…")
  and related-word RAG over the model's own vocab.
- **Instant feedback.** Generated code runs in the 2 s sandbox the moment it appears —
  result inline as `▷`, or the LED/screen fires. No "Apply".

Device builtins the agent targets: `show("…")` (screen), `led(1)`/`led("red")`/`led(0)`,
`flash()` (LED), plus `upper/lower/reverse` and the usual math/list builtins — all
within the sandbox so they auto-run. **Retraining for another language** (translate
the templates, swap the embedding model, re-curate + train) is documented in
[`tech.md`](tech.md#retraining-it--your-own-language-or-task).

### Embeddings & how well it works

The 0.45 M budget is too small to *also* learn what words mean, so it doesn't: the
input embedding table is **distilled from a bigger model and frozen** — every vocab
word is embedded with all-MiniLM (22 M params, via Ollama), PCA'd to 128 dims, folded
with a part-of-speech vector and row-normalized. The transformer only learns to *route*
those fixed meanings. The same 512 tokens + embeddings are then **served back to the
browser as a dictionary** — for input validation ("outside the model's vocabulary…")
and related-word RAG over the model's own vocab. (Full write-up: [`tech.md`](tech.md).)

Measured per category (greedy, model alone; "struct" = right skeleton — literal numbers
and strings are copied by the browser, not the model):

| Category | Struct ok | Compiles |
|---|---|---|
| device (`led`/`show`/`flash`) | 92 % | 95 % |
| arithmetic & math | 92 % | 100 % |
| list ops (`sum`/`max`/`min`/`len`) | 99 % | 100 % |
| string ops (`reverse`/`upper`/`lower`) | 95 % | 100 % |
| multi-line statements (`for`/`def`/`if`) | 68 % | 57 % |

Honest read: it's strong (92–99 %) on the single-line expression families it's *for*,
and weak on multi-line blocks (deliberately not the focus). It's a **narrow-domain**
model, not a general LLM — and the *system* beats the model: the browser copies the
literal content the model can't, and the sandbox verifies every result, so the
practical hit-rate on the target tasks is higher than the raw numbers suggest.

## Sizes (release)

- Standalone PySpell firmware (`firmware/esp32s3`): **460.8 kB** (< 500 kB).
- In the combined demo, **PySpell adds only ~62 kB** over tailscale
  (1585.2 kB vs 1523.3 kB) — measured by building with/without the `pyspell`
  feature. (The "what tailscale already has" is excluded from this count.)

> Tailscale **and** PySpell **and** a browser agent IDE on a chip with **512 kB
> SRAM and no PSRAM** only fit because of a long chain of memory tricks (SPKI
> pinning, streaming the netmap, serving pages from flash, admission control, …).
> They're all collected in **[docs/memory-512kb.md](docs/memory-512kb.md) — every
> memory trick**.

## Status

- Phase 1 — core + front-ends + CLI: **done**, `cargo test` green.
- Phase 2 — ESP32-S3 firmware (live evaluator over USB): **done**, flashed.
- Phase 3 — on-device parser + web/`/run` over Tailscale with timeout: **done**.
- Next — richer device `Env` (ADC/GPIO/RSSI, tailscale peer/packet metrics),
  multi-segment TX for a larger web UI.
