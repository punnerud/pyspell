# PySpell

A small, **sandboxed expression language** you write in **Rust** or **Python**
syntax, compile to a compact AST/IR on a host, and **push live to an ESP32** —
a little like MicroPython, but the parser never runs on the device. Only verified
IR crosses to the microcontroller, where a tiny native evaluator runs it against
live device state.

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

- literals (int, float, bool), arithmetic, comparison, boolean (`and`/`or`,
  short-circuiting), `if/else` (ternary), lists, indexing (negative ok),
- builtins: `len, abs, min, max, sum, any, all, round, int, float, bool,
  index, before, first, last`, and membership (`x in list` / `.contains`),
- **free identifiers**, resolved at evaluation time against the host `Env`
  (CLI `--set`, or live device readings like `free_heap`, `uptime_ms`).

Anything else (loops, functions, attribute access, imports, strings, I/O) is
rejected at compile time.

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

## Sizes (release)

- Standalone PySpell firmware (`firmware/esp32s3`): **460.8 kB** (< 500 kB).
- In the combined demo, **PySpell adds only ~62 kB** over tailscale
  (1585.2 kB vs 1523.3 kB) — measured by building with/without the `pyspell`
  feature. (The "what tailscale already has" is excluded from this count.)

## Status

- Phase 1 — core + front-ends + CLI: **done**, `cargo test` green.
- Phase 2 — ESP32-S3 firmware (live evaluator over USB): **done**, flashed.
- Phase 3 — on-device parser + web/`/run` over Tailscale with timeout: **done**.
- Next — richer device `Env` (ADC/GPIO/RSSI, tailscale peer/packet metrics),
  multi-segment TX for a larger web UI.
