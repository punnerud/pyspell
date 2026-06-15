# PySpell design

## Goal

A sandboxed expression language with two surface syntaxes (Rust and Python),
compiled to a portable AST/IR on a host and evaluated natively — on the host or
on a microcontroller. Think "MicroPython, but the parser stays off the device,
and you can write the same program in Rust syntax too."

## Three layers

```
Rust source  ─┐                                  ┌─ host: cargo run pyspell run …
              ├─►  pyspell-lang  ─►  IR (Program) ┤
Python source ┘   (syn / rustpython)             └─ device: hex line over USB → eval
                                       │
                                       ▼
                            pyspell-core::eval::run(program, env)
```

1. **Front-ends — `pyspell-lang` (host only).** `syn` parses the Rust subset;
   `rustpython-parser` parses the Python subset. Both lower a *whitelisted*
   subset to the shared IR (deny-by-default: any node outside the subset is a
   compile error, never a panic). These parsers are large and are the reason the
   device never sees source.

2. **IR + value model — `pyspell-core::ir` / `::value`.** `Program` = ordered
   `let` bindings + a return `Expr`. `Value` = `Int | Float | Bool | List`.
   Everything derives `serde`, so a `Program` serializes to a compact `postcard`
   blob (`pyspell-core::wire`) — the unit shipped to the device.

3. **Evaluator — `pyspell-core::eval` (no_std + alloc).** A tree-walk with a
   per-call instruction budget (runaway guard). Free identifiers resolve against
   a host-supplied `Env` (`pyspell-core::env`); that trait is the *only* bridge
   to outside data — there is no I/O, import, or attribute access in the grammar.

## The generalization from the original

The code was extracted from a VRP solver where the IR had a fixed schema
(`route.travel_time`, `vehicle.capacity`, `solution.*`, …) and the evaluator read
those fields straight off solver structs. That coupling is removed:

- `Field`/`SolutionField`/`ArcField`/`BrokerField`/`ListField` enums → deleted.
- A free identifier lowers to `Expr::Var(name)` instead of a schema field.
- At eval time `Expr::Var(name)` calls `Env::get(name)`. The host decides what
  names exist: CLI `--set k=v` on the host, live readings on the device.
- The result is a plain `Value` (not a solver `Verdict`); the caller interprets
  it (bool predicate, numeric score, list, …).

This keeps the whole evaluator (arithmetic, comparison, short-circuit booleans,
indexing, the builtin set) byte-for-byte reusable while making it domain-neutral.

## Portability split (mirrors tailscale-core/tailscale-rust)

- `pyspell-core` is `no_std + alloc`. The `std` feature only adds
  `impl std::error::Error` for host ergonomics; the firmware turns it off.
- `firmware/esp32s3` is its own cargo workspace (target `xtensa-esp32s3-espidf`,
  toolchain `esp`). It depends on `pyspell-core` by path with default features
  off, and implements the device `Env` (free heap, uptime, …).

## Wire + live update

- `wire::to_bytes` / `from_bytes` — postcard (de)serialization of a `Program`.
- Device link is a line protocol so it survives the esp-idf console cleanly:
  - host → device: `<hex of postcard Program>\n`
  - device → host: `OK <hex of postcard Value>` or `ERR <message>`
  - the host skips any other line (boot logs, `READY …`).
- `pyspell repl` compiles each typed line and pushes it — no reflashing, the
  MicroPython-like loop.

## Security model

- Deny-by-default grammar: only the whitelisted expression nodes and the fixed
  pure-builtin set exist. No loops, recursion, functions, attribute escape,
  imports, strings, or I/O.
- Per-evaluation instruction budget (`Program::max_steps`) bounds runaway work.
- The device additionally caps the request line length before allocating.
- Because parsing happens on the host, the device's attack surface is just the
  postcard decoder + the bounded evaluator.

## Non-goals (for now)

- Loops / functions / mutable state (would need IR + evaluator extensions).
- On-device parsing (deliberately excluded — that is the security boundary).
- A stable on-disk `.psb` format across versions (IR is versioned implicitly by
  the crate; treat blobs as build artifacts).

## Roadmap

1. Richer device `Env`: ADC, GPIO, temperature, RSSI.
2. When co-resident with a networking firmware, expose peer/packet metrics so a
   pushed spell can react to live traffic.
3. Live transport over WiFi / a Tailscale tunnel instead of USB-serial.
4. Optionally a flat bytecode (`Vec<Op>`) compile target for the hottest paths.
