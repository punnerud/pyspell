#![no_std]

extern crate alloc;

/// WiFi credentials (gitignored — copy config.rs.example → config.rs).
pub mod config;
/// `AsyncLeanNet` — device `pyspell_core::AsyncNet` over embassy-net + async TLS.
pub mod fetch_async;
/// SPKI leaf-key pinning `TlsVerifier` (low-memory verified fetches).
pub mod pinning;
/// Custom getrandom backends (esp-hal RNG) for tailscale-core's crypto.
pub mod rng_backend;
/// ST7735 status display + rolling PySpell job counters.
pub mod screen;
/// M3.2: tailscale control-plane registration over the lean async stack.
pub mod ts_control;

// --- stack headroom probe (safe: reads addresses only, never writes) --------
// The main/executor stack is the linker region [_stack_end_cpu0, _stack_start_cpu0]
// and grows DOWN. `stack_free_now()` = current SP − bottom = headroom to overflow
// at the call site; sampled at the deepest points it bounds how much we can move
// from stack into the heap.
unsafe extern "C" {
    static _stack_start_cpu0: u32;
    static _stack_end_cpu0: u32;
}

#[inline(always)]
pub fn stack_free_now() -> usize {
    let probe = 0u8;
    let sp = (&probe as *const u8) as usize;
    let bottom = (&raw const _stack_end_cpu0) as usize;
    sp.saturating_sub(bottom)
}

#[inline(always)]
pub fn stack_total() -> usize {
    let top = (&raw const _stack_start_cpu0) as usize;
    let bottom = (&raw const _stack_end_cpu0) as usize;
    top.saturating_sub(bottom)
}

// The synchronous M3.1 stack (`net.rs` raw-smoltcp `LeanStack` + `fetch.rs`
// blocking `LeanNet`) is superseded by the embassy/async path for Step 2's
// concurrency. The files are kept for reference but no longer compiled.
