//! On-device LLM inference (feasibility spike): run the tiny model on the ESP32 itself
//! instead of in the browser (WASM). Same `tinyllm` engine; the difference is WHERE it
//! runs and how memory is managed on a chip with only ~60 kB of (fragmented) free heap:
//!
//! * **Weights stay in flash** — `model_host::model_mmap()` maps the 489 kB `model.bin`
//!   into the address space; pages fault in through the flash cache on every forward, so
//!   the model is **never resident in the heap** (effectively streamed repeatedly).
//! * **int8 KV cache, bounded to `MAX_CTX`** — the f32 cache (256 kB) won't fit; the
//!   int8 + short-context cache keeps the whole working set ~25 kB.
//! * **Allocated up front, on the calling thread** — the RunState is built in [`start`]
//!   (where a large contiguous block still exists) with `try_reserve` (fallible — a tight
//!   heap returns a clean error, never an OOM-abort that reboots the chip), then moved
//!   into the worker. The worker does no big allocations.
//! * **Runs off the network thread** — a small background worker runs the (~tens of
//!   seconds) forward loop and yields each step, so the data plane keeps Tailscale online
//!   and `/run` serving in parallel. `POST /generate` starts it; `GET /generate` polls.

use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

use tinyllm::tokenizer::{BOS, EOS};
use tinyllm::{RunState, Sampler, Tokenizer, Transformer};

/// Bounded context for the int8 KV cache (tokens). The 512-vocab BPE is granular, so
/// prompts run 17–24 tokens; 32 leaves room for a short code output on top while the
/// cache stays ~16 kB (key_q/val_q 8 kB each). Larger would generate more but won't fit
/// alongside Tailscale on ~60 kB of (fragmented) heap.
const MAX_CTX: usize = 32;
/// Worker-thread stack (heap-backed on esp-idf). `forward` is iterative, no big arrays.
const WORKER_STACK: usize = 8 * 1024;
/// Generated tokens cap (beyond the prompt).
const MAX_NEW: usize = 24;
/// Refuse to start unless at least this much heap is free (plus a big-enough contiguous
/// block, checked in `start`) — so a tight moment returns cleanly instead of OOM-ing.
const MIN_FREE_HEAP: u32 = 50 * 1024;

struct GenState {
    running: bool,
    result: Option<String>,
}
static STATE: Mutex<GenState> = Mutex::new(GenState { running: false, result: None });

/// Sender to the single persistent worker thread. The worker's 8 kB stack is allocated
/// ONCE (when the heap is least fragmented) and reused for every job — spawning a fresh
/// thread per request fails after a few runs because the tight heap fragments and can't
/// find an 8 kB contiguous block.
static WORKER: OnceLock<Mutex<Sender<String>>> = OnceLock::new();

fn worker() -> &'static Mutex<Sender<String>> {
    WORKER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let _ = std::thread::Builder::new()
            .stack_size(WORKER_STACK)
            .spawn(move || {
                for prompt in rx {
                    // recv blocks until a job; reuses this thread + stack forever.
                    let out = match prepare(&prompt) {
                        Ok(p) => run_loop(p),
                        Err(e) => e,
                    };
                    let mut g = STATE.lock().unwrap();
                    g.result = Some(out);
                    g.running = false;
                }
            });
        Mutex::new(tx)
    })
}

/// Everything the worker needs, allocated up front on the caller's thread.
struct Prepared {
    model: &'static [u8],
    tok: Tokenizer,
    state: RunState,
    prompt_tokens: Vec<usize>,
    vocab: usize,
}

fn free_heap() -> u32 {
    unsafe { esp_idf_svc::sys::esp_get_free_heap_size() }
}
fn largest_block() -> u32 {
    let b = unsafe {
        esp_idf_svc::sys::heap_caps_get_largest_free_block(esp_idf_svc::sys::MALLOC_CAP_8BIT)
    };
    b as u32
}

/// Build the model context + RunState on the CALLING thread (fallible). Returns an error
/// string on no-image / parse / low-memory — never panics.
fn prepare(prompt: &str) -> Result<Prepared, String> {
    let model = crate::model_host::model_mmap().ok_or("error: no model image flashed")?;
    let tokb = crate::model_host::tokenizer_bytes().ok_or("error: no tokenizer")?;
    let cfg = {
        let t = Transformer::new(model).map_err(|e| format!("error: model parse {e}"))?;
        *t.config()
    };
    let tok =
        Tokenizer::from_bytes(&tokb, cfg.vocab_size).map_err(|_| "error: tokenizer parse")?;
    let prompt_tokens = tok.encode(prompt, true, false); // BOS, no EOS
    if prompt_tokens.is_empty() {
        return Err("error: empty prompt".into());
    }
    // The big allocation — fallible, so a fragmented/low heap is a clean error.
    let state = RunState::try_new_int8(&cfg, MAX_CTX).ok_or_else(|| {
        format!(
            "low memory: can't allocate the int8 KV cache (free {} B, largest block {} B) — try again when idle",
            free_heap(),
            largest_block()
        )
    })?;
    Ok(Prepared { model, tok, state, prompt_tokens, vocab: cfg.vocab_size })
}

/// Run the forward loop over a pre-built [`Prepared`] (no big allocations here). Greedy.
fn run_loop(mut p: Prepared) -> String {
    let t = match Transformer::new(p.model) {
        Ok(t) => t,
        Err(e) => return format!("error: model parse {e}"),
    };
    let mut sampler = Sampler::new(p.vocab, 0.0, 0.9, 1); // temperature 0 = greedy
    let max_steps = (p.prompt_tokens.len() + MAX_NEW).min(MAX_CTX);
    let mut out = String::new();
    let mut token = p.prompt_tokens[0];
    let mut pos = 0usize;
    let t0 = unsafe { esp_idf_svc::sys::esp_timer_get_time() };
    while pos < max_steps {
        t.forward(&mut p.state, token, pos);
        let next = if pos + 1 < p.prompt_tokens.len() {
            p.prompt_tokens[pos + 1] // still feeding the prompt
        } else {
            let mut logits = p.state.logits().to_vec();
            sampler.sample(&mut logits)
        };
        pos += 1;
        if next == BOS || next == EOS {
            break;
        }
        if pos >= p.prompt_tokens.len() {
            let piece = p.tok.decode(token, next);
            out.push_str(&String::from_utf8_lossy(&piece));
        }
        token = next;
        // Yield so the data-plane/Tailscale thread runs — this loop is the slow path and
        // must not starve the network (keepalives, map long-poll) or PySpell `/run`.
        unsafe { esp_idf_svc::sys::vTaskDelay(1) };
    }
    let dt = (unsafe { esp_idf_svc::sys::esp_timer_get_time() } - t0) / 1000;
    println!("device_llm: {pos} tok in {dt} ms; heap now {} B", free_heap());
    if out.is_empty() {
        "(no output)".into()
    } else {
        out
    }
}

/// Start a generation in the background (one at a time). All allocation happens here, on
/// the caller's thread, so the worker can't OOM mid-run; returns immediately so the HTTP
/// handler never holds the network thread for the whole (slow) run. Poll with [`poll`].
pub fn start(prompt: String) -> String {
    let (free, largest) = (free_heap(), largest_block());
    println!("device_llm: start: free {free} B, largest block {largest} B");
    {
        let g = STATE.lock().unwrap();
        if g.running {
            return "busy: a generation is already running — GET /generate".into();
        }
    }
    // Admission: the int8 KV cache needs ~8 kB contiguous blocks; Tailscale fragments the
    // heap, so check the largest block (not just total) and refuse cleanly if too tight.
    if free < MIN_FREE_HEAP || largest < 18 * 1024 {
        return format!(
            "low memory: free {free} B, largest block {largest} B (need ~{MIN_FREE_HEAP} + ≥18 kB block) — try again when idle (Tailscale/a fetch is using the heap)"
        );
    }
    STATE.lock().unwrap().running = true;
    // Hand the prompt to the persistent worker (allocates the RunState itself, fallibly).
    if worker().lock().unwrap().send(prompt).is_err() {
        STATE.lock().unwrap().running = false;
        return "error: generation worker is gone".into();
    }
    "started — GET /generate for the result (on-device, ~seconds)".into()
}

/// Poll the current/last generation: a status line while running, the text once done.
pub fn poll() -> String {
    let mut g = STATE.lock().unwrap();
    if g.running {
        return "generating on-device…".into();
    }
    match g.result.take() {
        Some(r) => r,
        None => "idle — POST a prompt to /generate to run the model on the chip".into(),
    }
}
