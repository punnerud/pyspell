//! Browser-WASM binding for `tinyllm`.
//!
//! The exact `no_std + alloc` inference engine from `tinyllm` (host-tested),
//! compiled to wasm32 and exposed to JavaScript via `wasm-bindgen`. The model
//! weights (`.bin`, ≤6 MB, llama2.c Q8_0 format) and `tokenizer.bin` are streamed
//! from the ESP32's flash over the lwIP TCP bridge, then inference runs entirely in
//! the user's browser — the dongle is just the (reliable, memory-light) file host.
//!
//! Usage from JS (token-by-token streaming so the UI never blocks):
//! ```js
//! import init, { Generator } from "./tinyllm_wasm.js";
//! await init();                              // loads the .wasm
//! const g = new Generator(modelBytes, tokBytes, "Once upon a time", 64, 0.9, 0.9, 1234n);
//! let piece;
//! while ((piece = g.step()) !== undefined) { out.textContent += piece; await raf(); }
//! ```

use wasm_bindgen::prelude::*;

use tinyllm::tokenizer::{BOS, EOS};
use tinyllm::{RunState, Sampler, Tokenizer, Transformer};

/// A streaming text generator. Holds the model bytes + run state across calls; each
/// [`Generator::step`] runs one decode step and returns the next text piece.
#[wasm_bindgen]
pub struct Generator {
    model: Vec<u8>,           // llama2.c Q8_0 checkpoint (borrowed per step, no copy)
    tok: Tokenizer,           // owned (does not borrow `model`)
    state: RunState,          // reused across tokens
    prompt_tokens: Vec<usize>,
    pos: usize,
    token: usize,             // current input token
    max_steps: usize,
    sampler: Sampler,
    done: bool,
}

#[wasm_bindgen]
impl Generator {
    /// Build a generator. `model`/`tokenizer` are the raw `.bin` bytes streamed from
    /// the device. `temperature` 0 = greedy; `topp` in (0,1) enables nucleus sampling.
    #[wasm_bindgen(constructor)]
    pub fn new(
        model: Vec<u8>,
        tokenizer: Vec<u8>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        topp: f32,
        seed: f64,
    ) -> Result<Generator, JsValue> {
        let cfg = {
            let t = Transformer::new(&model)
                .map_err(|e| JsValue::from_str(&format!("model parse: {e}")))?;
            *t.config()
        };
        let state = RunState::new(&cfg);
        let tok = Tokenizer::from_bytes(&tokenizer, cfg.vocab_size)
            .map_err(|e| JsValue::from_str(&format!("tokenizer parse: {e:?}")))?;
        let prompt_tokens = tok.encode(prompt, true, false); // BOS, no EOS
        if prompt_tokens.is_empty() {
            return Err(JsValue::from_str("empty prompt"));
        }
        let max_steps = if max_tokens == 0 {
            cfg.seq_len
        } else {
            max_tokens.min(cfg.seq_len)
        };
        let token = prompt_tokens[0];
        Ok(Generator {
            model,
            tok,
            state,
            prompt_tokens,
            pos: 0,
            token,
            max_steps,
            sampler: Sampler::new(cfg.vocab_size, temperature, topp, seed as u64),
            done: false,
        })
    }

    /// Run one decode step. Returns the next decoded text piece (an empty string
    /// while the prompt is being prefilled), or `undefined` when generation is done
    /// (hit `max_tokens`, the context window, or an end-of-text token). Call in a JS
    /// loop yielding to the event loop between steps so the page stays responsive.
    pub fn step(&mut self) -> Option<String> {
        if self.done || self.pos >= self.max_steps {
            self.done = true;
            return None;
        }
        // Re-parse the borrowed checkpoint (cheap: offset math, no copy) so the
        // struct needn't hold a self-referential Transformer.
        let t = match Transformer::new(&self.model) {
            Ok(t) => t,
            Err(_) => {
                self.done = true;
                return None;
            }
        };
        t.forward(&mut self.state, self.token, self.pos);

        let next = if self.pos + 1 < self.prompt_tokens.len() {
            self.prompt_tokens[self.pos + 1] // still feeding the prompt
        } else {
            let mut logits = self.state.logits().to_vec();
            self.sampler.sample(&mut logits)
        };
        self.pos += 1;

        if next == BOS || next == EOS {
            self.done = true;
            return None;
        }
        let piece = self.tok.decode(self.token, next);
        let in_prompt = self.pos < self.prompt_tokens.len();
        self.token = next;
        if in_prompt {
            Some(String::new()) // don't echo the prompt; keep the loop going
        } else {
            Some(String::from_utf8_lossy(&piece).into_owned())
        }
    }

    /// True once generation has finished.
    #[wasm_bindgen(getter)]
    pub fn done(&self) -> bool {
        self.done
    }

    /// Tokens generated/consumed so far (prompt + output).
    #[wasm_bindgen(getter)]
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// Parse just the model header and return a one-line description (dim/layers/vocab/
/// seq_len) — handy for the UI to show what model the dongle served.
#[wasm_bindgen]
pub fn model_info(model: &[u8]) -> Result<String, JsValue> {
    let t = Transformer::new(model).map_err(|e| JsValue::from_str(&format!("{e}")))?;
    let c = t.config();
    Ok(format!(
        "dim={} layers={} heads={} vocab={} seq_len={} group_size={}",
        c.dim, c.n_layers, c.n_heads, c.vocab_size, c.seq_len, c.group_size
    ))
}
