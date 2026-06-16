//! tinyllm — a portable, `no_std + alloc` int8 transformer inference engine that
//! reads the **llama2.c `version 2` (Q8_0) checkpoint format** verbatim.
//!
//! The same crate links into the host (tests + tooling) and into the ESP32-S3
//! firmware: on the device the weights are an `&'static [u8]` memory-mapped from
//! a flash partition (zero RAM for weights), and only the run state + KV cache
//! live on the heap. This mirrors the `pyspell-core` "portable core + platform
//! adapter" split.
//!
//! Scope is a *toy*: TinyStories-class models (Karpathy `llama2.c`). The format
//! reader, the forward pass, the tokenizer and the sampler are faithful ports of
//! `run.c`, so a host build can be validated token-for-token against it.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod format;
pub mod math;
pub mod model;
pub mod sampler;
pub mod tokenizer;

pub use format::{Config, Model, ModelError};
pub use model::{RunState, Transformer};
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;
