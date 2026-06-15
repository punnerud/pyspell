//! PySpell core — the portable IR + sandboxed evaluator + wire format.
//!
//! `no_std + alloc`, so the exact same crate links into the host tools and the
//! ESP32-S3 firmware (the "portable core + platform adapter" split). It has no
//! parser: source is compiled to a [`Program`] on the host by `pyspell-lang`
//! and shipped here over the wire; this crate only evaluates verified IR.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod env;
pub mod error;
pub mod eval;
pub mod ir;
pub mod json;
pub mod parse;
pub mod value;
pub mod wire;

pub use env::{EmptyEnv, Env, VecEnv};
pub use error::DslError;
pub use eval::{run, run_with, Limits, Net};
pub use ir::{Program, DEFAULT_MAX_STEPS};
pub use parse::{parse, Lang};
pub use value::Value;
