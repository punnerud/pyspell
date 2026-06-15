//! The host environment a program reads its free variables from.
//!
//! This is the single, mediated bridge between a sandboxed program and the
//! outside world. The grammar has no I/O, no imports, no attribute access — a
//! program can only see what its `Env` chooses to expose. On the host that
//! might be CLI-supplied values; on the ESP32 it is live device state (free
//! heap, uptime, ADC, …).

use alloc::string::String;
use alloc::vec::Vec;

use crate::value::Value;

/// Resolves a free identifier to a [`Value`]. Returning `None` makes the name
/// "unknown" and fails evaluation (deny-by-default).
pub trait Env {
    fn get(&self, name: &str) -> Option<Value>;
}

/// An environment that exposes nothing — every name is unknown.
pub struct EmptyEnv;

impl Env for EmptyEnv {
    fn get(&self, _name: &str) -> Option<Value> {
        None
    }
}

/// A simple name→value environment backed by a `Vec`. Convenient for the host
/// CLI (`--set k=v`) and for tests; small enough for the device too.
#[derive(Default, Clone, Debug)]
pub struct VecEnv {
    vars: Vec<(String, Value)>,
}

impl VecEnv {
    pub fn new() -> Self {
        VecEnv { vars: Vec::new() }
    }

    /// Bind (or rebind) a name. Builder-style so calls can chain.
    pub fn set(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        let name = name.into();
        let value = value.into();
        if let Some(slot) = self.vars.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = value;
        } else {
            self.vars.push((name, value));
        }
        self
    }

    /// In-place bind, for callers that don't want the builder style (e.g. a
    /// device refreshing live readings each loop).
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        let name = name.into();
        let value = value.into();
        if let Some(slot) = self.vars.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = value;
        } else {
            self.vars.push((name, value));
        }
    }
}

impl Env for VecEnv {
    fn get(&self, name: &str) -> Option<Value> {
        self.vars.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone())
    }
}

/// Anything callable `Fn(&str) -> Option<Value>` is an `Env` — handy for
/// wiring device registers without a struct.
impl<F: Fn(&str) -> Option<Value>> Env for F {
    fn get(&self, name: &str) -> Option<Value> {
        self(name)
    }
}
