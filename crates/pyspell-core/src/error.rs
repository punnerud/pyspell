//! Errors for PySpell — both compile-time (front-ends, host-only) and
//! eval-time (the portable evaluator). Kept `no_std`-compatible.

use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DslError {
    /// The front-end parser rejected the source (host-only).
    Parse(String),
    /// A syntactic construct outside the sandboxed subset (host-only).
    Forbidden(String),
    /// A name that is neither a bound local nor provided by the [`crate::env::Env`].
    UnknownName(String),
    /// A builtin called with the wrong number of arguments.
    Arity { builtin: &'static str, got: usize },
    /// A type mismatch detected while evaluating (e.g. indexing a scalar).
    Type(String),
    /// The per-evaluation instruction budget was exhausted (runaway guard).
    Budget,
    /// The wall-clock deadline (caller-supplied timeout) passed mid-evaluation.
    Timeout,
    /// (De)serialization of a program over the wire failed.
    Wire(String),
    /// A `fetch` network capability error (no capability, disallowed host, HTTP
    /// failure, …). The message is host/device-supplied.
    Net(String),
    /// A `show` display capability error (no capability, disabled in config, …).
    Display(String),
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslError::Parse(m) => write!(f, "parse error: {m}"),
            DslError::Forbidden(m) => write!(f, "not allowed: {m}"),
            DslError::UnknownName(n) => write!(f, "unknown name `{n}`"),
            DslError::Arity { builtin, got } => {
                write!(f, "builtin `{builtin}` called with {got} argument(s)")
            }
            DslError::Type(m) => write!(f, "type error: {m}"),
            DslError::Budget => write!(f, "program exceeded its evaluation step budget"),
            DslError::Timeout => write!(f, "program exceeded its time limit"),
            DslError::Wire(m) => write!(f, "wire error: {m}"),
            DslError::Net(m) => write!(f, "network error: {m}"),
            DslError::Display(m) => write!(f, "display error: {m}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DslError {}
