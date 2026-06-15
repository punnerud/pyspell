//! The shared, front-end-independent IR a PySpell program compiles to.
//!
//! Both the Rust (`syn`) and Python (`rustpython`) front-ends in `pyspell-lang`
//! lower to this, so `eval.rs` is the single native evaluator. A `Program` is
//! immutable, `serde`-serializable (the wire format that ships to the device),
//! and cheap to share behind an `Arc`.
//!
//! This was generalized from a VRP-constraint DSL: the old fixed `route.*` /
//! `vehicle.*` field schema is gone. A free identifier now lowers to
//! [`Expr::Var`], resolved at evaluation time against a host-supplied
//! [`crate::env::Env`] — so the same evaluator serves any domain (a routing
//! constraint on the host, a sensor predicate on the ESP32, …).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

pub use crate::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoolOp {
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Builtin {
    Len,
    Abs,
    Min,
    Max,
    Sum,
    Any,
    All,
    Round,
    Int,
    Float,
    Bool,
    /// `list.contains(x)` / `x in list` → Bool.
    Contains,
    /// `index(list, x)` → position of first x, or -1 if absent.
    IndexOf,
    /// `before(list, a, b)` → true iff a occurs before b; false if either absent.
    Before,
    /// `first(list)` → list[0], or -1 if empty.
    First,
    /// `last(list)` → last element, or -1 if empty.
    Last,
    /// `str(x)` → string representation of a value.
    Str,
    /// `json_get(text, "a.b.0.c")` → the scalar at a dotted/indexed JSON path.
    /// Path-directed: it scans the text and only materializes the matched value.
    JsonGet,
    /// `fetch(url)` → HTTP GET body as a string. A host-provided capability
    /// ([`crate::eval::Net`]); the evaluator itself does no I/O. Host/device
    /// enforce any allowlist. Errors if no network capability is installed.
    Fetch,
    /// `fetch_json(url, "a.b.0.c")` → stream the response and extract just the
    /// scalar at the JSON path, stopping as soon as it's found. Memory-optimal:
    /// the device never buffers the whole body. See [`crate::eval::Net::fetch_extract`].
    FetchJson,
    /// `show(x)` → render `x` to text and display it (a host-provided
    /// [`crate::eval::Display`] capability, e.g. the ESP32 screen), returning `x`
    /// so it composes. Errors if no display capability is installed.
    Show,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    Const(Value),
    /// A free identifier, resolved against the host [`crate::env::Env`] at eval
    /// time. This is the single bridge to host data — no other attribute access
    /// or I/O exists in the grammar (deny-by-default sandbox).
    Var(String),
    /// A `let`-bound local, by slot.
    Local(u16),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    Bool(BoolOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Builtin, Vec<Expr>),
    List(Vec<Expr>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LetBinding {
    pub slot: u16,
    pub expr: Expr,
}

/// A compiled program: ordered `let` bindings then a return expression. This is
/// the unit that compiles on the host and ships to the device.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub body: Vec<LetBinding>,
    pub ret: Expr,
    pub n_locals: u16,
    pub max_steps: u32,
}

/// Default per-evaluation instruction budget (runaway guard).
pub const DEFAULT_MAX_STEPS: u32 = 4096;
