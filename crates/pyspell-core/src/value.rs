//! The runtime value model, shared by the IR and the evaluator.

use alloc::sync::Arc;
use serde::{Deserialize, Serialize};

/// A runtime value. Scalars are unboxed; lists are refcounted so cloning a
/// `Value` during the tree-walk stays cheap. The same type travels over the wire
/// (host → device) and is what an evaluation finally returns.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// A text string (URLs, JSON fragments, extracted fields). Refcounted so
    /// cloning a `Value` stays cheap even for a large fetched body.
    Str(Arc<str>),
    List(Arc<[Value]>),
}

impl Value {
    /// A list value built from an iterator of values.
    pub fn list<I: IntoIterator<Item = Value>>(items: I) -> Value {
        Value::List(items.into_iter().collect())
    }
    /// A string value.
    pub fn str(s: impl AsRef<str>) -> Value {
        Value::Str(Arc::from(s.as_ref()))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            _ => false,
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(Arc::from(s))
    }
}
impl From<alloc::string::String> for Value {
    fn from(s: alloc::string::String) -> Value {
        Value::Str(Arc::from(s.as_str()))
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Value {
        Value::Int(n)
    }
}
impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::Float(x)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}
