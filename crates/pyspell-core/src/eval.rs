//! Native tree-walk evaluator for a compiled [`Program`].
//!
//! No FFI, no allocation in the scalar path — a typical predicate is a handful
//! of node visits. A per-call step budget guards against runaway programs.
//! Free identifiers are resolved against a host-supplied [`Env`]; everything
//! else is pure and side-effect free (the sandbox).

use alloc::format;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use alloc::string::String;

use crate::env::Env;
use crate::error::DslError;
use crate::ir::{BinOp, BoolOp, Builtin, CmpOp, Expr, Program, UnOp, Value};

/// A host-provided network capability for the `fetch(url)` builtin. The
/// evaluator itself performs no I/O — this is the single mediated effect, so the
/// host (CLI: an HTTP client; device: esp-idf + a config allowlist) controls
/// exactly what a program may reach. `None` installed → `fetch` errors.
pub trait Net {
    /// Fetch the whole response body as a string.
    fn fetch(&self, url: &str) -> Result<String, DslError>;

    /// Stream `url`, calling `probe` with the bytes received so far after each
    /// chunk, and return as soon as `probe` yields `Some` — letting an
    /// implementation extract one field without ever buffering the whole body
    /// (and abort the transfer early). The default implementation fetches the
    /// full body and probes once, which is fine on the host; the device
    /// overrides it with true streaming to fit in RAM.
    fn fetch_extract(
        &self,
        url: &str,
        probe: &dyn Fn(&[u8]) -> Option<Value>,
    ) -> Result<Value, DslError> {
        let body = self.fetch(url)?;
        probe(body.as_bytes())
            .ok_or_else(|| DslError::Net(String::from("field not found in response")))
    }
}

struct Frame<'a> {
    env: &'a dyn Env,
    net: Option<&'a dyn Net>,
    locals: Vec<Value>,
    budget: u32,
    /// Optional wall-clock guard: called periodically, returns `true` when the
    /// caller-supplied deadline has passed. `pyspell-core` has no clock of its
    /// own (it is `no_std`), so the host/device supplies one — e.g. the ESP32
    /// compares `esp_timer_get_time()` against a deadline. `None` = no timeout.
    deadline: Option<&'a dyn Fn() -> bool>,
    /// Steps since the last (relatively expensive) deadline check.
    since_check: u32,
}

/// How long a program may run, and what it may reach.
pub struct Limits<'a> {
    /// Per-evaluation instruction budget (runaway guard). Independent of wall time.
    pub max_steps: u32,
    /// Optional wall-clock deadline predicate (see [`Frame::deadline`]).
    pub deadline: Option<&'a dyn Fn() -> bool>,
    /// Optional network capability for `fetch` (see [`Net`]).
    pub net: Option<&'a dyn Net>,
}

impl Default for Limits<'_> {
    fn default() -> Self {
        Limits { max_steps: crate::ir::DEFAULT_MAX_STEPS, deadline: None, net: None }
    }
}

/// Check the wall-clock deadline at most this often (in steps) to keep the cost
/// of a clock read off the hot path.
const DEADLINE_CHECK_INTERVAL: u32 = 256;

/// Evaluate a compiled program against a host environment, returning its final
/// [`Value`]. Uses the program's own `max_steps` and no wall-clock timeout. The
/// result interpretation (bool predicate, numeric score, …) is the caller's.
pub fn run<E: Env>(program: &Program, env: &E) -> Result<Value, DslError> {
    run_with(program, env, Limits { max_steps: program.max_steps, deadline: None, net: None })
}

/// Evaluate with explicit [`Limits`] — a step budget plus an optional wall-clock
/// deadline. This is what the device uses to honor a request-supplied timeout
/// (e.g. "10 s").
pub fn run_with<E: Env>(program: &Program, env: &E, limits: Limits) -> Result<Value, DslError> {
    let mut f = Frame {
        env,
        net: limits.net,
        locals: vec_filled(program.n_locals as usize),
        budget: limits.max_steps,
        deadline: limits.deadline,
        since_check: 0,
    };
    for b in &program.body {
        let v = eval(&b.expr, &mut f)?;
        f.locals[b.slot as usize] = v;
    }
    eval(&program.ret, &mut f)
}

fn vec_filled(n: usize) -> Vec<Value> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(Value::Int(0));
    }
    v
}

fn eval(e: &Expr, f: &mut Frame) -> Result<Value, DslError> {
    {
        if f.budget == 0 {
            return Err(DslError::Budget);
        }
        f.budget -= 1;
        // Periodic wall-clock check (cheap amortized).
        if let Some(deadline) = f.deadline {
            f.since_check += 1;
            if f.since_check >= DEADLINE_CHECK_INTERVAL {
                f.since_check = 0;
                if deadline() {
                    return Err(DslError::Timeout);
                }
            }
        }
    }
    match e {
        Expr::Const(v) => Ok(v.clone()),
        Expr::Local(i) => Ok(f.locals[*i as usize].clone()),
        Expr::Var(name) => f.env.get(name).ok_or_else(|| DslError::UnknownName(name.clone())),
        Expr::Bin(op, a, b) => {
            let (x, y) = (eval(a, f)?, eval(b, f)?);
            num_binop(*op, x, y)
        }
        Expr::Cmp(op, a, b) => {
            let (x, y) = (eval(a, f)?, eval(b, f)?);
            Ok(Value::Bool(compare(*op, x, y)?))
        }
        Expr::Bool(op, a, b) => {
            let l = as_bool(&eval(a, f)?)?;
            match (op, l) {
                (BoolOp::And, false) => Ok(Value::Bool(false)),
                (BoolOp::Or, true) => Ok(Value::Bool(true)),
                _ => Ok(Value::Bool(as_bool(&eval(b, f)?)?)),
            }
        }
        Expr::Unary(op, a) => {
            let v = eval(a, f)?;
            match op {
                UnOp::Neg => match v {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(x) => Ok(Value::Float(-x)),
                    _ => Err(DslError::Type("cannot negate a non-number".to_string())),
                },
                UnOp::Not => Ok(Value::Bool(!as_bool(&v)?)),
            }
        }
        Expr::Index(l, i) => {
            let list = eval(l, f)?;
            let idx = eval(i, f)?;
            index(list, idx)
        }
        Expr::If(c, t, e2) => {
            if as_bool(&eval(c, f)?)? {
                eval(t, f)
            } else {
                eval(e2, f)
            }
        }
        Expr::Call(b, args) => call_builtin(*b, args, f),
        Expr::List(items) => {
            let mut v = Vec::with_capacity(items.len());
            for it in items {
                v.push(eval(it, f)?);
            }
            Ok(Value::List(v.into()))
        }
    }
}

// ---- value helpers -------------------------------------------------------

fn as_bool(v: &Value) -> Result<bool, DslError> {
    Ok(match v {
        Value::Bool(b) => *b,
        Value::Int(n) => *n != 0,
        Value::Float(x) => *x != 0.0,
        Value::Str(s) => !s.is_empty(),
        Value::List(l) => !l.is_empty(),
    })
}

fn as_f64(v: &Value) -> Result<f64, DslError> {
    match v {
        Value::Int(n) => Ok(*n as f64),
        Value::Float(x) => Ok(*x),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        // A numeric string coerces (lets `json_get` results compare numerically).
        Value::Str(s) => s.trim().parse::<f64>().map_err(|_| {
            DslError::Type("expected a number, got a non-numeric string".to_string())
        }),
        Value::List(_) => Err(DslError::Type("expected a number, got a list".to_string())),
    }
}

/// Render a value as a string (for `str()` and string concatenation).
fn to_str(v: &Value) -> String {
    match v {
        Value::Int(n) => format!("{n}"),
        Value::Float(x) => format!("{x}"),
        Value::Bool(b) => String::from(if *b { "true" } else { "false" }),
        Value::Str(s) => String::from(&**s),
        Value::List(l) => {
            let mut s = String::from("[");
            for (i, it) in l.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&to_str(it));
            }
            s.push(']');
            s
        }
    }
}

fn num_binop(op: BinOp, a: Value, b: Value) -> Result<Value, DslError> {
    // `+` with a string operand concatenates (the value is rendered to text).
    if op == BinOp::Add && (matches!(a, Value::Str(_)) || matches!(b, Value::Str(_))) {
        let mut s = to_str(&a);
        s.push_str(&to_str(&b));
        return Ok(Value::str(&s));
    }
    // If both are ints, stay integral (truncating div/rem); otherwise float.
    if let (Value::Int(x), Value::Int(y)) = (&a, &b) {
        let (x, y) = (*x, *y);
        return Ok(match op {
            BinOp::Add => Value::Int(x.wrapping_add(y)),
            BinOp::Sub => Value::Int(x.wrapping_sub(y)),
            BinOp::Mul => Value::Int(x.wrapping_mul(y)),
            BinOp::Div => {
                if y == 0 {
                    return Err(DslError::Type("division by zero".to_string()));
                }
                Value::Int(x / y)
            }
            BinOp::Rem => {
                if y == 0 {
                    return Err(DslError::Type("modulo by zero".to_string()));
                }
                Value::Int(x % y)
            }
        });
    }
    let (x, y) = (as_f64(&a)?, as_f64(&b)?);
    Ok(Value::Float(match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => x / y,
        BinOp::Rem => x % y,
    }))
}

fn compare(op: CmpOp, a: Value, b: Value) -> Result<bool, DslError> {
    // Bool == Bool / Bool != Bool handled directly; everything else numerically.
    if let (Value::Bool(x), Value::Bool(y)) = (&a, &b) {
        return match op {
            CmpOp::Eq => Ok(x == y),
            CmpOp::Ne => Ok(x != y),
            _ => Err(DslError::Type("booleans support only == and !=".to_string())),
        };
    }
    // String comparisons: equality and lexicographic ordering.
    if let (Value::Str(x), Value::Str(y)) = (&a, &b) {
        return Ok(match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt => x < y,
            CmpOp::Le => x <= y,
            CmpOp::Gt => x > y,
            CmpOp::Ge => x >= y,
        });
    }
    let (x, y) = (as_f64(&a)?, as_f64(&b)?);
    Ok(match op {
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
    })
}

fn index(list: Value, idx: Value) -> Result<Value, DslError> {
    let items = match list {
        Value::List(l) => l,
        _ => return Err(DslError::Type("cannot index a non-list".to_string())),
    };
    let i = match idx {
        Value::Int(n) => n,
        _ => return Err(DslError::Type("list index must be an integer".to_string())),
    };
    // Support Python-style negative indexing.
    let len = items.len() as i64;
    let real = if i < 0 { len + i } else { i };
    if real < 0 || real >= len {
        return Err(DslError::Type("list index out of range".to_string()));
    }
    Ok(items[real as usize].clone())
}

fn call_builtin(b: Builtin, args: &[Expr], f: &mut Frame) -> Result<Value, DslError> {
    let name = builtin_name(b);
    let mut vals: Vec<Value> = Vec::with_capacity(args.len());
    for a in args {
        vals.push(eval(a, f)?);
    }
    let arity_err = |got: usize| DslError::Arity { builtin: name, got };

    match b {
        Builtin::Len => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            match &vals[0] {
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                _ => Err(DslError::Type("len() expects a list or string".to_string())),
            }
        }
        Builtin::Abs => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            match &vals[0] {
                Value::Int(n) => Ok(Value::Int(n.abs())),
                Value::Float(x) => Ok(Value::Float(libm_abs(*x))),
                _ => Err(DslError::Type("abs() expects a number".to_string())),
            }
        }
        Builtin::Round => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Int(round_half_away(as_f64(&vals[0])?) as i64))
        }
        Builtin::Int => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Int(as_f64(&vals[0])? as i64))
        }
        Builtin::Float => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Float(as_f64(&vals[0])?))
        }
        Builtin::Bool => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::Bool(as_bool(&vals[0])?))
        }
        Builtin::Min | Builtin::Max => reduce_minmax(b, vals, name),
        Builtin::Sum => {
            let items = single_list(&vals, name)?;
            let mut int_acc: i64 = 0;
            let mut float_acc: f64 = 0.0;
            let mut any_float = false;
            for v in items.iter() {
                match v {
                    Value::Int(n) => int_acc += *n,
                    Value::Float(x) => {
                        any_float = true;
                        float_acc += *x;
                    }
                    _ => return Err(DslError::Type("sum() expects a list of numbers".to_string())),
                }
            }
            Ok(if any_float {
                Value::Float(float_acc + int_acc as f64)
            } else {
                Value::Int(int_acc)
            })
        }
        Builtin::Any => {
            let items = single_list(&vals, name)?;
            for v in items.iter() {
                if as_bool(v)? {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        }
        Builtin::All => {
            let items = single_list(&vals, name)?;
            for v in items.iter() {
                if !as_bool(v)? {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        }
        Builtin::Contains => {
            if vals.len() != 2 {
                return Err(arity_err(vals.len()));
            }
            let items = match &vals[0] {
                Value::List(l) => l.clone(),
                _ => return Err(DslError::Type("contains expects a list".to_string())),
            };
            let needle = as_f64(&vals[1])?;
            for v in items.iter() {
                if let Ok(x) = as_f64(v) {
                    if x == needle {
                        return Ok(Value::Bool(true));
                    }
                }
            }
            Ok(Value::Bool(false))
        }
        Builtin::IndexOf => {
            if vals.len() != 2 {
                return Err(arity_err(vals.len()));
            }
            let items = list_of(&vals[0], "index")?;
            let needle = as_f64(&vals[1])?;
            Ok(Value::Int(position_of(&items, needle).map(|i| i as i64).unwrap_or(-1)))
        }
        Builtin::Before => {
            if vals.len() != 3 {
                return Err(arity_err(vals.len()));
            }
            let items = list_of(&vals[0], "before")?;
            let a = as_f64(&vals[1])?;
            let b = as_f64(&vals[2])?;
            let verdict = match (position_of(&items, a), position_of(&items, b)) {
                (Some(ia), Some(ib)) => ia < ib,
                _ => false,
            };
            Ok(Value::Bool(verdict))
        }
        Builtin::First => {
            let items = single_list(&vals, "first")?;
            Ok(items.first().cloned().unwrap_or(Value::Int(-1)))
        }
        Builtin::Last => {
            let items = single_list(&vals, "last")?;
            Ok(items.last().cloned().unwrap_or(Value::Int(-1)))
        }
        Builtin::Str => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            Ok(Value::str(&to_str(&vals[0])))
        }
        Builtin::JsonGet => {
            if vals.len() != 2 {
                return Err(arity_err(vals.len()));
            }
            let text = match &vals[0] {
                Value::Str(s) => s.clone(),
                _ => return Err(DslError::Type("json_get() expects a string as 1st arg".to_string())),
            };
            let path = match &vals[1] {
                Value::Str(s) => s.clone(),
                _ => return Err(DslError::Type("json_get() expects a path string as 2nd arg".to_string())),
            };
            crate::json::get(&text, &path)
        }
        Builtin::Fetch => {
            if vals.len() != 1 {
                return Err(arity_err(vals.len()));
            }
            let url = match &vals[0] {
                Value::Str(s) => s.clone(),
                _ => return Err(DslError::Type("fetch() expects a url string".to_string())),
            };
            match f.net {
                Some(net) => Ok(Value::str(&net.fetch(&url)?)),
                None => Err(DslError::Net("no network capability installed".to_string())),
            }
        }
        Builtin::FetchJson => {
            if vals.len() != 2 {
                return Err(arity_err(vals.len()));
            }
            let url = match &vals[0] {
                Value::Str(s) => s.clone(),
                _ => return Err(DslError::Type("fetch_json() expects a url string".to_string())),
            };
            let path = match &vals[1] {
                Value::Str(s) => s.clone(),
                _ => return Err(DslError::Type("fetch_json() expects a path string".to_string())),
            };
            let net = match f.net {
                Some(net) => net,
                None => return Err(DslError::Net("no network capability installed".to_string())),
            };
            // Probe: try to extract the scalar from the bytes received so far.
            // Trim a trailing partial UTF-8 char so a mid-codepoint chunk
            // boundary doesn't abort the parse.
            let probe = |buf: &[u8]| -> Option<Value> {
                let s = match core::str::from_utf8(buf) {
                    Ok(s) => s,
                    Err(e) => core::str::from_utf8(&buf[..e.valid_up_to()]).ok()?,
                };
                crate::json::get(s, &path).ok()
            };
            net.fetch_extract(&url, &probe)
        }
    }
}

/// `f64::abs` without pulling `std` (works in `no_std`).
fn libm_abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Round half away from zero, matching Python's `round()` on .5 only loosely
/// but adequate for the integer-cast result and `no_std`-safe.
fn round_half_away(x: f64) -> f64 {
    if x >= 0.0 {
        (x + 0.5) as i64 as f64
    } else {
        -((-x + 0.5) as i64 as f64)
    }
}

fn position_of(items: &[Value], needle: f64) -> Option<usize> {
    items.iter().position(|v| as_f64(v).map(|x| x == needle).unwrap_or(false))
}

fn list_of(v: &Value, name: &'static str) -> Result<Arc<[Value]>, DslError> {
    match v {
        Value::List(l) => Ok(l.clone()),
        _ => Err(DslError::Type(format!("{name}() expects a list as its first argument"))),
    }
}

fn single_list(vals: &[Value], name: &'static str) -> Result<Arc<[Value]>, DslError> {
    if vals.len() != 1 {
        return Err(DslError::Arity { builtin: name, got: vals.len() });
    }
    match &vals[0] {
        Value::List(l) => Ok(l.clone()),
        _ => Err(DslError::Type(format!("{name}() expects a list"))),
    }
}

fn reduce_minmax(b: Builtin, vals: Vec<Value>, name: &'static str) -> Result<Value, DslError> {
    // min/max accept either a single list or 2+ scalar args (Python-like).
    let items: Vec<Value> = if vals.len() == 1 {
        match &vals[0] {
            Value::List(l) => l.to_vec(),
            _ => return Err(DslError::Type(format!("{name}() of a single non-list"))),
        }
    } else if vals.len() >= 2 {
        vals
    } else {
        return Err(DslError::Arity { builtin: name, got: vals.len() });
    };
    if items.is_empty() {
        return Err(DslError::Type(format!("{name}() of an empty list")));
    }
    let mut best = items[0].clone();
    let mut best_f = as_f64(&best)?;
    for v in items.iter().skip(1) {
        let val = as_f64(v)?;
        let take = match b {
            Builtin::Min => val < best_f,
            _ => val > best_f,
        };
        if take {
            best = v.clone();
            best_f = val;
        }
    }
    Ok(best)
}

fn builtin_name(b: Builtin) -> &'static str {
    match b {
        Builtin::Len => "len",
        Builtin::Abs => "abs",
        Builtin::Min => "min",
        Builtin::Max => "max",
        Builtin::Sum => "sum",
        Builtin::Any => "any",
        Builtin::All => "all",
        Builtin::Round => "round",
        Builtin::Int => "int",
        Builtin::Float => "float",
        Builtin::Bool => "bool",
        Builtin::Contains => "contains",
        Builtin::IndexOf => "index",
        Builtin::Before => "before",
        Builtin::First => "first",
        Builtin::Last => "last",
        Builtin::Str => "str",
        Builtin::JsonGet => "json_get",
        Builtin::Fetch => "fetch",
        Builtin::FetchJson => "fetch_json",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{EmptyEnv, VecEnv};
    use crate::ir::{DEFAULT_MAX_STEPS, LetBinding};
    use alloc::boxed::Box;
    use alloc::vec;

    fn prog(ret: Expr) -> Program {
        Program { body: vec![], ret, n_locals: 0, max_steps: DEFAULT_MAX_STEPS }
    }

    #[test]
    fn const_and_arithmetic() {
        let p = prog(Expr::Bin(
            BinOp::Add,
            Box::new(Expr::Const(Value::Int(2))),
            Box::new(Expr::Const(Value::Int(3))),
        ));
        assert_eq!(run(&p, &EmptyEnv).unwrap(), Value::Int(5));
    }

    #[test]
    fn var_resolves_from_env() {
        let env = VecEnv::new().set("x", 10i64).set("y", 4i64);
        let p = prog(Expr::Cmp(
            CmpOp::Gt,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        ));
        assert_eq!(run(&p, &env).unwrap(), Value::Bool(true));
    }

    #[test]
    fn unknown_var_errors() {
        let p = prog(Expr::Var("missing".into()));
        assert_eq!(run(&p, &EmptyEnv), Err(DslError::UnknownName("missing".into())));
    }

    #[test]
    fn short_circuit_and() {
        // false && (1/0) must not evaluate the rhs.
        let p = prog(Expr::Bool(
            BoolOp::And,
            Box::new(Expr::Const(Value::Bool(false))),
            Box::new(Expr::Bin(
                BinOp::Div,
                Box::new(Expr::Const(Value::Int(1))),
                Box::new(Expr::Const(Value::Int(0))),
            )),
        ));
        assert_eq!(run(&p, &EmptyEnv).unwrap(), Value::Bool(false));
    }

    #[test]
    fn list_builtins() {
        let p = prog(Expr::Call(
            Builtin::Sum,
            vec![Expr::List(vec![
                Expr::Const(Value::Int(1)),
                Expr::Const(Value::Int(2)),
                Expr::Const(Value::Int(3)),
            ])],
        ));
        assert_eq!(run(&p, &EmptyEnv).unwrap(), Value::Int(6));
    }

    #[test]
    fn let_bindings() {
        // let d = x - y; d <= 1000
        let env = VecEnv::new().set("x", 1100i64).set("y", 100i64);
        let p = Program {
            body: vec![LetBinding {
                slot: 0,
                expr: Expr::Bin(
                    BinOp::Sub,
                    Box::new(Expr::Var("x".into())),
                    Box::new(Expr::Var("y".into())),
                ),
            }],
            ret: Expr::Cmp(
                CmpOp::Le,
                Box::new(Expr::Local(0)),
                Box::new(Expr::Const(Value::Int(1000))),
            ),
            n_locals: 1,
            max_steps: DEFAULT_MAX_STEPS,
        };
        assert_eq!(run(&p, &env).unwrap(), Value::Bool(true));
    }

    #[test]
    fn budget_exhaustion_errors() {
        let mut p = prog(Expr::Const(Value::Bool(true)));
        p.max_steps = 0;
        assert_eq!(run(&p, &EmptyEnv), Err(DslError::Budget));
    }
}
