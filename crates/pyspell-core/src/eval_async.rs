//! Async sibling of [`crate::eval`]. Identical language semantics — the PySpell
//! program is the same sync expression tree — but the evaluator is `async` so the
//! one effect that blocks, `fetch`/`fetch_json`, can `.await` non-blocking I/O.
//! This is what lets many PySpell jobs run concurrently on a cooperative executor
//! (embassy) without OS threads: while one job awaits the network, others run.
//!
//! Everything pure (arithmetic, comparisons, list/string builtins, JSON probe)
//! is shared verbatim with the sync evaluator via `pub(crate)` helpers — only the
//! recursion and the network builtins differ.
//!
//! The network capability is a generic `N: AsyncNet` (not `&dyn`): native
//! `async fn` in traits is not yet dyn-compatible, and there is exactly one impl
//! per firmware, so monomorphization costs nothing here.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::Pin;

use crate::env::Env;
use crate::error::DslError;
use crate::eval::{
    apply_builtin, as_bool, builtin_name, compare, expect_str, index, json_probe, num_binop,
    vec_filled, Display, DEADLINE_CHECK_INTERVAL,
};
use crate::ir::{BoolOp, Builtin, Expr, Program, UnOp, Value};

/// Async network capability for `fetch`/`fetch_json`. Mirrors [`crate::eval::Net`]
/// but the methods `.await`. One impl per firmware (the device's lean TLS client).
#[allow(async_fn_in_trait)]
pub trait AsyncNet {
    /// Fetch the whole response body as a string.
    async fn fetch(&self, url: &str) -> Result<String, DslError>;

    /// Stream `url`, calling `probe` with the bytes received so far after each
    /// chunk, returning as soon as `probe` yields `Some` (early abort). Default
    /// buffers the whole body then probes once; the device overrides with true
    /// streaming.
    async fn fetch_extract(
        &self,
        url: &str,
        probe: &dyn Fn(&[u8]) -> Option<Value>,
    ) -> Result<Value, DslError> {
        let body = self.fetch(url).await?;
        probe(body.as_bytes())
            .ok_or_else(|| DslError::Net(String::from("field not found in response")))
    }
}

/// Evaluation context with interior mutability, so the recursive async evaluator
/// only ever needs a shared `&` reference (no `&mut` threading across `.await`).
struct AsyncCtx<'b, E: Env, N: AsyncNet> {
    env: &'b E,
    net: Option<&'b N>,
    display: Option<&'b dyn Display>,
    locals: RefCell<Vec<Value>>,
    budget: Cell<u32>,
    deadline: Option<&'b dyn Fn() -> bool>,
    since_check: Cell<u32>,
}

/// Evaluate a program asynchronously. Same contract as [`crate::eval::run_with`]
/// but the network capability is async, enabling cooperative concurrency.
pub async fn run_async<E: Env, N: AsyncNet>(
    program: &Program,
    env: &E,
    net: Option<&N>,
    max_steps: u32,
    deadline: Option<&dyn Fn() -> bool>,
    display: Option<&dyn Display>,
) -> Result<Value, DslError> {
    let ctx = AsyncCtx {
        env,
        net,
        display,
        locals: RefCell::new(vec_filled(program.n_locals as usize)),
        budget: Cell::new(max_steps),
        deadline,
        since_check: Cell::new(0),
    };
    for b in &program.body {
        let v = eval_async(&b.expr, &ctx).await?;
        ctx.locals.borrow_mut()[b.slot as usize] = v;
    }
    eval_async(&program.ret, &ctx).await
}

fn tick<E: Env, N: AsyncNet>(c: &AsyncCtx<'_, E, N>) -> Result<(), DslError> {
    if c.budget.get() == 0 {
        return Err(DslError::Budget);
    }
    c.budget.set(c.budget.get() - 1);
    if let Some(deadline) = c.deadline {
        let sc = c.since_check.get() + 1;
        if sc >= DEADLINE_CHECK_INTERVAL {
            c.since_check.set(0);
            if deadline() {
                return Err(DslError::Timeout);
            }
        } else {
            c.since_check.set(sc);
        }
    }
    Ok(())
}

fn eval_async<'a, 'b: 'a, E: Env, N: AsyncNet>(
    e: &'a Expr,
    c: &'a AsyncCtx<'b, E, N>,
) -> Pin<Box<dyn Future<Output = Result<Value, DslError>> + 'a>> {
    Box::pin(async move {
        tick(c)?;
        match e {
            Expr::Const(v) => Ok(v.clone()),
            Expr::Local(i) => Ok(c.locals.borrow()[*i as usize].clone()),
            Expr::Var(name) => c
                .env
                .get(name)
                .ok_or_else(|| DslError::UnknownName(name.clone())),
            Expr::Bin(op, a, b) => {
                let x = eval_async(a, c).await?;
                let y = eval_async(b, c).await?;
                num_binop(*op, x, y)
            }
            Expr::Cmp(op, a, b) => {
                let x = eval_async(a, c).await?;
                let y = eval_async(b, c).await?;
                Ok(Value::Bool(compare(*op, x, y)?))
            }
            Expr::Bool(op, a, b) => {
                let l = as_bool(&eval_async(a, c).await?)?;
                match (op, l) {
                    (BoolOp::And, false) => Ok(Value::Bool(false)),
                    (BoolOp::Or, true) => Ok(Value::Bool(true)),
                    _ => Ok(Value::Bool(as_bool(&eval_async(b, c).await?)?)),
                }
            }
            Expr::Unary(op, a) => {
                let v = eval_async(a, c).await?;
                match op {
                    UnOp::Neg => match v {
                        Value::Int(n) => Ok(Value::Int(-n)),
                        Value::Float(x) => Ok(Value::Float(-x)),
                        _ => Err(DslError::Type(String::from("cannot negate a non-number"))),
                    },
                    UnOp::Not => Ok(Value::Bool(!as_bool(&v)?)),
                }
            }
            Expr::Index(l, i) => {
                let list = eval_async(l, c).await?;
                let idx = eval_async(i, c).await?;
                index(list, idx)
            }
            Expr::If(cond, t, e2) => {
                if as_bool(&eval_async(cond, c).await?)? {
                    eval_async(t, c).await
                } else {
                    eval_async(e2, c).await
                }
            }
            Expr::Call(b, args) => call_builtin_async(*b, args, c).await,
            Expr::List(items) => {
                let mut v = Vec::with_capacity(items.len());
                for it in items {
                    v.push(eval_async(it, c).await?);
                }
                Ok(Value::List(v.into()))
            }
        }
    })
}

fn call_builtin_async<'a, 'b: 'a, E: Env, N: AsyncNet>(
    b: Builtin,
    args: &'a [Expr],
    c: &'a AsyncCtx<'b, E, N>,
) -> Pin<Box<dyn Future<Output = Result<Value, DslError>> + 'a>> {
    Box::pin(async move {
        let name = builtin_name(b);
        let mut vals: Vec<Value> = Vec::with_capacity(args.len());
        for a in args {
            vals.push(eval_async(a, c).await?);
        }
        match b {
            Builtin::Fetch => {
                if vals.len() != 1 {
                    return Err(DslError::Arity { builtin: name, got: vals.len() });
                }
                let url = expect_str(&vals[0], "fetch() expects a url string")?;
                let net = c
                    .net
                    .ok_or_else(|| DslError::Net(String::from("no network capability installed")))?;
                Ok(Value::str(&net.fetch(&url).await?))
            }
            Builtin::FetchJson => {
                if vals.len() != 2 {
                    return Err(DslError::Arity { builtin: name, got: vals.len() });
                }
                let url = expect_str(&vals[0], "fetch_json() expects a url string")?;
                let path = expect_str(&vals[1], "fetch_json() expects a path string")?;
                let net = c
                    .net
                    .ok_or_else(|| DslError::Net(String::from("no network capability installed")))?;
                let probe = json_probe(&path);
                net.fetch_extract(&url, &probe).await
            }
            _ => apply_builtin(b, vals, c.display, None),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::EmptyEnv;
    use crate::ir::{BinOp, DEFAULT_MAX_STEPS};
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::task::Wake;
    use alloc::vec;
    use core::task::{Context, Poll, Waker};

    /// Minimal `block_on` for tests — the mock net is always ready, so a busy
    /// poll with a no-op waker drives the future to completion. Uses the safe
    /// `Wake` trait (the crate forbids `unsafe`).
    struct NoopWake;
    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
        fn wake_by_ref(self: &Arc<Self>) {}
    }

    fn block_on<F: Future>(fut: F) -> F::Output {
        let waker = Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        let mut fut = Box::pin(fut);
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    struct MockNet {
        body: &'static str,
    }
    impl AsyncNet for MockNet {
        async fn fetch(&self, _url: &str) -> Result<String, DslError> {
            Ok(String::from(self.body))
        }
    }

    fn prog(ret: Expr) -> Program {
        Program { body: vec![], ret, n_locals: 0, max_steps: DEFAULT_MAX_STEPS }
    }

    #[test]
    fn async_arithmetic_no_net() {
        let p = prog(Expr::Bin(
            BinOp::Add,
            Box::new(Expr::Const(Value::Int(2))),
            Box::new(Expr::Const(Value::Int(3))),
        ));
        let out = block_on(run_async::<_, MockNet>(&p, &EmptyEnv, None, p.max_steps, None, None));
        assert_eq!(out.unwrap(), Value::Int(5));
    }

    #[test]
    fn async_fetch_json_extracts_field() {
        let net = MockNet { body: r#"{"a":{"b":42}}"# };
        let p = prog(Expr::Call(
            Builtin::FetchJson,
            vec![
                Expr::Const(Value::str("http://example/x")),
                Expr::Const(Value::str("a.b")),
            ],
        ));
        let out = block_on(run_async(&p, &EmptyEnv, Some(&net), p.max_steps, None, None));
        assert_eq!(out.unwrap(), Value::Int(42));
    }
}
