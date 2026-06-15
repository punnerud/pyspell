//! Rust-expression front-end: parse with `syn` 2.0 and lower a whitelisted
//! subset to the shared IR. Deny-by-default — anything outside the subset is a
//! compile error (`DslError`), never a panic. Host-only; never ships to a device.

use syn::{BinOp as SBin, Expr as SExpr, Lit, Stmt, UnOp as SUn};

use pyspell_core::error::DslError;
use pyspell_core::ir::{BinOp, BoolOp, Builtin, CmpOp, Expr, LetBinding, Program, UnOp, Value};

use crate::lower::{builtin_from, finish, resolve_name, Ctx};

/// Compile a program written in **Rust expression syntax**: optional `let`
/// bindings followed by a single trailing expression.
pub fn compile_rust(src: &str) -> Result<Program, DslError> {
    // Wrap as a block so `let` bindings + a trailing expression parse.
    let block = syn::parse_str::<syn::Block>(&format!("{{ {src} }}"))
        .map_err(|e| DslError::Parse(e.to_string()))?;
    let n = block.stmts.len();
    if n == 0 {
        return Err(DslError::Parse("empty program".into()));
    }

    let mut ctx = Ctx::new();
    let mut ret: Option<Expr> = None;

    for (i, stmt) in block.stmts.iter().enumerate() {
        let last = i + 1 == n;
        match stmt {
            Stmt::Local(local) if !last => {
                let name = pat_ident(&local.pat)?;
                let init = local
                    .init
                    .as_ref()
                    .ok_or_else(|| DslError::Forbidden("`let` without an initializer".into()))?;
                if init.diverge.is_some() {
                    return Err(DslError::Forbidden("`let ... else`".into()));
                }
                let e = lower(&init.expr, &mut ctx)?;
                let slot = ctx.declare(name);
                ctx.body.push(LetBinding { slot, expr: e });
            }
            Stmt::Expr(e, None) if last => {
                ret = Some(lower(e, &mut ctx)?);
            }
            Stmt::Local(_) => {
                return Err(DslError::Forbidden(
                    "`let` must be followed by more bindings and a final expression".into(),
                ))
            }
            Stmt::Expr(_, Some(_)) => {
                return Err(DslError::Forbidden(
                    "a `;`-terminated statement (expected a trailing expression)".into(),
                ))
            }
            _ => return Err(DslError::Forbidden("unsupported statement".into())),
        }
    }

    let ret = ret.ok_or_else(|| DslError::Parse("no trailing expression to return".into()))?;
    Ok(finish(ctx, ret))
}

fn pat_ident(pat: &syn::Pat) -> Result<String, DslError> {
    match pat {
        syn::Pat::Ident(pi) => Ok(pi.ident.to_string()),
        _ => Err(DslError::Forbidden("only `let <name> = ...` is supported".into())),
    }
}

fn path_ident(p: &syn::ExprPath) -> Option<String> {
    if p.qself.is_none() && p.path.segments.len() == 1 {
        Some(p.path.segments[0].ident.to_string())
    } else {
        None
    }
}

fn block_tail(block: &syn::Block) -> Result<&SExpr, DslError> {
    if block.stmts.len() == 1 {
        if let Stmt::Expr(e, None) = &block.stmts[0] {
            return Ok(e);
        }
    }
    Err(DslError::Forbidden("branch must be a single expression, e.g. `{ x }`".into()))
}

fn lower(e: &SExpr, ctx: &mut Ctx) -> Result<Expr, DslError> {
    match e {
        SExpr::Paren(p) => lower(&p.expr, ctx),
        SExpr::Group(g) => lower(&g.expr, ctx),

        SExpr::Lit(l) => lower_lit(&l.lit),

        SExpr::Unary(u) => match u.op {
            SUn::Not(_) => Ok(Expr::Unary(UnOp::Not, Box::new(lower(&u.expr, ctx)?))),
            SUn::Neg(_) => Ok(Expr::Unary(UnOp::Neg, Box::new(lower(&u.expr, ctx)?))),
            _ => Err(DslError::Forbidden("unary operator".into())),
        },

        SExpr::Binary(b) => {
            let l = Box::new(lower(&b.left, ctx)?);
            let r = Box::new(lower(&b.right, ctx)?);
            match b.op {
                SBin::Add(_) => Ok(Expr::Bin(BinOp::Add, l, r)),
                SBin::Sub(_) => Ok(Expr::Bin(BinOp::Sub, l, r)),
                SBin::Mul(_) => Ok(Expr::Bin(BinOp::Mul, l, r)),
                SBin::Div(_) => Ok(Expr::Bin(BinOp::Div, l, r)),
                SBin::Rem(_) => Ok(Expr::Bin(BinOp::Rem, l, r)),
                SBin::And(_) => Ok(Expr::Bool(BoolOp::And, l, r)),
                SBin::Or(_) => Ok(Expr::Bool(BoolOp::Or, l, r)),
                SBin::Eq(_) => Ok(Expr::Cmp(CmpOp::Eq, l, r)),
                SBin::Ne(_) => Ok(Expr::Cmp(CmpOp::Ne, l, r)),
                SBin::Lt(_) => Ok(Expr::Cmp(CmpOp::Lt, l, r)),
                SBin::Le(_) => Ok(Expr::Cmp(CmpOp::Le, l, r)),
                SBin::Gt(_) => Ok(Expr::Cmp(CmpOp::Gt, l, r)),
                SBin::Ge(_) => Ok(Expr::Cmp(CmpOp::Ge, l, r)),
                _ => Err(DslError::Forbidden("bitwise/shift operator".into())),
            }
        }

        SExpr::Path(p) => {
            let name = path_ident(p).ok_or_else(|| DslError::Forbidden("path expression".into()))?;
            Ok(resolve_name(&name, ctx))
        }

        SExpr::Index(ix) => Ok(Expr::Index(
            Box::new(lower(&ix.expr, ctx)?),
            Box::new(lower(&ix.index, ctx)?),
        )),

        SExpr::Array(a) => {
            let mut items = Vec::with_capacity(a.elems.len());
            for el in &a.elems {
                items.push(lower(el, ctx)?);
            }
            Ok(Expr::List(items))
        }

        SExpr::If(ifx) => {
            if ifx.else_branch.is_none() {
                return Err(DslError::Forbidden("`if` without `else`".into()));
            }
            let cond = Box::new(lower(&ifx.cond, ctx)?);
            let then = Box::new(lower(block_tail(&ifx.then_branch)?, ctx)?);
            let (_, else_expr) = ifx.else_branch.as_ref().unwrap();
            let els = Box::new(lower_branch(else_expr, ctx)?);
            Ok(Expr::If(cond, then, els))
        }

        SExpr::Call(c) => {
            let name = match &*c.func {
                SExpr::Path(p) => {
                    path_ident(p).ok_or_else(|| DslError::Forbidden("call target".into()))?
                }
                _ => return Err(DslError::Forbidden("call target".into())),
            };
            let b = builtin_from(&name)?;
            let mut args = Vec::with_capacity(c.args.len());
            for a in &c.args {
                args.push(lower(a, ctx)?);
            }
            Ok(Expr::Call(b, args))
        }

        SExpr::MethodCall(m) => {
            // Only `<list>.contains(<x>)` is allowed.
            if m.method != "contains" {
                return Err(DslError::Forbidden(format!("method `.{}()`", m.method)));
            }
            if m.args.len() != 1 {
                return Err(DslError::Arity { builtin: "contains", got: m.args.len() });
            }
            let recv = lower(&m.receiver, ctx)?;
            let arg = lower(&m.args[0], ctx)?;
            Ok(Expr::Call(Builtin::Contains, vec![recv, arg]))
        }

        other => Err(DslError::Forbidden(format!("{} expression", expr_kind(other)))),
    }
}

fn lower_branch(e: &SExpr, ctx: &mut Ctx) -> Result<Expr, DslError> {
    match e {
        SExpr::Block(b) => lower(block_tail(&b.block)?, ctx),
        _ => lower(e, ctx),
    }
}

fn lower_lit(lit: &Lit) -> Result<Expr, DslError> {
    match lit {
        Lit::Int(i) => Ok(Expr::Const(Value::Int(
            i.base10_parse::<i64>().map_err(|e| DslError::Parse(e.to_string()))?,
        ))),
        Lit::Float(fl) => Ok(Expr::Const(Value::Float(
            fl.base10_parse::<f64>().map_err(|e| DslError::Parse(e.to_string()))?,
        ))),
        Lit::Bool(b) => Ok(Expr::Const(Value::Bool(b.value))),
        Lit::Str(s) => Ok(Expr::Const(Value::str(s.value()))),
        _ => Err(DslError::Forbidden("char/byte literal".into())),
    }
}

fn expr_kind(e: &SExpr) -> &'static str {
    match e {
        SExpr::Closure(_) => "closure",
        SExpr::Macro(_) => "macro",
        SExpr::While(_) => "while",
        SExpr::ForLoop(_) => "for",
        SExpr::Loop(_) => "loop",
        SExpr::Match(_) => "match",
        SExpr::Assign(_) => "assignment",
        SExpr::Range(_) => "range",
        SExpr::Reference(_) => "reference",
        SExpr::Cast(_) => "cast",
        SExpr::Struct(_) => "struct literal",
        SExpr::Block(_) => "block",
        SExpr::Field(_) => "field/attribute access",
        SExpr::Async(_) => "async",
        SExpr::Await(_) => "await",
        SExpr::Try(_) => "try",
        SExpr::Return(_) => "return",
        _ => "unsupported",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyspell_core::{env::VecEnv, eval::run, value::Value};

    fn ok(src: &str) -> Program {
        compile_rust(src).unwrap_or_else(|e| panic!("expected ok for `{src}`: {e}"))
    }
    fn err(src: &str) -> DslError {
        compile_rust(src).unwrap_err()
    }

    #[test]
    fn accepts_typical_programs() {
        ok("temp <= 28800");
        ok("let d = high - low; d <= 3600");
        ok("abs(drift) < 100.0");
        ok("!peers.contains(20)");
        ok("if distance > 1000 { 250 } else { 0 }");
        ok("count <= limit && distance < 50000");
        ok("readings[0] >= 1");
        ok("sum(readings) > 0");
        ok("before(seq, 10, 20)");
        ok("first(seq) == 10");
    }

    #[test]
    fn free_names_become_vars() {
        // free identifiers resolve at eval time, not compile time
        let p = ok("free_heap > 100000");
        let env = VecEnv::new().set("free_heap", 120_000i64);
        assert_eq!(run(&p, &env).unwrap(), Value::Bool(true));
    }

    #[test]
    fn rejects_unsafe() {
        assert!(matches!(err("std::process::exit(0)"), DslError::Forbidden(_)));
        assert!(matches!(err("loop {}"), DslError::Parse(_) | DslError::Forbidden(_)));
        assert!(matches!(err("if x > 1 { 1 }"), DslError::Forbidden(_)));
        assert!(matches!(err("x.abs()"), DslError::Forbidden(_)));
        assert!(matches!(err("|x| x"), DslError::Forbidden(_)));
    }
}
