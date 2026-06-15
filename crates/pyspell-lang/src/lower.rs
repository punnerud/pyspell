//! Front-end-independent lowering helpers shared by the Rust and Python
//! front-ends: the local scope and the builtin table. Keeping these here
//! guarantees both syntaxes lower to exactly the same IR.

use std::collections::HashMap;

use pyspell_core::error::DslError;
use pyspell_core::ir::{Builtin, Expr, LetBinding, Program, DEFAULT_MAX_STEPS};

pub(crate) struct Ctx {
    pub locals: HashMap<String, u16>,
    pub next_slot: u16,
    pub body: Vec<LetBinding>,
}

impl Ctx {
    pub fn new() -> Self {
        Ctx { locals: HashMap::new(), next_slot: 0, body: Vec::new() }
    }
    pub fn declare(&mut self, name: String) -> u16 {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.locals.insert(name, slot);
        slot
    }
}

/// Resolve a bare identifier: a `let`-bound local becomes [`Expr::Local`];
/// anything else becomes a free [`Expr::Var`] resolved against the host env at
/// eval time. This is the generalization of the old fixed `route.*` schema.
pub(crate) fn resolve_name(name: &str, ctx: &Ctx) -> Expr {
    match ctx.locals.get(name) {
        Some(&slot) => Expr::Local(slot),
        None => Expr::Var(name.to_string()),
    }
}

pub(crate) fn builtin_from(name: &str) -> Result<Builtin, DslError> {
    Ok(match name {
        "len" => Builtin::Len,
        "abs" => Builtin::Abs,
        "min" => Builtin::Min,
        "max" => Builtin::Max,
        "sum" => Builtin::Sum,
        "any" => Builtin::Any,
        "all" => Builtin::All,
        "round" => Builtin::Round,
        "int" => Builtin::Int,
        "float" => Builtin::Float,
        "bool" => Builtin::Bool,
        "index" => Builtin::IndexOf,
        "before" => Builtin::Before,
        "first" => Builtin::First,
        "last" => Builtin::Last,
        _ => return Err(DslError::Forbidden(format!("function `{name}()`"))),
    })
}

/// Assemble a [`Program`] from a finished context and return expression.
pub(crate) fn finish(ctx: Ctx, ret: Expr) -> Program {
    Program { body: ctx.body, ret, n_locals: ctx.next_slot, max_steps: DEFAULT_MAX_STEPS }
}
