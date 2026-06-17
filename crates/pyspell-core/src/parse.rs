//! A tiny, dependency-free parser for the PySpell expression subset.
//!
//! Unlike `pyspell-lang` (which uses the heavy `syn` / `rustpython-parser`), this
//! is a hand-written recursive-descent parser small enough to run **on the
//! device** — a few kB of code, `no_std`, no allocation beyond the AST itself.
//! It accepts the same whitelisted subset both front-ends do, in either Python
//! or Rust surface syntax, and is deny-by-default: anything it doesn't recognize
//! is a `DslError`, never a panic.
//!
//! Having a parser here means "type code in a browser and run it on the ESP32"
//! works with nothing else installed, while still honoring "compile to AST before
//! running" — the AST is just built on the device by this small, safe parser.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::error::DslError;
use crate::ir::{
    BinOp, BoolOp, Builtin, CmpOp, Expr, LetBinding, Program, UnOp, Value, DEFAULT_MAX_STEPS,
};

/// Surface syntax to parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Python,
    Rust,
}

/// Parse source in the given language to a [`Program`].
pub fn parse(src: &str, lang: Lang) -> Result<Program, DslError> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0, lang, locals: Vec::new(), next_slot: 0, body: Vec::new() };
    let ret = p.program()?;
    Ok(Program { body: p.body, ret, n_locals: p.next_slot, max_steps: DEFAULT_MAX_STEPS })
}

// ---- tokens --------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ident(String),
    // keywords
    Let,
    If,
    Else,
    In,
    And,
    Or,
    Not,
    // punctuation / operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Dot,
    Assign,
    Semi,
}

fn unescape(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn lex(src: &str) -> Result<Vec<Tok>, DslError> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'0'..=b'9' => {
                let start = i;
                let mut is_float = false;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                    if b[i] == b'.' {
                        // a single decimal point makes it a float; reject "1.2.3"
                        if is_float {
                            return Err(DslError::Parse("malformed number".into()));
                        }
                        is_float = true;
                    }
                    i += 1;
                }
                let text = &src[start..i];
                if is_float {
                    let x: f64 = text.parse().map_err(|_| DslError::Parse("bad float".into()))?;
                    out.push(Tok::Float(x));
                } else {
                    let n: i64 = text.parse().map_err(|_| DslError::Parse("bad integer".into()))?;
                    out.push(Tok::Int(n));
                }
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let word = &src[start..i];
                out.push(match word {
                    "let" => Tok::Let,
                    "if" => Tok::If,
                    "else" => Tok::Else,
                    "in" => Tok::In,
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "not" => Tok::Not,
                    "true" | "True" => Tok::Bool(true),
                    "false" | "False" => Tok::Bool(false),
                    _ => Tok::Ident(word.to_string()),
                });
            }
            b'"' | b'\'' => {
                // scan to the matching unescaped quote (byte-wise is UTF-8 safe
                // since quote/backslash are ASCII), then unescape the slice.
                let quote = c;
                let mut j = i + 1;
                while j < b.len() {
                    match b[j] {
                        b'\\' => j += 2,
                        q if q == quote => break,
                        _ => j += 1,
                    }
                }
                if j >= b.len() {
                    return Err(DslError::Parse("unterminated string".into()));
                }
                out.push(Tok::Str(unescape(&src[i + 1..j])));
                i = j + 1;
            }
            _ => {
                // multi-char operators first
                let two = if i + 1 < b.len() { &src[i..i + 2] } else { "" };
                let tok = match two {
                    "==" => Some(Tok::EqEq),
                    "!=" => Some(Tok::Ne),
                    "<=" => Some(Tok::Le),
                    ">=" => Some(Tok::Ge),
                    "&&" => Some(Tok::AndAnd),
                    "||" => Some(Tok::OrOr),
                    // Python floor-division `//`: integer `/` already truncates,
                    // so accept `//` as an alias rather than rejecting it.
                    "//" => Some(Tok::Slash),
                    _ => None,
                };
                if let Some(t) = tok {
                    out.push(t);
                    i += 2;
                    continue;
                }
                let t = match c {
                    b'+' => Tok::Plus,
                    b'-' => Tok::Minus,
                    b'*' => Tok::Star,
                    b'/' => Tok::Slash,
                    b'%' => Tok::Percent,
                    b'<' => Tok::Lt,
                    b'>' => Tok::Gt,
                    b'!' => Tok::Bang,
                    b'(' => Tok::LParen,
                    b')' => Tok::RParen,
                    b'[' => Tok::LBracket,
                    b']' => Tok::RBracket,
                    b'{' => Tok::LBrace,
                    b'}' => Tok::RBrace,
                    b',' => Tok::Comma,
                    b'.' => Tok::Dot,
                    b'=' => Tok::Assign,
                    b';' => Tok::Semi,
                    other => {
                        return Err(DslError::Parse(alloc::format!(
                            "unexpected character `{}`",
                            other as char
                        )))
                    }
                };
                out.push(t);
                i += 1;
            }
        }
    }
    Ok(out)
}

// ---- parser --------------------------------------------------------------

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    lang: Lang,
    locals: Vec<(String, u16)>,
    next_slot: u16,
    body: Vec<LetBinding>,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn advance(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: &Tok, what: &str) -> Result<(), DslError> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(DslError::Parse(alloc::format!("expected {what}")))
        }
    }

    fn declare(&mut self, name: String) -> u16 {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.locals.push((name, slot));
        slot
    }
    fn resolve(&self, name: &str) -> Expr {
        // last binding wins (shadowing)
        for (n, slot) in self.locals.iter().rev() {
            if n == name {
                return Expr::Local(*slot);
            }
        }
        Expr::Var(name.to_string())
    }

    /// Top level: Rust = `let`s then a final expression; Python = one expression.
    fn program(&mut self) -> Result<Expr, DslError> {
        if self.lang == Lang::Rust {
            while self.peek() == Some(&Tok::Let) {
                self.advance();
                let name = match self.advance() {
                    Some(Tok::Ident(n)) => n,
                    _ => return Err(DslError::Parse("expected name after `let`".into())),
                };
                self.expect(&Tok::Assign, "`=`")?;
                let e = self.conditional()?;
                self.expect(&Tok::Semi, "`;`")?;
                let slot = self.declare(name);
                self.body.push(LetBinding { slot, expr: e });
            }
        }
        let ret = self.conditional()?;
        // trailing `;` tolerated in Rust mode
        self.eat(&Tok::Semi);
        if self.pos != self.toks.len() {
            return Err(DslError::Parse("trailing tokens after expression".into()));
        }
        Ok(ret)
    }

    /// Python ternary `body if test else orelse` sits below `or`.
    fn conditional(&mut self) -> Result<Expr, DslError> {
        let e = self.or_expr()?;
        if self.lang == Lang::Python && self.peek() == Some(&Tok::If) {
            self.advance();
            let test = self.or_expr()?;
            self.expect(&Tok::Else, "`else` in conditional expression")?;
            let orelse = self.conditional()?;
            return Ok(Expr::If(Box::new(test), Box::new(e), Box::new(orelse)));
        }
        Ok(e)
    }

    fn or_expr(&mut self) -> Result<Expr, DslError> {
        let mut left = self.and_expr()?;
        while matches!(self.peek(), Some(Tok::Or) | Some(Tok::OrOr)) {
            self.advance();
            let right = self.and_expr()?;
            left = Expr::Bool(BoolOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, DslError> {
        let mut left = self.not_expr()?;
        while matches!(self.peek(), Some(Tok::And) | Some(Tok::AndAnd)) {
            self.advance();
            let right = self.not_expr()?;
            left = Expr::Bool(BoolOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr, DslError> {
        if matches!(self.peek(), Some(Tok::Not) | Some(Tok::Bang)) {
            self.advance();
            return Ok(Expr::Unary(UnOp::Not, Box::new(self.not_expr()?)));
        }
        self.cmp_expr()
    }

    /// Comparison, with Python-style chaining (`a < b < c`) and `in` / `not in`.
    fn cmp_expr(&mut self) -> Result<Expr, DslError> {
        let first = self.add_expr()?;
        let mut operands = alloc::vec![first];
        let mut ops: Vec<CmpKind> = Vec::new();
        loop {
            let kind = match self.peek() {
                Some(Tok::EqEq) => CmpKind::Cmp(CmpOp::Eq),
                Some(Tok::Ne) => CmpKind::Cmp(CmpOp::Ne),
                Some(Tok::Lt) => CmpKind::Cmp(CmpOp::Lt),
                Some(Tok::Le) => CmpKind::Cmp(CmpOp::Le),
                Some(Tok::Gt) => CmpKind::Cmp(CmpOp::Gt),
                Some(Tok::Ge) => CmpKind::Cmp(CmpOp::Ge),
                Some(Tok::In) => CmpKind::In,
                Some(Tok::Not) => {
                    // `not in`
                    if self.toks.get(self.pos + 1) == Some(&Tok::In) {
                        self.advance(); // not
                        CmpKind::NotIn
                    } else {
                        break;
                    }
                }
                _ => break,
            };
            self.advance(); // the comparison/in token
            operands.push(self.add_expr()?);
            ops.push(kind);
        }
        if ops.is_empty() {
            return Ok(operands.pop().unwrap());
        }
        let mut terms: Vec<Expr> = Vec::with_capacity(ops.len());
        for (i, k) in ops.iter().enumerate() {
            let lhs = operands[i].clone();
            let rhs = operands[i + 1].clone();
            terms.push(match k {
                CmpKind::Cmp(op) => Expr::Cmp(*op, Box::new(lhs), Box::new(rhs)),
                // `x in list` → contains(list, x)
                CmpKind::In => Expr::Call(Builtin::Contains, alloc::vec![rhs, lhs]),
                CmpKind::NotIn => Expr::Unary(
                    UnOp::Not,
                    Box::new(Expr::Call(Builtin::Contains, alloc::vec![rhs, lhs])),
                ),
            });
        }
        let mut it = terms.into_iter();
        let mut acc = it.next().unwrap();
        for t in it {
            acc = Expr::Bool(BoolOp::And, Box::new(acc), Box::new(t));
        }
        Ok(acc)
    }

    fn add_expr(&mut self) -> Result<Expr, DslError> {
        let mut left = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => BinOp::Add,
                Some(Tok::Minus) => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.mul_expr()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> Result<Expr, DslError> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => BinOp::Mul,
                Some(Tok::Slash) => BinOp::Div,
                Some(Tok::Percent) => BinOp::Rem,
                _ => break,
            };
            self.advance();
            let right = self.unary()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, DslError> {
        if self.peek() == Some(&Tok::Minus) {
            self.advance();
            return Ok(Expr::Unary(UnOp::Neg, Box::new(self.unary()?)));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, DslError> {
        let mut e = self.primary()?;
        loop {
            match self.peek() {
                Some(Tok::LBracket) => {
                    self.advance();
                    let idx = self.conditional()?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    e = Expr::Index(Box::new(e), Box::new(idx));
                }
                Some(Tok::Dot) => {
                    self.advance();
                    let method = match self.advance() {
                        Some(Tok::Ident(n)) => n,
                        _ => return Err(DslError::Forbidden("method call".into())),
                    };
                    if method != "contains" {
                        return Err(DslError::Forbidden("only `.contains()` is allowed".into()));
                    }
                    self.expect(&Tok::LParen, "`(`")?;
                    let arg = self.conditional()?;
                    self.expect(&Tok::RParen, "`)`")?;
                    e = Expr::Call(Builtin::Contains, alloc::vec![e, arg]);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> Result<Expr, DslError> {
        match self.advance() {
            Some(Tok::Int(n)) => Ok(Expr::Const(Value::Int(n))),
            Some(Tok::Float(x)) => Ok(Expr::Const(Value::Float(x))),
            Some(Tok::Bool(b)) => Ok(Expr::Const(Value::Bool(b))),
            Some(Tok::Str(s)) => Ok(Expr::Const(Value::str(&s))),
            Some(Tok::LParen) => {
                let e = self.conditional()?;
                self.expect(&Tok::RParen, "`)`")?;
                Ok(e)
            }
            Some(Tok::LBracket) => {
                let mut items = Vec::new();
                if self.peek() != Some(&Tok::RBracket) {
                    loop {
                        items.push(self.conditional()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                        if self.peek() == Some(&Tok::RBracket) {
                            break; // trailing comma
                        }
                    }
                }
                self.expect(&Tok::RBracket, "`]`")?;
                Ok(Expr::List(items))
            }
            Some(Tok::If) if self.lang == Lang::Rust => self.rust_if(),
            Some(Tok::Ident(name)) => {
                if self.peek() == Some(&Tok::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    if self.peek() != Some(&Tok::RParen) {
                        loop {
                            args.push(self.conditional()?);
                            if !self.eat(&Tok::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect(&Tok::RParen, "`)`")?;
                    let b = builtin_from(&name)?;
                    Ok(Expr::Call(b, args))
                } else {
                    Ok(self.resolve(&name))
                }
            }
            Some(t) => Err(DslError::Parse(alloc::format!("unexpected token {t:?}"))),
            None => Err(DslError::Parse("unexpected end of input".into())),
        }
    }

    /// Rust `if cond { a } else { b }` (with `else if` chains).
    fn rust_if(&mut self) -> Result<Expr, DslError> {
        let cond = self.or_expr()?;
        self.expect(&Tok::LBrace, "`{` after `if`")?;
        let then = self.conditional()?;
        self.expect(&Tok::RBrace, "`}`")?;
        self.expect(&Tok::Else, "`else` (if without else is not allowed)")?;
        let els = if self.peek() == Some(&Tok::If) {
            self.advance();
            self.rust_if()?
        } else {
            self.expect(&Tok::LBrace, "`{` after `else`")?;
            let e = self.conditional()?;
            self.expect(&Tok::RBrace, "`}`")?;
            e
        };
        Ok(Expr::If(Box::new(cond), Box::new(then), Box::new(els)))
    }
}

enum CmpKind {
    Cmp(CmpOp),
    In,
    NotIn,
}

fn builtin_from(name: &str) -> Result<Builtin, DslError> {
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
        "str" => Builtin::Str,
        "json_get" => Builtin::JsonGet,
        "fetch" => Builtin::Fetch,
        "fetch_json" => Builtin::FetchJson,
        "show" => Builtin::Show,
        "print" => Builtin::Print,
        _ => return Err(DslError::Forbidden(alloc::format!("function `{name}()`"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::VecEnv;
    use crate::eval::run;

    fn run_py(src: &str, env: &VecEnv) -> Value {
        run(&parse(src, Lang::Python).unwrap(), env).unwrap()
    }
    fn run_rs(src: &str, env: &VecEnv) -> Value {
        run(&parse(src, Lang::Rust).unwrap(), env).unwrap()
    }

    #[test]
    fn python_subset() {
        let env = VecEnv::new().set("free_heap", 120_000i64).set("uptime_s", 30i64);
        assert_eq!(run_py("1 + 2 * 3", &env), Value::Int(7));
        assert_eq!(run_py("free_heap > 100000 and uptime_s < 60", &env), Value::Bool(true));
        assert_eq!(run_py("0 < uptime_s < 60", &env), Value::Bool(true)); // chained
        assert_eq!(run_py("250 if free_heap < 1000 else 0", &env), Value::Int(0));
        assert_eq!(run_py("3 not in [1, 2, 4]", &env), Value::Bool(true));
        assert_eq!(run_py("sum([1, 2, 3])", &env), Value::Int(6));
        assert_eq!(run_py("[10, 20, 30][-1]", &env), Value::Int(30));
        assert_eq!(run_py("max(uptime_s, 100)", &env), Value::Int(100));
        // `//` accepted as an alias for integer `/` (both truncate on ints).
        assert_eq!(run_py("7 // 2", &env), Value::Int(3));
    }

    #[test]
    fn rust_subset() {
        let env = VecEnv::new().set("total", 320_000i64).set("free", 80_000i64);
        assert_eq!(run_rs("let used = total - free; used * 100 / total", &env), Value::Int(75));
        assert_eq!(run_rs("free > 1000 && total < 1000000", &env), Value::Bool(true));
        assert_eq!(run_rs("if free > 1000 { 1 } else { 0 }", &env), Value::Int(1));
        assert_eq!(run_rs("!false", &env), Value::Bool(true));
        assert_eq!(run_rs("[1, 2, 3].contains(2)", &env), Value::Bool(true));
    }

    #[test]
    fn strings_and_json() {
        let env = VecEnv::new();
        // string literals + concatenation + comparison
        assert_eq!(run_py("\"ab\" + \"c\"", &env), Value::str("abc"));
        assert_eq!(run_py("'oslo' == 'oslo'", &env), Value::Bool(true));
        assert_eq!(run_py("len('hello')", &env), Value::Int(5));
        // json_get over a literal document
        let doc = r#"json_get('{"a":{"b":[10,20,30]}}', 'a.b.1')"#;
        assert_eq!(run_py(doc, &env), Value::Int(20));
    }

    #[test]
    fn rejects_unsafe() {
        assert!(matches!(parse("foo()", Lang::Python), Err(DslError::Forbidden(_))));
        assert!(matches!(parse("x.bar()", Lang::Python), Err(DslError::Forbidden(_))));
        assert!(matches!(parse("1 +", Lang::Python), Err(DslError::Parse(_))));
        assert!(matches!(parse("1 2 3", Lang::Python), Err(DslError::Parse(_))));
    }

    #[test]
    fn free_names_become_vars() {
        let p = parse("a + b", Lang::Python).unwrap();
        let env = VecEnv::new().set("a", 2i64).set("b", 5i64);
        assert_eq!(run(&p, &env).unwrap(), Value::Int(7));
    }
}
