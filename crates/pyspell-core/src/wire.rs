//! Wire format for shipping a compiled [`Program`] from the host to the device.
//!
//! The host parses + lowers source to a `Program` (the AST/IR), serializes it
//! here with `postcard` (compact, `no_std`-friendly), and streams the bytes to
//! the ESP32. The device deserializes and evaluates — it never sees source, so
//! the parser (and its attack surface) stays entirely on the host. This is the
//! MicroPython-like "push code live" path, minus the on-device parser.
//!
//! Framing on a byte stream (e.g. USB-serial) is `[u32-le len][postcard bytes]`;
//! see [`frame`] / [`MAX_FRAME`]. The reply a device sends back is up to the
//! firmware, but [`encode_value`] / [`decode_value`] give a matching codec.

use alloc::vec::Vec;

use crate::error::DslError;
use crate::ir::Program;
use crate::value::Value;

/// A sane upper bound on a single program frame (64 KiB). The device rejects
/// anything larger before allocating, so a corrupt length can't OOM it.
pub const MAX_FRAME: usize = 64 * 1024;

/// Serialize a program to postcard bytes.
pub fn to_bytes(program: &Program) -> Result<Vec<u8>, DslError> {
    postcard::to_allocvec(program).map_err(|e| DslError::Wire(wire_msg(e)))
}

/// Deserialize a program from postcard bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<Program, DslError> {
    postcard::from_bytes(bytes).map_err(|e| DslError::Wire(wire_msg(e)))
}

/// Wrap program bytes in a length-prefixed frame ready to write to a stream.
pub fn frame(program: &Program) -> Result<Vec<u8>, DslError> {
    let body = to_bytes(program)?;
    let mut out = Vec::with_capacity(body.len() + 4);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Encode an evaluation result `Value` for the reply channel.
pub fn encode_value(v: &Value) -> Result<Vec<u8>, DslError> {
    postcard::to_allocvec(v).map_err(|e| DslError::Wire(wire_msg(e)))
}

/// Decode an evaluation result `Value` from the reply channel.
pub fn decode_value(bytes: &[u8]) -> Result<Value, DslError> {
    postcard::from_bytes(bytes).map_err(|e| DslError::Wire(wire_msg(e)))
}

fn wire_msg(e: postcard::Error) -> alloc::string::String {
    use core::fmt::Write;
    let mut s = alloc::string::String::new();
    let _ = write!(s, "{e}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::VecEnv;
    use crate::eval::run;
    use crate::ir::{BinOp, CmpOp, Expr, DEFAULT_MAX_STEPS};
    use alloc::boxed::Box;
    use alloc::vec;

    #[test]
    fn round_trip_serialize_eval() {
        // (x + 1) <= 10
        let p = Program {
            body: vec![],
            ret: Expr::Cmp(
                CmpOp::Le,
                Box::new(Expr::Bin(
                    BinOp::Add,
                    Box::new(Expr::Var("x".into())),
                    Box::new(Expr::Const(Value::Int(1))),
                )),
                Box::new(Expr::Const(Value::Int(10))),
            ),
            n_locals: 0,
            max_steps: DEFAULT_MAX_STEPS,
        };
        let bytes = to_bytes(&p).unwrap();
        let back = from_bytes(&bytes).unwrap();
        let env = VecEnv::new().set("x", 4i64);
        assert_eq!(run(&back, &env).unwrap(), Value::Bool(true));

        // Value codec round-trips too.
        let v = Value::list([Value::Int(1), Value::Int(2)]);
        assert_eq!(decode_value(&encode_value(&v).unwrap()).unwrap(), v);
    }
}
