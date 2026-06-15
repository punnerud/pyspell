//! PySpell front-ends: compile Rust- or Python-expression source to the shared
//! `pyspell-core` IR. These parsers (`syn`, `rustpython-parser`) are heavy and
//! host-only — they are the reason source is compiled on the host and only the
//! verified IR is shipped to the device.

mod lower;
pub mod rust_frontend;
#[cfg(feature = "python")]
pub mod py_frontend;

pub use rust_frontend::compile_rust;
#[cfg(feature = "python")]
pub use py_frontend::compile_python;

/// Which surface syntax a source string is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Rust,
    Python,
}

/// Compile source in the given language to a `Program`.
pub fn compile(
    src: &str,
    lang: Lang,
) -> Result<pyspell_core::ir::Program, pyspell_core::error::DslError> {
    match lang {
        Lang::Rust => compile_rust(src),
        #[cfg(feature = "python")]
        Lang::Python => compile_python(src),
        #[cfg(not(feature = "python"))]
        Lang::Python => Err(pyspell_core::error::DslError::Forbidden(
            "the Python front-end was disabled at build time (feature `python`)".into(),
        )),
    }
}

/// Guess the language from a file extension (`.rs` → Rust, `.py` → Python).
pub fn lang_from_extension(path: &str) -> Option<Lang> {
    if path.ends_with(".rs") {
        Some(Lang::Rust)
    } else if path.ends_with(".py") {
        Some(Lang::Python)
    } else {
        None
    }
}
