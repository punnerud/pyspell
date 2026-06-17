//! `pyspell` — the host CLI.
//!
//! Compiles Rust/Python source to PySpell IR, runs it locally, or pushes it
//! live to a device (MicroPython-like) over USB-serial. The parser only ever
//! runs here; the device receives verified IR.

use std::io::Read;
#[allow(unused_imports)]
use std::io::Write; // brought in scope for write_all/flush on the serial link
use std::time::Duration;

use clap::{Parser, Subcommand};
use pyspell_core::{env::VecEnv, eval, value::Value, wire, Display, DslError, Limits, Net};
use pyspell_lang::{compile, lang_from_extension, Lang};

#[derive(Parser)]
#[command(name = "pyspell", about = "Compile Rust/Python to a sandboxed IR and run it on host or ESP32")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compile a source file to a `.psb` IR blob (the wire format).
    Compile {
        /// Source file (.rs or .py), or `-` for stdin (use --lang then).
        file: String,
        /// Output path (defaults to <file>.psb).
        #[arg(short, long)]
        out: Option<String>,
        /// Force the language instead of guessing from the extension.
        #[arg(long, value_enum)]
        lang: Option<LangArg>,
    },
    /// Compile and evaluate locally, printing the result value.
    Run {
        /// Source file (.rs or .py).
        file: String,
        /// Bind a free variable, e.g. `--set free_heap=120000`. Repeatable.
        #[arg(long = "set", value_name = "NAME=VALUE")]
        sets: Vec<String>,
        /// Allow `fetch()` to reach this host (e.g. `api.met.no`). Repeatable;
        /// subdomains of an allowed host are permitted. Empty = no network.
        #[arg(long = "allow-host", value_name = "HOST")]
        allow_hosts: Vec<String>,
        #[arg(long, value_enum)]
        lang: Option<LangArg>,
    },
    /// Compile and push one program to a device, printing the device's result.
    Push {
        file: String,
        /// Serial port, e.g. /dev/cu.usbmodem2101.
        #[arg(short, long)]
        port: String,
        #[arg(short, long, default_value_t = 115200)]
        baud: u32,
        #[arg(long, value_enum)]
        lang: Option<LangArg>,
    },
    /// Interactive REPL: type expressions, each is compiled and pushed live.
    Repl {
        #[arg(short, long)]
        port: String,
        #[arg(short, long, default_value_t = 115200)]
        baud: u32,
        /// Language for typed lines (default: python).
        #[arg(long, value_enum, default_value_t = LangArg::Python)]
        lang: LangArg,
    },
    /// List available serial ports.
    Ports,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum LangArg {
    Rust,
    Python,
}

impl From<LangArg> for Lang {
    fn from(l: LangArg) -> Lang {
        match l {
            LangArg::Rust => Lang::Rust,
            LangArg::Python => Lang::Python,
        }
    }
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Compile { file, out, lang } => {
            let (src, lang) = read_source(&file, lang)?;
            let program = compile(&src, lang).map_err(|e| e.to_string())?;
            let bytes = wire::to_bytes(&program).map_err(|e| e.to_string())?;
            let out = out.unwrap_or_else(|| format!("{file}.psb"));
            std::fs::write(&out, &bytes).map_err(|e| e.to_string())?;
            println!("compiled {file} -> {out} ({} bytes IR)", bytes.len());
            Ok(())
        }
        Cmd::Run { file, sets, allow_hosts, lang } => {
            let (src, lang) = read_source(&file, lang)?;
            let program = compile(&src, lang).map_err(|e| e.to_string())?;
            let env = build_env(&sets)?;
            let net = HttpNet { allow: allow_hosts };
            let disp = StdoutDisplay;
            let limits = Limits {
                max_steps: program.max_steps,
                max_bytes: u32::MAX,
                deadline: None,
                net: Some(&net),
                display: Some(&disp),
                actuator: None,
            };
            let value = eval::run_with(&program, &env, limits).map_err(|e| e.to_string())?;
            println!("{}", show(&value));
            Ok(())
        }
        Cmd::Push { file, port, baud, lang } => {
            let (src, lang) = read_source(&file, lang)?;
            let program = compile(&src, lang).map_err(|e| e.to_string())?;
            let mut link = open_port(&port, baud)?;
            let value = push_program(&mut *link, &program)?;
            println!("{}", show(&value));
            Ok(())
        }
        Cmd::Repl { port, baud, lang } => repl(&port, baud, lang.into()),
        Cmd::Ports => {
            let ports = serialport::available_ports().map_err(|e| e.to_string())?;
            if ports.is_empty() {
                println!("(no serial ports found)");
            }
            for p in ports {
                println!("{}", p.port_name);
            }
            Ok(())
        }
    }
}

fn read_source(file: &str, lang: Option<LangArg>) -> Result<(String, Lang), String> {
    let src = if file == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).map_err(|e| e.to_string())?;
        s
    } else {
        std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?
    };
    let lang = lang
        .map(Lang::from)
        .or_else(|| lang_from_extension(file))
        .ok_or_else(|| "cannot determine language; pass --lang rust|python".to_string())?;
    Ok((src, lang))
}

fn build_env(sets: &[String]) -> Result<VecEnv, String> {
    let mut env = VecEnv::new();
    for s in sets {
        let (name, raw) = s
            .split_once('=')
            .ok_or_else(|| format!("--set expects NAME=VALUE, got `{s}`"))?;
        env.insert(name.trim(), parse_value(raw.trim()));
    }
    Ok(env)
}

/// Parse a CLI scalar: int, else float, else bool, else treated as the integer 0
/// with a note would be confusing — instead we error.
fn parse_value(raw: &str) -> Value {
    if let Ok(n) = raw.parse::<i64>() {
        Value::Int(n)
    } else if let Ok(x) = raw.parse::<f64>() {
        Value::Float(x)
    } else if raw == "true" {
        Value::Bool(true)
    } else if raw == "false" {
        Value::Bool(false)
    } else {
        // Last resort: comma-separated integer list, e.g. "1,2,3".
        let parts: Vec<&str> = raw.split(',').collect();
        if parts.len() > 1 && parts.iter().all(|p| p.trim().parse::<i64>().is_ok()) {
            Value::list(parts.iter().map(|p| Value::Int(p.trim().parse().unwrap())))
        } else {
            // Anything else is a string value.
            Value::str(raw)
        }
    }
}

fn show(v: &Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(x) => x.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => s.to_string(),
        Value::List(l) => {
            let inner: Vec<String> = l.iter().map(show).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

// ---- host network capability (for `fetch` in `run`) ----------------------

/// Host-side `fetch` over HTTP(S), gated by a host allowlist. Mirrors what the
/// device enforces from its config — the allowlist is policy, kept out of the
/// pure evaluator.
struct HttpNet {
    allow: Vec<String>,
}

impl Net for HttpNet {
    fn fetch(&self, url: &str) -> Result<String, DslError> {
        let host = url_host(url);
        let ok = self
            .allow
            .iter()
            .any(|h| host == *h || host.ends_with(&format!(".{h}")));
        if !ok {
            return Err(DslError::Net(format!(
                "host `{host}` not allowed (use --allow-host {host})"
            )));
        }
        let resp = ureq::get(url)
            .set("User-Agent", "pyspell/0.1 (github.com/punnerud/pyspell)")
            .call()
            .map_err(|e| DslError::Net(format!("{e}")))?;
        resp.into_string().map_err(|e| DslError::Net(format!("read body: {e}")))
    }
}

/// Host `show()` target: just print to stdout (the device draws on its screen).
struct StdoutDisplay;
impl Display for StdoutDisplay {
    fn show(&self, text: &str) -> Result<(), DslError> {
        println!("[show] {text}");
        Ok(())
    }
}

/// Crude host extraction: strip scheme, take up to the first `/` or `:`.
fn url_host(url: &str) -> String {
    let after = url.split("://").nth(1).unwrap_or(url);
    after.split(['/', ':']).next().unwrap_or("").to_string()
}

// ---- device link ---------------------------------------------------------

fn open_port(port: &str, baud: u32) -> Result<Box<dyn serialport::SerialPort>, String> {
    serialport::new(port, baud)
        .timeout(Duration::from_secs(5))
        .open()
        .map_err(|e| format!("open {port}: {e}"))
}

/// Push a compiled program and read back the device's reply value.
///
/// Line protocol (matches the firmware):
///   host → device:  `<hex of postcard Program>\n`
///   device → host:  `OK <hex of postcard Value>` or `ERR <message>`
/// Any other line (boot logs, `READY …`) is skipped, so logging never breaks us.
fn push_program(
    link: &mut dyn serialport::SerialPort,
    program: &pyspell_core::ir::Program,
) -> Result<Value, String> {
    let bytes = wire::to_bytes(program).map_err(|e| e.to_string())?;
    let mut request = hex_encode(&bytes);
    request.push('\n');
    link.write_all(request.as_bytes()).map_err(|e| e.to_string())?;
    link.flush().map_err(|e| e.to_string())?;

    let line = read_reply_line(link)?;
    if let Some(rest) = line.strip_prefix("OK ") {
        let payload = hex_decode(rest.trim()).ok_or("device sent malformed hex")?;
        wire::decode_value(&payload).map_err(|e| e.to_string())
    } else if let Some(msg) = line.strip_prefix("ERR ") {
        Err(format!("device: {}", msg.trim()))
    } else {
        Err(format!("unexpected device reply: {line}"))
    }
}

/// Read lines until one starts with `OK ` or `ERR `, discarding logs/`READY`.
fn read_reply_line(link: &mut dyn serialport::SerialPort) -> Result<String, String> {
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match link.read(&mut byte) {
            Ok(0) => return Err("device closed the connection".into()),
            Ok(_) => {
                if byte[0] == b'\n' || byte[0] == b'\r' {
                    if line.is_empty() {
                        continue;
                    }
                    let s = String::from_utf8_lossy(&line).to_string();
                    if s.starts_with("OK ") || s.starts_with("ERR ") {
                        return Ok(s);
                    }
                    line.clear(); // a log/READY line — keep waiting
                } else {
                    line.push(byte[0]);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return Err("timed out waiting for device reply".into())
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.as_bytes();
    if s.len() % 2 != 0 {
        return None;
    }
    let val = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < s.len() {
        out.push((val(s[i])? << 4) | val(s[i + 1])?);
        i += 2;
    }
    Some(out)
}

fn repl(port: &str, baud: u32, lang: Lang) -> Result<(), String> {
    let mut link = open_port(port, baud)?;
    let mut rl = rustyline::DefaultEditor::new().map_err(|e| e.to_string())?;
    let prompt = match lang {
        Lang::Rust => "pyspell(rust)> ",
        Lang::Python => "pyspell(py)> ",
    };
    println!("Connected to {port}. Type an expression; Ctrl-D to quit.");
    loop {
        match rl.readline(prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(line);
                match compile(line, lang) {
                    Ok(program) => match push_program(&mut *link, &program) {
                        Ok(v) => println!("{}", show(&v)),
                        Err(e) => eprintln!("error: {e}"),
                    },
                    Err(e) => eprintln!("compile error: {e}"),
                }
            }
            Err(rustyline::error::ReadlineError::Eof)
            | Err(rustyline::error::ReadlineError::Interrupted) => break,
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
