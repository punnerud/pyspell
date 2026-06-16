//! PySpell web add-on for the in-tunnel HTTP server.
//!
//! This is the ESP-specific glue that ties the two clean dependencies together:
//! `tailscale-core` (networking) calls [`route`] via a `fn` pointer, and `route`
//! parses + evaluates the submitted code with `pyspell-core`, supplying the ESP
//! wall clock so a request can set a real timeout (e.g. 10 s).
//!
//! Routes:
//! * `GET /`     → a tiny single-segment web page (text box + run button).
//! * `GET  /run?lang=py|rs&timeout=<s>&code=<urlencoded>` → eval, `text/plain` result.
//! * `POST /run?lang=py|rs&timeout=<s>` with the program as the raw request body
//!   → same result. POST avoids URL-encoding overhead and URL length limits, so
//!   it fits more code in the single request segment.
//! Any other path returns `None`, falling back to the built-in control panel.

use esp_idf_svc::sys::esp_timer_get_time;

use pyspell_core::{eval, parse, value::Value, Lang, Limits, VecEnv};
use tailscale_core::tcp::HttpReply;

/// Largest accepted program (URL-decoded). Keeps a single request segment bounded.
const MAX_CODE: usize = 1024;

pub fn route(method: &str, path: &str, query: &str, body: &[u8]) -> Option<HttpReply> {
    match path {
        "/" => Some(HttpReply {
            content_type: "text/html; charset=utf-8",
            body: PAGE.as_bytes().to_vec(),
        }),
        "/run" => Some(HttpReply {
            content_type: "text/plain; charset=utf-8",
            body: run(method, query, body).into_bytes(),
        }),
        // MCP server: JSON-RPC 2.0 over HTTP (Streamable HTTP, stateless). Any MCP
        // agent can point at http://<tailscale-ip>/mcp and call run_pyspell — the
        // microcontroller itself speaks MCP.
        "/mcp" => Some(HttpReply {
            content_type: "application/json",
            body: mcp_handle(body).into_bytes(),
        }),
        _ => None,
    }
}

fn run(method: &str, query: &str, body: &[u8]) -> String {
    let lang = match query_get(query, "lang").as_str() {
        "rs" | "rust" => Lang::Rust,
        _ => Lang::Python,
    };
    let timeout_s = query_get(query, "timeout").parse::<i64>().unwrap_or(10);
    // POST → the raw request body is the program; GET → the URL-encoded `code` param.
    let code = if method.eq_ignore_ascii_case("POST") {
        String::from_utf8_lossy(body).trim().to_string()
    } else {
        url_decode(&query_get(query, "code"))
    };
    eval_program(&code, lang, timeout_s)
}

/// Parse + evaluate a PySpell program with the device's env/net/display and a
/// wall-clock deadline; returns the value (or `error: ...`) as a string. Shared by
/// the `/run` POST API and the MCP `run_pyspell` tool — and the single choke point
/// that records each job for the display counters.
fn eval_program(code: &str, lang: Lang, timeout_s: i64) -> String {
    if code.is_empty() {
        return "error: empty program".into();
    }
    if code.len() > MAX_CODE {
        return "error: program too long".into();
    }
    let program = match parse(code, lang) {
        Ok(p) => p,
        Err(e) => return format!("error: {e}"),
    };

    // Wall-clock deadline using the ESP timer (microseconds since boot).
    let start = unsafe { esp_timer_get_time() };
    let budget_us = timeout_s.clamp(1, 60) * 1_000_000;
    let deadline = move || unsafe { esp_timer_get_time() } - start > budget_us;

    let env = device_env();
    let net = crate::net::DeviceNet;
    let disp = crate::display::DeviceDisplay;
    let limits = Limits {
        max_steps: 2_000_000,
        deadline: Some(&deadline),
        net: Some(&net),
        display: Some(&disp),
    };
    let out = match eval::run_with(&program, &env, limits) {
        Ok(v) => show(&v),
        Err(e) => format!("error: {e}"),
    };
    crate::jobcount::record();
    out
}

// --- MCP server (JSON-RPC 2.0 over HTTP) -----------------------------------
// Minimal, stateless Model Context Protocol server so any MCP-capable agent can
// run PySpell live on this microcontroller. Handles initialize / tools/list /
// tools/call(run_pyspell); notifications get an empty 200.

fn mcp_handle(body: &[u8]) -> String {
    let req: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return jsonrpc_err(&serde_json::Value::Null, -32700, "parse error"),
    };
    let id = req.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

    match method {
        "initialize" => {
            let pv = req
                .get("params")
                .and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str())
                .unwrap_or("2024-11-05")
                .to_string();
            jsonrpc_ok(
                &id,
                serde_json::json!({
                    "protocolVersion": pv,
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "pyspell-esp32", "version": "0.1.0" }
                }),
            )
        }
        "tools/list" => jsonrpc_ok(
            &id,
            serde_json::json!({
                "tools": [{
                    "name": "run_pyspell",
                    "description": "Compile and run a sandboxed PySpell program (a Python/Rust expression subset) live on this ESP32, against live device state. Returns the evaluated value as text. Allowed: arithmetic, comparisons, bool, ternary, lists, strings, builtins (len,abs,min,max,sum,round,int,float,str), fetch_json(url,\"a.b.0.c\"), json_get, and free vars like free_heap, uptime_s. Not allowed: def, loops, imports, assignment.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "code": { "type": "string", "description": "PySpell program: let-bindings + one expression, e.g. free_heap > 100000" },
                            "lang": { "type": "string", "enum": ["py", "rs"], "description": "Syntax, default py" },
                            "timeout": { "type": "integer", "description": "Wall-clock seconds 1-60, default 10" }
                        },
                        "required": ["code"]
                    }
                }]
            }),
        ),
        "tools/call" => {
            let params = req.get("params");
            let name = params
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if name != "run_pyspell" {
                return jsonrpc_err(&id, -32602, "unknown tool");
            }
            let args = params.and_then(|p| p.get("arguments"));
            let code = args
                .and_then(|a| a.get("code"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let lang = match args.and_then(|a| a.get("lang")).and_then(|v| v.as_str()) {
                Some("rs") | Some("rust") => Lang::Rust,
                _ => Lang::Python,
            };
            let timeout = args
                .and_then(|a| a.get("timeout"))
                .and_then(|v| v.as_i64())
                .unwrap_or(10);
            let result = eval_program(code, lang, timeout);
            jsonrpc_ok(
                &id,
                serde_json::json!({
                    "content": [{ "type": "text", "text": result }],
                    "isError": result.starts_with("error:")
                }),
            )
        }
        // Notifications (no id, e.g. notifications/initialized): no response body.
        _ if id.is_null() => String::new(),
        _ => jsonrpc_err(&id, -32601, "method not found"),
    }
}

fn jsonrpc_ok(id: &serde_json::Value, result: serde_json::Value) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()
}

fn jsonrpc_err(id: &serde_json::Value, code: i64, message: &str) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        .to_string()
}

/// Live device variables a program may read.
fn device_env() -> VecEnv {
    let (free, min_free, uptime_us) = unsafe {
        (
            esp_idf_svc::sys::esp_get_free_heap_size() as i64,
            esp_idf_svc::sys::esp_get_minimum_free_heap_size() as i64,
            esp_timer_get_time(),
        )
    };
    VecEnv::new()
        .set("free_heap", free)
        .set("min_free_heap", min_free)
        .set("uptime_ms", uptime_us / 1000)
        .set("uptime_s", uptime_us / 1_000_000)
}

fn show(v: &Value) -> String {
    match v {
        Value::Int(n) => format!("{n}"),
        Value::Float(x) => format!("{x}"),
        Value::Bool(b) => format!("{b}"),
        Value::Str(s) => s.to_string(),
        Value::List(l) => {
            let mut s = String::from("[");
            for (i, it) in l.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&show(it));
            }
            s.push(']');
            s
        }
    }
}

// ---- tiny query helpers (self-contained; tailscale-core's are private) ----

fn query_get(query: &str, key: &str) -> String {
    for kv in query.split('&') {
        if let Some((k, v)) = kv.split_once('=') {
            if k == key {
                return v.into();
            }
        }
    }
    String::new()
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => match (hexval(b[i + 1]), hexval(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                }
                _ => {
                    out.push(b'%');
                    i += 1;
                }
            },
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// The web app: a Cursor-like split view (code left, agent chat right) served from
/// the dongle's HTTP origin so it can call `/run` (same-origin) and OpenAI (HTTPS),
/// and use localStorage (works on HTTP). Now larger than one TCP segment — the
/// in-tunnel server (core/tcp.rs) sends it across multiple segments. Lives in flash.
const PAGE: &str = r##"<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>PySpell Agent</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;height:100vh;display:flex;flex-direction:column}
header{padding:6px 10px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:8px;align-items:center;font-size:13px;flex-wrap:wrap}
header a{color:#58a6ff;text-decoration:none}
main{flex:1;display:flex;min-height:0}
.pane{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}.left{border-right:1px solid #30363d}
.crow{flex-shrink:0}
textarea,#chat,#msg,input,select{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px}
#code{flex:1;margin:8px;padding:8px;font-family:monospace;font-size:13px;resize:none}
.bar{display:flex;gap:6px;padding:0 8px 8px}
button{background:#238636;color:#fff;border:0;border-radius:6px;padding:6px 12px;cursor:pointer}button.sec{background:#30363d}
#out{margin:0 8px 8px;padding:8px;background:#161b22;border-radius:6px;font-family:monospace;font-size:13px;white-space:pre-wrap;max-height:32%;overflow:auto}
#chat{flex:1;margin:8px;padding:8px;overflow:auto;font-size:13px}.m{margin:6px 0;padding:6px 8px;border-radius:6px}.u{background:#1f6feb33}.a{background:#161b22}
.crow{display:flex;gap:6px;padding:0 8px 8px}#msg{flex:1;padding:6px}input,select{padding:4px}
</style></head><body>
<header><b>PySpell Agent</b>
<select id=lang><option value=py>Python</option><option value=rs>Rust</option></select>
<span style=flex:1></span>
<input id=key type=password placeholder="OpenAI key" size=12>
<input id=model value=gpt-4o-mini size=10>
<a href="https://punnerud.github.io/pyspell/" target=_blank>docs</a></header>
<main>
<div class="pane left"><textarea id=code spellcheck=false></textarea>
<div class=bar><button onclick=run()>Run</button><button class=sec onclick="ask('Improve this code')">Ask agent</button></div>
<pre id=out></pre></div>
<div class=pane><div id=chat></div>
<div class=crow><input id=msg placeholder="Ask the agent..." onkeydown="if(event.key=='Enter')send()"><button onclick=send()>Send</button></div></div>
</main>
<script>
const $=id=>document.getElementById(id),LS=localStorage
$('code').value=LS.ps_code||'free_heap > 100000'
$('key').value=LS.ps_key||'';$('model').value=LS.ps_model||'gpt-4o-mini'
let chat=JSON.parse(LS.ps_chat||'[]')
function save(){LS.ps_code=$('code').value;LS.ps_key=$('key').value;LS.ps_model=$('model').value;LS.ps_chat=JSON.stringify(chat)}
function render(){$('chat').innerHTML=chat.map(m=>'<div class="m '+(m.role=='user'?'u':'a')+'">'+m.content.replace(/</g,'&lt;')+'</div>').join('');$('chat').scrollTop=1e9}
render()
async function run(){save();$('out').textContent='running...';try{const r=await fetch('/run?lang='+$('lang').value+'&timeout=20',{method:'POST',body:$('code').value});$('out').textContent=await r.text()}catch(e){$('out').textContent='error: '+e}}
function sys(){return 'You are a coding assistant for PySpell: a sandboxed Python/Rust expression subset. Allowed: literals, arithmetic, comparisons, boolean, ternary, lists, strings, builtins (len,abs,min,max,sum,round,int,float,str), fetch_json(url,"a.b.0.c"), json_get, and free vars like free_heap, uptime_s. NOT allowed: def, loops, imports, assignment. Keep replies short; give a single PySpell expression when you give code.'}
function ask(p){const t=p+':\n'+$('code').value;send(t)}
async function send(pre){const text=pre||$('msg').value;if(!text)return
$('msg').value='';chat.push({role:'user',content:text});render();save()
const key=$('key').value;if(!key){chat.push({role:'assistant',content:'(set your OpenAI key, top-right)'});render();return}
const msgs=[{role:'system',content:sys()},{role:'user',content:'Current code:\n'+$('code').value},...chat]
try{const r=await fetch('https://api.openai.com/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+key},body:JSON.stringify({model:$('model').value,messages:msgs})})
const j=await r.json();const a=j.choices?j.choices[0].message.content:('error: '+JSON.stringify(j).slice(0,200))
chat.push({role:'assistant',content:a});render();save()}catch(e){chat.push({role:'assistant',content:'error: '+e+' — OpenAI browser CORS? may need a proxy'});render()}}
</script></body></html>"##;
