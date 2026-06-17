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

/// `Cache-Control` for the immutable offline assets (model, tokenizer, WASM runtime):
/// a normal reload serves them from the browser cache (no re-download of the multi-MB
/// model); a hard refresh (Cmd+Shift+R) re-fetches after a model swap.
const CACHE: &str = "public, max-age=604800";

/// Browser-WASM runtime, embedded in flash (served from `/tinyllm_wasm.js` +
/// `/tinyllm_wasm_bg.wasm`). Built by `wasm-pack` from `crates/tinyllm-wasm` and
/// copied into `web/`; regenerate with `wasm-pack build --target web --release`.
static WASM_JS: &[u8] = include_bytes!("../web/tinyllm_wasm.js");
static WASM_BG: &[u8] = include_bytes!("../web/tinyllm_wasm_bg.wasm");

pub fn route(method: &str, path: &str, query: &str, body: &[u8]) -> Option<HttpReply> {
    match path {
        // Served zero-copy from flash (`&'static`) — no per-request page copy.
        "/" => Some(HttpReply::ok_static(
            "text/html; charset=utf-8",
            PAGE.as_bytes(),
        )),
        "/run" => Some(HttpReply::ok_owned(
            "text/plain; charset=utf-8",
            run(method, query, body).into_bytes(),
        )),
        // MCP server: JSON-RPC 2.0 over HTTP (Streamable HTTP, stateless). Any MCP
        // agent can point at http://<tailscale-ip>/mcp and call run_pyspell — the
        // microcontroller itself speaks MCP.
        "/mcp" => Some(HttpReply::ok_owned(
            "application/json",
            mcp_handle(body).into_bytes(),
        )),
        // Offline browser-WASM model host: the weights + tokenizer stream straight off
        // the `model` flash partition (never materialised), with HTTP Range — so the
        // browser fetches the ≤6 MB model from the dongle and runs it with no internet.
        // `None` (→ 404) until the image is flashed to the partition.
        "/model" => crate::model_host::model_source().map(|source| HttpReply {
            status: 200,
            content_type: "application/octet-stream",
            source,
            cache_control: Some(CACHE),
        }),
        "/tokenizer" => crate::model_host::tokenizer_source().map(|source| HttpReply {
            status: 200,
            content_type: "application/octet-stream",
            source,
            cache_control: Some(CACHE),
        }),
        // Browser-WASM runtime, embedded in flash and served zero-copy so the whole
        // thing (page + runtime + weights) comes from the dongle — works offline.
        // Cached so a normal reload doesn't re-download them (hard-refresh to update).
        "/tinyllm_wasm.js" => {
            Some(HttpReply::ok_static("text/javascript; charset=utf-8", WASM_JS).cached(CACHE))
        }
        "/tinyllm_wasm_bg.wasm" => {
            Some(HttpReply::ok_static("application/wasm", WASM_BG).cached(CACHE))
        }
        // Phase B: the word dictionary (tokens + POS types) + the int8 embedding matrix,
        // for in-browser input validation + RAG/word-search over the same vocab. Served
        // from the packed image (v2); None (404) for older images.
        "/wordmeta" => crate::model_host::wordmeta_source().map(|source| HttpReply {
            status: 200,
            content_type: "application/json",
            source,
            cache_control: Some(CACHE),
        }),
        "/embeddings" => crate::model_host::embeddings_source().map(|source| HttpReply {
            status: 200,
            content_type: "application/octet-stream",
            source,
            cache_control: Some(CACHE),
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
    // Per-job memory budget in bytes ("container" memory limit). A caller that
    // knows the job needs more can request it (e.g. `?mem=65536`); the device then
    // admits or rejects up front instead of OOM-crashing midway. Default 16 kB.
    let max_bytes = query_get(query, "mem").parse::<i64>().unwrap_or(16384);
    // POST → the raw request body is the program; GET → the URL-encoded `code` param.
    let code = if method.eq_ignore_ascii_case("POST") {
        String::from_utf8_lossy(body).trim().to_string()
    } else {
        url_decode(&query_get(query, "code"))
    };
    eval_program(&code, lang, timeout_s, max_bytes)
}

/// Parse + evaluate a PySpell program with the device's env/net/display and a
/// wall-clock deadline; returns the value (or `error: ...`) as a string. Shared by
/// the `/run` POST API and the MCP `run_pyspell` tool — and the single choke point
/// that records each job for the display counters.
fn eval_program(code: &str, lang: Lang, timeout_s: i64, max_bytes_req: i64) -> String {
    if code.is_empty() {
        return "error: empty program".into();
    }
    if code.len() > MAX_CODE {
        return "error: program too long".into();
    }

    // Per-job ("container") memory budget, clamped to a sane range (1 kB .. 256 kB).
    let max_bytes = max_bytes_req.clamp(1024, 262_144) as u32;
    // Admission UP FRONT: only start if free heap covers the budget plus working
    // overhead (parse IR, env, a possible TLS fetch). Reject cleanly here rather
    // than letting the job OOM-crash the device midway.
    const MEM_MARGIN: u32 = 40 * 1024;
    let free = unsafe { esp_idf_svc::sys::esp_get_free_heap_size() };
    if free < max_bytes.saturating_add(MEM_MARGIN) {
        return format!(
            "error: insufficient memory: job budget {max_bytes} B + ~{MEM_MARGIN} B overhead > {free} B free (lower mem= or retry)"
        );
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
        max_bytes,
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
                            "timeout": { "type": "integer", "description": "Wall-clock seconds 1-60, default 10" },
                            "mem": { "type": "integer", "description": "Per-job memory budget in bytes (1024-262144, default 16384). Rejected up front if the device lacks free heap." }
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
            let mem = args
                .and_then(|a| a.get("mem"))
                .and_then(|v| v.as_i64())
                .unwrap_or(16384);
            let result = eval_program(code, lang, timeout, mem);
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
.badge{padding:2px 9px;border-radius:11px;background:#30363d;font-size:12px;white-space:nowrap}
#settings{display:none;position:fixed;top:42px;right:10px;z-index:50;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;width:min(320px,92vw);box-shadow:0 6px 24px #000a;font-size:13px}
#settings label{display:block;margin-bottom:8px}#settings input{width:100%;padding:5px;margin-top:3px}
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
<span id=backend class=badge title="which model answers — change under ⚙">● on-device model</span>
<input id=wsearch placeholder="related words" size=9 onkeydown="if(event.key=='Enter')words(this.value)">
<button class=sec onclick=verify()>Verify</button>
<button class=sec onclick=clr()>Clear</button>
<button class=sec onclick=gear() title="model settings — switch to OpenAI">⚙</button>
<a href="https://punnerud.github.io/pyspell/" target=_blank>docs</a></header>
<div id=settings>
<div style="margin-bottom:6px"><b>Model backend</b></div>
<div style="opacity:.7;margin-bottom:10px">Default is the <b>on-device offline model</b> (no internet). Add an OpenAI key to use GPT instead — it stays in this browser only.</div>
<label>OpenAI key (optional)<input id=key type=password placeholder="sk-… leave empty for on-device" oninput="save();updBackend()"></label>
<label>OpenAI model<input id=model value=gpt-4o-mini oninput="save();updBackend()"></label>
<button class=sec onclick="$('settings').style.display='none'">Done</button></div>
<main>
<div class="pane left"><textarea id=code spellcheck=false></textarea>
<div class=bar><button onclick=run()>Run</button><button class=sec onclick="ask('Improve this code')">Ask agent</button></div>
<pre id=out></pre></div>
<div class=pane><div id=chat></div>
<div class=crow><input id=msg placeholder="Ask the agent..." onkeydown="if(event.key=='Enter')send()"><button onclick=send()>Send</button><button class=sec onclick=editBtn() title="edit the matching code line in place (model emits a find/replace)">Edit</button><button class=sec onclick=applyLast() title="splice the last snippet into the best-matching code line">Apply</button></div></div>
</main>
<script>
const $=id=>document.getElementById(id),LS=localStorage
$('code').value=LS.ps_code||'free_heap > 100000'
$('key').value=LS.ps_key||'';$('model').value=LS.ps_model||'gpt-4o-mini'
let chat=JSON.parse(LS.ps_chat||'[]')
function gear(){const s=$('settings');s.style.display=s.style.display=='block'?'none':'block'}
// Reflect the active backend in the header badge: on-device by default, GPT if a key is set.
function updBackend(){const k=$('key').value.trim(),b=$('backend')
b.textContent=k?('● GPT · '+($('model').value.trim()||'gpt-4o-mini')):'● on-device model'
b.style.color=k?'#e3b341':'#7ee787'}
updBackend()
function save(){LS.ps_code=$('code').value;LS.ps_key=$('key').value;LS.ps_model=$('model').value;LS.ps_chat=JSON.stringify(chat)}
function render(){$('chat').innerHTML=chat.map(m=>'<div class="m '+(m.role=='user'?'u':'a')+'">'+(m.html||m.content.replace(/</g,'&lt;'))+'</div>').join('');$('chat').scrollTop=1e9}
function he(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function ccol(c){return c>0.8?'#7ee787':c>0.5?'#e3b341':'#f85149'} // confidence -> green/amber/red
// Scrollable overlay listing the whole vocabulary the model knows, grouped by POS type.
async function showVocab(){let wm;try{wm=await ensureWords()}catch(e){return}
const g={};wm.meta.tokens.forEach((t,i)=>{const w=t.trim();if(/^[A-Za-z]{2,}$/.test(w)){const ty=wm.meta.types[i];(g[ty]=g[ty]||[]).push(w)}})
let h='<div id=vov style="position:fixed;inset:0;background:#000a;z-index:99;display:flex;align-items:center;justify-content:center" onclick="if(event.target.id==\'vov\')vov.remove()">'
h+='<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;max-height:80vh;width:min(560px,92vw);overflow:auto;padding:14px;font-size:13px">'
h+='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px"><b>Vocabulary — '+wm.wlist.length+' words the model knows</b><button class=sec onclick=vov.remove()>close</button></div>'
for(const ty of Object.keys(g).sort())h+='<div style="margin:8px 0"><b style=color:#58a6ff>'+ty+'</b> ('+g[ty].length+'): '+g[ty].sort().map(he).join(', ')+'</div>'
h+='</div></div>';document.body.insertAdjacentHTML('beforeend',h)}
// Fully-specified arithmetic: copy the operands+operator VERBATIM from the request into
// print(...) — the browser does the copy, so digits are never regenerated (and reworded
// 'calculate' etc. doesn't matter). Returns code or null.
function genFastPath(instr){const op={'+':'+','-':'-','*':'*','/':'//','plus':'+','add':'+','minus':'-','subtract':'-','times':'*','multiply':'*','multiplied':'*','divide':'//','divided':'//'}
let m=instr.match(/(-?\d+)\s*([+\-*\/])\s*(-?\d+)/);if(m)return 'print('+m[1]+' '+op[m[2]]+' '+m[3]+')'
m=instr.match(/\b(add|subtract|multiply|divide|plus|minus|times)\b[^\d-]*(-?\d+)\D+?(-?\d+)/i);if(m&&op[m[1].toLowerCase()])return 'print('+m[2]+' '+op[m[1].toLowerCase()]+' '+m[3]+')'
return null}
// Auto-evaluate a suggestion in place: pyspell is an EXPRESSION sandbox (no loops/def/
// assignment, no print), so we extract the runnable expression from the snippet —
// print(expr)->expr, or a bare expression — and skip anything pyspell can't run or that
// would touch the network. The browser does the copy; the dongle runs it (2s, sandboxed).
function runnable(snip){snip=(snip||'').replace(/\s*⟨[^⟩]*⟩\s*$/,'').trim()
if(!snip||/[\n;]/.test(snip))return null            // single statement only
const m=snip.match(/^print\((.*)\)$/);if(m)snip=m[1].trim()
if(!snip||/\b(for|while|def|class|import|lambda|range|print)\b/.test(snip))return null
if(/(^|[^=!<>])=([^=]|$)/.test(snip))return null    // assignment (keeps == <= >= !=)
if(/\bfetch(_json)?\b/.test(snip))return null        // no network calls in auto-run
return snip}
async function autoEval(snip){const e=runnable(snip);if(!e)return null
try{const r=await fetch('/run?lang=py&timeout=2',{method:'POST',body:e});const t=(await r.text()).trim();return t||null}catch(_){return null}}
// Append the live result of message `i`'s snippet as a small green/red ▷ badge.
async function evalInto(i,snip){const v=await autoEval(snip);if(v==null)return
const err=/^error:/.test(v)
// Keep the result OUT of .content — Apply/applyLast copy .content, so it must stay
// the bare code. The ▷ result lives only in .html (display) + .eval (persisted).
chat[i].eval=v
chat[i].html=(chat[i].html||he(chat[i].content))+'<div style="margin-top:4px;font-family:ui-monospace,monospace;font-size:12px;color:'+(err?'#f85149':'#7ee787')+'" title="ran live in the pyspell sandbox — 2s, no network">▷ '+he(v)+'</div>'
render();save()}
render()
async function run(){save();$('out').textContent='running...';try{const r=await fetch('/run?lang='+$('lang').value+'&timeout=20',{method:'POST',body:$('code').value});$('out').textContent=await r.text()}catch(e){$('out').textContent='error: '+e}}
function sys(){return 'You are a coding assistant for PySpell: a sandboxed Python/Rust expression subset. Allowed: literals, arithmetic, comparisons, boolean, ternary, lists, strings, builtins (len,abs,min,max,sum,round,int,float,str), fetch_json(url,"a.b.0.c"), json_get, and free vars like free_heap, uptime_s. NOT allowed: def, loops, imports, assignment. Keep replies short; give a single PySpell expression when you give code.'}
function ask(p){const t=p+':\n'+$('code').value;send(t)}
async function send(pre){const text=pre||$('msg').value;if(!text)return
$('msg').value='';chat.push({role:'user',content:text});render();save()
if($('key').value){await openai()}else{
const bad=await validate(text);if(bad.length){chat.push({role:'assistant',content:'⚠ outside the model vocabulary: '+bad.join(', ')+' — rephrase with common words.',html:'⚠ outside the model vocabulary: <b>'+bad.map(he).join(', ')+'</b> — rephrase with <a style="color:#58a6ff;cursor:pointer;text-decoration:underline" onclick=showVocab()>common words</a>.'});render()}
if($('code').value.trim()&&/\b(change|rename|instead|replace|swap|make it|count down|upper bound|use the|delete|remove|move|put .* below|everywhere|all occurrences|uses of)\b/i.test(text)){await editLoop(text)}
else{const fp=genFastPath(text);if(fp){const j=chat.push({role:'assistant',content:fp+'  ⟨copied from your request⟩'})-1;render();save();await evalInto(j,fp)}else{await local(text)}}}}
// Phase B: word dictionary + embeddings served from the dongle, for input validation
// + related-word search (RAG over our OWN vocab — the same table the model thinks in).
let _wm
async function ensureWords(){if(_wm)return _wm
const meta=await (await fetch('/wordmeta')).json()
const eb=new Int8Array(await (await fetch('/embeddings')).arrayBuffer())
const words={},wlist=[]
meta.tokens.forEach((t,i)=>{const w=t.trim().toLowerCase();if(/^[a-z]{2,}$/.test(w)&&!(w in words)){words[w]=i;wlist.push(w)}})
return _wm={meta,eb,dim:meta.dim,words,wlist}}
async function validate(text){let wm;try{wm=await ensureWords()}catch(e){return[]}
const set=new Set(wm.wlist),bad=[]
for(const w of (text.toLowerCase().match(/[a-z]+/g)||[]))if(w.length>1&&!set.has(w)&&!bad.includes(w))bad.push(w)
return bad}
function cos(eb,dim,a,b){let d=0,na=0,nb=0;for(let i=0;i<dim;i++){const x=eb[a*dim+i],y=eb[b*dim+i];d+=x*y;na+=x*x;nb+=y*y}return d/(Math.sqrt(na*nb)+1e-9)}
async function words(q){q=(q||'').trim().toLowerCase();if(!q)return
let wm;try{wm=await ensureWords()}catch(e){chat.push({role:'assistant',content:'words unavailable: '+e});render();return}
const id=wm.words[q]
if(id===undefined){chat.push({role:'assistant',content:'“'+q+'” is not in the dictionary ('+wm.wlist.length+' words)'});render();return}
const sc=wm.wlist.filter(w=>w!==q).map(w=>[w,cos(wm.eb,wm.dim,id,wm.words[w])]).sort((a,b)=>b[1]-a[1]).slice(0,8)
chat.push({role:'assistant',content:'related to “'+q+'”: '+sc.map(x=>x[0]+' ('+x[1].toFixed(2)+')').join(', ')});render()}
// Line-local edit: the model generates a snippet (its trained task); we find the most
// relevant existing line by embedding similarity to the request and splice the snippet
// THERE — never rewriting the whole file. A tiny model can't hold all the code, but the
// browser does, so retrieval + a local splice keeps it targeted.
function meanEmb(wm,text){const ids=(text.toLowerCase().match(/[a-z]+/g)||[]).map(w=>wm.words[w]).filter(x=>x!==undefined)
if(!ids.length)return null;const v=new Float32Array(wm.dim)
for(const id of ids)for(let i=0;i<wm.dim;i++)v[i]+=wm.eb[id*wm.dim+i]
return v}
function cosv(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na*nb)+1e-9)}
async function applyLast(){
const u=[...chat].reverse().find(m=>m.role==='user')
const a=[...chat].reverse().find(m=>m.role==='assistant'&&m.content&&!/^[⚠]|^related to/.test(m.content))
if(!u||!a){$('out').textContent='nothing to apply';return}
const snippet=a.content.replace(/\s*⟨[^⟩]*⟩\s*$/,'').trim();if(!snippet)return
let wm=null;try{wm=await ensureWords()}catch(e){}
const code=$('code').value,lines=code.split('\n');let best=-1,bestsc=-1
if(wm){const q=meanEmb(wm,u.content);if(q)lines.forEach((ln,i)=>{const e=meanEmb(wm,ln);if(e){const s=cosv(q,e);if(s>bestsc){bestsc=s;best=i}}})}
if(best>=0&&bestsc>0.4&&lines[best].trim()){lines[best]=snippet;$('code').value=lines.join('\n');$('out').textContent='replaced line '+(best+1)+' (match '+bestsc.toFixed(2)+')'}
else{$('code').value=(code.trim()?code.replace(/\s*$/,'')+'\n':'')+snippet;$('out').textContent='inserted at end'}
save()}
// Anchor-based edit: the model emits a find/replace directive "@@ old ==> new" for a
// small window the browser retrieved; the browser does the actual edit (line.replace),
// so lists/long tails are never copied by the model.
// Parse the first directive in the model output: @@ replace / DEL / MOVE / RENAME.
function parseDirective(t){for(const l of t.split('\n')){
if(l.startsWith('@@ ')){const m=l.match(/^@@ (.*?) ==> ([\s\S]*)$/);if(m)return{op:'@@',a:m[1],b:m[2]}}
if(l.startsWith('DEL ')&&l.length>4){return{op:'DEL',a:l.slice(4)}}
if(l.startsWith('RENAME ')){const m=l.match(/^RENAME (.*?) ==> (.*)$/);if(m)return{op:'RENAME',a:m[1],b:m[2]}}
if(l.startsWith('MOVE ')){const m=l.match(/^MOVE (.*?) ==> (.*)$/);if(m)return{op:'MOVE',a:m[1],b:m[2]}}}
return null}
function esc(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}
// Apply a directive to `lines` (browser holds the file). Returns a status string, or null.
function applyDirective(lines,d,best){
const find=n=>{for(let o=0;o<lines.length;o++){const j=(best+o)%lines.length;if(lines[j].includes(n))return j}return -1}
if(d.op==='@@'){const j=find(d.a);if(j<0)return null;lines[j]=lines[j].replace(d.a,d.b);return 'replaced line '+(j+1)}
if(d.op==='DEL'){const j=find(d.a);if(j<0)return null;lines.splice(j,1);return 'deleted line '+(j+1)}
if(d.op==='RENAME'){const re=new RegExp('\\b'+esc(d.a)+'\\b','g');let c=0;for(let i=0;i<lines.length;i++){const b=lines[i];lines[i]=b.replace(re,d.b);if(lines[i]!==b)c++}return c?('renamed '+JSON.stringify(d.a)+'→'+JSON.stringify(d.b)+' in '+c+' line(s)'):null}
if(d.op==='MOVE'){const s=find(d.a);if(s<0)return null;const ln=lines.splice(s,1)[0];let dj=-1;for(let i=0;i<lines.length;i++){if(lines[i].includes(d.b)){dj=i;break}}if(dj<0){lines.splice(s,0,ln);return null}lines.splice(dj+1,0,ln);return 'moved after line '+(dj+1)}
return null}
function winAround(lines,best,pad){return lines.slice(Math.max(0,best-pad),Math.min(lines.length,best+1+pad)).join('\n')}
// Deterministic instruction parse for the copy-heavy ops (rename/move/delete) — these are
// fully specified by the text, so the browser builds the directive directly (no model,
// 100% reliable). The model is reserved for semantic @@ edits (change a value/operator).
function parseInstruction(instr,lines){let m
if((m=instr.match(/\brename\s+(\w+)\s+to\s+(\w+)/i))||(m=instr.match(/\bchange\s+every\s+(\w+)\s+to\s+(\w+)/i))||(m=instr.match(/\bcall\s+(\w+)\s+(\w+)\s+everywhere/i)))return{op:'RENAME',a:m[1],b:m[2]}
if((m=instr.match(/\bmove\s+(?:the\s+)?(\w+)\b[\s\S]*?\b(?:after|below)\s+(?:the\s+)?(\w+)/i))||(m=instr.match(/\bput\s+(\w+)\s+below\s+(\w+)/i)))return{op:'MOVE',a:m[1],b:m[2]}
if(/\b(delete|remove)\b/i.test(instr)){const stop=['delete','remove','the','line','that','this','a','please','print'];const ws=(instr.toLowerCase().match(/[a-z_]\w+/g)||[]).filter(w=>!stop.includes(w));let bw=null,bn=1e9;for(const w of ws){const c=lines.filter(l=>l.toLowerCase().includes(w)).length;if(c>=1&&c<bn){bn=c;bw=w}}if(bw)return{op:'DEL',a:bw}}
return null}
// Browser-driven edit loop: retrieve a window, generate a directive, apply it; if it
// doesn't apply, widen the window and retry (<=3 rounds). The model never holds the file.
async function editLoop(instr){const i=chat.push({role:'assistant',content:'⏳ editing…'})-1;render()
const set=t=>{chat[i].content=t;render()}
let m;try{m=await ensureLocal(set)}catch(e){set('load failed: '+e);return}
let wm=null;try{wm=await ensureWords()}catch(e){}
if(!$('code').value.trim()){set('(no code to edit)');return}
const base=$('code').value.split('\n');let best=0,bestsc=-1
if(wm){const q=meanEmb(wm,instr);if(q)base.forEach((ln,j)=>{const e=meanEmb(wm,ln);if(e){const s=cosv(q,e);if(s>bestsc){bestsc=s;best=j}}})}
set('')
// rename/move/delete: deterministic from the instruction — no model needed
const di=parseInstruction(instr,base)
if(di){const lines=$('code').value.split('\n');const res=applyDirective(lines,di,best);if(res){$('code').value=lines.join('\n');save();set('✎ '+res+'  ⟨browser⟩');return}}
for(const pad of [0,2,4]){
const win=winAround(base,best,pad)
let out='',p;const g=new m.mod.Generator(m.model,m.tok,'EDIT '+instr+'\n'+win,48,0,0.9,1+pad)
while((p=g.step())!==undefined)out+=p
const d=parseDirective(out);if(!d)continue
const lines=$('code').value.split('\n');const res=applyDirective(lines,d,best)
if(res){$('code').value=lines.join('\n');save();set('✎ '+res+'  ⟨local edit⟩');return}}
set('couldn’t apply an edit — try rephrasing')}
function editBtn(){const t=$('msg').value;if(!t)return;$('msg').value='';chat.push({role:'user',content:t});render();save();editLoop(t)}
async function openai(){const key=$('key').value
const msgs=[{role:'system',content:sys()},{role:'user',content:'Current code:\n'+$('code').value},...chat]
try{const r=await fetch('https://api.openai.com/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+key},body:JSON.stringify({model:$('model').value,messages:msgs})})
const j=await r.json();const a=j.choices?j.choices[0].message.content:('error: '+JSON.stringify(j).slice(0,200))
chat.push({role:'assistant',content:a});render();save()}catch(e){chat.push({role:'assistant',content:'error: '+e+' — OpenAI browser CORS? may need a proxy'});render()}}
// Local, fully-offline backend: the WASM runtime + model + tokenizer all come from
// the dongle (no internet). Loaded once, then generation streams token-by-token.
let _ml
// Fetch a URL with a streaming reader so we can show byte progress, returning the
// full body as a Uint8Array. (We feed the wasm bytes straight to init, bypassing
// instantiateStreaming — which could otherwise stall on the in-tunnel stream.)
async function prog(url,label,set){const r=await fetch(url);if(!r.ok)throw new Error(label+' HTTP '+r.status)
const total=+(r.headers.get('content-length')||0),rd=r.body.getReader(),cs=[];let got=0
for(;;){const{done,value}=await rd.read();if(done)break;cs.push(value);got+=value.length
set('⏳ '+label+' '+(got>>10)+(total?'/'+(total>>10):'')+' KB')}
const u=new Uint8Array(got);let o=0;for(const c of cs){u.set(c,o);o+=c.length}return u}
// Load (once) the runtime + model + tokenizer, all from the dongle. Shared by the
// chat backend and the Verify self-test.
async function ensureLocal(set){if(_ml)return _ml
set('⏳ importing runtime js…');const mod=await import('/tinyllm_wasm.js')
const wasm=await prog('/tinyllm_wasm_bg.wasm','wasm',set)
set('⏳ instantiating wasm…');await mod.default({module_or_path:wasm})
const model=await prog('/model','model (once)',set)
const tok=await prog('/tokenizer','tokenizer',set)
return _ml={mod,model,tok}}
async function local(prompt){const i=chat.push({role:'assistant',content:''})-1;render()
const set=t=>{chat[i].content=t;render()}
let m;try{m=await ensureLocal(set)}catch(e){set('load failed: '+e+' (flash a model image, or set an OpenAI key)');return}
set('')
try{const g=new m.mod.Generator(m.model,m.tok,prompt,64,0.9,0.9,Math.floor(Math.random()*1e9))
let p,n=0,html=''
while((p=g.step())!==undefined){const c=g.confidence;chat[i].content+=p;if(p)html+='<span style="color:'+ccol(c)+'" title="'+Math.round(c*100)+'% sure">'+he(p)+'</span>';chat[i].html=html;if(++n%2==0){render();await new Promise(r=>requestAnimationFrame(r))}}
chat[i].content=(chat[i].content||'(no output)')+'  ⟨local toy model, offline⟩'
chat[i].html=(html||'(no output)')+' <span style=opacity:.55>⟨coloured by confidence⟩</span>';render();save()
await evalInto(i,chat[i].content)}catch(e){chat[i].content='local gen error: '+e;chat[i].html=null;render()}}
// Clear the chat (with confirm); Verify self-tests the offline agent end-to-end.
function clr(){if(confirm('Clear chat?')){chat=[];save();render()}}
async function verify(){const i=chat.push({role:'assistant',content:'⏳ verifying offline agent…'})-1;render()
const set=t=>{chat[i].content=t;render()}
try{const m=await ensureLocal(set)
const g=new m.mod.Generator(m.model,m.tok,'Once upon a time',8,0.9,0.9,1)
let out='',p;while((p=g.step())!==undefined)out+=p
set('✅ agent OK — wasm loaded, model '+(m.model.length>>10)+' KB, tokenizer '+m.tok.length+' B, sample: '+JSON.stringify(out.slice(0,40)));save()}catch(e){set('❌ verify failed: '+e)}}
</script></body></html>"##;
