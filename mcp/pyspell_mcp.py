#!/usr/bin/env python3
"""PySpell MCP server (stdio bridge).

Exposes PySpell as a Model Context Protocol tool over **stdio** — the transport every
MCP client supports (Claude Desktop, Cursor, …) — so any agent can run sandboxed
Python/Rust expressions live on a PySpell device, regardless of the device being
HTTP-only inside a Tailscale tunnel.

The ESP32 also speaks MCP natively at `http://<device>/mcp`; this bridge is for
clients that require a local stdio server (or HTTPS for remote). It simply forwards
to the device's `/run` endpoint.

Run:
    pip install mcp            # or: uvx --from mcp ...
    PYSPELL_DEVICE=http://100.65.240.107 python pyspell_mcp.py

Add to Claude Desktop / Cursor (see README.md):
    { "mcpServers": { "pyspell": {
        "command": "python", "args": ["/abs/path/pyspell_mcp.py"],
        "env": { "PYSPELL_DEVICE": "http://100.65.240.107" } } } }
"""

import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

# The PySpell device base URL (its Tailscale IP by default; the in-tunnel server is
# HTTP — that's fine, traffic is already encrypted by WireGuard).
DEVICE = os.environ.get("PYSPELL_DEVICE", "http://100.65.240.107").rstrip("/")

mcp = FastMCP("pyspell")


@mcp.tool()
def run_pyspell(code: str, lang: str = "py", timeout: int = 10) -> str:
    """Compile and run a sandboxed PySpell program live on the device.

    PySpell is a Python/Rust *expression subset*: arithmetic, comparisons, boolean,
    ternary, lists, strings, builtins (len, abs, min, max, sum, round, int, float,
    str), `fetch_json(url, "a.b.0.c")` and `json_get` for live data, and free vars
    like `free_heap`, `uptime_s`. NOT allowed: def, loops, imports, assignment — the
    parser runs here on the host; only verified IR reaches the device.

    Args:
        code: the program, e.g. `free_heap > 100000` or a `fetch_json(...)` call.
        lang: "py" (default) or "rs".
        timeout: wall-clock seconds on the device, 1-60 (default 10).

    Returns the evaluated value as text (or an `error: ...` message).
    """
    timeout = max(1, min(60, int(timeout)))
    lang = "rs" if str(lang).lower() in ("rs", "rust") else "py"
    url = f"{DEVICE}/run?lang={lang}&timeout={timeout}"
    req = urllib.request.Request(url, data=code.encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout + 15) as resp:
            return resp.read().decode("utf-8", "replace").strip()
    except urllib.error.URLError as e:
        return f"error: cannot reach PySpell device at {DEVICE} ({e})"
    except Exception as e:  # noqa: BLE001 - surface anything as a tool error
        return f"error: {e}"


if __name__ == "__main__":
    mcp.run()
