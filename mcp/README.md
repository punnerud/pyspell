# PySpell MCP server

Run sandboxed **Python/Rust** expressions **live on a PySpell device** (e.g. an
ESP32 over Tailscale) from any MCP-capable agent.

There are two ways to use PySpell over MCP:

1. **Native (the chip speaks MCP).** The device exposes JSON-RPC at
   `http://<device>/mcp`. Point an MCP client that supports remote HTTP servers
   straight at it. *Caveat:* many clients require **HTTPS** (and sometimes auth) for
   remote servers, and the in-tunnel server is HTTP — so this doesn't work with every
   client.
2. **This stdio bridge (universal).** A tiny local MCP server over **stdio** — the
   transport every client supports — that forwards to the device's `/run`. Use this
   for Claude Desktop, Cursor, etc.

## Install & run

```sh
pip install mcp
PYSPELL_DEVICE=http://100.65.240.107 python pyspell_mcp.py
```

`PYSPELL_DEVICE` defaults to `http://100.65.240.107` (the device's Tailscale IP). The
machine running this bridge must be on the same tailnet as the device.

## Add to a client

**Claude Desktop** (`claude_desktop_config.json`) / **Cursor** (`mcp.json`):

```json
{
  "mcpServers": {
    "pyspell": {
      "command": "python",
      "args": ["/absolute/path/to/mcp/pyspell_mcp.py"],
      "env": { "PYSPELL_DEVICE": "http://100.65.240.107" }
    }
  }
}
```

Then ask the agent things like *"use run_pyspell to check the device's free heap"* or
*"fetch the Oslo air temperature on the device"* — it calls the `run_pyspell` tool.

## Tool

- **`run_pyspell(code, lang="py", timeout=10)`** — compile + run a PySpell program on
  the device, returning the value as text. PySpell is a deny-by-default expression
  subset (no `def`/loops/imports); the parser runs on the host, only verified IR
  reaches the device. `fetch_json(url, "a.b.0.c")` and live vars like `free_heap` are
  available.
