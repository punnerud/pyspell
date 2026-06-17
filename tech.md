# PySpell tech deep-dive: a browser LLM coding agent served off a 512 kB chip

This document explains the parts that are *not* obvious: how a ~0.45 M-parameter
language model turns plain English into runnable code, how it fits and ships from an
ESP32-S3 with **512 kB SRAM and no PSRAM**, the tricks that make a sub-0.5 MB model
actually useful, and **how to retrain it for your own language or task**.

For the network/TLS/Tailscale memory tricks see [`docs/memory-512kb.md`](docs/memory-512kb.md).
For the language reference see the [GitHub Pages site](https://punnerud.github.io/pyspell/).

---

## What it is

Open `http://<dongle>/` (over the Tailscale tunnel) and you get a Cursor-like agent:
a code editor on the left, a chat on the right. Type **"flash the light"**,
**"show the text \"hei og hopp\""**, **"what is 7 plus 5"**, or **"reverse the word
robot"** and it produces the code, **runs it live on the chip**, and shows you the
result — or the physical action (the screen lights up, the LED blinks).

The whole thing — the runtime, the model, the tokenizer, the dictionary — is served
**from the dongle, offline**. No cloud, no API key (unless you opt into OpenAI behind
the ⚙). A $10 USB stick is the entire stack.

---

## The hard part: a 0.45 M-param model is *tiny*

For scale: GPT-2 small is 124 M params; this model is **0.45 M** — ~280× smaller —
and quantized to **int8 it is under 500 kB**. A model that small cannot do what LLMs
normally do. The system is a set of tricks that route *around* its limitations so it
only ever does the one thing it can do well: **map intent to structure.**

### Architecture

A stripped llama2 (run.c-style), trained from scratch:

| | |
|---|---|
| dim | 128 |
| layers | 2 |
| heads / kv-heads | 4 / 4 (GQA) |
| hidden (SwiGLU) | 256 |
| context | 128 tokens |
| vocab | **512** |
| params | ~0.45 M |
| on disk | **< 500 kB** (int8, group-quantized) |

RMSNorm, interleaved-pair RoPE, SwiGLU, grouped-query attention. Dropout 0.1 +
best-checkpoint + early-stop, because at this size the model is **data-capped, not
compute-capped** — long runs overfit.

### Trick 1 — a curated 512-token vocabulary of *whole words*

A normal BPE vocab of 512 would be mostly sub-word fragments. Instead we **force**
whole instruction words (`print`, `reverse`, `largest`, `led`, `flash`, colour names,
…) and Python structural tokens (`print(`, `range(`, `for i in range(`, `@@ `, …)
into the vocab before BPE fills the rest by frequency. Result: the model "thinks" in
~190 real words and a few dozen code idioms, not letter-soup. (`train/train.py`
`WORDS` + `PY_TOKENS` → `FORCED`.)

### Trick 2 — frozen embeddings distilled from a bigger model

The 0.45 M budget is too small to *also* learn what words mean. So we don't. The
input embedding table is **built once and frozen**:

1. Embed every vocab word with **all-MiniLM** (a 22 M-param sentence model, run
   locally via Ollama).
2. PCA down to 128 dims.
3. Fold in a small **part-of-speech / type** vector (noun, verb, Python-keyword,
   colour, …) and row-normalize: `emb = normalize(0.9·semantic + 0.4·type)`.
4. Freeze it. The transformer only learns to *route* these fixed meanings; the
   output classifier is **untied** (separately learned).

This is sequence-level + representation-level **distillation**: a big model's word
geometry becomes the tiny model's starting prior. (`train/build_embeddings.py`,
`train/build_types.py`.)

### Trick 3 — "the model points, the browser copies" (the key idea)

A 0.45 M model **cannot reliably copy arbitrary tokens** — multi-digit numbers,
quoted strings, list literals all come out mangled (`3+2`→`3+21`, `"hello".upper()`→
`opper()`). The fix isn't a bigger model; it's **not asking it to copy.**

The model emits tiny *semantic directives*; the **browser** — which already holds the
file and the user's exact words — does the deterministic copying:

| You type | Browser builds (verbatim copy in **bold**) | Model's job |
|---|---|---|
| `calculate 3 + 2` | `print(`**3 + 2**`)` | (none — pure copy) |
| `print "hello world"` | `print("`**hello world**`")` | (none) |
| `make the light red` | `led("`**red**`")` | (none) |
| `change add to subtract` | `@@ ` **+** ` ==> ` **-** | find the anchor |
| `rename x to count` | `RENAME `**x**` ==> `**count** | (none) |

Anything quoted is treated as **literal content**: it's copied byte-for-byte and is
**excluded from vocabulary validation** (a name like `sunflower` inside quotes is data,
not an instruction). The tiny model is reserved for the genuinely semantic cases —
"which builtin, which structure" — where a wrong token is recoverable. (See
`genFastPath` / `genDeviceFastPath` / `genMinMaxFastPath` in
`demo/esp32-tailscale-pyspell/src/pyspell_web.rs` and `index.html`.)

**The same idea, now inside the model — delexicalization.** The hand-written fast-paths
cover the common intents, but for everything else the model used to fall back to copying
(and mangling) long literals. So the model no longer sees literals at all: before
training, copied numbers/strings are swapped for **slot markers** (`#0..#7`, `&a..&d`) in
*both* the English and the Python, so it learns the **template** —
`the largest of #0 and #1` → `print(max(#0, #1))` — and only has to route *"slot k goes
here"*. At inference the browser delexicalizes the prompt, runs the model, and copies the
real literals back into the slots (`relex`). No copy attention, no architecture change,
and the freed vocab budget buys more real words. The contract is shared by training
(`train/delex.py`) and the browser/device (`web/delex.js`) and pinned equal by
`train/parity_delex.py`; `index.html` auto-enables it when the tokenizer carries the
markers, so a delex model and the old literal model both just work. This is the general,
model-driven version of "the model points, the browser copies".

### Trick 4 — the device never runs the model; the browser does

Inference runs **client-side in WebAssembly** (`crates/tinyllm-wasm`). The dongle is a
*file server + sandbox*, not an inference engine — it has ~60 kB free heap, nowhere
near enough to hold a 0.5 MB model. This inverts the usual "edge inference" picture:
the constrained device ships the model to a capable browser and grades the output by
running it.

### Trick 5 — stream the model off flash, never materialize it

The model image lives in a dedicated 6 MB flash partition (`model` @ `0x810000`),
packed as one blob:

```
PSM1 | ver | tok_len | model_len | wordmeta_len | embed_len   (24-byte TOC)
      ├─ tokenizer.bin  (~6.7 kB)
      ├─ model.bin      (~489 kB, int8 llama2 weights)
      ├─ wordmeta.json  (~8.7 kB, tokens + POS types)
      └─ embeddings     (~65 kB, int8 vocab embedding matrix)
```

`GET /model`, `/tokenizer`, `/wordmeta`, `/embeddings` each stream their slice
**straight off flash, read on demand a segment at a time** (`BodySource::Flash` +
`esp_partition_read`) with HTTP **Range** support. Peak device RAM is **one TCP
segment**, regardless of body size — a 0.5 MB model serves from a chip that can't
hold 1/8 of it. The in-tunnel TCP server is ACK-clocked with go-back-N retransmit
(`core/src/tcp.rs`) so the transfer survives the lossy DERP/WireGuard path.

### Trick 6 — the vocabulary *is* the dictionary (validation + RAG)

The same 512 tokens and their frozen embeddings that the model thinks in are served
back to the browser and reused as a live dictionary:

- **Input validation** — words in your request that aren't in the vocab are flagged
  ("outside the model vocabulary — rephrase…"), with a clickable list of every word it
  knows (grouped by part of speech). Quoted content is skipped.
- **Related-word search (RAG)** — cosine similarity over the embedding matrix finds
  near-synonyms *in the model's own vocabulary*, so suggestions stay in-distribution.

### Trick 7 — instant feedback via the sandbox

Generated code is **run live** in the PySpell expression sandbox (2 s wall-clock, no
network) the moment it's produced — the result shows inline as a `▷` badge, or fires
the LED/screen. No "Apply" step. This only works because PySpell is a deny-by-default
*expression* evaluator (see the main [README](README.md)): `print(...)`, arithmetic,
comparisons, ternary, lists, and a fixed builtin set
(`len/abs/min/max/sum/round/int/float/str/first/last/index/upper/lower/reverse/`
`show/led/flash/fetch_json/…`). What it *can't* parse (loops, `def`, `.method()`,
`[::-1]`, assignment) is exactly the security boundary.

### Trick 8 — the data flywheel

Quality is data-bound, so we mine for the right data. `eval.py --harvest` measures
**per-token confidence** with literal positions masked (so number/string-copy
weakness — which is structural and *not* data-fixable — isn't chased), finds the
weak *families*, oversamples them, retrains a candidate, and **gates** it against the
champion (promote only if it doesn't regress). `flywheel.py` runs the loop.

### Bonus — running the model *on the chip itself* (feasibility spike)

The browser is the normal compute path, but the ESP32 can also run the model **on-device**
(`POST /generate`), proving the dongle is self-sufficient. Findings from the spike:

- **It works, and it's fast** — greedy generation produced correct code on-chip
  (`turn the led on` → `led(1)`, `what is 8 plus 3` → `print(8 + 3)`) in **~1.9 s for 24
  tokens** (the 40 s estimate was pessimistic; the model is tiny).
- **Weights never touch the heap** — `model.bin` is `esp_partition_mmap`'d; the forward
  pass reads it through the flash cache (streamed/paged), so the 489 kB model runs on a
  chip with ~60 kB free heap.
- **int8 KV cache, bounded context** (`RunState::try_new_int8`, `MAX_CTX=32`) — the f32
  cache (256 kB) won't fit; int8 + short context keeps the working set ~25 kB. Allocated
  with `try_reserve` so a tight heap returns a clean "low memory" instead of an
  OOM-reboot, on a **persistent worker thread** (stack allocated once) that **yields each
  step**, so Tailscale stays online and `/run` keeps serving *in parallel* during a
  generation.
- **The honest limit is memory.** Free heap is ~60 kB fresh, fragmenting toward ~30 kB
  under sustained load; the granular 512-vocab tokenizer makes prompts 17–24 tokens, so
  the longest commands truncate at `MAX_CTX`, and back-to-back runs cleanly refuse
  ("try again when idle"). Reliable sustained on-device generation wants PSRAM or a
  dedicated inference memory pool. So: **the chip *can* run its own model — best-effort,
  when idle — but the browser path (WASM, <0.4 s) is what you'd use day to day.**

---

## What's actually new here

- **A full offline LLM coding agent served from a 512 kB, no-PSRAM, $10 dongle** —
  runtime, model, tokenizer and dictionary all shipped from the chip, over a
  Tailscale tunnel, no cloud.
- **Inverted edge inference**: the constrained device *serves and grades*; the
  browser *computes*. The model streams off flash a segment at a time and is never
  resident on the device.
- **"Model points, browser copies"** — a labour split that makes a sub-0.5 MB model
  genuinely useful for code generation by never asking it to copy.
- **A sentence-embedding model distilled into a frozen 512-word vocabulary** as the
  tiny model's prior — and then *served back* as a validation + RAG dictionary.
- **English → live hardware action** on a microcontroller: "flash the light" becomes a
  blinking LED, end to end, on-device.

---

## Retraining it — your own language or task

The pipeline is deliberately small and template-driven, so an LLM can do most of the
porting for you. Everything is in `train/` (Python; a `venv` with `torch` +
`numpy`; Ollama for embeddings).

```
gen_data.py   templates: English instruction -> Python  (the only thing you translate)
curate.py     canonicalize + validate (must compile) + dedup + distill  -> data/*.jsonl
train.py      BPE(vocab=512, FORCED words) + frozen embeddings + train  -> out/
export_v2.py  pack TOC + tokenizer + model + wordmeta + embeddings      -> out/model.img
              espflash write-bin 0x810000 out/model.img
```

### To retrain in, say, Norwegian / German / Spanish

1. **Translate the instruction phrasings** in `gen_data.py`. Each family is a list of
   `en` strings with `{placeholders}`. Hand them to any LLM:
   > "Translate these instruction phrasings to Norwegian. Keep the `{a}`/`{w}`
   > placeholders and the meaning; give natural, varied phrasings." 

   The Python *target* (`py`) stays the same — you're only changing the input language.
2. **Swap the embedding source** in `build_embeddings.py` to a multilingual model
   (e.g. `paraphrase-multilingual-MiniLM` via sentence-transformers, or a multilingual
   Ollama embedding model) so the frozen prior captures your language's word geometry.
3. **Update `FORCED` words** in `train.py` to your language's instruction words (so the
   512-vocab is dominated by *your* words, not English).
4. (Optional) **LLM teacher distillation**: `curate.py --qwen <ollama-model>` asks a
   local model for paraphrases of your seed instructions — sequence-level distillation
   into the tiny student. Add hand-written gold pairs to `train/seeds.jsonl`
   (oversampled so natural phrasing dominates).
5. `python curate.py --n 90000 …` → `python train.py --preset full512 …` →
   `python export_v2.py --out out` → flash. Validate with `eval.py` and
   `roundtrip.py` before flashing.

### Notes for training at scale (GPU)

- The dataset is a **static, precomputed** JSONL (`data/train.jsonl`). The generator
  in `gen_data.py` can currently reach ~50 k unique `(en, py)` pairs; capture more of
  it by raising `curate.py --n` (dedup keeps the unique ones).
- For longer GPU runs, `train.py` supports **periodic regeneration** (`--regen-every
  N` steps): the dataset is re-curated and re-tokenized every N steps from fresh
  generator draws, so the model keeps seeing new phrasings without a per-batch
  pipeline. Not fully "live", but it keeps a GPU fed with non-repeating data and
  reduces overfit on a fixed sample.
- The real diversity ceiling is the **template set** — the highest-leverage change is
  always *more phrasings* (`_aug()` + the per-family `random.choice([...])` lists), or
  an LLM teacher, not a bigger static dump of the same templates.

---

## Where things live

| Concern | Path |
|---|---|
| Model architecture / training | `train/model.py`, `train/train.py` |
| Data templates + augmentation | `train/gen_data.py` (`_aug`, `DEVICE_FAMILIES`) |
| Curate / validate / distill | `train/curate.py` |
| Frozen embeddings / POS types | `train/build_embeddings.py`, `train/build_types.py` |
| Uncertainty mining / flywheel | `train/eval.py --harvest`, `train/flywheel.py` |
| Export / flash image (PSM1) | `train/export_v2.py` |
| Browser WASM inference | `crates/tinyllm-wasm`, `crates/tinyllm` |
| Agent web app + fast-paths | `demo/esp32-tailscale-pyspell/src/pyspell_web.rs` |
| Model host (stream off flash) | `demo/esp32-tailscale-pyspell/src/model_host.rs` |
| Device actions (screen/LED) | `…/src/display.rs`, `…/src/actuator.rs`, `…/src/ui.rs` |
| In-tunnel streaming TCP | `demo/esp32-tailscale-pyspell/core/src/tcp.rs` |
| 512 kB network/TLS tricks | `docs/memory-512kb.md` |
