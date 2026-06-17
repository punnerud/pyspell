# train/ — a tiny, focused English→Python model for the dongle (offline browser-WASM)

Trains a **<500 kB** llama2 model that maps an English instruction to a Python snippet,
then exports it to the on-device v2 (Q8_0) format and packs the flash image the dongle
serves. Architecture is **math-identical** to the `tinyllm` runtime (interleaved-pair
RoPE, RMSNorm 1e-5, SwiGLU, GQA, tied classifier), and the tokenizer is a **learned BPE
whose encode is a faithful port of tinyllm's** — verified equal by `tok_encode.rs`, so
the model sees the same tokens in training and in the browser.

## Why this design (focused vocab + strict grammar)
A small, relevant **~1024-piece vocabulary** (common English instruction words + Python
tokens like `def`, `return`, `range(`, `print(`, indentation) makes sequences short and
meaningful, so a sub-500 kB model learns the **strict instruction grammar** well — far
better than byte-level. The vocab doubles as an **input validator**: words/characters
outside it are out-of-distribution (see `validate.py`).

## Honest expectations
Trained ~1 h from scratch on an M3, this is genuinely useful **for the instruction
patterns it's trained on** (arithmetic, print, variables, ranges/loops, small functions,
list/string ops, conditionals) — not a general coder. Broaden it by adding templates in
`gen_data.py` and/or distilling extra pairs from a local Qwen.

## Setup
```bash
cd train
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended preset: `full512` (curated vocab + frozen semantic+POS embeddings)
The `full512` preset (vocab 512, untied learned classifier) loads a **frozen** input
embedding = `0.9·semantic + 0.4·type`, where `semantic` is an `all-minilm` embedding
PCA-compressed to 128 dims and `type` is a fixed per-POS vector. The model only learns
the mapping (and the classifier), so it generalizes across paraphrases at <500 kB.
Prereq: `ollama pull all-minilm` (~46 MB). The byte-level/learned `full` preset still
exists as a fallback.

## 1) Curate data (seeds + templates, validated; optional Qwen paraphrase)
```bash
python curate.py --n 40000 --out data
# optional: natural-phrasing paraphrases of the seeds from a local Qwen:
ollama pull qwen2.5-coder:1.5b
python curate.py --n 40000 --qwen qwen2.5-coder:1.5b --out data
```
`curate.py` mixes `seeds.jsonl` (oversampled) with `gen_data.py` templates, then
**canonicalizes + validates** every pair (Python must `compile`, ASCII-clean, dedup).

## 2) Train (~1 hour, pausable)
```bash
python train.py --preset full512 --max-minutes 60
```
First run trains the BPE vocab (`out/tokenizer.bin` + `out/bpe.json`), builds the POS
dictionary (`out/word_types.json`) and the frozen embedding (`out/embed_pca.npz`, via
ollama, cached), then trains. **Pause / resume:** a checkpoint is saved to `out/ckpt.pt`
every 200 steps and
on Ctrl-C / SIGTERM (atomic). Re-run the *same command* to resume exactly where you
stopped. Letting the Mac sleep is fine. Watch quality meanwhile:
```bash
python sample.py --prompt "print numbers from 1 to 5"
python validate.py "print numbers from 1 to 5"      # input validator
```

## 3) Export + flash
```bash
python export_v2.py --out out                 # -> out/model.img  (<500 kB)
espflash write-bin 0x810000 out/model.img
```
Serving is unchanged — this just replaces the model the dongle hosts. Hard-refresh the
page (assets are cached) and the offline agent runs your model.

## Smoke test (validates the whole pipeline fast)
```bash
python gen_data.py --n 3000 --val 200 --out data
python train.py --preset smoke --vocab 512 --max-steps 50
python export_v2.py --out out --image out/smoke.img
# from repo root: confirm it loads + runs on the real runtime, and encode parity:
cargo run -p tinyllm --example verify_toy_model -- train/out/smoke.img
cargo run -p tinyllm --example tok_encode      -- train/out/tokenizer.bin 504 "add 3 and 5"
```

## Files
- `model.py` — llama2 matching tinyllm (parity-critical: interleaved RoPE, etc.); `tie_classifier` flag.
- `bpe.py` — learned BPE; **encode is a faithful tinyllm port** + `tokenizer.bin` writer + validator + merge cap.
- `build_types.py` — POS/word-type dictionary (→ `out/word_types.json`), no NLP libs.
- `build_embeddings.py` — all-minilm (ollama) → numpy PCA 128 → frozen folded embedding (`out/embed_pca.npz`).
- `gen_data.py` — strict English→Python templates (+ optional Qwen teacher).
- `seeds.jsonl` / `curate.py` — hand seeds + validate/dedup ("compressor") → `data/*.jsonl`.
- `train.py` — MPS training; frozen-embedding aware; checkpoint + auto-resume + signal-safe save.
- `export_v2.py` — checkpoint → v2 int8 (untied wcls when `full512`) → packed `model.img`.
- `sample.py` / `validate.py` — quick generation / input validation.
