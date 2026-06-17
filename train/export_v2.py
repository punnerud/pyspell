"""Export a trained checkpoint to the on-device v2 (Q8_0) format and pack the dongle
image (`TOC + tokenizer.bin + model`), ready for `espflash write-bin 0x810000`.

The byte layout + quantization mirror `tinyllm` exactly (format.rs / math.rs):
header (256 B), then f32 rms_att/rms_ffn/rms_final, then Q8 tensors (q_tokens, then
wq/wk/wv/wo/w1/w2/w3 per layer; wcls only if not shared). Each Q8 tensor = N int8
values then N/gs f32 scales, scale = max|x|/127, q = round-half-away(x/scale).
"""

import argparse
import os
import struct

import numpy as np
import torch

from model import Config, Llama

MAGIC = 0x616B3432  # "ak42"


def quant_q8(x: np.ndarray, gs: int) -> bytes:
    x = x.astype(np.float32).reshape(-1)
    assert x.size % gs == 0, f"len {x.size} not a multiple of gs {gs}"
    g = x.reshape(-1, gs)
    wmax = np.abs(g).max(axis=1)                       # [num_groups]
    scale = wmax / 127.0
    safe = np.where(scale == 0.0, 1.0, scale)[:, None]
    q = np.sign(g) * np.floor(np.abs(g / safe) + 0.5)  # round-half-away-from-zero
    q = np.where(scale[:, None] == 0.0, 0.0, q)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.tobytes() + scale.astype(np.float32).tobytes()


def f32_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().to(torch.float32).numpy().reshape(-1).tobytes()


def f32_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().to(torch.float32).numpy()


def export(model: Llama, gs: int) -> bytes:
    c = model.c
    shared = model.lm_head.weight.data_ptr() == model.tok.weight.data_ptr()

    header = struct.pack(
        "<Ii7iBi", MAGIC, 2, c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
        c.vocab_size, c.seq_len, 1 if shared else 0, gs,
    )
    out = bytearray(header)
    out += bytes(256 - len(out))  # pad header to 256

    # fp32 norms (all layers concatenated, then final).
    out += b"".join(f32_bytes(blk.att_norm) for blk in model.blocks)
    out += b"".join(f32_bytes(blk.ffn_norm) for blk in model.blocks)
    out += f32_bytes(model.final_norm)

    # Q8 token embedding, then each weight group across all layers, then wcls (if untied).
    out += quant_q8(f32_np(model.tok.weight), gs)
    for attr in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        for blk in model.blocks:
            out += quant_q8(f32_np(getattr(blk, attr).weight), gs)
    if not shared:
        out += quant_q8(f32_np(model.lm_head.weight), gs)  # separate learned classifier
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="dir with ckpt.pt")
    ap.add_argument("--image", default=None, help="output image path (default out/model.img)")
    ap.add_argument("--emit-web", default=None, metavar="DIR",
                    help="also write DIR/model.bin + DIR/tokenizer.bin for the Pages/WASM "
                         "demo (the same raw payloads the model.img packs) — no offset fiddling")
    args = ap.parse_args()

    best = os.path.join(args.out, "best.pt")
    ckpt_path = best if os.path.exists(best) else os.path.join(args.out, "ckpt.pt")  # prefer best val
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own ckpt (trusted)
    cfg = Config(**ck["cfg"])
    model = Llama(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path}: step={ck['step']} vocab={cfg.vocab_size} params={model.num_params()/1e6:.3f}M")

    model_bytes = export(model, cfg.group_size)
    with open(os.path.join(args.out, "tokenizer.bin"), "rb") as f:
        tok_bytes = f.read()  # the BPE tokenizer written during training

    # Browser/Pages assets: the SAME raw payloads packed into model.img, written as the
    # two files the WASM demo fetches (web/model.bin + web/tokenizer.bin).
    if args.emit_web:
        os.makedirs(args.emit_web, exist_ok=True)
        with open(os.path.join(args.emit_web, "model.bin"), "wb") as f:
            f.write(model_bytes)
        with open(os.path.join(args.emit_web, "tokenizer.bin"), "wb") as f:
            f.write(tok_bytes)
        print(f"emit-web: wrote {args.emit_web}/model.bin ({len(model_bytes)} B) + "
              f"tokenizer.bin ({len(tok_bytes)} B)")

    # Optional Phase-B blobs: word metadata (tokens+POS types) + the frozen embedding
    # matrix (int8). Served to the browser for input validation + RAG. Present when the
    # full512 (frozen-embedding) pipeline produced bpe.json + word_types.json.
    wordmeta, embed = build_wordmeta_and_embed(args.out, model, cfg)

    image_path = args.image or os.path.join(args.out, "model.img")
    if wordmeta is not None:
        # v2 TOC: magic + version(2) + tok_len + model_len + wordmeta_len + embed_len.
        img = bytearray(b"PSM1")
        img += struct.pack("<IIIII", 2, len(tok_bytes), len(model_bytes), len(wordmeta), len(embed))
        img += tok_bytes + model_bytes + wordmeta + embed
        toc = 24
        extra = f" + wordmeta {len(wordmeta)} B + embed {len(embed)} B"
    else:
        img = bytearray(b"PSM1")
        img += struct.pack("<III", 1, len(tok_bytes), len(model_bytes))
        img += tok_bytes + model_bytes
        toc = 16
        extra = ""
    with open(image_path, "wb") as f:
        f.write(img)
    print(f"wrote {image_path}: TOC {toc} + tokenizer {len(tok_bytes)} B + model {len(model_bytes)} B"
          f"{extra} = {len(img)} B ({len(img)/1024/1024:.2f} MB)")
    print(f"flash with:  espflash write-bin 0x810000 {image_path}")


def build_wordmeta_and_embed(out_dir, model, cfg):
    """Return (wordmeta_json_bytes, embed_int8_bytes) for the browser, or (None, None)
    if the frozen-embedding artifacts aren't present (e.g. the byte-level `full` preset)."""
    import json
    import bpe as bpemod
    bpe_path = os.path.join(out_dir, "bpe.json")
    types_path = os.path.join(out_dir, "word_types.json")
    if not (os.path.exists(bpe_path) and os.path.exists(types_path)):
        return None, None
    tk = bpemod.BPE.load_json(bpe_path)
    types = json.load(open(types_path))
    emb = f32_np(model.tok.weight)  # the frozen folded embedding the model uses
    scale = float(np.abs(emb).max() / 127.0) or 1.0
    q = np.clip(np.round(emb / scale), -127, 127).astype(np.int8)
    meta = {
        "tokens": [t.decode("latin1") for t in tk.tokens],
        "types": types["types"],
        "type_set": types["type_set"],
        "dim": cfg.dim,
        "vocab": cfg.vocab_size,
        "embed_scale": scale,
    }
    return json.dumps(meta).encode("utf-8"), q.tobytes()


if __name__ == "__main__":
    main()
