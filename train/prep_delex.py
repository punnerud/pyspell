"""Torch-free prep: build the tokenizer (bpe.json + tokenizer.bin), POS types, and the
FROZEN semantic+POS embedding (embed_pca.npz) for the delexicalized vocab — locally, using
numpy + ollama (all-minilm). The GPU trainer (e.g. on Modal) then only needs torch+numpy:
it loads bpe.json + embed_pca.npz as-is (no ollama in the cloud).

  cp out/embed_cache.json out_delex/embed_cache.json   # reuse cached word embeddings
  python prep_delex.py --data data --out out_delex
"""
import argparse
import json
import os

import bpe as bpemod
import build_embeddings
import build_types
from vocab_spec import FORCED


def corpus_texts(jsonl):
    out = []
    with open(jsonl) as f:
        for line in f:
            o = json.loads(line)
            out.append((o["en"] + "\n" + o["py"]).encode("ascii", "ignore").decode("ascii"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out_delex")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    texts = corpus_texts(os.path.join(args.data, "train.jsonl"))
    tk = bpemod.BPE.train(texts, args.vocab, forced=FORCED)
    tk.save_json(os.path.join(args.out, "bpe.json"))
    tk.write_tokenizer_bin(os.path.join(args.out, "tokenizer.bin"))
    print(f"vocab_size={tk.vocab_size}, {len(tk.readable_words())} readable word pieces")

    types, _ = build_types.build(tk)
    json.dump({"types": types, "type_set": build_types.TYPES},
              open(os.path.join(args.out, "word_types.json"), "w"))

    emb = build_embeddings.build(tk, args.out, args.dim)  # uses out/embed_cache.json + ollama
    print(f"frozen embedding {emb.shape} -> {args.out}/embed_pca.npz")
    # sanity: the delex markers must be present + atomic
    for p in ("#0", "#7", "&a", "&d"):
        ids = tk.encode("x " + p)
        assert any(tk.tokens[i].decode("latin1").strip() == p for i in ids), f"{p} not atomic"
    print("delex markers present + atomic in the tokenizer")


if __name__ == "__main__":
    main()
