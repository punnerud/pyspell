"""Build the FROZEN input embedding matrix [vocab, dim] for the model.

Pipeline:
  * all-minilm (ollama) embedding (384-d) of each alpha vocab piece's trimmed text
    (cached); symbols/digits/byte/special get no semantic vector.
  * PCA 384 -> dim with numpy SVD (center on real rows, top-`dim` right singular vectors).
  * Fold semantic + a fixed per-TYPE vector. Tokens WITHOUT a semantic vector (digits,
    symbols) instead get a distinct frozen per-token random vector + the type bias, so
    e.g. every digit has a *different* embedding (required to copy numbers).
  * Row-normalize to L2 = 1 (RMSNorm-friendly scale).

Reused by train.py (set_frozen_embedding) and export. Compute once; cache to out/.
"""

import json
import os
import urllib.request

import numpy as np

import build_types

_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _has_letters(piece: bytes) -> bool:
    return any(chr(b) in _LETTERS for b in piece)


def _is_placeholder(piece: bytes) -> bool:
    """Delexicalization slot markers (#0.. / &a..) carry a letter (a/b/c/d) but are NOT
    words — they get the non-semantic per-token + NUM/STR type embedding, like digits."""
    t = piece.decode("latin1").strip()
    return (len(t) == 2 and ((t[0] == "#" and t[1].isdigit())
                             or (t[0] == "&" and t[1].isalpha())))


def _embed_text(piece: bytes) -> str:
    return piece.decode("latin1").strip()


def ollama_embed(text, cache):
    if text in cache:
        return cache[text]
    body = json.dumps({"model": "all-minilm", "prompt": text}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        emb = json.loads(r.read())["embedding"]
    cache[text] = emb
    return emb


def type_vectors(n_types, dim, seed=0):
    """Fixed, ~orthonormal per-type vectors (Gram-Schmidt, row L2=1)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_types, dim)).astype(np.float32)
    # Gram-Schmidt
    for i in range(n_types):
        for j in range(i):
            M[i] -= np.dot(M[i], M[j]) * M[j]
        M[i] /= np.linalg.norm(M[i]) + 1e-8
    return M


def build(tk, out_dir, dim):
    cache_path = os.path.join(out_dir, "embed_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    V = tk.vocab_size
    E = np.zeros((V, 384), dtype=np.float32)
    has_sem = np.zeros(V, dtype=bool)
    for i, piece in enumerate(tk.tokens):
        if _has_letters(piece) and not _is_placeholder(piece):
            txt = _embed_text(piece)
            if txt:
                E[i] = np.array(ollama_embed(txt, cache), dtype=np.float32)
                has_sem[i] = True
    json.dump(cache, open(cache_path, "w"))

    # PCA 384 -> dim on the semantic rows.
    mu = E[has_sem].mean(axis=0)
    Ec = E - mu
    Ec[~has_sem] = 0.0
    _, _, Vt = np.linalg.svd(Ec[has_sem], full_matrices=False)
    P = Vt[:dim].T                      # [384, dim]
    Z = Ec @ P                          # [V, dim]
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Znorm = np.where(norms > 1e-6, Z / np.maximum(norms, 1e-6), 0.0).astype(np.float32)

    # Type bias + per-token uniqueness for non-semantic tokens.
    _, type_idx = build_types.build(tk)
    TV = type_vectors(len(build_types.TYPES), dim)
    Tvec = TV[np.array(type_idx)]       # [V, dim]
    rng = np.random.default_rng(1)
    PerTok = rng.standard_normal((V, dim)).astype(np.float32)
    PerTok /= (np.linalg.norm(PerTok, axis=1, keepdims=True) + 1e-8)

    emb = np.where(has_sem[:, None],
                   0.9 * Znorm + 0.4 * Tvec,        # words: semantic + type
                   0.9 * PerTok + 0.4 * Tvec)       # digits/symbols: distinct + type
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)  # row L2 = 1
    emb = emb.astype(np.float32)

    np.savez(os.path.join(out_dir, "embed_pca.npz"),
             emb=emb, Znorm=Znorm, has_sem=has_sem, type_idx=np.array(type_idx))
    return emb


def load_or_build(tk, out_dir, dim):
    npz = os.path.join(out_dir, "embed_pca.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        if d["emb"].shape == (tk.vocab_size, dim):
            return d["emb"].astype(np.float32)
    return build(tk, out_dir, dim)


if __name__ == "__main__":
    import bpe as bpemod
    tk = bpemod.BPE.load_json("out/bpe.json")
    emb = build(tk, "out", 128)
    print(f"built frozen embedding {emb.shape}, row-RMS {np.sqrt((emb**2).sum(1)).mean():.3f}")
