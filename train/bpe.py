"""A small BPE tokenizer whose ENCODE is a faithful port of the on-device
`tinyllm` tokenizer (tokenizer.rs), so training-time and browser-time tokenization
are identical. Verified against the Rust runtime by `examples/tok_encode.rs`.

Vocab layout (tinyllm-compatible): id 0 `<unk>`, 1 `<s>`(BOS), 2 `</s>`(EOS),
ids 3..258 the byte tokens `<0xXX>` (decode + rare-byte fallback), then learned
pieces (single chars first, then merges) as literal bytes with scores. tinyllm's
greedy "merge the highest-scoring adjacent pair present in vocab" reproduces our
tokenization because we assign earlier (more frequent) merges higher scores.
"""

import json
import struct
from collections import Counter

BOS = 1
EOS = 2


def _parse_hex_byte(piece: bytes):
    if len(piece) == 6 and piece[:3] == b"<0x" and piece[5:6] == b">":
        try:
            return int(piece[3:5], 16)
        except ValueError:
            return None
    return None


class BPE:
    def __init__(self, tokens, scores):
        self.tokens = tokens          # list[bytes]
        self.scores = scores          # list[float]
        self.vocab_size = len(tokens)
        self.index = {t: i for i, t in enumerate(tokens)}

    # ---- training ----
    # Frequency merges longer than this (decoded bytes), or that join across more than
    # one internal space (cross-word phrases), are skipped — keeps the vocab made of
    # whole words + short Python tokens instead of dead template phrases.
    MAX_MERGE_LEN = 12

    @classmethod
    def train(cls, texts, vocab_size, forced=()):
        # Base: specials + 256 byte tokens + the single chars seen in the corpus.
        byte_tokens = [f"<0x{b:02X}>".encode() for b in range(256)]
        chars = sorted({ch for t in texts for ch in t})
        base = [c.encode() for c in chars]
        # Seed a few must-have multi-char pieces (Python tokens) so they exist even if
        # rare; BPE will also discover frequent ones.
        forced = [f.encode() for f in forced]

        tokens = [b"<unk>", b"<s>", b"</s>"] + byte_tokens + base
        scores = [0.0, 0.0, 0.0] + [0.0] * 256 + [0.0] * len(base)
        index = {t: i for i, t in enumerate(tokens)}

        # Work on unique sequences with counts (templated data has few uniques).
        freq = Counter(texts)
        words = {t: [c for c in t] for t in freq}  # list of single-char strings
        rank = 0
        budget = vocab_size - len(tokens)

        def add(piece_str):
            nonlocal rank
            b = piece_str.encode()
            if b in index:
                return
            index[b] = len(tokens)
            tokens.append(b)
            scores.append(-(rank + 1.0))  # earlier merge -> higher (less negative) score
            rank += 1

        # Force the must-haves first (highest priority merges).
        for f in forced:
            s = f.decode()
            if len(s) >= 2 and budget > 0:
                add(s)
                budget -= 1

        def acceptable(piece):
            if len(piece) > cls.MAX_MERGE_LEN:
                return False
            # at most one internal space (a single " word" piece is fine; phrases aren't)
            return piece.strip().count(" ") == 0

        while budget > 0:
            pairs = Counter()
            for t, seq in words.items():
                c = freq[t]
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += c
            if not pairs:
                break
            # most frequent pair whose merge is acceptable (whole-word / short Python token)
            a = b = None
            for (pa, pb), _ in pairs.most_common():
                if acceptable(pa + pb):
                    a, b = pa, pb
                    break
            if a is None:
                break
            merged = a + b
            add(merged)
            budget -= 1
            # apply the merge everywhere
            for t in words:
                seq = words[t]
                out = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                words[t] = out
        return cls(tokens, scores)

    # ---- encode (faithful tinyllm port) ----
    def _find(self, b: bytes):
        return self.index.get(b)

    def encode(self, text, bos=False, eos=False):
        toks = []
        if bos:
            toks.append(BOS)
        if text:
            sid = self._find(b" ")
            if sid is not None:
                toks.append(sid)
        for ch in text:
            s = ch.encode("utf-8")
            cid = self._find(s)
            if cid is not None:
                toks.append(cid)
            else:
                for byte in s:
                    i = byte + 3
                    toks.append(i if i < self.vocab_size else 0)
        # greedy merge by score
        while True:
            best_score = -1e30
            best_id = None
            best_idx = None
            for i in range(len(toks) - 1):
                cat = self.tokens[toks[i]] + self.tokens[toks[i + 1]]
                cid = self._find(cat)
                if cid is not None and self.scores[cid] > best_score:
                    best_score = self.scores[cid]
                    best_id = cid
                    best_idx = i
            if best_idx is None:
                break
            toks[best_idx] = best_id
            del toks[best_idx + 1]
        if eos:
            toks.append(EOS)
        return toks

    def decode(self, ids):
        out = bytearray()
        prev = BOS
        for tid in ids:
            if tid < 0 or tid >= self.vocab_size:
                continue
            piece = self.tokens[tid]
            if prev == BOS and piece[:1] == b" ":
                piece = piece[1:]
            hb = _parse_hex_byte(piece)
            out += bytes([hb]) if hb is not None else piece
            prev = tid
        return out.decode("utf-8", "replace")

    # ---- validation ----
    def unknown_ratio(self, text):
        """Fraction of chars that fall to byte-fallback (not in the learned vocab) —
        an input-distribution check."""
        if not text:
            return 0.0
        miss = sum(1 for ch in text if self._find(ch.encode()) is None)
        return miss / len(text)

    def readable_words(self):
        """Learned multi-char pieces that are printable ASCII (for an input validator
        word list)."""
        words = []
        for t in self.tokens[259:]:
            if len(t) >= 2 and all(32 <= c < 127 for c in t):
                words.append(t.decode())
        return words

    # ---- persistence ----
    def write_tokenizer_bin(self, path):
        max_len = max(len(t) for t in self.tokens)
        out = bytearray(struct.pack("<i", max_len))
        for t, s in zip(self.tokens, self.scores):
            out += struct.pack("<f", float(s))
            out += struct.pack("<i", len(t))
            out += t
        with open(path, "wb") as f:
            f.write(out)
        return bytes(out)

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump({"tokens": [t.decode("latin1") for t in self.tokens], "scores": self.scores}, f)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls([t.encode("latin1") for t in d["tokens"]], d["scores"])
