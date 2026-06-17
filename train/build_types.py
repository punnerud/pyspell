"""Assign a part-of-speech / role TYPE to each vocab piece (no NLP libs).

Deterministic: transcribe gen_data.py's own role groups + small hand sets + heuristics.
Output `out/word_types.json` (a list aligned to bpe.json's `tokens`) and a parallel list
of type *indices* into TYPES. Also reused at train time to fold a type bias into the
frozen input embedding, and later served to the browser.
"""

import json
import os

import bpe as bpemod

# 10 types (index = TypeVec row).
TYPES = ["VERB", "NOUN", "NUM", "STR", "PYKW", "PYFN", "SYM", "PREP", "ART", "OTHER"]

PYKW = {"def", "return", "for", "in", "if", "else", "while", "range"}
PYFN = {"print", "len", "sum", "max", "min", "upper", "lower", "int", "str", "float"}
STR_WORDS = {"hello", "world", "cat", "dog", "lily", "tom", "apple", "tree", "sun",
             "code", "python", "robot"}
VERB = {"print", "say", "output", "display", "add", "subtract", "multiply", "divide",
        "compute", "set", "let", "assign", "define", "write", "make", "create", "find",
        "count", "loop", "reverse", "check", "combine", "square", "cube"}
PREP = {"to", "from", "in", "of", "over", "than", "up", "down", "through", "with", "and"}
ART = {"a", "an", "the", "two", "each"}
NOUN = {"x", "y", "n", "count", "total", "result", "value", "num", "a", "b", "temp",
        "nums", "item", "number", "list", "function", "word", "numbers", "items"}

_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def type_of(piece: bytes) -> str:
    s = piece.decode("latin1")
    t = s.strip()
    if t == "" or s in ("<unk>", "<s>", "</s>"):
        return "SYM"
    if all(c.isdigit() for c in t):
        return "NUM"
    # any non-letter, non-space char -> structural/symbol (e.g. "print(", " = ", "[::-1]",
    # "<0xXX>", "\n    "). Bare alpha words fall through to linguistic types.
    if any((c not in _LETTERS and c != " ") for c in t):
        return "SYM"
    low = t.lower()
    if low in PYKW:
        return "PYKW"
    if low in PYFN:
        return "PYFN"
    if low in STR_WORDS:
        return "STR"
    if low in VERB:
        return "VERB"
    if low in PREP:
        return "PREP"
    if low in ART:
        return "ART"
    if low in NOUN:
        return "NOUN"
    return "OTHER"


def build(tk: "bpemod.BPE"):
    types = [type_of(t) for t in tk.tokens]
    idx = [TYPES.index(x) for x in types]
    return types, idx


def main():
    out = "out"
    tk = bpemod.BPE.load_json(os.path.join(out, "bpe.json"))
    types, idx = build(tk)
    with open(os.path.join(out, "word_types.json"), "w") as f:
        json.dump({"types": types, "type_set": TYPES}, f)
    from collections import Counter
    print("type counts:", dict(Counter(types)))
    print(f"wrote {out}/word_types.json ({len(types)} tokens)")


if __name__ == "__main__":
    main()
