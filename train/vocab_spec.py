"""The forced-vocabulary spec, factored out of train.py so torch-free tools (the local
embedding precompute, parity checks) can build the exact same BPE the trainer uses.

Whole instruction words are forced (bare + leading-space form, since encode prepends a
space) so the ~512 vocab is dominated by real words, not template fragments. Plus the
Python structural tokens and the delexicalization slot markers."""

import delex

WORDS = [
    "print", "say", "output", "display", "add", "subtract", "multiply", "divide",
    "compute", "what", "is", "set", "let", "assign", "define", "function", "returns",
    "write", "make", "create", "list", "sum", "find", "largest", "smallest", "length",
    "uppercase", "lowercase", "word", "count", "from", "to", "through", "numbers",
    "each", "item", "loop", "over", "square", "cube", "even", "odd", "reverse",
    "backwards", "greater", "bigger", "than", "check", "if", "the", "of", "and", "two",
    "down", "up", "total", "plus", "combine", "in", "for", "else", "while", "result",
    "value", "num", "temp", "nums", "number", "items", "a", "an", "calculate",
    # natural-phrasing augmentation words (please/can you/i want to/…)
    "please", "can", "you", "could", "want", "need", "me", "it", "how", "much",
    "switch", "power", "enable", "disable", "change", "up",
    # string literals for print("…")/show("…") (programming/device-leaning)
    "hello", "world", "code", "Python", "robot", "star", "hi", "done", "ok", "yes", "no",
    # math data-aug words
    "percent", "average", "mean", "power", "remainder", "divided", "round", "places",
    "decimals", "larger", "smaller", "double", "raise", "mod",
    # device words (screen + LED) — let beginners drive the ESP32 in plain English
    "show", "text", "screen", "light", "led", "turn", "on", "off", "flash", "blink",
    "memory", "free", "heap", "uptime", "seconds", "boot", "color", "set",
    "red", "green", "blue", "yellow", "white", "orange", "purple", "pink",
    # string-op phrasing words
    "flip", "sentence", "convert", "around", "are", "your",
]
# Python structural tokens (kept as whole pieces for clean code generation).
PY_TOKENS = [
    "    ", "\n    ", "print(", "def ", "return ", "range(", "for ", " in ", "if ",
    "else", "while ", "):", "()", " = ", " + ", " - ", " * ", " // ", " % ", "len(",
    "sum(", "max(", "min(", "upper(", "lower(", "reverse(", "abs(", "== 0", '("', '")',
    ", ", " % 2", " == ", " > ", "** 2", "** 3", ", 0, -1)", "for i in range(",
    "for item in ", "range", "def", "return",
    "round(", "** ", ") / 2", " % ",
    # device builtins (screen + LED)
    "show(", "led(", "flash()", "led(1)", "led(0)",
    # Edit-mode protocol tokens (anchor-based directives: replace/delete/move/rename).
    "EDIT", " EDIT", "@@ ", " ==> ", "DEL ", "MOVE ", "RENAME ",
    "EXPLAIN ",  # reverse direction: EXPLAIN <code> -> english
]
# Delexicalization slot markers (#0..#7 numbers, &a..&d strings). Force BOTH the bare
# form (after '('/'[' in Python: `max(#0`) and the space-prefixed form (after a word in
# English: `of #0`) so each marker is a SINGLE atomic token in either context — the model
# then learns "carry slot k", never how to spell a literal. See delex.py.
PLACEHOLDERS = delex.NUM_PH + delex.STR_PH

FORCED = (PY_TOKENS + PLACEHOLDERS + [" " + p for p in PLACEHOLDERS]
          + [" " + w for w in WORDS])
