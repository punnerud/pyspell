"""Generate an English -> Python instruction dataset (JSONL of {"en","py"}).

Two sources, combined:
  * a rich **template generator** (always on, no deps) covering arithmetic, variables,
    printing, ranges/loops, functions, lists, strings, conditionals, etc.
  * an **optional Qwen teacher** (`--qwen MODEL`, requires a local `ollama` with that
    model pulled) — sequence-level distillation: Qwen writes extra (en, py) pairs.

The browser feeds an English instruction as the prompt; the model is trained to
continue with the Python (BOS + en + "\n" + py + EOS), so generation emits the code.

Usage:
  python gen_data.py --n 30000 --out data
  python gen_data.py --n 30000 --qwen qwen2.5-coder:1.5b --qwen-n 2000 --out data
"""

import argparse
import json
import os
import random


def _int(lo=0, hi=99):
    return random.randint(lo, hi)


def _name():
    return random.choice(["x", "y", "n", "count", "total", "result", "value", "num", "a", "b", "temp"])


def _word():
    return random.choice(["hello", "world", "cat", "dog", "Lily", "Tom", "apple", "tree", "sun", "code", "Python", "robot"])


def _list():
    k = random.randint(2, 5)
    return [random.randint(1, 20) for _ in range(k)]


# Families that the model struggles with most (from eval.py): list aggregations,
# function defs, countdown loops. curate.py can oversample these.
WEAK_FAMILIES = [4, 6, 7, 10]


def gen_example(fam=None):
    """Return one (en, py) pair from a template family (random, or a forced `fam`)."""
    if fam is None:
        fam = random.randint(0, 21)
    if fam == 0:
        w = _word()
        en = random.choice([f"print {w}", f"say {w}", f"output the word {w}", f"display {w}"])
        py = f'print("{w}")'
    elif fam == 1:
        a, b = _int(), _int()
        op, sym, verb = random.choice([("+", "+", "add"), ("-", "-", "subtract"), ("*", "*", "multiply"), ("//", "/", "divide")])
        en = random.choice([f"{verb} {a} and {b}", f"what is {a} {verb} {b}", f"compute {a} {sym} {b}"])
        py = f"print({a} {op} {b})"
    elif fam == 2:
        nm, v = _name(), _int()
        en = random.choice([f"set {nm} to {v}", f"let {nm} be {v}", f"assign {v} to {nm}"])
        py = f"{nm} = {v}"
    elif fam == 3:
        a, b = _int(1, 5), _int(6, 12)
        en = random.choice([f"print numbers from {a} to {b}", f"count from {a} to {b}", f"list numbers {a} through {b}"])
        py = f"for i in range({a}, {b + 1}):\n    print(i)"
    elif fam == 4:
        nm = random.choice(["add", "total", "plus", "combine"])
        en = random.choice([f"define a function {nm} that returns a plus b", f"write a function {nm} that adds two numbers a and b"])
        py = f"def {nm}(a, b):\n    return a + b"
    elif fam == 5:
        lst = _list()
        en = random.choice([f"make a list of {', '.join(map(str, lst))}", f"create a list with {', '.join(map(str, lst))}"])
        py = f"nums = {lst}"
    elif fam == 6:
        lst = _list()
        en = random.choice([f"sum the list {lst}", f"add up the numbers {lst}", f"what is the total of {lst}"])
        py = f"print(sum({lst}))"
    elif fam == 7:
        lst = _list()
        fn, verb = random.choice([("max", "largest"), ("min", "smallest"), ("len", "length")])
        en = random.choice([f"find the {verb} of {lst}", f"the {verb} in {lst}"])
        py = f"print({fn}({lst}))"
    elif fam == 8:
        w = _word()
        meth, verb = random.choice([("upper", "uppercase"), ("lower", "lowercase")])
        en = random.choice([f"{verb} the word {w}", f"make {w} {verb}"])
        py = f'print("{w}".{meth}())'
    elif fam == 9:
        a, b = _int(), _int()
        en = random.choice([f"if {a} is greater than {b} print yes", f"check if {a} is bigger than {b}"])
        py = f"if {a} > {b}:\n    print(\"yes\")"
    elif fam == 10:
        a = _int(3, 8)
        en = random.choice([f"count down from {a} to 1", f"print {a} down to 1"])
        py = f"for i in range({a}, 0, -1):\n    print(i)"
    elif fam == 11:
        lst = [_word() for _ in range(random.randint(2, 3))]
        en = random.choice([f"print each item in the list {lst}", f"loop over {lst} and print them"])
        py = f"for item in {lst}:\n    print(item)"
    elif fam == 12:
        a = _int(2, 12)
        op, verb = random.choice([("** 2", "square"), ("** 3", "cube")])
        en = random.choice([f"{verb} the number {a}", f"what is {a} {verb}d"])
        py = f"print({a} {op})"
    elif fam == 13:
        a = _int()
        en = random.choice([f"check if {a} is even", f"is {a} even or odd"])
        py = f"print(\"even\" if {a} % 2 == 0 else \"odd\")"
    elif fam == 14:
        w = _word()
        en = random.choice([f"reverse the word {w}", f"print {w} backwards"])
        py = f'print("{w}"[::-1])'
    elif fam == 15:  # percent
        a, p = _int(5, 95), random.choice([10, 15, 20, 25, 50])
        en = random.choice([f"what is {p}% of {a}", f"{p} percent of {a}"])
        py = f"print({p / 100} * {a})"
    elif fam == 16:  # average of two
        a, b = _int(), _int()
        en = random.choice([f"average of {a} and {b}", f"the mean of {a} and {b}"])
        py = f"print(({a} + {b}) / 2)"
    elif fam == 17:  # power
        a, b = _int(2, 9), _int(2, 4)
        en = random.choice([f"{a} to the power of {b}", f"raise {a} to {b}"])
        py = f"print({a} ** {b})"
    elif fam == 18:  # modulo / remainder
        a, b = _int(10, 40), _int(2, 9)
        en = random.choice([f"remainder of {a} divided by {b}", f"{a} mod {b}"])
        py = f"print({a} % {b})"
    elif fam == 19:  # rounding
        x, k = round(random.uniform(1, 99), 3), _int(0, 2)
        en = random.choice([f"round {x} to {k} places", f"round {x} to {k} decimals"])
        py = f"print(round({x}, {k}))"
    elif fam == 20:  # larger / smaller of two
        a, b = _int(), _int()
        fn, verb = random.choice([("max", "larger"), ("min", "smaller")])
        en = random.choice([f"the {verb} of {a} and {b}", f"which is {verb}, {a} or {b}"])
        py = f"print({fn}({a}, {b}))"
    else:  # fam 21 — multi-step
        a, b = _int(1, 20), _int(1, 20)
        en = random.choice([f"add {a} and {b} then double it", f"double the sum of {a} and {b}"])
        py = f"print(({a} + {b}) * 2)"
    return en, py


# Math families (for curate --boost and eval grouping).
MATH_FAMILIES = [15, 16, 17, 18, 19, 20, 21]


def gen_edit_example():
    """Return one (en, window, block) edit triple. `block` is a find/replace directive
    `@@ <old> ==> <new>`; applying it to `window` (window.replace(old, new, 1)) yields a
    line that compiles. The model copies only the tiny `old` substring (often a single
    in-vocab token) — the browser does the rest, so list/long content is never emitted."""
    m = random.randint(0, 5)
    if m == 0:  # change a range upper bound (2-line window so it compiles)
        a = _int(1, 5)
        b1, b2 = _int(6, 9), _int(10, 15)
        window = f"for i in range({a}, {b1 + 1}):\n    print(i)"
        old, new = f"{b1 + 1})", f"{b2 + 1})"
        en = random.choice([f"change the upper bound to {b2}", f"make it go up to {b2}",
                            f"count up to {b2}"])
    elif m == 1:  # swap arithmetic operator
        a, b = _int(0, 20), _int(1, 20)
        (o1, v1), (o2, v2) = random.sample(
            [("+", "add"), ("-", "subtract"), ("*", "multiply"), ("//", "divide")], 2)
        window = f"print({a} {o1} {b})"
        old, new = f" {o1} ", f" {o2} "
        en = random.choice([f"make it {v2} instead", f"use {v2}", f"change {v1} to {v2}"])
    elif m == 2:  # rename a variable
        n1, n2 = random.sample(["x", "y", "n", "count", "total", "result", "value", "num", "temp"], 2)
        v = _int(0, 99)
        window = f"{n1} = {v}"
        old, new = f"{n1} =", f"{n2} ="
        en = random.choice([f"rename {n1} to {n2}", f"call it {n2} instead"])
    elif m == 3:  # swap a list builtin (no list copy — only the fn token changes)
        (f1, w1), (f2, w2) = random.sample(
            [("sum", "sum"), ("max", "largest"), ("min", "smallest"), ("len", "length")], 2)
        window = f"print({f1}({_list()}))"
        old, new = f"{f1}(", f"{f2}("
        en = random.choice([f"use the {w2} instead of the {w1}", f"find the {w2} not the {w1}"])
    elif m == 4:  # change a printed word
        s1, s2 = random.sample(["hello", "world", "cat", "dog", "apple", "tree", "sun", "code"], 2)
        window = f'print("{s1}")'
        old, new = f'"{s1}"', f'"{s2}"'
        en = random.choice([f"print {s2} instead", f"change the word to {s2}"])
    else:  # turn a counting-up loop into a countdown (2-line window so it compiles)
        a = _int(3, 9)
        window = f"for i in range(1, {a + 1}):\n    print(i)"
        old, new = f"range(1, {a + 1})", f"range({a}, 0, -1)"
        en = random.choice(["count down instead", "reverse the direction", "go downwards"])
    return en, window, f"@@ {old} ==> {new}"


def _unique_anchor(window, line):
    """Shortest prefix of `line` (stripped) that occurs in exactly one window line —
    keeps the anchor the model must copy as small as possible."""
    rows = window.split("\n")
    target = line.strip()
    for end in range(2, len(target) + 1):
        cand = target[:end]
        if sum(1 for r in rows if cand in r) == 1:
            return cand
    return target


def gen_delete_example():
    """Delete a whole line: `DEL <anchor>`; removing it must still compile."""
    templates = [
        ("total = 0\nfor i in nums:\n    total = total + i", "total = 0",
         ["remove the initializer", "delete the total line"]),
        ('print("debug")\nprint(result)', 'print("debug")',
         ["delete the debug print", "remove the debug line"]),
        ("x = 5\ny = 10\nprint(x + y)", "y = 10", ["remove the y line", "delete the y line"]),
        ("a = 1\nb = 2\nc = 3\nprint(a + c)", "b = 2", ["delete the b line", "remove b"]),
    ]
    window, line, instrs = random.choice(templates)
    return random.choice(instrs), window, f"DEL {_unique_anchor(window, line)}"


def gen_rename_example():
    """Global rename a name that appears 2-3 times: `RENAME <old> ==> <new>`."""
    n1, n2 = random.sample(["count", "total", "x", "y", "result", "value", "num"], 2)
    forms = [
        f"{n1} = 0\n{n1} = {n1} + 1\nprint({n1})",
        f"{n1} = 5\nprint({n1})\nprint({n1} * 2)",
        f"def {n1}(a, b):\n    return a + b\nprint({n1}(1, 2))",
    ]
    window = random.choice(forms)
    instrs = [f"rename {n1} to {n2}", f"call {n1} {n2} everywhere", f"change every {n1} to {n2}"]
    return random.choice(instrs), window, f"RENAME {n1} ==> {n2}"


def gen_move_example():
    """Move a line after another: `MOVE <anchor> ==> <dest>` (reorder independent stmts)."""
    (na, va), (nb, vb) = random.sample([("x", "5"), ("y", "10"), ("z", "3"), ("n", "7")], 2)
    window = f"{na} = {va}\n{nb} = {vb}\nprint({na} + {nb})"
    instrs = [f"move the {na} line after the {nb} line", f"put {na} below {nb}"]
    src = _unique_anchor(window, f"{na} = {va}")
    dest = _unique_anchor(window, f"{nb} = {vb}")
    return random.choice(instrs), window, f"MOVE {src} ==> {dest}"


def qwen_examples(model: str, n: int):
    """Optional: ask a local ollama Qwen model for extra (en, py) pairs."""
    import urllib.request
    seeds = [
        "string manipulation", "list comprehension", "dictionary use", "simple math",
        "a small loop", "a tiny function", "conditionals", "reading user input",
    ]
    prompt = (
        "Generate {k} short, beginner Python tasks as JSONL. Each line: "
        '{{"en": "<one-line English instruction>", "py": "<short correct Python, may use \\n>"}}. '
        "Topic: {topic}. Output ONLY JSONL, no prose."
    )
    out = []
    per = max(5, n // len(seeds))
    for topic in seeds:
        if len(out) >= n:
            break
        body = json.dumps({
            "model": model,
            "prompt": prompt.format(k=per, topic=topic),
            "stream": False,
            "options": {"temperature": 0.8},
        }).encode()
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())["response"]
        except Exception as e:
            print(f"  qwen unavailable ({e}); skipping teacher augmentation")
            break
        for line in resp.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                o = json.loads(line)
                if isinstance(o.get("en"), str) and isinstance(o.get("py"), str) and o["en"] and o["py"]:
                    out.append((o["en"].strip(), o["py"].rstrip()))
            except Exception:
                pass
    print(f"  qwen produced {len(out)} pairs")
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="template examples")
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--qwen", type=str, default=None, help="ollama model for teacher pairs")
    ap.add_argument("--qwen-n", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    pairs = [gen_example() for _ in range(args.n)]
    if args.qwen:
        pairs += qwen_examples(args.qwen, args.qwen_n)
    random.shuffle(pairs)

    val = pairs[: args.val]
    train = pairs[args.val:]
    for name, rows in (("train", train), ("val", val)):
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for en, py in rows:
                f.write(json.dumps({"en": en, "py": py}) + "\n")
        print(f"wrote {path}: {len(rows)} examples")


if __name__ == "__main__":
    main()
