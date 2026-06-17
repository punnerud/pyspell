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


def gen_example():
    """Return one (en, py) pair from a randomly chosen template family."""
    fam = random.randint(0, 14)
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
    else:
        w = _word()
        en = random.choice([f"reverse the word {w}", f"print {w} backwards"])
        py = f'print("{w}"[::-1])'
    return en, py


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
