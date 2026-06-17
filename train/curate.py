"""Curate + distill + validate the training data ("compressor").

Bulk coverage from gen_data's strict templates, plus hand seeds (seeds.jsonl) oversampled
so natural phrasing dominates, plus optional Qwen paraphrase of the English (py kept).
Every pair is CANONICALIZED + VALIDATED: Python must compile, text/code ASCII-clean,
dedup. Writes data/train.jsonl + data/val.jsonl (same format train.py consumes).

  python curate.py --n 30000 --out data
  python curate.py --n 30000 --qwen qwen2.5-coder:1.5b --out data
"""

import argparse
import json
import os
import random
import re

import gen_data

EDIT_RE = re.compile(r"^@@ (.*?) ==> (.*)$")


def canon(en, py):
    en = " ".join(en.encode("ascii", "ignore").decode("ascii").split())
    py = py.replace("\t", "    ").rstrip()
    py = "\n".join(line.rstrip() for line in py.split("\n"))
    return en, py


def valid(en, py):
    if not en or not py:
        return False
    if en.encode("ascii", "ignore").decode("ascii") != en:
        return False
    if py.encode("ascii", "ignore").decode("ascii") != py:
        return False
    try:
        compile(py, "<curate>", "exec")
    except SyntaxError:
        return False
    return True


def valid_edit(window, block):
    """An edit row's body is `window + "\\n" + block` where block is `@@ old ==> new`.
    Valid iff old occurs in the window, no marker collisions, and the spliced line
    compiles."""
    m = EDIT_RE.match(block)
    if not m:
        return False
    old, new = m.group(1), m.group(2)
    if not old or old not in window:
        return False
    if any(s in (window + new) for s in ("@@", "==>")):
        return False
    if not (window.isascii() and block.isascii()):
        return False
    try:
        compile(window.replace(old, new, 1), "<edit>", "exec")
    except SyntaxError:
        return False
    return True


def load_seeds(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    out.append((o["en"], o["py"]))
    return out


def qwen_paraphrase(seeds, model, per=2):
    """Ask a local ollama model for English paraphrases of each seed (py kept fixed)."""
    import urllib.request
    out = []
    for en, py in seeds:
        prompt = (f'Rewrite this instruction in {per} different natural ways people might '
                  f'phrase it. One per line, no numbering, keep the same meaning:\n{en}')
        body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.8}}).encode()
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())["response"]
        except Exception as e:
            print(f"  qwen unavailable ({e}); skipping paraphrase")
            break
        for line in resp.splitlines():
            t = line.strip(" -*0123456789.").strip()
            if t:
                out.append((t, py))
    print(f"  qwen paraphrases: {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="bulk template examples")
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seeds", default="seeds.jsonl")
    ap.add_argument("--oversample", type=int, default=8, help="repeat seeds N×")
    ap.add_argument("--boost", type=int, default=0, help="extra examples from weak families")
    ap.add_argument("--edit-frac", type=float, default=0.0, help="fraction of EDIT rows (0..1)")
    ap.add_argument("--qwen", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    seeds = load_seeds(args.seeds)
    print(f"seeds: {len(seeds)}")
    extra = qwen_paraphrase(seeds, args.qwen) if args.qwen else []
    n_edit = int(args.n * args.edit_frac)
    bulk = [gen_data.gen_example() for _ in range(args.n - n_edit)]
    if args.boost:
        bulk += [gen_data.gen_example(random.choice(gen_data.WEAK_FAMILIES)) for _ in range(args.boost)]
        print(f"boosted weak families with {args.boost} extra examples")
    edits = []
    for _ in range(n_edit):
        en, window, block = gen_data.gen_edit_example()
        if valid_edit(window, block):
            edits.append(("EDIT " + en, window + "\n" + block))
    if n_edit:
        print(f"edit rows: {len(edits)} (target {n_edit})")

    allp = bulk + edits + (seeds + extra) * args.oversample
    seen, out = set(), []
    rejected = 0
    for en, py in allp:
        en, py = canon(en, py)
        if "\n@@ " in py:  # EDIT row (window + "\n" + block)
            window, block = py.rsplit("\n", 1)
            if not (en.startswith("EDIT ") and valid_edit(window, block)):
                rejected += 1
                continue
        elif not valid(en, py):
            rejected += 1
            continue
        k = en + "\x00" + py
        if k in seen:
            continue
        seen.add(k)
        out.append((en, py))
    random.shuffle(out)
    n_e = sum(1 for _, py in out if "\n@@ " in py)
    print(f"kept {len(out)} unique valid pairs ({n_e} edit, rejected {rejected})")

    val, train = out[: args.val], out[args.val:]
    for name, rows in (("train", train), ("val", val)):
        path = os.path.join(args.out, f"{name}.jsonl")
        with open(path, "w") as f:
            for en, py in rows:
                f.write(json.dumps({"en": en, "py": py}) + "\n")
        print(f"wrote {path}: {len(rows)}")


if __name__ == "__main__":
    main()
