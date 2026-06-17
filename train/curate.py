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


def _compiles(src):
    try:
        compile(src, "<dir>", "exec")
        return True
    except SyntaxError:
        return False


def valid_delete(window, block):
    if not block.startswith("DEL "):
        return False
    anchor = block[4:]
    rows = window.split("\n")
    hits = [i for i, r in enumerate(rows) if anchor and anchor in r]
    if len(hits) != 1 or not (window.isascii() and block.isascii()):
        return False
    return _compiles("\n".join(r for i, r in enumerate(rows) if i != hits[0]))


def valid_rename(window, block):
    m = re.match(r"^RENAME (.*?) ==> (.*)$", block)
    if not m:
        return False
    old, new = m.group(1), m.group(2)
    ident = re.compile(r"^[A-Za-z_]\w*$")
    if not (ident.match(old) and ident.match(new)):
        return False
    if len(re.findall(r"\b" + re.escape(old) + r"\b", window)) < 2:
        return False
    if not (window.isascii() and block.isascii()):
        return False
    return _compiles(re.sub(r"\b" + re.escape(old) + r"\b", new, window))


def valid_move(window, block):
    m = re.match(r"^MOVE (.*?) ==> (.*)$", block)
    if not m:
        return False
    src, dest = m.group(1), m.group(2)
    rows = window.split("\n")
    si = [i for i, r in enumerate(rows) if src and src in r]
    di = [i for i, r in enumerate(rows) if dest and dest in r]
    if len(si) != 1 or len(di) != 1 or si[0] == di[0]:
        return False
    if not (window.isascii() and block.isascii()):
        return False
    s = rows.pop(si[0])
    di2 = [i for i, r in enumerate(rows) if dest in r][0]
    rows.insert(di2 + 1, s)
    return _compiles("\n".join(rows))


def split_directive(py):
    """If `py` is `window + "\\n" + directive`, return (window, directive), else None."""
    if "\n" not in py:
        return None
    window, block = py.rsplit("\n", 1)
    if block.startswith("@@ ") or block.split(" ", 1)[0] in ("DEL", "MOVE", "RENAME"):
        return window, block
    return None


def valid_directive(window, block):
    if block.startswith("@@ "):
        return valid_edit(window, block)
    if block.startswith("DEL "):
        return valid_delete(window, block)
    if block.startswith("RENAME "):
        return valid_rename(window, block)
    if block.startswith("MOVE "):
        return valid_move(window, block)
    return False


def gen_any_edit():
    """Weighted mix of edit ops: replace 45% / delete 15% / rename 18% / move 22%."""
    r = random.random()
    if r < 0.45:
        return gen_data.gen_edit_example()
    if r < 0.60:
        return gen_data.gen_delete_example()
    if r < 0.78:
        return gen_data.gen_rename_example()
    return gen_data.gen_move_example()


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
    ap.add_argument("--boost-families", default=None, help="comma list of fams to boost (else WEAK_FAMILIES)")
    ap.add_argument("--edit-frac", type=float, default=0.0, help="fraction of EDIT rows (0..1)")
    ap.add_argument("--reverse-frac", type=float, default=0.0, help="fraction of EXPLAIN (py->en) rows")
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
        bf = [int(x) for x in args.boost_families.split(",")] if args.boost_families else gen_data.WEAK_FAMILIES
        bulk += [gen_data.gen_example(random.choice(bf)) for _ in range(args.boost)]
        print(f"boosted families {bf} with {args.boost} extra examples")
    edits = []
    for _ in range(n_edit):
        en, window, block = gen_any_edit()
        if valid_directive(window, block):
            edits.append(("EDIT " + en, window + "\n" + block))
    if n_edit:
        print(f"edit rows: {len(edits)} (target {n_edit})")

    allp = bulk + edits + (seeds + extra) * args.oversample
    seen, out = set(), []
    rejected = 0
    for en, py in allp:
        en, py = canon(en, py)
        sd = split_directive(py)
        if sd:  # EDIT row (window + "\n" + directive)
            if not (en.startswith("EDIT ") and valid_directive(*sd)):
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
    # Reverse direction: EXPLAIN <code> -> english (py->en). Appended pre-validated so
    # canon (which collapses instruction whitespace) never mangles the code in `en`.
    if args.reverse_frac > 0:
        n_rev = int(len(out) * args.reverse_frac)
        added = 0
        for _ in range(n_rev * 3):
            if added >= n_rev:
                break
            en_t, code = gen_data.gen_example(random.randint(0, 21))
            if code.isascii() and en_t.isascii() and "EXPLAIN" not in code:
                out.append(("EXPLAIN " + code, en_t))
                added += 1
        print(f"reverse (EXPLAIN) rows: {added}")
    random.shuffle(out)
    n_e = sum(1 for _, py in out if split_directive(py))
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
