"""Evaluate the model on held-out test data and report real accuracy (not just loss).

Generates a test set (templates with unseen values + the canonical seeds), runs greedy
generation, and scores three ways:
  * exact   — output == expected (normalized)
  * compiles— output is valid Python
  * struct  — matches with numbers/strings masked (did it pick the right TEMPLATE?)
Prints overall + per-structure breakdown + sample failures. With --dump it writes the
failures as extra training examples (hard-example mining) to data/hard.jsonl.

  python eval.py --n 400
  python eval.py --n 400 --dump        # also save failures for the next training run
"""

import argparse
import json
import os
import re
import random

import torch

import bpe as bpemod
import curate
import delex
import gen_data
from model import Config, Llama
from sample import generate, generate_scored

# Delexicalized models (the default pipeline) are prompted with slot markers and emit
# them; the real inference path delexes the prompt and relexes the output. Eval mirrors
# it so accuracy reflects end-to-end behaviour (toggle with --no-delex for a legacy model).
DELEX = True


def gen_delex(model, tk, device, en, **kw):
    """Generate the way the browser/device does: delex the prompt, relex the output."""
    if not DELEX:
        return generate(model, tk, device, en, **kw)
    prompt, nums, strs = delex.delex_en(en)
    raw = generate(model, tk, device, prompt, **kw)
    return delex.relex(raw, nums, strs)


def norm(s):
    return "\n".join(l.rstrip() for l in s.replace("\t", "    ").strip().split("\n"))


def mask(s):
    s = re.sub(r'"[^"]*"', '"S"', s)
    s = re.sub(r"\[[^\]]*\]", "[L]", s)
    s = re.sub(r"\d+", "N", s)
    return s


def compiles(py):
    try:
        compile(py, "<eval>", "exec")
        return True
    except SyntaxError:
        return False


def eval_edit(model, tk, device, n, seed, show):
    """Edit-mode eval: located (anchor present) / applied-exact / applied-struct / compiles."""
    random.seed(seed)
    cases = [gen_data.gen_edit_example() for _ in range(n)]
    cases = [(en, w, b) for en, w, b in cases if curate.valid_edit(w, b)]
    loc = ae = astr = comp = 0
    by, fails = {}, []
    for en, window, block in cases:
        m = curate.EDIT_RE.match(block)
        gold = norm(window.replace(m.group(1), m.group(2), 1))
        out = generate(model, tk, device, "EDIT " + en + "\n" + window, max_new=48, temperature=0.0)
        ed = re.search(r"@@ (.*?) ==> (.*?)(?:\n|$)", out)
        applied = None
        located = bool(ed) and ed.group(1) in window
        if located:
            applied = norm(window.replace(ed.group(1), ed.group(2), 1))
        loc += located
        ae += applied == gold
        s = applied is not None and mask(applied) == mask(gold)
        c = applied is not None and compiles(applied)
        astr += s
        comp += c
        k = mask(window.replace("\n", " ; "))
        st = by.setdefault(k, [0, 0])
        st[1] += 1
        st[0] += s
        if not s:
            fails.append((en, gold, out.strip()))
    tot = len(cases)
    print(f"\n=== {tot} edit tests ===")
    for name, v in (("located", loc), ("exact", ae), ("struct", astr), ("compiles", comp)):
        print(f"{name:8}: {v}/{tot}  ({100*v//tot}%)")
    print("\n=== per-template (struct) ===")
    for k, (ok, t) in sorted(by.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {ok:3}/{t:<3} {100*ok//t:3}%  {k[:55]}")
    print(f"\n=== sample failures ({min(show, len(fails))}) ===")
    for en, gold, out in fails[:show]:
        print(f"  EN  : {en}\n  GOLD: {gold!r}\n  GOT : {out!r}\n")


def apply_directive(window, block):
    """Apply a directive to a window (mirrors the browser/curate); None if not applicable."""
    rows = window.split("\n")
    if block.startswith("@@ "):
        m = re.match(r"^@@ (.*?) ==> (.*)$", block)
        if m:
            for i, r in enumerate(rows):
                if m.group(1) in r:
                    rows[i] = r.replace(m.group(1), m.group(2), 1)
                    return "\n".join(rows)
    elif block.startswith("DEL "):
        hit = [i for i, r in enumerate(rows) if block[4:] in r]
        if len(hit) == 1:
            return "\n".join(r for i, r in enumerate(rows) if i != hit[0])
    elif block.startswith("RENAME "):
        m = re.match(r"^RENAME (.*?) ==> (.*)$", block)
        if m:
            return re.sub(r"\b" + re.escape(m.group(1)) + r"\b", m.group(2), window)
    elif block.startswith("MOVE "):
        m = re.match(r"^MOVE (.*?) ==> (.*)$", block)
        if m:
            si = [i for i, r in enumerate(rows) if m.group(1) in r]
            di = [i for i, r in enumerate(rows) if m.group(2) in r]
            if len(si) == 1 and len(di) == 1 and si[0] != di[0]:
                s = rows.pop(si[0])
                d2 = [i for i, r in enumerate(rows) if m.group(2) in r][0]
                rows.insert(d2 + 1, s)
                return "\n".join(rows)
    return None


def _first_directive(out):
    for l in out.split("\n"):
        if l.startswith("@@ ") or l.split(" ", 1)[0] in ("DEL", "MOVE", "RENAME"):
            return l
    return None


def eval_directive(model, tk, device, n, seed, show, gen, label):
    import curate
    random.seed(seed)
    cases = [gen() for _ in range(n)]
    cases = [(en, w, b) for en, w, b in cases if curate.valid_directive(w, b)]
    loc = ae = astr = comp = 0
    fails = []
    for en, window, block in cases:
        gold = norm(apply_directive(window, block))
        out = generate(model, tk, device, "EDIT " + en + "\n" + window, max_new=48, temperature=0.0)
        line = _first_directive(out)
        applied = apply_directive(window, line) if line else None
        located = applied is not None
        if applied is not None:
            applied = norm(applied)
        loc += located
        ae += applied == gold
        s = applied is not None and mask(applied) == mask(gold)
        c = applied is not None and compiles(applied)
        astr += s
        comp += c
        if not s:
            fails.append((en, gold, out.strip()))
    tot = max(len(cases), 1)
    print(f"\n=== {len(cases)} {label} tests ===")
    for name, v in (("located", loc), ("exact", ae), ("struct", astr), ("compiles", comp)):
        print(f"{name:8}: {v}/{len(cases)}  ({100*v//tot}%)")
    print(f"=== sample failures ({min(show, len(fails))}) ===")
    for en, gold, out in fails[:show]:
        print(f"  EN  : {en}\n  GOLD: {gold!r}\n  GOT : {out!r}\n")


def _scan_literal(piece, in_str, in_brk):
    """Is this token a literal (digit / inside a string / inside a list)? Returns
    (is_literal, new_in_str, new_in_brk) so callers track state across tokens."""
    lit = in_str or in_brk > 0 or any(c.isdigit() for c in piece)
    for c in piece:
        if c == '"':
            in_str = not in_str
        elif c == "[":
            in_brk += 1
        elif c == "]":
            in_brk = max(0, in_brk - 1)
    return lit, in_str, in_brk


def eval_harvest(model, tk, device, n, seed, out_dir, conf_thr=0.5):
    """Uncertainty mining: per-example confidence over NON-literal positions (so the
    structural number/list ceiling isn't chased) + per-family stats. Writes
    harvest.json (boost families) and data/hard.jsonl (GOLD pairs only)."""
    random.seed(seed)
    fam_conf, fam_fail, hard = {}, {}, []
    for _ in range(n):
        fam = random.randint(0, gen_data.N_FAM)
        en, py = gen_data.gen_example(fam)
        prompt, nums, strs = (delex.delex_en(en) if DELEX else (en, [], []))
        text, pieces, confs = generate_scored(model, tk, device, prompt, max_new=64)
        in_str, in_brk, nl = False, 0, []
        for pc, cf in zip(pieces, confs):
            lit, in_str, in_brk = _scan_literal(pc, in_str, in_brk)
            if not lit:
                nl.append(cf)
        cmin = min(nl) if nl else 1.0
        ok = mask(norm(delex.relex(text, nums, strs) if DELEX else text)) == mask(norm(py))
        fam_conf.setdefault(fam, []).append(cmin)
        f = fam_fail.setdefault(fam, [0, 0])
        f[1] += 1
        f[0] += (not ok)
        if (not ok) or cmin < conf_thr:
            hard.append({"en": en, "py": py})
    rows = []
    for fam in sorted(fam_conf):
        cs = fam_conf[fam]
        avg = sum(cs) / len(cs)
        fails, tot = fam_fail[fam]
        rows.append((fam, avg, fails, tot))
    rows.sort(key=lambda r: r[1])
    print("fam  avg_conf(non-literal)  fails")
    for fam, avg, fails, tot in rows:
        print(f"  {fam:2}   {avg:.3f}   {fails}/{tot}")
    # DATA-FIXABLE = low non-literal confidence (the model is unsure → more data helps).
    # STRUCTURAL = confident but failing (can't copy numbers/lists) → don't chase here.
    boost = [fam for fam, avg, fails, tot in rows if avg < 0.85]
    structural = [fam for fam, avg, fails, tot in rows if avg >= 0.85 and fails / tot > 0.3]
    harvest = {"boost_families": boost, "structural_skip": structural, "n_hard": len(hard),
               "conf_thr": conf_thr, "fam_conf": {f: round(a, 3) for f, a, _, _ in rows}}
    os.makedirs("data", exist_ok=True)
    json.dump(harvest, open(os.path.join(out_dir, "harvest.json"), "w"), indent=1)
    with open(os.path.join("data", "hard.jsonl"), "w") as fh:
        for h in hard:
            fh.write(json.dumps(h) + "\n")
    print(f"harvest: boost (data-fixable) {boost}; structural-skip {structural}; "
          f"{len(hard)} gold hard cases -> data/hard.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--harvest", action="store_true", help="uncertainty mining -> harvest.json")
    ap.add_argument("--task", choices=["generate", "edit", "delete", "rename", "move"],
                    default="generate")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242, help="!= train seed -> unseen values")
    ap.add_argument("--show", type=int, default=12, help="sample failures to print")
    ap.add_argument("--dump", action="store_true", help="write failures to data/hard.jsonl")
    ap.add_argument("--no-delex", dest="delex", action="store_false", default=True,
                    help="evaluate a legacy (non-delexicalized) model literally")
    args = ap.parse_args()
    global DELEX
    DELEX = args.delex

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bestp = os.path.join(args.out, "best.pt")
    ck = torch.load(bestp if os.path.exists(bestp) else os.path.join(args.out, "ckpt.pt"),
                    map_location=device, weights_only=False)
    model = Llama(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tk = bpemod.BPE.load_json(os.path.join(args.out, "bpe.json"))
    print(f"eval model step {ck['step']} val {ck.get('best_val', float('nan')):.3f}")

    if args.harvest:
        eval_harvest(model, tk, device, args.n, args.seed, args.out)
        return
    if args.task == "edit":
        eval_edit(model, tk, device, args.n, args.seed, args.show)
        return
    if args.task in ("delete", "rename", "move"):
        gen = {"delete": gen_data.gen_delete_example, "rename": gen_data.gen_rename_example,
               "move": gen_data.gen_move_example}[args.task]
        eval_directive(model, tk, device, args.n, args.seed, args.show, gen, args.task)
        return

    # held-out test set: templates with a fresh seed (unseen values) + the seeds file.
    random.seed(args.seed)
    tests = [gen_data.gen_example() for _ in range(args.n)]
    seeds_path = "seeds.jsonl"
    if os.path.exists(seeds_path):
        with open(seeds_path) as f:
            for line in f:
                o = json.loads(line.strip())
                tests.append((o["en"], o["py"]))

    n = exact = comp = struct = 0
    by_struct = {}   # masked-py -> [correct, total]
    fails = []
    for en, py in tests:
        out = norm(gen_delex(model, tk, device, en, max_new=64, temperature=0.0))
        gold = norm(py)
        e = out == gold
        s = mask(out) == mask(gold)
        c = compiles(out)
        n += 1
        exact += e
        struct += s
        comp += c
        k = mask(gold)
        st = by_struct.setdefault(k, [0, 0])
        st[1] += 1
        st[0] += s
        if not s:
            fails.append((en, gold, out))

    print(f"\n=== {n} tests ===")
    print(f"exact   : {exact}/{n}  ({100*exact/n:.0f}%)")
    print(f"struct  : {struct}/{n}  ({100*struct/n:.0f}%)   (right template, values aside)")
    print(f"compiles: {comp}/{n}  ({100*comp/n:.0f}%)")

    print("\n=== per-template (struct match) ===")
    for k, (ok, tot) in sorted(by_struct.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {ok:3}/{tot:<3} {100*ok//tot:3}%  {k[:60]}")

    print(f"\n=== sample failures ({min(args.show, len(fails))}) ===")
    for en, gold, out in fails[: args.show]:
        print(f"  EN  : {en}")
        print(f"  GOLD: {gold!r}")
        print(f"  GOT : {out!r}\n")

    if args.dump and fails:
        os.makedirs("data", exist_ok=True)
        with open("data/hard.jsonl", "w") as f:
            for en, gold, _ in fails:
                f.write(json.dumps({"en": en, "py": gold}) + "\n")
        print(f"dumped {len(fails)} failures -> data/hard.jsonl (oversample in next train)")


if __name__ == "__main__":
    main()
