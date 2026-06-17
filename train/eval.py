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
import gen_data
from model import Config, Llama
from sample import generate


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242, help="!= train seed -> unseen values")
    ap.add_argument("--show", type=int, default=12, help="sample failures to print")
    ap.add_argument("--dump", action="store_true", help="write failures to data/hard.jsonl")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bestp = os.path.join(args.out, "best.pt")
    ck = torch.load(bestp if os.path.exists(bestp) else os.path.join(args.out, "ckpt.pt"),
                    map_location=device, weights_only=False)
    model = Llama(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tk = bpemod.BPE.load_json(os.path.join(args.out, "bpe.json"))
    print(f"eval model step {ck['step']} val {ck.get('best_val', float('nan')):.3f}")

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
        out = norm(generate(model, tk, device, en, max_new=64, temperature=0.0))
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
