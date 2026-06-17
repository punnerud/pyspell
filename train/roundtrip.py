"""Round-trip back-translation signal: en -> py (forward) then EXPLAIN py_gold -> en'
(reverse). A low en/en' overlap flags an ambiguous/hard instruction; the distinct-ratio
of en' guards against the reverse head collapsing. Requires a model trained with
--reverse-frac (the EXPLAIN direction).

  python roundtrip.py --out out --n 200
"""

import argparse
import os
import re

import torch

import bpe as bpemod
import gen_data
from model import Config, Llama
from sample import generate


def words(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def jaccard(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()
    import random
    random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bp = os.path.join(args.out, "best.pt")
    ck = torch.load(bp if os.path.exists(bp) else os.path.join(args.out, "ckpt.pt"),
                    map_location=device, weights_only=False)
    model = Llama(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tk = bpemod.BPE.load_json(os.path.join(args.out, "bpe.json"))

    sims, en2s, hard = [], [], []
    for _ in range(args.n):
        en, py = gen_data.gen_example(random.randint(0, 21))
        en2 = generate(model, tk, device, "EXPLAIN " + py, max_new=48, temperature=0.0).strip()
        s = jaccard(en, en2)
        sims.append(s)
        en2s.append(en2)
        if s < 0.3:
            hard.append((en, en2))
    distinct = len(set(en2s)) / max(1, len(en2s))
    print(f"reverse round-trip on {args.n}: mean en/en' jaccard {sum(sims)/len(sims):.2f}, "
          f"distinct-ratio {distinct:.2f} {'(OK)' if distinct > 0.5 else '(DEGENERATE - lower --reverse-frac)'}")
    print(f"low-overlap (ambiguous) cases: {len(hard)}")
    for en, en2 in hard[:6]:
        print(f"  en : {en}\n  en': {en2}\n")


if __name__ == "__main__":
    main()
