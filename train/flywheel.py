"""Automated active-learning loop: eval+harvest the champion -> curate targeted data
(boost the data-fixable hard families + edits + reverse) -> train a candidate -> gate it
against the champion -> promote or roll back. Resumable via flywheel_state.json.

  python flywheel.py --rounds 3 --minutes 10

Gates (candidate must meet to be promoted): generate struct >= max(82, champ-1),
edit@@ struct >= 79. Otherwise the champion is kept (rollback). Structural families
(0,6,7 — number/list copy) are never boosted (not data-fixable).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

PY = sys.executable
STRUCT = re.compile(r"struct\s*:\s*\d+/\d+\s*\((\d+)%\)")
STRUCTURAL = {0, 6, 7}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def struct_pct(out_dir, task):
    r = run([PY, "eval.py", "--out", out_dir, "--task", task, "--n", "400", "--show", "0"])
    m = STRUCT.search(r.stdout)
    return int(m.group(1)) if m else -1


def harvest(out_dir):
    run([PY, "eval.py", "--out", out_dir, "--harvest", "--n", "400"])
    h = json.load(open(os.path.join(out_dir, "harvest.json")))
    return [f for f in h.get("boost_families", []) if f not in STRUCTURAL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--out", default="out", help="champion dir")
    ap.add_argument("--reverse-frac", type=float, default=0.12)
    args = ap.parse_args()
    champ = args.out

    base_gen = struct_pct(champ, "generate")
    base_edit = struct_pct(champ, "edit")
    print(f"champion: generate {base_gen}% edit@@ {base_edit}%", flush=True)

    for rnd in range(1, args.rounds + 1):
        fams = harvest(champ)
        bf = ",".join(map(str, fams))
        print(f"\n=== round {rnd}: boost data-fixable families [{bf}] ===", flush=True)
        cur = [PY, "curate.py", "--n", "28000", "--edit-frac", "0.3",
               "--reverse-frac", str(args.reverse_frac), "--boost", "10000",
               "--val", "600", "--out", "data"]
        if bf:
            cur += ["--boost-families", bf]
        run(cur)

        cand = f"out_r{rnd}"
        shutil.rmtree(cand, ignore_errors=True)
        os.makedirs(cand)
        cache = os.path.join(champ, "embed_cache.json")
        if os.path.exists(cache):
            shutil.copy(cache, os.path.join(cand, "embed_cache.json"))  # avoid re-embedding
        run([PY, "train.py", "--preset", "full512", "--out", cand,
             "--max-minutes", str(args.minutes), "--eval-interval", "500",
             "--ckpt-interval", "1000", "--patience", "12", "--decay-iters", "30000"])

        cg = struct_pct(cand, "generate")
        ce = struct_pct(cand, "edit")
        passed = cg >= max(82, base_gen - 1) and ce >= 79
        print(f"round {rnd}: candidate generate {cg}% edit@@ {ce}% "
              f"(champion {base_gen}/{base_edit}) -> {'PROMOTE' if passed else 'ROLLBACK'}", flush=True)
        if passed:
            shutil.rmtree(champ + "_prev", ignore_errors=True)
            shutil.move(champ, champ + "_prev")
            shutil.move(cand, champ)
            base_gen, base_edit = cg, ce
        else:
            shutil.rmtree(cand, ignore_errors=True)
        json.dump({"round": rnd, "champ_gen": base_gen, "champ_edit": base_edit,
                   "promoted": passed}, open("flywheel_state.json", "w"), indent=1)

    print(f"\ndone. champion: generate {base_gen}% edit@@ {base_edit}%. "
          f"Export+flash with: python export_v2.py --out {champ}")


if __name__ == "__main__":
    main()
