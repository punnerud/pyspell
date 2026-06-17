"""Cross-check that train/delex.py (training) and web/delex.js (inference) delexicalize
IDENTICALLY — if they ever diverge, the model sees one thing in training and another in
the browser and silently degrades. Generates many examples, runs both implementations,
and asserts the delexed prompt, the slots, and the relexed code all match.

  python parity_delex.py            # needs `node` on PATH
"""
import json
import os
import random
import subprocess
import sys

import delex
import gen_data

N = 4000


def main():
    random.seed(7)
    cases = []
    for _ in range(N):
        en, py = gen_data.gen_example()
        prompt, nums, strs = delex.delex_en(en)
        py2 = delex.delex_py(py, nums, strs)
        back = delex.relex(py2, nums, strs)
        cases.append({"en": en, "py": py, "prompt": prompt, "nums": nums,
                      "strs": strs, "py2": py2, "back": back})

    # 1) Python self-consistency: relex(delex_py) == original py for the COPY families.
    pyfail = [c for c in cases if c["back"] != c["py"]]
    # range/percent now symbolic; remaining mismatches would be a real bug.
    if pyfail:
        for c in pyfail[:5]:
            print("PY ROUND-TRIP MISMATCH:", c["en"], "|", c["py"], "->", c["back"])
        print(f"python round-trip failures: {len(pyfail)}/{N}")
    else:
        print(f"python round-trip: {N}/{N} ok")

    # 2) JS parity: feed the same EN/py2 to web/delex.js and compare.
    harness = r"""
import { delexEn, relex } from '../web/delex.js'
import { readFileSync } from 'fs'
const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'))
let bad = 0, shown = 0
for (const c of cases) {
  const { prompt, nums, strs } = delexEn(c.en)
  const back = relex(c.py2, nums, strs)
  const eq = prompt === c.prompt &&
             JSON.stringify(nums) === JSON.stringify(c.nums) &&
             JSON.stringify(strs) === JSON.stringify(c.strs) &&
             back === c.back
  if (!eq) { bad++; if (shown++ < 5) console.error('JS DIFF', JSON.stringify({en:c.en, py_prompt:c.prompt, js_prompt:prompt, py_nums:c.nums, js_nums:nums, py_back:c.back, js_back:back})) }
}
console.log(bad === 0 ? `js parity: ${cases.length}/${cases.length} ok` : `js parity FAILURES: ${bad}`)
process.exit(bad === 0 ? 0 : 1)
"""
    d = os.path.dirname(os.path.abspath(__file__))
    cjson = os.path.join(d, "_parity_cases.json")
    hpath = os.path.join(d, "_parity_harness.mjs")
    with open(cjson, "w") as f:
        json.dump(cases, f)
    with open(hpath, "w") as f:
        f.write(harness)
    try:
        r = subprocess.run(["node", hpath, cjson], cwd=d, capture_output=True, text=True)
    except FileNotFoundError:
        print("node not found on PATH — skipping JS parity (install Node to run it)")
        return
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    os.remove(cjson)
    os.remove(hpath)
    if pyfail or r.returncode != 0:
        sys.exit(1)
    print("PARITY OK — Python and JS delexicalize identically")


if __name__ == "__main__":
    main()
