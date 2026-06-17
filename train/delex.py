r"""Delexicalization: turn copied literals (numbers, quoted strings) into placeholder
SLOTS so a tiny model never has to *reproduce* them — it only has to learn "a number
goes here, carry slot k forward". The browser/device copies the real literal back into
the slot at decode time ("the model points, the browser copies", now model-driven).

THE CONTRACT (must match the JS at inference, see web/delex.js / index.html / pyspell_web.rs):
  * EN is delexicalized on its own (it is all we have at inference):
      1. quoted strings first  ("..." or '...')  -> &a, &b, ...  (digit-free markers,
         so the number pass below can't see digits inside them), keeping the quote chars.
      2. then numbers (-?\d+(\.\d+)?)             -> #0, #1, ...
      slots are assigned in order of FIRST appearance, deduped by value/content (the
      same literal reuses its slot), and ALL occurrences are replaced.
  * PY is delexicalized against the SAME slots (training only): a quoted string / number
    that matches a slot value becomes that slot's marker; anything else stays literal
    (structural constants like the 2 in `% 2`, the 100 in `/ 100`).
  * relex(code, nums, strs) substitutes the markers back -> runnable code.

Because EN-delex is self-contained, training and inference produce identical prompts.
Derived constants (e.g. `range(a, b+1)`, `20% -> 0.2`) would break the contract, so
gen_data.py emits those symbolically (`range(a, b + 1)`, `p / 100 * a`) — every operand
is then a pure copy.
"""

import re

# Slot markers. NUMs carry digits (inserted last, so nothing rescans them); STRs are
# digit-free letters (inserted first, so the number pass never matches inside them).
NUM_SLOTS = 8
STR_SLOTS = 4
NUM_PH = [f"#{i}" for i in range(NUM_SLOTS)]          # #0 .. #7
STR_PH = [f"&{chr(ord('a') + i)}" for i in range(STR_SLOTS)]  # &a .. &d

_STR_RE = re.compile(r"(['\"])(.*?)\1")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Markers, longest-first, for relex (all length 2 here, but be safe).
_PH_RE = re.compile("|".join(re.escape(p) for p in sorted(NUM_PH + STR_PH, key=len, reverse=True)))


def delex_en(en):
    """Delexicalize an English instruction on its own.
    Returns (prompt, nums, strs) where nums[k] is the literal for #k, strs[k] for &<k>."""
    nums, strs = [], []

    def _str_sub(m):
        q, content = m.group(1), m.group(2)
        if content not in strs:
            if len(strs) >= STR_SLOTS:
                return m.group(0)  # overflow: leave literal
            strs.append(content)
        return q + STR_PH[strs.index(content)] + q

    s = _STR_RE.sub(_str_sub, en)

    def _num_sub(m):
        v = m.group(0)
        if v not in nums:
            if len(nums) >= NUM_SLOTS:
                return v
            nums.append(v)
        return NUM_PH[nums.index(v)]

    s = _NUM_RE.sub(_num_sub, s)
    return s, nums, strs


def delex_py(py, nums, strs):
    """Delexicalize PY against EN's slots: literals that match a slot become its marker;
    everything else (structural constants, derived literals) stays."""
    def _str_sub(m):
        q, content = m.group(1), m.group(2)
        return q + STR_PH[strs.index(content)] + q if content in strs else m.group(0)

    s = _STR_RE.sub(_str_sub, py)

    def _num_sub(m):
        v = m.group(0)
        return NUM_PH[nums.index(v)] if v in nums else v

    return _NUM_RE.sub(_num_sub, s)


def delex_pair(en, py):
    """Delexicalize an (en, py) training pair consistently. Returns (en2, py2)."""
    en2, nums, strs = delex_en(en)
    return en2, delex_py(py, nums, strs)


def relex(code, nums, strs):
    """Substitute slot markers in generated code back to the real literals. Markers the
    model emitted but that have no value (e.g. it over-generated list elements) are
    dropped along with an adjacent comma, so `[17, 15, #2, #3]` -> `[17, 15]` stays valid."""
    def _sub(m):
        p = m.group(0)
        if p[0] == "#":
            i = int(p[1:])
            return nums[i] if i < len(nums) else "\x00"
        i = ord(p[1]) - ord("a")
        return strs[i] if i < len(strs) else "\x00"
    s = _PH_RE.sub(_sub, code)
    s = re.sub(r"\s*,\s*\x00", "", s)
    s = re.sub(r"\x00\s*,\s*", "", s)
    return s.replace("\x00", "")


if __name__ == "__main__":
    # Round-trip + contract self-tests.
    CASES = [
        ("the largest of 46 and 96215", "print(max(46, 96215))"),
        ("which is larger, 5 or 9", "print(max(5, 9))"),
        ("subtract 7 from the sum", "print(7)"),
        ("what is 7 plus 5", "print(7 + 5)"),
        ('show the text "hello world"', 'show("hello world")'),
        ("make a list of 3, 5, 7, 9, 11", "nums = [3, 5, 7, 9, 11]"),
        ("loop over ['hi', 'world'] and print them", "for item in ['hi', 'world']:\n    print(item)"),
        ("20 percent of 50", "print(20 / 100 * 50)"),
        ("print numbers from 1 to 7", "for i in range(1, 7 + 1):\n    print(i)"),
        ("round 12.5 to 2 decimals", "print(round(12.5, 2))"),
        ("the larger of -5 and 9", "print(max(-5, 9))"),
    ]
    ok = True
    for en, py in CASES:
        en2, py2 = delex_pair(en, py)
        # inference re-derives slots from EN alone, then relexes the (delexed) py:
        _, nums, strs = delex_en(en)
        back = relex(py2, nums, strs)
        status = "ok" if back == py else "MISMATCH"
        if back != py:
            ok = False
        print(f"[{status}] {en!r}\n   en2={en2!r}\n   py2={py2!r}\n   back={back!r}\n")
    print("ALL OK" if ok else "FAILURES ABOVE")
