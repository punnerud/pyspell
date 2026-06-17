"""Input validator: check an English instruction against the model's vocabulary.

The model has a small, focused vocab (~1k word/subword pieces) and is trained on a
strict instruction grammar — so input that uses out-of-vocab words / characters is
out of distribution and won't work well. This reports that up front.

  python validate.py "add 3 and 5"
"""

import sys

import bpe as bpemod


def main():
    tk = bpemod.BPE.load_json("out/bpe.json")
    text = " ".join(sys.argv[1:]) or "add 3 and 5"
    ratio = tk.unknown_ratio(text)
    ids = tk.encode(text, bos=True)
    words = set(tk.readable_words())
    unknown = [w for w in text.split() if w.lower() not in words and not w.isdigit()]
    print(f"input  : {text!r}")
    print(f"tokens : {len(ids)}   out-of-vocab chars: {ratio:.0%}")
    if unknown:
        print(f"words not seen as whole pieces: {unknown}")
    if ratio < 0.02:
        print("OK — in the model's vocabulary.")
    else:
        print("WARNING: out-of-vocabulary input; the model may not handle this well. "
              "Rephrase using common words, or add such examples to the training data.")


if __name__ == "__main__":
    main()
