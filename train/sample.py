"""Generate from a checkpoint (torch) to eyeball quality — same tokenizer + model the
dongle runs.  python sample.py --prompt "add 3 and 5"
"""

import argparse
import os

import torch
import torch.nn.functional as F

import bpe as bpemod
from model import Config, Llama


@torch.no_grad()
def generate(model, tk, device, prompt, max_new=96, temperature=0.8, top_p=0.9):
    ids = tk.encode(prompt, bos=True)  # BOS + prompt, matching the browser
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.c.seq_len:])
        logits = logits[0, -1, :]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, 0)
            sp[cum - sp > top_p] = 0.0
            sp /= sp.sum()
            nxt = int(si[torch.multinomial(sp, 1)])
        if nxt in (bpemod.BOS, bpemod.EOS):
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return tk.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--prompt", default="add 3 and 5")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=96)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(os.path.join(args.out, "ckpt.pt"), map_location=device, weights_only=False)
    model = Llama(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tk = bpemod.BPE.load_json(os.path.join(args.out, "bpe.json"))
    print(f"[step {ck['step']}] prompt: {args.prompt!r}\n---")
    print(generate(model, tk, device, args.prompt, args.max_new, args.temperature))


if __name__ == "__main__":
    main()
