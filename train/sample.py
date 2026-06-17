"""Generate from a checkpoint (torch) to eyeball quality — same tokenizer + model the
dongle runs.  python sample.py --prompt "add 3 and 5"
"""

import argparse
import os

import torch
import torch.nn.functional as F

import bpe as bpemod
import delex
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


@torch.no_grad()
def generate_scored(model, tk, device, prompt, max_new=64):
    """Greedy generate, returning (text, token_pieces, confs) where conf is the softmax
    probability of each chosen token — the model's confidence (for uncertainty mining)."""
    ids = tk.encode(prompt, bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out, pieces, confs = [], [], []
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.c.seq_len:])
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        nxt = int(logits.argmax())
        if nxt in (bpemod.BOS, bpemod.EOS):
            break
        out.append(nxt)
        pieces.append(tk.decode([nxt]))  # decoded text of this token (for literal masking)
        confs.append(float(probs[nxt]))
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return tk.decode(out), pieces, confs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--prompt", default="add 3 and 5")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--no-delex", dest="delex", action="store_false", default=True,
                    help="feed the prompt literally (legacy non-delexicalized model)")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(os.path.join(args.out, "ckpt.pt"), map_location=device, weights_only=False)
    model = Llama(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tk = bpemod.BPE.load_json(os.path.join(args.out, "bpe.json"))
    # Mirror the browser/device: delex the prompt, generate, relex the output.
    prompt, nums, strs = (delex.delex_en(args.prompt) if args.delex else (args.prompt, [], []))
    raw = generate(model, tk, device, prompt, args.max_new, args.temperature)
    out = delex.relex(raw, nums, strs) if args.delex else raw
    print(f"[step {ck['step']}] prompt: {args.prompt!r}  (delex: {prompt!r})\n---")
    print(out)


if __name__ == "__main__":
    main()
