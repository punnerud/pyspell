"""Train the English->Python model on Apple Silicon (MPS), with a learned ~1024
word/subword vocab (see bpe.py — tokenization is identical to the on-device runtime).

PAUSE / RESUME: a checkpoint (model + optimizer + step + RNG + config) is saved to
`out/ckpt.pt` every `--ckpt-interval` steps and on Ctrl-C / SIGTERM (atomic). Re-run
the SAME command to resume exactly where you stopped. Letting the Mac sleep is fine.

  python gen_data.py --n 40000 --out data
  python train.py --preset full --max-minutes 60      # ~1h run
  # ...Ctrl-C any time...  then re-run the same line to continue.
"""

import argparse
import json
import os
import signal
import time
from dataclasses import asdict, replace

import numpy as np
import torch

import bpe as bpemod
import build_embeddings
import build_types
from model import PRESETS, Config, Llama

# Whole instruction words forced into the vocab (bare + leading-space form, since encode
# prepends a space) so the ~512 vocab is dominated by real words, not template fragments.
WORDS = [
    "print", "say", "output", "display", "add", "subtract", "multiply", "divide",
    "compute", "what", "is", "set", "let", "assign", "define", "function", "returns",
    "write", "make", "create", "list", "sum", "find", "largest", "smallest", "length",
    "uppercase", "lowercase", "word", "count", "from", "to", "through", "numbers",
    "each", "item", "loop", "over", "square", "cube", "even", "odd", "reverse",
    "backwards", "greater", "bigger", "than", "check", "if", "the", "of", "and", "two",
    "down", "up", "total", "plus", "combine", "in", "for", "else", "while", "result",
    "value", "num", "temp", "nums", "number", "items", "a", "an",
    "hello", "world", "cat", "dog", "Lily", "Tom", "apple", "tree", "sun", "code",
    "Python", "robot",
]
# Python structural tokens (kept as whole pieces for clean code generation).
PY_TOKENS = [
    "    ", "\n    ", "print(", "def ", "return ", "range(", "for ", " in ", "if ",
    "else", "while ", "):", "()", " = ", " + ", " - ", " * ", " // ", " % ", "len(",
    "sum(", "max(", "min(", ".upper()", ".lower()", "[::-1]", "== 0", '("', '")',
    ", ", " % 2", " == ", " > ", "** 2", "** 3", ", 0, -1)", "for i in range(",
    "for item in ", "range", "def", "return",
    # Edit-mode protocol tokens (anchor-based find/replace edits).
    "EDIT", " EDIT", "@@ ", " ==> ",
]
# vocab 512 = 3 specials + 256 reserved byte tokens + ~60 base chars leaves ~190 slots,
# so force the space-prefixed word forms (what appears mid-sentence) + Python tokens; BPE
# learns line-start/bare words and the rest by frequency.
FORCED = PY_TOKENS + [" " + w for w in WORDS]

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n[signal {signum}] finishing the current step, then checkpointing…", flush=True)


def corpus_texts(jsonl_path):
    out = []
    with open(jsonl_path) as f:
        for line in f:
            o = json.loads(line)
            # ASCII only (keeps byte-fallback unused); strict, simple grammar.
            t = (o["en"] + "\n" + o["py"]).encode("ascii", "ignore").decode("ascii")
            out.append(t)
    return out


def get_tokenizer(data_dir, out_dir, vocab_size):
    jpath = os.path.join(out_dir, "bpe.json")
    if os.path.exists(jpath):
        return bpemod.BPE.load_json(jpath)
    print(f"training BPE vocab (target {vocab_size})…")
    texts = corpus_texts(os.path.join(data_dir, "train.jsonl"))
    tk = bpemod.BPE.train(texts, vocab_size, forced=FORCED)
    tk.save_json(jpath)
    tk.write_tokenizer_bin(os.path.join(out_dir, "tokenizer.bin"))
    print(f"  vocab_size={tk.vocab_size}, {len(tk.readable_words())} readable word pieces")
    return tk


def build_bin(jsonl_path, bin_path, tk):
    ids = []
    with open(jsonl_path) as f:
        for line in f:
            o = json.loads(line)
            full = (o["en"] + "\n" + o["py"]).encode("ascii", "ignore").decode("ascii")
            ids.extend(tk.encode(full, bos=True, eos=True))
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    return arr


def load_split(data_dir, out_dir, name, tk):
    jsonl = os.path.join(data_dir, f"{name}.jsonl")
    binp = os.path.join(out_dir, f"{name}.bin")
    if (not os.path.exists(binp)) or os.path.getmtime(jsonl) > os.path.getmtime(binp):
        print(f"tokenizing {jsonl} -> {binp}")
        return build_bin(jsonl, binp, tk)
    return np.fromfile(binp, dtype=np.uint16)


def get_batch(data, bs, seq, device):
    ix = np.random.randint(0, len(data) - seq - 1, size=bs)
    x = np.stack([data[i : i + seq] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + seq] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(step, base_lr, warmup, decay_iters, min_lr):
    import math
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= decay_iters:
        return min_lr
    r = (step - warmup) / max(1, decay_iters - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (base_lr - min_lr)


@torch.no_grad()
def eval_loss(model, data, bs, seq, device, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, seq, device)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def save_ckpt(path, model, opt, step, best_val, cfg):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
        "best_val": best_val, "cfg": asdict(cfg),
        "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
    }, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save can't corrupt the checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--vocab", type=int, default=None, help="BPE vocab size (default: preset)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-iters", type=int, default=20000)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--max-steps", type=int, default=10_000_000)
    ap.add_argument("--ckpt-interval", type=int, default=200)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--patience", type=int, default=6, help="stop after N evals w/o val improvement")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    base = PRESETS[args.preset]
    vocab = args.vocab or base.vocab_size
    tk = get_tokenizer(args.data, args.out, vocab)
    cfg = replace(base, vocab_size=tk.vocab_size)
    bs = args.batch or (64 if args.preset == "full" else 16)
    print(f"device={device} preset={args.preset} dim={cfg.dim} layers={cfg.n_layers} "
          f"vocab={cfg.vocab_size} seq={cfg.seq_len} gs={cfg.group_size} batch={bs}")

    train_data = load_split(args.data, args.out, "train", tk)
    val_data = load_split(args.data, args.out, "val", tk)
    print(f"tokens: train={len(train_data):,} val={len(val_data):,}")

    torch.manual_seed(1337)
    np.random.seed(1337)
    model = Llama(cfg).to(device)

    # Frozen semantic+POS input embedding. Built once, cached. With tie_classifier the
    # output is a RAG lookup against this same frozen table; otherwise a learned wcls.
    if cfg.frozen_emb:
        types, _ = build_types.build(tk)
        with open(os.path.join(args.out, "word_types.json"), "w") as f:
            json.dump({"types": types, "type_set": build_types.TYPES}, f)
        emb = build_embeddings.load_or_build(tk, args.out, cfg.dim)
        model.set_frozen_embedding(torch.from_numpy(emb).to(device))
        mode = "tied (RAG lookup)" if cfg.tie_classifier else "untied (learned)"
        print(f"frozen input embedding installed; classifier {mode}")

    print(f"params: {model.num_params()/1e6:.3f}M (~{model.num_params()/1024:.0f} kB int8)")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    step, best_val = 0, float("inf")
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # our own ckpt (trusted)
        if ck.get("cfg") != asdict(cfg):
            raise SystemExit("checkpoint config differs from current; use a fresh --out dir")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_val = ck.get("best_val", float("inf"))
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))  # rng state lives on CPU
        np.random.set_state(ck["np_rng"])
        if cfg.frozen_emb:
            model.tok.weight.requires_grad_(False)  # stay frozen after load
        print(f"RESUMED from {ckpt_path} at step {step} (best_val {best_val:.3f})")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    best_path = os.path.join(args.out, "best.pt")
    no_improve = 0
    model.train()
    t0 = time.time()
    tok_per_step = bs * cfg.seq_len
    last_log = t0
    while step < args.max_steps and (time.time() - t0) / 60.0 < args.max_minutes:
        lr = lr_at(step, args.lr, args.warmup, args.decay_iters, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, bs, cfg.seq_len, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1

        if step % 20 == 0:
            now = time.time()
            tps = 20 * tok_per_step / (now - last_log)
            last_log = now
            mins = (now - t0) / 60.0
            print(f"step {step} loss {loss.item():.3f} lr {lr:.2e} {tps/1000:.0f}k tok/s "
                  f"{mins:.1f}/{args.max_minutes:.0f} min", flush=True)
        if step % args.eval_interval == 0:
            vl = eval_loss(model, val_data, bs, cfg.seq_len, device)
            if vl < best_val:
                best_val = vl
                no_improve = 0
                save_ckpt(best_path, model, opt, step, best_val, cfg)  # keep the BEST
                print(f"  >> val loss {vl:.3f} (best {best_val:.3f}) [saved best]", flush=True)
            else:
                no_improve += 1
                print(f"  >> val loss {vl:.3f} (best {best_val:.3f}) [{no_improve}/{args.patience}]", flush=True)
                if no_improve >= args.patience:
                    print("early stop: val not improving")
                    break
        if step % args.ckpt_interval == 0:
            save_ckpt(ckpt_path, model, opt, step, best_val, cfg)
        if _STOP:
            break

    save_ckpt(ckpt_path, model, opt, step, best_val, cfg)
    print(f"latest -> {ckpt_path}; best (val {best_val:.3f}) -> {best_path}")
    print("done. export with:  python export_v2.py --out", args.out)


if __name__ == "__main__":
    main()
