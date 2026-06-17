"""Train the tiny English->Python (delexicalized) model on a Modal GPU and pull the
browser assets back. The frozen embedding is precomputed locally (numpy + ollama, see
prep_delex.py) and shipped as embed_pca.npz, so the cloud job needs only torch+numpy —
no ollama, no GPU-side embedding build.

  python prep_delex.py --data data --out out_delex     # (once, locally)
  modal run modal_train.py                              # ~A10G, a few min, well under $10

Writes web/model.bin + web/tokenizer.bin in the repo. index.html auto-detects the delex
markers in the new tokenizer and switches the slot machinery on.
"""
import os

import modal

ASSETS = "/tmp/modal_pkg"  # staged by the shell step: code + data/*.jsonl + out_delex/{bpe.json,tokenizer.bin,embed_pca.npz}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(ASSETS, "/assets", copy=True)
)

app = modal.App("pyspell-delex-train", image=image)


@app.function(gpu="A10G", timeout=3600)
def train_and_export():
    import shutil
    import subprocess

    shutil.copytree("/assets", "/work")  # writable copy (the image layer is read-only)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def run(cmd):
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd="/work", env=env, check=True)

    # Train (loads the shipped bpe.json + embed_pca.npz as-is; no ollama). The model is
    # tiny, so early-stopping on val usually finishes well before the time cap.
    run(["python", "train.py", "--preset", "full512", "--out", "out_delex",
         "--device", "cuda", "--max-minutes", "30", "--batch", "128"])
    # Export the browser assets directly (model.bin + tokenizer.bin).
    run(["python", "export_v2.py", "--out", "out_delex", "--emit-web", "web_out"])

    ev = subprocess.run(["python", "eval.py", "--out", "out_delex", "--n", "250"],
                        cwd="/work", env=env, capture_output=True, text=True)
    model_bin = open("/work/web_out/model.bin", "rb").read()
    tok_bin = open("/work/web_out/tokenizer.bin", "rb").read()
    return model_bin, tok_bin, (ev.stdout + "\n" + ev.stderr)[-4000:]


@app.local_entrypoint()
def main():
    model_bin, tok_bin, eval_out = train_and_export.remote()
    web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    with open(os.path.join(web, "model.bin"), "wb") as f:
        f.write(model_bin)
    with open(os.path.join(web, "tokenizer.bin"), "wb") as f:
        f.write(tok_bin)
    print(f"\nwrote {web}/model.bin ({len(model_bin)} B) + tokenizer.bin ({len(tok_bin)} B)")
    print("\n===== eval (on the new delex model) =====\n" + eval_out)
