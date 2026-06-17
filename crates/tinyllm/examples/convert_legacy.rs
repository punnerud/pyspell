//! Convert a llama2.c **legacy (v0) fp32 checkpoint** (e.g. Karpathy's
//! `stories260K.bin`) into our packed flash image: `TOC + tokenizer + model`, where
//! the model is re-quantized to the v2 (Q8_0) layout with [`write_v2`]. No PyTorch —
//! we read the raw fp32 weights and quantize in Rust.
//!
//! Get the public stories260K checkpoint + tokenizer (Karpathy, MIT):
//!   curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin -o stories260K.bin
//!   curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin   -o tok512.bin
//! Run:
//!   cargo run -p tinyllm --example convert_legacy -- stories260K.bin tok512.bin model.img
//! then flash:  espflash write-bin 0x810000 model.img
//!
//! Legacy layout (run.c `memory_map_weights`), little-endian:
//!   header: 7×i32 (dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
//!     (negative vocab_size ⇒ separate classifier; positive ⇒ shared with embeddings)
//!   f32: token_embedding[vocab*dim], rms_att[L*dim], wq[L*dim*dim], wk[L*dim*kv],
//!        wv[L*dim*kv], wo[L*dim*dim], rms_ffn[L*dim], w1[L*dim*hid], w2[L*hid*dim],
//!        w3[L*dim*hid], rms_final[dim], freq_cis_real[seq*hs/2], freq_cis_imag[...],
//!        wcls[vocab*dim] (only if not shared)

use std::io::Write;

use tinyllm::format::{write_v2, Config};

fn main() {
    let mut a = std::env::args().skip(1);
    let model_in = a.next().expect("usage: convert_legacy <legacy.bin> <tokenizer.bin> <out.img>");
    let tok_in = a.next().expect("missing tokenizer.bin");
    let out_path = a.next().unwrap_or_else(|| "model.img".to_string());

    let raw = std::fs::read(&model_in).expect("read legacy model");
    let mut rd = Reader { buf: &raw, pos: 0 };

    let dim = rd.i32() as usize;
    let hidden_dim = rd.i32() as usize;
    let n_layers = rd.i32() as usize;
    let n_heads = rd.i32() as usize;
    let n_kv_heads = rd.i32() as usize;
    let vocab_raw = rd.i32();
    let seq_len = rd.i32() as usize;
    let shared_classifier = vocab_raw > 0;
    let vocab_size = vocab_raw.unsigned_abs() as usize;
    let head_size = dim / n_heads;
    let kv_dim = (dim * n_kv_heads) / n_heads;

    // Group size must divide every matmul input width (dim and hidden_dim) and every
    // tensor length. Use the largest power-friendly divisor of gcd(dim, hidden_dim).
    let group_size = largest_gs(gcd(dim, hidden_dim));

    eprintln!(
        "legacy: dim={dim} hidden={hidden_dim} layers={n_layers} heads={n_heads} kv={n_kv_heads} vocab={vocab_size} seq={seq_len} shared={shared_classifier} → group_size={group_size}"
    );

    // Read in the exact legacy order.
    let q_tokens = rd.f32s(vocab_size * dim);
    let rms_att = rd.f32s(n_layers * dim);
    let wq = rd.layers(n_layers, dim * dim);
    let wk = rd.layers(n_layers, dim * kv_dim);
    let wv = rd.layers(n_layers, dim * kv_dim);
    let wo = rd.layers(n_layers, dim * dim);
    let rms_ffn = rd.f32s(n_layers * dim);
    let w1 = rd.layers(n_layers, dim * hidden_dim);
    let w2 = rd.layers(n_layers, hidden_dim * dim);
    let w3 = rd.layers(n_layers, dim * hidden_dim);
    let rms_final = rd.f32s(dim);
    rd.skip(seq_len * (head_size / 2)); // freq_cis_real (RoPE computed at runtime)
    rd.skip(seq_len * (head_size / 2)); // freq_cis_imag
    let wcls = if shared_classifier { None } else { Some(rd.f32s(vocab_size * dim)) };
    assert!(rd.pos == raw.len(), "trailing bytes: read {} of {}", rd.pos, raw.len());

    let cfg = Config {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        seq_len,
        group_size,
        shared_classifier,
    };
    let model = write_v2(
        &cfg, &rms_att, &rms_ffn, &rms_final, &q_tokens, &wq, &wk, &wv, &wo, &w1, &w2, &w3,
        wcls.as_deref(),
    );
    let tokenizer = std::fs::read(&tok_in).expect("read tokenizer");

    let mut img = Vec::with_capacity(16 + tokenizer.len() + model.len());
    img.extend_from_slice(b"PSM1");
    img.extend_from_slice(&1u32.to_le_bytes());
    img.extend_from_slice(&(tokenizer.len() as u32).to_le_bytes());
    img.extend_from_slice(&(model.len() as u32).to_le_bytes());
    img.extend_from_slice(&tokenizer);
    img.extend_from_slice(&model);
    std::fs::File::create(&out_path).unwrap().write_all(&img).unwrap();

    eprintln!(
        "wrote {out_path}: TOC 16 + tokenizer {} B + model {} B = {} B ({:.2} MB)",
        tokenizer.len(),
        model.len(),
        img.len(),
        img.len() as f64 / (1024.0 * 1024.0),
    );
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        let bytes = n * 4;
        let out: Vec<f32> = self.buf[self.pos..self.pos + bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.pos += bytes;
        out
    }
    fn layers(&mut self, n_layers: usize, each: usize) -> Vec<Vec<f32>> {
        (0..n_layers).map(|_| self.f32s(each)).collect()
    }
    fn skip(&mut self, n_floats: usize) {
        self.pos += n_floats * 4;
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
/// Largest power of two ≤ `g` that divides `g` (a good Q8 group size).
fn largest_gs(g: usize) -> usize {
    let mut gs = 1;
    while gs * 2 <= g && g % (gs * 2) == 0 {
        gs *= 2;
    }
    gs.min(64)
}
