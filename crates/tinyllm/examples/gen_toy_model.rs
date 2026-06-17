//! Generate a packed flash image for the ESP32 `model` partition.
//!
//! Emits `TOC(16) + tokenizer.bin + model.bin`, where the model is a small but
//! *valid* llama2.c v2 (Q8_0) checkpoint (built with [`tinyllm::format::write_v2`])
//! and the tokenizer is a byte-fallback `tokenizer.bin` matching its `vocab_size`.
//! The weights are deterministic pseudo-random, so the model **runs** end-to-end
//! (the browser can load + generate) but produces toy word-salad — swap in a real
//! trained checkpoint by re-running with real weights, the on-device serving path is
//! unchanged.
//!
//! Run:  `cargo run -p tinyllm --example gen_toy_model -- model-part.img`
//!
//! TOC header (16 bytes, little-endian): magic b"PSM1", u32 version=1,
//! u32 tok_len, u32 model_len. Then the tokenizer bytes, then the model bytes.

use std::io::Write;

use tinyllm::format::{write_v2, Config};

// Toy hyper-parameters → ~3.8 MB int8 (fits the 6 MB partition, a real multi-MB
// streamed transfer). dim is a multiple of group_size so every tensor quantizes.
const DIM: usize = 256;
// Both dim and hidden_dim must be multiples of GROUP_SIZE: they are the *input*
// widths of the matmuls (the activation is quantized in gs-sized groups).
const HIDDEN: usize = 704; // 64 * 11
const N_LAYERS: usize = 4;
const N_HEADS: usize = 8;
const N_KV_HEADS: usize = 8;
const VOCAB: usize = 512;
const SEQ_LEN: usize = 256;
const GROUP_SIZE: usize = 64;

/// Tiny deterministic LCG → f32 in roughly [-scale, scale]. Deterministic so the
/// image is reproducible (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self, scale: f32) -> f32 {
        // Numerical Recipes LCG.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((self.0 >> 40) as u32) as f32 / (1u32 << 24) as f32; // [0,1)
        (u * 2.0 - 1.0) * scale
    }
}

fn vec_rng(rng: &mut Lcg, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32(scale)).collect()
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "model-part.img".to_string());

    let cfg = Config {
        dim: DIM,
        hidden_dim: HIDDEN,
        n_layers: N_LAYERS,
        n_heads: N_HEADS,
        n_kv_heads: N_KV_HEADS,
        vocab_size: VOCAB,
        seq_len: SEQ_LEN,
        group_size: GROUP_SIZE,
        shared_classifier: true, // no separate wcls; reuse token embeddings
    };
    let kv_dim = (DIM * N_KV_HEADS) / N_HEADS;

    let mut rng = Lcg(0x5151_5151_2026_0617);
    // rms weights initialise near 1.0 (LayerNorm-ish).
    let rms_att: Vec<f32> = (0..N_LAYERS * DIM).map(|_| 1.0 + rng.next_f32(0.02)).collect();
    let rms_ffn: Vec<f32> = (0..N_LAYERS * DIM).map(|_| 1.0 + rng.next_f32(0.02)).collect();
    let rms_final: Vec<f32> = (0..DIM).map(|_| 1.0 + rng.next_f32(0.02)).collect();

    let q_tokens = vec_rng(&mut rng, VOCAB * DIM, 0.08);
    let per_layer = |rng: &mut Lcg, n: usize| (0..N_LAYERS).map(|_| vec_rng(rng, n, 0.06)).collect::<Vec<_>>();
    let wq = per_layer(&mut rng, DIM * DIM);
    let wk = per_layer(&mut rng, DIM * kv_dim);
    let wv = per_layer(&mut rng, DIM * kv_dim);
    let wo = per_layer(&mut rng, DIM * DIM);
    let w1 = per_layer(&mut rng, DIM * HIDDEN);
    let w2 = per_layer(&mut rng, HIDDEN * DIM);
    let w3 = per_layer(&mut rng, DIM * HIDDEN);

    let model = write_v2(
        &cfg, &rms_att, &rms_ffn, &rms_final, &q_tokens, &wq, &wk, &wv, &wo, &w1, &w2, &w3, None,
    );

    let tokenizer = build_tokenizer(VOCAB);

    // Pack: TOC + tokenizer + model.
    let mut img = Vec::with_capacity(16 + tokenizer.len() + model.len());
    img.extend_from_slice(b"PSM1");
    img.extend_from_slice(&1u32.to_le_bytes()); // version
    img.extend_from_slice(&(tokenizer.len() as u32).to_le_bytes());
    img.extend_from_slice(&(model.len() as u32).to_le_bytes());
    img.extend_from_slice(&tokenizer);
    img.extend_from_slice(&model);

    let mut f = std::fs::File::create(&out_path).expect("create output");
    f.write_all(&img).expect("write image");

    eprintln!(
        "wrote {out_path}: TOC 16 + tokenizer {} B + model {} B = {} B ({:.2} MB)\n  config dim={DIM} hidden={HIDDEN} layers={N_LAYERS} heads={N_HEADS} vocab={VOCAB} seq={SEQ_LEN} gs={GROUP_SIZE}",
        tokenizer.len(),
        model.len(),
        img.len(),
        img.len() as f64 / (1024.0 * 1024.0),
    );
}

/// Build a byte-fallback `tokenizer.bin` (llama2.c layout) of exactly `vocab` tokens:
/// 0=`<unk>`, 1=`<s>`, 2=`</s>`, ids 3..259 the 256 raw bytes as `<0xXX>` (so encode's
/// `byte+3` fallback and decode's raw-byte parse line up), and the remainder filled
/// with common English fragments so generation yields printable word-salad.
fn build_tokenizer(vocab: usize) -> Vec<u8> {
    let mut toks: Vec<(Vec<u8>, f32)> = Vec::with_capacity(vocab);
    toks.push((b"<unk>".to_vec(), 0.0));
    toks.push((b"<s>".to_vec(), 0.0));
    toks.push((b"</s>".to_vec(), 0.0));
    for b in 0u8..=255 {
        toks.push((format!("<0x{b:02X}>").into_bytes(), 0.0));
    }
    const FRAGMENTS: &[&str] = &[
        " the", " and", " to", " a", " of", " in", " was", " it", " he", " she",
        " they", " said", " little", " one", " day", " went", " big", " happy",
        " play", " friend", " saw", " then", " very", " up", " down", " all",
        " could", " not", " with", " for", " his", " her", " on", " out", ".",
        ",", " I", " you", " we", " is", " are", " had", " but", " so", " when",
        " home", " tree", " good", " time",
    ];
    let mut i = 0usize;
    while toks.len() < vocab {
        let frag = FRAGMENTS[i % FRAGMENTS.len()];
        // Keep entries distinct past one cycle by appending the cycle count.
        let cycle = i / FRAGMENTS.len();
        let s = if cycle == 0 { frag.to_string() } else { format!("{frag}{cycle}") };
        // Higher scores than raw bytes so merges prefer words.
        toks.push((s.into_bytes(), 1.0 + (i as f32) * 0.001));
        i += 1;
    }
    toks.truncate(vocab);

    let max_len = toks.iter().map(|(b, _)| b.len()).max().unwrap_or(0) as i32;
    let mut out = Vec::new();
    out.extend_from_slice(&max_len.to_le_bytes());
    for (bytes, score) in &toks {
        out.extend_from_slice(&score.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}
