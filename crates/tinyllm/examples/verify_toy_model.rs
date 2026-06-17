//! Smoke-test a packed model image (`TOC + tokenizer + model`, as produced by the
//! `gen_toy_model` example): split it, parse the model + tokenizer, and generate a
//! few tokens. This is the exact path the browser-WASM runtime takes, so if this
//! prints tokens the image is loadable on-device.
//!
//! Run:  `cargo run -p tinyllm --example verify_toy_model -- model-part.img`

use tinyllm::{RunState, Sampler, Tokenizer, Transformer};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "model-part.img".to_string());
    let img = std::fs::read(&path).expect("read image");
    assert!(img.len() >= 16 && &img[0..4] == b"PSM1", "bad TOC magic");
    let u = |i: usize| u32::from_le_bytes([img[i], img[i + 1], img[i + 2], img[i + 3]]) as usize;
    let version = u(4);
    let tok_len = u(8);
    let model_len = u(12);
    let base = if version >= 2 { 24 } else { 16 }; // v2 TOC adds wordmeta+embed lengths
    let tok_bytes = &img[base..base + tok_len];
    let model_bytes = &img[base + tok_len..base + tok_len + model_len];
    eprintln!("split: tokenizer {tok_len} B, model {model_len} B");

    let t = Transformer::new(model_bytes).expect("parse model");
    let cfg = *t.config();
    eprintln!(
        "model: dim={} hidden={} layers={} heads={} vocab={} seq={} gs={}",
        cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size, cfg.seq_len, cfg.group_size
    );
    let tok = Tokenizer::from_bytes(tok_bytes, cfg.vocab_size).expect("parse tokenizer");

    let prompt = "Once upon a time";
    let prompt_tokens = tok.encode(prompt, true, false);
    let mut state = RunState::new(&cfg);
    let mut sampler = Sampler::new(cfg.vocab_size, 0.9, 0.9, 1234);

    let mut token = prompt_tokens[0];
    let mut out = String::from(prompt);
    let max_steps = 40usize.min(cfg.seq_len);
    for pos in 0..max_steps {
        t.forward(&mut state, token, pos);
        let next = if pos + 1 < prompt_tokens.len() {
            prompt_tokens[pos + 1]
        } else {
            let mut logits = state.logits().to_vec();
            sampler.sample(&mut logits)
        };
        if pos + 1 >= prompt_tokens.len() {
            let piece = tok.decode(token, next);
            out.push_str(&String::from_utf8_lossy(&piece));
        }
        token = next;
    }
    println!("--- generated (toy weights → word-salad is expected) ---\n{out}");
    eprintln!("OK: image parses and generates");
}
