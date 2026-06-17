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
    let tok_len = u32::from_le_bytes([img[8], img[9], img[10], img[11]]) as usize;
    let model_len = u32::from_le_bytes([img[12], img[13], img[14], img[15]]) as usize;
    let tok_bytes = &img[16..16 + tok_len];
    let model_bytes = &img[16 + tok_len..16 + tok_len + model_len];
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
