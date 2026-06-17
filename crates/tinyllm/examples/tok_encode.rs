//! Print how the on-device `tinyllm` tokenizer encodes a string, to verify the
//! Python BPE port (train/bpe.py) produces identical token ids.
//!
//!   cargo run -p tinyllm --example tok_encode -- tokenizer.bin 1024 "add 3 and 5"

use tinyllm::Tokenizer;

fn main() {
    let mut a = std::env::args().skip(1);
    let tok_path = a.next().expect("usage: tok_encode <tokenizer.bin> <vocab_size> <text>");
    let vocab: usize = a.next().expect("vocab_size").parse().expect("vocab_size int");
    let text = a.collect::<Vec<_>>().join(" ");
    let buf = std::fs::read(&tok_path).expect("read tokenizer.bin");
    let tk = Tokenizer::from_bytes(&buf, vocab).expect("parse tokenizer");
    let ids = tk.encode(&text, true, false); // BOS, no EOS — as the browser does
    let s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    println!("{}", s.join(" "));
}
