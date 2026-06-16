//! BPE tokenizer reading llama2.c's `tokenizer.bin` format:
//! ```text
//!   i32  max_token_length
//!   repeat vocab_size times:
//!     f32  score
//!     i32  len
//!     u8   bytes[len]
//! ```
//! `encode`/`decode` mirror `run.c` (BOS=1, EOS=2, byte-fallback at `byte+3`,
//! `<0xXX>` raw-byte pieces, and the leading-space strip after BOS). The merge
//! search is O(n²) per step — fine for the short prompts a toy gets. This port
//! is exercised on-device; the byte-exact gate against `run.c` happens once the
//! real `tokenizer.bin` is on hand (see the plan).

use alloc::vec;
use alloc::vec::Vec;

pub const BOS: usize = 1;
pub const EOS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    Truncated,
    /// Declared vocab in the file did not match the model's vocab_size.
    VocabMismatch,
}

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
}

impl Tokenizer {
    /// Parse `tokenizer.bin` for a model with `vocab_size` tokens.
    pub fn from_bytes(buf: &[u8], vocab_size: usize) -> Result<Self, TokenizerError> {
        let mut off = 0usize;
        let rd_i32 = |buf: &[u8], o: usize| -> Option<i32> {
            buf.get(o..o + 4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let rd_f32 = |buf: &[u8], o: usize| -> Option<f32> {
            buf.get(o..o + 4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        // max_token_length — read for format fidelity, not otherwise needed.
        let _max_len = rd_i32(buf, off).ok_or(TokenizerError::Truncated)?;
        off += 4;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let score = rd_f32(buf, off).ok_or(TokenizerError::Truncated)?;
            off += 4;
            let len = rd_i32(buf, off).ok_or(TokenizerError::Truncated)? as usize;
            off += 4;
            let bytes = buf.get(off..off + len).ok_or(TokenizerError::Truncated)?;
            off += len;
            vocab.push(bytes.to_vec());
            scores.push(score);
        }
        if vocab.len() != vocab_size {
            return Err(TokenizerError::VocabMismatch);
        }
        Ok(Tokenizer { vocab, scores })
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn find(&self, s: &[u8]) -> Option<usize> {
        self.vocab.iter().position(|t| t.as_slice() == s)
    }

    /// Encode `text` to token ids, optionally adding BOS/EOS, then greedily
    /// applying the highest-scoring BPE merges.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens: Vec<usize> = Vec::new();
        if bos {
            tokens.push(BOS);
        }
        // TinyStories prepends a space when the text is non-empty.
        if !text.is_empty() {
            if let Some(id) = self.find(b" ") {
                tokens.push(id);
            }
        }
        // Per-codepoint: whole-char vocab hit, else byte-fallback (byte + 3).
        let mut cbuf = [0u8; 4];
        for ch in text.chars() {
            let s = ch.encode_utf8(&mut cbuf).as_bytes();
            if let Some(id) = self.find(s) {
                tokens.push(id);
            } else {
                for &b in s {
                    // Byte fallback: real models reserve ids 3..259 for the 256
                    // raw bytes. Guard against a vocab too small to hold them.
                    let id = b as usize + 3;
                    tokens.push(if id < self.vocab.len() { id } else { 0 });
                }
            }
        }
        // Greedy BPE: repeatedly merge the best adjacent pair present in vocab.
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = None;
            let mut best_idx = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let mut cat = self.vocab[tokens[i]].clone();
                cat.extend_from_slice(&self.vocab[tokens[i + 1]]);
                if let Some(id) = self.find(&cat) {
                    if self.scores[id] > best_score {
                        best_score = self.scores[id];
                        best_id = Some(id);
                        best_idx = Some(i);
                    }
                }
            }
            match best_idx {
                Some(i) => {
                    tokens[i] = best_id.unwrap();
                    tokens.remove(i + 1);
                }
                None => break,
            }
        }
        if eos {
            tokens.push(EOS);
        }
        tokens
    }

    /// Decode one token to its UTF-8 bytes, given the previous token (to strip
    /// the leading space right after BOS, as `run.c` does).
    pub fn decode(&self, prev: usize, token: usize) -> Vec<u8> {
        let piece: &[u8] = match self.vocab.get(token) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let mut p = piece;
        if prev == BOS && p.first() == Some(&b' ') {
            p = &p[1..];
        }
        if let Some(b) = parse_hex_byte(p) {
            return vec![b];
        }
        p.to_vec()
    }
}

/// Parse a `<0xXX>` raw-byte piece into its byte value.
fn parse_hex_byte(p: &[u8]) -> Option<u8> {
    if p.len() == 6 && &p[0..3] == b"<0x" && p[5] == b'>' {
        let hi = hex_val(p[3])?;
        let lo = hex_val(p[4])?;
        Some(hi << 4 | lo)
    } else {
        None
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `tokenizer.bin` in memory.
    fn build(vocab: &[(&str, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let max_len = vocab.iter().map(|(s, _)| s.len()).max().unwrap_or(0) as i32;
        out.extend_from_slice(&max_len.to_le_bytes());
        for (s, score) in vocab {
            out.extend_from_slice(&score.to_le_bytes());
            out.extend_from_slice(&(s.len() as i32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        out
    }

    fn tiny_tok() -> Tokenizer {
        // ids:    0       1      2       3    4    5     6      7
        let vocab = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
            (" ", 1.0),
            ("h", 1.0),
            ("i", 1.0),
            ("hi", 5.0),
            (" hi", 9.0),
        ];
        let bytes = build(&vocab);
        Tokenizer::from_bytes(&bytes, vocab.len()).expect("tok parse")
    }

    #[test]
    fn merges_to_best_scoring_token() {
        let tok = tiny_tok();
        // " " (dummy prefix) + h + i  ->  merge hi -> merge " hi"
        assert_eq!(tok.encode("hi", false, false), alloc::vec![7]);
    }

    #[test]
    fn bos_eos_and_decode_strip_space() {
        let tok = tiny_tok();
        let ids = tok.encode("hi", true, true);
        assert_eq!(ids.first(), Some(&BOS));
        assert_eq!(ids.last(), Some(&EOS));
        // decode of " hi" right after BOS strips the leading space.
        assert_eq!(tok.decode(BOS, 7), b"hi".to_vec());
        assert_eq!(tok.decode(5, 7), b" hi".to_vec());
    }

    #[test]
    fn decodes_raw_byte_piece() {
        let vocab = [("<unk>", 0.0), ("<s>", 0.0), ("</s>", 0.0), ("<0x0A>", 0.0)];
        let bytes = build(&vocab);
        let tok = Tokenizer::from_bytes(&bytes, vocab.len()).unwrap();
        assert_eq!(tok.decode(0, 3), alloc::vec![b'\n']);
    }
}
