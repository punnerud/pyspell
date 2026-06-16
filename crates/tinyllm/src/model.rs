//! The transformer itself: a [`RunState`] (all the per-token scratch + the KV
//! cache) and [`Transformer::forward`], a faithful port of llama2.c's quantized
//! `forward()`.
//!
//! The KV cache here is **f32** — it matches `run.c` exactly, which is what the
//! host correctness gate needs. On the device the cache dominates RAM; an int8
//! KV variant is a later, swap-in optimization (see the plan), but the math and
//! the format reader are identical either way.

use alloc::vec;
use alloc::vec::Vec;

use crate::format::{Config, Model, ModelError};
use crate::math::{matmul, quantize, rmsnorm, silu, softmax};

/// All mutable inference state for one sequence. Sized from the [`Config`] at
/// construction; reused across tokens.
pub struct RunState {
    x: Vec<f32>,     // dim: residual stream
    xb: Vec<f32>,    // dim: norm/attention scratch
    xb2: Vec<f32>,   // dim
    hb: Vec<f32>,    // hidden_dim
    hb2: Vec<f32>,   // hidden_dim
    xq_q: Vec<i8>,   // dim: quantized activation values
    xq_s: Vec<f32>,  // dim/gs: quantized activation scales
    hq_q: Vec<i8>,   // hidden_dim
    hq_s: Vec<f32>,  // hidden_dim/gs
    q: Vec<f32>,     // dim
    k: Vec<f32>,     // kv_dim
    v: Vec<f32>,     // kv_dim
    att: Vec<f32>,   // n_heads * seq_len
    logits: Vec<f32>, // vocab
    key_cache: Vec<f32>,   // n_layers * seq_len * kv_dim
    value_cache: Vec<f32>, // n_layers * seq_len * kv_dim
}

impl RunState {
    pub fn new(c: &Config) -> Self {
        let kv_dim = c.kv_dim();
        let gs = c.group_size;
        RunState {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            xq_q: vec![0; c.dim],
            xq_s: vec![0.0; c.dim / gs],
            hq_q: vec![0; c.hidden_dim],
            hq_s: vec![0.0; c.hidden_dim / gs],
            q: vec![0.0; c.dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; c.vocab_size],
            key_cache: vec![0.0; c.n_layers * c.seq_len * kv_dim],
            value_cache: vec![0.0; c.n_layers * c.seq_len * kv_dim],
        }
    }

    /// The logits produced by the last [`Transformer::forward`] call.
    #[inline]
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    /// Approximate heap footprint in bytes — handy for the device heap budget.
    pub fn heap_bytes(&self) -> usize {
        let f = core::mem::size_of::<f32>();
        (self.x.len()
            + self.xb.len()
            + self.xb2.len()
            + self.hb.len()
            + self.hb2.len()
            + self.xq_s.len()
            + self.hq_s.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.att.len()
            + self.logits.len()
            + self.key_cache.len()
            + self.value_cache.len())
            * f
            + self.xq_q.len()
            + self.hq_q.len()
    }
}

/// A parsed model + its forward pass.
pub struct Transformer<'a> {
    model: Model<'a>,
}

impl<'a> Transformer<'a> {
    /// Parse a checkpoint buffer (see [`Model::parse`]).
    pub fn new(buf: &'a [u8]) -> Result<Self, ModelError> {
        Ok(Transformer {
            model: Model::parse(buf)?,
        })
    }

    #[inline]
    pub fn config(&self) -> &Config {
        &self.model.config
    }

    /// Run one decode step for `token` at position `pos`, leaving the logits in
    /// `s.logits()`. `pos` must be `< config.seq_len`.
    pub fn forward(&self, s: &mut RunState, token: usize, pos: usize) {
        let c = &self.model.config;
        let dim = c.dim;
        let kv_dim = c.kv_dim();
        let kv_mul = c.n_heads / c.n_kv_heads;
        let hidden = c.hidden_dim;
        let head_size = c.head_size();
        let gs = c.group_size;

        self.model.embed_row(token, &mut s.x);

        for l in 0..c.n_layers {
            rmsnorm(&mut s.xb, &s.x, self.model.rms_att(l));
            quantize(&mut s.xq_q, &mut s.xq_s, &s.xb, gs);
            matmul(&mut s.q, &s.xq_q, &s.xq_s, &self.model.wq(l), dim, dim, gs);
            matmul(&mut s.k, &s.xq_q, &s.xq_s, &self.model.wk(l), dim, kv_dim, gs);
            matmul(&mut s.v, &s.xq_q, &s.xq_s, &self.model.wv(l), dim, kv_dim, gs);

            // RoPE: rotate q (all dim) and k (first kv_dim) in 2D pairs.
            let mut i = 0;
            while i < dim {
                let head_dim = (i % head_size) as f32;
                let freq = 1.0 / libm::powf(10000.0, head_dim / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = libm::cosf(val);
                let fci = libm::sinf(val);
                let rotn = if i < kv_dim { 2 } else { 1 };
                for vsel in 0..rotn {
                    if vsel == 0 {
                        let v0 = s.q[i];
                        let v1 = s.q[i + 1];
                        s.q[i] = v0 * fcr - v1 * fci;
                        s.q[i + 1] = v0 * fci + v1 * fcr;
                    } else {
                        let v0 = s.k[i];
                        let v1 = s.k[i + 1];
                        s.k[i] = v0 * fcr - v1 * fci;
                        s.k[i + 1] = v0 * fci + v1 * fcr;
                    }
                }
                i += 2;
            }

            // Append k,v to the cache at this position.
            let loff = l * c.seq_len * kv_dim;
            let row = loff + pos * kv_dim;
            s.key_cache[row..row + kv_dim].copy_from_slice(&s.k[..kv_dim]);
            s.value_cache[row..row + kv_dim].copy_from_slice(&s.v[..kv_dim]);

            // Multi-head attention over positions 0..=pos.
            let scale = 1.0 / libm::sqrtf(head_size as f32);
            for h in 0..c.n_heads {
                let qoff = h * head_size;
                let aoff = h * c.seq_len;
                let kvh = (h / kv_mul) * head_size;
                for t in 0..=pos {
                    let koff = loff + t * kv_dim + kvh;
                    let mut score = 0.0f32;
                    for j in 0..head_size {
                        score += s.q[qoff + j] * s.key_cache[koff + j];
                    }
                    s.att[aoff + t] = score * scale;
                }
                softmax(&mut s.att[aoff..aoff + pos + 1]);
                let xboff = h * head_size;
                for j in 0..head_size {
                    s.xb[xboff + j] = 0.0;
                }
                for t in 0..=pos {
                    let voff = loff + t * kv_dim + kvh;
                    let a = s.att[aoff + t];
                    for j in 0..head_size {
                        s.xb[xboff + j] += a * s.value_cache[voff + j];
                    }
                }
            }

            // Output projection, residual add.
            quantize(&mut s.xq_q, &mut s.xq_s, &s.xb, gs);
            matmul(&mut s.xb2, &s.xq_q, &s.xq_s, &self.model.wo(l), dim, dim, gs);
            for i in 0..dim {
                s.x[i] += s.xb2[i];
            }

            // FFN: w2( silu(w1 x) * w3 x ), residual add.
            rmsnorm(&mut s.xb, &s.x, self.model.rms_ffn(l));
            quantize(&mut s.xq_q, &mut s.xq_s, &s.xb, gs);
            matmul(&mut s.hb, &s.xq_q, &s.xq_s, &self.model.w1(l), dim, hidden, gs);
            matmul(&mut s.hb2, &s.xq_q, &s.xq_s, &self.model.w3(l), dim, hidden, gs);
            for i in 0..hidden {
                s.hb[i] = silu(s.hb[i]) * s.hb2[i];
            }
            quantize(&mut s.hq_q, &mut s.hq_s, &s.hb, gs);
            matmul(&mut s.xb, &s.hq_q, &s.hq_s, &self.model.w2(l), hidden, dim, gs);
            for i in 0..dim {
                s.x[i] += s.xb[i];
            }
        }

        // Final norm + classifier. rmsnorm can't alias in/out, so norm via xb.
        s.xb.copy_from_slice(&s.x);
        rmsnorm(&mut s.x, &s.xb, self.model.rms_final());
        quantize(&mut s.xq_q, &mut s.xq_s, &s.x, gs);
        matmul(
            &mut s.logits,
            &s.xq_q,
            &s.xq_s,
            &self.model.wcls(),
            dim,
            c.vocab_size,
            gs,
        );
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::format::build_tiny;

    #[test]
    fn forward_is_finite_and_deterministic() {
        let (c, bytes) = build_tiny();
        let t = Transformer::new(&bytes).expect("parse");
        assert_eq!(t.config(), &c);

        // Decode a few positions; logits must be finite and reproducible.
        let mut s1 = RunState::new(t.config());
        let mut s2 = RunState::new(t.config());
        let tokens = [1usize, 5, 2, 7];
        let mut first_logits = alloc::vec::Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            t.forward(&mut s1, tok, pos);
            assert!(s1.logits().iter().all(|x| x.is_finite()), "pos {pos} non-finite");
            if pos == tokens.len() - 1 {
                first_logits = s1.logits().to_vec();
            }
        }
        for (pos, &tok) in tokens.iter().enumerate() {
            t.forward(&mut s2, tok, pos);
        }
        assert_eq!(first_logits, s2.logits(), "forward not deterministic");
        assert_eq!(first_logits.len(), c.vocab_size);
    }

    #[test]
    fn heap_bytes_is_reasonable() {
        let (_, bytes) = build_tiny();
        let t = Transformer::new(&bytes).unwrap();
        let s = RunState::new(t.config());
        // tiny model: a few kB, definitely under 1 MB
        assert!(s.heap_bytes() > 0 && s.heap_bytes() < 1_000_000);
    }
}
