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

/// Allocate a zeroed `Vec<f32>` of length `n` with `try_reserve` (returns `None` on OOM
/// instead of aborting — for tight, fragmented device heaps).
fn zf(n: usize) -> Option<Vec<f32>> {
    let mut v: Vec<f32> = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0.0);
    Some(v)
}

/// Same for `Vec<i8>`.
fn zi(n: usize) -> Option<Vec<i8>> {
    let mut v: Vec<i8> = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

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
    att: Vec<f32>,   // n_heads * ctx
    logits: Vec<f32>, // vocab
    // KV cache. `ctx` = allocated context length (seq_len for f32; max_ctx for int8).
    // Two representations: f32 (run.c-exact, for the host gate) or int8 + a per-vector
    // scale (4× smaller — the device path, since the cache dominates RAM). Only one set
    // is allocated; the other stays empty.
    ctx: usize,
    int8: bool,
    key_cache: Vec<f32>,   // f32: n_layers * ctx * kv_dim
    value_cache: Vec<f32>,
    key_q: Vec<i8>,        // int8: n_layers * ctx * kv_dim
    val_q: Vec<i8>,
    key_s: Vec<f32>,       // int8: n_layers * ctx (one scale per cached kv vector)
    val_s: Vec<f32>,
}

impl RunState {
    pub fn new(c: &Config) -> Self {
        Self::try_with_cache(c, c.seq_len, false).expect("RunState alloc")
    }

    /// Device path: int8 KV cache bounded to `max_ctx` tokens (4× smaller than the f32
    /// cache, and `max_ctx` ≪ `seq_len` keeps it tiny). `pos` passed to `forward` must
    /// stay `< max_ctx`. Math is otherwise identical to [`new`]. Panics on OOM — use
    /// [`try_new_int8`](Self::try_new_int8) on a memory-constrained device.
    pub fn new_int8(c: &Config, max_ctx: usize) -> Self {
        Self::try_with_cache(c, max_ctx.min(c.seq_len), true).expect("RunState alloc")
    }

    /// Fallible int8 constructor for tight heaps (the ESP32): every buffer is allocated
    /// with `try_reserve`, returning `None` instead of aborting if the (fragmented) heap
    /// can't satisfy it — so a low-memory moment is a clean error, never a reboot.
    pub fn try_new_int8(c: &Config, max_ctx: usize) -> Option<Self> {
        Self::try_with_cache(c, max_ctx.min(c.seq_len), true)
    }

    fn try_with_cache(c: &Config, ctx: usize, int8: bool) -> Option<Self> {
        let kv_dim = c.kv_dim();
        let gs = c.group_size;
        let cache = c.n_layers * ctx * kv_dim;
        // One int8 scale per (layer, position, kv-head) — i.e. quantize each head's
        // head_size values independently, so one head's outliers don't crush another's
        // precision. Storage is tiny (n_kv_heads scales per cached vector).
        let scales = c.n_layers * ctx * c.n_kv_heads;
        Some(RunState {
            x: zf(c.dim)?,
            xb: zf(c.dim)?,
            xb2: zf(c.dim)?,
            hb: zf(c.hidden_dim)?,
            hb2: zf(c.hidden_dim)?,
            xq_q: zi(c.dim)?,
            xq_s: zf(c.dim / gs)?,
            hq_q: zi(c.hidden_dim)?,
            hq_s: zf(c.hidden_dim / gs)?,
            q: zf(c.dim)?,
            k: zf(kv_dim)?,
            v: zf(kv_dim)?,
            att: zf(c.n_heads * ctx)?,
            logits: zf(c.vocab_size)?,
            ctx,
            int8,
            key_cache: zf(if int8 { 0 } else { cache })?,
            value_cache: zf(if int8 { 0 } else { cache })?,
            key_q: zi(if int8 { cache } else { 0 })?,
            val_q: zi(if int8 { cache } else { 0 })?,
            key_s: zf(if int8 { scales } else { 0 })?,
            val_s: zf(if int8 { scales } else { 0 })?,
        })
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
            + self.value_cache.len()
            + self.key_s.len()
            + self.val_s.len())
            * f
            + self.xq_q.len()
            + self.hq_q.len()
            + self.key_q.len()
            + self.val_q.len()
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

            // Append k,v to the cache at this position. f32 = run.c-exact copy; int8 =
            // quantize each kv vector with one scale (gs = kv_dim) — 4× smaller.
            let loff = l * s.ctx * kv_dim;
            let row = loff + pos * kv_dim;
            if s.int8 {
                let nkv = c.n_kv_heads;
                let si = (l * s.ctx + pos) * nkv;
                quantize(&mut s.key_q[row..row + kv_dim], &mut s.key_s[si..si + nkv], &s.k[..kv_dim], head_size);
                quantize(&mut s.val_q[row..row + kv_dim], &mut s.val_s[si..si + nkv], &s.v[..kv_dim], head_size);
            } else {
                s.key_cache[row..row + kv_dim].copy_from_slice(&s.k[..kv_dim]);
                s.value_cache[row..row + kv_dim].copy_from_slice(&s.v[..kv_dim]);
            }

            // Multi-head attention over positions 0..=pos (dequantizing the int8 cache
            // on the fly — one scale per kv vector, hoisted out of the inner loop).
            let scale = 1.0 / libm::sqrtf(head_size as f32);
            for h in 0..c.n_heads {
                let qoff = h * head_size;
                let aoff = h * s.ctx;
                let kvh = (h / kv_mul) * head_size;
                for t in 0..=pos {
                    let koff = loff + t * kv_dim + kvh;
                    let mut score = 0.0f32;
                    if s.int8 {
                        let ks = s.key_s[(l * s.ctx + t) * c.n_kv_heads + h / kv_mul];
                        for j in 0..head_size {
                            score += s.q[qoff + j] * (s.key_q[koff + j] as f32) * ks;
                        }
                    } else {
                        for j in 0..head_size {
                            score += s.q[qoff + j] * s.key_cache[koff + j];
                        }
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
                    if s.int8 {
                        let vs = s.val_s[(l * s.ctx + t) * c.n_kv_heads + h / kv_mul];
                        for j in 0..head_size {
                            s.xb[xboff + j] += a * (s.val_q[voff + j] as f32) * vs;
                        }
                    } else {
                        for j in 0..head_size {
                            s.xb[xboff + j] += a * s.value_cache[voff + j];
                        }
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

    #[test]
    fn int8_cache_matches_f32_and_is_smaller() {
        let (_c, bytes) = build_tiny();
        let t = Transformer::new(&bytes).unwrap();
        let tokens = [1usize, 5, 2, 7];
        let mut f = RunState::new(t.config());
        let mut q = RunState::new_int8(t.config(), 16);
        let (mut lf, mut lq) = (vec![], vec![]);
        for (pos, &tok) in tokens.iter().enumerate() {
            t.forward(&mut f, tok, pos);
            t.forward(&mut q, tok, pos);
            lf = f.logits().to_vec();
            lq = q.logits().to_vec();
        }
        // int8 KV is lossy but must track the f32 path closely and pick the same argmax.
        let amax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(amax(&lf), amax(&lq), "int8 cache changed the predicted token");
        let mse: f32 = lf.iter().zip(&lq).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / lf.len() as f32;
        assert!(mse < 1.0, "int8 cache logits drift too far (mse {mse})");
        // int8 cache (16 ctx) must use far less RAM than the f32 cache (seq_len ctx).
        assert!(q.heap_bytes() < f.heap_bytes(), "int8 cache not smaller");
    }
}
