//! Reader for the llama2.c **`version 2` (Q8_0) checkpoint** layout.
//!
//! Header (little-endian, padded to 256 bytes):
//! ```text
//!   u32  magic = 0x616b3432 ("ak42")
//!   i32  version = 2
//!   i32  dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//!   u8   shared_classifier
//!   i32  group_size (GS)
//!   ...  zero padding up to offset 256
//! ```
//! Then the payload at offset 256:
//! ```text
//!   f32  rms_att_weight   [n_layers * dim]
//!   f32  rms_ffn_weight   [n_layers * dim]
//!   f32  rms_final_weight [dim]
//!   Q8   q_tokens         (1 tensor, vocab * dim)
//!   Q8   wq, wk, wv, wo   (n_layers tensors each)
//!   Q8   w1, w2, w3       (n_layers tensors each)
//!   Q8   wcls             (1 tensor, only if !shared_classifier)
//! ```
//! Each Q8 tensor of N elements is `N` int8 values followed by `N/GS` f32 group
//! scales. We never copy the weights: a [`QTensor`] is just two sub-slices of the
//! backing buffer, and scalars are read on the fly with `from_le_bytes` (so no
//! alignment assumptions — the buffer may be an mmap'd flash window).

#[cfg(feature = "std")]
use alloc::vec::Vec;
use core::fmt;

pub const MAGIC: u32 = 0x616b_3432; // "ak42"
pub const VERSION: i32 = 2;
const HEADER_BYTES: usize = 256;

/// Transformer hyper-parameters, as stored in the checkpoint header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub group_size: usize,
    pub shared_classifier: bool,
}

impl Config {
    #[inline]
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }
    /// Width of one key/value vector (GQA-aware).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        (self.dim * self.n_kv_heads) / self.n_heads
    }
}

/// What can go wrong while parsing a checkpoint buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    TooShort,
    BadMagic(u32),
    BadVersion(i32),
    /// A dimension was zero, or `group_size` does not divide a tensor length.
    BadConfig,
    /// The declared tensors run past the end of the buffer.
    Truncated,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::TooShort => f.write_str("checkpoint shorter than the 256-byte header"),
            ModelError::BadMagic(m) => write!(f, "bad magic 0x{m:08x} (expected 0x616b3432 'ak42')"),
            ModelError::BadVersion(v) => write!(f, "unsupported version {v} (expected 2)"),
            ModelError::BadConfig => f.write_str("invalid config (zero dim or group_size mismatch)"),
            ModelError::Truncated => f.write_str("checkpoint truncated: tensors exceed buffer"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ModelError {}

#[inline]
fn rd_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
pub(crate) fn rd_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// A borrowed Q8_0 weight tensor: `n` int8 values + `n/gs` f32 scales, both as
/// raw byte sub-slices of the checkpoint buffer (no copy, no alignment needs).
#[derive(Clone, Copy)]
pub struct QTensor<'a> {
    q: &'a [u8], // n int8 values (one byte each, reinterpreted as i8)
    s: &'a [u8], // n/gs f32 scales, little-endian
}

impl<'a> QTensor<'a> {
    #[inline]
    pub fn qi(&self, i: usize) -> i32 {
        self.q[i] as i8 as i32
    }
    #[inline]
    pub fn scale(&self, group: usize) -> f32 {
        rd_f32(self.s, group * 4)
    }
}

/// A contiguous run of `count` Q8 tensors that all have the same element count
/// `size_each` (e.g. the per-layer `wq` weights). Stored as the byte offset of
/// the first tensor; `tensor(l)` slices out tensor `l` on demand.
#[derive(Clone, Copy)]
struct QGroup {
    base: usize,
    count: usize,
    size_each: usize,
    gs: usize,
}

impl QGroup {
    /// Bytes occupied by one tensor: int8 values + f32 group scales.
    #[inline]
    fn stride(&self) -> usize {
        self.size_each + (self.size_each / self.gs) * 4
    }
    #[inline]
    fn total(&self) -> usize {
        self.stride() * self.count
    }
    #[inline]
    fn tensor<'a>(&self, buf: &'a [u8], l: usize) -> QTensor<'a> {
        let off = self.base + l * self.stride();
        let q = &buf[off..off + self.size_each];
        let s_off = off + self.size_each;
        let s = &buf[s_off..s_off + (self.size_each / self.gs) * 4];
        QTensor { q, s }
    }
}

/// A parsed, zero-copy view of a checkpoint. Holds the backing buffer plus the
/// computed byte offsets of every weight group; nothing is materialized.
pub struct Model<'a> {
    pub config: Config,
    buf: &'a [u8],
    // fp32 norm weights, as byte offsets into `buf`.
    rms_att: usize,
    rms_ffn: usize,
    rms_final: usize,
    // quantized weight groups.
    q_tokens: QGroup,
    wq: QGroup,
    wk: QGroup,
    wv: QGroup,
    wo: QGroup,
    w1: QGroup,
    w2: QGroup,
    w3: QGroup,
    wcls: QGroup,
}

impl<'a> Model<'a> {
    /// Parse a checkpoint buffer, validating the header and that every declared
    /// tensor fits. Returns a view that borrows `buf` for its whole lifetime.
    pub fn parse(buf: &'a [u8]) -> Result<Self, ModelError> {
        if buf.len() < HEADER_BYTES {
            return Err(ModelError::TooShort);
        }
        let magic = rd_u32(buf, 0);
        if magic != MAGIC {
            return Err(ModelError::BadMagic(magic));
        }
        let version = rd_i32(buf, 4);
        if version != VERSION {
            return Err(ModelError::BadVersion(version));
        }
        let dim = rd_i32(buf, 8);
        let hidden_dim = rd_i32(buf, 12);
        let n_layers = rd_i32(buf, 16);
        let n_heads = rd_i32(buf, 20);
        let n_kv_heads = rd_i32(buf, 24);
        let vocab_size = rd_i32(buf, 28);
        let seq_len = rd_i32(buf, 32);
        let shared_classifier = buf[36] != 0;
        let group_size = rd_i32(buf, 37);

        // Reject anything non-positive before casting to usize.
        if dim <= 0
            || hidden_dim <= 0
            || n_layers <= 0
            || n_heads <= 0
            || n_kv_heads <= 0
            || vocab_size <= 0
            || seq_len <= 0
            || group_size <= 0
        {
            return Err(ModelError::BadConfig);
        }
        let config = Config {
            dim: dim as usize,
            hidden_dim: hidden_dim as usize,
            n_layers: n_layers as usize,
            n_heads: n_heads as usize,
            n_kv_heads: n_kv_heads as usize,
            vocab_size: vocab_size as usize,
            seq_len: seq_len as usize,
            group_size: group_size as usize,
            shared_classifier,
        };
        let gs = config.group_size;
        if config.dim % config.n_heads != 0 {
            return Err(ModelError::BadConfig);
        }
        let head_size = config.head_size();
        let kv_dim = config.kv_dim();
        // Every quantized tensor length must be a multiple of the group size.
        let lens = [
            config.vocab_size * config.dim,
            config.dim * config.dim,         // wq: dim * (n_heads*head_size) = dim*dim
            config.dim * kv_dim,             // wk
            config.dim * kv_dim,             // wv
            config.dim * config.dim,         // wo
            config.dim * config.hidden_dim,  // w1
            config.hidden_dim * config.dim,  // w2
            config.dim * config.hidden_dim,  // w3
        ];
        for &len in &lens {
            if len == 0 || len % gs != 0 {
                return Err(ModelError::BadConfig);
            }
        }
        let _ = head_size; // documented above; kept for clarity

        // --- walk the payload, computing byte offsets ---
        let mut off = HEADER_BYTES;
        let rms_att = off;
        off += config.n_layers * config.dim * 4;
        let rms_ffn = off;
        off += config.n_layers * config.dim * 4;
        let rms_final = off;
        off += config.dim * 4;

        let take = |count: usize, size_each: usize, off: &mut usize| -> QGroup {
            let g = QGroup {
                base: *off,
                count,
                size_each,
                gs,
            };
            *off += g.total();
            g
        };

        let q_tokens = take(1, config.vocab_size * config.dim, &mut off);
        let wq = take(config.n_layers, config.dim * config.dim, &mut off);
        let wk = take(config.n_layers, config.dim * kv_dim, &mut off);
        let wv = take(config.n_layers, config.dim * kv_dim, &mut off);
        let wo = take(config.n_layers, config.dim * config.dim, &mut off);
        let w1 = take(config.n_layers, config.dim * config.hidden_dim, &mut off);
        let w2 = take(config.n_layers, config.hidden_dim * config.dim, &mut off);
        let w3 = take(config.n_layers, config.dim * config.hidden_dim, &mut off);
        let wcls = if shared_classifier {
            q_tokens
        } else {
            take(1, config.dim * config.vocab_size, &mut off)
        };

        if off > buf.len() {
            return Err(ModelError::Truncated);
        }

        Ok(Model {
            config,
            buf,
            rms_att,
            rms_ffn,
            rms_final,
            q_tokens,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            wcls,
        })
    }

    // --- fp32 norm-weight rows ---
    #[inline]
    pub(crate) fn rms_att(&self, l: usize) -> &[u8] {
        let o = self.rms_att + l * self.config.dim * 4;
        &self.buf[o..o + self.config.dim * 4]
    }
    #[inline]
    pub(crate) fn rms_ffn(&self, l: usize) -> &[u8] {
        let o = self.rms_ffn + l * self.config.dim * 4;
        &self.buf[o..o + self.config.dim * 4]
    }
    #[inline]
    pub(crate) fn rms_final(&self) -> &[u8] {
        &self.buf[self.rms_final..self.rms_final + self.config.dim * 4]
    }

    /// Dequantize a single token-embedding row (`dim` floats) into `out`. We
    /// dequantize on demand rather than materializing the whole `vocab*dim`
    /// table — that table would be ~512 kB for a 2048-vocab/64-dim model.
    pub fn embed_row(&self, token: usize, out: &mut [f32]) {
        let t = self.q_tokens.tensor(self.buf, 0);
        let dim = self.config.dim;
        let gs = self.config.group_size;
        let base = token * dim;
        for i in 0..dim {
            let idx = base + i;
            out[i] = (t.qi(idx) as f32) * t.scale(idx / gs);
        }
    }

    // --- per-layer quantized weights ---
    #[inline]
    pub(crate) fn wq(&self, l: usize) -> QTensor<'a> {
        self.wq.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn wk(&self, l: usize) -> QTensor<'a> {
        self.wk.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn wv(&self, l: usize) -> QTensor<'a> {
        self.wv.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn wo(&self, l: usize) -> QTensor<'a> {
        self.wo.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn w1(&self, l: usize) -> QTensor<'a> {
        self.w1.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn w2(&self, l: usize) -> QTensor<'a> {
        self.w2.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn w3(&self, l: usize) -> QTensor<'a> {
        self.w3.tensor(self.buf, l)
    }
    #[inline]
    pub(crate) fn wcls(&self) -> QTensor<'a> {
        self.wcls.tensor(self.buf, 0)
    }
}

/// Serialize a model to the `version 2` byte layout. Host-only helper used by the
/// tests (and the export tooling) to build checkpoints without llama2.c.
#[cfg(feature = "std")]
pub fn write_v2(
    config: &Config,
    rms_att: &[f32],
    rms_ffn: &[f32],
    rms_final: &[f32],
    q_tokens: &[f32],
    wq: &[Vec<f32>],
    wk: &[Vec<f32>],
    wv: &[Vec<f32>],
    wo: &[Vec<f32>],
    w1: &[Vec<f32>],
    w2: &[Vec<f32>],
    w3: &[Vec<f32>],
    wcls: Option<&[f32]>,
) -> Vec<u8> {
    use crate::math::quantize_to_bytes;
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    for v in [
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.seq_len,
    ] {
        out.extend_from_slice(&(v as i32).to_le_bytes());
    }
    out.push(config.shared_classifier as u8);
    out.extend_from_slice(&(config.group_size as i32).to_le_bytes());
    out.resize(HEADER_BYTES, 0);

    let put_f32 = |out: &mut Vec<u8>, xs: &[f32]| {
        for &x in xs {
            out.extend_from_slice(&x.to_le_bytes());
        }
    };
    put_f32(&mut out, rms_att);
    put_f32(&mut out, rms_ffn);
    put_f32(&mut out, rms_final);

    let gs = config.group_size;
    quantize_to_bytes(&mut out, q_tokens, gs);
    for group in [wq, wk, wv, wo, w1, w2, w3] {
        for t in group {
            quantize_to_bytes(&mut out, t, gs);
        }
    }
    if let Some(w) = wcls {
        quantize_to_bytes(&mut out, w, gs);
    }
    out
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::math::{matmul, quantize, quantize_to_bytes};

    #[test]
    fn matmul_matches_dequantized_reference() {
        // W is (d, n) row-major, x is (n,). Both Q8_0 with group size gs.
        let (n, d, gs) = (16usize, 4usize, 8usize);
        let w: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.137).sin()).collect();
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.317).cos() * 2.0).collect();

        let mut wbytes = Vec::new();
        quantize_to_bytes(&mut wbytes, &w, gs);
        let qt = QTensor {
            q: &wbytes[..n * d],
            s: &wbytes[n * d..],
        };

        let mut xq = alloc::vec![0i8; n];
        let mut xs = alloc::vec![0.0f32; n / gs];
        quantize(&mut xq, &mut xs, &x, gs);

        let mut out = alloc::vec![0.0f32; d];
        matmul(&mut out, &xq, &xs, &qt, n, d, gs);

        // Reference: dequantize element-by-element and sum (same arithmetic).
        for i in 0..d {
            let mut r = 0.0f32;
            for j in 0..n {
                let wq = qt.qi(i * n + j) as f32;
                let ws = qt.scale((i * n + j) / gs);
                let xv = xq[j] as f32 * xs[j / gs];
                r += wq * ws * xv;
            }
            assert!((out[i] - r).abs() < 1e-3, "row {i}: {} vs {}", out[i], r);
        }
    }

    fn tiny_config() -> Config {
        Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            vocab_size: 16,
            seq_len: 8,
            group_size: 8,
            shared_classifier: true,
        }
    }

    // Deterministic pseudo-random f32 in [-1, 1] from an LCG (no rand crate).
    fn fill(seed: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    pub(crate) fn build_tiny() -> (Config, Vec<u8>) {
        let c = tiny_config();
        let mut s = 12345u64;
        let nl = c.n_layers;
        let layer = |s: &mut u64, len: usize| -> Vec<Vec<f32>> {
            (0..nl).map(|_| fill(s, len)).collect()
        };
        let rms_att = fill(&mut s, nl * c.dim);
        let rms_ffn = fill(&mut s, nl * c.dim);
        let rms_final = fill(&mut s, c.dim);
        let q_tokens = fill(&mut s, c.vocab_size * c.dim);
        let wq = layer(&mut s, c.dim * c.dim);
        let wk = layer(&mut s, c.dim * c.kv_dim());
        let wv = layer(&mut s, c.dim * c.kv_dim());
        let wo = layer(&mut s, c.dim * c.dim);
        let w1 = layer(&mut s, c.dim * c.hidden_dim);
        let w2 = layer(&mut s, c.hidden_dim * c.dim);
        let w3 = layer(&mut s, c.dim * c.hidden_dim);
        let bytes = write_v2(
            &c, &rms_att, &rms_ffn, &rms_final, &q_tokens, &wq, &wk, &wv, &wo, &w1, &w2, &w3, None,
        );
        (c, bytes)
    }

    #[test]
    fn parse_round_trips_config() {
        let (c, bytes) = build_tiny();
        let m = Model::parse(&bytes).expect("parse");
        assert_eq!(m.config, c);
    }

    #[test]
    fn parse_rejects_bad_input() {
        // `Model` is a zero-copy view and intentionally not Debug, so match on
        // the error instead of using unwrap_err().
        let err = |r: Result<Model, ModelError>| r.err().unwrap();
        assert_eq!(err(Model::parse(&[0u8; 10])), ModelError::TooShort);
        let mut bytes = build_tiny().1;
        bytes[0] ^= 0xff;
        assert!(matches!(err(Model::parse(&bytes)), ModelError::BadMagic(_)));
        let mut bytes = build_tiny().1;
        bytes[4] = 9; // version
        assert_eq!(err(Model::parse(&bytes)), ModelError::BadVersion(9));
        let bytes = build_tiny().1;
        let short = &bytes[..bytes.len() - 4];
        assert_eq!(err(Model::parse(short)), ModelError::Truncated);
    }
}

#[cfg(all(test, feature = "std"))]
pub(crate) use tests::build_tiny;
