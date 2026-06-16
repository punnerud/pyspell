//! The numeric primitives, ported from llama2.c `run.c` so a host build matches
//! it. All transcendentals go through `libm` (no std) for host/device parity.

use crate::format::{rd_f32, QTensor};

/// RMSNorm: `out = x / rms(x) * weight`, with llama2.c's `1e-5` epsilon. The
/// weight is read as f32 from a (possibly unaligned) byte slice.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight_bytes: &[u8]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    ss = ss / (n as f32) + 1e-5;
    ss = 1.0 / libm::sqrtf(ss);
    for i in 0..n {
        out[i] = weight_bytes_at(weight_bytes, i) * (ss * x[i]);
    }
}

#[inline]
fn weight_bytes_at(b: &[u8], i: usize) -> f32 {
    rd_f32(b, i * 4)
}

/// In-place numerically-stable softmax over `x`.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = x[0];
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// Quantize an activation vector `x` (length `n`, a multiple of `gs`) into int8
/// values `q` with per-group f32 scales `s` (`n/gs` of them) — the Q8_0 scheme.
pub fn quantize(q: &mut [i8], s: &mut [f32], x: &[f32], gs: usize) {
    const Q_MAX: f32 = 127.0;
    let num_groups = x.len() / gs;
    for g in 0..num_groups {
        let base = g * gs;
        let mut wmax = 0.0f32;
        for k in 0..gs {
            let a = libm::fabsf(x[base + k]);
            if a > wmax {
                wmax = a;
            }
        }
        let scale = wmax / Q_MAX;
        s[g] = scale;
        for k in 0..gs {
            let qv = if scale != 0.0 {
                libm::roundf(x[base + k] / scale)
            } else {
                0.0
            };
            q[base + k] = qv as i8;
        }
    }
}

/// `out[d] = W[d,n] @ x[n]` where both the weight `w` and the activation
/// (`xq` int8 + `xs` scales) are Q8_0 with the same group size `gs`. Inner
/// products accumulate in i32 per group, then rescale to f32 — exactly
/// llama2.c's `matmul`.
pub fn matmul(out: &mut [f32], xq: &[i8], xs: &[f32], w: &QTensor, n: usize, d: usize, gs: usize) {
    for i in 0..d {
        let mut val = 0.0f32;
        let row = i * n;
        let mut j = 0;
        while j < n {
            let mut ival: i32 = 0;
            for k in 0..gs {
                ival += (xq[j + k] as i32) * w.qi(row + j + k);
            }
            val += (ival as f32) * w.scale((row + j) / gs) * xs[j / gs];
            j += gs;
        }
        out[i] = val;
    }
}

/// SiLU/swish: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + libm::expf(-x)))
}

/// Host-only: quantize `x` and append it to `out` in the on-disk Q8_0 layout
/// (all int8 values, then all f32 group scales). Used by `format::write_v2`.
#[cfg(feature = "std")]
pub fn quantize_to_bytes(out: &mut alloc::vec::Vec<u8>, x: &[f32], gs: usize) {
    const Q_MAX: f32 = 127.0;
    let num_groups = x.len() / gs;
    let mut scales = alloc::vec::Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let base = g * gs;
        let mut wmax = 0.0f32;
        for k in 0..gs {
            let a = libm::fabsf(x[base + k]);
            if a > wmax {
                wmax = a;
            }
        }
        let scale = wmax / Q_MAX;
        scales.push(scale);
        for k in 0..gs {
            let qv = if scale != 0.0 {
                libm::roundf(x[base + k] / scale)
            } else {
                0.0
            };
            out.push((qv as i8) as u8);
        }
    }
    for s in scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn quantize_dequantize_round_trips() {
        let gs = 8usize;
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.21).sin() * 3.0).collect();
        let mut q = vec![0i8; x.len()];
        let mut s = vec![0.0f32; x.len() / gs];
        quantize(&mut q, &mut s, &x, gs);
        for i in 0..x.len() {
            let deq = q[i] as f32 * s[i / gs];
            // Per-group scale = max/127, so error per element <= scale.
            assert!((deq - x[i]).abs() <= s[i / gs] + 1e-6, "i={i}: {deq} vs {}", x[i]);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -1.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(v.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn silu_is_zero_at_zero() {
        assert!(silu(0.0).abs() < 1e-7);
        assert!(silu(10.0) > 9.9); // approaches x for large x
    }
}
