//! Logit sampling, ported from llama2.c: greedy argmax, plus temperature +
//! top-p (nucleus) with the same xorshift64 RNG so a seeded host run can be
//! reproduced bit-for-bit.

use alloc::vec::Vec;

use crate::math::softmax;

pub struct Sampler {
    temperature: f32,
    topp: f32,
    rng: u64,
    // Scratch reused across calls for the (index, prob) pairs in top-p.
    probindex: Vec<(usize, f32)>,
}

impl Sampler {
    /// `temperature` 0.0 = greedy (argmax). `topp` in (0,1) enables nucleus
    /// sampling; `topp >= 1.0` samples from the full distribution.
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Sampler {
            temperature,
            topp,
            rng: if seed == 0 { 1 } else { seed },
            probindex: Vec::with_capacity(vocab_size),
        }
    }

    #[inline]
    fn random_u32(&mut self) -> u32 {
        // xorshift64* — identical to llama2.c's `random_u32`.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }

    #[inline]
    fn random_f32(&mut self) -> f32 {
        (self.random_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Pick the next token from `logits` (modified in place when temperature
    /// applies). Returns the chosen token id.
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            return argmax(logits);
        }
        for v in logits.iter_mut() {
            *v /= self.temperature;
        }
        softmax(logits);
        let coin = self.random_f32();
        if self.topp <= 0.0 || self.topp >= 1.0 {
            sample_mult(logits, coin)
        } else {
            self.sample_topp(logits, coin)
        }
    }

    fn sample_topp(&mut self, probs: &[f32], coin: f32) -> usize {
        let n = probs.len();
        self.probindex.clear();
        // Pre-filter: only tokens with prob above the cutoff can be in the
        // nucleus (llama2.c's optimization).
        let cutoff = (1.0 - self.topp) / (n as f32 - 1.0);
        for (i, &p) in probs.iter().enumerate() {
            if p >= cutoff {
                self.probindex.push((i, p));
            }
        }
        self.probindex
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

        // Truncate where cumulative prob exceeds topp.
        let mut cumulative = 0.0f32;
        let mut last = self.probindex.len().saturating_sub(1);
        for (i, &(_, p)) in self.probindex.iter().enumerate() {
            cumulative += p;
            if cumulative > self.topp {
                last = i;
                break;
            }
        }
        // Sample within the truncated nucleus.
        let r = coin * cumulative;
        let mut cdf = 0.0f32;
        for &(idx, p) in &self.probindex[..=last] {
            cdf += p;
            if r < cdf {
                return idx;
            }
        }
        self.probindex[last].0
    }
}

/// Index of the largest logit.
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Sample an index from a probability distribution given a coin in [0,1).
fn sample_mult(probs: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if coin < cdf {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(4, 0.0, 0.0, 42);
        let mut logits = vec![0.1, 0.9, 0.3, -2.0];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn temperature_sampling_is_seed_reproducible() {
        let logits0 = vec![1.0f32, 2.0, 0.5, 3.0, -1.0, 0.2, 1.5, 0.0];
        let mut a = Sampler::new(8, 1.0, 0.9, 777);
        let mut b = Sampler::new(8, 1.0, 0.9, 777);
        for _ in 0..20 {
            let ia = a.sample(&mut logits0.clone());
            let ib = b.sample(&mut logits0.clone());
            assert_eq!(ia, ib);
            assert!(ia < 8);
        }
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[3.0, 1.0, 9.0, 2.0]), 2);
    }
}
