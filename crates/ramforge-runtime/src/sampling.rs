//! Minimal sampling: greedy and temperature

use rand::Rng;

#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
        }
    }

    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }

    /// Sample token ID from logits
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        if self.temperature <= 0.0 {
            // Greedy argmax
            let mut max_id = 0;
            let mut max_val = logits[0];
            for (i, &v) in logits.iter().enumerate().skip(1) {
                if v > max_val {
                    max_val = v;
                    max_id = i;
                }
            }
            return max_id as u32;
        }

        // Temperature scaling
        let mut scaled = logits.to_vec();
        for v in scaled.iter_mut() {
            *v /= self.temperature;
        }

        // Single reusable scratch table for top-k/top-p filtering. Peak
        // scratch memory is exactly `scaled` + `indexed` (one vocab table);
        // the caller budget-guards this via `tmp:sampling`.
        let mut indexed: Vec<(usize, f32)> = Vec::new();

        // Apply top-k filtering if set: keep exactly k entries.
        if let Some(k) = self.top_k {
            if k > 0 && k < scaled.len() {
                indexed.extend(scaled.iter().enumerate().map(|(i, &v)| (i, v)));
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(k);
                for v in scaled.iter_mut() {
                    *v = f32::NEG_INFINITY;
                }
                for &(i, v) in &indexed {
                    scaled[i] = v;
                }
            }
        }

        // Softmax
        let max = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for v in scaled.iter_mut() {
            if *v == f32::NEG_INFINITY {
                *v = 0.0;
            } else {
                *v = (*v - max).exp();
                sum += *v;
            }
        }
        if sum > 0.0 {
            for v in scaled.iter_mut() {
                *v /= sum;
            }
        }

        // Apply top-p (nucleus) filtering if set: reuse the same scratch
        // table, zero everything, then restore the kept prefix.
        if let Some(p) = self.top_p {
            if p < 1.0 && p > 0.0 {
                indexed.clear();
                indexed.extend(scaled.iter().enumerate().map(|(i, &v)| (i, v)));
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut cumsum = 0.0;
                let mut cutoff = scaled.len();
                for (i, (_, prob)) in indexed.iter().enumerate() {
                    cumsum += prob;
                    if cumsum >= p {
                        cutoff = i + 1;
                        break;
                    }
                }
                indexed.truncate(cutoff);
                for v in scaled.iter_mut() {
                    *v = 0.0;
                }
                for &(i, v) in &indexed {
                    scaled[i] = v;
                }
                // Renormalize
                let sum: f32 = scaled.iter().sum();
                if sum > 0.0 {
                    for v in scaled.iter_mut() {
                        *v /= sum;
                    }
                }
            }
        }

        // Sample
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &prob) in scaled.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum {
                return i as u32;
            }
        }
        // Fallback to last
        (scaled.len() - 1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let sampler = Sampler::greedy();
        let logits = vec![0.1, 0.5, 0.2];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_temperature_zero_greedy() {
        let sampler = Sampler::new(0.0, None, None);
        let logits = vec![1.0, 5.0, 2.0];
        assert_eq!(sampler.sample(&logits), 1);
    }

    #[test]
    fn test_temperature_sampling() {
        let sampler = Sampler::new(1.0, None, None);
        let logits = vec![0.0, 10.0, 0.0];
        // With high logit for id 1, should almost always sample 1
        // We test deterministic by sampling many times and checking that 1 is most common, but for unit test just check it doesn't panic and returns valid id
        let id = sampler.sample(&logits);
        assert!(id < 3);
    }

    #[test]
    fn test_top_k() {
        let sampler = Sampler::new(1.0, Some(1), None);
        let logits = vec![1.0, 5.0, 2.0];
        // Top-k 1 should always return argmax
        // Since after filtering only top 1 remains, sampling should return 1
        let id = sampler.sample(&logits);
        assert_eq!(id, 1);
    }
}
