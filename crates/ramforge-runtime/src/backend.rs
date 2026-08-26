//! Compute backend abstraction with SIMD and threading support
//!
//! - AVX2 SIMD for x86_64 with scalar fallback (runtime-detected)
//! - Configurable threading via rayon
//!
//! ## Matrix layout convention (GGML/GGUF)
//!
//! `matvec` follows the single explicit ggml convention:
//! - `w_shape = [in_features, out_features]`
//! - `w` holds `out` contiguous rows of `in` elements (row-major `[out][in]`)
//! - `y[o] = sum_i w[o * in + i] * x[i]`
//!
//! Arity mismatches are hard errors, never orientation guesses.

use std::fmt::Debug;

use rayon::prelude::*;

use crate::simd;

pub trait ComputeBackend: Debug + Send + Sync {
    fn name(&self) -> &'static str;

    /// F32 matvec in explicit ggml layout; errors on arity mismatch.
    fn matvec(&self, w: &[f32], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), String>;

    fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32, y: &mut [f32]);

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    fn silu(&self, x: &[f32], out: &mut [f32]);

    fn softmax(&self, x: &mut [f32]);
}

#[derive(Debug, Clone)]
pub struct CpuBackend {
    pub num_threads: usize,
    pub use_simd: bool,
}

impl Default for CpuBackend {
    fn default() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let use_simd = simd::is_avx2_available();
        Self {
            num_threads,
            use_simd,
        }
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            ..Self::default()
        }
    }

    pub fn scalar() -> Self {
        Self {
            num_threads: 1,
            use_simd: false,
        }
    }

    pub fn with_simd(use_simd: bool) -> Self {
        Self {
            use_simd: use_simd && simd::is_avx2_available(),
            ..Self::default()
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        if self.use_simd {
            "CPU-SIMD"
        } else {
            "CPU-scalar"
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn matvec(&self, w: &[f32], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), String> {
        if w_shape.len() != 2 {
            return Err(format!(
                "backend matvec expects 2D w_shape [in, out] (ggml layout), got {:?}",
                w_shape
            ));
        }
        let in_dim = w_shape[0];
        let out_dim = w_shape[1];

        if x.len() != in_dim || y.len() != out_dim || w.len() < in_dim * out_dim {
            return Err(format!(
                "backend matvec arity mismatch (ggml layout [in, out]): shape {:?} implies in={}, out={}, but got w.len()={}, x.len()={}, y.len()={}",
                w_shape,
                in_dim,
                out_dim,
                w.len(),
                x.len(),
                y.len()
            ));
        }

        // Buffer is row-major [out][in]: y[o] = sum_i w[o*in + i] * x[i].
        if self.num_threads > 1 && out_dim >= 4 {
            // Parallelize rows via rayon (global pool).
            y.par_iter_mut().enumerate().for_each(|(o, yo)| {
                let row_offset = o * in_dim;
                let row = &w[row_offset..row_offset + in_dim];
                if self.use_simd {
                    *yo = simd::dot_f32_avx2(row, x);
                } else {
                    *yo = simd::dot_f32_scalar(row, x);
                }
            });
        } else if self.use_simd {
            simd::matvec_f32_avx2(w, out_dim, in_dim, x, y);
        } else {
            for o in 0..out_dim {
                let mut sum = 0.0;
                let row_offset = o * in_dim;
                for i in 0..in_dim {
                    sum += w[row_offset + i] * x[i];
                }
                y[o] = sum;
            }
        }
        Ok(())
    }

    fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32, y: &mut [f32]) {
        let mut sum = 0.0f32;
        for &v in x {
            sum += v * v;
        }
        let mean = sum / (x.len() as f32);
        let rms = (mean + eps).sqrt();
        for i in 0..x.len() {
            y[i] = x[i] / rms * weight.get(i).copied().unwrap_or(1.0);
        }
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len().min(b.len()).min(out.len()) {
            out[i] = a[i] + b[i];
        }
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len().min(b.len()).min(out.len()) {
            out[i] = a[i] * b[i];
        }
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) {
        for i in 0..x.len().min(out.len()) {
            let v = x[i];
            out[i] = v / (1.0 + (-v).exp());
        }
    }

    fn softmax(&self, x: &mut [f32]) {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in x.iter_mut() {
                *v /= sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec() {
        let backend = CpuBackend::scalar();
        // ggml layout [in=2, out=2], rows [1,2],[3,4]
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        backend.matvec(&w, &[2, 2], &x, &mut y).unwrap();
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_matvec_non_square_ggml_layout() {
        let backend = CpuBackend::scalar();
        // ggml [in=3, out=2]: rows [1,2,3] and [4,5,6]
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0; 2];
        backend.matvec(&w, &[3, 2], &x, &mut y).unwrap();
        assert_eq!(y, vec![6.0, 15.0]);
    }

    #[test]
    fn test_matvec_arity_mismatch_is_error() {
        let backend = CpuBackend::scalar();
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 3];
        // x.len()==2 and y.len()==3 happen to match the ORIENTATION-FLIPPED
        // reading of [3,2] – the backend must still refuse to guess.
        let err = backend.matvec(&w, &[3, 2], &x, &mut y).unwrap_err();
        assert!(err.contains("arity mismatch"), "got: {}", err);
    }

    #[test]
    fn test_matvec_simd_vs_scalar() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y_scalar = vec![0.0; 2];
        let mut y_simd = vec![0.0; 2];

        let scalar_backend = CpuBackend::scalar();
        scalar_backend
            .matvec(&w, &[3, 2], &x, &mut y_scalar)
            .unwrap();

        let simd_backend = CpuBackend {
            num_threads: 1,
            use_simd: true,
        };
        simd_backend.matvec(&w, &[3, 2], &x, &mut y_simd).unwrap();

        for i in 0..2 {
            assert!((y_scalar[i] - y_simd[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_matvec_threaded() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = vec![1.0, 1.0];
        let mut y_single = vec![0.0; 4];
        let mut y_multi = vec![0.0; 4];

        let single = CpuBackend::with_threads(1);
        single.matvec(&w, &[2, 4], &x, &mut y_single).unwrap();

        let multi = CpuBackend::with_threads(4);
        multi.matvec(&w, &[2, 4], &x, &mut y_multi).unwrap();

        assert_eq!(y_single, y_multi);
    }

    #[test]
    fn test_rmsnorm() {
        let backend = CpuBackend::new();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 4];
        backend.rmsnorm(&x, &w, 1e-5, &mut y);
        for &v in &y {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_silu() {
        let backend = CpuBackend::new();
        let x = vec![0.0, 1.0];
        let mut y = vec![0.0; 2];
        backend.silu(&x, &mut y);
        assert_eq!(y[0], 0.0);
        assert!((y[1] - 0.7310586).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let backend = CpuBackend::new();
        let mut x = vec![1.0, 1.0, 1.0];
        backend.softmax(&mut x);
        assert!((x[0] - 1.0 / 3.0).abs() < 1e-4);
    }
}
