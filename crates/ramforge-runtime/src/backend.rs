//! Compute backend abstraction with SIMD and threading support
//!
//! - AVX2 SIMD for x86_64 with scalar fallback (runtime-detected)
//! - Configurable threading via rayon
//!
//! The backend owns dispatch and parallelism, not model mathematics. Scalar
//! operations delegate to `ramforge_core::compute`; optimized paths are
//! equivalence-tested against that contract.
//!
//! ## Matrix layout convention (GGML/GGUF)
//!
//! `matvec` follows the single explicit ggml convention:
//! - `w_shape = [in_features, out_features]`
//! - `w` holds `out` contiguous rows of `in` elements (row-major `[out][in]`)
//! - `y[o] = sum_i w[o * in + i] * x[i]`
//!
//! Arity mismatches are hard errors, never orientation guesses.

use std::fmt;
use std::sync::Arc;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::simd;

pub trait ComputeBackend: fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;

    /// F32 matvec in explicit ggml layout; errors on arity mismatch.
    fn matvec(&self, w: &[f32], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), String>;

    /// Number of worker threads configured for this backend. Returns 1 for
    /// the scalar backend; 1+ for backends using an explicit thread pool.
    fn num_threads(&self) -> usize;

    /// Run `work` inside this backend's thread pool so that any Rayon
    /// parallel iterators spawned inside it are bounded to the configured
    /// thread count. Single-threaded backends run work inline. Returns
    /// whatever `work` returns so parallel kernels can propagate errors.
    fn in_worker_pool<R: Send>(&self, work: impl FnOnce() -> R + Send) -> R;

    fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32, y: &mut [f32]);

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    fn silu(&self, x: &[f32], out: &mut [f32]);

    fn softmax(&self, x: &mut [f32]);
}

#[derive(Clone)]
pub struct CpuBackend {
    thread_count: usize,
    pub use_simd: bool,
    thread_pool: Option<Arc<ThreadPool>>,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CpuBackend")
            .field("num_threads", &self.thread_count)
            .field("use_simd", &self.use_simd)
            .field("thread_pool_configured", &self.thread_pool.is_some())
            .finish()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        let thread_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let use_simd = simd::is_avx2_available();
        Self {
            thread_count,
            use_simd,
            thread_pool: None,
        }
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self::try_with_threads(num_threads.max(1))
            .expect("failed to construct explicitly sized CPU thread pool")
    }

    pub fn try_with_threads(num_threads: usize) -> Result<Self, String> {
        if num_threads == 0 {
            return Err("CPU thread count must be greater than zero".to_string());
        }
        let thread_pool = if num_threads > 1 {
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .map_err(|error| format!("failed to construct CPU thread pool: {error}"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            thread_count: num_threads,
            use_simd: simd::is_avx2_available(),
            thread_pool,
        })
    }

    pub fn scalar() -> Self {
        Self {
            thread_count: 1,
            use_simd: false,
            thread_pool: None,
        }
    }

    pub fn with_simd(use_simd: bool) -> Self {
        Self {
            use_simd: use_simd && simd::is_avx2_available(),
            ..Self::default()
        }
    }
}

fn matvec_rows_parallel(use_simd: bool, w: &[f32], in_dim: usize, x: &[f32], y: &mut [f32]) {
    y.par_iter_mut().enumerate().for_each(|(o, yo)| {
        let row_offset = o * in_dim;
        let row = &w[row_offset..row_offset + in_dim];
        if use_simd {
            *yo = simd::dot_f32_avx2(row, x);
        } else {
            ramforge_core::compute::matvec_f32_reference(
                row,
                ramforge_core::compute::MatrixShape::new(in_dim, 1),
                x,
                std::slice::from_mut(yo),
            )
            .expect("threaded scalar backend row shape must be valid");
        }
    });
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        if self.use_simd {
            "CPU-SIMD"
        } else {
            "CPU-scalar"
        }
    }

    fn num_threads(&self) -> usize {
        self.thread_count
    }

    fn in_worker_pool<R: Send>(&self, work: impl FnOnce() -> R + Send) -> R {
        if let Some(pool) = &self.thread_pool {
            pool.install(work)
        } else {
            work()
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

        let required_weights = in_dim.checked_mul(out_dim).ok_or_else(|| {
            format!(
                "backend matvec shape size overflow for ggml layout {:?}",
                w_shape
            )
        })?;
        if x.len() != in_dim || y.len() != out_dim || w.len() < required_weights {
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
        if self.thread_count > 1 && out_dim >= 4 {
            self.in_worker_pool(|| {
                matvec_rows_parallel(self.use_simd, w, in_dim, x, y);
            });
        } else if self.use_simd {
            simd::matvec_f32_avx2(w, out_dim, in_dim, x, y);
        } else {
            ramforge_core::compute::matvec_f32_reference(
                w,
                ramforge_core::compute::MatrixShape::new(in_dim, out_dim),
                x,
                y,
            )
            .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32, y: &mut [f32]) {
        ramforge_core::compute::rms_norm_reference(x, weight, eps, y)
            .expect("backend RMSNorm input dimensions must match");
    }

    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        ramforge_core::compute::add_reference(a, b, out)
            .expect("backend add input dimensions must match");
    }

    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        ramforge_core::compute::mul_reference(a, b, out)
            .expect("backend multiply input dimensions must match");
    }

    fn silu(&self, x: &[f32], out: &mut [f32]) {
        ramforge_core::compute::silu_reference(x, out)
            .expect("backend SiLU input dimensions must match");
    }

    fn softmax(&self, x: &mut [f32]) {
        ramforge_core::compute::softmax_reference(x)
            .expect("backend softmax input must be non-empty");
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
            thread_count: 1,
            use_simd: true,
            thread_pool: None,
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

        let multi = CpuBackend::with_threads(2);
        assert_eq!(
            multi
                .thread_pool
                .as_ref()
                .map(|pool| pool.current_num_threads()),
            Some(2)
        );
        multi.matvec(&w, &[2, 4], &x, &mut y_multi).unwrap();

        assert_eq!(y_single, y_multi);
    }

    #[test]
    fn test_explicit_zero_thread_count_is_rejected() {
        assert!(CpuBackend::try_with_threads(0).is_err());
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
