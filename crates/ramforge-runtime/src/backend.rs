//! Compute backend abstraction
//!
//! Milestone 3 only has CPU backend, but designed to allow future GPU backends.

use std::fmt::Debug;

pub trait ComputeBackend: Debug + Send + Sync {
    fn name(&self) -> &'static str;

    /// Matrix-vector multiplication: y = W * x
    /// W is [out_dim, in_dim] in row-major (or column-major handled by caller)
    /// x is [in_dim], y is [out_dim]
    fn matvec(&self, w: &[f32], w_shape: &[usize], x: &[f32], y: &mut [f32]);

    /// RMSNorm: y = x / sqrt(mean(x^2)+eps) * weight
    fn rmsnorm(&self, x: &[f32], weight: &[f32], eps: f32, y: &mut [f32]);

    /// Elementwise addition
    fn add(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// Elementwise multiplication
    fn mul(&self, a: &[f32], b: &[f32], out: &mut [f32]);

    /// SiLU activation: y = x * sigmoid(x)
    fn silu(&self, x: &[f32], out: &mut [f32]);

    /// Softmax in-place
    fn softmax(&self, x: &mut [f32]);
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU"
    }

    #[allow(clippy::needless_range_loop)]
    fn matvec(&self, w: &[f32], w_shape: &[usize], x: &[f32], y: &mut [f32]) {
        // w_shape is [out, in] or [in, out] depending on storage.
        // We assume w is stored as [out, in] row-major: w[out * in + in]
        // If shape is [in, out], we transpose logic: we detect by comparing dimensions to x len
        // For simplicity, we assume row-major [out, in]
        // But we also handle case where w_shape is [in, out] by checking
        if w_shape.len() != 2 {
            // Fallback to naive if not 2D
            for (i, yi) in y.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (j, xj) in x.iter().enumerate() {
                    if i * x.len() + j < w.len() {
                        sum += w[i * x.len() + j] * xj;
                    }
                }
                *yi = sum;
            }
            return;
        }

        let out_dim = w_shape[0];
        let in_dim = w_shape[1];

        // If in_dim == x.len(), then w is [out, in] row-major
        // If out_dim == x.len(), then w is [in, out] and we need transpose: y[j] = sum_i W[i,j]*x[i] ??? Let's handle
        if in_dim == x.len() && out_dim == y.len() {
            // Row-major [out, in]
            for i in 0..out_dim {
                let mut sum = 0.0;
                let row_offset = i * in_dim;
                for j in 0..in_dim {
                    sum += w[row_offset + j] * x[j];
                }
                y[i] = sum;
            }
        } else if out_dim == x.len() && in_dim == y.len() {
            // W is [in, out] row-major, need to compute y = W^T * x? Actually if W is [in, out], then W^T is [out, in]
            // So y[j] = sum_i W[i*out + j] * x[i]
            for j in 0..in_dim {
                let mut sum = 0.0;
                for i in 0..out_dim {
                    sum += w[i * in_dim + j] * x[i];
                }
                y[j] = sum;
            }
        } else {
            // Fallback: try to handle as column-major [in, out] where in is contiguous
            // Assume w is [in, out] column-major: first in elements are col0, etc.
            // Then y[j] = sum_i W[i + j*in] * x[i]
            if w.len() == out_dim * in_dim {
                // Try both interpretations and pick one that matches
                // We'll assume column-major [in, out] if in_dim == x.len() and out_dim == y.len() is false, but we already handled row-major
                // For column-major [in, out]: w is [in, out] with in contiguous: offset = j*in + i
                if in_dim == x.len() && out_dim == y.len() {
                    // This case already handled as row-major, but column-major would be same size, need to decide
                    // We'll use row-major as default
                    for i in 0..out_dim {
                        let mut sum = 0.0;
                        for j in 0..in_dim {
                            sum += w[j + i * in_dim] * x[j];
                        }
                        y[i] = sum;
                    }
                } else {
                    // Generic fallback
                    for i in 0..y.len() {
                        y[i] = 0.0;
                        for j in 0..x.len() {
                            if j < w_shape[0] && i < w_shape[1] {
                                // Assume w is [in, out] row-major
                                y[i] += w[j * w_shape[1] + i] * x[j];
                            }
                        }
                    }
                }
            } else {
                // Zero out
                for yi in y.iter_mut() {
                    *yi = 0.0;
                }
            }
        }
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
        let backend = CpuBackend::new();
        // W = [[1,2],[3,4]] row-major [2,2], x=[1,1] => y=[3,7]
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        backend.matvec(&w, &[2, 2], &x, &mut y);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_rmsnorm() {
        let backend = CpuBackend::new();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 4];
        backend.rmsnorm(&x, &w, 1e-5, &mut y);
        // rms = sqrt(1+eps) ~1, so y ~ x
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
