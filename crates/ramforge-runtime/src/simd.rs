//! SIMD optimization layer – AVX2 for x86_64, scalar fallback

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "x86_64")]
pub fn is_avx2_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_avx2_available() -> bool {
    false
}

/// Scalar dot product
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// AVX2 dot product – unsafe, isolated
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_inner(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    let len = a.len();
    // Process 8 at a time
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(av, bv, sum);
        i += 8;
    }
    // Horizontal sum
    let mut result = [0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total: f32 = result.iter().sum();
    // Remainder
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
pub fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    if is_avx2_available() && a.len() >= 8 {
        unsafe { dot_f32_avx2_inner(a, b) }
    } else {
        dot_f32_scalar(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_scalar(a, b)
}

/// Matvec with AVX2: y = W * x, W row-major [out, in]
#[cfg(target_arch = "x86_64")]
pub fn matvec_f32_avx2(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
    if is_avx2_available() && in_dim >= 8 {
        for i in 0..out_dim {
            let row_offset = i * in_dim;
            let row = &w[row_offset..row_offset + in_dim];
            unsafe {
                y[i] = dot_f32_avx2_inner(row, x);
            }
        }
    } else {
        // scalar fallback
        for i in 0..out_dim {
            let mut sum = 0.0;
            let row_offset = i * in_dim;
            for j in 0..in_dim {
                sum += w[row_offset + j] * x[j];
            }
            y[i] = sum;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn matvec_f32_avx2(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
    for i in 0..out_dim {
        let mut sum = 0.0;
        let row_offset = i * in_dim;
        for j in 0..in_dim {
            sum += w[row_offset + j] * x[j];
        }
        y[i] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_scalar_vs_simd() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let scalar = dot_f32_scalar(&a, &b);
        let simd = dot_f32_avx2(&a, &b);
        assert!((scalar - simd).abs() < 1e-4, "scalar {} vs simd {}", scalar, simd);
    }

    #[test]
    fn test_matvec_scalar_vs_simd() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let x = vec![1.0, 1.0, 1.0];
        let mut y_scalar = [0.0; 2];
        let mut y_simd = [0.0; 2];

        // scalar
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += w[i * 3 + j] * x[j];
            }
            y_scalar[i] = sum;
        }

        matvec_f32_avx2(&w, 2, 3, &x, &mut y_simd);

        for i in 0..2 {
            assert!((y_scalar[i] - y_simd[i]).abs() < 1e-4);
        }
    }
}
