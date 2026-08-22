//! GGUF quantized tensor support – Q4_0, Q8_0, Q4_K
//!
//! Implements actual GGML/GGUF block layouts and dequantization semantics.
//!
//! Documented layouts:
//!
//! Q4_0:
//! - block size: 32 values
//! - bytes per block: 18
//!   - 2 bytes: half (fp16) scale `d`
//!   - 16 bytes: 32 x 4-bit quants packed as 2 per byte, unsigned 0..15, mapped to -8..7
//! - dequant: `d * (q - 8)`
//!
//! Q8_0:
//! - block size: 32 values
//! - bytes per block: 34
//!   - 2 bytes: half scale `d`
//!   - 32 bytes: 32 x int8 quants (signed)
//! - dequant: `d * q`
//!
//! Q4_K:
//! - block size: 256 values (super-block), divided into 8 sub-blocks of 32
//! - bytes per block: 144
//!   - 2 bytes: half `d` (super-block scale)
//!   - 2 bytes: half `dmin` (super-block min)
//!   - 12 bytes: `scales[12]` – 8 sub-block scales + 8 mins packed as 6-bit values
//!   - 128 bytes: `qs[128]` – 256 x 4-bit quants
//! - scales packing: `get_scale_min_k4(j, scales, &mut sc, &mut m)`:
//!   if j<4: sc = scales[j] & 63, m = scales[j+4] & 63
//!   else: sc = (scales[j+4] & 0xF) | ((scales[j-4]>>6)<<4), m = (scales[j+4]>>4) | ((scales[j]>>6)<<4)
//! - dequant per sub-block: `d * sc * q - dmin * m` where q is 0..15

#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]

use crate::error::DataSourceError;

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

pub const BLOCK_SIZE_Q4_0: usize = 18;
pub const BLOCK_SIZE_Q8_0: usize = 34;
pub const BLOCK_SIZE_Q4_K: usize = 144;

// ---------- F16 helpers ----------
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 0;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let exp = (127 - 15 - e) as u32;
            (sign << 31) | (exp << 23) | (f << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        let exp = exp + (127 - 15);
        (sign << 31) | (exp << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

fn read_f16_le(bytes: &[u8]) -> Result<f32, DataSourceError> {
    if bytes.len() < 2 {
        return Err(DataSourceError::General("truncated f16".to_string()));
    }
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    Ok(f16_to_f32(bits))
}

// ---------- Q4_0 ----------
#[derive(Debug, Clone)]
pub struct BlockQ4_0 {
    pub d: f32,
    pub qs: [u8; 16], // 32 nibbles
}

impl BlockQ4_0 {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q4_0 {
            return Err(DataSourceError::General(format!(
                "Q4_0 block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q4_0,
                bytes.len()
            )));
        }
        let d = read_f16_le(&bytes[0..2])?;
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&bytes[2..18]);
        Ok(Self { d, qs })
    }

    pub fn dequantize(&self, out: &mut [f32; 32]) {
        for j in 0..16 {
            let byte = self.qs[j];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = (byte >> 4) as i8 - 8;
            out[j] = q0 as f32 * self.d;
            out[16 + j] = q1 as f32 * self.d;
        }
    }
}

// ---------- Q8_0 ----------
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    pub d: f32,
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q8_0 {
            return Err(DataSourceError::General(format!(
                "Q8_0 block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q8_0,
                bytes.len()
            )));
        }
        let d = read_f16_le(&bytes[0..2])?;
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = bytes[2 + i] as i8;
        }
        Ok(Self { d, qs })
    }

    pub fn dequantize(&self, out: &mut [f32; 32]) {
        for i in 0..32 {
            out[i] = self.qs[i] as f32 * self.d;
        }
    }
}

// ---------- Q4_K ----------
#[derive(Debug, Clone)]
pub struct BlockQ4K {
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

impl BlockQ4K {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q4_K {
            return Err(DataSourceError::General(format!(
                "Q4_K block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q4_K,
                bytes.len()
            )));
        }
        let d = read_f16_le(&bytes[0..2])?;
        let dmin = read_f16_le(&bytes[2..4])?;
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qs = [0u8; 128];
        qs.copy_from_slice(&bytes[16..144]);
        Ok(Self { d, dmin, scales, qs })
    }

    fn get_scale_min(&self, j: usize) -> (u8, u8) {
        // j in 0..8 sub-blocks
        if j < 4 {
            let d = self.scales[j] & 63;
            let m = self.scales[j + 4] & 63;
            (d, m)
        } else {
            let d = (self.scales[j + 4] & 0xF) | ((self.scales[j - 4] >> 6) << 4);
            let m = (self.scales[j + 4] >> 4) | ((self.scales[j] >> 6) << 4);
            (d, m)
        }
    }

    pub fn dequantize(&self, out: &mut [f32; 256]) {
        // For each 64-block chunk (256 = 4*64), but Q4_K dequant is per 32 with 2 scales per 64
        // Reference: for j in 0..256 step 64, is = j/32? Actually is increments per 32
        // Implementation from ggml:
        // for j in 0..256 step 64:
        //   get_scale_min(is+0) -> d1,m1, get_scale_min(is+1) -> d2,m2
        //   for l 0..32: y = d*d1*q - dmin*m1
        //   for l 0..32: y = d*d2*q - dmin*m2
        let mut y_idx = 0;
        let mut is = 0usize;
        let mut q_offset = 0usize;

        for _ in 0..QK_K / 64 {
            let (sc0, m0) = self.get_scale_min(is);
            let (sc1, m1) = self.get_scale_min(is + 1);
            let d1 = self.d * (sc0 as f32);
            let min1 = self.dmin * (m0 as f32);
            let d2 = self.d * (sc1 as f32);
            let min2 = self.dmin * (m1 as f32);

            for l in 0..32 {
                let byte = self.qs[q_offset + l];
                let q0 = (byte & 0x0F) as f32;
                out[y_idx + l] = d1 * q0 - min1;
            }
            y_idx += 32;
            for l in 0..32 {
                let byte = self.qs[q_offset + l];
                let q1 = (byte >> 4) as f32;
                out[y_idx + l] = d2 * q1 - min2;
            }
            y_idx += 32;
            q_offset += 32;
            is += 2;
        }
    }
}

// ---------- Row dequantization ----------
pub fn dequantize_row_q4_0(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK4_0 != 0 {
        return Err(DataSourceError::General(format!(
            "Q4_0 row size {} not divisible by block size {}",
            n_elements, QK4_0
        )));
    }
    let n_blocks = n_elements / QK4_0;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q4_0 {
        return Err(DataSourceError::General(format!(
            "Q4_0 row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q4_0,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q4_0).enumerate().take(n_blocks) {
        let block = BlockQ4_0::from_bytes(chunk)?;
        let mut tmp = [0f32; 32];
        block.dequantize(&mut tmp);
        out[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q8_0(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK8_0 != 0 {
        return Err(DataSourceError::General(format!(
            "Q8_0 row size {} not divisible by block size {}",
            n_elements, QK8_0
        )));
    }
    let n_blocks = n_elements / QK8_0;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q8_0 {
        return Err(DataSourceError::General(format!(
            "Q8_0 row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q8_0,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q8_0).enumerate().take(n_blocks) {
        let block = BlockQ8_0::from_bytes(chunk)?;
        let mut tmp = [0f32; 32];
        block.dequantize(&mut tmp);
        out[i * QK8_0..(i + 1) * QK8_0].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q4_k(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q4_K row size {} not divisible by block size {}",
            n_elements, QK_K
        )));
    }
    let n_blocks = n_elements / QK_K;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q4_K {
        return Err(DataSourceError::General(format!(
            "Q4_K row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q4_K,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q4_K).enumerate().take(n_blocks) {
        let block = BlockQ4K::from_bytes(chunk)?;
        let mut tmp = [0f32; 256];
        block.dequantize(&mut tmp);
        out[i * QK_K..(i + 1) * QK_K].copy_from_slice(&tmp);
    }
    Ok(())
}

// ---------- Quantized matvec (scalar, block-wise) ----------
pub fn matvec_q4_0(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    // W shape [out, in], row-major quantized per row
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q4_0 matvec expects 2D weight".to_string()));
    }
    let out_dim = w_shape[0];
    let in_dim = w_shape[1];
    if x.len() != in_dim || y.len() != out_dim {
        return Err(DataSourceError::General(format!(
            "matvec shape mismatch: W {:?}, x {}, y {}",
            w_shape,
            x.len(),
            y.len()
        )));
    }
    if in_dim % QK4_0 != 0 {
        return Err(DataSourceError::General(format!(
            "Q4_0 in_dim {} not divisible by {}",
            in_dim, QK4_0
        )));
    }
    let row_bytes = (in_dim / QK4_0) * BLOCK_SIZE_Q4_0;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q4_0 weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_bytes_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        // Iterate blocks
        for (block_idx, block_bytes) in row_bytes_slice.chunks(BLOCK_SIZE_Q4_0).enumerate() {
            let block = BlockQ4_0::from_bytes(block_bytes)?;
            let mut deq = [0f32; 32];
            block.dequantize(&mut deq);
            let x_offset = block_idx * QK4_0;
            for j in 0..QK4_0 {
                sum += deq[j] * x[x_offset + j];
            }
        }
        y[i] = sum;
    }
    Ok(())
}

pub fn matvec_q8_0(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q8_0 matvec expects 2D weight".to_string()));
    }
    let out_dim = w_shape[0];
    let in_dim = w_shape[1];
    if x.len() != in_dim || y.len() != out_dim {
        return Err(DataSourceError::General(format!(
            "matvec shape mismatch: W {:?}, x {}, y {}",
            w_shape,
            x.len(),
            y.len()
        )));
    }
    if in_dim % QK8_0 != 0 {
        return Err(DataSourceError::General(format!(
            "Q8_0 in_dim {} not divisible by {}",
            in_dim, QK8_0
        )));
    }
    let row_bytes = (in_dim / QK8_0) * BLOCK_SIZE_Q8_0;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q8_0 weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q8_0).enumerate() {
            let block = BlockQ8_0::from_bytes(block_bytes)?;
            let mut deq = [0f32; 32];
            block.dequantize(&mut deq);
            let x_offset = block_idx * QK8_0;
            for j in 0..QK8_0 {
                sum += deq[j] * x[x_offset + j];
            }
        }
        y[i] = sum;
    }
    Ok(())
}

pub fn matvec_q4_k(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q4_K matvec expects 2D weight".to_string()));
    }
    let out_dim = w_shape[0];
    let in_dim = w_shape[1];
    if x.len() != in_dim || y.len() != out_dim {
        return Err(DataSourceError::General(format!(
            "matvec shape mismatch: W {:?}, x {}, y {}",
            w_shape,
            x.len(),
            y.len()
        )));
    }
    if in_dim % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q4_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let row_bytes = (in_dim / QK_K) * BLOCK_SIZE_Q4_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q4_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q4_K).enumerate() {
            let block = BlockQ4K::from_bytes(block_bytes)?;
            let mut deq = [0f32; 256];
            block.dequantize(&mut deq);
            let x_offset = block_idx * QK_K;
            for j in 0..QK_K {
                sum += deq[j] * x[x_offset + j];
            }
        }
        y[i] = sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_block_size() {
        assert_eq!(QK4_0, 32);
        assert_eq!(BLOCK_SIZE_Q4_0, 18);
    }

    #[test]
    fn test_q8_0_block_size() {
        assert_eq!(QK8_0, 32);
        assert_eq!(BLOCK_SIZE_Q8_0, 34);
    }

    #[test]
    fn test_q4_k_block_size() {
        assert_eq!(QK_K, 256);
        assert_eq!(BLOCK_SIZE_Q4_K, 144);
    }

    #[test]
    fn test_q4_0_decode() {
        // Construct a Q4_0 block with d=1.0 (fp16 0x3C00) and all qs = 8 (which dequant to 0)
        // qs 8 means q-8 =0
        let d_fp16: u16 = 0x3C00; // 1.0
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&[0x88; 16]); // each nibble 8
        let block = BlockQ4_0::from_bytes(&bytes).unwrap();
        assert!((block.d - 1.0).abs() < 1e-3);
        let mut out = [0f32; 32];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-5);
        }

        // Test with qs 0 => q-8 = -8, d=1 => -8
        let mut bytes2 = Vec::new();
        bytes2.extend_from_slice(&d_fp16.to_le_bytes());
        bytes2.extend_from_slice(&[0x00; 16]);
        let block2 = BlockQ4_0::from_bytes(&bytes2).unwrap();
        let mut out2 = [0f32; 32];
        block2.dequantize(&mut out2);
        for &v in &out2 {
            assert!((v + 8.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_q8_0_decode() {
        let d_fp16: u16 = 0x3C00; // 1.0
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&[1i8 as u8; 32]); // qs =1
        let block = BlockQ8_0::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 32];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_q4_k_decode() {
        // Construct a synthetic Q4_K block with known values
        // d=1.0, dmin=0.0, scales all 1, qs all 1
        let d_fp16: u16 = 0x3C00; // 1.0
        let dmin_fp16: u16 = 0x0000; // 0.0
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        bytes.extend_from_slice(&[1u8; 12]); // scales =1 for all sub-blocks
        bytes.extend_from_slice(&[0x11; 128]); // qs nibbles =1

        let block = BlockQ4K::from_bytes(&bytes).unwrap();
        assert!((block.d - 1.0).abs() < 1e-3);
        assert!(block.dmin.abs() < 1e-3);

        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        // With scales 1, d=1, dmin=0, q=1 => 1*1*1 -0 =1
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {}", v);
        }
    }

    #[test]
    fn test_q4_k_scale_unpack() {
        // Test get_scale_min packing
        let mut scales = [0u8; 12];
        // Set first 4 scales to 1,2,3,4 and mins to 5,6,7,8
        scales[0] = 1;
        scales[1] = 2;
        scales[2] = 3;
        scales[3] = 4;
        scales[4] = 5;
        scales[5] = 6;
        scales[6] = 7;
        scales[7] = 8;
        // For j>=4, packing is more complex, leave zero for now
        let block = BlockQ4K {
            d: 1.0,
            dmin: 1.0,
            scales,
            qs: [0; 128],
        };
        let (sc0, m0) = block.get_scale_min(0);
        assert_eq!(sc0, 1);
        assert_eq!(m0, 5);
        let (sc1, m1) = block.get_scale_min(1);
        assert_eq!(sc1, 2);
        assert_eq!(m1, 6);
    }

    #[test]
    fn test_truncated_block_rejection() {
        let bytes = vec![0u8; 5];
        assert!(BlockQ4_0::from_bytes(&bytes).is_err());
        assert!(BlockQ8_0::from_bytes(&bytes).is_err());
        assert!(BlockQ4K::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_quantized_matvec_q4_0() {
        // Tiny matrix 2x32, all weights dequant to 1.0, input all 1.0 => output 32 per row
        let d_fp16: u16 = 0x3C00;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        // qs: need q such that d*(q-8)=1 => q-8=1 => q=9 => nibble 9 = 0x9
        row_bytes.extend_from_slice(&[0x99; 16]); // each nibble 9

        // Two rows
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);

        let x = vec![1.0f32; 32];
        let mut y = vec![0.0f32; 2];
        matvec_q4_0(&w_bytes, &[2, 32], &x, &mut y).unwrap();
        assert!((y[0] - 32.0).abs() < 1e-3);
        assert!((y[1] - 32.0).abs() < 1e-3);

        // Compare against reference dequant + f32 matvec
        let mut deq_row = vec![0.0f32; 32];
        dequantize_row_q4_0(&row_bytes, 32, &mut deq_row).unwrap();
        let mut y_ref = vec![0.0f32; 2];
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..32 {
                sum += deq_row[j] * x[j];
            }
            y_ref[i] = sum;
        }
        assert!((y[0] - y_ref[0]).abs() < 1e-3);
    }

    #[test]
    fn test_quantized_matvec_q8_0() {
        let d_fp16: u16 = 0x3C00;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&[1u8; 32]); // qs=1 => 1*1=1

        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);

        let x = vec![2.0f32; 32];
        let mut y = vec![0.0f32; 2];
        matvec_q8_0(&w_bytes, &[2, 32], &x, &mut y).unwrap();
        assert!((y[0] - 64.0).abs() < 1e-3);
    }

    #[test]
    fn test_quantized_matvec_q4_k() {
        // 2x256 matrix, each block dequant to 1.0, input 1.0 => output 256 per row
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&[1u8; 12]);
        row_bytes.extend_from_slice(&[0x11; 128]);

        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);

        let x = vec![1.0f32; 256];
        let mut y = vec![0.0f32; 2];
        matvec_q4_k(&w_bytes, &[2, 256], &x, &mut y).unwrap();
        assert!((y[0] - 256.0).abs() < 1e-3);

        // Reference
        let mut deq = vec![0.0f32; 256];
        dequantize_row_q4_k(&row_bytes, 256, &mut deq).unwrap();
        let mut y_ref = vec![0.0f32; 2];
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..256 {
                sum += deq[j] * x[j];
            }
            y_ref[i] = sum;
        }
        assert!((y[0] - y_ref[0]).abs() < 1e-3);
    }
}
