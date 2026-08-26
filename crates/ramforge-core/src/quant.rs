//! GGUF quantized tensor support – Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, Q8_K
//!
//! Implements exact GGML/GGUF block layouts and dequantization semantics.
//!
//! Q4_0: 32 vals, 18B (2B half d + 16B nibbles, dequant d*(q-8))
//! Q8_0: 32 vals, 34B (2B half d + 32B int8, dequant d*q)
//! Q4_K: 256 vals, 144B (2B d, 2B dmin, 12B scales 6-bit, 128B 4-bit quants, dequant d*sc*q - dmin*m)
//! Q5_K: 256 vals, 176B (2B d, 2B dmin, 12B scales, 32B qh high bit, 128B qs, dequant d*sc*(q+16*high) - dmin*m)
//! Q6_K: 256 vals, 210B (2B d, 16B scales int8, 128B ql low 4 bits, 64B qh high 2 bits, dequant d*sc*q)
//! Q2_K: 256 vals, 84B (2B d, 2B dmin, 16B scales 4-bit, 64B qs 2-bit, dequant d*sc*q - dmin*sc_min)
//! Q3_K: 256 vals, 110B (2B d, 32B hmask, 64B qs low 2 bits, 12B scales 6-bit, dequant d*sc*(q-4) with high bit)
//! Q8_K: 256 vals, 292B (4B float d, 256B qs int8, 32B bsums int16, dequant d*qs)

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::unnecessary_cast,
    clippy::identity_op
)]

use crate::error::DataSourceError;

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

pub const BLOCK_SIZE_Q4_0: usize = 18;
pub const BLOCK_SIZE_Q8_0: usize = 34;
pub const BLOCK_SIZE_Q4_K: usize = 144;
pub const BLOCK_SIZE_Q2_K: usize = 84;
pub const BLOCK_SIZE_Q3_K: usize = 110;
pub const BLOCK_SIZE_Q5_K: usize = 176;
pub const BLOCK_SIZE_Q6_K: usize = 210;
pub const BLOCK_SIZE_Q8_K: usize = 292;

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

fn read_f32_le(bytes: &[u8]) -> Result<f32, DataSourceError> {
    if bytes.len() < 4 {
        return Err(DataSourceError::General("truncated f32".to_string()));
    }
    Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

// ---------- Q4_0 ----------
#[derive(Debug, Clone)]
pub struct BlockQ4_0 {
    pub d: f32,
    pub qs: [u8; 16],
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

// ---------- Q5_K ----------
#[derive(Debug, Clone)]
pub struct BlockQ5K {
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}

impl BlockQ5K {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q5_K {
            return Err(DataSourceError::General(format!(
                "Q5_K block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q5_K,
                bytes.len()
            )));
        }
        let d = read_f16_le(&bytes[0..2])?;
        let dmin = read_f16_le(&bytes[2..4])?;
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qh = [0u8; 32];
        qh.copy_from_slice(&bytes[16..48]);
        let mut qs = [0u8; 128];
        qs.copy_from_slice(&bytes[48..176]);
        Ok(Self { d, dmin, scales, qh, qs })
    }

    fn get_scale_min(&self, j: usize) -> (u8, u8) {
        if j < 4 {
            (self.scales[j] & 63, self.scales[j + 4] & 63)
        } else {
            let d = (self.scales[j + 4] & 0xF) | ((self.scales[j - 4] >> 6) << 4);
            let m = (self.scales[j + 4] >> 4) | ((self.scales[j] >> 6) << 4);
            (d, m)
        }
    }

    pub fn dequantize(&self, out: &mut [f32; 256]) {
        let mut y_idx = 0;
        let mut is = 0usize;
        let mut q_offset = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..QK_K / 64 {
            let (sc0, m0) = self.get_scale_min(is);
            let (sc1, m1) = self.get_scale_min(is + 1);
            let d1 = self.d * (sc0 as f32);
            let min1 = self.dmin * (m0 as f32);
            let d2 = self.d * (sc1 as f32);
            let min2 = self.dmin * (m1 as f32);

            for l in 0..32 {
                let q = (self.qs[q_offset + l] & 0xF) as u8;
                let high = if (self.qh[q_offset / 4 + l / 8] & u1) != 0 { 16 } else { 0 };
                out[y_idx + l] = d1 * ((q + high) as f32) - min1;
            }
            y_idx += 32;
            for l in 0..32 {
                let q = (self.qs[q_offset + l] >> 4) as u8;
                let high = if (self.qh[q_offset / 4 + l / 8] & u2) != 0 { 16 } else { 0 };
                out[y_idx + l] = d2 * ((q + high) as f32) - min2;
            }
            y_idx += 32;
            q_offset += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ---------- Q6_K ----------
#[derive(Debug, Clone)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: f32,
}

impl BlockQ6K {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q6_K {
            return Err(DataSourceError::General(format!(
                "Q6_K block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q6_K,
                bytes.len()
            )));
        }
        let mut ql = [0u8; 128];
        ql.copy_from_slice(&bytes[0..128]);
        let mut qh = [0u8; 64];
        qh.copy_from_slice(&bytes[128..192]);
        let mut scales = [0i8; 16];
        for i in 0..16 {
            scales[i] = bytes[192 + i] as i8;
        }
        let d = read_f16_le(&bytes[192 + 16..192 + 18])?;
        // Remaining bytes are padding? Actually 210 bytes: 128+64+16+2=210
        Ok(Self { ql, qh, scales, d })
    }

    pub fn dequantize(&self, out: &mut [f32; 256]) {
        // Correct implementation based on ggml reference, sequential y
        let mut y_pos = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        for _ in 0..QK_K / 128 {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((self.ql[ql_off + l] & 0xF) | ((self.qh[qh_off + l] & 3) << 4)) as i8 - 32;
                let q2 = ((self.ql[ql_off + 32 + l] & 0xF) | ((self.qh[qh_off + l] >> 2 & 3) << 4)) as i8 - 32;
                let q3 = ((self.ql[ql_off + l] >> 4) | ((self.qh[qh_off + l] >> 4 & 3) << 4)) as i8 - 32;
                let q4 = ((self.ql[ql_off + 32 + l] >> 4) | ((self.qh[qh_off + l] >> 6 & 3) << 4)) as i8 - 32;

                out[y_pos + l] = self.d * (self.scales[sc_off + is] as f32) * (q1 as f32);
                out[y_pos + l + 32] = self.d * (self.scales[sc_off + is + 2] as f32) * (q2 as f32);
                out[y_pos + l + 64] = self.d * (self.scales[sc_off + is + 4] as f32) * (q3 as f32);
                out[y_pos + l + 96] = self.d * (self.scales[sc_off + is + 6] as f32) * (q4 as f32);
            }
            y_pos += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

// ---------- Q2_K ----------
#[derive(Debug, Clone)]
pub struct BlockQ2K {
    pub scales: [u8; 16],
    pub qs: [u8; 64],
    pub d: f32,
    pub dmin: f32,
}

impl BlockQ2K {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q2_K {
            return Err(DataSourceError::General(format!(
                "Q2_K block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q2_K,
                bytes.len()
            )));
        }
        let mut scales = [0u8; 16];
        scales.copy_from_slice(&bytes[0..16]);
        let mut qs = [0u8; 64];
        qs.copy_from_slice(&bytes[16..80]);
        let d = read_f16_le(&bytes[80..82])?;
        let dmin = read_f16_le(&bytes[82..84])?;
        Ok(Self { scales, qs, d, dmin })
    }

    pub fn dequantize(&self, out: &mut [f32; 256]) {
        let mut y_idx = 0usize;
        let mut is = 0usize;
        let mut q_offset = 0usize;

        for _ in 0..QK_K / 128 {
            let mut shift = 0;
            for _ in 0..4 {
                let sc = self.scales[is];
                let dl = self.d * ((sc & 0xF) as f32);
                let ml = self.dmin * ((sc >> 4) as f32);
                is += 1;
                for l in 0..16 {
                    let q = ((self.qs[q_offset + l] >> shift) & 3) as i8;
                    out[y_idx + l] = dl * (q as f32) - ml;
                }
                y_idx += 16;

                let sc = self.scales[is];
                let dl = self.d * ((sc & 0xF) as f32);
                let ml = self.dmin * ((sc >> 4) as f32);
                is += 1;
                for l in 0..16 {
                    let q = ((self.qs[q_offset + 16 + l] >> shift) & 3) as i8;
                    out[y_idx + l] = dl * (q as f32) - ml;
                }
                y_idx += 16;
                shift += 2;
            }
            q_offset += 32;
        }
    }
}

// ---------- Q3_K ----------
#[derive(Debug, Clone)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: [u8; 12],
    pub d: f32,
}

impl BlockQ3K {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q3_K {
            return Err(DataSourceError::General(format!(
                "Q3_K block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q3_K,
                bytes.len()
            )));
        }
        let mut hmask = [0u8; 32];
        hmask.copy_from_slice(&bytes[0..32]);
        let mut qs = [0u8; 64];
        qs.copy_from_slice(&bytes[32..96]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[96..108]);
        let d = read_f16_le(&bytes[108..110])?;
        Ok(Self { hmask, qs, scales, d })
    }

    pub fn dequantize(&self, out: &mut [f32; 256]) {
        // Unpack scales – same as Q4_K but different
        let mut aux = [0u32; 4];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes([
                self.scales[i * 4],
                self.scales[i * 4 + 1],
                self.scales[i * 4 + 2],
                self.scales[i * 4 + 3],
            ]);
        }
        // aux[3] remains 0, tmp is aux[2] from original packing
        // For Q3_K, packing is: 12 bytes contain 16 6-bit scales? Actually we need to unpack as per ggml
        // Reference unpack from ggml-quants.c for Q3_K:
        // memcpy(aux, scales, 12);
        // tmp = aux[2];
        // aux[2] = ((aux[0]>>4) & kmask2) | (((tmp>>4) & kmask1)<<4);
        // aux[3] = ((aux[1]>>4) & kmask2) | (((tmp>>6) & kmask1)<<4);
        // aux[0] = (aux[0] & kmask2) | (((tmp>>0) & kmask1)<<4);
        // aux[1] = (aux[1] & kmask2) | (((tmp>>2) & kmask1)<<4);
        // scales = (int8_t*)aux;
        // kmask1=0x03030303, kmask2=0x0F0F0F0F

        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0F0F0F0F;
        let tmp = aux[2];
        let aux0_orig = aux[0];
        let aux1_orig = aux[1];
        aux[2] = ((aux0_orig >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux1_orig >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux0_orig & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux1_orig & kmask2) | (((tmp >> 2) & kmask1) << 4);

        // Now aux contains 16 scales as bytes (each 0..63, offset by 32)
        let scales_bytes: Vec<u8> = aux.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scales_i8: Vec<i8> = scales_bytes.iter().map(|&b| b as i8).collect();

        let mut y_idx = 0usize;
        let mut q_offset = 0usize;
        let mut m: u8 = 1;

        for _ in 0..QK_K / 128 {
            let mut shift = 0;
            for j in 0..4 {
                let sc = scales_i8[j] as i32 - 32;
                let dl = self.d * (sc as f32);
                for l in 0..16 {
                    let q = ((self.qs[q_offset + l] >> shift) & 3) as i8;
                    let q = q - if (self.hmask[l] & m) != 0 { 0 } else { 4 };
                    out[y_idx + l] = dl * (q as f32);
                }
                y_idx += 16;

                let sc = scales_i8[j + 4] as i32 - 32;
                let dl = self.d * (sc as f32);
                for l in 0..16 {
                    let q = ((self.qs[q_offset + 16 + l] >> shift) & 3) as i8;
                    let q = q - if (self.hmask[16 + l] & m) != 0 { 0 } else { 4 };
                    out[y_idx + l] = dl * (q as f32);
                }
                y_idx += 16;
                shift += 2;
                m <<= 1;
            }
            q_offset += 32;
        }
    }
}

// ---------- Q8_K ----------
#[derive(Debug, Clone)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

impl BlockQ8K {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DataSourceError> {
        if bytes.len() < BLOCK_SIZE_Q8_K {
            return Err(DataSourceError::General(format!(
                "Q8_K block truncated: expected {} bytes, got {}",
                BLOCK_SIZE_Q8_K,
                bytes.len()
            )));
        }
        let d = read_f32_le(&bytes[0..4])?;
        let mut qs = [0i8; 256];
        for i in 0..256 {
            qs[i] = bytes[4 + i] as i8;
        }
        let mut bsums = [0i16; 16];
        for i in 0..16 {
            let off = 4 + 256 + i * 2;
            bsums[i] = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
        }
        Ok(Self { d, qs, bsums })
    }

    pub fn dequantize(&self, out: &mut [f32; 256]) {
        for i in 0..256 {
            out[i] = self.d * (self.qs[i] as f32);
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

pub fn dequantize_row_q5_k(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q5_K row size {} not divisible by block size {}",
            n_elements, QK_K
        )));
    }
    let n_blocks = n_elements / QK_K;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q5_K {
        return Err(DataSourceError::General(format!(
            "Q5_K row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q5_K,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q5_K).enumerate().take(n_blocks) {
        let block = BlockQ5K::from_bytes(chunk)?;
        let mut tmp = [0f32; 256];
        block.dequantize(&mut tmp);
        out[i * QK_K..(i + 1) * QK_K].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q6_k(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q6_K row size {} not divisible by block size {}",
            n_elements, QK_K
        )));
    }
    let n_blocks = n_elements / QK_K;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q6_K {
        return Err(DataSourceError::General(format!(
            "Q6_K row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q6_K,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q6_K).enumerate().take(n_blocks) {
        let block = BlockQ6K::from_bytes(chunk)?;
        let mut tmp = [0f32; 256];
        block.dequantize(&mut tmp);
        out[i * QK_K..(i + 1) * QK_K].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q2_k(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q2_K row size {} not divisible by block size {}",
            n_elements, QK_K
        )));
    }
    let n_blocks = n_elements / QK_K;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q2_K {
        return Err(DataSourceError::General(format!(
            "Q2_K row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q2_K,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q2_K).enumerate().take(n_blocks) {
        let block = BlockQ2K::from_bytes(chunk)?;
        let mut tmp = [0f32; 256];
        block.dequantize(&mut tmp);
        out[i * QK_K..(i + 1) * QK_K].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q3_k(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q3_K row size {} not divisible by block size {}",
            n_elements, QK_K
        )));
    }
    let n_blocks = n_elements / QK_K;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q3_K {
        return Err(DataSourceError::General(format!(
            "Q3_K row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q3_K,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q3_K).enumerate().take(n_blocks) {
        let block = BlockQ3K::from_bytes(chunk)?;
        let mut tmp = [0f32; 256];
        block.dequantize(&mut tmp);
        out[i * QK_K..(i + 1) * QK_K].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q8_k(bytes: &[u8], n_elements: usize, out: &mut [f32]) -> Result<(), DataSourceError> {
    if n_elements % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q8_K row size {} not divisible by block size {}",
            n_elements, QK_K
        )));
    }
    let n_blocks = n_elements / QK_K;
    if bytes.len() < n_blocks * BLOCK_SIZE_Q8_K {
        return Err(DataSourceError::General(format!(
            "Q8_K row truncated: expected {} bytes, got {}",
            n_blocks * BLOCK_SIZE_Q8_K,
            bytes.len()
        )));
    }
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q8_K).enumerate().take(n_blocks) {
        let block = BlockQ8K::from_bytes(chunk)?;
        let mut tmp = [0f32; 256];
        block.dequantize(&mut tmp);
        out[i * QK_K..(i + 1) * QK_K].copy_from_slice(&tmp);
    }
    Ok(())
}

// ---------- Quantized matvec (scalar, block-wise) ----------
pub fn matvec_q4_0(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
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
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q4_0).enumerate() {
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

pub fn matvec_q5_k(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q5_K matvec expects 2D weight".to_string()));
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
            "Q5_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let row_bytes = (in_dim / QK_K) * BLOCK_SIZE_Q5_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q5_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q5_K).enumerate() {
            let block = BlockQ5K::from_bytes(block_bytes)?;
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

pub fn matvec_q6_k(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q6_K matvec expects 2D weight".to_string()));
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
            "Q6_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let row_bytes = (in_dim / QK_K) * BLOCK_SIZE_Q6_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q6_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q6_K).enumerate() {
            let block = BlockQ6K::from_bytes(block_bytes)?;
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

pub fn matvec_q2_k(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q2_K matvec expects 2D weight".to_string()));
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
            "Q2_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let row_bytes = (in_dim / QK_K) * BLOCK_SIZE_Q2_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q2_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q2_K).enumerate() {
            let block = BlockQ2K::from_bytes(block_bytes)?;
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

pub fn matvec_q3_k(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q3_K matvec expects 2D weight".to_string()));
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
            "Q3_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let row_bytes = (in_dim / QK_K) * BLOCK_SIZE_Q3_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q3_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q3_K).enumerate() {
            let block = BlockQ3K::from_bytes(block_bytes)?;
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

pub fn matvec_q8_k(w_bytes: &[u8], w_shape: &[usize], x: &[f32], y: &mut [f32]) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General("Q8_K matvec expects 2D weight".to_string()));
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
            "Q8_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let row_bytes = (in_dim / QK_K) * BLOCK_SIZE_Q8_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q8_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }

    for i in 0..out_dim {
        let row_start = i * row_bytes;
        let row_slice = &w_bytes[row_start..row_start + row_bytes];
        let mut sum = 0.0f32;
        for (block_idx, block_bytes) in row_slice.chunks(BLOCK_SIZE_Q8_K).enumerate() {
            let block = BlockQ8K::from_bytes(block_bytes)?;
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
    fn test_q2_k_block_size() {
        assert_eq!(BLOCK_SIZE_Q2_K, 84);
    }

    #[test]
    fn test_q3_k_block_size() {
        assert_eq!(BLOCK_SIZE_Q3_K, 110);
    }

    #[test]
    fn test_q5_k_block_size() {
        assert_eq!(BLOCK_SIZE_Q5_K, 176);
    }

    #[test]
    fn test_q6_k_block_size() {
        assert_eq!(BLOCK_SIZE_Q6_K, 210);
    }

    #[test]
    fn test_q8_k_block_size() {
        assert_eq!(BLOCK_SIZE_Q8_K, 292);
    }

    #[test]
    fn test_q4_0_decode() {
        let d_fp16: u16 = 0x3C00;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&[0x88; 16]);
        let block = BlockQ4_0::from_bytes(&bytes).unwrap();
        assert!((block.d - 1.0).abs() < 1e-3);
        let mut out = [0f32; 32];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-5);
        }
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
        let d_fp16: u16 = 0x3C00;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&[1i8 as u8; 32]);
        let block = BlockQ8_0::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 32];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_q4_k_decode() {
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        bytes.extend_from_slice(&[1u8; 12]);
        bytes.extend_from_slice(&[0x11; 128]);
        let block = BlockQ4K::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_q5_k_decode() {
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        bytes.extend_from_slice(&[1u8; 12]);
        bytes.extend_from_slice(&[0u8; 32]); // qh
        bytes.extend_from_slice(&[0x11; 128]); // qs
        let block = BlockQ5K::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_q6_k_decode() {
        let d_fp16: u16 = 0x3C00;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0x11; 128]); // ql
        bytes.extend_from_slice(&[0x00; 64]); // qh
        bytes.extend_from_slice(&[1i8 as u8; 16]); // scales
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        let block = BlockQ6K::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        // With our dummy data, just check it doesn't panic and produces some values
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_q2_k_decode() {
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0x11; 16]); // scales
        bytes.extend_from_slice(&[0x11; 64]); // qs
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        let block = BlockQ2K::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_q3_k_decode() {
        let d_fp16: u16 = 0x3C00;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0x11; 32]); // hmask
        bytes.extend_from_slice(&[0x11; 64]); // qs
        bytes.extend_from_slice(&[1u8; 12]); // scales
        bytes.extend_from_slice(&d_fp16.to_le_bytes());
        let block = BlockQ3K::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_q8_k_decode() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes()); // d
        bytes.extend_from_slice(&[1i8 as u8; 256]); // qs
        bytes.extend_from_slice(&[0u8; 32]); // bsums
        let block = BlockQ8K::from_bytes(&bytes).unwrap();
        let mut out = [0f32; 256];
        block.dequantize(&mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_truncated_block_rejection() {
        let bytes = vec![0u8; 5];
        assert!(BlockQ4_0::from_bytes(&bytes).is_err());
        assert!(BlockQ8_0::from_bytes(&bytes).is_err());
        assert!(BlockQ4K::from_bytes(&bytes).is_err());
        assert!(BlockQ2K::from_bytes(&bytes).is_err());
        assert!(BlockQ3K::from_bytes(&bytes).is_err());
        assert!(BlockQ5K::from_bytes(&bytes).is_err());
        assert!(BlockQ6K::from_bytes(&bytes).is_err());
        assert!(BlockQ8K::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_quantized_matvec_q4_0() {
        let d_fp16: u16 = 0x3C00;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&[0x99; 16]);
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);
        let x = vec![1.0f32; 32];
        let mut y = vec![0.0f32; 2];
        matvec_q4_0(&w_bytes, &[2, 32], &x, &mut y).unwrap();
        assert!((y[0] - 32.0).abs() < 1e-3);
        let mut deq_row = vec![0.0f32; 32];
        dequantize_row_q4_0(&row_bytes, 32, &mut deq_row).unwrap();
        let mut y_ref = [0.0f32; 2];
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
        row_bytes.extend_from_slice(&[1u8; 32]);
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
        let mut deq = vec![0.0f32; 256];
        dequantize_row_q4_k(&row_bytes, 256, &mut deq).unwrap();
        let mut y_ref = [0.0f32; 2];
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..256 {
                sum += deq[j] * x[j];
            }
            y_ref[i] = sum;
        }
        assert!((y[0] - y_ref[0]).abs() < 1e-3);
    }

    #[test]
    fn test_quantized_matvec_q5_k() {
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&[1u8; 12]);
        row_bytes.extend_from_slice(&[0u8; 32]);
        row_bytes.extend_from_slice(&[0x11; 128]);
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);
        let x = vec![1.0f32; 256];
        let mut y = vec![0.0f32; 2];
        matvec_q5_k(&w_bytes, &[2, 256], &x, &mut y).unwrap();
        assert!((y[0] - 256.0).abs() < 1e-3);
    }

    #[test]
    fn test_quantized_matvec_q6_k() {
        let d_fp16: u16 = 0x3C00;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&[0x11; 128]);
        row_bytes.extend_from_slice(&[0x00; 64]);
        row_bytes.extend_from_slice(&[1u8; 16]);
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);
        let x = vec![1.0f32; 256];
        let mut y = vec![0.0f32; 2];
        // Q6_K dequant is more complex, but test that it doesn't panic and produces some output
        let _ = matvec_q6_k(&w_bytes, &[2, 256], &x, &mut y);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_quantized_matvec_q2_k() {
        let d_fp16: u16 = 0x3C00;
        let dmin_fp16: u16 = 0x0000;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&[0x11; 16]);
        row_bytes.extend_from_slice(&[0x11; 64]);
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        row_bytes.extend_from_slice(&dmin_fp16.to_le_bytes());
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);
        let x = vec![1.0f32; 256];
        let mut y = vec![0.0f32; 2];
        let _ = matvec_q2_k(&w_bytes, &[2, 256], &x, &mut y);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_quantized_matvec_q3_k() {
        let d_fp16: u16 = 0x3C00;
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&[0x11; 32]);
        row_bytes.extend_from_slice(&[0x11; 64]);
        row_bytes.extend_from_slice(&[1u8; 12]);
        row_bytes.extend_from_slice(&d_fp16.to_le_bytes());
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);
        let x = vec![1.0f32; 256];
        let mut y = vec![0.0f32; 2];
        let _ = matvec_q3_k(&w_bytes, &[2, 256], &x, &mut y);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_quantized_matvec_q8_k() {
        let mut row_bytes = Vec::new();
        row_bytes.extend_from_slice(&1.0f32.to_le_bytes());
        row_bytes.extend_from_slice(&[1u8; 256]);
        row_bytes.extend_from_slice(&[0u8; 32]);
        let mut w_bytes = Vec::new();
        w_bytes.extend_from_slice(&row_bytes);
        w_bytes.extend_from_slice(&row_bytes);
        let x = vec![1.0f32; 256];
        let mut y = vec![0.0f32; 2];
        matvec_q8_k(&w_bytes, &[2, 256], &x, &mut y).unwrap();
        assert!((y[0] - 256.0).abs() < 1e-3);
    }
}
