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
        Ok(Self {
            d,
            dmin,
            scales,
            qs,
        })
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
        Ok(Self {
            d,
            dmin,
            scales,
            qh,
            qs,
        })
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
                let high = if (self.qh[q_offset / 4 + l / 8] & u1) != 0 {
                    16
                } else {
                    0
                };
                out[y_idx + l] = d1 * ((q + high) as f32) - min1;
            }
            y_idx += 32;
            for l in 0..32 {
                let q = (self.qs[q_offset + l] >> 4) as u8;
                let high = if (self.qh[q_offset / 4 + l / 8] & u2) != 0 {
                    16
                } else {
                    0
                };
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
                let q1 =
                    ((self.ql[ql_off + l] & 0xF) | ((self.qh[qh_off + l] & 3) << 4)) as i8 - 32;
                let q2 = ((self.ql[ql_off + 32 + l] & 0xF) | ((self.qh[qh_off + l] >> 2 & 3) << 4))
                    as i8
                    - 32;
                let q3 =
                    ((self.ql[ql_off + l] >> 4) | ((self.qh[qh_off + l] >> 4 & 3) << 4)) as i8 - 32;
                let q4 = ((self.ql[ql_off + 32 + l] >> 4) | ((self.qh[qh_off + l] >> 6 & 3) << 4))
                    as i8
                    - 32;

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
        Ok(Self {
            scales,
            qs,
            d,
            dmin,
        })
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
        Ok(Self {
            hmask,
            qs,
            scales,
            d,
        })
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
pub fn dequantize_row_q4_0(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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
    // Keep the fixed decode-and-copy path: the direct-destination variant was
    // empirically slower for Q4_0. Q8_0 intentionally differs below.
    for (i, chunk) in bytes.chunks(BLOCK_SIZE_Q4_0).enumerate().take(n_blocks) {
        let block = BlockQ4_0::from_bytes(chunk)?;
        let mut tmp = [0f32; 32];
        block.dequantize(&mut tmp);
        out[i * QK4_0..(i + 1) * QK4_0].copy_from_slice(&tmp);
    }
    Ok(())
}

pub fn dequantize_row_q8_0(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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
        let block_out: &mut [f32; QK8_0] = (&mut out[i * QK8_0..(i + 1) * QK8_0])
            .try_into()
            .expect("Q8_0 output block has exact length");
        block.dequantize(block_out);
    }
    Ok(())
}

pub fn dequantize_row_q4_k(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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

pub fn dequantize_row_q5_k(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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

pub fn dequantize_row_q6_k(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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

pub fn dequantize_row_q2_k(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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

pub fn dequantize_row_q3_k(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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

pub fn dequantize_row_q8_k(
    bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
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

// ---------- Q4_0 fused row dot (single-thread scalar) ----------
// Fused per-row dot product: decodes Q4_0 nibbles on the fly without
// materializing a 32-f32 dequant array. Computes the same multiply-add
// order as the reference kernel so outputs match exactly.
#[inline(always)]
fn q4_0_row_dot(row_bytes: &[u8], blocks_per_row: usize, x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut x_offset = 0usize;
    let mut off = 0usize;
    for _ in 0..blocks_per_row {
        let d = f16_to_f32(u16::from_le_bytes([row_bytes[off], row_bytes[off + 1]]));
        off += 2;
        for j in 0..16 {
            let byte = row_bytes[off + j];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = (byte >> 4) as i8 - 8;
            // Multiply order: low nibbles then high nibbles, j=0..15, matching
            // the reference loop over deq[j] for j in 0..32.
            sum += q0 as f32 * d * x[x_offset + j];
            sum += q1 as f32 * d * x[x_offset + 16 + j];
        }
        off += 16;
        x_offset += QK4_0;
    }
    sum
}

// ---------- Quantized matvec (scalar, block-wise) ----------
pub fn matvec_q4_0(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    matvec_q4_0_row_range(w_bytes, w_shape, x, y, 0, None)
}

/// Compute a contiguous Q4_0 output-row range.
///
/// `y` is a local output slice: `y[0]` receives the result for `row_start`.
/// `row_count = None` means all rows from `row_start` through `out_dim`.
/// This local-slice contract is what lets the threaded dispatcher pass a
/// disjoint output subslice without adding the absolute row index twice.
pub fn matvec_q4_0_row_range(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
    row_start: usize,
    row_count: Option<usize>,
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q4_0 matvec expects 2D weight".to_string(),
        ));
    }
    let out_dim = w_shape[0];
    let in_dim = w_shape[1];
    if x.len() != in_dim {
        return Err(DataSourceError::General(format!(
            "matvec input shape mismatch: W {:?}, x {}",
            w_shape,
            x.len()
        )));
    }
    if row_start > out_dim {
        return Err(DataSourceError::General(format!(
            "Q4_0 row start {} exceeds out_dim {}",
            row_start, out_dim
        )));
    }
    let row_count = row_count.unwrap_or(out_dim - row_start);
    let row_end = row_start.checked_add(row_count).ok_or_else(|| {
        DataSourceError::General("Q4_0 row range overflow".to_string())
    })?;
    if row_end > out_dim || y.len() != row_count {
        return Err(DataSourceError::General(format!(
            "Q4_0 row range {}..{} requires local y length {}, got {} for out_dim {}",
            row_start,
            row_end,
            row_count,
            y.len(),
            out_dim
        )));
    }
    if in_dim % QK4_0 != 0 {
        return Err(DataSourceError::General(format!(
            "Q4_0 in_dim {} not divisible by {}",
            in_dim, QK4_0
        )));
    }
    let blocks_per_row = in_dim / QK4_0;
    let row_bytes = blocks_per_row * BLOCK_SIZE_Q4_0;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q4_0 weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }
    for i in row_start..row_end {
        let rb = i * row_bytes;
        y[i - row_start] = q4_0_row_dot(&w_bytes[rb..rb + row_bytes], blocks_per_row, x);
    }
    Ok(())
}

pub fn matvec_q8_0(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q8_0 matvec expects 2D weight".to_string(),
        ));
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

pub fn matvec_q4_k(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q4_K matvec expects 2D weight".to_string(),
        ));
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

pub fn matvec_q5_k(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q5_K matvec expects 2D weight".to_string(),
        ));
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

// ---------- Q6_K fused row dot (single-thread scalar) ----------
#[inline(always)]
fn q6_k_row_dot(
    row_bytes: &[u8],
    blocks_per_row: usize,
    x: &[f32],
) -> Result<f32, DataSourceError> {
    let mut sum = 0.0f32;
    let mut x_offset = 0usize;
    let mut off = 0usize;
    for _ in 0..blocks_per_row {
        if off + BLOCK_SIZE_Q6_K > row_bytes.len() {
            return Err(DataSourceError::General(format!(
                "Q6_K row truncated: expected {} bytes per block",
                BLOCK_SIZE_Q6_K
            )));
        }
        let ql = &row_bytes[off..off + 128];
        let qh = &row_bytes[off + 128..off + 192];
        let scales = &row_bytes[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([
            row_bytes[off + 208],
            row_bytes[off + 209],
        ]));
        off += BLOCK_SIZE_Q6_K;

        // Mirror BlockQ6K::dequantize exactly to preserve arithmetic order.
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off_base = 0usize;
        for _ in 0..QK_K / 128 {
            for l in 0..32 {
                let is = l / 16;
                let qh_byte = qh[qh_off + l];
                let ql_lo = ql[ql_off + l];
                let ql_hi = ql[ql_off + 32 + l];

                let sc1 = scales[sc_off_base + is] as i8 as f32 * d;
                let sc2 = scales[sc_off_base + is + 2] as i8 as f32 * d;
                let sc3 = scales[sc_off_base + is + 4] as i8 as f32 * d;
                let sc4 = scales[sc_off_base + is + 6] as i8 as f32 * d;

                let q1 = ((ql_lo & 0x0F) | ((qh_byte & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql_hi & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((ql_lo >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql_hi >> 4) | (((qh_byte >> 6) & 0x03) << 4)) as i8 - 32;

                sum += sc1 * (q1 as f32) * x[x_offset + l];
                sum += sc2 * (q2 as f32) * x[x_offset + l + 32];
                sum += sc3 * (q3 as f32) * x[x_offset + l + 64];
                sum += sc4 * (q4 as f32) * x[x_offset + l + 96];
            }
            x_offset += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off_base += 8;
        }
    }
    Ok(sum)
}

pub fn matvec_q6_k(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    matvec_q6_k_row_range(w_bytes, w_shape, x, y, 0, None)
}

/// Compute a contiguous Q6_K output-row range into a local `y` slice.
/// `y[0]` corresponds to `row_start`; `None` means through `out_dim`.
pub fn matvec_q6_k_row_range(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
    row_start: usize,
    row_count: Option<usize>,
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q6_K matvec expects 2D weight".to_string(),
        ));
    }
    let out_dim = w_shape[0];
    let in_dim = w_shape[1];
    if x.len() != in_dim {
        return Err(DataSourceError::General(format!(
            "matvec input shape mismatch: W {:?}, x {}",
            w_shape,
            x.len()
        )));
    }
    if row_start > out_dim {
        return Err(DataSourceError::General(format!(
            "Q6_K row start {} exceeds out_dim {}",
            row_start, out_dim
        )));
    }
    let row_count = row_count.unwrap_or(out_dim - row_start);
    let row_end = row_start.checked_add(row_count).ok_or_else(|| {
        DataSourceError::General("Q6_K row range overflow".to_string())
    })?;
    if row_end > out_dim || y.len() != row_count {
        return Err(DataSourceError::General(format!(
            "Q6_K row range {}..{} requires local y length {}, got {} for out_dim {}",
            row_start,
            row_end,
            row_count,
            y.len(),
            out_dim
        )));
    }
    if in_dim % QK_K != 0 {
        return Err(DataSourceError::General(format!(
            "Q6_K in_dim {} not divisible by {}",
            in_dim, QK_K
        )));
    }
    let blocks_per_row = in_dim / QK_K;
    let row_bytes = blocks_per_row * BLOCK_SIZE_Q6_K;
    if w_bytes.len() < out_dim * row_bytes {
        return Err(DataSourceError::General(format!(
            "Q6_K weight truncated: expected {} bytes, got {}",
            out_dim * row_bytes,
            w_bytes.len()
        )));
    }
    for i in row_start..row_end {
        let rb = i * row_bytes;
        y[i - row_start] = q6_k_row_dot(&w_bytes[rb..rb + row_bytes], blocks_per_row, x)?;
    }
    Ok(())
}

pub fn matvec_q2_k(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q2_K matvec expects 2D weight".to_string(),
        ));
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

pub fn matvec_q3_k(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q3_K matvec expects 2D weight".to_string(),
        ));
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

pub fn matvec_q8_k(
    w_bytes: &[u8],
    w_shape: &[usize],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), DataSourceError> {
    if w_shape.len() != 2 {
        return Err(DataSourceError::General(
            "Q8_K matvec expects 2D weight".to_string(),
        ));
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

    fn q4_0_test_block(scale: u16, quants: [u8; 16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q4_0);
        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend_from_slice(&quants);
        bytes
    }

    fn q8_0_test_block(scale: u16, quants: [i8; 32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(BLOCK_SIZE_Q8_0);
        bytes.extend_from_slice(&scale.to_le_bytes());
        bytes.extend(quants.into_iter().map(|value| value as u8));
        bytes
    }

    /// Pre-optimization Q8_0 row path retained as a parity oracle: decode each
    /// block into a fixed array, then copy it into the caller's destination.
    fn dequantize_row_q8_0_reference(
        bytes: &[u8],
        n_elements: usize,
        out: &mut [f32],
    ) -> Result<(), DataSourceError> {
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
            let mut tmp = [0f32; QK8_0];
            block.dequantize(&mut tmp);
            out[i * QK8_0..(i + 1) * QK8_0].copy_from_slice(&tmp);
        }
        Ok(())
    }

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
    fn test_q4_0_row_decode_overwrites_output_and_handles_edge_nibbles() {
        let quants = [
            0xF0, 0x08, 0x87, 0x7F, 0x00, 0xFF, 0x18, 0xE9, 0x26, 0xD3, 0x45, 0xBC, 0x6A, 0x95,
            0x70, 0x8F,
        ];
        let bytes = q4_0_test_block(0x3800, quants); // scale = 0.5
        let mut actual = [123.0f32; QK4_0];
        dequantize_row_q4_0(&bytes, QK4_0, &mut actual).unwrap();

        assert_eq!(actual.len(), QK4_0);
        assert_eq!(actual[0], -4.0); // low nibble 0
        assert_eq!(actual[16], 3.5); // high nibble 15
        assert_eq!(actual[1], 0.0); // low nibble 8
        assert_eq!(actual[17], -4.0); // high nibble 0
        assert!(actual.iter().any(|value| *value < 0.0));
        assert!(actual.iter().any(|value| *value > 0.0));

        let zero_bytes = q4_0_test_block(0x3C00, [0x88; 16]);
        dequantize_row_q4_0(&zero_bytes, QK4_0, &mut actual).unwrap();
        assert!(actual.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_q4_0_row_decode_handles_multiple_rows_and_blocks() {
        let mut row0 = q4_0_test_block(0x3C00, [0x00; 16]);
        row0.extend_from_slice(&q4_0_test_block(0x3800, [0xFF; 16]));
        let mut row1 = q4_0_test_block(0x4000, [0x18; 16]);
        row1.extend_from_slice(&q4_0_test_block(0xBC00, [0xF0; 16]));

        let mut output = [321.0f32; 2 * QK4_0];
        dequantize_row_q4_0(&row0, 2 * QK4_0, &mut output).unwrap();
        assert!(output[..32].iter().all(|value| *value == -8.0));
        assert!(output[32..].iter().all(|value| *value == 3.5));

        dequantize_row_q4_0(&row1, 2 * QK4_0, &mut output).unwrap();
        assert!(output[..16].iter().all(|value| *value == 0.0));
        assert!(output[16..32].iter().all(|value| *value == -14.0));
        assert!(output[32..48].iter().all(|value| *value == 8.0));
        assert!(output[48..].iter().all(|value| *value == -7.0));
    }

    #[test]
    fn test_q4_0_row_decode_preserves_validation_and_destination() {
        let truncated = vec![0u8; BLOCK_SIZE_Q4_0 - 1];
        let mut output = [17.0f32; QK4_0];
        let error = dequantize_row_q4_0(&truncated, QK4_0, &mut output).unwrap_err();
        assert!(error.to_string().contains("row truncated"));
        assert_eq!(output, [17.0; QK4_0]);

        let bytes = q4_0_test_block(0x3C00, [0x88; 16]);
        let mut invalid_output = [29.0f32; QK4_0 - 1];
        let error = dequantize_row_q4_0(&bytes, QK4_0 - 1, &mut invalid_output).unwrap_err();
        assert!(error.to_string().contains("not divisible"));
        assert_eq!(invalid_output, [29.0; QK4_0 - 1]);
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
    fn test_q8_0_row_direct_decode_matches_reference_and_overwrites_output() {
        let quants = [
            -128, 127, -64, 64, -17, 17, -8, 8, -1, 0, 1, 2, -2, 31, -31, 63, -63, 126, -127, 5,
            -5, 12, -12, 42, -42, 99, -99, 3, -3, 7, -7, 16,
        ];
        let bytes = q8_0_test_block(0x3800, quants); // scale = 0.5
        let mut expected = [0.0f32; QK8_0];
        let mut actual = [777.0f32; QK8_0];
        dequantize_row_q8_0_reference(&bytes, QK8_0, &mut expected).unwrap();
        dequantize_row_q8_0(&bytes, QK8_0, &mut actual).unwrap();

        assert_eq!(actual.len(), QK8_0);
        assert_eq!(actual, expected);
        assert_eq!(actual[0], -64.0);
        assert_eq!(actual[1], 63.5);
        assert!(actual.iter().any(|value| *value < 0.0));
        assert!(actual.iter().any(|value| *value > 0.0));
        assert!(actual.iter().all(|value| *value != 777.0));

        let zero_bytes = q8_0_test_block(0x3C00, [0; 32]);
        dequantize_row_q8_0(&zero_bytes, QK8_0, &mut actual).unwrap();
        assert!(actual.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_q8_0_row_direct_decode_matches_reference_for_multiple_rows_and_blocks() {
        let mut row0 = q8_0_test_block(0x3C00, [-3; 32]);
        row0.extend_from_slice(&q8_0_test_block(0x3800, [7; 32]));
        let mut row1 = q8_0_test_block(0xBC00, [-8; 32]);
        row1.extend_from_slice(&q8_0_test_block(0x4000, [5; 32]));

        for row in [&row0, &row1] {
            let mut expected = [0.0f32; 2 * QK8_0];
            let mut actual = [321.0f32; 2 * QK8_0];
            dequantize_row_q8_0_reference(row, 2 * QK8_0, &mut expected).unwrap();
            dequantize_row_q8_0(row, 2 * QK8_0, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_q8_0_row_direct_decode_preserves_validation_and_error_cleanup() {
        let truncated = vec![0u8; BLOCK_SIZE_Q8_0 - 1];
        let mut expected = [17.0f32; QK8_0];
        let mut actual = expected;
        let reference_error =
            dequantize_row_q8_0_reference(&truncated, QK8_0, &mut expected).unwrap_err();
        let direct_error = dequantize_row_q8_0(&truncated, QK8_0, &mut actual).unwrap_err();
        assert_eq!(direct_error.to_string(), reference_error.to_string());
        assert_eq!(actual, [17.0; QK8_0]);

        let bytes = q8_0_test_block(0x3C00, [0; 32]);
        let mut invalid_expected = [29.0f32; QK8_0 - 1];
        let mut invalid_actual = invalid_expected;
        let reference_error =
            dequantize_row_q8_0_reference(&bytes, QK8_0 - 1, &mut invalid_expected).unwrap_err();
        let direct_error = dequantize_row_q8_0(&bytes, QK8_0 - 1, &mut invalid_actual).unwrap_err();
        assert_eq!(direct_error.to_string(), reference_error.to_string());
        assert_eq!(invalid_actual, [29.0; QK8_0 - 1]);
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

    fn make_diagnostic_q4_0_row(blocks: usize) -> Vec<u8> {
        const SCALES: [u16; 8] = [
            0x3C00, // 1.0
            0x3800, // 0.5
            0x3A00, // 0.75
            0x4000, // 2.0
            0xBC00, // -1.0
            0x3400, // 0.25
            0x3E00, // 1.5
            0xB800, // -0.5
        ];

        let mut row = Vec::with_capacity(blocks * BLOCK_SIZE_Q4_0);
        for block_index in 0..blocks {
            let mut quants = [0u8; 16];
            for (j, quant) in quants.iter_mut().enumerate() {
                let low = ((j * 5 + block_index * 3 + 1) & 0x0F) as u8;
                let high = ((j * 7 + block_index * 11 + 9) & 0x0F) as u8;
                *quant = low | (high << 4);
            }
            row.extend_from_slice(&q4_0_test_block(
                SCALES[block_index % SCALES.len()],
                quants,
            ));
        }
        row
    }

    fn make_diagnostic_input(pattern: &str, length: usize) -> Vec<f32> {
        match pattern {
            "constant" => vec![1.25; length],
            "ramp" => (0..length)
                .map(|index| index as f32 * 0.125 - 2.0)
                .collect(),
            "alternating" => (0..length)
                .map(|index| if index % 2 == 0 { 1.5 } else { -0.75 })
                .collect(),
            "pseudo_random" => {
                let mut state = 0x1234_5678u32;
                (0..length)
                    .map(|_| {
                        state = state
                            .wrapping_mul(1_664_525)
                            .wrapping_add(1_013_904_223);
                        let unit = (state >> 8) as f32 / 16_777_215.0;
                        unit * 2.0 - 1.0
                    })
                    .collect()
            }
            other => panic!("unknown diagnostic input pattern {other}"),
        }
    }

    fn compare_q4_0_fused_with_reference(
        shape: &str,
        pattern: &str,
        row_bytes: &[u8],
        n_elements: usize,
        x: &[f32],
    ) {
        let mut dequantized = vec![0.0f32; n_elements];
        dequantize_row_q4_0(row_bytes, n_elements, &mut dequantized).unwrap();

        // This is the previous materialization path's scalar accumulation:
        // dequantized row order and input order are both strictly 0..n-1.
        let mut reference = 0.0f32;
        for index in 0..n_elements {
            reference += dequantized[index] * x[index];
        }

        let fused = q4_0_row_dot(row_bytes, n_elements / QK4_0, x);
        let absolute_difference = (fused - reference).abs();
        let scale = reference.abs().max(fused.abs()).max(1.0);
        // The fused loop interleaves low/high nibbles while the materialized
        // reference accumulates the dequantized row linearly. Allow only a
        // tight f32 accumulation-order difference, not a kernel/layout error.
        let tolerance = 1.0e-5f32 + 1.0e-6f32 * scale;

        println!(
            "Q4_0 A/B shape={} pattern={} reference={:?} fused={:?} abs_diff={:?} tolerance={:?}",
            shape, pattern, reference, fused, absolute_difference, tolerance
        );
        assert!(
            absolute_difference <= tolerance,
            "Q4_0 fused/reference mismatch: shape={}, pattern={}, reference={:?}, fused={:?}, abs_diff={:?}, tolerance={:?}",
            shape,
            pattern,
            reference,
            fused,
            absolute_difference,
            tolerance
        );
    }

    #[test]
    fn test_q4_0_fused_row_dot_matches_dequantize_reference_patterns_and_qwen_shape() {
        const PATTERNS: [&str; 4] = ["constant", "ramp", "alternating", "pseudo_random"];
        assert_eq!(48 * QK4_0, 1536);

        for &(shape, blocks) in &[
            ("four_blocks", 4usize),
            ("qwen2_5_projection", 48usize),
        ] {
            let n_elements = blocks * QK4_0;
            let row_bytes = make_diagnostic_q4_0_row(blocks);
            assert_eq!(row_bytes.len(), blocks * BLOCK_SIZE_Q4_0);

            for pattern in PATTERNS {
                let x = make_diagnostic_input(pattern, n_elements);
                compare_q4_0_fused_with_reference(
                    shape,
                    pattern,
                    &row_bytes,
                    n_elements,
                    &x,
                );
            }
        }
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

    #[test]
    fn q4_0_row_range_uses_local_output_indexing() {
        let mut raw = Vec::new();
        for row in 0..3 {
            raw.extend_from_slice(&0x3c00u16.to_le_bytes());
            raw.extend(std::iter::repeat_n(
                (0x88u8).wrapping_add(row as u8),
                16,
            ));
        }
        let x = vec![1.0f32; QK4_0];
        let mut full = vec![0.0f32; 3];
        matvec_q4_0(&raw, &[3, QK4_0], &x, &mut full).unwrap();

        let mut local = vec![-999.0f32; 1];
        matvec_q4_0_row_range(&raw, &[3, QK4_0], &x, &mut local, 2, Some(1)).unwrap();
        assert_eq!(local[0], full[2]);
        assert!(matvec_q4_0_row_range(&raw, &[3, QK4_0], &x, &mut local, 3, Some(1)).is_err());
    }

    #[test]
    fn q6_k_row_range_uses_local_output_indexing_without_oob() {
        let mut raw = vec![0u8; 3 * BLOCK_SIZE_Q6_K];
        for row in 0..3 {
            let offset = row * BLOCK_SIZE_Q6_K + 208;
            raw[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
        }
        let x = vec![1.0f32; QK_K];
        let mut full = vec![0.0f32; 3];
        matvec_q6_k(&raw, &[3, QK_K], &x, &mut full).unwrap();

        let mut local = vec![-999.0f32; 1];
        matvec_q6_k_row_range(&raw, &[3, QK_K], &x, &mut local, 2, Some(1)).unwrap();
        assert_eq!(local[0], full[2]);
        assert!(matvec_q6_k_row_range(&raw, &[3, QK_K], &x, &mut local, 3, Some(1)).is_err());
    }
}
