/// HQQ (Half-Quadratic Quantization) implementation for Rust/Candle
///
/// **Status**: Research/proof-of-concept implementation. Not integrated with main lib.rs
/// or production inference pipeline (QMatMul). See HQQ_QUANTIZATION.md for details.
///
/// HQQ is an advanced quantization method that optimizes scales and zero-points
/// to minimize quantization error using a half-quadratic approach.
///
/// This module is standalone and used only by the `quantize_hqq` example tool.
/// It demonstrates superior quantization accuracy but requires custom inference
/// kernels for practical use.
///
/// References:
/// - HQQ paper: https://arxiv.org/abs/2309.15531
/// - Optimal Brain Quantization and related works

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// HQQ quantization parameters
#[derive(Debug, Clone)]
pub struct HqqConfig {
    /// Number of bits for quantization (2, 3, 4, 8)
    pub nbits: u8,
    /// Group size for group-wise quantization (e.g., 128)
    pub group_size: usize,
    /// Whether to use symmetric quantization (no zero-point)
    pub symmetric: bool,
    /// Number of optimization iterations for scale/zero-point
    pub optimize_iters: usize,
}

impl Default for HqqConfig {
    fn default() -> Self {
        Self {
            nbits: 4,
            group_size: 128,
            symmetric: false,
            optimize_iters: 20,
        }
    }
}

/// Quantized tensor in HQQ format
#[derive(Debug, Clone)]
pub struct HqqTensor {
    /// Quantized weights as u8 (packed bits)
    pub qweight: Vec<u8>,
    /// Scales per group [num_groups]
    pub scales: Vec<f32>,
    /// Zero-points per group [num_groups] (optional, for asymmetric quantization)
    pub zeros: Option<Vec<f32>>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Config used
    pub config: HqqConfig,
}

impl HqqTensor {
    /// Quantize a tensor using HQQ
    pub fn quantize(tensor: &Tensor, config: HqqConfig) -> Result<Self> {
        // Only support 2D tensors (weight matrices)
        if tensor.rank() != 2 {
            anyhow::bail!("HQQ only supports 2D tensors, got rank {}", tensor.rank());
        }

        let shape = tensor.dims();
        let (rows, cols) = (shape[0], shape[1]);

        // Convert to 2D Vec<Vec<f32>>
        let weight_matrix = tensor.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // Quantize row-by-row with grouping
        let num_groups = (cols + config.group_size - 1) / config.group_size;
        let mut scales = Vec::with_capacity(rows * num_groups);
        let mut zeros = if config.symmetric {
            None
        } else {
            Some(Vec::with_capacity(rows * num_groups))
        };

        let max_q = (1 << config.nbits) - 1; // e.g., 15 for 4-bit
        let mut all_qweights = Vec::new();

        for row in 0..rows {
            for group_idx in 0..num_groups {
                let start = group_idx * config.group_size;
                let end = (start + config.group_size).min(cols);
                let group = &weight_matrix[row][start..end];

                // Compute optimal scale and zero-point for this group
                let (scale, zero) = if config.symmetric {
                    optimize_symmetric(group, max_q, config.optimize_iters)
                } else {
                    optimize_asymmetric(group, max_q, config.optimize_iters)
                };

                scales.push(scale);
                if let Some(ref mut z) = zeros {
                    z.push(zero);
                }

                // Quantize group
                for &w in group {
                    let q = quantize_value(w, scale, zero, max_q);
                    all_qweights.push(q);
                }
            }
        }

        // Pack quantized weights based on nbits
        let qweight = pack_weights(&all_qweights, config.nbits)?;

        Ok(Self {
            qweight,
            scales,
            zeros,
            shape: vec![rows, cols],
            config,
        })
    }

    /// Dequantize back to FP32 tensor
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let num_groups = (cols + self.config.group_size - 1) / self.config.group_size;

        // Unpack weights
        let qweights = unpack_weights(&self.qweight, self.config.nbits, rows * cols)?;

        // Dequantize
        let mut data = Vec::with_capacity(rows * cols);
        let mut qw_idx = 0;

        for row in 0..rows {
            for group_idx in 0..num_groups {
                let scale_idx = row * num_groups + group_idx;
                let scale = self.scales[scale_idx];
                let zero = self.zeros.as_ref().map(|z| z[scale_idx]).unwrap_or(0.0);

                let start = group_idx * self.config.group_size;
                let end = (start + self.config.group_size).min(cols);
                let group_size = end - start;

                for _ in 0..group_size {
                    let qw = qweights[qw_idx] as f32;
                    let w = dequantize_value(qw, scale, zero);
                    data.push(w);
                    qw_idx += 1;
                }
            }
        }

        Ok(Tensor::from_vec(data, (rows, cols), device)?)
    }
}

/// Optimize scale for symmetric quantization (zero-point = 0)
/// Uses grid search over scale values to minimize MSE
fn optimize_symmetric(weights: &[f32], max_q: u8, iters: usize) -> (f32, f32) {
    let max_w = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);

    if max_w == 0.0 {
        return (1.0, 0.0);
    }

    let mut best_scale = max_w / (max_q as f32 / 2.0);
    let mut best_error = f32::INFINITY;

    // Grid search over scale
    for i in 0..iters {
        let scale_factor = 0.5 + (i as f32 / iters as f32) * 1.5; // Search from 0.5x to 2.0x
        let scale = (max_w / (max_q as f32 / 2.0)) * scale_factor;

        if scale == 0.0 {
            continue;
        }

        let mut error = 0.0f32;
        for &w in weights {
            let q = quantize_value(w, scale, 0.0, max_q);
            let w_hat = dequantize_value(q as f32, scale, 0.0);
            error += (w - w_hat).powi(2);
        }

        if error < best_error {
            best_error = error;
            best_scale = scale;
        }
    }

    (best_scale, 0.0)
}

/// Optimize scale and zero-point for asymmetric quantization
/// Uses grid search to minimize MSE
fn optimize_asymmetric(weights: &[f32], max_q: u8, iters: usize) -> (f32, f32) {
    let min_w = weights.iter().copied().fold(f32::INFINITY, f32::min);
    let max_w = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max_w - min_w).abs() < 1e-10 {
        return (1.0, min_w);
    }

    // Initial scale and zero-point
    let mut best_scale = (max_w - min_w) / max_q as f32;
    let mut best_zero = min_w;
    let mut best_error = f32::INFINITY;

    // Grid search
    for i in 0..iters {
        for j in 0..iters {
            let scale_factor = 0.8 + (i as f32 / iters as f32) * 0.4; // 0.8x to 1.2x
            let zero_factor = j as f32 / iters as f32; // Interpolate between min and adjusted min

            let scale = (max_w - min_w) / max_q as f32 * scale_factor;
            let zero = min_w + (max_w - min_w) * 0.1 * (zero_factor - 0.5);

            if scale == 0.0 {
                continue;
            }

            let mut error = 0.0f32;
            for &w in weights {
                let q = quantize_value(w, scale, zero, max_q);
                let w_hat = dequantize_value(q as f32, scale, zero);
                error += (w - w_hat).powi(2);
            }

            if error < best_error {
                best_error = error;
                best_scale = scale;
                best_zero = zero;
            }
        }
    }

    (best_scale, best_zero)
}

/// Quantize a single value
fn quantize_value(w: f32, scale: f32, zero: f32, max_q: u8) -> u8 {
    let q = ((w - zero) / scale).round();
    q.clamp(0.0, max_q as f32) as u8
}

/// Dequantize a single value
fn dequantize_value(q: f32, scale: f32, zero: f32) -> f32 {
    q * scale + zero
}

/// Pack quantized weights into bytes based on bit width
fn pack_weights(weights: &[u8], nbits: u8) -> Result<Vec<u8>> {
    match nbits {
        2 => pack_2bit(weights),
        3 => pack_3bit(weights),
        4 => pack_4bit(weights),
        8 => Ok(weights.to_vec()),
        _ => anyhow::bail!("Unsupported nbits: {}", nbits),
    }
}

/// Pack 4-bit values (2 values per byte)
fn pack_4bit(weights: &[u8]) -> Result<Vec<u8>> {
    let mut packed = Vec::with_capacity((weights.len() + 1) / 2);

    for chunk in weights.chunks(2) {
        let low = chunk[0] & 0x0F;
        let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
        packed.push((high << 4) | low);
    }

    Ok(packed)
}

/// Pack 2-bit values (4 values per byte)
fn pack_2bit(weights: &[u8]) -> Result<Vec<u8>> {
    let mut packed = Vec::with_capacity((weights.len() + 3) / 4);

    for chunk in weights.chunks(4) {
        let mut byte = 0u8;
        for (i, &w) in chunk.iter().enumerate() {
            byte |= (w & 0x03) << (i * 2);
        }
        packed.push(byte);
    }

    Ok(packed)
}

/// Pack 3-bit values
fn pack_3bit(weights: &[u8]) -> Result<Vec<u8>> {
    // 3-bit packing: 8 values = 24 bits = 3 bytes
    let mut packed = Vec::with_capacity((weights.len() * 3 + 7) / 8);

    for chunk in weights.chunks(8) {
        let mut bits = 0u32;
        for (i, &w) in chunk.iter().enumerate() {
            bits |= ((w & 0x07) as u32) << (i * 3);
        }
        packed.push((bits & 0xFF) as u8);
        packed.push(((bits >> 8) & 0xFF) as u8);
        packed.push(((bits >> 16) & 0xFF) as u8);
    }

    Ok(packed)
}

/// Unpack quantized weights from bytes
fn unpack_weights(packed: &[u8], nbits: u8, total_count: usize) -> Result<Vec<u8>> {
    match nbits {
        2 => unpack_2bit(packed, total_count),
        3 => unpack_3bit(packed, total_count),
        4 => unpack_4bit(packed, total_count),
        8 => Ok(packed.to_vec()),
        _ => anyhow::bail!("Unsupported nbits: {}", nbits),
    }
}

/// Unpack 4-bit values
fn unpack_4bit(packed: &[u8], total_count: usize) -> Result<Vec<u8>> {
    let mut unpacked = Vec::with_capacity(total_count);

    for &byte in packed {
        unpacked.push(byte & 0x0F);
        if unpacked.len() < total_count {
            unpacked.push((byte >> 4) & 0x0F);
        }
    }

    unpacked.truncate(total_count);
    Ok(unpacked)
}

/// Unpack 2-bit values
fn unpack_2bit(packed: &[u8], total_count: usize) -> Result<Vec<u8>> {
    let mut unpacked = Vec::with_capacity(total_count);

    for &byte in packed {
        for i in 0..4 {
            if unpacked.len() >= total_count {
                break;
            }
            unpacked.push((byte >> (i * 2)) & 0x03);
        }
    }

    unpacked.truncate(total_count);
    Ok(unpacked)
}

/// Unpack 3-bit values
fn unpack_3bit(packed: &[u8], total_count: usize) -> Result<Vec<u8>> {
    let mut unpacked = Vec::with_capacity(total_count);

    for chunk in packed.chunks(3) {
        if chunk.len() < 3 {
            break;
        }

        let bits = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);

        for i in 0..8 {
            if unpacked.len() >= total_count {
                break;
            }
            unpacked.push(((bits >> (i * 3)) & 0x07) as u8);
        }
    }

    unpacked.truncate(total_count);
    Ok(unpacked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_4bit() {
        let weights = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let packed = pack_4bit(&weights).unwrap();
        let unpacked = unpack_4bit(&packed, weights.len()).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_pack_unpack_2bit() {
        let weights = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let packed = pack_2bit(&weights).unwrap();
        let unpacked = unpack_2bit(&packed, weights.len()).unwrap();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_quantize_dequantize() {
        let device = Device::Cpu;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data.clone(), (2, 4), &device).unwrap();

        let config = HqqConfig {
            nbits: 4,
            group_size: 4,
            symmetric: false,
            optimize_iters: 10,
        };

        let hqq = HqqTensor::quantize(&tensor, config).unwrap();
        let dequant = hqq.dequantize(&device).unwrap();
        let result = dequant.to_vec2::<f32>().unwrap();

        // Check that dequantized values are close to original
        for i in 0..2 {
            for j in 0..4 {
                let original = data[i * 4 + j];
                let reconstructed = result[i][j];
                let error = (original - reconstructed).abs() / original.abs().max(1.0);
                assert!(error < 0.2, "Error too large: {} vs {}", original, reconstructed);
            }
        }
    }
}
