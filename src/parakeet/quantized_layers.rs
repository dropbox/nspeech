//! Quantized neural network layers for efficient inference
//!
//! These layers keep weights in quantized format and only dequantize
//! during matrix multiplication operations, reducing memory usage.

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_core::quantized::{QTensor, QMatMul};

/// Quantized Linear layer
///
/// Keeps weights in quantized format (Q8_0, Q4_K, etc.) and uses
/// quantized matrix multiplication for efficient inference.
pub struct QLinear {
    weight: QTensor,
    bias: Option<Tensor>,
}

impl QLinear {
    pub fn new(weight: QTensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Load from a quantized weight tensor and optional bias
    pub fn from_parts(weight: QTensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self::new(weight, bias))
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = QMatMul.forward(x, &self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Quantized Embedding layer
///
/// Keeps embedding weights in quantized format.
pub struct QEmbedding {
    embeddings: QTensor,
    hidden_size: usize,
}

impl QEmbedding {
    pub fn new(embeddings: QTensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &QTensor {
        &self.embeddings
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for QEmbedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        // Dequantize embeddings and index
        // Note: Embeddings typically need full dequantization for indexing
        let embeddings = self.embeddings.dequantize(&indexes.device())?;
        embeddings.index_select(indexes, 0)
    }
}

/// Helper to check if a tensor name should be quantized
pub fn should_quantize_tensor(name: &str) -> bool {
    // Quantize linear layer weights in encoder and joint network
    // Do NOT quantize predictor (LSTM) weights
    if name.contains("predictor.") {
        return false;  // LSTM requires dequantized weights
    }

    // Quantize encoder and joint network linear layers
    name.contains("encoder.") && (
        name.contains(".linear1.weight") ||
        name.contains(".linear2.weight") ||
        name.contains(".q_proj.weight") ||
        name.contains(".k_proj.weight") ||
        name.contains(".v_proj.weight") ||
        name.contains(".o_proj.weight") ||
        name.contains(".relative_k_proj.weight") ||
        name.contains(".pointwise_conv1.weight") ||
        name.contains(".pointwise_conv2.weight")
    ) || (
        name.contains("joint.") && (
            name.contains(".enc_proj.weight") ||
            name.contains(".pred_proj.weight") ||
            name.contains(".output.weight")
        )
    )
}
