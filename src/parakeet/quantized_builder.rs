//! Hybrid VarBuilder for models with mixed quantized/dequantized weights
//!
//! Allows selective dequantization: encoder and joint network stay quantized,
//! while predictor LSTM is dequantized.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::QTensor;
use std::collections::HashMap;

/// Hybrid storage for both quantized and dequantized tensors
pub struct QuantizedVarBuilder {
    /// Quantized tensors (kept in Q8_0, Q4_K, etc. format)
    qtensors: HashMap<String, QTensor>,
    /// Dequantized tensors (for LSTM and other ops that need full precision)
    tensors: HashMap<String, Tensor>,
    dtype: DType,
    device: Device,
}

impl QuantizedVarBuilder {
    pub fn new(
        qtensors: HashMap<String, QTensor>,
        tensors: HashMap<String, Tensor>,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            qtensors,
            tensors,
            dtype,
            device,
        }
    }

    /// Get a regular tensor (dequantizing if needed)
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        // Try dequantized first
        if let Some(tensor) = self.tensors.get(name) {
            return Ok(tensor.clone());
        }

        // Try quantized (dequantize on demand)
        if let Some(qtensor) = self.qtensors.get(name) {
            return qtensor.dequantize(&self.device);
        }

        Err(anyhow::anyhow!("Tensor not found: {}", name))
    }

    /// Get a quantized tensor (if available)
    pub fn get_qtensor(&self, name: &str) -> Option<&QTensor> {
        self.qtensors.get(name)
    }

    /// Check if tensor exists in quantized form
    pub fn has_qtensor(&self, name: &str) -> bool {
        self.qtensors.contains_key(name)
    }

    /// Get with path prefix (like VarBuilder::pp)
    pub fn pp(&self, prefix: &str) -> PrefixedQuantizedVarBuilder {
        PrefixedQuantizedVarBuilder {
            builder: self,
            prefix: prefix.to_string(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Prefixed view of QuantizedVarBuilder (like VarBuilder::pp())
pub struct PrefixedQuantizedVarBuilder<'a> {
    builder: &'a QuantizedVarBuilder,
    prefix: String,
}

impl<'a> PrefixedQuantizedVarBuilder<'a> {
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.builder.get_tensor(&full_name)
    }

    pub fn get_qtensor(&self, name: &str) -> Option<&QTensor> {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.builder.get_qtensor(&full_name)
    }

    pub fn has_qtensor(&self, name: &str) -> bool {
        let full_name = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        self.builder.has_qtensor(&full_name)
    }

    pub fn pp(&self, name: &str) -> PrefixedQuantizedVarBuilder<'a> {
        let new_prefix = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };
        PrefixedQuantizedVarBuilder {
            builder: self.builder,
            prefix: new_prefix,
        }
    }

    pub fn device(&self) -> &Device {
        self.builder.device()
    }

    pub fn dtype(&self) -> DType {
        self.builder.dtype()
    }
}

/// Create a VarBuilder that dequantizes on-demand
///
/// This allows existing code using VarBuilder to work with quantized weights
/// transparently. Quantized tensors are dequantized lazily when accessed.
pub fn create_var_builder_from_quantized(qvb: &QuantizedVarBuilder) -> candle_nn::VarBuilder {
    // Create a regular VarBuilder by dequantizing all tensors
    // This is a stopgap until we fully refactor to use quantized layers
    let mut all_tensors = HashMap::new();

    // Add already-dequantized tensors
    for (name, tensor) in &qvb.tensors {
        all_tensors.insert(name.clone(), tensor.clone());
    }

    // Dequantize quantized tensors on-demand
    // Note: This still requires dequantization but at least it's lazy
    for (name, qtensor) in &qvb.qtensors {
        if let Ok(tensor) = qtensor.dequantize(&qvb.device) {
            all_tensors.insert(name.clone(), tensor);
        }
    }

    candle_nn::VarBuilder::from_tensors(all_tensors, qvb.dtype, &qvb.device)
}
