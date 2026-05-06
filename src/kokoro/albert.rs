//! CustomALBERT — shared-weight transformer for phoneme context encoding.
//!
//! ALBERT uses 128-dim embeddings projected to 768-dim hidden, with all 12
//! layers sharing a single set of attention + FFN weights.
//! Output: [B, T, 768] contextual phoneme embeddings.

use anyhow::Result;
use candle_core::{DType, Tensor, D};
use candle_nn::{self as nn, Embedding, Linear, Module, VarBuilder};

pub struct Albert {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embed_layernorm: LayerNorm,
    embedding_projection: Linear,
    shared_layer: AlbertLayer,
    num_layers: usize,
}

struct AlbertLayer {
    attention: AlbertAttention,
    ffn: Linear,
    ffn_output: Linear,
    attention_norm: LayerNorm,
    output_norm: LayerNorm,
}

struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    num_heads: usize,
    head_dim: usize,
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm {
    fn load(vb: VarBuilder, size: usize) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = diff.broadcast_div(&(var + 1e-12)?.sqrt()?)?;
        norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias).map_err(Into::into)
    }
}

impl Albert {
    pub fn load(vb: VarBuilder, num_layers: usize) -> Result<Self> {
        // Embeddings are 128-dim
        let emb = vb.pp("embeddings");
        let word_embeddings = Embedding::new(
            emb.get((178, 128), "word_embeddings.weight")?,
            128,
        );
        let position_embeddings = Embedding::new(
            emb.get((512, 128), "position_embeddings.weight")?,
            128,
        );
        let token_type_embeddings = Embedding::new(
            emb.get((2, 128), "token_type_embeddings.weight")?,
            128,
        );
        let embed_layernorm = LayerNorm::load(emb.pp("LayerNorm"), 128)?;

        // 128 -> 768 projection
        let embedding_projection = candle_nn::linear(
            128, 768, vb.pp("encoder").pp("embedding_hidden_mapping_in"),
        )?;

        let layer_vb = vb.pp("encoder").pp("albert_layer_groups.0.albert_layers.0");
        let shared_layer = AlbertLayer::load(layer_vb)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_layernorm,
            embedding_projection,
            shared_layer,
            num_layers,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();

        let input_ids = input_ids.contiguous()?;
        let word_emb = self.word_embeddings.forward(&input_ids)?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?.contiguous()?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;
        let token_type_ids = Tensor::zeros((1, seq_len), DType::U32, device)?;
        let type_emb = self.token_type_embeddings.forward(&token_type_ids)?;

        // Sum embeddings (all 128-dim)
        let embeddings = ((word_emb + pos_emb)? + type_emb)?;
        let embeddings = self.embed_layernorm.forward(&embeddings)?;

        // Project 128 -> 768
        let mut hidden = self.embedding_projection.forward(&embeddings)?;

        for _ in 0..self.num_layers {
            hidden = self.shared_layer.forward(&hidden)?;
        }

        Ok(hidden)
    }
}

impl AlbertLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let attn_vb = vb.pp("attention");
        let attention = AlbertAttention::load(attn_vb.clone())?;
        let ffn = candle_nn::linear(768, 2048, vb.pp("ffn"))?;
        let ffn_output = candle_nn::linear(2048, 768, vb.pp("ffn_output"))?;
        let attention_norm = LayerNorm::load(attn_vb.pp("LayerNorm"), 768)?;
        let output_norm = LayerNorm::load(vb.pp("full_layer_layer_norm"), 768)?;
        Ok(Self { attention, ffn, ffn_output, attention_norm, output_norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn_out = self.attention.forward(x)?;
        let x = self.attention_norm.forward(&(x + attn_out)?)?;
        let ffn_out = self.ffn.forward(&x)?.gelu_erf()?;
        let ffn_out = self.ffn_output.forward(&ffn_out)?;
        self.output_norm.forward(&(x + ffn_out)?)
    }
}

impl AlbertAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        let head_dim = 768 / 12;
        Ok(Self {
            query: candle_nn::linear(768, 768, vb.pp("query"))?,
            key: candle_nn::linear(768, 768, vb.pp("key"))?,
            value: candle_nn::linear(768, 768, vb.pp("value"))?,
            dense: candle_nn::linear(768, 768, vb.pp("dense"))?,
            num_heads: 12,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let q = self.query.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = self.key.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = self.value.forward(x)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?;
        let attn = nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?
            .transpose(1, 2)?.contiguous()?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        self.dense.forward(&out).map_err(Into::into)
    }
}
