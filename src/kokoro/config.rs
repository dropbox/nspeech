use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct KokoroConfig {
    pub istftnet: ISTFTNetConfig,
    pub dim_in: usize,
    pub dropout: f64,
    pub hidden_dim: usize,
    pub max_conv_dim: usize,
    pub max_dur: usize,
    pub multispeaker: bool,
    pub n_layer: usize,
    pub n_mels: usize,
    pub n_token: usize,
    pub style_dim: usize,
    pub text_encoder_kernel_size: usize,
    pub plbert: PLBertConfig,
    pub vocab: HashMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ISTFTNetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PLBertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub dropout: f64,
}

impl KokoroConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn total_upsample_factor(&self) -> usize {
        self.istftnet.upsample_rates.iter().product::<usize>() * self.istftnet.gen_istft_hop_size
    }
}
