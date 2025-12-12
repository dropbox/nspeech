use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use parakeet::{FastConformerConfig, ParakeetFastConformerCtc};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let cfg = FastConformerConfig {
        feat_in: 80,
        d_model: 256,
        num_heads: 4,
        ff_mult: 4,
        num_layers: 4,
        conv_kernel_size: 31,
        dropout: 0.1,
        subsampling_channels: 128,
        vocab_size: 40, // [blank] + 39 chars
        blank_id: 0,
    };

    let mut id2token = Vec::new();
    id2token.push("<blank>".to_string());
    for c in b'a'..=b'z' {
        id2token.push((c as char).to_string());
    }
    while id2token.len() < cfg.vocab_size {
        id2token.push(format!("<{}>", id2token.len()));
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = ParakeetFastConformerCtc::new(cfg.clone(), vb, id2token)?;

    // Dummy batch: B=1, T=320 frames, F=80 log-mel features
    let xs = Tensor::zeros((1, 320, cfg.feat_in), DType::F32, &device)?;
    let logits = model.forward(&xs, false)?;
    println!("logits shape: {:?}", logits.shape());

    let transcripts = model.greedy_decode(&logits)?;
    println!("decoded: {:?}", transcripts);

    Ok(())
}
