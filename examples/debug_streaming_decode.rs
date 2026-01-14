/// Debug streaming model decoding to understand why it produces so few tokens
use anyhow::Result;
use speech::parakeet::{
    get_device, load_parakeet_streaming_tdt_from_local,
    ParakeetFeatureExtractor,
};
use candle_core::{DType, Tensor, D};
use candle_nn;

fn main() -> Result<()> {
    let device = get_device()?;
    println!("Loading streaming model...\n");
    let model = load_parakeet_streaming_tdt_from_local(".cache/parakeet-streaming-tdt", &device)?;

    println!("Model config:");
    println!("  vocab_size: {}", model.config.vocab_size);
    println!("  blank_id: {}", model.config.blank_id);
    println!("  joint_vocab_size: {:?}\n", model.config.joint_vocab_size);

    // Load audio (just first chunk)
    let mut reader = hound::WavReader::open("dots.wav")?;
    let audio_samples: Vec<f32> = reader
        .samples::<i16>()
        .take(16640)  // Just first 1.04s chunk
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    println!("Loaded {} samples\n", audio_samples.len());

    // Extract features (136 mel bins)
    let feat_extractor = ParakeetFeatureExtractor::new(136);
    let features = feat_extractor.extract_to_tensor(&audio_samples, &device)?;
    let features = if !device.is_cpu() {
        features.to_dtype(DType::BF16)?
    } else {
        features
    };

    println!("Feature dims: {:?}\n", features.dims());

    // Run encoder (no cache for debugging)
    println!("Running encoder...");
    let encoder_out = model.encoder.forward(&features, false)?;
    println!("Encoder output dims: {:?}\n", encoder_out.dims());

    // Greedy decode with detailed logging
    let (_, time_steps, _) = encoder_out.dims3()?;
    println!("Decoding {} timesteps...\n", time_steps);

    let mut tokens = Vec::new();
    let mut pred_states = None;
    let mut last_token = model.config.blank_id as u32;

    for t in 0..time_steps.min(5) {  // Just first 5 timesteps for debugging
        println!("=== Timestep {} ===", t);

        let mut inner_step = 0;
        loop {
            inner_step += 1;
            if inner_step > 10 {
                println!("  Hit max inner steps, emitting blank");
                break;
            }

            // Get encoder frame
            let enc_t = encoder_out.narrow(1, t, 1)?;

            // Predictor
            let pred_input = Tensor::new(&[last_token], encoder_out.device())?.unsqueeze(0)?;
            let (pred_out, new_states) = model.predictor.forward(&pred_input, pred_states.as_ref())?;
            pred_states = Some(new_states);

            // Joint network
            let logits = model.joint.forward(&enc_t, &pred_out)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.squeeze(0)?;
            let logits_f32 = logits.to_dtype(DType::F32)?;

            // Get log probs
            let log_probs = candle_nn::ops::log_softmax(&logits_f32, D::Minus1)?;
            let log_probs_vec: Vec<f32> = log_probs.to_vec1()?;

            // Find top tokens
            let mut indexed: Vec<(usize, f32)> = log_probs_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let blank_id = model.config.blank_id;
            let blank_score = log_probs_vec[blank_id];

            println!("  Inner step {}: last_token={}", inner_step, last_token);
            println!("    Blank ({}): {:.3}", blank_id, blank_score);
            println!("    Top 5 non-blank:");
            for (tok_id, score) in indexed.iter().take(10) {
                if *tok_id != blank_id {
                    println!("      {}: {:.3}", tok_id, score);
                }
            }

            // Get best token
            let best_token = indexed[0].0 as u32;
            let best_score = indexed[0].1;

            if best_token == blank_id as u32 {
                println!("  → Blank wins ({:.3}), moving to next timestep\n", best_score);
                break;
            } else {
                println!("  → Token {} wins ({:.3}), continuing\n", best_token, best_score);
                tokens.push(best_token);
                last_token = best_token;
            }
        }
    }

    println!("=== Results ===");
    println!("Tokens: {:?}", tokens);
    if !tokens.is_empty() {
        let text = model.decode_tokens(&tokens)?;
        println!("Text: '{}'", text);
    }

    Ok(())
}
