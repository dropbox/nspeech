/// Debug quantized model output to understand why transcription is empty
///
/// Usage:
///   PARAKEET_DEVICE=cpu cargo run --example debug_quantized_output --release -- dots.wav

use anyhow::Result;
use candle_core::D;
use parakeet::{get_device, load_parakeet_ctc_from_gguf_local, load_wav_as_features};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];
    let device = get_device()?;

    println!("Loading quantized model...");
    let model = load_parakeet_ctc_from_gguf_local("hf_parakeet", &device)?;

    println!("Processing audio...");
    let features = load_wav_as_features(audio_path, model.cfg.feat_in, &device)?;
    let (batch, frames, _) = features.dims3()?;
    println!("  Input: batch={}, frames={}, feat_dim={}\n", batch, frames, model.cfg.feat_in);

    println!("Running inference...");
    let logits = model.forward(&features, false)?;
    let (b, t, v) = logits.dims3()?;
    println!("  Logits: [{}, {}, {}]\n", b, t, v);

    // Analyze logits
    println!("Analyzing logits...");
    let logits_flat = logits.flatten_all()?.to_vec1::<f32>()?;
    let mean = logits_flat.iter().sum::<f32>() / logits_flat.len() as f32;
    let min = logits_flat.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Logits stats: mean={:.4}, min={:.4}, max={:.4}", mean, min, max);

    // Get predictions
    let pred_ids = logits.argmax(D::Minus1)?;
    let pred_ids = pred_ids.to_vec2::<u32>()?;

    // Analyze predictions
    let blank_id = model.cfg.blank_id as u32;
    let mut blank_count = 0;
    let mut non_blank_count = 0;
    let mut token_counts = std::collections::HashMap::new();

    println!("\nPredictions:");
    for tidx in 0..t.min(20) {
        let token = pred_ids[0][tidx];
        *token_counts.entry(token).or_insert(0) += 1;

        if token == blank_id {
            blank_count += 1;
        } else {
            non_blank_count += 1;
        }

        let marker = if token == blank_id { " (blank)" } else { "" };
        println!("  Frame {:3}: token {:4}{}", tidx, token, marker);
    }

    println!("\nToken statistics (all {} frames):", t);
    for tidx in 0..t {
        let token = pred_ids[0][tidx];
        if tidx >= 20 {
            *token_counts.entry(token).or_insert(0) += 1;
            if token == blank_id {
                blank_count += 1;
            } else {
                non_blank_count += 1;
            }
        }
    }

    println!("  Blank tokens: {} ({:.1}%)", blank_count, 100.0 * blank_count as f32 / t as f32);
    println!("  Non-blank tokens: {} ({:.1}%)", non_blank_count, 100.0 * non_blank_count as f32 / t as f32);

    println!("\nTop 10 predicted tokens:");
    let mut counts: Vec<_> = token_counts.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    for (token, count) in counts.iter().take(10) {
        let marker = if **token == blank_id { " (blank)" } else { "" };
        println!("  Token {:4}: {:4} times{}", token, count, marker);
    }

    // Try decoding
    println!("\nGreedy decode result:");
    let transcripts = model.greedy_decode(&logits)?;
    println!("  \"{}\"", transcripts[0]);

    // Manual CTC decode to see what's happening
    println!("\nManual CTC decode:");
    let mut prev = blank_id;
    let mut tokens = Vec::new();
    for tidx in 0..t {
        let cur = pred_ids[0][tidx];
        if cur == blank_id {
            prev = cur;
            continue;
        }
        if cur == prev {
            continue;
        }
        tokens.push(cur);
        prev = cur;
    }
    println!("  Unique non-blank tokens: {:?}", tokens);
    if !tokens.is_empty() {
        let text = model.decode_tokens(&tokens)?;
        println!("  Decoded text: \"{}\"", text);
    } else {
        println!("  No non-blank tokens found!");
    }

    Ok(())
}
