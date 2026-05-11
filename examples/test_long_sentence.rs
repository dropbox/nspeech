use std::path::Path;

fn main() -> anyhow::Result<()> {
    let assets = Path::new("assets");
    let config_json = std::fs::read_to_string(assets.join("kokoro-config.json"))?;
    let config = speech::kokoro::KokoroConfig::from_json(&config_json)?;
    let device = speech::parakeet::get_device()?;
    let model = speech::kokoro::KokoroModel::load_gguf(
        &assets.join("kokoro_q8_0.gguf"), config.clone(), &device,
    )?;
    let gold_json = std::fs::read_to_string(assets.join("us_gold.json")).unwrap_or_default();
    let silver_json = std::fs::read_to_string(assets.join("us_silver.json")).unwrap_or_default();
    let phonemizer = speech::kokoro::Phonemizer::new(&gold_json, &silver_json, &config.vocab)?;

    let base = "The quick brown fox jumps over the lazy dog near the riverbank. ";
    let mut text = String::new();
    for i in 1..=5 {
        text.push_str(base);
        let tokens = phonemizer.phonemize(&text);
        let voice_path = assets.join("kokoro-af_heart.safetensors");
        let style = speech::kokoro::KokoroModel::load_voice(&voice_path, tokens.len() + 2, &device)?;

        speech::kokoro::reset_rng();
        let gpu = model.synthesize(&tokens, &style, 1.0)?;
        speech::kokoro::reset_rng();
        let cpu = model.synthesize_cpu(&tokens, &style, 1.0)?;

        let n = gpu.len().min(cpu.len());
        let gpu_max: f32 = gpu[..n].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let cpu_max: f32 = cpu[..n].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let has_nan = gpu.iter().any(|v| v.is_nan());

        let signal: f64 = cpu[..n].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64;
        let noise: f64 = gpu[..n].iter().zip(cpu[..n].iter())
            .map(|(g, c)| { let d = (*g - *c) as f64; d * d }).sum::<f64>() / n as f64;
        let snr = if noise == 0.0 { f64::INFINITY } else { 10.0 * (signal / noise).log10() };

        let status = if has_nan { "FAIL:NaN" }
            else if gpu_max < 0.001 { "FAIL:SILENCE" }
            else if snr == f64::INFINITY { "PASS:exact" }
            else if snr > 5.0 { "PASS" }
            else { "FAIL:quality" };
        let audio_sec = n as f64 / 24000.0;
        eprintln!("rep={i} tok={:3} n={n} audio={:.2}s gpu_max={:.4} cpu_max={:.4} snr={:.1}dB  {status}",
            tokens.len(), audio_sec, gpu_max, cpu_max, snr);
    }
    Ok(())
}
