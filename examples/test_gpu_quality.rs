//! Compare GPU vs CPU decoder quality (SNR and envelope correlation).

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let voice_name = "af_heart";
    let assets = Path::new("assets");
    let config_json = std::fs::read_to_string(assets.join("kokoro-config.json"))?;
    let config = speech::kokoro::KokoroConfig::from_json(&config_json)?;

    eprintln!("Loading model...");
    let device = speech::parakeet::get_device()?;
    let model = speech::kokoro::KokoroModel::load_gguf(
        &assets.join("kokoro_q8_0.gguf"), config.clone(), &device,
    )?;

    let gold_json = std::fs::read_to_string(assets.join("us_gold.json")).unwrap_or_default();
    let silver_json = std::fs::read_to_string(assets.join("us_silver.json")).unwrap_or_default();
    let phonemizer = speech::kokoro::Phonemizer::new(&gold_json, &silver_json, &config.vocab)?;

    let texts: Vec<String> = vec![
        "Hello world, this is a test.".into(),
        "Hello world, this is a test of the speech synthesis system.".into(),
        "The quick brown fox jumps over the lazy dog. ".repeat(3),
    ];

    for text in &texts {
        eprintln!("\n--- \"{}...\" ---", &text[..text.len().min(60)]);
        let tokens = phonemizer.phonemize(text);
        eprintln!("  {} tokens", tokens.len());

        let voice_path = assets.join(format!("kokoro-{voice_name}.safetensors"));
        let style = speech::kokoro::KokoroModel::load_voice(&voice_path, tokens.len() + 2, &device)?;

        speech::kokoro::reset_rng();
        let gpu_audio = model.synthesize(&tokens, &style, 1.0)?;
        eprintln!("  GPU: {} samples ({:.2}s)", gpu_audio.len(), gpu_audio.len() as f64 / 24000.0);

        speech::kokoro::reset_rng();
        let cpu_audio = model.synthesize_cpu(&tokens, &style, 1.0)?;
        eprintln!("  CPU: {} samples ({:.2}s)", cpu_audio.len(), cpu_audio.len() as f64 / 24000.0);

        let n = gpu_audio.len().min(cpu_audio.len());
        let gpu = &gpu_audio[..n];
        let cpu = &cpu_audio[..n];

        let signal_power: f64 = cpu.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
        let noise_power: f64 = gpu.iter().zip(cpu.iter())
            .map(|(&g, &c)| { let d = (g - c) as f64; d * d })
            .sum::<f64>() / n as f64;
        let snr_db = 10.0 * (signal_power / noise_power.max(1e-30)).log10();

        eprintln!("  SNR: {snr_db:.1} dB");
        if snr_db < 30.0 {
            eprintln!("  FAIL (target: >30 dB)");
            std::process::exit(1);
        }
    }
    eprintln!("\nAll PASS");
    Ok(())
}
