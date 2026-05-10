//! Compare GPU vs CPU decoder quality (SNR and envelope correlation).

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let text = "Hello world, this is a test of the speech synthesis system.";
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

    let tokens = phonemizer.phonemize(text);
    eprintln!("  {} tokens", tokens.len());

    let voice_path = assets.join(format!("kokoro-{voice_name}.safetensors"));
    let style = speech::kokoro::KokoroModel::load_voice(&voice_path, tokens.len() + 2, &device)?;

    eprintln!("Synthesizing GPU...");
    let gpu_audio = model.synthesize(&tokens, &style, 1.0)?;
    eprintln!("  {} samples ({:.2}s)", gpu_audio.len(), gpu_audio.len() as f64 / 24000.0);

    eprintln!("Synthesizing CPU...");
    let cpu_audio = model.synthesize_cpu(&tokens, &style, 1.0)?;
    eprintln!("  {} samples ({:.2}s)", cpu_audio.len(), cpu_audio.len() as f64 / 24000.0);

    let n = gpu_audio.len().min(cpu_audio.len());
    let gpu = &gpu_audio[..n];
    let cpu = &cpu_audio[..n];

    // SNR: signal = cpu, noise = gpu - cpu
    let signal_power: f64 = cpu.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
    let noise_power: f64 = gpu.iter().zip(cpu.iter())
        .map(|(&g, &c)| { let d = (g - c) as f64; d * d })
        .sum::<f64>() / n as f64;
    let snr_db = 10.0 * (signal_power / noise_power.max(1e-30)).log10();

    // Envelope correlation (downsample to ~100 Hz blocks)
    let block = 240; // 24000/100
    let n_blocks = n / block;
    let mut env_gpu = vec![0.0f64; n_blocks];
    let mut env_cpu = vec![0.0f64; n_blocks];
    for b in 0..n_blocks {
        let base = b * block;
        for i in 0..block {
            env_gpu[b] += (gpu[base + i] as f64).abs();
            env_cpu[b] += (cpu[base + i] as f64).abs();
        }
    }
    let mean_g: f64 = env_gpu.iter().sum::<f64>() / n_blocks as f64;
    let mean_c: f64 = env_cpu.iter().sum::<f64>() / n_blocks as f64;
    let mut cov = 0.0f64;
    let mut var_g = 0.0f64;
    let mut var_c = 0.0f64;
    for b in 0..n_blocks {
        let dg = env_gpu[b] - mean_g;
        let dc = env_cpu[b] - mean_c;
        cov += dg * dc;
        var_g += dg * dg;
        var_c += dc * dc;
    }
    let corr = cov / (var_g.sqrt() * var_c.sqrt()).max(1e-30);

    // Max absolute error
    let max_err: f32 = gpu.iter().zip(cpu.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    eprintln!("\n=== GPU vs CPU Quality ===");
    eprintln!("  SNR:         {snr_db:.1} dB (target: >30)");
    eprintln!("  Envelope R:  {corr:.4} (target: >0.99)");
    eprintln!("  Max error:   {max_err:.4}");
    eprintln!("  Samples:     {n}");

    if snr_db > 30.0 && corr > 0.99 {
        eprintln!("\n  PASS");
    } else {
        eprintln!("\n  FAIL");
        std::process::exit(1);
    }

    Ok(())
}
