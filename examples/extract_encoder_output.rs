/// Extract encoder output for comparison with NeMo
///
/// This example extracts features and encoder output from our Rust implementation
/// so we can compare with NeMo to identify where differences occur.

use anyhow::Result;
use candle_core::IndexOp;
use speech::parakeet::{
    get_device, load_parakeet_tdt_from_local, ParakeetFeatureExtractor,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav>", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];

    println!("Extracting encoder output from Rust implementation");
    println!("==================================================\n");
    println!("Audio: {}\n", audio_path);

    // Load model
    let device = get_device()?;
    println!("Device: {:?}", device);

    println!("Loading TDT model...");
    let model = load_parakeet_tdt_from_local(".cache/parakeet-tdt", &device)?;
    println!("✓ Model loaded\n");

    // Load audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()?;

    let samples = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };

    println!("Audio samples: {}", samples.len());

    // Extract features
    println!("\nExtracting features...");
    let feat_extractor = ParakeetFeatureExtractor::new(128);
    let features = feat_extractor.extract_to_tensor(&samples, &device)?;
    let features = if !device.is_cpu() {
        features.to_dtype(candle_core::DType::BF16)?
    } else {
        features
    };

    let (batch, mel_frames, feat_dim) = features.dims3()?;
    println!("Features shape: [batch={}, time={}, features={}]", batch, mel_frames, feat_dim);

    // Print feature statistics
    let features_f32 = features.to_dtype(candle_core::DType::F32)?;
    let features_vec: Vec<f32> = features_f32.flatten_all()?.to_vec1()?;
    let mean = features_vec.iter().sum::<f32>() / features_vec.len() as f32;
    let variance = features_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / features_vec.len() as f32;
    let std = variance.sqrt();
    let min = features_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = features_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\nFeatures statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std: {:.6}", std);
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}", max);

    // Print first few values
    println!("\nFirst 10 frames, first 5 features:");
    for t in 0..10.min(mel_frames) {
        let frame_data: Vec<f32> = (0..5)
            .map(|f| features_f32.i((0, t, f)).unwrap().to_scalar::<f32>().unwrap())
            .collect();
        println!("  Frame {}: {:?}", t, frame_data);
    }

    // Run encoder
    println!("\nRunning encoder...");
    let encoder_out = model.encoder.forward(&features, false)?;
    let (batch, enc_frames, enc_dim) = encoder_out.dims3()?;
    println!("Encoder output shape: [batch={}, time={}, features={}]", batch, enc_frames, enc_dim);

    // Print encoder statistics
    let encoder_f32 = encoder_out.to_dtype(candle_core::DType::F32)?;
    let encoder_vec: Vec<f32> = encoder_f32.flatten_all()?.to_vec1()?;
    let mean = encoder_vec.iter().sum::<f32>() / encoder_vec.len() as f32;
    let variance = encoder_vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / encoder_vec.len() as f32;
    let std = variance.sqrt();
    let min = encoder_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = encoder_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\nEncoder output statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std: {:.6}", std);
    println!("  Min: {:.6}", min);
    println!("  Max: {:.6}", max);

    // Print first few values
    println!("\nFirst 5 frames, first 5 features:");
    for t in 0..5.min(enc_frames) {
        let frame_data: Vec<f32> = (0..5)
            .map(|f| encoder_f32.i((0, t, f)).unwrap().to_scalar::<f32>().unwrap())
            .collect();
        println!("  Frame {}: {:?}", t, frame_data);
    }

    println!("\n✓ Extraction complete!");
    println!("\nCompare with NeMo:");
    println!("  ~/bin/jrpython compare_encoder_outputs.py {}", audio_path);

    Ok(())
}
