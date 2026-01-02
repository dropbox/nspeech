use anyhow::Result;
use candle_core::Device;

// Submodules
pub mod assets;
pub mod fast_conformer;
pub mod features;

// Re-export commonly used types and functions
pub use fast_conformer::{
    FastConformerConfig, ParakeetFastConformerCtc,
    load_parakeet_ctc_from_hf, load_parakeet_ctc_from_local,
    load_parakeet_ctc_from_gguf_hf, load_parakeet_ctc_from_gguf_local,
    VAD_CONFIG, VAD_MODEL, // Re-export VAD assets for use in silero module
};
pub use features::{
    ParakeetFeatureExtractor, extract_features_from_samples, load_wav_as_features,
    load_python_encoder_input,
};

/// Select the best available device for inference
/// Prefers Metal on macOS if PARAKEET_DEVICE env var is not set to "cpu"
/// Falls back to CPU with Accelerate framework
pub fn get_device() -> Result<Device> {
    // Allow forcing CPU mode via environment variable
    if std::env::var("PARAKEET_DEVICE").as_deref() == Ok("cpu") {
        println!("Using CPU (forced by PARAKEET_DEVICE=cpu)");
        return Ok(Device::Cpu);
    }

    #[cfg(target_os = "macos")]
    {
        // Note: Metal acceleration has some known issues with certain tensor operations
        // in Candle. If you encounter errors, set PARAKEET_DEVICE=cpu
        match Device::new_metal(0) {
            Ok(device) => {
                println!("Using Metal GPU acceleration");
                println!("  (If you encounter errors, try: PARAKEET_DEVICE=cpu)");
                return Ok(device);
            }
            Err(e) => {
                println!("Metal not available ({}), using CPU with Accelerate", e);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    println!("Using CPU");

    #[cfg(target_os = "macos")]
    println!("Using CPU with Accelerate framework");

    Ok(Device::Cpu)
}
