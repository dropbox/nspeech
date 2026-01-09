/// Qwen2.5 model for text correction (punctuation, capitalization)
///
/// **Status**: Work in progress. Requires Qwen model files to be downloaded.
/// See scripts/download_qwen_model.sh for setup instructions.
///
/// Uses quantized Qwen2.5-0.5B-Instruct model to correct raw ASR transcriptions
/// by adding proper punctuation and capitalization.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Model;
use std::io::{Error, ErrorKind};
use std::path::Path;
use tokenizers::Tokenizer;

// Import Qwen assets (declared in parakeet fast_conformer module)
use crate::parakeet::fast_conformer::{QWEN_MODEL_Q4, QWEN_TOKENIZER};

/// Qwen text correction model
pub struct QwenCorrector {
    model: Qwen2Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl QwenCorrector {
    /// Load Qwen model from assets directory
    ///
    /// Requires model files to be downloaded and compressed with zstd.
    /// See scripts/download_qwen_model.sh for setup.
    pub fn load<P: AsRef<Path>>(assets_dir: P, device: &Device) -> Result<Self> {
        let assets = assets_dir.as_ref().to_path_buf();

        // Load tokenizer
        let tokenizer_bytes = QWEN_TOKENIZER.bytes(&assets).map_err(|_| {
            Error::new(ErrorKind::Other, "Failed to load Qwen tokenizer - run scripts/download_qwen_model.sh")
        })?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Tokenizer error: {}", e)))?;

        // Load GGUF quantized model
        let gguf_bytes = QWEN_MODEL_Q4.bytes(&assets).map_err(|_| {
            Error::new(ErrorKind::Other, "Failed to load Qwen GGUF model - run scripts/download_qwen_model.sh")
        })?;

        // Parse GGUF file
        let mut cursor = std::io::Cursor::new(gguf_bytes);
        let content = gguf_file::Content::read(&mut cursor)?;

        // Load model from GGUF
        let model = Qwen2Model::from_gguf(content, &mut cursor, device)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Correct raw transcription text by adding punctuation and capitalization
    ///
    /// Takes raw lowercase text without punctuation and returns corrected text.
    /// Uses a single-shot prompt to the model.
    pub fn correct_text(&mut self, raw_text: &str) -> Result<String> {
        if raw_text.trim().is_empty() {
            return Ok(String::new());
        }

        // Construct prompt for text correction
        let prompt = format!(
            "<|im_start|>system\n\
            You are a helpful assistant that corrects transcriptions by adding proper punctuation and capitalization. \
            Only output the corrected text, nothing else.<|im_end|>\n\
            <|im_start|>user\n\
            Correct this transcription by adding punctuation and capitalization:\n\
            {}<|im_end|>\n\
            <|im_start|>assistant\n",
            raw_text
        );

        // Tokenize
        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?
            .get_ids()
            .to_vec();

        // Generate with simple greedy decoding
        let max_new_tokens = raw_text.split_whitespace().count() + 20; // Allow some extra tokens
        let generated = self.generate_tokens(&tokens, max_new_tokens)?;

        // Decode
        let output = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("Decoding error: {}", e))?;

        // Extract just the assistant's response (after the last <|im_start|>assistant\n)
        let corrected = output
            .split("<|im_start|>assistant\n")
            .last()
            .unwrap_or(&output)
            .split("<|im_end|>")
            .next()
            .unwrap_or(&output)
            .trim()
            .to_string();

        Ok(corrected)
    }

    /// Simple greedy token generation
    fn generate_tokens(&mut self, prompt_tokens: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        use candle_core::Tensor;

        let mut tokens = prompt_tokens.to_vec();
        let eos_token_id = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .unwrap_or(151645); // Qwen2.5 EOS token ID

        for _ in 0..max_new_tokens {
            // Create input tensor [1, seq_len]
            let input = Tensor::new(tokens.as_slice(), &self.device)?
                .unsqueeze(0)?;

            // Forward pass
            let logits = self.model.forward(&input, 0)?; // position 0 for KV cache (simplified)

            // Get last token logits [vocab_size]
            // Quantized Qwen2 only returns logits for last position: [1, vocab_size]
            let last_logits = logits.squeeze(0)?;  // [vocab_size]

            // Greedy: argmax
            let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

            // Stop on EOS
            if next_token == eos_token_id {
                break;
            }

            tokens.push(next_token);
        }

        // Return only generated tokens (skip prompt)
        Ok(tokens[prompt_tokens.len()..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model files
    fn test_qwen_correction() {
        let device = Device::Cpu;
        let mut corrector = QwenCorrector::load("assets", &device).unwrap();

        let raw = "hello world this is a test";
        let corrected = corrector.correct_text(raw).unwrap();

        // Should have capitalization
        assert!(corrected.starts_with("Hello") || corrected.starts_with("HELLO"));
        // Should have some punctuation
        assert!(corrected.contains(".") || corrected.contains("!") || corrected.contains("?"));
    }
}
