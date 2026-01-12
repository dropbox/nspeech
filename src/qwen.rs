/// Qwen3 model for text correction (punctuation, capitalization)
///
/// **Status**: Work in progress. Requires Qwen model files to be downloaded.
/// See scripts/download_qwen3.py for setup instructions.
///
/// Uses quantized Qwen3-0.6B-Instruct model to correct raw ASR transcriptions
/// by adding proper punctuation and capitalization.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Model;
use std::io::{Error, ErrorKind};
use std::path::Path;
use tokenizers::Tokenizer;
use crate::embed_zst_asset;

// Import Qwen assets (declared in parakeet fast_conformer module)
// Qwen3-0.6B-Instruct for text correction (only when "qwen" feature enabled)
embed_zst_asset!(QWEN_CONFIG,                    "qwen3-0.6b-instruct-config.json.zst");
embed_zst_asset!(QWEN_TOKENIZER,                 "qwen3-0.6b-instruct-tokenizer.json.zst");
embed_zst_asset!(QWEN_MODEL_Q4,                  "qwen3-0.6b-instruct-q4_k_m.gguf.zst");


/// Qwen text correction model
pub struct QwenCorrector {
    model: Qwen3Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl QwenCorrector {
    /// Load Qwen model from assets directory
    ///
    /// Requires model files to be downloaded and compressed with zstd.
    /// See scripts/download_qwen3.py for setup.
    pub fn load<P: AsRef<Path>>(assets_dir: P, device: &Device) -> Result<Self> {
        let assets = assets_dir.as_ref().to_path_buf();

        // Load tokenizer
        let tokenizer_bytes = QWEN_TOKENIZER.bytes(&assets).map_err(|_| {
            Error::new(ErrorKind::Other, "Failed to load Qwen tokenizer - run: python scripts/download_qwen3.py")
        })?;

        // Validate that the tokenizer bytes are valid JSON (not an error message)
        if tokenizer_bytes.len() < 100 {
            return Err(Error::new(
                ErrorKind::Other,
                format!(
                    "Qwen tokenizer file is corrupt or incomplete ({} bytes). Contents: {}. Run: python scripts/download_qwen3.py",
                    tokenizer_bytes.len(),
                    String::from_utf8_lossy(&tokenizer_bytes)
                )
            ).into());
        }

        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Tokenizer error: {}", e)))?;

        // Load GGUF quantized model
        let gguf_bytes = QWEN_MODEL_Q4.bytes(&assets).map_err(|_| {
            Error::new(ErrorKind::Other, "Failed to load Qwen GGUF model - run: python scripts/download_qwen3.py")
        })?;

        // Validate that the GGUF file is large enough (should be hundreds of MB)
        if gguf_bytes.len() < 1_000_000 {
            return Err(Error::new(
                ErrorKind::Other,
                format!(
                    "Qwen GGUF model file is corrupt or incomplete ({} bytes). Expected > 1MB. Run: python scripts/download_qwen3.py",
                    gguf_bytes.len()
                )
            ).into());
        }

        // Parse GGUF file
        let mut cursor = std::io::Cursor::new(gguf_bytes);
        let content = gguf_file::Content::read(&mut cursor)?;

        // Load model from GGUF
        let model = Qwen3Model::from_gguf(content, &mut cursor, device)?;

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
            You are an ASR transcript post-processor.

            Goal: improve readability while preserving the speaker's meaning.
            Do NOT summarize. Do NOT shorten sentences. Do NOT drop subjects, verbs, or intent.

            Edits allowed:
            - Remove filler words (\"um\", \"uh\", \"hm\", \"like\") when they are fillers.
            - Fix obvious punctuation/casing.

            Output:
            - Return ONLY the corrected transcript (no explanations, no quotes).
            <|im_end|>
            {raw_text}"
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

            // Quantized Qwen3 already returns squeezed logits: [vocab_size]
            // Greedy: argmax
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

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
