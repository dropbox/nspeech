#!/usr/bin/env node
/**
 * Example usage of the Parakeet transcription stream
 */

const fs = require('fs');
const path = require('path');
const { TranscriptionStream, init_logging } = require('./index.js');

// Initialize logging
init_logging();

async function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.error('Usage: node example.js <assets_path> <audio.wav>');
    console.error('');
    console.error('assets_path should contain:');
    console.error('  - vad16.safetensors');
    console.error('  - vad16.config.json');
    console.error('  - hf_parakeet/config.json');
    console.error('  - hf_parakeet/model_q8_0.gguf (or model_q4k.gguf)');
    console.error('  - hf_parakeet/tokenizer.json');
    process.exit(1);
  }

  const assetsPath = args[0];
  const wavPath = args[1];

  console.log(`Loading models from: ${assetsPath}`);
  console.log(`Audio file: ${wavPath}`);
  console.log('');

  // Create transcription stream with callback
  const stream = new TranscriptionStream(assetsPath, (transcription) => {
    console.log(`[${transcription.start_time.toFixed(2)}s - ${transcription.end_time.toFixed(2)}s]`);
    console.log(`  "${transcription.text}"`);
    console.log('');
  });

  console.log('✓ Models loaded\n');

  // Read WAV file
  console.log('Loading audio...');
  const wavBuffer = fs.readFileSync(wavPath);

  // Parse WAV header (simple parser for 16-bit PCM)
  const dataView = new DataView(wavBuffer.buffer, wavBuffer.byteOffset, wavBuffer.byteLength);

  // Find 'data' chunk
  let dataOffset = 44; // Standard WAV header size
  let dataSize = dataView.getUint32(dataOffset - 4, true);

  // Read 16-bit PCM samples
  const numSamples = dataSize / 2;
  const samples = new Float64Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    const sample = dataView.getInt16(dataOffset + i * 2, true);
    samples[i] = sample / 32768.0; // Normalize to [-1, 1]
  }

  console.log(`✓ Audio loaded: ${numSamples} samples (${(numSamples / 16000).toFixed(2)}s)\n`);
  console.log('Processing...\n');

  // Stream audio in chunks (simulate real-time streaming)
  const chunkSize = 16000; // 1 second chunks
  for (let i = 0; i < samples.length; i += chunkSize) {
    const end = Math.min(i + chunkSize, samples.length);
    const chunk = samples.slice(i, end);
    stream.input(chunk);
  }

  // For WAV file processing, feed silence to transcribe any remaining speech
  // (In true streaming, you'd just keep feeding live audio indefinitely)
  console.log('Feeding silence to trigger final transcription...');
  const silence = new Float64Array(8000); // 500ms silence
  stream.input(silence);

  // Wait a bit for final callback to fire
  setTimeout(() => {
    console.log('\n✓ Transcription complete!');
  }, 100);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
