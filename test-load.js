#!/usr/bin/env node

import { createRequire } from 'module';
import fs from 'fs';
const require = createRequire(import.meta.url);

// Enable manual garbage collection
// Run with: node --expose-gc test-load.js
const gcAvailable = typeof global.gc === 'function';
if (gcAvailable) {
  console.log('✓ Manual GC enabled');
} else {
  console.log('⚠ Manual GC not available (run with: node --expose-gc test-load.js)');
}

// Simple WAV file parser for 16-bit PCM mono
function readWav(filename) {
  const buffer = fs.readFileSync(filename);

  // Parse WAV header
  const riff = buffer.toString('ascii', 0, 4);
  if (riff !== 'RIFF') {
    throw new Error('Not a valid WAV file');
  }

  const wave = buffer.toString('ascii', 8, 12);
  if (wave !== 'WAVE') {
    throw new Error('Not a valid WAV file');
  }

  // Find data chunk
  let offset = 12;
  let dataOffset = 0;
  let dataSize = 0;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;

  while (offset < buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);

    if (chunkId === 'fmt ') {
      channels = buffer.readUInt16LE(offset + 8 + 2);
      sampleRate = buffer.readUInt32LE(offset + 8 + 4);
      bitsPerSample = buffer.readUInt16LE(offset + 8 + 14);
    } else if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize;
  }

  console.log(`WAV: ${sampleRate}Hz, ${channels} channel(s), ${bitsPerSample}-bit`);

  // Read 16-bit PCM samples and convert to float64 [-1.0, 1.0]
  const samples = [];
  let minSample = Infinity;
  let maxSample = -Infinity;
  let minFloat = Infinity;
  let maxFloat = -Infinity;

  for (let i = dataOffset; i < dataOffset + dataSize; i += 2) {
    const sample = buffer.readInt16LE(i);
    const floatSample = sample / 32768.0;
    samples.push(floatSample);

    minSample = Math.min(minSample, sample);
    maxSample = Math.max(maxSample, sample);
    minFloat = Math.min(minFloat, floatSample);
    maxFloat = Math.max(maxFloat, floatSample);
  }

  console.log(`Sample range (int16): [${minSample}, ${maxSample}]`);
  console.log(`Sample range (float): [${minFloat.toFixed(6)}, ${maxFloat.toFixed(6)}]`);

  return { samples, sampleRate, channels };
}

try {
  console.log('Loading native module...');
  const speech = require('./index.node');
  console.log('Module loaded successfully!');
  console.log('Exports:', Object.keys(speech));

  if (speech.Speech) {
    console.log('Speech constructor found');

    // Set up logging (suppress trace logs for cleaner output)
    speech.setLogCallback((s) => {
      console.log(`[${s.level}] ${s.message}`);
    }, "info");

    // Track transcription results
    const transcriptions = [];

    // Create Speech instance with callback
    const transcriber = new speech.Speech("assets", (transcription) => {
      console.log('\n=== TRANSCRIPTION ===');
      console.log(`Text: ${transcription.text}`);
      const startTime = transcription.startTime ?? transcription.start_time ?? 0;
      const endTime = transcription.endTime ?? transcription.end_time ?? 0;
      console.log(`Time: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`);
      console.log('====================\n');
      transcriptions.push(transcription);

      // Force garbage collection after each transcription to free beam search memory
      if (gcAvailable) {
        global.gc();
        console.log('  [GC triggered]');
      }
    });

    console.log('Transcriber created, loading audio...');

    // Read WAV file
    const wavFile = process.argv[2] || 'MLKDream_16k.wav';
    console.log(`Reading ${wavFile}...`);
    const { samples, sampleRate } = readWav(wavFile);

    console.log(`Loaded ${samples.length} samples (${(samples.length / sampleRate).toFixed(2)}s)`);

    // Send audio in chunks (simulate streaming with optimal chunk size)
    // 512 samples = 32ms at 16kHz, aligned with VAD processing
    const chunkSize = 512; // Optimal chunk size for VAD + TDT
    let sent = 0;

    while (sent < samples.length) {
      const end = Math.min(sent + chunkSize, samples.length);
      const chunk = samples.slice(sent, end);
      transcriber.input(chunk);
      sent = end;

      // Report progress every 5 seconds
      if (sent % (16000 * 5) === 0 || sent === samples.length) {
        console.log(`Sent ${sent}/${samples.length} samples (${(sent / sampleRate).toFixed(1)}s)`);
      }
    }

    console.log('All audio sent, waiting for processing...');
    console.log('(Worker processes at ~1x real-time, so 35s audio needs ~40s wait)');

    // Wait for worker thread to process all queued audio
    // Worker processes at roughly real-time speed: 35s audio = ~40s processing
    setTimeout(() => {
      console.log('Flushing...');
      transcriber.flush();

        // Wait a bit more for final transcriptions
        setTimeout(() => {
          console.log(`\nTotal transcriptions: ${transcriptions.length}`);
          if (transcriptions.length > 0) {
            const fullText = transcriptions.map(t => t.text).join(' ');
            console.log('\n=== FULL TRANSCRIPT ===');
            console.log(fullText);
            console.log('=======================\n');
          }

          transcriber.shutdown();
          console.log('✓ Test complete!');
        }, 2000);
    }, 40000);  // Wait 40 seconds for worker to process all audio
  }
} catch (err) {
  console.error('Failed to load module:');
  console.error(err.message);
  console.error('\nStack trace:');
  console.error(err.stack);
  process.exit(1);
}
