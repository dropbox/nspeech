const { Speech } = require('./index.node');
const fs = require('fs');

// Simple WAV file reader (assumes 16-bit PCM, mono, 16kHz)
function readWav(filename) {
  const buffer = fs.readFileSync(filename);

  // Skip 44-byte WAV header
  const dataStart = 44;
  const samples = [];

  // Read 16-bit PCM samples and convert to Float64
  for (let i = dataStart; i < buffer.length; i += 2) {
    const sample = buffer.readInt16LE(i);
    // Convert to float in range [-1, 1]
    samples.push(sample / 32768.0);
  }

  return samples;
}

console.log('Testing Speech API flush() with real audio...\n');

// Create speech instance with callback
const transcriptions = [];
const speech = new Speech('assets', (transcription) => {
  console.log(`\n[Transcription ${transcriptions.length + 1}] "${transcription.text}"`);
  console.log(`  Time: ${transcription.startTime.toFixed(2)}s - ${transcription.endTime.toFixed(2)}s\n`);
  transcriptions.push(transcription);
});

console.log('✓ Speech instance created\n');

// Load audio file
console.log('Loading dots.wav...');
const audioSamples = readWav('dots.wav');
console.log(`✓ Loaded ${audioSamples.length} samples (${(audioSamples.length / 16000).toFixed(2)}s)\n`);

// Send audio in chunks (simulate streaming)
const chunkSize = 4096; // 256ms chunks at 16kHz
let offset = 0;

console.log('Streaming audio in chunks...');
const streamInterval = setInterval(() => {
  if (offset >= audioSamples.length) {
    clearInterval(streamInterval);

    // After all audio sent, wait for queue to drain
    console.log('\n✓ All audio sent');
    console.log('⏳ Waiting 5 seconds for queue to drain...');

    setTimeout(() => {
      console.log('📤 Calling flush() to force final transcription...\n');
      speech.flush();

      // Wait for flush to complete
      setTimeout(() => {
        console.log(`\n✓ Test complete - received ${transcriptions.length} transcription(s)`);
        speech.shutdown();
        process.exit(0);
      }, 3000);
    }, 5000);

    return;
  }

  const end = Math.min(offset + chunkSize, audioSamples.length);
  const chunk = audioSamples.slice(offset, end);
  speech.input(chunk);

  offset = end;
  process.stdout.write(`\r  Progress: ${((offset / audioSamples.length) * 100).toFixed(1)}%`);
}, 100); // Send chunk every 100ms
