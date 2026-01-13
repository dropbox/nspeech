const { Speech, setLogCallback } = require('./index.node');
const fs = require('fs');

// Enable logging
setLogCallback((event) => {
  console.log(`[${event.level.toUpperCase()}] ${event.message}`);
}, 'info');

// Simple WAV file reader (assumes 16-bit PCM, mono, 16kHz)
function readWav(filename) {
  const buffer = fs.readFileSync(filename);
  const dataStart = 44;
  const samples = [];
  for (let i = dataStart; i < buffer.length; i += 2) {
    const sample = buffer.readInt16LE(i);
    samples.push(sample / 32768.0);
  }
  return samples;
}

console.log('Testing with debug_input.wav...\n');

// Create speech instance with callback
const transcriptions = [];
const speech = new Speech('assets', (transcription) => {
  console.log(`\n[Transcription ${transcriptions.length + 1}] "${transcription.text}"`);
  console.log(`  Time: ${transcription.startTime.toFixed(2)}s - ${transcription.endTime.toFixed(2)}s\n`);
  transcriptions.push(transcription);
});

// Load audio file
console.log('Loading debug_input.wav...');
const audioSamples = readWav('debug_input.wav');
console.log(`✓ Loaded ${audioSamples.length} samples (${(audioSamples.length / 16000).toFixed(2)}s)\n`);

// Send all audio at once
console.log('Sending audio...\n');
speech.input(audioSamples);

// Wait for processing
setTimeout(() => {
  console.log('\n📤 Calling flush()...\n');
  speech.flush();

  setTimeout(() => {
    console.log(`\n✓ Test complete - received ${transcriptions.length} transcription(s)`);
    speech.shutdown();
    process.exit(0);
  }, 3000);
}, 2000);
