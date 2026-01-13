const { Speech, setLogCallback } = require('./index.node');
const fs = require('fs');

// Enable logging to see startup buffer accumulation
setLogCallback((event) => {
  console.log(`[${event.level.toUpperCase()}] ${event.message}`);
}, 'info');

// Simple WAV file reader
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

console.log('Testing startup buffer with first 5 seconds of dots.wav...\n');

const transcriptions = [];
const speech = new Speech('assets', (transcription) => {
  console.log(`\n[RESULT] "${transcription.text}"`);
  console.log(`  Time: ${transcription.startTime.toFixed(3)}s - ${transcription.endTime.toFixed(3)}s\n`);
  transcriptions.push(transcription);
});

// Load first 5 seconds of dots.wav (80,000 samples)
const allSamples = readWav('dots.wav');
const samples = allSamples.slice(0, 80000); // First 5 seconds
console.log(`Loaded ${samples.length} samples (${(samples.length / 16000).toFixed(2)}s)\n`);

// Send in chunks to simulate streaming
const chunkSize = 4096;
let offset = 0;

console.log('Streaming audio...\n');
const interval = setInterval(() => {
  if (offset >= samples.length) {
    clearInterval(interval);
    console.log('\n✓ All audio sent, waiting for transcription...\n');

    setTimeout(() => {
      console.log('Calling flush()...\n');
      speech.flush();

      setTimeout(() => {
        console.log(`\n✓ Complete - received ${transcriptions.length} transcription(s)`);
        speech.shutdown();
        process.exit(0);
      }, 2000);
    }, 2000);
    return;
  }

  const end = Math.min(offset + chunkSize, samples.length);
  speech.input(samples.slice(offset, end));
  offset = end;
}, 50);
