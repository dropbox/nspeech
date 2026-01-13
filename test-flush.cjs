const { Speech } = require('./index.node');
const fs = require('fs');

console.log('Testing Speech API with flush()...\n');

// Create speech instance with callback
const speech = new Speech('assets', (transcription) => {
  console.log(`[Transcription] "${transcription.text}"`);
  console.log(`  Time: ${transcription.start_time.toFixed(2)}s - ${transcription.end_time.toFixed(2)}s\n`);
});

console.log('✓ Speech instance created');
console.log('✓ Methods available:', Object.getOwnPropertyNames(Object.getPrototypeOf(speech)));

// Test that flush() method exists
if (typeof speech.flush === 'function') {
  console.log('✓ flush() method is available\n');

  // Simulate sending some audio samples
  console.log('Sending audio samples...');
  const sampleCount = 16000; // 1 second of audio at 16kHz
  const samples = new Array(sampleCount).fill(0).map(() => Math.random() * 0.1);

  speech.input(samples);
  console.log(`✓ Sent ${sampleCount} samples\n`);

  // Wait a bit for processing
  setTimeout(() => {
    console.log('Calling flush() to force transcription...');
    speech.flush();

    // Give time for async flush to complete
    setTimeout(() => {
      console.log('\n✓ Test complete');
      speech.shutdown();
      process.exit(0);
    }, 2000);
  }, 1000);

} else {
  console.error('✗ flush() method not found!');
  process.exit(1);
}
