#!/usr/bin/env node

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

const speech = require('./index.node');

console.log('Creating Speech instance...');
console.log('Note: This test demonstrates that input() returns immediately');
console.log('while processing happens asynchronously in the background.\n');

// Create a simple callback
let callbackCount = 0;
const callback = (transcription) => {
  callbackCount++;
  console.log(`[Callback ${callbackCount}] Transcription received:`, transcription);
};

try {
  // This would normally load real models, but for this test we just want to
  // show that the API works and input() returns immediately
  const instance = new speech.Speech('./assets', callback);

  console.log('Speech instance created successfully');
  console.log('Calling input() with sample data...');

  // Create some dummy audio samples (16kHz, 1 second = 16000 samples)
  const samples = new Array(16000).fill(0);

  const start = Date.now();
  instance.input(samples);
  const elapsed = Date.now() - start;

  console.log(`input() returned in ${elapsed}ms (should be < 1ms)`);
  console.log('Processing is happening in the background!');
  console.log('\nNote: Callbacks will fire asynchronously as speech segments are detected.');

  // Keep process alive briefly to allow background processing
  setTimeout(() => {
    console.log(`\nTotal callbacks received: ${callbackCount}`);
    instance.shutdown();
    process.exit(0);
  }, 1000);

} catch (err) {
  console.error('Error:', err.message);
  process.exit(1);
}
