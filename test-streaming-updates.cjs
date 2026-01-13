const { Speech } = require('./index.node');

console.log('Testing Streaming Buffer with Continuous Updates\n');
console.log('This demonstrates:');
console.log('  1. Continuous updates every ~750ms (streaming buffer)');
console.log('  2. VAD-based segmentation on pauses');
console.log('  3. Manual flush() at the end\n');

let updateCount = 0;
const speech = new Speech('assets', (transcription) => {
  updateCount++;
  const duration = (transcription.endTime - transcription.startTime).toFixed(2);
  console.log(`[Update ${updateCount}] ${transcription.startTime.toFixed(2)}s - ${transcription.endTime.toFixed(2)}s (${duration}s)`);
  console.log(`  "${transcription.text}"\n`);
});

console.log('Sending short audio bursts with pauses...\n');

// Simulate streaming audio with pauses
const SAMPLE_RATE = 16000;
const samples_200ms = new Array(SAMPLE_RATE * 0.2).fill(0).map(() => Math.random() * 0.01);
const samples_1s = new Array(SAMPLE_RATE * 1.0).fill(0).map(() => Math.random() * 0.01);

// Send first burst
console.log('[0.0s] Sending 1s of audio...');
speech.input(samples_1s);

setTimeout(() => {
  console.log('[1.0s] Sending another 1s of audio...');
  speech.input(samples_1s);

  setTimeout(() => {
    console.log('[2.0s] Sending final 200ms burst...');
    speech.input(samples_200ms);

    setTimeout(() => {
      console.log('[2.5s] Calling flush() to get final content...\n');
      speech.flush();

      setTimeout(() => {
        console.log(`\n✓ Test complete - received ${updateCount} update(s) total`);
        console.log('\nKey observations:');
        console.log('  - Updates come every ~750ms (streaming buffer interval)');
        console.log('  - flush() forces transcription of remaining buffer');
        console.log('  - All updates invoke the callback immediately');
        speech.shutdown();
        process.exit(0);
      }, 2000);
    }, 500);
  }, 1000);
}, 1000);
