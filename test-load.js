#!/usr/bin/env node

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

try {
  console.log('Loading native module...');
  const speech = require('./index.node');
  console.log('Module loaded successfully!');
  console.log('Exports:', Object.keys(speech));

  // Try to access Speech constructor
  if (speech.Speech) {
    console.log('Speech constructor found');
  }
} catch (err) {
  console.error('Failed to load module:');
  console.error(err.message);
  console.error('\nStack trace:');
  console.error(err.stack);
  process.exit(1);
}
