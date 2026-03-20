#!/usr/bin/env node
/**
 * Compare NAPI streaming transcription against pure Rust reference.
 *
 * Feeds all audio at once (fast), then waits for the worker to finish
 * processing and flush. Compares against pure-Rust streaming output.
 *
 * Usage:
 *   node test-napi-compare.cjs [audio.wav] [assets_dir]
 *
 * Prerequisites:
 *   cargo build --release --lib --features use-moonshine
 *   cp target/release/libspeech.dylib index.node
 */

const fs = require('fs');
const speech = require('./index.node');

const wavFile = process.argv[2] || 'dots.wav';
const assetsDir = process.argv[3] || 'assets';

// --- WAV reader (16-bit PCM or 32-bit float, mono 16kHz) ---
function readWav(filename) {
  const buf = fs.readFileSync(filename);
  if (buf.toString('ascii', 0, 4) !== 'RIFF') throw new Error('Not a WAV file');

  let offset = 12;
  let fmt = null, dataOffset = 0, dataSize = 0;

  while (offset < buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'fmt ') {
      fmt = {
        format: buf.readUInt16LE(offset + 8),
        channels: buf.readUInt16LE(offset + 10),
        sampleRate: buf.readUInt32LE(offset + 12),
        bitsPerSample: buf.readUInt16LE(offset + 22),
      };
    } else if (id === 'data') {
      dataOffset = offset + 8;
      dataSize = size;
      break;
    }
    offset += 8 + size;
  }

  if (!fmt) throw new Error('No fmt chunk');
  if (fmt.channels !== 1) throw new Error(`Expected mono, got ${fmt.channels} channels`);
  if (fmt.sampleRate !== 16000) throw new Error(`Expected 16kHz, got ${fmt.sampleRate}Hz`);

  const samples = [];
  if (fmt.format === 1 && fmt.bitsPerSample === 16) {
    for (let i = dataOffset; i < dataOffset + dataSize; i += 2) {
      samples.push(buf.readInt16LE(i) / 32768.0);
    }
  } else if (fmt.format === 3 && fmt.bitsPerSample === 32) {
    for (let i = dataOffset; i < dataOffset + dataSize; i += 4) {
      samples.push(buf.readFloatLE(i));
    }
  } else {
    throw new Error(`Unsupported format: ${fmt.format}, ${fmt.bitsPerSample}-bit`);
  }

  return { samples, sampleRate: fmt.sampleRate };
}

// --- Load Rust reference ---
const rustJsonPath = '/tmp/rust_streaming.json';
let rustRef = null;
if (fs.existsSync(rustJsonPath)) {
  rustRef = JSON.parse(fs.readFileSync(rustJsonPath, 'utf8'));
} else {
  console.log(`Warning: no Rust reference at ${rustJsonPath}`);
  console.log('Generate with: cargo run --example transcribe_moonshine_streaming --release -- dots.wav assets --json /tmp/rust_streaming.json\n');
}

// --- Run NAPI transcription ---
console.log(`=== NAPI Streaming Test: ${wavFile} ===\n`);

const { samples, sampleRate } = readWav(wavFile);
const durationSec = samples.length / sampleRate;
console.log(`Audio: ${durationSec.toFixed(2)}s, ${samples.length} samples`);

const events = [];
let gotFinal = false;

speech.setLogCallback(() => {}, 'warn');

const t0 = Date.now();

const transcriber = new speech.Speech(assetsDir, (t) => {
  const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
  events.push({
    type: t.isPartial ? 'partial' : 'final',
    segmentIndex: t.segmentIndex,
    text: t.text,
    stableText: t.stableText,
    startTime: t.startTime,
    endTime: t.endTime,
    wallTime: parseFloat(elapsed),
  });

  if (t.isPartial) {
    process.stderr.write(`\r\x1b[K  [partial ${String(events.filter(e=>e.type==='partial').length).padStart(2)}] (${elapsed}s) seg=${t.segmentIndex} "${t.text.slice(0,80)}${t.text.length>80?'...':''}"`);
  } else {
    gotFinal = true;
    process.stderr.write(`\r\x1b[K  [FINAL] (${elapsed}s) seg=${t.segmentIndex} "${t.text.slice(0,80)}${t.text.length>80?'...':''}"\n`);
  }
});

// Feed all audio at once in 1-second chunks (fits in queue: 35 items << 1000 limit)
const CHUNK = 16000;
let sent = 0;
while (sent < samples.length) {
  const end = Math.min(sent + CHUNK, samples.length);
  transcriber.input(Array.from(samples.slice(sent, end)));
  sent = end;
}
console.log(`Fed ${sent} samples. Waiting for worker...\n`);

// Poll: once we see a final event, or events stop arriving for 5s after all
// expected partials, flush (if not already done) and report.
let flushed = false;
let lastEventCount = 0;
let stableCount = 0;

const poll = setInterval(() => {
  // If we got a final, we're done
  if (gotFinal) {
    clearInterval(poll);
    setTimeout(() => { report(); transcriber.shutdown(); process.exit(0); }, 500);
    return;
  }

  // Track whether events are still arriving
  if (events.length > lastEventCount) {
    lastEventCount = events.length;
    stableCount = 0;
  } else {
    stableCount++;
  }

  // If events stopped for 5s (10 polls), flush and wait for the final
  if (!flushed && stableCount >= 10) {
    flushed = true;
    console.log(`\nNo new events for 5s (${events.length} events so far). Flushing...`);
    transcriber.flush();
  }

  // If flushed and still no final after 10s, give up
  if (flushed && stableCount >= 30) {
    clearInterval(poll);
    console.log('\nTimeout waiting for final after flush.');
    report();
    transcriber.shutdown();
    process.exit(1);
  }
}, 500);


function report() {
  const totalMs = Date.now() - t0;
  const napiPartials = events.filter(e => e.type === 'partial');
  const napiFinals = events.filter(e => e.type === 'final');
  const napiFullText = napiFinals.map(e => e.text).join(' ');

  console.log('\n========================================');
  console.log('           COMPARISON REPORT');
  console.log('========================================\n');

  // --- Finals ---
  console.log('--- FINAL TRANSCRIPTS ---\n');
  if (rustRef) {
    const rustFullText = rustRef.full_text;
    console.log(`Rust: "${rustFullText.slice(0, 120)}..."`);
    console.log(`NAPI: "${napiFullText.slice(0, 120)}${napiFullText.length > 120 ? '...' : ''}"\n`);

    if (napiFullText === rustFullText) {
      console.log('  MATCH: Final transcripts are identical.\n');
    } else if (!napiFullText) {
      console.log('  FAIL: No final transcript from NAPI.\n');
    } else {
      console.log('  MISMATCH: Final transcripts differ!\n');
      const rw = rustFullText.split(/\s+/), nw = napiFullText.split(/\s+/);
      let diffs = 0;
      for (let i = 0; i < Math.max(rw.length, nw.length); i++) {
        if (rw[i] !== nw[i]) {
          diffs++;
          if (diffs <= 10) console.log(`    word ${i}: rust="${rw[i]||'(missing)'}" napi="${nw[i]||'(missing)'}"`);
        }
      }
      if (diffs > 10) console.log(`    ... and ${diffs - 10} more differences`);
      console.log();
    }
  } else {
    console.log(`NAPI (${napiFinals.length} final(s)): "${napiFullText}"\n`);
  }

  // --- Partials ---
  console.log('--- PARTIAL TRANSCRIPTS ---\n');
  console.log(`NAPI: ${napiPartials.length} partials`);
  if (rustRef) {
    const rustPartials = rustRef.events.filter(e => e.type === 'partial');
    console.log(`Rust: ${rustPartials.length} partials\n`);

    let matching = 0;
    const minP = Math.min(rustPartials.length, napiPartials.length);
    for (let i = 0; i < minP; i++) {
      if (rustPartials[i].text === napiPartials[i].text) matching++;
    }
    console.log(`Match rate: ${matching}/${minP} (${minP ? ((matching/minP)*100).toFixed(0) : 0}%)\n`);

    // Show first divergence
    for (let i = 0; i < minP; i++) {
      if (rustPartials[i].text !== napiPartials[i].text) {
        console.log(`First diff at partial #${i+1}:`);
        console.log(`  rust: "${rustPartials[i].text.slice(0,100)}"`);
        console.log(`  napi: "${napiPartials[i].text.slice(0,100)}"`);
        console.log();
        break;
      }
    }
  }
  console.log();

  // --- Timing ---
  console.log('--- TIMING ---\n');
  console.log(`Audio: ${durationSec.toFixed(2)}s`);
  if (rustRef) console.log(`Rust:  ${rustRef.total_ms.toFixed(0)}ms (${rustRef.realtime_factor.toFixed(2)}x RT)`);
  console.log(`NAPI:  ${totalMs}ms (${(totalMs / 1000 / durationSec).toFixed(2)}x RT)\n`);

  // --- Write event log ---
  const outputPath = '/tmp/napi_streaming.json';
  fs.writeFileSync(outputPath, JSON.stringify({
    audio_file: wavFile,
    audio_duration_sec: durationSec,
    total_ms: totalMs,
    num_partials: napiPartials.length,
    num_finals: napiFinals.length,
    full_text: napiFullText,
    events,
  }, null, 2));
  console.log(`Events written to ${outputPath}`);
}
