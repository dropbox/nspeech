# Debug WAV File Update

## Summary

Changed the debug WAV file (`debug_input.wav`) to capture **post-VAD audio** (what Parakeet actually transcribes) instead of raw input audio.

## Motivation

Previously, the debug WAV file contained all incoming audio samples, including silence and non-speech segments. This made it difficult to debug transcription issues because the WAV file didn't match what Parakeet was actually processing.

Now, the debug WAV file contains only the speech segments that are sent to Parakeet for transcription, making it much easier to:
- Debug transcription accuracy issues
- Verify VAD is segmenting audio correctly
- Reproduce transcription results exactly

## Changes

### Before
```rust
fn process_samples(&mut self, samples: &[f32], ...) {
    // Write ALL incoming samples to debug WAV
    if let Some(ref mut writer) = self.debug_wav_writer {
        for &sample in samples {
            let _ = writer.write_sample(sample);
        }
    }
    // ... VAD processing ...
}
```

**Result**: `debug_input.wav` contained everything (speech + silence + noise)

### After
```rust
fn transcribe_segment(&mut self) {
    // ... after segment is complete via VAD ...

    // Write ONLY the speech segment to debug WAV
    if let Some(ref mut writer) = self.debug_wav_writer {
        for &sample in &self.current_segment {
            let _ = writer.write_sample(sample);
        }
        let _ = writer.flush();
    }

    // ... then transcribe the segment ...
}
```

**Result**: `debug_input.wav` contains only speech segments (post-VAD filtering)

## Usage

The debug WAV file is automatically created when the Speech module initializes:

```javascript
const speech = new Speech('./assets', (transcription) => {
  console.log(transcription);
});

// debug_input.wav is automatically created in the current directory
// It will contain only the audio segments that were transcribed
```

## What's in the Debug WAV File Now

The file contains:
- ✅ Speech segments detected by VAD
- ✅ Pre-buffer audio (captured before speech start)
- ✅ Post-buffer audio (captured after speech end)
- ✅ Audio between pauses (for comma/period detection)
- ❌ Silence gaps between utterances
- ❌ Non-speech noise filtered by VAD

## Benefits

1. **Easier debugging**: WAV file matches exactly what Parakeet transcribes
2. **Reproducible**: Can feed the debug WAV back into Parakeet to reproduce results
3. **Smaller files**: No silence/noise padding
4. **Better analysis**: Can verify VAD segmentation accuracy

## Technical Details

- **Format**: 16kHz mono, 32-bit float WAV
- **Location**: `debug_input.wav` in current working directory
- **Appends**: Each segment is appended sequentially (multiple utterances concatenated)
- **Flush**: Flushed after each segment for immediate availability

## Files Modified

- `src/lib.rs`:
  - Removed debug WAV writing from `process_samples()` (line 263-270)
  - Added debug WAV writing to `transcribe_segment()` (line 483-491)
  - Changed `transcribe_segment()` signature to `&mut self` (line 474)
