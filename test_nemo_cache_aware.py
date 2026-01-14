#!/usr/bin/env python3
"""Test NeMo cache-aware streaming inference"""

import nemo.collections.asr as nemo_asr
import torch
import numpy as np
import time
import os

print("\nLoading model from HuggingFace...")
start = time.time()

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)

load_time = time.time() - start
print(f"✓ Model loaded ({load_time:.2f}s)")

# Find audio file
audio_paths = ["dots.wav", "/workspace/dots.wav", "assets/assets/dots.wav"]
audio_file = None
for path in audio_paths:
    if os.path.exists(path):
        audio_file = path
        break

if not audio_file:
    print(f"ERROR: dots.wav not found")
    import sys
    sys.exit(1)

# Load audio
import soundfile as sf
audio, sr = sf.read(audio_file)
print(f"\nAudio: {audio_file}")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(audio)/sr:.2f}s")
print(f"  Samples: {len(audio)}")

# Try to use cache-aware streaming if available
if hasattr(asr_model, 'conformer_stream_step'):
    print("\n=== CACHE-AWARE STREAMING MODE ===")
    print("Model supports cache-aware streaming!")

    # Get model config for chunk size
    att_context_size = getattr(asr_model.encoder, 'att_context_size', [70, 6])
    print(f"att_context_size: {att_context_size}")

    # Calculate chunk size (560ms for [70, 6])
    # Each encoder frame = 80ms, so (right_context + 1) * 80ms
    right_context = att_context_size[1] if isinstance(att_context_size, list) else 6
    chunk_ms = (right_context + 1) * 80
    chunk_samples = chunk_ms * 16  # 16 samples per ms at 16kHz

    print(f"Chunk size: {chunk_ms}ms ({chunk_samples} samples)")

    # Initialize encoder caches
    batch_size = 1
    cache_last_channel, cache_last_time, cache_last_channel_len = \
        asr_model.encoder.get_initial_cache_state(batch_size=batch_size)

    # Initialize decoder state
    previous_hypotheses = None
    pred_out_stream = None

    print("\nProcessing audio in chunks...")
    all_tokens = []
    chunk_count = 0

    # Process in chunks
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        if len(chunk) < chunk_samples:
            # Pad last chunk
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunk_count += 1

        # Convert to tensor and add batch dimension
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
        chunk_lengths = torch.tensor([len(chunk)])

        # Run cache-aware streaming step
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = asr_model.conformer_stream_step(
            processed_signal=chunk_tensor,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=None,
        )

        # Extract tokens if available
        if previous_hypotheses is not None and len(previous_hypotheses) > 0:
            hyp = previous_hypotheses[0]
            if hasattr(hyp, 'y_sequence'):
                tokens = hyp.y_sequence.tolist()
                all_tokens = tokens  # Update with latest

                if chunk_count <= 5 or chunk_count % 10 == 0:
                    print(f"  Chunk {chunk_count}: {len(tokens)} tokens total")

    print(f"\n✓ Processed {chunk_count} chunks")
    print(f"  Total tokens: {len(all_tokens)}")

    # Get final transcription
    if previous_hypotheses and len(previous_hypotheses) > 0:
        final_text = previous_hypotheses[0].text
        print(f"\n=== STREAMING TRANSCRIPTION ===")
        print(final_text)
        print(f"\nFirst 10 tokens: {all_tokens[:10]}")

else:
    print("\nModel does not support cache-aware streaming")
    print("Using regular transcribe() instead")

    transcription = asr_model.transcribe([audio_file])[0]
    print("\n=== TRANSCRIPTION ===")
    print(transcription)

print("\n✓ Done")
