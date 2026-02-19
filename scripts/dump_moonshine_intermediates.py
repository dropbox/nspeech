#!/usr/bin/env python3
"""Phase 2: Dump intermediate tensors from Moonshine V2 for Rust verification.

Uses HuggingFace transformers implementation to dump:
- Frontend (embedder) output
- Encoder output
- Decoder output (first few tokens)
- Final logits

Saves as numpy .npy files in moonshine_intermediates/
"""

import os
import sys
import wave
import struct
import json
import numpy as np
import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def load_wav_pcm16(path: str):
    """Load a 16-bit PCM WAV file and return float samples."""
    with wave.open(path, 'rb') as wf:
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        raw = wf.readframes(n_frames)
        samples = struct.unpack(f'<{n_frames}h', raw)
        return np.array([s / 32768.0 for s in samples], dtype=np.float32), sample_rate


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "dots.wav"
    output_dir = "moonshine_intermediates"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading audio: {wav_path}")
    samples, sample_rate = load_wav_pcm16(wav_path)
    print(f"  Samples: {len(samples)}, Sample rate: {sample_rate}")

    print("\nLoading model from HuggingFace...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "UsefulSensors/moonshine-streaming-medium",
        torch_dtype=torch.float32,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-medium")
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Use the processor to get proper input_values and attention_mask
    inputs = processor(samples, return_tensors="pt", sampling_rate=sample_rate)
    input_values = inputs["input_values"]  # [1, audio_len] - padded to frame_len multiple
    attention_mask = inputs.get("attention_mask")  # [1, audio_len] or None
    print(f"  Processor input_values shape: {input_values.shape}")
    if attention_mask is not None:
        print(f"  Processor attention_mask shape: {attention_mask.shape}")
        print(f"  Attention mask sum: {attention_mask.sum().item()} / {attention_mask.numel()}")

    # Also save raw audio padded to frame_len for Rust
    frame_len = 80
    pad_len = (frame_len - len(samples) % frame_len) % frame_len
    padded_samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.float32)]) if pad_len > 0 else samples
    np.save(os.path.join(output_dir, "audio_input.npy"), padded_samples)
    print(f"  Raw padded audio: {len(padded_samples)} samples ({len(padded_samples) / sample_rate:.3f}s)")

    with torch.no_grad():
        # Run encoder (embedder + transformer layers)
        encoder = model.model.encoder
        decoder = model.model.decoder

        # Step 1: Embedder (frontend) - use raw padded audio (no processor padding)
        raw_input = torch.tensor(padded_samples, dtype=torch.float32).unsqueeze(0)
        embedder = encoder.embedder
        hidden_states = embedder.cmvn(raw_input.reshape(1, -1, embedder.frame_len))
        np.save(os.path.join(output_dir, "cmvn_output.npy"), hidden_states.numpy())
        print(f"\n  CMVN output: {hidden_states.shape}")

        hidden_states = embedder.comp(hidden_states)
        np.save(os.path.join(output_dir, "asinh_comp_output.npy"), hidden_states.numpy())
        print(f"  Asinh compression output: {hidden_states.shape}")

        hidden_states = torch.nn.functional.silu(embedder.linear(hidden_states))
        np.save(os.path.join(output_dir, "linear_silu_output.npy"), hidden_states.numpy())
        print(f"  Linear+SiLU output: {hidden_states.shape}")

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states, _ = embedder.conv1(hidden_states, None)
        hidden_states_post_conv1 = torch.nn.functional.silu(hidden_states)
        np.save(os.path.join(output_dir, "conv1_output.npy"), hidden_states_post_conv1.numpy())
        print(f"  Conv1+SiLU output: {hidden_states_post_conv1.shape}")

        hidden_states, _ = embedder.conv2(hidden_states_post_conv1, None)
        hidden_states = hidden_states.transpose(1, 2)
        np.save(os.path.join(output_dir, "embedder_output.npy"), hidden_states.numpy())
        print(f"  Embedder output: {hidden_states.shape}")

        # Step 2: Full encoder output (using processor input for correct result)
        encoder_output = encoder(input_values, attention_mask=attention_mask)
        encoder_hidden = encoder_output.last_hidden_state
        np.save(os.path.join(output_dir, "encoder_output.npy"), encoder_hidden.numpy())
        print(f"  Encoder output: {encoder_hidden.shape}")

        # Also save encoder output from raw audio (no processor attention_mask)
        # This is what our Rust code will produce
        encoder_output_raw = encoder(raw_input)
        encoder_hidden_raw = encoder_output_raw.last_hidden_state
        np.save(os.path.join(output_dir, "encoder_output_raw.npy"), encoder_hidden_raw.numpy())
        print(f"  Encoder output (raw, no mask): {encoder_hidden_raw.shape}")

        # Step 3: Generate tokens with greedy decode using processor inputs
        print("\n  Running generation (with processor)...")
        generated = model.generate(
            input_values,
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=False,  # greedy
        )
        generated_ids = generated[0].tolist()
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        print(f"  Generated tokens ({len(generated_ids)}): {generated_ids[:20]}...")
        print(f"  Transcription: {text}")

        # Also try with raw audio (no attention_mask)
        print("\n  Running generation (raw, no mask)...")
        generated_raw = model.generate(
            raw_input,
            max_new_tokens=200,
            do_sample=False,
        )
        generated_raw_ids = generated_raw[0].tolist()
        text_raw = processor.batch_decode(generated_raw, skip_special_tokens=True)[0]
        print(f"  Generated tokens (raw): {generated_raw_ids[:20]}...")
        print(f"  Transcription (raw): {text_raw}")

        # Save generated tokens
        np.save(os.path.join(output_dir, "generated_tokens.npy"), np.array(generated_ids, dtype=np.int64))
        np.save(os.path.join(output_dir, "generated_tokens_raw.npy"), np.array(generated_raw_ids, dtype=np.int64))

        # Step 4: Run decoder step-by-step for intermediate logits
        # Use the encoder_hidden that produced correct output
        print("\n  Step-by-step decoder (with processor encoder output):")
        bos = torch.tensor([[1]], dtype=torch.long)
        decoder_out = decoder(
            input_ids=bos,
            encoder_hidden_states=encoder_hidden,
            use_cache=True,
        )
        first_hidden = decoder_out.last_hidden_state
        first_logits = model.proj_out(first_hidden)
        np.save(os.path.join(output_dir, "decoder_first_hidden.npy"), first_hidden.numpy())
        np.save(os.path.join(output_dir, "decoder_first_logits.npy"), first_logits.numpy())
        print(f"  Decoder first step hidden: {first_hidden.shape}")
        print(f"  Decoder first step logits: {first_logits.shape}")
        print(f"  Top-5 first token predictions: {torch.topk(first_logits[0, 0], 5)}")

        # Also dump decoder step-by-step with raw encoder output (for Rust comparison)
        print("\n  Step-by-step decoder (with raw encoder output):")
        decoder_out_raw = decoder(
            input_ids=bos,
            encoder_hidden_states=encoder_hidden_raw,
            use_cache=True,
        )
        first_hidden_raw = decoder_out_raw.last_hidden_state
        first_logits_raw = model.proj_out(first_hidden_raw)
        np.save(os.path.join(output_dir, "decoder_first_hidden_raw.npy"), first_hidden_raw.numpy())
        np.save(os.path.join(output_dir, "decoder_first_logits_raw.npy"), first_logits_raw.numpy())
        print(f"  Decoder first step hidden (raw): {first_hidden_raw.shape}")
        print(f"  Top-5 first token (raw): {torch.topk(first_logits_raw[0, 0], 5)}")

        # Run a few more decoder steps with KV cache (using correct encoder output)
        past_kv = decoder_out.past_key_values
        cache_position = torch.tensor([1], dtype=torch.long)
        next_token = first_logits.argmax(dim=-1)
        print(f"\n  First predicted token: {next_token.item()}")

        for step in range(4):
            decoder_out = decoder(
                input_ids=next_token,
                encoder_hidden_states=encoder_hidden,
                past_key_values=past_kv,
                use_cache=True,
                cache_position=cache_position,
            )
            step_logits = model.proj_out(decoder_out.last_hidden_state)
            np.save(os.path.join(output_dir, f"decoder_step{step+2}_logits.npy"), step_logits.numpy())
            past_kv = decoder_out.past_key_values
            next_token = step_logits.argmax(dim=-1)
            cache_position = cache_position + 1
            token_text = processor.decode([next_token.item()])
            print(f"  Step {step+2}: token={next_token.item()}, text='{token_text}'")

    # Save config summary
    config_summary = {
        "encoder_dim": 768,
        "decoder_dim": 640,
        "depth": 14,
        "nheads": 10,
        "head_dim": 64,
        "vocab_size": 32768,
        "bos_id": 1,
        "eos_id": 2,
        "frame_len": 80,
        "encoder_intermediate_size": 3072,
        "decoder_intermediate_size": 2560,
        "audio_samples": len(padded_samples),
        "sample_rate": sample_rate,
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "transcription": text,
        "transcription_raw": text_raw,
        "generated_tokens": generated_ids,
        "generated_tokens_raw": generated_raw_ids,
    }
    with open(os.path.join(output_dir, "config_summary.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    print(f"\nAll intermediates saved to {output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f}: {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
