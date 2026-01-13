"""
Compare encoder outputs between NeMo and Rust implementation.
This helps identify if the issue is in feature extraction or encoder.
"""

import torch
import numpy as np
import nemo.collections.asr as nemo_asr
import librosa

def load_audio(audio_path):
    """Load audio file at 16kHz"""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    return audio

def get_nemo_encoder_output(audio_path):
    """Get encoder output from NeMo"""
    print("Loading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Load audio
    audio = load_audio(audio_path)
    print(f"Audio shape: {audio.shape}")

    # Get features (preprocessor)
    with torch.no_grad():
        audio_signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_length = torch.tensor([len(audio)], dtype=torch.long)

        if torch.cuda.is_available():
            audio_signal = audio_signal.cuda()
            audio_length = audio_length.cuda()

        # Extract features through preprocessor
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=audio_signal,
            length=audio_length
        )

        print(f"Preprocessor output shape: {processed_signal.shape}")
        print(f"Preprocessor output length: {processed_signal_length}")

        # Run encoder
        encoder_output, _ = model.encoder(audio_signal=processed_signal, length=processed_signal_length)

        print(f"Encoder output shape: {encoder_output.shape}")

        # Convert to numpy for comparison
        encoder_np = encoder_output.cpu().numpy()
        features_np = processed_signal.cpu().numpy()

        return {
            'features': features_np,
            'encoder_output': encoder_np,
            'audio': audio,
        }

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compare_encoder_outputs.py <audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    print(f"Processing {audio_path}...")
    print("=" * 60)

    result = get_nemo_encoder_output(audio_path)

    print("\n" + "=" * 60)
    print("NeMo Results:")
    print("=" * 60)
    print(f"Audio samples: {result['audio'].shape}")
    print(f"Features shape: {result['features'].shape}")
    print(f"Encoder output shape: {result['encoder_output'].shape}")

    # Print some statistics
    features = result['features'][0]  # Remove batch dimension
    encoder = result['encoder_output'][0]  # Remove batch dimension

    print("\nFeatures statistics (first 10 frames, first 5 features):")
    print(features[:10, :5])
    print(f"Features mean: {features.mean():.6f}")
    print(f"Features std: {features.std():.6f}")
    print(f"Features min: {features.min():.6f}")
    print(f"Features max: {features.max():.6f}")

    print("\nEncoder output statistics (first 5 frames, first 5 features):")
    print(encoder[:5, :5])
    print(f"Encoder mean: {encoder.mean():.6f}")
    print(f"Encoder std: {encoder.std():.6f}")
    print(f"Encoder min: {encoder.min():.6f}")
    print(f"Encoder max: {encoder.max():.6f}")

    # Save for comparison with Rust
    output_file = audio_path.replace('.wav', '_nemo_encoder.npz')
    np.savez(output_file,
             features=result['features'],
             encoder_output=result['encoder_output'],
             audio=result['audio'])
    print(f"\nSaved to {output_file}")

if __name__ == '__main__':
    main()
