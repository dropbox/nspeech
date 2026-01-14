#!/usr/bin/env python3
"""Check how NeMo's predictor handles blank token"""

import nemo.collections.asr as nemo_asr
import torch

print("\nLoading model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-speech-streaming-en-0.6b"
)
print("✓ Model loaded\n")

# Check decoder/predictor config
print("=== Predictor/Decoder Info ===")
print(f"Decoder type: {type(asr_model.decoder).__name__}")
print(f"Vocab size: {asr_model.decoder.vocab_size}")
print(f"Blank idx: {asr_model.decoder.blank_idx}")
print()

# Check predictor embedding
if hasattr(asr_model.decoder, 'prediction'):
    pred = asr_model.decoder.prediction
    print("Prediction network found:")
    print(f"  Embedding shape: {pred.embed.weight.shape}")
    print(f"  This means predictor accepts token IDs: 0 to {pred.embed.weight.shape[0]-1}")
    print()

    # Try to understand blank token handling
    print("Testing predictor with different token inputs:")

    # Test with token 0 (config blank_id)
    with torch.no_grad():
        token_0 = torch.tensor([[0]])
        out_0 = pred(token_0, None)
        print(f"  Token 0 (config blank): output shape {out_0[0].shape}")

        # Test with token 1024 (runtime blank_idx)
        token_1024 = torch.tensor([[1024]])
        out_1024 = pred(token_1024, None)
        print(f"  Token 1024 (runtime blank): output shape {out_1024[0].shape}")

        # Check if outputs are the same
        if torch.allclose(out_0[0], out_1024[0], atol=1e-6):
            print(f"  → Outputs are IDENTICAL! Both represent blank.")
        else:
            print(f"  → Outputs are DIFFERENT! They represent different tokens.")

print("\n=== Joint Network Info ===")
if hasattr(asr_model.decoder, 'joint'):
    joint = asr_model.decoder.joint
    # The joint network outputs logits for all tokens
    # Check the output projection layer
    if hasattr(joint, 'joint_net'):
        for i, layer in enumerate(joint.joint_net):
            print(f"  Layer {i}: {layer}")
            if hasattr(layer, 'out_features'):
                print(f"    Output features: {layer.out_features}")

print("\n✓ Done")
