#!/usr/bin/env python3
"""Convert Kokoro .pth weights to safetensors format for Rust loading.

Handles:
- Nested state dict (top-level keys: bert, bert_encoder, predictor, decoder, text_encoder)
- DataParallel `.module.` prefix removal
- Weight norm folding (weight_g + weight_v -> weight = g/||v|| * v)
- Voice pack conversion (.pt -> .safetensors)
"""

import sys
from pathlib import Path

def fold_weight_norm(state_dict):
    """Fold weight_g and weight_v into a single weight tensor."""
    import torch

    keys_to_remove = set()
    keys_to_add = {}

    # Find all weight_v keys and their corresponding weight_g
    for key in list(state_dict.keys()):
        if key.endswith('.weight_v'):
            prefix = key[:-len('.weight_v')]
            g_key = prefix + '.weight_g'
            if g_key in state_dict:
                v = state_dict[key]
                g = state_dict[g_key]
                # weight_norm: weight = g * v / ||v||
                # v shape: [out, in, kernel] or [out, in]
                # g shape: [out, 1, 1] or [out, 1]
                norm = torch.norm(v.reshape(v.shape[0], -1), dim=1)
                # Reshape norm to broadcast
                shape = [v.shape[0]] + [1] * (v.dim() - 1)
                weight = g.reshape(shape) * v / norm.reshape(shape)
                keys_to_add[prefix + '.weight'] = weight
                keys_to_remove.add(key)
                keys_to_remove.add(g_key)

    for k in keys_to_remove:
        del state_dict[k]
    state_dict.update(keys_to_add)
    return state_dict


def main():
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        print("pip install torch safetensors")
        sys.exit(1)

    model_dir = Path("hf_kokoro")
    pth_path = model_dir / "kokoro-v1_0.pth"
    out_path = model_dir / "kokoro-v1_0.safetensors"

    if not pth_path.exists():
        print(f"Missing: {pth_path}")
        print("Run: python scripts/download_kokoro.py")
        sys.exit(1)

    if out_path.exists():
        print(f"Already exists: {out_path}")
        print("Delete it to regenerate.")
        return

    print(f"Loading {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # Flatten nested module dicts and remove .module. prefix
    flat = {}
    for module_name, module_dict in state_dict.items():
        if isinstance(module_dict, dict):
            for k, v in module_dict.items():
                if isinstance(v, torch.Tensor):
                    # Remove .module. prefix from DataParallel
                    clean_key = k.replace("module.", "", 1) if k.startswith("module.") else k
                    flat[f"{module_name}.{clean_key}"] = v.float().contiguous()
        elif isinstance(module_dict, torch.Tensor):
            flat[module_name] = module_dict.float().contiguous()

    print(f"  {len(flat)} tensors (before weight norm folding)")

    # Fold weight norm
    flat = fold_weight_norm(flat)
    print(f"  {len(flat)} tensors (after weight norm folding)")

    total_params = sum(t.numel() for t in flat.values())
    print(f"  {total_params:,} parameters ({total_params * 4 / 1e6:.1f} MB)")

    # Show a sample of keys for verification
    print("\n  Sample keys:")
    for k in sorted(flat.keys())[:10]:
        print(f"    {k}: {flat[k].shape}")
    print("    ...")

    print(f"\nSaving {out_path}...")
    save_file(flat, str(out_path))
    print("Done.")

    # Convert voice packs
    voices_dir = model_dir / "voices"
    if voices_dir.exists():
        for pt_file in voices_dir.glob("*.pt"):
            st_file = pt_file.with_suffix(".safetensors")
            if st_file.exists():
                continue
            print(f"  Converting voice: {pt_file.name}")
            voice = torch.load(pt_file, map_location="cpu", weights_only=True)
            if isinstance(voice, torch.Tensor):
                # Squeeze middle dim: [N, 1, 256] -> [N, 256]
                voice = voice.squeeze(1).float()
                save_file({"voice": voice}, str(st_file))
            elif isinstance(voice, dict):
                tensors = {k: v.squeeze(1).float() if v.dim() == 3 else v.float()
                           for k, v in voice.items() if isinstance(v, torch.Tensor)}
                save_file(tensors, str(st_file))


if __name__ == "__main__":
    main()
